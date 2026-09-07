"""Harvey LAB adapter — dataset, workspace, and rubric results.

LAB ships a `ModelAdapter` interface (`chat`, `make_tool_result_messages`, ...) that assumes
the benchmark drives the agent loop. PrimeAgent has its own loop — RLM turns, a REPL,
sub-agents — so forcing it through that interface would mean discarding the thing under test.
Instead this uses LAB as **dataset + rubric + judge** and drives the agent ourselves:

    task.json + documents/  ->  LabTask  ->  workspace/{documents,output}
                                                     |
                                    PrimeAgent.run_task(instructions, workdir)
                                                     |
                                 output/*  ->  judge  ->  to_feedback  ->  refine()

Two deliberate choices:

- **Documents are converted to text, not handed over raw.** LAB's own harness parses
  documents (Pandoc/MarkItDown/pdfplumber) before the agent sees them, so extracting text is
  faithful rather than a shortcut. `.docx` is a zip of XML, so the stdlib is enough and the
  agent needs no new dependency.
- **The workspace is a copy.** `LocalInterpreter` runs with this process's permissions and is
  explicitly not a sandbox, so an agent bug must not be able to reach the benchmark corpus.

Scoring note: `Feedback.score` here is the **pooled criterion pass rate**, not LAB's headline
all-pass score — `all_pass()` reports that separately. At <10% completion with dozens of
ANDed binary criteria, all-pass has almost no variance to learn from, while the per-criterion
verdicts are exactly the label-free signal `refine()` already consumes.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from sentinelprime.feedback import Feedback, parse_lab_result

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class LabTask:
    task_id: str
    title: str
    instructions: str
    work_type: str
    criteria: list[dict]
    deliverables: list[str]
    dir: Path
    tags: list[str] = field(default_factory=list)

    @property
    def documents(self) -> list[Path]:
        docs = self.dir / "documents"
        if not docs.is_dir():
            return []
        return sorted(p for p in docs.iterdir() if p.is_file())


def load_task(task_dir: str | Path) -> LabTask:
    """Read one `tasks/<area>/<task>/` directory into a LabTask."""
    task_dir = Path(task_dir)
    manifest = task_dir / "task.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"no task.json in {task_dir}")
    raw = json.loads(manifest.read_text())
    # `deliverables` is a {filename: filename} map in the real schema; the values carry no
    # extra information, so normalize to the list of expected output filenames.
    deliverables = raw.get("deliverables") or {}
    if isinstance(deliverables, dict):
        deliverables = list(deliverables)
    return LabTask(
        task_id=task_dir.name,
        title=raw.get("title", ""),
        instructions=raw.get("instructions", ""),
        work_type=raw.get("work_type", ""),
        criteria=list(raw.get("criteria") or []),
        deliverables=list(deliverables),
        tags=list(raw.get("tags") or []),
        dir=task_dir,
    )


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    # One line per <w:p>, so section numbers do not run into neighbouring paragraphs.
    paragraphs = re.findall(rf"<{_W_NS[1:-1]}?:?p[ >].*?</.*?:?p>", xml, re.S) or []
    if not paragraphs:
        paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    out = []
    for para in paragraphs:
        # Strip markup first, then unescape once: real LAB documents carry entities like
        # `&amp;nbsp;`, and unescaping before the strip would let an entity forge a tag.
        text = html.unescape(_TAG_RE.sub("", para))
        if text.strip():
            out.append(text.strip())
    return "\n".join(out)


def document_text(path: str | Path) -> str:
    """Extract readable text. `.docx` via the stdlib; plaintext passes straight through.

    The extension is a hint, not a guarantee. Asked for `report.docx`, an agent will often
    write markdown under that name — which is a reasonable deliverable, and must not crash
    the grader — so a `.docx` that is not a valid zip is read as text.
    """
    path = Path(path)
    if path.suffix.lower() == ".docx":
        try:
            return _docx_text(path)
        except zipfile.BadZipFile:
            pass
    return path.read_text(errors="replace")


def prepare_workspace(task: LabTask, dest: str | Path) -> Path:
    """Build a disposable `{documents,output}` workspace for one run.

    Documents land as `<original name>.txt` so the agent can read them with `open()` while
    the filename still points back at the source document a rubric criterion will cite.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "documents").mkdir(parents=True)
    (dest / "output").mkdir(parents=True)
    for doc in task.documents:
        (dest / "documents" / f"{doc.name}.txt").write_text(document_text(doc))
    return dest


def to_feedback(task_id: str, criteria_results: list[dict]) -> Feedback:
    """LAB rubric verdicts -> the label-free signal `refine()` consumes.

    `Feedback.score` is the pooled criterion pass rate. The judge's `reasoning` on a failed
    criterion becomes the failure text the proposer reads — it is the only thing carrying
    *why* the run fell short, so it must not be dropped.
    """
    return parse_lab_result({
        "task_id": task_id,
        "criteria": [
            {
                "id": c.get("id", ""),
                "passed": str(c.get("verdict", "")).strip().lower() == "pass",
                "reason": c.get("reasoning", "") or c.get("title", ""),
            }
            for c in criteria_results
        ],
    })


def all_pass(feedback: Feedback) -> bool:
    """LAB's headline metric: complete only if every criterion passed."""
    return bool(feedback.criteria) and not feedback.failures


DEFAULT_RUBRIC_PROMPT = Path("lab_tasks/_lab_eval/prompts/rubric_criterion.txt")


class RubricJudge:
    """Grades a deliverable criterion-by-criterion using LAB's own rubric prompt.

    The prompt template is **loaded from LAB's file**, not paraphrased here, so the grading
    contract is theirs: one criterion per call, only the relevant output in context, a JSON
    `{reasoning, verdict}` reply. Missing that file is an error rather than a silent
    fallback to a prompt of our own — self-grading a self-improvement loop against a rubric
    we wrote would not be an evaluation.

    This is still *not* an official LAB score: their harness runs its own judge models
    (claude-sonnet-4-6 by default, optionally dual-judge) through their SDK path. Same
    prompt and same contract, different runner. Report it that way.
    """

    def __init__(self, lm, prompt_path: str | Path | None = None) -> None:
        self.lm = lm
        path = Path(prompt_path or DEFAULT_RUBRIC_PROMPT)
        if not path.is_file():
            raise FileNotFoundError(
                f"LAB rubric prompt not found at {path}. Fetch it from "
                "harveyai/harvey-labs (evaluation/prompts/rubric_criterion.txt) — "
                "grading with a substitute prompt would not be a LAB result."
            )
        self.template = path.read_text()

    def _verdict(self, reply: str) -> tuple[str, str]:
        # Accept bare or fenced JSON. Anything unparseable is a FAIL: a judge that cannot
        # read its own output must never hand out a passing criterion.
        text = reply.strip()
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                verdict = str(data.get("verdict", "")).strip().lower()
                if verdict in ("pass", "fail"):
                    return verdict, str(data.get("reasoning", ""))
            except json.JSONDecodeError:
                pass
        return "fail", f"unparseable judge reply: {text[:200]!r}"

    def judge(self, task: LabTask, agent_output: str,
              criteria: list[dict] | None = None,
              available_files: list[str] | None = None) -> list[dict]:
        """Grade each criterion, skipping any whose deliverable was never written.

        `available_files` is what the agent actually produced. A criterion naming a file
        that is absent is failed deterministically, without an LM call — asked to grade a
        report that does not exist, a judge will confabulate one: on the first LAB warmup
        it invented detailed passing justifications for 26 of 55 criteria against an empty
        output directory. Passing `None` disables the check (grade everything).
        """
        available = None if available_files is None else set(available_files)
        results: list[dict] = []
        for criterion in (criteria if criteria is not None else task.criteria):
            required = criterion.get("deliverables") or task.deliverables
            if available is not None and required and not (available & set(required)):
                results.append({
                    "id": criterion.get("id", ""),
                    "title": criterion.get("title", ""),
                    "verdict": "fail",
                    "reasoning": (f"deliverable not produced: none of {sorted(required)} "
                                  f"found in output (present: {sorted(available) or 'nothing'})"),
                })
                continue
            prompt = self.template.format(
                task_description=f"{task.title}\n\n{task.instructions}",
                agent_output=agent_output,
                criterion_title=criterion.get("title", ""),
                match_criteria=criterion.get("match_criteria", ""),
            )
            reply = self.lm(messages=[{"role": "user", "content": prompt}])
            verdict, reasoning = self._verdict(reply[0] if reply else "")
            results.append({
                "id": criterion.get("id", ""),
                "title": criterion.get("title", ""),
                "verdict": verdict,
                "reasoning": reasoning,
            })
        return results


def output_files(workspace: str | Path) -> list[str]:
    """Names of the files the agent actually wrote to `output/`."""
    out = Path(workspace) / "output"
    if not out.is_dir():
        return []
    return sorted(p.name for p in out.rglob("*") if p.is_file())


def read_output(workspace: str | Path) -> str:
    """Concatenate everything the agent wrote to `output/`, for the judge to read."""
    out = Path(workspace) / "output"
    if not out.is_dir():
        return ""
    chunks = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        chunks.append(f"===== {path.relative_to(out)} =====\n{document_text(path)}")
    return "\n\n".join(chunks)
