"""Harvey LAB adapter: task loading, document text, and rubric results -> Feedback."""
import json
import zipfile

import pytest


def _docx(path, paragraphs):
    """A minimal but real .docx: a zip whose word/document.xml holds w:p/w:r/w:t runs."""
    body = "".join(
        f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", xml)


def _task_dir(tmp_path, *, criteria=None, docs=("contract.docx",)):
    d = tmp_path / "extract-change-of-control-provisions"
    (d / "documents").mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({
        "title": "Change of Control Provision Extraction",
        "work_type": "review",
        "tags": ["Mergers & Acquisitions", "change-of-control"],
        "instructions": "Review the contracts and prepare a report. Output: `report.docx`.",
        "deliverables": {"report.docx": "report.docx"},
        "criteria": criteria if criteria is not None else [
            {"id": "C-001", "title": "Identifies Northland anti-assignment clause",
             "deliverables": ["report.docx"],
             "match_criteria": "PASS if the report identifies Section 14.2."},
        ],
    }))
    for name in docs:
        _docx(d / "documents" / name, ["MASTER SUPPLY AGREEMENT",
                                       "Section 14.2 Anti-Assignment."])
    return d


# ---- load_task -----------------------------------------------------------------------

def test_load_task_reads_the_real_schema(tmp_path):
    from sentinelprime.lab import load_task

    task = load_task(_task_dir(tmp_path))
    assert task.task_id == "extract-change-of-control-provisions"
    assert task.work_type == "review"
    assert "prepare a report" in task.instructions
    assert task.deliverables == ["report.docx"]      # dict in the file, list here
    assert [c["id"] for c in task.criteria] == ["C-001"]


def test_load_task_lists_its_documents(tmp_path):
    from sentinelprime.lab import load_task

    task = load_task(_task_dir(tmp_path, docs=("a.docx", "b.docx")))
    assert sorted(p.name for p in task.documents) == ["a.docx", "b.docx"]


def test_load_task_rejects_a_directory_with_no_task_json(tmp_path):
    from sentinelprime.lab import load_task

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_task(tmp_path / "empty")


# ---- document text -------------------------------------------------------------------

def test_docx_text_extraction_is_dependency_free(tmp_path):
    """A .docx is a zip of XML; stdlib is enough, so the agent needs no new dependency."""
    from sentinelprime.lab import document_text

    path = tmp_path / "c.docx"
    _docx(path, ["MASTER SUPPLY AGREEMENT", "Section 14.2 Anti-Assignment."])
    text = document_text(path)
    assert "MASTER SUPPLY AGREEMENT" in text
    assert "Section 14.2 Anti-Assignment." in text


def test_docx_paragraphs_stay_separated(tmp_path):
    """Runs must not be glued together, or section numbers merge into neighbouring text."""
    from sentinelprime.lab import document_text

    path = tmp_path / "c.docx"
    _docx(path, ["Section 14.2", "Section 15.1"])
    assert "14.2\nSection" in document_text(path)


def test_plaintext_documents_pass_through(tmp_path):
    from sentinelprime.lab import document_text

    p = tmp_path / "note.txt"
    p.write_text("governing law: Delaware")
    assert document_text(p) == "governing law: Delaware"


# ---- workspace -----------------------------------------------------------------------

def test_workspace_gives_the_agent_readable_text_and_an_output_dir(tmp_path):
    """LAB parses documents for the agent; mirror that instead of handing it a zip."""
    from sentinelprime.lab import load_task, prepare_workspace

    task = load_task(_task_dir(tmp_path))
    ws = prepare_workspace(task, tmp_path / "run")

    assert (ws / "output").is_dir()
    docs = sorted(p.name for p in (ws / "documents").iterdir())
    assert docs == ["contract.docx.txt"]
    assert "Section 14.2" in (ws / "documents" / "contract.docx.txt").read_text()


def test_workspace_never_mutates_the_benchmark_corpus(tmp_path):
    """LocalInterpreter is not a sandbox; an agent bug must not reach the source tasks."""
    from sentinelprime.lab import load_task, prepare_workspace

    task = load_task(_task_dir(tmp_path))
    ws = prepare_workspace(task, tmp_path / "run")
    assert task.dir not in ws.parents and ws != task.dir


# ---- rubric results -> Feedback ------------------------------------------------------

def test_to_feedback_maps_lab_verdicts_to_the_label_free_signal():
    from sentinelprime.lab import to_feedback

    fb = to_feedback("ma-001", [
        {"id": "C-001", "verdict": "pass", "reasoning": "found it"},
        {"id": "C-002", "verdict": "fail", "reasoning": "no mention of Section 9"},
    ])
    assert fb.task_id == "ma-001"
    assert fb.score == 0.5                     # pooled criterion pass rate
    assert [c.id for c in fb.failures] == ["C-002"]
    assert "Section 9" in fb.as_text()         # the reason is what refine() learns from


def test_to_feedback_scores_an_all_pass_run_as_one():
    from sentinelprime.lab import to_feedback

    fb = to_feedback("t", [{"id": "C-001", "verdict": "pass", "reasoning": ""}])
    assert fb.score == 1.0
    assert fb.failures == []


def test_all_pass_is_reported_separately_from_the_pooled_rate():
    """LAB's headline metric is all-or-nothing; the learning signal is the pooled rate."""
    from sentinelprime.lab import all_pass, to_feedback

    fb = to_feedback("t", [{"id": "C-001", "verdict": "pass", "reasoning": ""},
                           {"id": "C-002", "verdict": "fail", "reasoning": "x"}])
    assert fb.score == 0.5
    assert all_pass(fb) is False


def test_docx_text_unescapes_xml_entities(tmp_path):
    """Real LAB documents contain &amp;nbsp; — the agent must not read escaped markup."""
    from sentinelprime.lab import document_text

    path = tmp_path / "c.docx"
    _docx(path, ["Section 5 &amp; 6", "fee of &amp;nbsp;$100 &lt;cap&gt;"])
    text = document_text(path)
    assert "Section 5 & 6" in text
    assert "&amp;" not in text
    assert "<cap>" in text


# ---- rubric judge (LAB's own prompt) --------------------------------------------------

class _StubLM:
    """Stands in for a dspy LM: records prompts, returns canned JSON verdicts."""

    def __init__(self, replies):
        self.prompts = []
        self._replies = list(replies)

    def __call__(self, messages=None, **kw):
        self.prompts.append(messages[-1]["content"])
        return [self._replies.pop(0)]


def _prompt_file(tmp_path):
    p = tmp_path / "rubric_criterion.txt"
    p.write_text(
        "## Task\n{task_description}\n\n## Agent's Output\n{agent_output}\n\n"
        "## Criterion\n**{criterion_title}**\n\n{match_criteria}\n\n"
        'Respond with JSON only:\n```json\n{{\n  "reasoning": "...", "verdict": "pass" | "fail"\n}}\n```'
    )
    return p


def test_judge_sends_labs_prompt_with_the_criterion_and_output(tmp_path):
    from sentinelprime.lab import RubricJudge, load_task

    task = load_task(_task_dir(tmp_path))
    lm = _StubLM(['{"reasoning": "found Section 14.2", "verdict": "pass"}'])
    results = RubricJudge(lm, prompt_path=_prompt_file(tmp_path)).judge(task, "the report text")

    assert results == [{"id": "C-001",
                        "title": "Identifies Northland anti-assignment clause",
                        "verdict": "pass",
                        "reasoning": "found Section 14.2"}]
    sent = lm.prompts[0]
    assert "PASS if the report identifies Section 14.2." in sent   # match_criteria
    assert "the report text" in sent                               # agent output
    assert "Change of Control Provision Extraction" in sent        # task description


def test_judge_parses_a_fenced_json_reply(tmp_path):
    from sentinelprime.lab import RubricJudge, load_task

    task = load_task(_task_dir(tmp_path))
    lm = _StubLM(['```json\n{"reasoning": "no mention", "verdict": "fail"}\n```'])
    r = RubricJudge(lm, prompt_path=_prompt_file(tmp_path)).judge(task, "out")
    assert r[0]["verdict"] == "fail"
    assert r[0]["reasoning"] == "no mention"


def test_judge_treats_an_unparseable_reply_as_a_failure_not_a_pass(tmp_path):
    """A judge that errors must never silently award a criterion."""
    from sentinelprime.lab import RubricJudge, load_task

    task = load_task(_task_dir(tmp_path))
    lm = _StubLM(["I cannot evaluate this"])
    r = RubricJudge(lm, prompt_path=_prompt_file(tmp_path)).judge(task, "out")
    assert r[0]["verdict"] == "fail"
    assert "unparseable" in r[0]["reasoning"].lower()


def test_judge_scores_every_criterion(tmp_path):
    from sentinelprime.lab import RubricJudge, load_task, to_feedback

    task = load_task(_task_dir(tmp_path, criteria=[
        {"id": "C-001", "title": "a", "match_criteria": "PASS if a"},
        {"id": "C-002", "title": "b", "match_criteria": "PASS if b"},
    ]))
    lm = _StubLM(['{"reasoning": "yes", "verdict": "pass"}',
                  '{"reasoning": "no", "verdict": "fail"}'])
    fb = to_feedback(task.task_id, RubricJudge(lm, prompt_path=_prompt_file(tmp_path)).judge(task, "out"))
    assert fb.score == 0.5
    assert len(lm.prompts) == 2


# ---- the judge must never grade a deliverable that was not produced -------------------

def test_judge_fails_criteria_whose_deliverable_is_missing_without_asking_the_lm(tmp_path):
    """Warmup finding: asked to grade a report that did not exist, the judge invented
    detailed passes for 26 of 55 criteria. A missing file is a deterministic FAIL."""
    from sentinelprime.lab import RubricJudge, load_task

    task = load_task(_task_dir(tmp_path))
    lm = _StubLM([])  # any LM call here would raise IndexError
    results = RubricJudge(lm, prompt_path=_prompt_file(tmp_path)).judge(
        task, "", available_files=[])

    assert results[0]["verdict"] == "fail"
    assert "not produced" in results[0]["reasoning"]
    assert lm.prompts == []          # no tokens spent grading nothing


def test_judge_grades_normally_when_the_deliverable_is_present(tmp_path):
    from sentinelprime.lab import RubricJudge, load_task

    task = load_task(_task_dir(tmp_path))
    lm = _StubLM(['{"reasoning": "found it", "verdict": "pass"}'])
    results = RubricJudge(lm, prompt_path=_prompt_file(tmp_path)).judge(
        task, "the report", available_files=["report.docx"])
    assert results[0]["verdict"] == "pass"
    assert len(lm.prompts) == 1


def test_judge_fails_only_the_criteria_whose_own_deliverable_is_missing(tmp_path):
    """Criteria declare their own deliverables; a missing one must not fail the rest."""
    from sentinelprime.lab import RubricJudge, load_task

    task = load_task(_task_dir(tmp_path, criteria=[
        {"id": "C-001", "title": "a", "match_criteria": "x", "deliverables": ["report.docx"]},
        {"id": "C-002", "title": "b", "match_criteria": "y", "deliverables": ["memo.docx"]},
    ]))
    lm = _StubLM(['{"reasoning": "ok", "verdict": "pass"}'])
    results = RubricJudge(lm, prompt_path=_prompt_file(tmp_path)).judge(
        task, "the report", available_files=["report.docx"])

    assert [r["verdict"] for r in results] == ["pass", "fail"]
    assert len(lm.prompts) == 1      # only the criterion that could be graded


def test_output_files_lists_what_the_agent_actually_wrote(tmp_path):
    from sentinelprime.lab import output_files

    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "report.docx").write_text("x")
    assert output_files(tmp_path) == ["report.docx"]
    assert output_files(tmp_path / "nope") == []


def test_document_text_falls_back_when_a_docx_is_really_plain_text(tmp_path):
    """Agents write markdown under the requested .docx name; reading it must not crash."""
    from sentinelprime.lab import document_text

    p = tmp_path / "report.docx"
    p.write_text("# Change-of-Control Extraction Report\n\nSection 14.2 anti-assignment.")
    text = document_text(p)
    assert "Section 14.2 anti-assignment." in text


def test_read_output_survives_a_mixture_of_real_and_fake_docx(tmp_path):
    from sentinelprime.lab import read_output

    out = tmp_path / "output"
    out.mkdir()
    _docx(out / "real.docx", ["genuine zip content"])
    (out / "fake.docx").write_text("plain text body")
    text = read_output(tmp_path)
    assert "genuine zip content" in text
    assert "plain text body" in text


# ---- noise floor: how much do repeated verdicts move on their own? --------------------

def test_stability_reports_zero_spread_for_identical_replicates():
    from sentinelprime.lab import verdict_stability

    rep = [{"C-001": "pass", "C-002": "fail"}] * 3
    s = verdict_stability(rep)
    assert s["replicates"] == 3
    assert s["rates"] == [0.5, 0.5, 0.5]
    assert s["spread"] == 0.0
    assert s["flipped"] == []


def test_stability_names_the_criteria_that_flipped():
    """The unstable criteria matter more than the aggregate — they say where noise lives."""
    from sentinelprime.lab import verdict_stability

    s = verdict_stability([
        {"C-001": "pass", "C-002": "fail", "C-003": "pass"},
        {"C-001": "pass", "C-002": "pass", "C-003": "pass"},
        {"C-001": "fail", "C-002": "pass", "C-003": "pass"},
    ])
    assert s["flipped"] == ["C-001", "C-002"]
    assert s["flip_rate"] == pytest.approx(2 / 3)


def test_stability_spread_is_the_range_of_pooled_rates():
    from sentinelprime.lab import verdict_stability

    s = verdict_stability([
        {"C-001": "pass", "C-002": "pass"},   # 1.0
        {"C-001": "pass", "C-002": "fail"},   # 0.5
    ])
    assert s["rates"] == [1.0, 0.5]
    assert s["spread"] == 0.5
    assert s["mean"] == 0.75


def test_stability_needs_at_least_two_replicates():
    from sentinelprime.lab import verdict_stability

    with pytest.raises(ValueError):
        verdict_stability([{"C-001": "pass"}])


def test_verdict_map_extracts_id_to_verdict_from_judge_results():
    from sentinelprime.lab import verdict_map

    assert verdict_map([{"id": "C-001", "verdict": "pass"},
                        {"id": "C-002", "verdict": "fail"}]) == {"C-001": "pass",
                                                                 "C-002": "fail"}


def _xlsx(path, strings):
    """A minimal .xlsx: shared strings are where cell text actually lives."""
    shared = "".join(f"<si><t>{s}</t></si>" for s in strings)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/sharedStrings.xml",
                   '<?xml version="1.0"?><sst>' + shared + "</sst>")
        z.writestr("xl/worksheets/sheet1.xml",
                   '<?xml version="1.0"?><worksheet><sheetData/></worksheet>')


def test_xlsx_deliverables_are_readable(tmp_path):
    """One task in the set expects .xlsx; reading it as bytes would feed the judge noise."""
    from sentinelprime.lab import document_text

    p = tmp_path / "risk.xlsx"
    _xlsx(p, ["Contract", "Change of Control", "High"])
    text = document_text(p)
    assert "Change of Control" in text
    assert "High" in text


def test_office_formats_share_one_extraction_path(tmp_path):
    from sentinelprime.lab import document_text

    d, x = tmp_path / "a.docx", tmp_path / "b.xlsx"
    _docx(d, ["docx body"])
    _xlsx(x, ["xlsx body"])
    assert "docx body" in document_text(d)
    assert "xlsx body" in document_text(x)


def test_unreadable_binary_does_not_crash_the_grader(tmp_path):
    from sentinelprime.lab import document_text

    p = tmp_path / "weird.xlsx"
    p.write_bytes(b"\x00\x01\x02not a zip and not utf8 \xff\xfe")
    document_text(p)   # must not raise


# ---- paired A/B analysis --------------------------------------------------------------

def test_paired_summary_reports_the_mean_delta_and_its_error():
    from sentinelprime.lab import paired_summary

    s = paired_summary([("t1", 0.50, 0.60), ("t2", 0.40, 0.45), ("t3", 0.30, 0.30)])
    assert s["n"] == 3
    assert s["mean_delta"] == pytest.approx(0.05)
    assert s["wins"] == 2 and s["losses"] == 0 and s["ties"] == 1
    assert s["se"] > 0


def test_paired_summary_flags_an_effect_smaller_than_its_own_error():
    """The whole point: a delta inside the noise is not a result."""
    from sentinelprime.lab import paired_summary

    s = paired_summary([("t1", 0.50, 0.52), ("t2", 0.40, 0.30), ("t3", 0.30, 0.40)])
    assert s["significant"] is False


def test_paired_summary_calls_a_large_consistent_effect_significant():
    from sentinelprime.lab import paired_summary

    s = paired_summary([(f"t{i}", 0.30, 0.55) for i in range(6)])
    assert s["mean_delta"] == pytest.approx(0.25)
    assert s["significant"] is True


def test_paired_summary_needs_at_least_two_pairs():
    from sentinelprime.lab import paired_summary

    with pytest.raises(ValueError):
        paired_summary([("t1", 0.5, 0.6)])
