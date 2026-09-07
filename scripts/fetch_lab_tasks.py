"""fetch_lab_tasks.py — download a named set of Harvey LAB tasks.

The benchmark corpus is ~2.1GB and is not vendored, but *which* tasks an experiment ran on
is part of the experiment. This fetches a named set into `lab_tasks/` (gitignored) so a
result can be reproduced from the task list in the eval plan.

    .venv/bin/python scripts/fetch_lab_tasks.py --list        # sizes, no download
    .venv/bin/python scripts/fetch_lab_tasks.py              # fetch the default set
    .venv/bin/python scripts/fetch_lab_tasks.py --task extract-closing-conditions

Auth: uses the github.com credential already in the local git credential helper, so no
`gh auth login` is required. Public repo, so an unauthenticated fetch also works if the
rate limit allows.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

REPO = "harveyai/harvey-labs"
AREA = "corporate-ma"
DEST = Path("lab_tasks")

# A single practice-area family: contract review and diligence extraction. Chosen so a
# lesson learned on one task *could* transfer to another — the premise the A/B tests. A set
# drawn from unrelated areas would test nothing, since no honest lesson would carry over.
DEFAULT_SET = [
    "extract-credit-agreement-covenants",
    "extract-closing-conditions",
    "extract-post-closing-obligations",
    "extract-change-of-control-provisions",
    "identify-pe-target-contract-issues",
    "review-material-contracts-coc",
    "review-material-contract-review",
    "identify-issues-in-portfolio-company-contracts",
    "identify-target-org-docs-issues",
    "review-enterprise-saas-agreement",
    "compare-target-representations-vs-diligence",
    "extract-key-terms-from-fund-term-sheet",
]


def _token() -> str | None:
    try:
        out = subprocess.run(["git", "credential", "fill"],
                             input="protocol=https\nhost=github.com\n\n",
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1]
    except Exception:
        pass
    return None


def _api(path: str, token: str | None):
    req = urllib.request.Request(f"https://api.github.com/{path}")
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def fetch(task: str, token: str | None, list_only: bool) -> tuple[int, int, int] | None:
    base = f"repos/{REPO}/contents/tasks/{AREA}/{task}"
    try:
        manifest = _api(f"{base}/task.json", token)
        docs = _api(f"{base}/documents", token)
    except Exception as exc:
        print(f"  {task:<52} SKIP ({exc})")
        return None
    raw = json.loads(_get(manifest["download_url"]))
    n_criteria = len(raw.get("criteria") or [])
    n_docs = len(docs)
    n_bytes = sum(d["size"] for d in docs)
    print(f"  {task:<52} docs={n_docs:<3} {n_bytes // 1024:>5}KB  criteria={n_criteria}")
    if list_only:
        return n_docs, n_bytes, n_criteria

    out = DEST / AREA / task
    (out / "documents").mkdir(parents=True, exist_ok=True)
    (out / "task.json").write_bytes(_get(manifest["download_url"]))
    for d in docs:
        target = out / "documents" / d["name"]
        if not target.exists():          # resumable: skip what is already local
            target.write_bytes(_get(d["download_url"]))
    return n_docs, n_bytes, n_criteria


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", action="append", help="task name (repeatable)")
    ap.add_argument("--list", action="store_true", help="show sizes without downloading")
    args = ap.parse_args()

    tasks = args.task or DEFAULT_SET
    token = _token()
    print(f"[fetch] {len(tasks)} tasks from {REPO}/tasks/{AREA} "
          f"({'listing only' if args.list else f'into {DEST}'})")
    stats = [s for t in tasks if (s := fetch(t, token, args.list))]
    if stats:
        docs = sum(s[0] for s in stats)
        kb = sum(s[1] for s in stats) // 1024
        crit = sum(s[2] for s in stats)
        print(f"\n[fetch] {len(stats)} tasks | {docs} documents | {kb}KB | "
              f"{crit} rubric criteria total")


if __name__ == "__main__":
    main()
