"""watch_run.py — live, readable tail of a sweep's event log.

`tail -f run.jsonl` shows raw JSON. This renders the same stream grouped and coloured so
the interesting layers — ledger edits, verifier rejections, retirements, replan events,
sub-query cache hits — stand out from the per-criterion judge noise.

    .venv/bin/python scripts/watch_run.py                 # follow live
    .venv/bin/python scripts/watch_run.py --no-follow     # replay what is there
    .venv/bin/python scripts/watch_run.py --only ledger_create,verifier_reject
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

LOG = pathlib.Path("lab_runs/paired/run.jsonl")

# Colour by layer, so the mechanisms we added are visually separable from the run itself.
C = {
    "task_start": "\033[1;36m", "task_done": "\033[1;32m", "skip": "\033[2;37m",
    "arm_start": "\033[1;35m", "arm_done": "\033[1;35m",
    "ledger_create": "\033[1;33m", "ledger_update": "\033[0;33m",
    "verifier_reject": "\033[1;31m", "credit_retire": "\033[1;31m",
    "audit": "\033[0;36m", "ledger_size": "\033[0;33m",
    "replan": "\033[1;31m", "subquery_cache": "\033[0;34m",
    "rlm_trace": "\033[0;35m", "judge": "\033[2;37m",
    "concurrency_forced": "\033[1;31m", "ungrounded_pass": "\033[1;31m",
    "tokens": "\033[0;32m",
}
RESET = "\033[0m"


def render(rec: dict) -> str:
    kind = rec.get("kind", "?")
    stamp = time.strftime("%H:%M:%S", time.localtime(rec.get("ts", 0)))
    colour = C.get(kind, "")
    body = " ".join(f"{k}={v}" for k, v in rec.items() if k not in ("ts", "kind"))
    if len(body) > 150:
        body = body[:147] + "..."
    return f"{colour}[{stamp}] {kind:<18}{RESET} {body}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--no-follow", action="store_true")
    ap.add_argument("--only", help="comma-separated event kinds")
    ap.add_argument("--quiet-judge", action="store_true",
                    help="hide per-criterion judge lines (there are hundreds)")
    args = ap.parse_args()

    path = pathlib.Path(args.log)
    only = set(args.only.split(",")) if args.only else None
    print(f"watching {path}  (ctrl-c to stop)\n")

    def show(line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return
        kind = rec.get("kind")
        if only and kind not in only:
            return
        if args.quiet_judge and kind == "judge":
            return
        print(render(rec), flush=True)

    while not path.exists():
        if args.no_follow:
            print(f"no log at {path} yet")
            return
        time.sleep(0.5)

    with open(path) as f:
        for line in f:
            show(line)
        if args.no_follow:
            return
        while True:
            line = f.readline()
            if line:
                show(line)
            else:
                time.sleep(0.4)


if __name__ == "__main__":
    main()
