"""Live smoke test: run PrimeAgent against a real LM over a tiny document task.

Provider is auto-detected from whichever API key is present in the environment
(.env is loaded manually). Override the model with SMOKE_MODEL=... if desired.

Usage:
    .venv/bin/python scripts/smoke_live.py
"""
from __future__ import annotations

import os
import pathlib
import tempfile


def _load_dotenv(path: str = ".env") -> None:
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ.setdefault(key.strip(), val)


def _pick_model() -> str:
    if os.environ.get("SMOKE_MODEL"):
        return os.environ["SMOKE_MODEL"]
    # OpenAI first: this repo is configured for the gpt-5.6 "luna" endpoint.
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-5-20250929"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini/gemini-2.5-flash"
    raise SystemExit("No provider API key found in environment (.env).")


def _lm_kwargs() -> dict:
    # Custom endpoint (e.g. the luna proxy) via OPENAI_BASE_URL; blank -> api.openai.com.
    kwargs = {"max_tokens": 4000}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
    return kwargs


def main() -> None:
    _load_dotenv()
    model = _pick_model()
    print(f"[smoke] using model: {model}")

    import dspy

    from sentinelprime.agent import PrimeAgent
    from sentinelprime.harness import ContinualHarness
    from sentinelprime.memory import JsonMemoryBackend

    lm = dspy.LM(model, **_lm_kwargs())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        # A tiny "legal" document for the agent to read from its workdir.
        workdir = tmp / "matter"
        workdir.mkdir()
        (workdir / "contract.txt").write_text(
            "MASTER SERVICES AGREEMENT\n\n"
            "Section 3. Term. This Agreement begins on 2026-01-01 and continues "
            "for twelve (12) months, renewing automatically unless either party "
            "gives 30 days written notice.\n\n"
            "Section 7. Governing Law. This Agreement is governed by the laws of "
            "the State of Delaware.\n"
        )

        backend = JsonMemoryBackend(str(tmp / "ledger.json"))
        harness = ContinualHarness(backend)
        agent = PrimeAgent(harness, root_lm=lm)

        task = (
            "Read contract.txt in the working directory. In one or two sentences, "
            "state the governing law and the notice period required to prevent "
            "automatic renewal."
        )
        print(f"[smoke] running task in workdir: {workdir}")
        pred = agent.run_task(task, workdir=str(workdir))
        print("\n=== DELIVERABLE ===")
        print(getattr(pred, "deliverable", pred))
        print("===================\n")
        print("[smoke] OK")


if __name__ == "__main__":
    main()
