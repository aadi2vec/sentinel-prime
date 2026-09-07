# sentinelprime/config.py
from __future__ import annotations
from dataclasses import dataclass
import os
import yaml
import dspy


@dataclass
class Models:
    root_lm: dspy.LM
    sub_lm: dspy.LM
    reflection_lm: dspy.LM
    judge_lm: dspy.LM


def _make_lm(spec: dict) -> dspy.LM:
    # Keep provider secrets/endpoints out of the committed config: for OpenAI-style
    # models, honor a custom endpoint from OPENAI_BASE_URL unless the config pins one.
    params = dict(spec.get("params", {}))
    model = spec["model"]
    if model.startswith("openai/") and "api_base" not in params:
        base = os.environ.get("OPENAI_BASE_URL")
        if base:
            params["api_base"] = base
    return dspy.LM(model=model, **params)


def load_config(path: str | None = None) -> Models:
    if path is None:
        raise ValueError("config path is required (no default provider is baked in)")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Models(
        root_lm=_make_lm(raw["root_lm"]),
        sub_lm=_make_lm(raw["sub_lm"]),
        reflection_lm=_make_lm(raw["reflection_lm"]),
        judge_lm=_make_lm(raw["judge_lm"]),
    )
