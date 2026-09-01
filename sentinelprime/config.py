# sentinelprime/config.py
from __future__ import annotations
from dataclasses import dataclass
import yaml
import dspy


@dataclass
class Models:
    root_lm: dspy.LM
    sub_lm: dspy.LM
    reflection_lm: dspy.LM
    judge_lm: dspy.LM


def _make_lm(spec: dict) -> dspy.LM:
    return dspy.LM(model=spec["model"], **spec.get("params", {}))


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
