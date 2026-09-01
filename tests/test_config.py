# tests/test_config.py
import textwrap
import dspy
from primeagent.config import load_config, _make_lm, Models


def test_make_lm_builds_dspy_lm_with_params():
    lm = _make_lm({"model": "openai/gpt-4o-mini", "params": {"temperature": 0.0}})
    assert isinstance(lm, dspy.LM)
    assert lm.model == "openai/gpt-4o-mini"


def test_load_config_maps_all_four_roles(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(textwrap.dedent("""
        root_lm: {model: "openai/root"}
        sub_lm: {model: "openai/sub"}
        reflection_lm: {model: "openai/reflect"}
        judge_lm: {model: "openai/judge"}
    """))
    models = load_config(str(cfg))
    assert isinstance(models, Models)
    assert models.root_lm.model == "openai/root"
    assert models.sub_lm.model == "openai/sub"
    assert models.reflection_lm.model == "openai/reflect"
    assert models.judge_lm.model == "openai/judge"
