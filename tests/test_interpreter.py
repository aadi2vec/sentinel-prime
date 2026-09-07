import pytest
from dspy.primitives.code_interpreter import FinalOutput, CodeExecutionError, CodeInterpreter
from sentinelprime.interpreter import LocalInterpreter


def test_state_persists_and_stdout_captured():
    interp = LocalInterpreter()
    interp.start()
    assert interp.execute("x = 21") is None            # no print -> None
    assert interp.execute("print(x * 2)") == "42\n"    # state carried across calls


def test_submit_returns_final_output():
    interp = LocalInterpreter()
    interp.output_fields = [{"name": "deliverable"}]
    interp.start()
    result = interp.execute("SUBMIT(deliverable='done')")
    assert isinstance(result, FinalOutput)
    assert result.output == {"deliverable": "done"}


def test_runtime_error_becomes_code_execution_error():
    interp = LocalInterpreter()
    interp.start()
    with pytest.raises(CodeExecutionError):
        interp.execute("1 / 0")


def test_syntax_error_propagates():
    interp = LocalInterpreter()
    interp.start()
    with pytest.raises(SyntaxError):
        interp.execute("def (:")


def test_injected_tool_is_callable_from_code():
    interp = LocalInterpreter()
    interp.tools["echo"] = lambda **kw: kw["msg"]
    interp.start()
    assert interp.execute("print(echo(msg='hi'))") == "hi\n"


def test_satisfies_protocol():
    assert isinstance(LocalInterpreter(), CodeInterpreter)


from dspy.primitives.code_interpreter import _validate_interpreter_factory
from sentinelprime.interpreter import _confine_path, InterpreterFactory


def test_confine_path_allows_inside(tmp_path):
    p = _confine_path(str(tmp_path), "docs/a.txt")
    assert p.startswith(str(tmp_path))


def test_confine_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        _confine_path(str(tmp_path), "../../etc/passwd")


def test_factory_is_zero_arg_and_makes_fresh_interpreters(tmp_path):
    factory = InterpreterFactory(str(tmp_path))
    _validate_interpreter_factory(factory)          # dspy accepts it
    a, b = factory(), factory()
    assert a is not b
    assert a.workdir == str(tmp_path)
    assert isinstance(factory.execution_instructions, str)
    assert factory.execution_instructions            # non-empty


def test_factory_accepts_callable_workdir_source():
    current = {"dir": "/tmp/one"}
    factory = InterpreterFactory(lambda: current["dir"])
    assert factory().workdir == "/tmp/one"
    current["dir"] = "/tmp/two"
    assert factory().workdir == "/tmp/two"


# ---- workdir is the resolution base, not decoration -----------------------------------

def test_relative_reads_resolve_against_the_workdir(tmp_path):
    """`workdir` was stored and never used, so the agent read the process CWD instead."""
    from sentinelprime.interpreter import LocalInterpreter

    (tmp_path / "doc.txt").write_text("change of control clause")
    interp = LocalInterpreter(workdir=str(tmp_path))
    interp.start()
    out = interp.execute("print(open('doc.txt').read())")
    assert "change of control clause" in out


def test_relative_writes_land_in_the_workdir(tmp_path):
    """The deliverable contract is a file in ./output — it must not go to the repo root."""
    from sentinelprime.interpreter import LocalInterpreter

    (tmp_path / "output").mkdir()
    interp = LocalInterpreter(workdir=str(tmp_path))
    interp.start()
    interp.execute("open('output/report.md', 'w').write('# Report')")
    assert (tmp_path / "output" / "report.md").read_text() == "# Report"


def test_absolute_paths_are_left_alone(tmp_path):
    from sentinelprime.interpreter import LocalInterpreter

    other = tmp_path / "elsewhere.txt"
    other.write_text("absolute")
    interp = LocalInterpreter(workdir=str(tmp_path / "wd"))
    (tmp_path / "wd").mkdir()
    interp.start()
    assert "absolute" in interp.execute(f"print(open({str(other)!r}).read())")


def test_workdir_is_exposed_to_the_model_as_a_variable(tmp_path):
    """Anything that is not open() — glob, pathlib, subprocess — needs the absolute base."""
    from sentinelprime.interpreter import LocalInterpreter

    interp = LocalInterpreter(workdir=str(tmp_path))
    interp.start()
    assert str(tmp_path) in interp.execute("print(WORKDIR)")


def test_workdir_resolution_does_not_chdir_the_process(tmp_path):
    """Children run on threads; a process-global chdir would clobber the parent."""
    import os
    from sentinelprime.interpreter import LocalInterpreter

    before = os.getcwd()
    interp = LocalInterpreter(workdir=str(tmp_path))
    interp.start()
    interp.execute("open('x.txt', 'w').write('y')")
    assert os.getcwd() == before


def test_no_workdir_keeps_plain_cwd_behaviour(tmp_path, monkeypatch):
    from sentinelprime.interpreter import LocalInterpreter

    monkeypatch.chdir(tmp_path)
    (tmp_path / "here.txt").write_text("cwd")
    interp = LocalInterpreter()
    interp.start()
    assert "cwd" in interp.execute("print(open('here.txt').read())")
