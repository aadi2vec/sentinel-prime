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
