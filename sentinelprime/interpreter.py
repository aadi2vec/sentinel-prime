"""In-process CodeInterpreter for dspy.RLM.

This replaces DSPy's default Deno/Pyodide WASM interpreter (which has no
filesystem or network) with a plain in-process Python REPL, so the RLM can
read the task's document bundle and shell out with `subprocess` directly.

SECURITY: this is NOT a sandbox. Executed code runs with THIS process's OS
permissions — full filesystem and subprocess access. It is a durable control
environment for a trusted agent, not an isolation boundary. Never run it
against untrusted input.
"""
from __future__ import annotations

import contextlib
import io
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeExecutionError, FinalOutput


class _SubmitSignal(Exception):
    """Raised by the injected SUBMIT() to unwind the exec and carry outputs."""

    def __init__(self, output: dict):
        self.output = output


class LocalInterpreter:
    """A CodeInterpreter that execs model code in a persistent namespace."""

    def __init__(self) -> None:
        self._ns: dict[str, Any] | None = None
        self._tools: dict[str, Callable[..., Any]] = {}
        # dspy.RLM sets output_fields (SUBMIT field metadata) and toggles
        # _tools_registered before each forward pass; expose both.
        self.output_fields: list[dict] = []
        self._tools_registered = False

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        return self._tools

    def start(self) -> None:
        if self._ns is None:
            self._ns = {"__name__": "__rlm__"}
        self._register()

    def _register(self) -> None:
        assert self._ns is not None

        def SUBMIT(**kwargs: Any):
            raise _SubmitSignal(kwargs)

        self._ns["SUBMIT"] = SUBMIT
        self._ns.update(self._tools)
        self._tools_registered = True

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        if self._ns is None:
            self.start()
        if not self._tools_registered:
            self._register()
        if variables:
            self._ns.update(variables)

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<rlm>", "exec"), self._ns)
        except _SubmitSignal as sig:
            return FinalOutput(sig.output)
        except SyntaxError:
            raise
        except Exception as exc:  # runtime error in model code -> recoverable
            raise CodeExecutionError(str(exc)) from exc

        out = buf.getvalue()
        return out if out else None

    def shutdown(self) -> None:
        self._ns = None
        self._tools_registered = False
