from __future__ import annotations
"""
REPL environment for Recursive Language Models.

Contains:
- Sub_RLM   : a lightweight inner LM used inside the REPL as llm_query()
- REPLResult: dataclass for execution output
- REPLEnv   : exec-based notebook-style REPL that pre-loads context and
              exposes llm_query() / FINAL_VAR() to the model's code
"""

import sys
import io
import threading
import json
import tempfile
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from rlm import RLM


# ---------------------------------------------------------------------------
# Sub-LM (depth = 1)
# ---------------------------------------------------------------------------

class Sub_RLM(RLM):
    """
    Minimal, non-recursive LM used inside the REPL environment as `llm_query`.

    Replacing this class with `RLM_REPL` makes the system truly recursive at
    depth > 1, but requires REPL environments to be composable.
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        self.model = model
        from rlm.utils.llm import GeminiClient
        self.client = GeminiClient(api_key=self.api_key, model=model)

    def completion(self, prompt) -> str:
        """Issue a single LM call (string or message list)."""
        try:
            return self.client.completion(messages=prompt, timeout=300)
        except Exception as e:
            return f"Error making LLM query: {str(e)}"

    def cost_summary(self) -> dict:
        raise NotImplementedError("Cost tracking not implemented for Sub_RLM.")

    def reset(self):
        raise NotImplementedError("Reset not implemented for Sub_RLM.")


# ---------------------------------------------------------------------------
# REPL result dataclass
# ---------------------------------------------------------------------------

@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: Optional[float] = None

    def __str__(self):
        return (
            f"REPLResult(stdout={self.stdout!r}, stderr={self.stderr!r}, "
            f"locals={list(self.locals.keys())}, execution_time={self.execution_time})"
        )


# ---------------------------------------------------------------------------
# REPL environment
# ---------------------------------------------------------------------------

class REPLEnv:
    """
    Exec-based Python REPL that:
      - Pre-loads the user context into a `context` variable
      - Exposes `llm_query(prompt)` for recursive sub-LM calls
      - Exposes `FINAL_VAR(varname)` for retrieving built-up answer variables
      - Captures stdout / stderr per execution (thread-safe)
      - Auto-prints the last expression of each cell (Jupyter-style)
    """

    # Safe built-ins passed to exec — deliberately limited
    _SAFE_BUILTINS = {
        "print": print, "len": len, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
        "type": type, "isinstance": isinstance, "issubclass": issubclass,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "range": range,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "pow": pow, "divmod": divmod,
        "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
        "repr": repr, "ascii": ascii, "format": format,
        "any": any, "all": all,
        "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
        "delattr": delattr, "dir": dir, "vars": vars,
        "slice": slice, "iter": iter, "next": next,
        "hash": hash, "id": id, "callable": callable,
        "complex": complex, "bytes": bytes, "bytearray": bytearray,
        "memoryview": memoryview,
        "super": super, "object": object,
        "property": property, "staticmethod": staticmethod, "classmethod": classmethod,
        # Allow imports and file access
        "__import__": __import__, "open": open,
        # Exception hierarchy
        "BaseException": BaseException, "Exception": Exception,
        "ValueError": ValueError, "TypeError": TypeError,
        "KeyError": KeyError, "IndexError": IndexError,
        "AttributeError": AttributeError, "FileNotFoundError": FileNotFoundError,
        "OSError": OSError, "IOError": IOError, "RuntimeError": RuntimeError,
        "NameError": NameError, "ImportError": ImportError,
        "ArithmeticError": ArithmeticError, "LookupError": LookupError,
        "EnvironmentError": EnvironmentError, "AssertionError": AssertionError,
        "NotImplementedError": NotImplementedError, "UnicodeError": UnicodeError,
        "StopIteration": StopIteration, "GeneratorExit": GeneratorExit,
        "SystemExit": SystemExit, "KeyboardInterrupt": KeyboardInterrupt,
        # Warnings
        "Warning": Warning, "UserWarning": UserWarning,
        "DeprecationWarning": DeprecationWarning,
        "PendingDeprecationWarning": PendingDeprecationWarning,
        "SyntaxWarning": SyntaxWarning, "RuntimeWarning": RuntimeWarning,
        "FutureWarning": FutureWarning, "ImportWarning": ImportWarning,
        "ResourceWarning": ResourceWarning,
        # Blocked
        "input": None, "eval": None, "exec": None,
        "compile": None, "globals": None, "locals": None,
    }

    def __init__(
        self,
        recursive_model: str = "gemini-2.0-flash",
        context_json: Optional[dict | list] = None,
        context_str: Optional[str] = None,
        setup_code: Optional[str] = None,
    ):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="repl_env_")

        # Inner sub-LM for recursive calls
        self.sub_rlm: RLM = Sub_RLM(model=recursive_model)

        # Execution namespace
        self.globals: dict = {"__builtins__": dict(self._SAFE_BUILTINS)}
        self.locals: dict = {}
        self._lock = threading.Lock()

        # Load context into namespace
        self.load_context(context_json, context_str)

        # Inject helper functions
        def llm_query(prompt: str) -> str:
            """Query the recursive sub-LM."""
            return self.sub_rlm.completion(prompt)

        self.globals["llm_query"] = llm_query

        def FINAL_VAR(variable_name: str) -> str:  # noqa: N802 — name must match prompt
            """Retrieve a REPL variable as the final answer."""
            variable_name = variable_name.strip().strip('"').strip("'").strip()
            if variable_name in self.locals:
                return str(self.locals[variable_name])
            return f"Error: Variable '{variable_name}' not found in REPL environment"

        self.globals["FINAL_VAR"] = FINAL_VAR

        if setup_code:
            self.code_execution(setup_code)

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def load_context(
        self,
        context_json: Optional[dict | list] = None,
        context_str: Optional[str] = None,
    ):
        """Write context to a temp file and load it into the REPL namespace."""
        if context_json is not None:
            context_path = os.path.join(self.temp_dir, "context.json")
            with open(context_path, "w") as f:
                json.dump(context_json, f, indent=2)
            self.code_execution(
                f"import json\n"
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = json.load(f)\n"
            )

        if context_str is not None:
            context_path = os.path.join(self.temp_dir, "context.txt")
            with open(context_path, "w") as f:
                f.write(context_str)
            self.code_execution(
                f"import os\n"
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = f.read()\n"
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @contextmanager
    def _capture_output(self):
        """Thread-safe stdout / stderr capture."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            try:
                sys.stdout = stdout_buf
                sys.stderr = stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    @contextmanager
    def _temp_working_directory(self):
        """Temporarily switch to the REPL's temp dir."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def code_execution(self, code: str) -> REPLResult:
        """
        Execute a code string notebook-style:
        - Imports are hoisted so they persist across cells.
        - The last non-comment expression is auto-printed (like Jupyter).
        - Variables created in a cell are merged back into self.locals.
        """
        start_time = time.time()

        with self._capture_output() as (stdout_buf, stderr_buf):
            with self._temp_working_directory():
                try:
                    lines = code.split("\n")
                    import_lines = []
                    other_lines = []
                    for line in lines:
                        if line.startswith(("import ", "from ")) and not line.startswith("#"):
                            import_lines.append(line)
                        else:
                            other_lines.append(line)

                    # Execute imports into globals so they persist
                    if import_lines:
                        exec("\n".join(import_lines), self.globals, self.globals)

                    if other_lines:
                        combined = {**self.globals, **self.locals}
                        non_comment = [
                            l for l in other_lines if l.strip() and not l.startswith("#")
                        ]

                        if non_comment:
                            last = non_comment[-1]
                            is_expr = (
                                not last.startswith(
                                    (
                                        "import ", "from ", "def ", "class ",
                                        "if ", "for ", "while ", "try:", "with ",
                                        "return ", "yield ", "break", "continue", "pass",
                                    )
                                )
                                and "=" not in last.split("#")[0]
                                and not last.endswith(":")
                                and not last.startswith("print(")
                            )

                            if is_expr:
                                # Try to eval the last line and print its result
                                try:
                                    last_idx = next(
                                        i for i, l in enumerate(other_lines) if l == last
                                    )
                                    if last_idx > 0:
                                        exec(
                                            "\n".join(other_lines[:last_idx]),
                                            combined,
                                            combined,
                                        )
                                    result_val = eval(last, combined, combined)
                                    if result_val is not None:
                                        print(repr(result_val))
                                except Exception:
                                    exec("\n".join(other_lines), combined, combined)
                            else:
                                exec("\n".join(other_lines), combined, combined)

                        else:
                            exec("\n".join(other_lines), combined, combined)

                        # Merge newly defined names back into locals
                        for key, value in combined.items():
                            if key not in self.globals:
                                self.locals[key] = value

                    stdout_content = stdout_buf.getvalue()
                    stderr_content = stderr_buf.getvalue()

                except Exception as e:
                    stdout_content = stdout_buf.getvalue()
                    stderr_content = stderr_buf.getvalue() + str(e)

        execution_time = time.time() - start_time
        self.locals["_stdout"] = stdout_content
        self.locals["_stderr"] = stderr_content

        return REPLResult(stdout_content, stderr_content, self.locals.copy(), execution_time)

    def get_cost_summary(self):
        raise NotImplementedError("Cost tracking not implemented for REPLEnv.")
