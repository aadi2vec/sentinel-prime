"""
Utility functions for the RLM REPL client.
"""

import re
from typing import List, Dict, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Code block parsing
# ---------------------------------------------------------------------------

def find_code_blocks(text: str) -> Optional[List[str]]:
    """
    Extract all ```repl ... ``` code blocks from the model response.

    Returns:
        A list of code strings, or None if no blocks were found.
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    results = [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]
    return results if results else None


# ---------------------------------------------------------------------------
# Final-answer detection
# ---------------------------------------------------------------------------

def find_final_answer(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect FINAL(...) or FINAL_VAR(...) in the model response.

    Returns:
        A tuple (type, content) where type is 'FINAL' or 'FINAL_VAR',
        or None if neither pattern is present.
    """
    # FINAL_VAR must be checked first to avoid matching FINAL inside FINAL_VAR
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ("FINAL_VAR", match.group(1).strip())

    final_pattern = r"^\s*FINAL\((.*?)\)"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ("FINAL", match.group(1).strip())

    return None


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def add_execution_result_to_messages(
    messages: List[Dict[str, str]],
    code: str,
    result: str,
    max_character_length: int = 100_000,
) -> List[Dict[str, str]]:
    """
    Append the REPL execution result to the conversation history.

    Args:
        messages: Current message list.
        code: The Python code that was executed.
        result: Formatted output from the REPL.
        max_character_length: Hard cap on returned output length.

    Returns:
        Updated message list.
    """
    if len(result) > max_character_length:
        result = result[:max_character_length] + "..."

    execution_message = {
        "role": "user",
        "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
    }
    messages.append(execution_message)
    return messages


def format_execution_result(
    stdout: str,
    stderr: str,
    locals_dict: Dict[str, Any],
    truncate_length: int = 100,
) -> str:
    """
    Format REPL stdout/stderr and a subset of local variables for display.

    Args:
        stdout: Standard output captured during execution.
        stderr: Standard error captured during execution.
        locals_dict: Variables in the REPL namespace after execution.
        truncate_length: Maximum chars shown per variable value.

    Returns:
        A formatted string ready to be fed back to the model.
    """
    result_parts = []

    if stdout:
        result_parts.append(f"\n{stdout}")

    if stderr:
        result_parts.append(f"\n{stderr}")

    # Show non-private variables with simple types
    important_vars: Dict[str, str] = {}
    for key, value in locals_dict.items():
        if key.startswith("_") or key in {"__builtins__", "__name__", "__doc__"}:
            continue
        try:
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                if isinstance(value, str) and len(value) > truncate_length:
                    important_vars[key] = f"'{value[:truncate_length]}...'"
                else:
                    important_vars[key] = repr(value)
        except Exception:
            important_vars[key] = f"<{type(value).__name__}>"

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def execute_code(repl_env, code: str, repl_env_logger, logger) -> str:
    """
    Execute a code string in the REPL environment and return a formatted result.

    Args:
        repl_env: The active REPLEnv instance.
        code: Python code to run.
        repl_env_logger: Logger for rendering notebook-style output.
        logger: Root logger for high-level events.

    Returns:
        Formatted execution result string.
    """
    try:
        result = repl_env.code_execution(code)

        formatted_result = format_execution_result(
            result.stdout, result.stderr, result.locals
        )

        repl_env_logger.log_execution(code, result.stdout, result.stderr, result.execution_time)
        repl_env_logger.display_last()

        logger.log_tool_execution("CODE_EXECUTION", formatted_result)

        return formatted_result

    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        return error_msg


def process_code_execution(
    response: str,
    messages: List[Dict[str, str]],
    repl_env,
    repl_env_logger,
    logger,
) -> List[Dict[str, str]]:
    """
    Extract all ```repl blocks from the response, execute them, and update messages.

    Args:
        response: Raw model response text.
        messages: Current conversation history.
        repl_env: Active REPLEnv.
        repl_env_logger: REPL logger.
        logger: Root logger.

    Returns:
        Updated message list with execution results appended.
    """
    code_blocks = find_code_blocks(response)
    if code_blocks:
        for code in code_blocks:
            execution_result = execute_code(repl_env, code, repl_env_logger, logger)
            messages = add_execution_result_to_messages(messages, code, execution_result)
    return messages


def check_for_final_answer(response: str, repl_env, logger) -> Optional[str]:
    """
    Check the model response for a FINAL / FINAL_VAR declaration.

    Returns:
        The final answer string, or None if the model hasn't finished yet.
    """
    result = find_final_answer(response)
    if result is None:
        return None

    answer_type, content = result

    if answer_type == "FINAL":
        return content

    if answer_type == "FINAL_VAR":
        variable_name = content.strip().strip('"').strip("'").strip("\n").strip("\r")
        if variable_name in repl_env.locals:
            return str(repl_env.locals[variable_name])
        else:
            error_msg = f"Variable '{variable_name}' not found in REPL environment"
            logger.log_tool_execution("FINAL_VAR", error_msg)
            return None

    return None


# ---------------------------------------------------------------------------
# Context conversion
# ---------------------------------------------------------------------------

def convert_context_for_repl(context) -> Tuple[Optional[Any], Optional[str]]:
    """
    Normalise the user-supplied context into (context_json, context_str).

    - str  → (None, context)
    - dict → (context, None)
    - list of message dicts with 'content' key → extract contents as list
    - list of other dicts / primitives → (context, None)
    """
    if isinstance(context, dict):
        return context, None

    if isinstance(context, str):
        return None, context

    if isinstance(context, list):
        if context and isinstance(context[0], dict):
            if "content" in context[0]:
                return [msg.get("content", "") for msg in context], None
            return context, None
        return context, None

    return context, None
