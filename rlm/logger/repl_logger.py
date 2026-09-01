"""
REPL environment logger — renders Jupyter-notebook-style cells using `rich`.
"""

from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.rule import Rule


@dataclass
class CodeExecution:
    code: str
    stdout: str
    stderr: str
    execution_number: int
    execution_time: Optional[float] = None


class REPLEnvLogger:
    """Renders REPL executions like Jupyter notebook cells using `rich`."""

    def __init__(self, max_output_length: int = 2000, enabled: bool = True):
        self.enabled = enabled
        self.console = Console()
        self.executions: List[CodeExecution] = []
        self.execution_count = 0
        self.max_output_length = max_output_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_length:
            return text
        half = self.max_output_length // 2
        dropped = len(text) - self.max_output_length
        return f"{text[:half]}\n\n... [TRUNCATED {dropped} characters] ...\n\n{text[-half:]}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_execution(
        self,
        code: str,
        stdout: str,
        stderr: str = "",
        execution_time: Optional[float] = None,
    ) -> None:
        """Record a code execution."""
        self.execution_count += 1
        self.executions.append(
            CodeExecution(
                code=code,
                stdout=stdout,
                stderr=stderr,
                execution_number=self.execution_count,
                execution_time=execution_time,
            )
        )

    def display_last(self) -> None:
        """Display the most recently logged execution."""
        if not self.enabled or not self.executions:
            return
        self._render(self.executions[-1])

    def display_all(self) -> None:
        """Display all logged executions."""
        if not self.enabled:
            return
        for i, execution in enumerate(self.executions):
            self._render(execution)
            if i < len(self.executions) - 1:
                self.console.print(Rule(style="dim", characters="─"))
                self.console.print()

    def clear(self) -> None:
        self.executions.clear()
        self.execution_count = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, execution: CodeExecution) -> None:
        if not self.enabled:
            return

        # Input cell
        self.console.print(
            Panel(
                Syntax(
                    self._truncate(execution.code),
                    "python",
                    theme="monokai",
                    line_numbers=True,
                ),
                title=f"[bold blue]In [{execution.execution_number}]:[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )

        # Output cell
        timing_panel = None

        if execution.stderr:
            output_panel = Panel(
                Text(self._truncate(execution.stderr), style="bold red"),
                title=f"[bold red]Error in [{execution.execution_number}]:[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        elif execution.stdout:
            output_panel = Panel(
                Text(self._truncate(execution.stdout), style="white"),
                title=f"[bold green]Out [{execution.execution_number}]:[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
            if execution.execution_time is not None:
                timing_panel = Panel(
                    Text(f"Execution time: {execution.execution_time:.4f}s", style="bright_black"),
                    border_style="grey37",
                    box=box.ROUNDED,
                    title=f"[bold grey37]Timing [{execution.execution_number}]:[/bold grey37]",
                )
        else:
            if execution.execution_time is not None:
                output_panel = Panel(
                    Text(f"Execution time: {execution.execution_time:.4f}s", style="dim"),
                    title=f"[bold dim]Out [{execution.execution_number}]:[/bold dim]",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            else:
                output_panel = Panel(
                    Text("No output", style="dim"),
                    title=f"[bold dim]Out [{execution.execution_number}]:[/bold dim]",
                    border_style="dim",
                    box=box.ROUNDED,
                )

        self.console.print(output_panel)
        if timing_panel:
            self.console.print(timing_panel)
