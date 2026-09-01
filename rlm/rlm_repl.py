from __future__ import annotations
"""
Recursive Language Model with REPL environment (RLM_REPL).

This is the root orchestrator: it wraps an LLM in an iterative loop where
the model can write Python code, execute it in a REPL, see the output, and
repeat — until it emits FINAL(...) or FINAL_VAR(...).

Usage (drop-in replacement for a plain LLM call):

    rlm = RLM_REPL(model="gpt-4o", recursive_model="gpt-4o-mini")
    answer = rlm.completion(context=my_huge_text, query="What is X?")
"""

from typing import Dict, List, Optional, Any

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.llm import GeminiClient
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils
from rlm.utils.trajectory import parse_trajectory, TrajectoryStep
from rlm.utils.exporter import TrajectoryExporter

from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_REPL(RLM):
    """
    Recursive Language Model that uses a Python REPL environment to
    iteratively explore large contexts before producing a final answer.

    The interface is identical to a plain LLM client:
        rlm.completion(context, query)  ≡  llm.completion(messages)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        recursive_model: str = "gemini-2.0-flash",
        max_iterations: int = 20,
        depth: int = 0,
        enable_logging: bool = False,
    ):
        """
        Args:
            api_key: Google API key (falls back to GOOGLE_API_KEY env var).
            model: Root LM model name.
            recursive_model: Sub-LM model name (used inside the REPL).
            max_iterations: Maximum REPL loop iterations before forcing an answer.
            depth: Recursive depth (unused in depth-1 implementation).
            enable_logging: Enable colourful step-by-step console logging.
        """
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.llm = GeminiClient(api_key, model)

        self.repl_env: Optional[REPLEnv] = None
        self.depth = depth  # reserved for future depth > 1 support
        self._max_iterations = max_iterations

        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)

        self.messages: List[Dict[str, str]] = []
        self.query: Optional[str] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_context(
        self,
        context: List[str] | str | List[Dict[str, str]],
        query: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Initialise the system prompt and REPL environment for a new query.

        Args:
            context: The large context as a string, dict, or list of messages.
            query: The user's question (falls back to a default if None).

        Returns:
            The initial message list.
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        # Build system prompt
        self.messages = build_system_prompt()
        self.logger.log_initial_messages(self.messages)

        # Convert context and load into REPL
        context_data, context_str = utils.convert_context_for_repl(context)
        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
        )

        return self.messages

    # ------------------------------------------------------------------
    # Main completion loop
    # ------------------------------------------------------------------

    def completion(
        self,
        context: List[str] | str | List[Dict[str, str]],
        query: Optional[str] = None,
    ) -> str:
        """
        Given a query and a (potentially enormous) context, recursively call
        the LM to explore and analyse the context, then return a final answer.

        This is the drop-in replacement for a plain `llm.completion(messages)`.
        """
        self.messages = self.setup_context(context, query)

        for iteration in range(self._max_iterations):
            # Query root LM
            response = self.llm.completion(
                self.messages + [next_action_prompt(self.query, iteration)]
            )

            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)

            if code_blocks is not None:
                # Execute REPL code and feed output back into conversation
                self.messages = utils.process_code_execution(
                    response,
                    self.messages,
                    self.repl_env,
                    self.repl_env_logger,
                    self.logger,
                )
            else:
                # No code — just record the assistant turn
                self.messages.append(
                    {"role": "assistant", "content": "You responded with:\n" + response}
                )

            # Check whether the model has emitted a final answer
            final_answer = utils.check_for_final_answer(response, self.repl_env, self.logger)
            if final_answer:
                self.logger.log_final_response(final_answer)
                return final_answer

        # Max iterations reached — force a final answer
        print("Max iterations reached. Forcing final answer.")
        self.messages.append(next_action_prompt(self.query, iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.logger.log_final_response(final_answer)
        return final_answer

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def cost_summary(self) -> Dict[str, Any]:
        raise NotImplementedError("Cost tracking not implemented for RLM_REPL.")

    def get_trajectory(self) -> List[TrajectoryStep]:
        """Parse and return the structured trajectory from message history."""
        return parse_trajectory(self.messages)

    def save_trajectory(self, path: str, format: str = "html"):
        """Save the execution trajectory to a file (json or html)."""
        trajectory = self.get_trajectory()
        if format.lower() == "json":
            TrajectoryExporter.to_json(trajectory, path)
        else:
            TrajectoryExporter.to_html(trajectory, path, query=self.query or "")

    def reset(self):
        """Reset the REPL environment and conversation history."""
        self.repl_env = REPLEnv()
        self.messages = []
        self.query = None


if __name__ == "__main__":
    pass
