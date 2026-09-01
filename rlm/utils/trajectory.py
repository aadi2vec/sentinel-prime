"""
Trajectory parsing for RLM executions.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import re


class TrajectoryStep:
    """A single step in the RLM trajectory."""

    def __init__(
        self,
        iteration: int,
        thought: str,
        code: Optional[str] = None,
        output: Optional[str] = None,
        final_answer: Optional[str] = None,
    ):
        self.iteration = iteration
        self.thought = thought
        self.code = code
        self.output = output
        self.final_answer = final_answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "code": self.code,
            "output": self.output,
            "final_answer": self.final_answer,
        }


def parse_trajectory(messages: List[Dict[str, str]]) -> List[TrajectoryStep]:
    """
    Parse the conversation message history into a structured trajectory.
    
    The history typically looks like:
    1. System Prompt
    2. User: Safeguard + Query (Iteration 0)
    3. Assistant: Thought + ```repl info
    4. User: Code executed + Output
    5. User: Iteration query (Iteration 1)
    6. Assistant: ...
    7. FINAL(...) or FINAL_VAR(...)
    """
    trajectory = []
    
    current_thought = ""
    current_code = None
    current_iteration = 0
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "assistant":
            # Extract code blocks
            code_pattern = r"```repl\s*\n(.*?)\n```"
            match = re.search(code_pattern, content, re.DOTALL)
            
            if match:
                current_code = match.group(1).strip()
                # Thought is everything before the code block
                current_thought = content.split("```repl")[0].strip()
            else:
                # Might be a final answer or just a follow-up
                current_thought = content
            
            # Check for final answer patterns
            final_patterns = [r"FINAL\((.*?)\)", r"FINAL_VAR\((.*?)\)"]
            for pattern in final_patterns:
                f_match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
                if f_match:
                    trajectory.append(TrajectoryStep(
                        iteration=current_iteration,
                        thought=current_thought,
                        final_answer=f_match.group(1).strip()
                    ))
                    break
            else:
                # If no final answer yet, we wait for the REPL output in the next message
                pass

        elif role == "user":
            # Check if this identifies an iteration
            iter_match = re.search(r"to answer the original query: \".*?\"\.\n\nContinue .*? Your next action:", content)
            
            # Check if this is a REPL output
            if "REPL output:" in content:
                output_content = content.split("REPL output:\n")[-1].strip()
                trajectory.append(TrajectoryStep(
                    iteration=current_iteration,
                    thought=current_thought,
                    code=current_code,
                    output=output_content
                ))
                current_iteration += 1
                current_thought = ""
                current_code = None
                
    return trajectory
