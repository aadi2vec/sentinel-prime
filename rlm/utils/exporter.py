"""
Trajectory export utilities (JSON/HTML).
"""

from __future__ import annotations
import json
import os
from typing import List, Any
from rlm.utils.trajectory import TrajectoryStep


class TrajectoryExporter:
    """Exports structured RLM trajectories to various formats."""

    @staticmethod
    def to_json(trajectory: List[TrajectoryStep], path: str):
        """Save trajectory as a JSON file."""
        data = [step.to_dict() for step in trajectory]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def to_html(trajectory: List[TrajectoryStep], path: str, query: str = ""):
        """Save trajectory as a simple, pretty HTML file."""
        steps_html = ""
        for step in trajectory:
            step_html = f"""
            <div class="step">
                <h3>Step {step.iteration}</h3>
                <div class="thought"><strong>Thought:</strong> {step.thought}</div>
            """
            if step.code:
                step_html += f"""
                <div class="code"><strong>Code:</strong> <pre><code>{step.code}</code></pre></div>
                <div class="output"><strong>Result:</strong> <pre>{step.output}</pre></div>
                """
            if step.final_answer:
                step_html += f"""
                <div class="final"><strong>Final Answer:</strong> {step.final_answer}</div>
                """
            step_html += "</div><hr>"
            steps_html += step_html

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RLM Trajectory</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica; line-height: 1.5; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; }}
                .step {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 13px; }}
                code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }}
                .thought {{ margin-bottom: 10px; white-space: pre-wrap; }}
                .code {{ margin-top: 15px; color: #005cc5; }}
                .output {{ margin-top: 10px; color: #22863a; }}
                .final {{ margin-top: 15px; font-size: 1.2em; color: #d73a49; border-top: 2px solid #eaecef; padding-top: 10px; }}
                h1 {{ color: #24292e; }}
                h3 {{ margin-top: 0; color: #586069; }}
            </style>
        </head>
        <body>
            <h1>RLM Trajectory</h1>
            <p><strong>Query:</strong> {query}</p>
            <hr>
            {steps_html}
        </body>
        </html>
        """
        with open(path, "w") as f:
            f.write(html_content)
