# Recursive Language Models (RLMs)

This project is a Python implementation of **Recursive Language Models (RLMs)**, an inference strategy where language models can decompose and recursively interact with input context of unbounded length through REPL environments.

It is based on the research and minimal implementation by [Alex L. Zhang](https://alexzhang13.github.io/blog/2025/rlm/).

## Project Structure

```
rlm_project/
├── main.py                        # Needle-in-a-haystack demo entry point
├── requirements.txt               # Dependencies (google-generativeai, rich, dotenv)
├── .env                           # Your API keys (rename from .env.example)
└── rlm/                           # Core package
    ├── rlm.py                     # Abstract Base Class for RLMs
    ├── repl.py                    # REPL Environment and Sub-LM implementation
    ├── rlm_repl.py                # Main RLM class with REPL orchestration
    ├── logger/                   # Logging utilities
    │   ├── root_logger.py         # Colorful step-by-step terminal logger
    │   └── repl_logger.py         # Rich-based Jupyter-like REPL display
    └── utils/                    # Utility functions
        ├── llm.py                 # Gemini API client wrapper
        ├── prompts.py             # System and recursive prompt templates
        └── utils.py               # Parsing and execution helpers
```

## How It Works

RLMs solve the "context rot" and "context window" limitations of standard LLMs by never showing the root model the entire context directly. Instead:

1.  **Iterative Loop**: The root model receives only the user query and a tiny system prompt.
2.  **REPL Environment**: The context is loaded into a Python REPL environment as a variable.
3.  **Active Exploration**: The root model writes Python code in ` ```repl ` blocks to:
    -   Peek at substrings of the context.
    -   Search for keywords using regex.
    -   Chunk the context and call `llm_query()` on specific parts (recursive sub-calls).
4.  **Final Answer**: Once the root model identifies the information it needs, it returns a final answer using `FINAL(answer)` or `FINAL_VAR(variable_name)`.

## Design Decisions

-   **Provider**: Adopted to use **Google Gemini** (v1.5 Flash/Flash-8B) for high performance and long context support.
-   **Architecture**: Follows a "Depth-1" recursion model where a root agent calls a standard LLM inside the REPL. This can be extended to deeper recursion by nesting `RLM_REPL` instances.
-   **Logging**: Includes a custom rich-rendered logger and support for exporting **trajectories** (the full sequence of thoughts, code, and results) to JSON or HTML for post-run analysis.
-   **Compatibility**: Uses `from __future__ import annotations` to support Python 3.9+ typing.

## Getting Started

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set up environment:
    ```bash
    cp .env.example .env
    # Edit .env and add your GOOGLE_API_KEY
    ```
3.  Run the demo:
    ```bash
    python3 main.py
    ```

The demo generates a 1-million-line "haystack" of random text with a hidden 7-digit number and tasks the RLM with finding it.
