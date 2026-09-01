"""
Needle-in-a-haystack demo for Recursive Language Models (RLMs).

Generates ~1 million lines of random filler text, hides a 7-digit magic
number somewhere in the middle, then asks RLM_REPL to find it.

RLM.completion() is a drop-in replacement for a plain LLM completion call:

    rlm.completion(context=context, query=query)
    ≡  llm.completion(messages)

Run:
    cp .env.example .env   # add your OPENAI_API_KEY
    python main.py
"""

import random
from rlm.rlm_repl import RLM_REPL


def generate_massive_context(
    num_lines: int = 1_000_000,
    answer: str = "1298418",
) -> str:
    """
    Generate a large block of random text with one hidden magic number.

    Args:
        num_lines: Total number of lines to generate.
        answer: The magic number string to embed.

    Returns:
        The context as a newline-delimited string.
    """
    print(f"Generating massive context with {num_lines:,} lines...")

    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]

    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        lines.append(" ".join(random.choice(random_words) for _ in range(num_words)))

    # Hide the magic number somewhere in the middle half
    magic_position = random.randint(num_lines // 4, 3 * num_lines // 4)
    lines[magic_position] = f"The magic number is {answer}"

    print(f"Magic number '{answer}' inserted at line {magic_position:,}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Recursive Language Model — Needle-in-a-Haystack Demo")
    print("=" * 60)

    answer = str(random.randint(1_000_000, 9_999_999))
    context = generate_massive_context(num_lines=1_000_000, answer=answer)

    rlm = RLM_REPL(
        model="gemini-2.0-flash",           # root LM
        recursive_model="gemini-2.0-flash",  # sub-LM inside the REPL
        enable_logging=True,
        max_iterations=10,
    )

    query = "I'm looking for a magic number hidden in the context. What is it?"
    print(f"\nQuery: {query}\n")

    result = rlm.completion(context=context, query=query)

    # Save the trajectory for visualization
    trajectory_path = "trajectory.html"
    rlm.save_trajectory(trajectory_path, format="html")
    print(f"\nTrajectory saved to {trajectory_path}")

    print("\n" + "=" * 60)
    print(f"RLM answer : {result}")
    print(f"True answer: {answer}")
    print(f"Correct    : {answer in result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
