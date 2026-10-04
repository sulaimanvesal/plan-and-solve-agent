"""Offline demo: solve multi-step math word problems with Plan-and-Solve.

Runs entirely offline on MockBackend — no API key, no network.

    python examples/demo.py                 # PS on 3 demo problems
    python examples/demo.py --variant ps-plus
    python examples/demo.py --backend openai  # live run, needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_and_solve import MockBackend, OpenAIBackend, PlanAndSolveAgent  # noqa: E402

DEMO_PROBLEMS = [
    (
        "apples",
        "Janet starts with 16 apples. She gives 3 apples to her friend and "
        "4 apples to her sister. Then she buys 12 more apples at the market. "
        "How many apples does Janet have now?",
    ),
    (
        "books",
        "A bookstore sells each book for $8. A customer buys 5 books and "
        "gets a $5 discount on the total. How much does the customer pay?",
    ),
    (
        "train",
        "A train travels at 60 miles per hour for 2 hours, then continues "
        "at the same speed for another 1.5 hours. What is the total distance "
        "the train traveled?",
    ),
]


def print_result(result, *, verbose: bool = True) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"[{result.variant}] {result.question}")
    print("-" * 64)
    print("PLAN:")
    for i, step in enumerate(result.plan, 1):
        print(f"  {i}. {step}")
    print("-" * 64)
    print("EXECUTION:")
    for i, r in enumerate(result.subtask_results, 1):
        print(f"  Subtask {i}: {r}")
    print("-" * 64)
    print(f"ANSWER: {result.answer}")
    cov = result.coverage
    print(
        f"COVERAGE: {cov.executed_subtasks}/{cov.expected_subtasks} subtasks "
        f"executed {'(complete)' if cov.complete else f'(missing: {cov.missing})'}"
    )
    if verbose:
        print(bar)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan-and-Solve offline demo")
    parser.add_argument(
        "--variant",
        choices=["ps", "ps-plus"],
        default="ps",
        help="Prompt variant (default: ps)",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "openai"],
        default="mock",
        help="LLM backend (default: mock, fully offline)",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print answers")
    args = parser.parse_args()

    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("error: OPENAI_API_KEY is not set", file=sys.stderr)
            return 2
        backend = OpenAIBackend()
    else:
        backend = MockBackend(seed=7)

    agent = PlanAndSolveAgent(backend=backend, use_ps_plus=(args.variant == "ps-plus"))

    for name, question in DEMO_PROBLEMS:
        result = agent.solve(question)
        if args.quiet:
            print(f"[{name}] answer: {result.answer}")
        else:
            print_result(result, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
