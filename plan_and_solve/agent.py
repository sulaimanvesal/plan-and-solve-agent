"""Plan-and-Solve agent: plan → execute → extract answer.

The two-stage loop mirrors the paper exactly:

1. **Plan** — "devise a plan to divide the task into smaller subtasks".
2. **Execute** — "carry out the subtasks according to the plan".
3. **Answer** — extract the final answer and self-check subtask coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .answer import CoverageReport, check_coverage, extract_answer
from .backends import LLMBackend, MockBackend
from .executor import Executor
from .planner import Plan, Planner


@dataclass
class SolveResult:
    """Structured outcome of one plan-and-solve run."""

    question: str
    plan: List[str]
    subtask_results: List[str]
    raw_plan: str
    raw_execution: str
    answer: Optional[str]
    coverage: CoverageReport
    variant: str = "PS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "variant": self.variant,
            "plan": self.plan,
            "subtask_results": self.subtask_results,
            "answer": self.answer,
            "coverage": {
                "expected": self.coverage.expected_subtasks,
                "executed": self.coverage.executed_subtasks,
                "missing": self.coverage.missing,
                "complete": self.coverage.complete,
            },
        }


class PlanAndSolveAgent:
    """Zero-shot plan-and-solve reasoning agent.

    Parameters
    ----------
    backend:
        Any :class:`~plan_and_solve.backends.LLMBackend`. Defaults to
        :class:`~plan_and_solve.backends.MockBackend` so the package works
        offline out of the box.
    use_ps_plus:
        If ``True``, use the PS+ variant (variable/numeral extraction +
        careful-calculation instruction), which the paper shows helps on
        calculation errors.
    """

    def __init__(self, backend: Optional[LLMBackend] = None, *, use_ps_plus: bool = False) -> None:
        self.backend = backend or MockBackend()
        self.use_ps_plus = use_ps_plus
        self.planner = Planner(self.backend, use_ps_plus=use_ps_plus)
        self.executor = Executor(self.backend, use_ps_plus=use_ps_plus)

    def solve(self, question: str) -> SolveResult:
        plan = self.planner.plan(question)
        subtask_results = self.executor.execute(plan)

        # Re-render the numbered plan so we can recover the raw execution
        # text for answer extraction (backends return per-subtask results).
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(plan.subtasks, 1))
        raw_execution = "\n".join(
            f"Subtask {i}: {r}" for i, r in enumerate(subtask_results, 1)
        )

        answer = extract_answer(raw_execution) or extract_answer(plan.raw)
        coverage = check_coverage(subtask_results)

        return SolveResult(
            question=question,
            plan=plan.subtasks,
            subtask_results=subtask_results,
            raw_plan=plan.raw,
            raw_execution=raw_execution,
            answer=answer,
            coverage=coverage,
            variant="PS+" if self.use_ps_plus else "PS",
        )
