"""Stage 2 — Execution.

The executor sends the question *plus the generated plan* back to the
model with the paper's second-stage instruction — "carry out the plan
and solve the problem step by step" — and maps the execution trace back
onto the individual subtasks so answer extraction and the coverage
self-check can operate per-subtask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .backends import LLMBackend
from .planner import Plan
from .prompts import build_ps_execute_prompt, build_psplus_execute_prompt

_SUBTASK_LINE_RE = re.compile(r"(?im)^\s*subtask\s*\d+\s*[:\-]?\s*(.+)$")


@dataclass
class Executor:
    """Stage-2 executor: plan -> per-subtask execution results."""

    backend: LLMBackend
    use_ps_plus: bool = False

    def numbered_plan(self, plan: Plan) -> str:
        """Render the plan with explicit 1..n numbering (fills placeholders)."""
        return "\n".join(f"{i}. {s}" for i, s in enumerate(plan.subtasks, 1))

    def execute(self, plan: Plan) -> List[str]:
        numbered = self.numbered_plan(plan)
        prompt = (
            build_psplus_execute_prompt(plan.question, numbered)
            if self.use_ps_plus
            else build_ps_execute_prompt(plan.question, numbered)
        )
        trace = self.backend.complete(prompt)
        return self._trace_to_subtask_results(trace, len(plan))

    @staticmethod
    def _trace_to_subtask_results(trace: str, n_subtasks: int) -> List[str]:
        per = _SUBTASK_LINE_RE.findall(trace)
        if len(per) >= n_subtasks:
            return per[:n_subtasks]
        # Fallback: split the trace into lines, one bucket per subtask.
        lines = [ln.strip() for ln in trace.splitlines() if ln.strip()]
        results = [""] * n_subtasks
        for i, line in enumerate(lines):
            bucket = min(i, n_subtasks - 1)
            results[bucket] += (" " if results[bucket] else "") + line
        return results
