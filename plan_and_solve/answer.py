"""Answer extraction and plan-coverage self-check.

After execution, the agent extracts the final answer from the trace and
checks that every subtask received an execution result — the in-code
analogue of the paper's claim that Plan-and-Solve reduces *missing-step*
errors relative to Zero-shot-CoT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


_ANSWER_PATTERNS = [
    re.compile(r"(?im)^\s*(?:the\s+)?answer(?:\s+is)?\s*[:\-]?\s*(.+?)\s*$"),
    re.compile(r"(?im)\banswer\s*[:\-=]\s*(.+?)\s*$"),
]


def extract_answer(trace: str) -> Optional[str]:
    """Pull the final answer from an execution trace.

    Looks for an explicit ``Answer: ...`` line first, then falls back to
    the last bare number in the trace.
    """
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(trace)
        if matches:
            return matches[-1].strip().rstrip(".")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", trace)
    return numbers[-1] if numbers else None


@dataclass
class CoverageReport:
    """Result of the subtask-coverage self-check."""

    expected_subtasks: int
    executed_subtasks: int
    missing: List[int] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def check_coverage(subtask_results: List[str]) -> CoverageReport:
    """Verify every planned subtask produced a non-empty execution result."""
    missing = [i + 1 for i, r in enumerate(subtask_results) if not r.strip()]
    return CoverageReport(
        expected_subtasks=len(subtask_results),
        executed_subtasks=len(subtask_results) - len(missing),
        missing=missing,
    )
