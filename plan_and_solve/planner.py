"""Stage 1 — Planning.

The planner sends the question with the paper's Plan-and-Solve prompt and
parses the model's response into an ordered list of subtasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .backends import LLMBackend
from .prompts import build_ps_plan_prompt, build_psplus_variable_prompt


@dataclass
class Plan:
    """An ordered list of subtasks produced by stage 1."""

    question: str
    subtasks: List[str] = field(default_factory=list)
    raw: str = ""

    def __len__(self) -> int:
        return len(self.subtasks)


_STEP_RE = re.compile(r"^\s*(?:step\s*)?(\d+)[\.\)\:]?\s+(.*)$", re.IGNORECASE)


def parse_subtasks(raw_plan: str) -> List[str]:
    """Extract numbered subtasks from a plan response.

    Accepts forms like ``1. Do X``, ``1) Do X``, ``Step 1: Do X``.
    Falls back to non-empty lines when no numbering is found.
    """
    subtasks: List[str] = []
    for line in raw_plan.splitlines():
        m = _STEP_RE.match(line.strip())
        if m:
            subtasks.append(m.group(2).strip())
    if not subtasks:
        subtasks = [line.strip() for line in raw_plan.splitlines() if line.strip()]
    return subtasks


@dataclass
class Planner:
    """Stage-1 planner: question -> ordered subtasks."""

    backend: LLMBackend
    use_ps_plus: bool = False

    def plan(self, question: str) -> Plan:
        prompt = (
            build_psplus_variable_prompt(question)
            if self.use_ps_plus
            else build_ps_plan_prompt(question)
        )
        raw = self.backend.complete(prompt)
        return Plan(question=question, subtasks=parse_subtasks(raw), raw=raw)
