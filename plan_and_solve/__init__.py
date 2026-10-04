"""Plan-and-Solve agent — a runnable implementation of
"Plan-and-Solve Prompting" (Wang et al., ACL 2023, arXiv:2305.04091).
"""

from .agent import PlanAndSolveAgent, SolveResult
from .answer import CoverageReport, check_coverage, extract_answer
from .backends import LLMBackend, MockBackend, OpenAIBackend
from .executor import Executor
from .planner import Plan, Planner, parse_subtasks
from .prompts import (
    build_ps_execute_prompt,
    build_ps_plan_prompt,
    build_psplus_execute_prompt,
    build_psplus_variable_prompt,
)

__all__ = [
    "PlanAndSolveAgent",
    "SolveResult",
    "Plan",
    "Planner",
    "Executor",
    "MockBackend",
    "OpenAIBackend",
    "LLMBackend",
    "CoverageReport",
    "check_coverage",
    "extract_answer",
    "parse_subtasks",
    "build_ps_plan_prompt",
    "build_ps_execute_prompt",
    "build_psplus_variable_prompt",
    "build_psplus_execute_prompt",
]

__version__ = "0.1.0"
