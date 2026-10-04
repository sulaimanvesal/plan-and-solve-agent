"""Tests for plan-and-solve-agent."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from plan_and_solve import (
    Executor,
    MockBackend,
    PlanAndSolveAgent,
    Planner,
    check_coverage,
    extract_answer,
    parse_subtasks,
)
from plan_and_solve.prompts import (
    PLAN_INSTRUCTION,
    build_ps_execute_prompt,
    build_ps_plan_prompt,
    build_psplus_execute_prompt,
    build_psplus_variable_prompt,
)

QUESTION = "Janet has 16 apples. She gives away 7 and buys 12. How many apples does she have now?"


# --- prompt templates -------------------------------------------------------


def test_ps_plan_prompt_contains_paper_wording():
    prompt = build_ps_plan_prompt(QUESTION)
    assert "devise a plan to solve the problem" in prompt
    assert "understand the problem" in prompt
    assert QUESTION in prompt


def test_ps_execute_prompt_contains_carry_out_wording():
    plan = "1. Do A.\n2. Do B."
    prompt = build_ps_execute_prompt(QUESTION, plan)
    assert "carry out the plan and solve the problem step by step" in prompt
    assert plan in prompt


def test_psplus_prompt_extracts_variables_and_numerals():
    prompt = build_psplus_variable_prompt(QUESTION)
    assert "extract relevant variables" in prompt
    assert "numerals" in prompt
    assert "make a complete plan" in prompt


def test_psplus_execute_prompt_warns_about_calculation():
    prompt = build_psplus_execute_prompt(QUESTION, "1. Do A.")
    assert "calculation" in prompt
    assert "commonsense" in prompt
    assert "final answer" in prompt


# --- planner ----------------------------------------------------------------


def test_planner_produces_numbered_subtasks():
    planner = Planner(MockBackend(seed=1))
    plan = planner.plan(QUESTION)
    assert len(plan) >= 2
    assert all(step.strip() for step in plan.subtasks)


def test_parse_subtasks_handles_numbering_styles():
    raw = "1. First step\n2) Second step\nStep 3: Third step"
    assert parse_subtasks(raw) == ["First step", "Second step", "Third step"]


def test_planner_falls_back_to_lines_without_numbering():
    raw = "Think about it\nThen act"
    assert parse_subtasks(raw) == ["Think about it", "Then act"]


# --- executor ---------------------------------------------------------------


def test_executor_fills_plan_placeholders():
    planner = Planner(MockBackend(seed=1))
    plan = planner.plan(QUESTION)
    executor = Executor(MockBackend(seed=1))
    results = executor.execute(plan)
    assert len(results) == len(plan)
    assert all(r.strip() for r in results)


def test_executor_numbered_plan_rendering():
    executor = Executor(MockBackend(seed=1))
    plan = planner_plan_with_steps()
    numbered = executor.numbered_plan(plan)
    lines = numbered.splitlines()
    assert lines[0].startswith("1. ")
    assert lines[1].startswith("2. ")


def planner_plan_with_steps():
    planner = Planner(MockBackend(seed=1))
    return planner.plan(QUESTION)


# --- agent ------------------------------------------------------------------


def test_agent_returns_structured_result():
    agent = PlanAndSolveAgent(MockBackend(seed=3))
    result = agent.solve(QUESTION)
    assert result.question == QUESTION
    assert result.plan and len(result.plan) >= 2
    assert len(result.subtask_results) == len(result.plan)
    assert result.answer is not None
    assert result.variant == "PS"
    d = result.to_dict()
    assert d["answer"] == result.answer
    assert d["coverage"]["complete"] is True


def test_agent_ps_plus_variant_flag():
    agent = PlanAndSolveAgent(MockBackend(seed=3), use_ps_plus=True)
    result = agent.solve(QUESTION)
    assert result.variant == "PS+"
    assert result.answer is not None


def test_agent_deterministic_same_seed():
    r1 = PlanAndSolveAgent(MockBackend(seed=9)).solve(QUESTION)
    r2 = PlanAndSolveAgent(MockBackend(seed=9)).solve(QUESTION)
    assert r1.plan == r2.plan
    assert r1.answer == r2.answer
    assert r1.subtask_results == r2.subtask_results


def test_offline_demo_path_deterministic():
    script = Path(__file__).resolve().parents[1] / "examples" / "demo.py"
    cmd = [sys.executable, str(script), "--quiet"]
    out1 = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    out2 = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert out1 == out2
    assert "answer:" in out1


def test_demo_problems_have_expected_answers():
    agent = PlanAndSolveAgent(MockBackend(seed=7))
    expected = {"apples": "21", "books": "35", "train": "210"}
    problems = {
        "apples": "Janet starts with 16 apples. She gives 3 apples to her friend and 4 apples to her sister. Then she buys 12 more apples at the market. How many apples does Janet have now?",
        "books": "A bookstore sells each book for $8. A customer buys 5 books and gets a $5 discount on the total. How much does the customer pay?",
        "train": "A train travels at 60 miles per hour for 2 hours, then continues at the same speed for another 1.5 hours. What is the total distance the train traveled?",
    }
    for name, q in problems.items():
        result = agent.solve(q)
        assert result.answer == expected[name], (name, result.answer)


# --- answer extraction / coverage -------------------------------------------


def test_extract_answer_from_answer_line():
    assert extract_answer("Some trace\nAnswer: 42") == "42"


def test_extract_answer_falls_back_to_last_number():
    assert extract_answer("First we get 10, then 32") == "32"


def test_extract_answer_none_without_numbers():
    assert extract_answer("no digits here") is None


def test_check_coverage_flags_missing_subtasks():
    report = check_coverage(["ok", "", "ok"])
    assert not report.complete
    assert report.missing == [2]
    assert report.executed_subtasks == 2


def test_check_coverage_complete():
    report = check_coverage(["a", "b"])
    assert report.complete
    assert report.missing == []
