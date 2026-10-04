"""Plan-and-Solve (PS / PS+) prompt templates.

Implements the two-stage zero-shot prompting scheme from:
    "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning
     by Large Language Models" (Wang et al., ACL 2023, arXiv:2305.04091).

Stage 1 — Planning:   elicit a plan that divides the task into subtasks.
Stage 2 — Execution:  carry out the subtasks according to the plan.

The wording of the PS prompt is a faithful rendering of the paper's:
    "Let's first understand the problem and devise a plan to solve the
     problem. Then, let's carry out the plan and solve the problem step
     by step."
"""

PLAN_INSTRUCTION = (
    "Let's first understand the problem and devise a plan to solve the problem."
)

EXECUTE_INSTRUCTION = (
    "Then, let's carry out the plan and solve the problem step by step."
)

# PS (Plan-and-Solve): plan, then execute.
PS_PLAN_PROMPT = """{question}

{plan_instruction} List the subtasks as numbered steps (1., 2., 3., ...)."""

PS_EXECUTE_PROMPT = """{question}

Plan:
{plan}

{execute_instruction} Work through each subtask in order and show the final answer."""

# PS+ (Plan-and-Solve Plus): additionally extracts variables and numerals
# before planning, instructs careful numeral calculation, and asks the model
# to show its work. This is the paper's extension targeting calculation
# errors (PS primarily targets missing-step errors).
PSPLUS_VARIABLE_PROMPT = """{question}

Let's first understand the problem, extract relevant variables and their \
numerals, and make a complete plan. Then, let's carry out the plan, \
calculate intermediate results, pay attention to calculation and \
commonsense, and solve the problem step by step."""

PSPLUS_EXECUTE_PROMPT = """{question}

Plan:
{plan}

Let's carry out the plan, calculate intermediate results (pay attention to \
calculation and commonsense), and solve the problem step by step. Show the \
final answer."""


def build_ps_plan_prompt(question: str) -> str:
    """Stage-1 prompt: ask the model to devise a plan for ``question``."""
    return PS_PLAN_PROMPT.format(
        question=question, plan_instruction=PLAN_INSTRUCTION
    )


def build_ps_execute_prompt(question: str, plan: str) -> str:
    """Stage-2 prompt: carry out ``plan`` for ``question``."""
    return PS_EXECUTE_PROMPT.format(
        question=question, plan=plan, execute_instruction=EXECUTE_INSTRUCTION
    )


def build_psplus_variable_prompt(question: str) -> str:
    """PS+ stage-1 prompt: extract variables/numerals, then plan."""
    return PSPLUS_VARIABLE_PROMPT.format(question=question)


def build_psplus_execute_prompt(question: str, plan: str) -> str:
    """PS+ stage-2 prompt: execute with careful calculation."""
    return PSPLUS_EXECUTE_PROMPT.format(question=question, plan=plan)
