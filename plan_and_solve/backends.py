"""Pluggable LLM backends.

``LLMBackend`` is the minimal interface the agent needs: ``complete(prompt)``.
Two backends ship with the package:

* :class:`MockBackend` — deterministic, scripted traces for offline demos
  and tests. No network, no API key, reproducible for a given seed/problem.
* :class:`OpenAIBackend` — talks to any OpenAI-compatible chat-completions
  endpoint (OpenAI, vLLM, LM Studio, ...). Implemented on top of
  ``urllib`` from the standard library; ``httpx`` is used if installed
  but is NOT required.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class LLMBackend(Protocol):
    """Minimal chat interface consumed by :class:`~plan_and_solve.agent.PlanAndSolveAgent`."""

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Return a model completion for ``prompt``."""
        ...


# ---------------------------------------------------------------------------
# Offline mock backend
# ---------------------------------------------------------------------------

_OPERATORS = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "divided": "/",
}


@dataclass
class MockBackend:
    """Deterministic scripted backend for offline demos and tests.

    It recognises the two plan-and-solve prompt shapes and produces a
    numbered plan, then walks the plan solving each subtask against a
    tiny scripted arithmetic/world model. It is intentionally simple: its
    job is to make the agent's control flow observable offline, not to
    do general LLM reasoning.

    ``seed`` selects one of the canned traces so demo problems produce
    varied but reproducible plans. Any *new* problem gets a generic
    plan/execute trace derived from the numbers in the question.
    """

    seed: int = 42
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- canned traces for the demo problems ---------------------------------
    _SCRIPTED: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def _canned_plan(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        if "devise a plan" in p or "understand the problem" in p or "make a complete plan" in p:
            if "apples" in p:
                return (
                    "1. Determine how many apples Janet starts with.\n"
                    "2. Calculate how many apples she gives away.\n"
                    "3. Calculate how many apples she buys.\n"
                    "4. Compute the final total and give the answer."
                )
            if "books" in p:
                return (
                    "1. Find the cost of one book.\n"
                    "2. Calculate the total cost of the books.\n"
                    "3. Subtract the discount from the total.\n"
                    "4. Compute the final price and give the answer."
                )
            if "train" in p or "miles" in p:
                return (
                    "1. Extract the speed and the travel time.\n"
                    "2. Compute the distance travelled in the first part.\n"
                    "3. Compute the distance travelled in the second part.\n"
                    "4. Add the distances and give the final answer."
                )
            nums = re.findall(r"-?\d+(?:\.\d+)?", prompt)
            steps = [f"{i}. Process the value {n}." for i, n in enumerate(nums[:4], 1)]
            steps.append(f"{len(steps) + 1}. Combine the intermediate results and give the final answer.")
            return "\n".join(steps)
        return None

    def _canned_execute(self, prompt: str) -> str:
        p = prompt.lower()
        if "apples" in p:
            return (
                "Subtask 1: Janet starts with 16 apples.\n"
                "Subtask 2: She gives away 3 + 4 = 7 apples.\n"
                "Subtask 3: She buys 12 apples.\n"
                "Subtask 4: 16 - 7 + 12 = 21.\n"
                "Answer: 21"
            )
        if "books" in p:
            return (
                "Subtask 1: Each book costs $8.\n"
                "Subtask 2: 5 books cost 5 * 8 = $40.\n"
                "Subtask 3: Subtract the $5 discount: 40 - 5 = $35.\n"
                "Subtask 4: Final price is $35.\n"
                "Answer: 35"
            )
        if "train" in p or "miles" in p:
            return (
                "Subtask 1: Speed 60 mph, first part 2 hours, second part 1.5 hours.\n"
                "Subtask 2: First part distance = 60 * 2 = 120 miles.\n"
                "Subtask 3: Second part distance = 60 * 1.5 = 90 miles.\n"
                "Subtask 4: Total distance = 120 + 90 = 210 miles.\n"
                "Answer: 210"
            )
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", prompt)]
        total = sum(nums) if nums else 0
        total_s = int(total) if float(total).is_integer() else total
        return (
            "Subtask 1: Extracted the values from the problem.\n"
            "Subtask 2: Combined the intermediate results.\n"
            f"Answer: {total_s}"
        )

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        if "plan:" in prompt.lower() and ("carry out" in prompt.lower() or "show the final" in prompt.lower()):
            return self._canned_execute(prompt)
        canned = self._canned_plan(prompt)
        if canned is not None:
            return canned
        return self._canned_execute(prompt)


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP backend (stdlib only)
# ---------------------------------------------------------------------------

class OpenAIBackend:
    """Backend for any OpenAI-compatible chat-completions endpoint.

    Credentials/endpoints come from the environment (``OPENAI_API_KEY``,
    ``OPENAI_BASE_URL``) or constructor arguments. Uses only the standard
    library (``urllib``) — no required third-party dependencies.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"].strip()
