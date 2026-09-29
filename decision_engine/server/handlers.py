# decision_engine/server/handlers.py
"""
Question Handler Strategy & Registry Pattern for TAMEV Decision Server.
Decouples question-specific formatting, extraction, and confidence scoring
into modular, pluggable handler strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from decision_engine.server.engine import TamevEngine


def choice_confidence(probs: list[float]) -> float:
    """
    Confidence normalized against uniform chance:
    c = (max(p) - 1/K) / (1 - 1/K)
    Returns 1.0 for K == 1.
    """
    K = len(probs)
    if K <= 1:
        return 1.0
    return round(float(max(probs) - 1.0 / K) / (1.0 - 1.0 / K), 4)


def score_confidence(probs: list[float]) -> float:
    """
    Expected distance from modal level:
    c = 1.0 - E[|level - mode|] / (L - 1)
    """
    L = len(probs)
    if L <= 1:
        return 1.0
    mode = max(range(L), key=lambda i: probs[i])
    exp_dist = sum(pi * abs(i - mode) for i, pi in enumerate(probs))
    return round(float(1.0 - exp_dist / (L - 1)), 4)


def render(v: Any, indent: int = 0) -> str:
    """Flatten str | dict | list into clean hierarchical text representation."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    if isinstance(v, dict):
        return "\n".join(
            f"{pad}{k}:\n{render(x, indent + 1)}"
            if isinstance(x, (dict, list))
            else f"{pad}{k}: {render(x)}"
            for k, x in v.items()
        )
    return str(v)


def option_text(name: str, desc: Any) -> str:
    """Format option label and optional description."""
    if desc is None or desc == "":
        return name
    return f"{name}: {render(desc)}"


def round_prob(x: float) -> float:
    return round(float(x), 4)


class BaseQuestionHandler(ABC):
    """
    Abstract Strategy class for question processing.
    """

    question_type: str

    @abstractmethod
    def handle(
        self,
        engine: TamevEngine,
        state_str: str,
        instructions: str,
        criteria: Any,
    ) -> dict[str, Any]:
        """Evaluates question criteria against engine and returns formatted answer."""


class ChoiceQuestionHandler(BaseQuestionHandler):
    """Handles categorical Choice questions."""

    question_type = "choice"

    def handle(
        self,
        engine: TamevEngine,
        state_str: str,
        instructions: str,
        criteria: Any,
    ) -> dict[str, Any]:
        crit = criteria or {}
        opt_keys = list(crit.keys())
        if not opt_keys:
            raise ValueError("Choice question requires at least one option in criteria.")

        options = [(k, option_text(k, crit.get(k))) for k in opt_keys]
        probs = engine._score_options(state_str, instructions, options)

        chosen_idx = int(np.argmax(probs))
        chosen_key = opt_keys[chosen_idx]
        conf = choice_confidence(probs)
        prob_dict = {k: round_prob(p) for k, p in zip(opt_keys, probs)}

        return {
            "type": "choice",
            "choice": chosen_key,
            "confidence": conf,
            "probabilities": prob_dict,
        }


class NoulQuestionHandler(BaseQuestionHandler):
    """Handles binary Yes/No (Noul) questions."""

    question_type = "noul"

    def handle(
        self,
        engine: TamevEngine,
        state_str: str,
        instructions: str,
        criteria: Any,
    ) -> dict[str, Any]:
        crit = criteria or {}
        false_desc = crit.get(False) or crit.get("false") or crit.get("no")
        true_desc = crit.get(True) or crit.get("true") or crit.get("yes")

        options = [
            ("false", option_text("No / False", false_desc)),
            ("true", option_text("Yes / True", true_desc)),
        ]
        probs = engine._score_options(state_str, instructions, options)
        p_true = round_prob(probs[1])

        return {"type": "noul", "noul": p_true}


class ScoreQuestionHandler(BaseQuestionHandler):
    """Handles ordinal rubric Score questions."""

    question_type = "score"

    def handle(
        self,
        engine: TamevEngine,
        state_str: str,
        instructions: str,
        criteria: Any,
    ) -> dict[str, Any]:
        if not criteria:
            raise ValueError("Score question requires non-empty criteria list.")

        options = [(str(i), render(c)) for i, c in enumerate(criteria)]
        probs = engine._score_options(state_str, instructions, options)

        expected_score = round_prob(sum(i * p for i, p in enumerate(probs)))
        conf = score_confidence(probs)
        legend = {i: criteria[i] for i in range(len(criteria))}
        prob_dict = {i: round_prob(p) for i, p in enumerate(probs)}

        return {
            "type": "score",
            "score": expected_score,
            "confidence": conf,
            "legend": legend,
            "probabilities": prob_dict,
        }


class QuestionHandlerRegistry:
    """Registry managing available question handler strategies."""

    _handlers: dict[str, BaseQuestionHandler] = {}

    @classmethod
    def register(cls, handler: BaseQuestionHandler) -> None:
        cls._handlers[handler.question_type.lower()] = handler

    @classmethod
    def get(cls, question_type: str | None) -> BaseQuestionHandler:
        q_type = (question_type or "choice").strip().lower()
        if q_type not in cls._handlers:
            # Fallback to choice handler for backward compatibility
            return cls._handlers.get("choice", ChoiceQuestionHandler())
        return cls._handlers[q_type]


# Register standard handlers
QuestionHandlerRegistry.register(ChoiceQuestionHandler())
QuestionHandlerRegistry.register(NoulQuestionHandler())
QuestionHandlerRegistry.register(ScoreQuestionHandler())
