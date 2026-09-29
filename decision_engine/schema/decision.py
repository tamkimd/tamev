# decision_engine/schema/decision.py
"""
Pydantic v2 Canonical Schemas for TAMEV Decision Engine.
Provides high-performance, strictly typed validation for decision tasks,
candidate options, rubrics, and teacher soft targets.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DecisionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


class OptionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptionItem:
        return cls.model_validate(data)


class ScoreScale(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min: int
    max: int

    def to_dict(self) -> dict[str, int]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoreScale:
        return cls.model_validate(data)


class DecisionExample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    state: str
    question: str
    decision_type: DecisionType
    label: str | int | bool
    options: list[OptionItem] | None = None
    scale: ScoreScale | None = None
    teacher_probs: dict[str, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_semantics(self) -> DecisionExample:
        # Validation rules
        if self.decision_type == DecisionType.CHOICE:
            if not self.options or len(self.options) < 2:
                raise ValueError("Choice decision requires at least 2 options")
            valid_ids = {opt.id for opt in self.options}
            if self.label not in valid_ids:
                raise ValueError(f"Label '{self.label}' not present in options {valid_ids}")
            if self.teacher_probs:
                if set(self.teacher_probs.keys()) != valid_ids:
                    raise ValueError(
                        f"teacher_probs keys {set(self.teacher_probs.keys())} must match option IDs {valid_ids}"
                    )
                total_prob = sum(self.teacher_probs.values())
                if abs(total_prob - 1.0) > 0.02:
                    raise ValueError(f"teacher_probs must sum to ~1.0, got {total_prob}")
        elif self.decision_type == DecisionType.SCORE:
            if not self.scale:
                raise ValueError("Score decision requires scale (min, max)")
            if not isinstance(self.label, int) or not (
                self.scale.min <= self.label <= self.scale.max
            ):
                raise ValueError(
                    f"Score label must be int between {self.scale.min} and {self.scale.max}"
                )
        elif self.decision_type == DecisionType.NOUL:
            if not isinstance(self.label, bool):
                raise ValueError(f"Noul label must be boolean True or False, got {self.label}")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionExample:
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "question": self.question,
            "decision_type": self.decision_type.value,
            "options": [opt.to_dict() for opt in self.options] if self.options else None,
            "scale": self.scale.to_dict() if self.scale else None,
            "label": self.label,
            "teacher_probs": self.teacher_probs,
            "metadata": self.metadata,
        }
