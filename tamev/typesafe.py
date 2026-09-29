"""
Native TypeSafe SDK Protocol and Client Support for TAMEV.

Provides:
- 100% compliant data models for TypeSafe System One requests and responses
- Choice, Noul, and Score question specifications
- Normalized confidence and scoring formulas
- TypeSafeDirectClient for high-speed in-process decision making
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typesafe_sdk import (
        Answer,
        Choice,
        ChoiceAnswer,
        ListModelsResponse,
        ModelMetadata,
        Noul,
        NoulAnswer,
        Score,
        ScoreAnswer,
        SystemOneResponse,
        TypeSafeClient,
        Usage,
    )
else:
    try:
        from typesafe_sdk import (
            Answer,
            Choice,
            ChoiceAnswer,
            ListModelsResponse,
            ModelMetadata,
            Noul,
            NoulAnswer,
            Score,
            ScoreAnswer,
            SystemOneResponse,
            TypeSafeClient,
            Usage,
        )
    except ImportError:
        # Graceful pure-python fallback if typesafe_sdk is not installed in runtime
        from typing import Literal

        from pydantic import BaseModel

        class ChoiceAnswer(BaseModel):
            type: Literal["choice"] = "choice"
            choice: str
            confidence: float
            probabilities: dict[str, float]

        class NoulAnswer(BaseModel):
            type: Literal["noul"] = "noul"
            noul: float

        class ScoreAnswer(BaseModel):
            type: Literal["score"] = "score"
            score: float
            confidence: float
            legend: dict[int, Any]
            probabilities: dict[int, float]

        class Usage(BaseModel):
            input_tokens: int = 0
            output_tokens: int = 0

        class SystemOneResponse(BaseModel):
            model: str
            answers: dict[str, Any]
            usage: Usage

            @property
            def choices(self) -> dict[str, ChoiceAnswer]:
                return {
                    k: v for k, v in self.answers.items() if getattr(v, "type", None) == "choice"
                }

            @property
            def nouls(self) -> dict[str, NoulAnswer]:
                return {k: v for k, v in self.answers.items() if getattr(v, "type", None) == "noul"}

            @property
            def scores(self) -> dict[str, ScoreAnswer]:
                return {
                    k: v for k, v in self.answers.items() if getattr(v, "type", None) == "score"
                }

        class Choice:
            def __init__(
                self, instructions: str | None = None, criteria: dict[str, Any] | None = None
            ):
                self.instructions = instructions
                self.criteria = criteria or {}

        class Noul:
            def __init__(
                self, instructions: str | None = None, criteria: dict[Any, Any] | None = None
            ):
                self.instructions = instructions
                self.criteria = criteria or {}

        class Score:
            def __init__(self, instructions: str | None = None, criteria: list[Any] | None = None):
                self.instructions = instructions
                self.criteria = criteria or []

        class ModelMetadata(BaseModel):
            name: str
            description: str
            release_date: str

        class ListModelsResponse(BaseModel):
            models: list[ModelMetadata]

        TypeSafeClient = Any
        Answer = Any


from decision_engine.server.handlers import choice_confidence, score_confidence


def validate_choice_criteria(criteria: dict[str, Any]) -> bool:
    """Validates that choice criteria is non-empty and has valid keys."""
    if not isinstance(criteria, dict) or len(criteria) == 0:
        raise ValueError("Choice criteria must be a non-empty dictionary of option choices.")
    return True


class TypeSafeDirectClient:
    """
    Direct in-process client for TAMEV models.
    Implements the exact same `.system_one(...)` API as `TypeSafeClient` but
    executes directly in-memory with sub-5ms CPU latency, bypassing HTTP networking.
    """

    def __init__(
        self,
        engine: Any | None = None,
        checkpoint_path: str | None = None,
        backbone: str = "huawei-noah/TinyBERT_General_4L_312D",
        device: str | None = None,
        model_name: str = "tamev-latest",
    ):
        self.model_name = model_name
        if engine is not None:
            self.engine = engine
        else:
            from decision_engine.server.typesafe_server import TamevEngine

            self.engine = TamevEngine(
                checkpoint_path=checkpoint_path, device=device, backbone=backbone
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def system_one(
        self, state: Any, questions: dict[str, Any], model: str | None = None
    ) -> SystemOneResponse:
        """
        Executes System One decision making over the state for all questions.
        Returns a 100% TypeSafe-compliant SystemOneResponse object.
        """
        chosen_model = model or self.model_name
        batch_res = self.engine.predict_batch(state, questions, model_name=chosen_model)

        answers_map: dict[str, Any] = {}
        for q_name, res in batch_res["answers"].items():
            q_type = res.get("type", "choice")
            if q_type == "choice":
                ans = ChoiceAnswer(
                    type="choice",
                    choice=res["choice"],
                    confidence=res["confidence"],
                    probabilities=res["probabilities"],
                )
            elif q_type == "noul":
                ans = NoulAnswer(type="noul", noul=res["noul"])
            elif q_type == "score":
                int_legend = {int(k): v for k, v in res["legend"].items()}
                int_probs = {int(k): v for k, v in res["probabilities"].items()}
                ans = ScoreAnswer(
                    type="score",
                    score=res["score"],
                    confidence=res["confidence"],
                    legend=int_legend,
                    probabilities=int_probs,
                )
            else:
                raise ValueError(f"Unsupported question type: {q_type} for question '{q_name}'")
            answers_map[q_name] = ans

        usage_dict = batch_res.get("usage", {})
        return SystemOneResponse(
            model=chosen_model,
            answers=answers_map,
            usage=Usage(
                input_tokens=usage_dict.get("input_tokens", 0),
                output_tokens=usage_dict.get("output_tokens", 0),
            ),
        )
