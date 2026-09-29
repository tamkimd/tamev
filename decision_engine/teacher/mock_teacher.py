# decision_engine/teacher/mock_teacher.py
"""
Mock Teacher Model for deterministic testing, CI, and local rapid development.
"""

import hashlib

import numpy as np

from decision_engine.teacher.base import BaseTeacherModel, TeacherRegistry


@TeacherRegistry.register("mock")
class MockTeacher(BaseTeacherModel):
    """
    Produces deterministic pseudo-logits using string hashing for unit tests and CI.
    """

    def __init__(self, model_name: str = "mock_teacher", temperature: float = 2.0):
        super().__init__(model_name=model_name, temperature=temperature)

    def get_soft_targets(
        self, context: str, options: list[str], temperature: float | None = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self.temperature
        k = len(options)
        if k == 0:
            return np.array([], dtype=np.float32)

        # Deterministic logits based on hashing context + option
        logits = []
        for i, opt in enumerate(options):
            h = hashlib.sha256(f"{context}:{opt}:{i}".encode()).hexdigest()
            # map hash to a pseudo-float between -2.0 and 2.0
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 4.0 - 2.0
            logits.append(val)

        arr = np.array(logits, dtype=np.float32) / T
        arr = arr - np.max(arr)  # numerical stability
        exp_arr = np.exp(arr)
        probs = exp_arr / np.sum(exp_arr)
        return probs.astype(np.float32)
