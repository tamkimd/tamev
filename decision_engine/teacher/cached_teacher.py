# decision_engine/teacher/cached_teacher.py
"""
Cached Teacher Wrapper.
Stores computed soft targets on disk (JSON) to avoid recomputing expensive LLM calls.
"""

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from decision_engine.teacher.base import BaseTeacherModel, TeacherRegistry


@TeacherRegistry.register("cached")
class CachedTeacher(BaseTeacherModel):
    """
    Wraps another BaseTeacherModel with persistent disk caching.
    """

    def __init__(
        self,
        wrapped_teacher: BaseTeacherModel | None = None,
        cache_path: str | Path = "runs/teacher_cache.json",
        model_name: str = "cached_teacher",
        temperature: float = 2.0,
    ):
        if wrapped_teacher is None:
            from decision_engine.teacher.mock_teacher import MockTeacher

            wrapped_teacher = MockTeacher(model_name=model_name, temperature=temperature)

        super().__init__(
            model_name=f"cached_{wrapped_teacher.model_name}",
            temperature=wrapped_teacher.temperature,
        )
        self.wrapped_teacher = wrapped_teacher
        self.cache_path = Path(cache_path)
        self.cache: dict[str, list[float]] = {}
        # Concurrent annotation shares one cache across worker threads.
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self):
        """Atomic replace: a killed run must never leave a truncated cache behind."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)
        tmp.replace(self.cache_path)

    def _make_key(self, context: str, options: list[str], temperature: float) -> str:
        key_raw = f"{self.wrapped_teacher.model_name}:{temperature}:{context}:{','.join(options)}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def get_soft_targets(
        self, context: str, options: list[str], temperature: float | None = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self.temperature
        key = self._make_key(context, options, T)

        with self._lock:
            hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return np.array(hit, dtype=np.float32)

        self.cache_misses += 1
        # Call the wrapped teacher *outside* the lock so N threads really overlap.
        probs = self.wrapped_teacher.get_soft_targets(context, options, temperature=T)
        with self._lock:
            self.cache[key] = probs.tolist()
            self._save_cache()
        return probs
