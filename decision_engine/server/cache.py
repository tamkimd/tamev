# decision_engine/server/cache.py
"""
Flyweight / Cache Pattern for TAMEV High-Performance Inference.
Caches:
1. Option Embeddings: Fixed option sets (e.g. game actions, rubric scales, Noul Yes/No, intent menus)
   are cached to avoid redundant tokenization and neural network forward passes.
2. Context Embeddings: Repeated context texts (e.g. multi-question evaluation over the same state)
   are cached to eliminate redundant context encoding.
"""

from __future__ import annotations

import collections
import threading
from typing import Any

import torch


class LRUEmbeddingCache:
    """
    Thread-safe, bounded Least Recently Used (LRU) embedding cache.
    Applies the Flyweight Pattern to share immutable pooled vector representations.
    """

    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._cache: collections.OrderedDict[Any, torch.Tensor] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> torch.Tensor | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, key: Any, value: torch.Tensor) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                # Store detached tensor to prevent computational graph memory retention
                self._cache[key] = value.detach().clone()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Global shared cache instances
_option_cache = LRUEmbeddingCache(maxsize=512)
_context_cache = LRUEmbeddingCache(maxsize=256)


def get_option_cache() -> LRUEmbeddingCache:
    """Returns the shared option embedding flyweight cache."""
    return _option_cache


def get_context_cache() -> LRUEmbeddingCache:
    """Returns the shared context embedding flyweight cache."""
    return _context_cache
