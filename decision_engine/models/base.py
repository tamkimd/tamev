# decision_engine/models/base.py
"""
Base Decision Model Abstraction, Model Registry, and Model Factory.
Enables dead-simple swapping of student models across any model family:
- Encoders (BERT, TinyBERT, MiniLM, ModernBERT, RoBERTa, DeBERTa)
- Causal LMs (SmolLM2, Qwen2.5, Qwen3.5, Llama-3.2)
- Bi-Encoders
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from decision_engine.config.base import ModelConfig
from decision_engine.models.pointer_head import PointerHead, order_invariant_softmax


class BaseDecisionModel(nn.Module, ABC):
    """
    Abstract Base Class for all TAMEV decision models.
    Guarantees consistent encoding interfaces and decoupled pointer head scoring.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.pointer_head: PointerHead = None  # Must be set by subclass

    @property
    def hidden_dim(self) -> int:
        return self.pointer_head.hidden_dim

    @property
    def projection_dim(self) -> int:
        return self.pointer_head.projection_dim

    @property
    def temperature(self) -> torch.Tensor:
        return self.pointer_head.temperature

    def set_temperature(self, t: float) -> None:
        """Sets the calibrated temperature on the pointer head."""
        if self.pointer_head is not None:
            self.pointer_head.set_temperature(t)

    @abstractmethod
    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode context text into pooled representation (B, hidden_dim)."""

    @abstractmethod
    def encode_options(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode candidate options into representations (B, K, hidden_dim)."""

    def forward(
        self,
        ctx_input_ids: torch.Tensor | None = None,
        ctx_attention_mask: torch.Tensor | None = None,
        opt_input_ids: torch.Tensor | None = None,
        opt_attention_mask: torch.Tensor | None = None,
        num_options: list[int] | None = None,
        context_input_ids: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
        option_input_ids: torch.Tensor | None = None,
        option_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: context + candidate options -> logits & calibrated probabilities.
        """
        c_ids = ctx_input_ids if ctx_input_ids is not None else context_input_ids
        c_mask = ctx_attention_mask if ctx_attention_mask is not None else context_attention_mask
        o_ids = opt_input_ids if opt_input_ids is not None else option_input_ids
        o_mask = opt_attention_mask if opt_attention_mask is not None else option_attention_mask

        ctx_vec = self.encode_context(c_ids, c_mask)
        opt_vecs = self.encode_options(o_ids, o_mask)
        logits, probs = self.pointer_head(ctx_vec, opt_vecs)

        if num_options is not None:
            K = logits.shape[-1]
            mask = torch.arange(K, device=logits.device).unsqueeze(0) < torch.as_tensor(
                num_options, device=logits.device
            ).unsqueeze(1)
            logits = logits.masked_fill(~mask, -1e4)
            probs = order_invariant_softmax(logits, self.pointer_head.temperature)

        return {"logits": logits, "probs": probs, "ctx_vec": ctx_vec, "opt_vecs": opt_vecs}


class ModelRegistry:
    """Registry for dynamically pluggable decision model types."""

    _registry: dict[str, type[BaseDecisionModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[BaseDecisionModel]):
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseDecisionModel]:
        key = name.lower()
        if key not in cls._registry:
            raise KeyError(
                f"Model type '{name}' not found in ModelRegistry. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[key]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_models(cls) -> list[str]:
        return cls.list_available()


class ModelFactory:
    """Factory to instantiate any configured Decision Model."""

    @staticmethod
    def create(config: ModelConfig) -> BaseDecisionModel:
        # Ensure built-in models are imported and registered
        import decision_engine.models  # noqa: F401

        model_cls = ModelRegistry.get(config.model_type)
        return model_cls(config)

    @classmethod
    def create_from_name(
        cls,
        name_or_type: str,
        backbone: str | None = None,
        projection_dim: int = 64,
        temperature: float = 2.2,
        **kwargs: Any,
    ) -> BaseDecisionModel:
        """Create a decision model directly by type or preset name."""
        key = name_or_type.lower()
        if key in ("tinybert", "tamev_tinybert"):
            mtype = "tinybert"
            bb = backbone or "huawei-noah/TinyBERT_General_4L_312D"
        elif key in ("incontext_causal", "incontext", "kev"):
            mtype = "incontext_causal"
            bb = backbone or "Qwen/Qwen2.5-0.5B"
        elif key in ("causal", "causallm", "smollm", "smollm2", "qwen", "llama"):
            mtype = "causal"
            bb = backbone or "HuggingFaceTB/SmolLM2-135M"
        elif key in ("bi_encoder", "biencoder", "sentence_transformer"):
            mtype = "bi_encoder"
            bb = backbone or "sentence-transformers/all-MiniLM-L6-v2"
        elif key in ("encoder", "bert", "minilm", "modernbert", "roberta"):
            mtype = "encoder"
            bb = backbone or "huawei-noah/TinyBERT_General_4L_312D"
        else:
            mtype = key
            bb = backbone or "huawei-noah/TinyBERT_General_4L_312D"

        cfg = ModelConfig(
            name=key,
            model_type=mtype,
            backbone=bb,
            projection_dim=projection_dim,
            temperature=temperature,
            **kwargs,
        )
        return cls.create(cfg)
