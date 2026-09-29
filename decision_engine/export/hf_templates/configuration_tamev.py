# configuration_tamev.py
"""
Hugging Face Configuration class for TAMEV Universal Decision Models.
Supports both Bidirectional Encoders (TinyBERT, MiniLM, ModernBERT)
and In-Context Causal Models (Qwen3.5-0.8B, Qwen3.5-4B).
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class TamevConfig(PretrainedConfig):
    """
    Configuration class for TAMEV Universal Decision Models.
    Compatible with Hugging Face AutoConfig and AutoModel.
    """

    model_type = "tamev"

    def __init__(  # noqa: PLR0917
        self,
        backbone: str = "huawei-noah/TinyBERT_General_4L_312D",
        architecture_type: str = "encoder",
        hidden_dim: int = 312,
        projection_dim: int = 64,
        temperature: float = 1.05,
        calibration_by_k: dict[str, float] | None = None,
        pooling: str = "cls",
        normalize: bool = False,  # noqa: FBT001, FBT002
        option_isolation: bool = True,  # noqa: FBT001, FBT002
        tier: str = "nano",
        max_state_len: int = 512,
        max_opt_len: int = 64,
        auto_map: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.architecture_type = architecture_type
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.calibration_by_k = calibration_by_k or {}
        self.pooling = pooling
        self.normalize = normalize
        self.option_isolation = option_isolation
        self.tier = tier
        self.max_state_len = max_state_len
        self.max_opt_len = max_opt_len
        self.auto_map = auto_map or {
            "AutoConfig": "configuration_tamev.TamevConfig",
            "AutoModel": "modeling_tamev.TamevModel",
            "AutoModelForSequenceClassification": "modeling_tamev.TamevForDecision",
        }
