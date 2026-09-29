# decision_engine/models/tinybert_model.py
"""
TAMEV Super-Tiny Decision Model (TinyBERT Edition).
Maintains 100% backward compatibility with existing saved weights and downstream scripts.
"""

from __future__ import annotations

from decision_engine.config.base import ModelConfig
from decision_engine.models.base import ModelRegistry
from decision_engine.models.encoder_model import EncoderDecisionModel
from decision_engine.models.pointer_head import PointerHead, TinyBertPointerHead


@ModelRegistry.register("tinybert")
class TamevTinyBertDecisionModel(EncoderDecisionModel):
    """
    Backward-compatible TinyBERT decision model.
    Instantiates an EncoderDecisionModel configured for TinyBERT.
    """

    def __init__(
        self,
        config_or_backbone: ModelConfig | str = "huawei-noah/TinyBERT_General_4L_312D",
        backbone_name_or_path: str | None = None,
        projection_dim: int = 64,
        freeze_early_layers: bool = False,
    ):
        if isinstance(config_or_backbone, ModelConfig):
            config = config_or_backbone
            backbone = config.backbone
        else:
            backbone = backbone_name_or_path or config_or_backbone
            config = ModelConfig(
                name="tinybert",
                size_profile="nano",
                model_type="encoder",
                backbone=backbone,
                projection_dim=projection_dim,
                freeze_embeddings=freeze_early_layers,
                pooling="cls",
            )
        super().__init__(config)
        self.backbone_name = backbone


__all__ = ["PointerHead", "TamevTinyBertDecisionModel", "TinyBertPointerHead"]
