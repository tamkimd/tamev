# decision_engine/models/bi_encoder_model.py
"""
Bi-Encoder Decision Model for High-Cardinality Option Sets (K > 50).
Employs decoupled sentence transformer embeddings with normalized inner-product scoring.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel

from decision_engine.config.base import ModelConfig
from decision_engine.models.base import BaseDecisionModel, ModelRegistry
from decision_engine.models.pointer_head import PointerHead


@ModelRegistry.register("bi_encoder")
class BiEncoderDecisionModel(BaseDecisionModel):
    """
    Bi-Encoder architecture optimized for large action spaces (e.g. Banking77).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.backbone)
        hidden_dim = getattr(self.encoder.config, "hidden_size", 384)

        self.pointer_head = PointerHead(
            hidden_dim=hidden_dim,
            projection_dim=config.projection_dim,
            temperature=config.temperature,
        )

    def _mean_pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        vec = self._mean_pool(out.last_hidden_state, attention_mask)
        return F.normalize(vec, p=2, dim=-1)

    def encode_options(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, K, L_opt = input_ids.shape
        flat_ids = input_ids.view(B * K, L_opt)
        flat_mask = attention_mask.view(B * K, L_opt).clone()
        flat_mask[:, 0] = 1

        out = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        pooled = self._mean_pool(out.last_hidden_state, flat_mask)
        normed = F.normalize(pooled, p=2, dim=-1)
        return normed.view(B, K, -1)
