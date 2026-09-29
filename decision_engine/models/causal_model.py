# decision_engine/models/causal_model.py
"""
Generic HuggingFace Causal LM Decision Model.
Supports SmolLM2 (135M, 360M), Qwen2.5, Qwen3.5 (0.8B, 4B), Llama-3.2.
Uses decoupled pointer head scoring with last-token sequence representations.
"""

from __future__ import annotations

import torch
from transformers import AutoModel

from decision_engine.config.base import ModelConfig
from decision_engine.models.base import BaseDecisionModel, ModelRegistry
from decision_engine.models.pointer_head import PointerHead


@ModelRegistry.register("causal")
class CausalLMDecisionModel(BaseDecisionModel):
    """
    Decoupled Decision Model backed by an autoregressive Causal LM backbone.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        device_type = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        load_dtype = torch.bfloat16 if device_type in ("mps", "cuda") else torch.float32

        self.backbone = AutoModel.from_pretrained(
            config.backbone,
            dtype=load_dtype,
        )

        self.freeze_backbone = bool(getattr(config, "freeze_backbone", False))
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif getattr(config, "freeze_embeddings", False) and hasattr(self.backbone, "embed_tokens"):
            for p in self.backbone.embed_tokens.parameters():
                p.requires_grad = False

        if config.hidden_dim is not None:
            hidden_dim = config.hidden_dim
        elif (
            hasattr(self.backbone.config, "hidden_size")
            and self.backbone.config.hidden_size is not None
        ):
            hidden_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "text_config") and hasattr(
            self.backbone.config.text_config, "hidden_size"
        ):
            hidden_dim = self.backbone.config.text_config.hidden_size
        else:
            hidden_dim = 576

        self.pooling = getattr(config, "pooling", "last")
        self.pointer_head = PointerHead(
            hidden_dim=hidden_dim,
            projection_dim=config.projection_dim,
            temperature=config.temperature,
            normalize=getattr(config, "normalize", False),
        )
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
            self.pointer_head = self.pointer_head.to(backbone_dtype)
        except StopIteration:
            pass

    def _pool_last_token(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract hidden state of the last non-padded token."""
        seq_lens = (attention_mask.sum(dim=1) - 1).long()
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_idx = torch.arange(
            hidden_states.shape[0], device=hidden_states.device, dtype=torch.long
        )
        return hidden_states[batch_idx, seq_lens]

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states using configured pooling strategy (mean or last)."""
        if self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return (sum_embeddings / sum_mask).to(hidden_states.dtype)
        return self._pool_last_token(hidden_states, attention_mask)

    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)

    def encode_options(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, K, L_opt = input_ids.shape
        flat_ids = input_ids.view(B * K, L_opt)
        flat_mask = attention_mask.view(B * K, L_opt).clone()
        flat_mask[:, 0] = 1

        # Adaptive chunk size: 32 on large models or CPU, 64 on MPS/CUDA
        chunk_size = 64 if input_ids.device.type in ("mps", "cuda") else 32
        if flat_ids.shape[0] > chunk_size:
            chunks = []
            for i in range(0, flat_ids.shape[0], chunk_size):
                sub_ids = flat_ids[i : i + chunk_size]
                sub_mask = flat_mask[i : i + chunk_size]
                if self.freeze_backbone:
                    with torch.no_grad():
                        sub_out = self.backbone(input_ids=sub_ids, attention_mask=sub_mask)
                else:
                    sub_out = self.backbone(input_ids=sub_ids, attention_mask=sub_mask)
                sub_p = self._pool(sub_out.last_hidden_state, sub_mask)
                chunks.append(sub_p)
            pooled = torch.cat(chunks, dim=0)
        else:
            if self.freeze_backbone:
                with torch.no_grad():
                    out = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
            else:
                out = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
            pooled = self._pool(out.last_hidden_state, flat_mask)
        return pooled.view(B, K, -1)
