# decision_engine/models/encoder_model.py
"""
Generic HuggingFace Transformer Encoder Decision Model.
Supports TinyBERT, MiniLM, ModernBERT, RoBERTa, DeBERTa, ELECTRA.
"""

from __future__ import annotations

import torch
from transformers import AutoModel

from decision_engine.config.base import ModelConfig
from decision_engine.models.base import BaseDecisionModel, ModelRegistry
from decision_engine.models.pointer_head import PointerHead


@ModelRegistry.register("encoder")
class EncoderDecisionModel(BaseDecisionModel):
    """
    Decoupled Decision Model backed by any HuggingFace Encoder backbone.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.backbone)

        # Discover hidden dimension
        if config.hidden_dim is not None:
            hidden_dim = config.hidden_dim
        elif hasattr(self.encoder.config, "hidden_size"):
            hidden_dim = self.encoder.config.hidden_size
        elif hasattr(self.encoder.config, "d_model"):
            hidden_dim = self.encoder.config.d_model
        else:
            hidden_dim = 312  # Fallback

        if config.freeze_embeddings and hasattr(self.encoder, "embeddings"):
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False

        self.pointer_head = PointerHead(
            hidden_dim=hidden_dim,
            projection_dim=config.projection_dim,
            temperature=config.temperature,
            normalize=getattr(config, "normalize", False),
        )
        self.pooling = config.pooling
        self.option_fusion = getattr(config, "option_fusion", "bilinear")
        # Context captured by the last encode_context call and consumed by the next
        # encode_options call. Only read when option_fusion == "cross".
        self._option_ctx: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)

        # In-process option-vector memo used only in eval (see encode_options). Keyed on the
        # unpadded token ids plus a backbone/max_opt_len/dtype/device prefix: batches pad options
        # to their own widest row, so a key on the padded row misses on nearly every batch. The
        # bilinear head scores an option independently of the context, so the vector is a pure
        # function of the text.
        # `option_cache_max <= 0` disables it; `option_cache_stats` exposes the hit/miss counts.
        self.option_cache_max = 20000
        self._option_vec_cache: dict[tuple, torch.Tensor] = {}
        self.option_cache_stats = {"encoded": 0, "hits": 0}

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            # Token 0 ([CLS])
            return last_hidden_state[:, 0, :]
        if self.pooling == "mean":
            # Mean pooling masked over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        return last_hidden_state[:, 0, :]

    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self._option_ctx = (input_ids, attention_mask)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)

    def encode_options(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ctx_input_ids: torch.Tensor | None = None,
        ctx_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training:
            # Weights move every step, so a vector cached under the previous weights is stale the
            # next time eval runs. Clearing on the training forward (not on eval entry) means a
            # train->eval transition starts cold, while a pure eval loop keeps its entries.
            self._option_vec_cache.clear()
        if self.option_fusion == "cross":
            # Cross fusion: prepend the context so every option is encoded jointly with it
            # (option tokens stay last, so mean pooling attends to both). Each option is
            # still encoded independently, so permutation equivariance stays exact.
            ctx_ids, ctx_mask = self._option_ctx
            self._option_ctx = (None, None)
            ctx_ids = ctx_input_ids if ctx_input_ids is not None else ctx_ids
            ctx_mask = ctx_attention_mask if ctx_attention_mask is not None else ctx_mask
            if ctx_ids is None:
                raise RuntimeError(
                    "option_fusion='cross' needs the context: call encode_context first or pass "
                    "ctx_input_ids/ctx_attention_mask."
                )
            n_options = input_ids.shape[1]
            input_ids = torch.cat(
                [ctx_ids.unsqueeze(1).expand(-1, n_options, -1), input_ids], dim=-1
            )
            attention_mask = torch.cat(
                [ctx_mask.unsqueeze(1).expand(-1, n_options, -1), attention_mask], dim=-1
            )
        B, K, L_opt = input_ids.shape
        flat_ids = input_ids.view(B * K, L_opt)
        flat_mask = attention_mask.view(B * K, L_opt).clone()

        # Ensure at least token 0 is attended to avoid NaN issues on empty padding options
        flat_mask[:, 0] = 1

        # The cache only applies when the option encode is context-free: with option_fusion="cross"
        # the same token ids pool against a different context every call, so nothing may be reused.
        # Also off while tracing: the key lookup is data-dependent Python that a traced/exported
        # graph cannot capture, and torch.jit.trace's own sanity check then fails (test broke).
        if (
            self.option_cache_max > 0
            and not self.training
            and self.option_fusion != "cross"
            and not torch.jit.is_tracing()
        ):
            pooled = self._encode_options_cached(flat_ids, flat_mask)
        else:
            self.option_cache_stats["encoded"] += flat_ids.shape[0]
            pooled = self._encode_flat(flat_ids, flat_mask)
        return pooled.view(B, K, -1)

    def _encode_flat(self, flat_ids: torch.Tensor, flat_mask: torch.Tensor) -> torch.Tensor:
        chunk_size = 128 if flat_ids.device.type in ("mps", "cuda") else 64
        if flat_ids.shape[0] > chunk_size:
            chunks = []
            for i in range(0, flat_ids.shape[0], chunk_size):
                sub_ids = flat_ids[i : i + chunk_size]
                sub_mask = flat_mask[i : i + chunk_size]
                sub_out = self.encoder(input_ids=sub_ids, attention_mask=sub_mask)
                sub_p = self._pool(sub_out.last_hidden_state, sub_mask)
                chunks.append(sub_p)
            return torch.cat(chunks, dim=0)
        out = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        return self._pool(out.last_hidden_state, flat_mask)

    def _option_cache_key_prefix(self, ids: torch.Tensor) -> tuple:
        """Everything that makes two pooled option vectors comparable, minus the weights.

        Weights are intentionally absent: the cache is read only while `not self.training`, and a
        training forward clears it, so an entry cannot outlive the weights that produced it.
        """
        return (
            str(self.config.backbone),
            getattr(self.config, "max_opt_len", None),
            str(next(self.encoder.parameters()).dtype),
            str(ids.device),
        )

    def _encode_options_cached(
        self, flat_ids: torch.Tensor, flat_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode only the option rows whose unpadded ids are unseen, reusing the rest.

        A hit returns a vector produced under some other batch's padding, which differs from the
        full-batch encode by ~1e-7 -- the same order as the batch-vs-per-item padding drift the
        runner documents (benchmark/runner.py:222-228). Bounded, not zero: see the probe.
        """
        host = flat_ids.detach().to("cpu")
        prefix = self._option_cache_key_prefix(flat_ids)
        lengths = flat_mask.sum(dim=1).tolist()
        keys = [
            (*prefix, host[n, : int(length)].numpy().tobytes()) for n, length in enumerate(lengths)
        ]

        rows: list[torch.Tensor | None] = [None] * len(keys)
        miss_rows: list[int] = []
        miss_keys: list[tuple] = []
        rows_of_key: dict[tuple, list[int]] = {}
        for n, key in enumerate(keys):
            hit = self._option_vec_cache.get(key)
            if hit is not None:
                rows[n] = hit
                self.option_cache_stats["hits"] += 1
                continue
            first = rows_of_key.get(key)
            if first is None:
                rows_of_key[key] = [n]
                miss_keys.append(key)
                miss_rows.append(n)
            else:
                first.append(n)

        if miss_rows:
            # Encode the first occurrence of each unseen key once; duplicates in this batch reuse it.
            encoded = self._encode_flat(flat_ids[miss_rows], flat_mask[miss_rows])
            self.option_cache_stats["encoded"] += len(miss_rows)
            for key, vec in zip(miss_keys, encoded):
                vec = vec.detach().clone()  # a view would pin the whole sub-batch in the cache
                for n in rows_of_key[key]:
                    rows[n] = vec
                if len(self._option_vec_cache) < self.option_cache_max:
                    self._option_vec_cache[key] = vec
        return torch.stack(rows)  # type: ignore[arg-type]
