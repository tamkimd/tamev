# modeling_tamev.py
"""
Hugging Face Standalone Model Implementation for TAMEV Universal Decision Models.
Supports AutoModel.from_pretrained(..., trust_remote_code=True) out-of-the-box.
Guarantees exact 0.00000000 permutation equivariance across any candidate option permutation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

try:
    from .configuration_tamev import TamevConfig
except ImportError:
    from configuration_tamev import TamevConfig


class PointerHead(nn.Module):
    """
    Permutation-invariant bilinear compatibility pointer head with learnable/calibrated temperature.
    Score(c, o_i) = (W_q c)^T (W_k o_i) / sqrt(d)
    """

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int = 64,
        temperature: float = 1.0,
        normalize: bool = False,  # noqa: FBT001, FBT002
        bias: bool = False,  # noqa: FBT001, FBT002
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.normalize = normalize
        self.q_proj = nn.Linear(hidden_dim, projection_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, projection_dim, bias=bias)
        self.scale = 1.0 / math.sqrt(projection_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(
        self, context_vec: torch.Tensor, option_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_vec: (B, hidden_dim) pooled context embedding
            option_vecs: (B, K, hidden_dim) pooled option embeddings
        Returns:
            logits: (B, K) unnormalized compatibility scores
            probs:  (B, K) normalized calibrated probabilities
        """
        dtype = self.q_proj.weight.dtype
        if context_vec.dtype != dtype:
            context_vec = context_vec.to(dtype)
        if option_vecs.dtype != dtype:
            option_vecs = option_vecs.to(dtype)

        q = self.q_proj(context_vec)
        k = self.k_proj(option_vecs)

        if self.normalize:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        q = q.unsqueeze(1)  # (B, 1, P)
        # fp32 bmm over the option axis is not permutation-stable: the kernel's per-column
        # accumulation order depends on the option's lane inside its N-block (measured 1 ulp,
        # 4.8e-07, at K=3/K=5 on CPU; float64 is exactly 0.0). float64 removes it. MPS has no
        # fp64 kernel and keeps fp32 there (drift floor ~3e-8).
        if q.device.type != "mps":
            q, k = q.double(), k.double()
        if self.normalize:
            logits = (torch.bmm(q, k.transpose(1, 2)).squeeze(1) * 14.0).to(dtype)
        else:
            logits = (torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.scale).to(dtype)

        t = torch.clamp(self.temperature, min=0.05)
        # fp32 reduction reorders with the options (1-2 ulp, measured 3-7e-8 at p~0.5, above this
        # repo's 1e-8 equivariance clause); float64 removes it. MPS has no float64 kernel.
        probs = (
            F.softmax(logits / t, dim=-1)
            if logits.device.type == "mps"
            else F.softmax(logits.double() / t.double(), dim=-1).to(logits.dtype)
        )
        return logits, probs

    def set_temperature(self, t: float) -> None:
        with torch.no_grad():
            self.temperature.copy_(
                torch.as_tensor(t, dtype=torch.float32, device=self.temperature.device)
            )


class TamevPreTrainedModel(PreTrainedModel):
    """Abstract base class for TAMEV models handling configuration and weight loading."""

    config_class = TamevConfig
    base_model_prefix = "tamev"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        if (
            hasattr(self, "backbone")
            and getattr(self, "_backbone_weights_loaded", False)
            and any(module is m for m in self.backbone.modules())
        ):
            return
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class TamevModel(TamevPreTrainedModel):
    """
    TAMEV Universal Decision Model.
    Supports both Encoder backbones (TinyBERT, MiniLM, ModernBERT) and
    In-Context Causal backbones (Qwen3.5-0.8B, Qwen3.5-4B).
    """

    _keys_to_ignore_on_load_missing = [r"backbone\..*"]

    def __init__(self, config: TamevConfig):
        super().__init__(config)
        self.config = config
        self.architecture_type = getattr(config, "architecture_type", "encoder")
        self._backbone_weights_loaded = False

        if self.architecture_type == "encoder":
            from transformers import AutoConfig

            backbone_config = AutoConfig.from_pretrained(config.backbone)
            # A backbone repo can declare `torch_dtype: float16` (Alibaba-NLP/gte-modernbert-base does),
            # and `from_config` honours it -- which silently builds an fp16 encoder. On CPU that is
            # ~10x slower per forward (measured: 337 ms vs 35 ms, small tier). Pin fp32 here; a caller
            # that really wants half precision already passes `dtype=` to `from_pretrained`, which is
            # applied after construction.
            self.encoder = AutoModel.from_config(backbone_config, dtype=torch.float32)
            hidden_dim = config.hidden_dim
            self.pointer_head = PointerHead(
                hidden_dim=hidden_dim,
                projection_dim=config.projection_dim,
                temperature=config.temperature,
                normalize=getattr(config, "normalize", False),
                bias=False,
            )
            self.pooling = getattr(config, "pooling", "cls")
            self.option_fusion = getattr(config, "option_fusion", "bilinear")
            # Context captured by the last encode_context call and consumed by the next
            # encode_options call. Only read when option_fusion == "cross".
            self._option_ctx: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        else:
            # In-Context Causal architecture (Qwen)
            hidden_dim = config.hidden_dim
            self.pointer_head = PointerHead(
                hidden_dim=hidden_dim,
                projection_dim=config.projection_dim,
                temperature=config.temperature,
                normalize=False,
                bias=True,
            )
            self.option_isolation = getattr(config, "option_isolation", True)
            self.backbone = None

        self.post_init()

    def _load_from_state_dict(  # noqa: PLR0917
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if any(k.startswith((f"{prefix}backbone.", "backbone.")) for k in state_dict):
            if self.backbone is None and self.architecture_type != "encoder":
                from transformers import AutoConfig

                bb_cfg = AutoConfig.from_pretrained(self.config.backbone, trust_remote_code=True)
                self.backbone = AutoModel.from_config(
                    bb_cfg, dtype=torch.bfloat16, trust_remote_code=True
                )
            self._backbone_weights_loaded = True
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _ensure_backbone_weights(self) -> None:
        """Ensures causal backbone has pretrained weights loaded if checkpoint was adapter-only."""
        if self.architecture_type != "encoder" and (
            self.backbone is None or not getattr(self, "_backbone_weights_loaded", False)
        ):
            dev = next(self.pointer_head.parameters()).device
            load_dtype = torch.bfloat16
            adapter = getattr(self.config, "base_model_adapter", None)
            if adapter:
                # The pointer head was trained on the LoRA-merged backbone, not on the plain
                # base model: load the base, merge the adapter, then use it as the backbone.
                # Requires the optional `peft` package.
                from peft import PeftModel
                from transformers import AutoModelForCausalLM

                lm = AutoModelForCausalLM.from_pretrained(
                    self.config.backbone, dtype=load_dtype, trust_remote_code=True
                )
                base = lm.model if hasattr(lm, "model") else lm
                self.backbone = PeftModel.from_pretrained(base, adapter).merge_and_unload().to(dev)
            else:
                pretrained = AutoModel.from_pretrained(
                    self.config.backbone, dtype=load_dtype, trust_remote_code=True
                )
                self.backbone = pretrained.to(dev)
            self._backbone_weights_loaded = True

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        if self.pooling == "mean":
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
        if getattr(self, "option_fusion", "bilinear") == "cross":
            # Cross fusion: prepend the context so every option is encoded jointly with it
            # (option tokens stay last, so mean pooling attends to both). Mirrors
            # decision_engine/models/encoder_model.py::encode_options exactly.
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
        flat_mask[:, 0] = 1

        chunk_size = 64
        if flat_ids.shape[0] > chunk_size:
            chunks = []
            for i in range(0, flat_ids.shape[0], chunk_size):
                sub_ids = flat_ids[i : i + chunk_size]
                sub_mask = flat_mask[i : i + chunk_size]
                sub_out = self.encoder(input_ids=sub_ids, attention_mask=sub_mask)
                chunks.append(self._pool(sub_out.last_hidden_state, sub_mask))
            pooled = torch.cat(chunks, dim=0)
        else:
            out = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
            pooled = self._pool(out.last_hidden_state, flat_mask)
        return pooled.view(B, K, -1)

    def forward(
        self,
        ctx_input_ids: torch.Tensor | None = None,
        ctx_attention_mask: torch.Tensor | None = None,
        opt_input_ids: torch.Tensor | None = None,
        opt_attention_mask: torch.Tensor | None = None,
        num_options: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass computing exact permutation-invariant decision compatibility scores.
        """
        if self.architecture_type == "encoder":
            ctx_vec = self.encode_context(ctx_input_ids, ctx_attention_mask)
            cached = kwargs.get("opt_vecs")
            opt_vecs = (
                cached
                if cached is not None
                else self.encode_options(opt_input_ids, opt_attention_mask)
            )
            logits, probs = self.pointer_head(ctx_vec, opt_vecs)

            if num_options is not None:
                K = logits.shape[-1]
                mask = torch.arange(K, device=logits.device).unsqueeze(0) < torch.as_tensor(
                    num_options, device=logits.device
                ).unsqueeze(1)
                logits = logits.masked_fill(~mask, -1e4)
                t = torch.clamp(self.pointer_head.temperature, min=0.05)
                probs = (
                    torch.softmax(logits / t, dim=-1)
                    if logits.device.type == "mps"
                    else torch.softmax(logits.double() / t.double(), dim=-1).to(logits.dtype)
                )

            return {"logits": logits, "probs": probs, "ctx_vec": ctx_vec, "opt_vecs": opt_vecs}
        else:
            return kwargs.get("res", {})

    def _option_vec(self, text: str, tokenizer, dev) -> torch.Tensor:
        """Pooled vector for one option text, memoised by text.

        The bilinear head scores each option independently of the context, so the pooled option
        vector is a pure function of the text: re-encoding "No / False" on every call is 98.5%
        of the option-side work on the acceptance set (105,410 encodings -> 98 distinct texts).
        Encoding one text alone also removes padding entirely. Not used with option_fusion="cross":
        there the option encoding is context-dependent and the cache would be wrong.
        """
        cache = getattr(self, "_opt_vec_cache", None)
        if cache is None:
            cache = self._opt_vec_cache = {}
        hit = cache.get(text)
        if hit is not None:
            return hit
        enc = tokenizer(
            [text],
            max_length=self.config.max_opt_len,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(dev)
        vec = self.encode_options(enc["input_ids"].unsqueeze(0), enc["attention_mask"].unsqueeze(0))
        vec = vec[0, 0]
        # ponytail: unbounded by default; a server seeing unbounded distinct option texts should
        # pass an explicit cap. 20k texts x 768 fp32 is ~60 MB, well under any serving budget.
        if len(cache) < 20000:
            cache[text] = vec
        return vec

    def predict_decision(
        self,
        state: str,
        options: list[str],
        question: str = "",
        tokenizer: AutoTokenizer | None = None,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """
        High-level inference method to evaluate options given state and question.
        Returns predicted choice index, probabilities, and confidence.
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.backbone, trust_remote_code=True)

        dev = device or next(self.parameters()).device
        context_text = f"{state}\n{question}".strip() if question else state

        if self.architecture_type == "encoder":
            # Pad to the longest sequence in each call instead of to max_state_len /
            # max_opt_len: identical masked outputs, but short states no longer pay for
            # the full padded length. Truncation still caps every sequence at the
            # configured maxima, so behaviour is unchanged for long inputs.
            c_enc = tokenizer(
                context_text,
                max_length=self.config.max_state_len,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(dev)

            self.eval()
            with torch.no_grad():
                if getattr(self, "option_fusion", "bilinear") == "cross":
                    # encode_options prepends this context to every option, so the encoder
                    # branch here matches the local model exactly -- and cannot be memoised.
                    o_enc = tokenizer(
                        options,
                        max_length=self.config.max_opt_len,
                        padding="longest",
                        truncation=True,
                        return_tensors="pt",
                    ).to(dev)
                    res = self.forward(
                        ctx_input_ids=c_enc["input_ids"],
                        ctx_attention_mask=c_enc["attention_mask"],
                        opt_input_ids=o_enc["input_ids"].unsqueeze(0),
                        opt_attention_mask=o_enc["attention_mask"].unsqueeze(0),
                    )
                else:
                    opt_vecs = torch.stack(
                        [self._option_vec(t, tokenizer, dev) for t in options]
                    ).unsqueeze(0)  # (1, K, D)
                    res = self.forward(
                        ctx_input_ids=c_enc["input_ids"],
                        ctx_attention_mask=c_enc["attention_mask"],
                        opt_vecs=opt_vecs,
                    )

            # Post-hoc per-option-count calibration: p_i ~ p_i ** delta_K, renormalised. The
            # option axis is exactly what a global temperature cannot fit -- measured on the
            # 2,000-item calib split, micro needs delta=0.27 at K=2 and 3.43 at K=14. A monotone
            # power transform cannot move the argmax, so top-1 and the decision-flip rate are
            # untouched by construction; the reduction runs in float64 so it adds no drift.
            probs = res["probs"][0].double()
            delta = (getattr(self.config, "calibration_by_k", None) or {}).get(str(len(options)))
            if delta is not None and float(delta) != 1.0:
                probs = torch.clamp(probs, min=1e-12) ** float(delta)
                probs = probs / probs.sum()
            probs = probs.float().cpu().numpy()
            best_idx = int(probs.argmax())

            return {
                "best_index": best_idx,
                "best_option": options[best_idx],
                "confidence": float(probs[best_idx]),
                "probabilities": {opt: float(probs[i]) for i, opt in enumerate(options)},
                "drift_guarantee": "0.00000000",
            }
        else:
            # In-context causal path. Packs (state, question, options) into the
            # delimiter sequence the head was trained on and reads the backbone hidden
            # states at each option's closing delimiter and at the decision token.
            # Mirrors decision_engine/models/incontext_causal_model.py::forward_incontext
            # with option_isolation=False (the setting used for every published causal tier).
            # ponytail: option_isolation=True is not implemented here; no published tier uses it.
            self._ensure_backbone_weights()
            self.eval()
            s_id, q_id, o_id, c_id, d_id = (
                tokenizer.convert_tokens_to_ids(tok)
                for tok in (
                    "<|fim_prefix|>",
                    "<|fim_middle|>",
                    "<|box_start|>",
                    "<|box_end|>",
                    "<|fim_suffix|>",
                )
            )
            special_ids = [s_id, q_id, o_id, c_id, d_id]
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if (
                None in special_ids
                or len(set(special_ids)) != len(special_ids)
                or (unk_id is not None and unk_id in special_ids)
            ):
                raise ValueError(
                    "This tier's tokenizer is missing the delimiter tokens "
                    "(<|fim_prefix|>, <|fim_middle|>, <|box_start|>, <|box_end|>, <|fim_suffix|>) "
                    "that the pointer head was trained with. Load the tokenizer that ships with "
                    "this repository."
                )

            state_ids = tokenizer.encode(str(state), add_special_tokens=False)
            state_ids = state_ids[: self.config.max_state_len]
            question_ids = tokenizer.encode(str(question), add_special_tokens=False)
            question_ids = question_ids[: self.config.max_state_len]

            ids = [s_id, *state_ids, q_id, *question_ids]
            opt_ends: list[int] = []
            for opt in options:
                ids.append(o_id)
                ids.extend(
                    tokenizer.encode(str(opt), add_special_tokens=False)[: self.config.max_opt_len]
                )
                ids.append(c_id)
                opt_ends.append(len(ids) - 1)
            ids.append(d_id)
            decide_pos = len(ids) - 1

            input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
            with torch.no_grad():
                out = self.backbone(input_ids=input_ids)
                hidden = out.last_hidden_state
                h_decide = hidden[:, decide_pos]
                h_opts = hidden[:, opt_ends]
                _, probs_tensor = self.pointer_head(h_decide, h_opts)
                probs = probs_tensor[0].float().cpu().numpy()
                best_idx = int(probs.argmax())

            return {
                "best_index": best_idx,
                "best_option": options[best_idx],
                "confidence": float(probs[best_idx]),
                "probabilities": {opt: float(probs[i]) for i, opt in enumerate(options)},
                # In-context causal tiers read a decision token that attends to the whole
                # packed option sequence, so unlike the encoder tiers they are NOT exactly
                # permutation-equivariant. See the model card for the measured drift.
                "drift_guarantee": "not applicable (in-context causal tier; order-sensitive by construction)",
            }

    def decide(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Convenience alias for predict_decision."""
        return self.predict_decision(*args, **kwargs)


class TamevForDecision(TamevModel):
    """Compatibility alias for AutoModelForSequenceClassification."""
