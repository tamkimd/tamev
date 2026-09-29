# decision_engine/models/incontext_causal_model.py
"""
In-Context Causal Decision Model Architecture.
Combines causal autoregressive backbones with special delimiter tokens, joint attention,
and TAMEV's Option Isolation Attention Mask for exact 0.00000000 permutation invariance.

Supports:
- Native Qwen (Qwen2.5, Qwen3.5) special tokens (<|fim_prefix|>, <|fim_middle|>, <|box_start|>, <|box_end|>, <|fim_suffix|>).
- LoRA parameter-efficient fine-tuning via PEFT.
- Option Isolation attention masking to eliminate order bias.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from decision_engine.config.base import ModelConfig
from decision_engine.models.base import BaseDecisionModel, ModelRegistry
from decision_engine.models.pointer_head import PointerHead

DEFAULT_SPECIAL_TOKENS = [
    "<|fim_prefix|>",  # State prefix
    "<|fim_middle|>",  # Question prefix
    "<|box_start|>",  # Option opening delimiter
    "<|box_end|>",  # Option closing delimiter
    "<|fim_suffix|>",  # Decision query readout token
]


@ModelRegistry.register("incontext_causal")
@ModelRegistry.register("incontext")
@ModelRegistry.register("kev")
class InContextCausalDecisionModel(BaseDecisionModel):
    """
    In-Context Causal Decision Model with joint attention and optional Option Isolation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device_type = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.load_dtype = torch.bfloat16 if self.device_type in ("mps", "cuda") else torch.float32

        self.option_isolation = bool(getattr(config, "option_isolation", True))
        self.init_from = getattr(config, "init_from", None)
        self.use_lora = bool(getattr(config, "use_lora", False))
        self.freeze_backbone = bool(getattr(config, "freeze_backbone", False))

        # 1. Initialize or Load Tokenizer
        backbone_path = config.backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_path, trust_remote_code=True)
        self._setup_delimiter_tokens()

        # 2. Initialize Backbone Model
        if self.init_from:
            self._load_from_pretrained_kev(self.init_from, config)
        else:
            self._init_standard_backbone(config)

        # 3. Setup Bilinear Pointer Readout Head
        hidden_dim = self._resolve_hidden_dim(config)
        projection_dim = getattr(config, "projection_dim", 256)
        temperature = float(getattr(config, "temperature", 2.351))

        if not hasattr(self, "q_proj"):
            self.q_proj = nn.Linear(hidden_dim, projection_dim, bias=True)
            self.k_proj = nn.Linear(hidden_dim, projection_dim, bias=True)
            self.scale = 1.0 / math.sqrt(projection_dim)
            self.temperature_param = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

        # Maintain compatibility with BaseDecisionModel pointer_head
        self.pointer_head = PointerHead(
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            temperature=temperature,
        )

        try:
            bb_dtype = next(self.backbone.parameters()).dtype
            self.q_proj = self.q_proj.to(bb_dtype)
            self.k_proj = self.k_proj.to(bb_dtype)
            self.pointer_head = self.pointer_head.to(bb_dtype)
        except StopIteration:
            pass

    @property
    def temperature(self) -> torch.Tensor:
        return self.temperature_param

    def set_temperature(self, t: float) -> None:
        with torch.no_grad():
            self.temperature_param.copy_(
                torch.as_tensor(t, dtype=torch.float32, device=self.temperature_param.device)
            )
            self.pointer_head.set_temperature(t)

    def _resolve_hidden_dim(self, config: ModelConfig) -> int:
        if config.hidden_dim is not None:
            return config.hidden_dim
        bb_cfg = getattr(self.backbone, "config", None)
        if bb_cfg:
            if hasattr(bb_cfg, "hidden_size") and bb_cfg.hidden_size is not None:
                return bb_cfg.hidden_size
            if hasattr(bb_cfg, "text_config") and hasattr(bb_cfg.text_config, "hidden_size"):
                return bb_cfg.text_config.hidden_size
        return 576

    def _setup_delimiter_tokens(self) -> None:
        """Ensure delimiter tokens have corresponding IDs in tokenizer."""
        vocab = self.tokenizer.get_vocab()
        all_present = all(tok in vocab for tok in DEFAULT_SPECIAL_TOKENS)
        if not all_present:
            missing = [tok for tok in DEFAULT_SPECIAL_TOKENS if tok not in vocab]
            self.tokenizer.add_special_tokens({"additional_special_tokens": missing})

        self.s_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPECIAL_TOKENS[0])
        self.q_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPECIAL_TOKENS[1])
        self.o_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPECIAL_TOKENS[2])
        self.c_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPECIAL_TOKENS[3])
        self.d_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPECIAL_TOKENS[4])

    def _init_standard_backbone(self, config: ModelConfig) -> None:
        """Initialize standard base model with optional LoRA."""
        self.backbone = AutoModel.from_pretrained(
            config.backbone,
            dtype=self.load_dtype,
            trust_remote_code=True,
        )

        bb_cfg = getattr(self.backbone, "config", None)
        vocab_size = (
            getattr(bb_cfg, "vocab_size", None)
            or getattr(getattr(bb_cfg, "text_config", None), "vocab_size", None)
            or 151936
        )
        if len(self.tokenizer) > vocab_size:
            self.backbone.resize_token_embeddings(len(self.tokenizer))

        if self.use_lora:
            try:
                from peft import LoraConfig, get_peft_model

                lora_cfg = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    lora_dropout=0.05,
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_cfg)
            except ImportError:
                pass
        elif self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _load_from_pretrained_kev(self, checkpoint_source: str, config: ModelConfig) -> None:
        """Load weights from a Kev checkpoint repository or local directory."""
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        lm = AutoModelForCausalLM.from_pretrained(
            config.backbone,
            dtype=self.load_dtype,
            trust_remote_code=True,
        )
        base_model = lm.model if hasattr(lm, "model") else lm
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_source)
        self.backbone = peft_model.merge_and_unload()
        self.backbone.eval()

        # Download or load pointer head
        if os.path.exists(os.path.join(checkpoint_source, "head.pt")):
            head_path = os.path.join(checkpoint_source, "head.pt")
        else:
            head_path = hf_hub_download(checkpoint_source, "head.pt")

        head_data = torch.load(head_path, map_location="cpu", weights_only=False)
        head_state = head_data.get("head", head_data)
        temp = float(head_data.get("temperature", 2.351))

        proj_dim = head_state["q.weight"].shape[0]
        hidden_dim = head_state["q.weight"].shape[1]

        self.q_proj = nn.Linear(hidden_dim, proj_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, proj_dim, bias=True)
        self.q_proj.weight.data.copy_(head_state["q.weight"].to(self.load_dtype))
        self.q_proj.bias.data.copy_(head_state["q.bias"].to(self.load_dtype))
        self.k_proj.weight.data.copy_(head_state["k.weight"].to(self.load_dtype))
        self.k_proj.bias.data.copy_(head_state["k.bias"].to(self.load_dtype))
        self.temperature_param = nn.Parameter(torch.tensor(temp, dtype=torch.float32))
        self.scale = 1.0 / math.sqrt(proj_dim)

    def build_incontext_sequence(
        self,
        state: str,
        question: str,
        options: list[str],
        max_state_len: int = 512,
        max_opt_len: int = 128,
    ) -> tuple[list[int], list[int], int, list[tuple[int, int]], int]:
        """
        Packs (state, question, options) into Kev token sequence:
        [<|fim_prefix|>, state_tokens, <|fim_middle|>, question_tokens,
         <|box_start|>, opt0, <|box_end|>, ..., <|fim_suffix|>]

        Returns:
            input_ids: full token sequence
            opt_ends: positions of each option's <|box_end|>
            decide_pos: position of <|fim_suffix|>
            opt_spans: (start, end) index pairs for each option
            prefix_len: length of state + question prefix
        """
        tok = self.tokenizer
        state_tokens = tok.encode(str(state), add_special_tokens=False)[:max_state_len]
        question_tokens = tok.encode(str(question), add_special_tokens=False)[:max_state_len]

        ids = [self.s_id, *state_tokens, self.q_id, *question_tokens]
        prefix_len = len(ids)

        opt_ends = []
        opt_spans = []

        for opt in options:
            start_pos = len(ids)
            ids.append(self.o_id)
            ids.extend(tok.encode(str(opt), add_special_tokens=False)[:max_opt_len])
            ids.append(self.c_id)
            end_pos = len(ids) - 1
            opt_ends.append(end_pos)
            opt_spans.append((start_pos, end_pos))

        ids.append(self.d_id)
        decide_pos = len(ids) - 1

        return ids, opt_ends, decide_pos, opt_spans, prefix_len

    def build_option_isolation_mask_and_positions(
        self,
        seq_len: int,
        prefix_len: int,
        opt_spans: list[tuple[int, int]],
        decide_pos: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs an Option Isolation 4D attention mask and position IDs.
        Guarantees options cannot attend to each other, ensuring 0.00000000 permutation drift!
        """
        mask = torch.full((1, 1, seq_len, seq_len), -1e4, device=device, dtype=torch.float32)
        pos_ids = torch.zeros((1, seq_len), device=device, dtype=torch.long)

        # 1. Prefix (State + Question): standard causal attention
        for i in range(prefix_len):
            pos_ids[0, i] = i
            mask[0, 0, i, : i + 1] = 0.0

        # 2. Options: attend to prefix + own option tokens (isolated from other options)
        max_opt_tokens = 0
        for _opt_idx, (start_idx, end_idx) in enumerate(opt_spans):
            opt_len = end_idx - start_idx + 1
            if opt_len > max_opt_tokens:
                max_opt_tokens = opt_len

            for i in range(start_idx, end_idx + 1):
                rel_pos = i - start_idx
                # Options share the exact same relative position offset after prefix
                pos_ids[0, i] = prefix_len + rel_pos

                # Attend to full state and question prefix
                mask[0, 0, i, :prefix_len] = 0.0
                # Attend within current option up to token i
                mask[0, 0, i, start_idx : i + 1] = 0.0

        # 3. Decision Query Readout token (<|fim_suffix|>)
        # Attends to state and question prefix (and itself) to remain order-invariant
        pos_ids[0, decide_pos] = prefix_len + max_opt_tokens + 1
        mask[0, 0, decide_pos, :prefix_len] = 0.0
        mask[0, 0, decide_pos, decide_pos] = 0.0

        return mask, pos_ids

    def forward_incontext(
        self,
        state: str,
        question: str,
        options: list[str],
        option_isolation: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Performs full forward pass for an in-context decision query.
        """
        isolate = self.option_isolation if option_isolation is None else option_isolation
        device = next(self.parameters()).device

        seq_ids, opt_ends, decide_pos, opt_spans, prefix_len = self.build_incontext_sequence(
            state, question, options
        )
        L = len(seq_ids)
        input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)

        if isolate:
            attn_mask, pos_ids = self.build_option_isolation_mask_and_positions(
                seq_len=L,
                prefix_len=prefix_len,
                opt_spans=opt_spans,
                decide_pos=decide_pos,
                device=device,
            )
            attn_mask = attn_mask.to(self.load_dtype)
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask, position_ids=pos_ids)
        else:
            # Standard causal attention
            out = self.backbone(input_ids=input_ids)

        hidden_states = out.last_hidden_state  # (1, L, hidden_dim)

        h_decide = hidden_states[:, decide_pos]  # (1, hidden_dim)
        h_opts = hidden_states[:, opt_ends]  # (1, K, hidden_dim)

        # Linear projections
        q = self.q_proj(h_decide)  # (1, projection_dim)
        k = self.k_proj(h_opts)  # (1, K, projection_dim)

        # Bilinear compatibility dot product: (1, K, D) x (1, D) -> (1, K)
        t = torch.clamp(self.temperature_param, min=0.05)
        logits = torch.einsum("bkd,bd->bk", k, q) * self.scale
        calibrated_logits = logits / t
        probs = F.softmax(calibrated_logits.float(), dim=-1)

        return {
            "logits": logits,
            "calibrated_logits": calibrated_logits,
            "probs": probs,
            "q": q,
            "k": k,
        }

    def predict_decision(
        self,
        state: str,
        question: str,
        options: list[str | dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """High-level decision prediction."""
        opt_texts = [o["text"] if isinstance(o, dict) else str(o) for o in options]
        with torch.no_grad():
            res = self.forward_incontext(state, question, opt_texts, **kwargs)

        probs_np = res["probs"][0].float().cpu().numpy()
        ranked_indices = np.argsort(-probs_np)

        ranked = []
        for idx in ranked_indices:
            orig_opt = options[idx]
            ranked.append(
                {
                    "index": int(idx),
                    "option": orig_opt,
                    "probability": float(probs_np[idx]),
                }
            )

        return {
            "logits": res["logits"],
            "probs": res["probs"],
            "best_index": int(ranked_indices[0]),
            "best_option": options[ranked_indices[0]],
            "ranked": ranked,
        }

    def predict_proba(
        self,
        context: str,
        options: list[str],
        tokenizer: Any = None,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        BenchmarkRunner and evaluate_local_model compatibility method.
        """
        # Context may contain "state\nquestion" or just state
        parts = context.split("\n", 1)
        state = parts[0]
        question = parts[1] if len(parts) > 1 else "What is the optimal action?"

        with torch.no_grad():
            res = self.forward_incontext(state, question, options)
        return res["probs"][0].float().cpu().numpy()

    # Decoupled Fallback Methods for BaseDecisionModel compatibility
    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq_lens = torch.clamp(attention_mask.sum(dim=1) - 1, min=0).long()
        b_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        return out.last_hidden_state[b_idx, seq_lens]

    def encode_options(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, K, L = input_ids.shape
        flat_ids = input_ids.view(B * K, L)
        flat_mask = attention_mask.view(B * K, L).clone()
        flat_mask[:, 0] = 1
        out = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
        seq_lens = torch.clamp(flat_mask.sum(dim=1) - 1, min=0).long()
        b_idx = torch.arange(B * K, device=input_ids.device)
        pooled = out.last_hidden_state[b_idx, seq_lens]
        return pooled.view(B, K, -1)
