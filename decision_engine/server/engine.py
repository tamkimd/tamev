# decision_engine/server/engine.py
"""
Unified TAMEV Inference Engine.
Coordinates model loading, tokenizer, and prediction for Choice, Noul, and Score questions
with exact mathematical permutation equivariance and microsecond Flyweight embedding caching.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

from decision_engine.config.base import ModelConfig, TamevConfig
from decision_engine.models import BaseDecisionModel, ModelFactory
from decision_engine.models.tinybert_model import TamevTinyBertDecisionModel
from decision_engine.server.cache import (
    LRUEmbeddingCache,
    get_context_cache,
    get_option_cache,
)
from decision_engine.server.handlers import (
    QuestionHandlerRegistry,
    choice_confidence,
    option_text,
    render,
    round_prob,
    score_confidence,
)

logger = logging.getLogger("tamev.engine")


class TamevEngine:
    """
    TAMEV Decision Model Inference Engine.
    Executes pointer-network scoring over candidate options with exact permutation equivariance.
    Supports pluggable backbones, model families, and Flyweight LRU caching for ultra-low latency.
    """

    def __init__(  # noqa: PLR0917
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        backbone: str | None = None,
        model_type: str | None = None,
        model: BaseDecisionModel | None = None,
        model_config: ModelConfig | None = None,
        config: TamevConfig | str | Path | None = None,
        *,
        enable_cache: bool = True,
    ):
        if config is not None:
            cfg = TamevConfig.from_yaml(config) if isinstance(config, (str, Path)) else config
        else:
            cfg = TamevConfig()

        if model_type is not None:
            cfg.model.model_type = model_type
        if backbone is not None:
            cfg.model.backbone = backbone
        if model_config is not None:
            cfg.model = model_config

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.backbone = cfg.model.backbone
        self.model_type = cfg.model.model_type
        self.enable_cache = enable_cache
        self.option_cache: LRUEmbeddingCache = get_option_cache()
        self.context_cache: LRUEmbeddingCache = get_context_cache()

        logger.info(
            "Initializing TAMEV %s model (%s) on device: %s",
            self.model_type,
            self.backbone,
            self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "[PAD]"

        if model is not None:
            self.model = model
        elif model_config is not None:
            self.model = ModelFactory.create(model_config)
        elif model_type is not None and model_type != "tinybert":
            self.model = ModelFactory.create(cfg.model)
        else:
            self.model = TamevTinyBertDecisionModel(backbone_name_or_path=self.backbone)

        # Locate best available trained checkpoint if model was not pre-instantiated
        if model is None:
            ckpt_candidate = None
            if checkpoint_path:
                ckpt_candidate = Path(checkpoint_path)
            else:
                for p in [
                    Path("runs/tamev_distill_rl/tamev_tinybert_distill_rl_best.pt"),
                    Path("runs/tamev_super_tiny_tinybert/tamev_tinybert_best.pt"),
                    Path("models/exported/nano/tamev_nano_int8.pt"),
                    Path("models/exported/nano/tamev_nano_fp32.pt"),
                    Path("models/exported/tamev_tinybert_int8.pt"),
                ]:
                    if p.exists():
                        ckpt_candidate = p
                        break

            if ckpt_candidate and ckpt_candidate.exists():
                logger.info("Loading model weights from: %s", ckpt_candidate)
                try:
                    ckpt = torch.load(ckpt_candidate, map_location="cpu", weights_only=False)
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    self.model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    logger.warning("Could not load weights from %s: %s", ckpt_candidate, e)
            elif checkpoint_path and "/" in checkpoint_path and not Path(checkpoint_path).exists():
                try:
                    from huggingface_hub import hf_hub_download
                    from safetensors.torch import load_file

                    logger.info("Downloading weights from Hugging Face Hub: %s", checkpoint_path)
                    weight_file = hf_hub_download(
                        repo_id=checkpoint_path, filename="model.safetensors"
                    )
                    state_dict = load_file(weight_file)
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(
                        "Successfully loaded weights from Hugging Face Hub: %s", checkpoint_path
                    )
                except Exception as e:
                    logger.warning(
                        "Could not load from Hugging Face Hub (%s): %s", checkpoint_path, e
                    )
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    from safetensors.torch import load_file

                    hf_repo = "Tamkimd/tamev-nano-tinybert"
                    logger.info(
                        "No local checkpoint found; auto-downloading canonical model from Hugging Face: %s...",
                        hf_repo,
                    )
                    weight_file = hf_hub_download(repo_id=hf_repo, filename="model.safetensors")
                    state_dict = load_file(weight_file)
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(
                        "Successfully loaded canonical weights from Hugging Face Hub: %s", hf_repo
                    )
                except Exception as e:
                    logger.warning(
                        "No trained checkpoint found and HF download failed (%s); using initialized weights.",
                        e,
                    )

        self.model.to(self.device)
        self.model.eval()

        # Warmup forward pass
        self._warmup()

    def _warmup(self) -> None:
        try:
            self.predict_choice(
                "system warmup", "choose action", {"ping": "test ping", "pong": "test pong"}
            )
        except Exception as e:
            logger.warning("Warmup failed: %s", e)

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Returns statistics on Flyweight LRU embedding cache hits and misses."""
        return {
            "option_cache": {
                "hits": self.option_cache.hits,
                "misses": self.option_cache.misses,
                "hit_rate": round(self.option_cache.hit_rate, 4),
                "size": len(self.option_cache._cache),
                "maxsize": self.option_cache.maxsize,
            },
            "context_cache": {
                "hits": self.context_cache.hits,
                "misses": self.context_cache.misses,
                "hit_rate": round(self.context_cache.hit_rate, 4),
                "size": len(self.context_cache._cache),
                "maxsize": self.context_cache.maxsize,
            },
        }

    def clear_cache(self) -> None:
        """Clears both option and context embedding caches."""
        self.option_cache.clear()
        self.context_cache.clear()

    def _score_options(
        self, state_text: str, question_text: str, options: list[tuple[str, str]]
    ) -> list[float]:
        """
        Runs pointer network forward pass over options.
        Guarantees exact permutation invariance because each option is encoded independently.
        Uses Flyweight Cache to eliminate redundant context and option encodings.
        """
        k = len(options)
        if k == 0:
            return []

        sep = (
            self.tokenizer.sep_token
            if hasattr(self.tokenizer, "sep_token") and self.tokenizer.sep_token
            else " "
        )
        ctx_text = (
            f"{state_text} {sep} {question_text}".strip() if question_text else state_text.strip()
        )
        opt_texts = tuple(opt[1] for opt in options)

        # Cross fusion mixes the context into every option encoding, so a cached option vector
        # is only valid for the context that produced it -- the (model, options, device) cache
        # key cannot express that dependency. Keep the decoupled fast path for bilinear only.
        can_decouple = (
            self.enable_cache
            and getattr(self.model, "option_fusion", "bilinear") != "cross"
            and hasattr(self.model, "encode_context")
            and hasattr(self.model, "encode_options")
            and hasattr(self.model, "pointer_head")
            and self.model.pointer_head is not None
        )

        with torch.no_grad():
            if can_decouple:
                # 1. Flyweight context embedding retrieval
                ctx_cache_key = (id(self.model), ctx_text, str(self.device))
                ctx_vec = self.context_cache.get(ctx_cache_key)
                if ctx_vec is None:
                    ctx_enc = self.tokenizer(
                        [ctx_text],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    )
                    ctx_ids = ctx_enc["input_ids"].to(self.device)
                    ctx_mask = ctx_enc["attention_mask"].to(self.device)
                    ctx_vec = self.model.encode_context(ctx_ids, ctx_mask)
                    self.context_cache.put(ctx_cache_key, ctx_vec)

                # 2. Flyweight option embedding retrieval
                opt_cache_key = (id(self.model), opt_texts, str(self.device))
                opt_vecs = self.option_cache.get(opt_cache_key)
                if opt_vecs is None:
                    opt_enc = self.tokenizer(
                        list(opt_texts),
                        padding=True,
                        truncation=True,
                        max_length=48,
                        return_tensors="pt",
                    )
                    opt_ids = opt_enc["input_ids"].unsqueeze(0).to(self.device)
                    opt_mask = opt_enc["attention_mask"].unsqueeze(0).to(self.device)
                    opt_vecs = self.model.encode_options(opt_ids, opt_mask)
                    self.option_cache.put(opt_cache_key, opt_vecs)

                # 3. Sub-millisecond bilinear pointer projection
                _, probs = self.model.pointer_head(ctx_vec, opt_vecs)
                return probs[0, :k].float().cpu().numpy().tolist()

            # Fallback path for non-decoupled / monolithic models
            ctx_enc = self.tokenizer(
                [ctx_text], padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            opt_enc = self.tokenizer(
                list(opt_texts),
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors="pt",
            )
            ctx_ids = ctx_enc["input_ids"].to(self.device)
            ctx_mask = ctx_enc["attention_mask"].to(self.device)
            opt_ids = opt_enc["input_ids"].unsqueeze(0).to(self.device)
            opt_mask = opt_enc["attention_mask"].unsqueeze(0).to(self.device)
            out = self.model(ctx_ids, ctx_mask, opt_ids, opt_mask, num_options=[k])
            return out["probs"][0, :k].float().cpu().numpy().tolist()

    def predict_choice(
        self, state_str: str, instructions: str, criteria: dict[str, Any]
    ) -> dict[str, Any]:
        """Answers a Choice question using ChoiceQuestionHandler strategy."""
        return QuestionHandlerRegistry.get("choice").handle(self, state_str, instructions, criteria)

    def predict_noul(
        self, state_str: str, instructions: str, criteria: dict[Any, Any] | None = None
    ) -> dict[str, Any]:
        """Answers a Noul (Yes/No) question using NoulQuestionHandler strategy."""
        return QuestionHandlerRegistry.get("noul").handle(self, state_str, instructions, criteria)

    def predict_score(
        self, state_str: str, instructions: str, criteria: list[Any]
    ) -> dict[str, Any]:
        """Answers a Score question using ScoreQuestionHandler strategy."""
        return QuestionHandlerRegistry.get("score").handle(self, state_str, instructions, criteria)

    def predict_batch(
        self, state: Any, questions: dict[str, Any], model_name: str = "tamev-latest"
    ) -> dict[str, Any]:
        """
        Evaluates a complete System One request containing 1 or more questions.
        Dispatches each question dynamically through QuestionHandlerRegistry.
        """
        t0 = time.perf_counter()
        state_str = render(state) if not isinstance(state, str) else state
        answers = {}

        for qid, q_data in questions.items():
            if isinstance(q_data, dict):
                qtype = q_data.get("type")
                instructions = render(q_data.get("instructions", ""))
                criteria = q_data.get("criteria")
            else:
                qtype = getattr(q_data, "type", None)
                instructions = render(getattr(q_data, "instructions", ""))
                criteria = getattr(q_data, "criteria", None)

            # Auto-detect question type if not explicitly specified
            if qtype is None:
                if isinstance(criteria, list):
                    qtype = "score"
                elif isinstance(criteria, dict):
                    qtype = "choice"
                else:
                    qtype = "noul"

            handler = QuestionHandlerRegistry.get(qtype)
            answers[qid] = handler.handle(self, state_str, instructions, criteria)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Token usage billing approximation
        input_tokens = len(self.tokenizer(state_str, add_special_tokens=False)["input_ids"])
        output_tokens = len(
            self.tokenizer(json.dumps(answers), add_special_tokens=False)["input_ids"]
        )

        return {
            "model": model_name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "latency_ms": round(dt_ms, 2),
        }

    def check_permutation(
        self, state: Any, question: dict[str, Any], num_perms: int = 4
    ) -> dict[str, Any]:
        """
        Tests option permutation invariance for a Choice question.
        Returns drift and probability distributions under various permutations.
        """
        state_str = render(state) if not isinstance(state, str) else state
        instructions = render(question.get("instructions", ""))
        criteria = question.get("criteria", {})
        if not isinstance(criteria, dict) or len(criteria) < 2:
            return {
                "permutation_invariant": True,
                "is_invariant": True,
                "max_drift": 0.0,
                "tested_permutations": [],
            }

        keys = list(criteria.keys())
        base_res = self.predict_choice(state_str, instructions, criteria)
        base_probs = base_res["probabilities"]

        perms_tested = [{"order": keys, "drift": 0.0, "probabilities": base_probs}]
        max_drift = 0.0

        for i in range(1, num_perms):
            if i == 1:
                perm_keys = keys[::-1]  # reverse
            elif i == 2:
                perm_keys = keys[1:] + keys[:1]  # cyclic shift
            else:
                perm_keys = list(np.random.permutation(keys))

            perm_crit = {k: criteria[k] for k in perm_keys}
            perm_res = self.predict_choice(state_str, instructions, perm_crit)
            perm_probs = perm_res["probabilities"]

            drift = sum(abs(base_probs[k] - perm_probs[k]) for k in keys)
            max_drift = max(max_drift, drift)
            perms_tested.append(
                {"order": perm_keys, "drift": round(float(drift), 8), "probabilities": perm_probs}
            )

        return {
            "permutation_invariant": bool(max_drift < 1e-4),
            "is_invariant": bool(max_drift < 1e-4),
            "max_drift": round(float(max_drift), 8),
            "original_order": keys,
            "permuted_order": perms_tested[-1]["order"] if perms_tested else keys,
            "tested_permutations": perms_tested,
        }


__all__ = [
    "TamevEngine",
    "choice_confidence",
    "option_text",
    "render",
    "round_prob",
    "score_confidence",
]
