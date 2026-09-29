# decision_engine/data/collator.py
"""
Data Collator for Vectorized Batch Inference and Training in TAMEV.
Properly pads variable-length contexts and variable-cardinality candidate option sets.
"""

import json
import random
from pathlib import Path
from typing import Any

import torch

from decision_engine.data.schema import DecisionItem, OptionItem, build_context


def _cell_label(item: DecisionItem) -> str:
    """Canonical "<source>/<qid>" cell key (e.g. "yelp/rating") used by cell-weight lookup."""
    meta = item.metadata or {}
    source = meta.get("source") or item.task
    qid = meta.get("qid")
    return f"{source}/{qid}" if qid else str(source)


class TamevCollator:
    """
    Collates a list of DecisionItem instances into padded PyTorch tensors.
    Supports context [SEP] formatting, negative option subsampling during training,
    and valid teacher probability masking.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_ctx_len: int = 128,
        max_opt_len: int = 48,
        max_options: int | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
        max_context_len: int | None = None,
        max_option_len: int | None = None,
        hard_negatives_path: str | Path | None = None,
    ):
        self.tokenizer = tokenizer
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "[PAD]"
        self.max_ctx_len = max_context_len if max_context_len is not None else max_ctx_len
        self.max_opt_len = max_option_len if max_option_len is not None else max_opt_len
        self.max_options = max_options
        self.device = device
        # Explicit RNG so subsampling is reproducible; None keeps OS-entropy seeding
        self.rng = random.Random(seed)
        # Offline-mined hard negatives (row id -> ranked wrong-option ids). None keeps the
        # historical random subsampling byte-identical.
        self.hard_negatives = self._load_hard_negatives(hard_negatives_path)

    @staticmethod
    def _load_hard_negatives(path: str | Path | None) -> dict[str, list[str]]:
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"hard negatives file not found: {p}")
        rows = json.loads(p.read_text(encoding="utf-8")).get("rows", {})
        return {str(k): list(v.get("hard_negatives", [])) for k, v in rows.items()}

    def __call__(self, batch: list[DecisionItem]) -> dict[str, Any]:
        b_size = len(batch)
        if b_size == 0:
            return {}

        # The same string the serving paths build (`build_context`), so nothing is trained on
        # text the model is never scored on. Length is governed by `max_ctx_len` alone.
        contexts = [build_context(item.state, item.question) for item in batch]

        # Tokenize contexts
        ctx_enc = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_ctx_len,
            return_tensors="pt",
        )

        processed_options: list[list[OptionItem]] = []
        processed_labels: list[int] = []
        is_ordinal: list[bool] = []
        has_teacher_list: list[bool] = []
        teacher_dist_list: list[list[float]] = []

        for item in batch:
            opts = list(item.options)
            target_idx = item.target_index
            # `score` items are generated ascending in value, so the option index is the ordinal
            # position. Subsampling reshuffles the option axis and would invalidate that.
            ordinal = item.decision_type == "score"

            # Negative option subsampling if max_options is specified and cardinality exceeds threshold
            if self.max_options and len(opts) > self.max_options:
                ordinal = False
                target_opt = opts[target_idx]
                other_opts = [o for idx, o in enumerate(opts) if idx != target_idx]
                sampled = self._subsample_negatives(item, other_opts)
                new_opts = [*sampled, target_opt]
                self.rng.shuffle(new_opts)
                opts = new_opts
                target_idx = opts.index(target_opt)

            processed_options.append(opts)
            processed_labels.append(target_idx)
            is_ordinal.append(ordinal)

            k = len(opts)
            if item.teacher_probs:
                p_arr = [float(item.teacher_probs.get(o.id, 0.0)) for o in opts]
                p_sum = sum(p_arr)
                if p_sum > 0:
                    norm_p = [p / p_sum for p in p_arr]
                    is_uniform = all(abs(p - 1.0 / k) < 1e-4 for p in norm_p)
                    if not is_uniform:
                        has_teacher_list.append(True)
                        teacher_dist_list.append(norm_p)
                        continue
            has_teacher_list.append(False)
            teacher_dist_list.append([1.0 / k] * k)

        k_list = [len(opts) for opts in processed_options]
        max_k = max(k_list)

        # Collect all option texts, padding missing slots with dummy token
        flat_options: list[str] = []
        for opts in processed_options:
            opt_texts = [o.text for o in opts]
            opt_texts += [""] * (max_k - len(opt_texts))
            flat_options.extend(opt_texts)

        # Tokenize options
        opt_enc = self.tokenizer(
            flat_options,
            padding=True,
            truncation=True,
            max_length=self.max_opt_len,
            return_tensors="pt",
        )

        opt_len = opt_enc["input_ids"].shape[-1]
        opt_input_ids = opt_enc["input_ids"].view(b_size, max_k, opt_len)
        opt_attention_mask = opt_enc["attention_mask"].view(b_size, max_k, opt_len)

        # Zero out attention mask for dummy padded options
        for i, k in enumerate(k_list):
            if k < max_k:
                opt_attention_mask[i, k:, :] = 0

        labels = torch.tensor(processed_labels, dtype=torch.long)
        teacher_probs = torch.zeros((b_size, max_k), dtype=torch.float32)
        has_teacher = torch.tensor(has_teacher_list, dtype=torch.bool)

        for i, dist in enumerate(teacher_dist_list):
            teacher_probs[i, : len(dist)] = torch.tensor(dist, dtype=torch.float32)

        result = {
            "ctx_input_ids": ctx_enc["input_ids"],
            "ctx_attention_mask": ctx_enc["attention_mask"],
            "opt_input_ids": opt_input_ids,
            "opt_attention_mask": opt_attention_mask,
            "labels": labels,
            "teacher_probs": teacher_probs,
            "has_teacher": has_teacher,
            "num_options": k_list,
            "ordinal": torch.tensor(is_ordinal, dtype=torch.bool),
            "tasks": [item.task for item in batch],
            "cells": [_cell_label(item) for item in batch],
        }

        if self.device is not None:
            result = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }

        return result

    def _subsample_negatives(
        self, item: DecisionItem, other_opts: list[OptionItem]
    ) -> list[OptionItem]:
        """Mined hard negatives when available for this row, else a random sample.

        The mined entry (row id -> ranked wrong-option ids) is only used if it supplies the
        full ``max_options - 1`` slots from *this* item's options, so a row-id collision or a
        stale cache falls back to random instead of under-filling the option set.
        """
        n = self.max_options - 1
        mined_ids = self.hard_negatives.get(str(item.id))
        if mined_ids:
            by_id = {o.id: o for o in other_opts}
            picked = [by_id[oid] for oid in mined_ids if oid in by_id]
            if len(picked) >= n:
                return picked[:n]
        return self.rng.sample(other_opts, n)


# Convenience alias for batch collation in fine-tuning and inference pipelines
DecisionBatchCollator = TamevCollator
