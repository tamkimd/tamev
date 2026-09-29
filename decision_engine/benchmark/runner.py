# decision_engine/benchmark/runner.py
"""
Unified Benchmark Runner for TAMEV.
Evaluates local models (PyTorch/ONNX) and remote TypeSafe endpoints on:
- Classification accuracy (Top-1, Top-3)
- Calibration (ECE, Brier Score, NLL)
- Permutation Equivariance & Invariance Drift
- Inference Latency Percentiles (p50, p95, mean, QPS)
- Task/Domain Breakdown
"""

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from decision_engine.benchmark.metrics import (
    compute_classification_and_calibration,
    compute_latency_percentiles,
    compute_permutation_drift,
)
from decision_engine.benchmark.scorecard import (
    generate_json_report,
    generate_markdown_scorecard,
)


class BenchmarkRunner:
    """
    Standardized benchmark runner across all model backbones and deployment targets.
    """

    def __init__(self, model_name: str = "tamev_model", batch_size: int = 32):
        self.model_name = model_name
        #: Rows per forward in the batched passes. 0/None forces the old one-row-at-a-time path,
        #: which the equivalence check uses as the reference.
        self.batch_size = batch_size

    def evaluate_local_model(
        self,
        model: Any,
        test_examples: list[dict[str, Any]],
        tokenizer: Any = None,
        device: str = "cpu",
        num_latency_runs: int = 50,
        warmup_runs: int = 5,
    ) -> dict[str, Any]:
        """
        Evaluates a PyTorch BaseDecisionModel locally.
        Each example is a dict:
        {
            "context": str,
            "options": List[str],
            "target": int (ground truth index),
            "task": Optional[str]
        }
        """
        model.eval()
        if hasattr(model, "to"):
            model.to(device)

        if (
            tokenizer is not None
            and hasattr(tokenizer, "pad_token")
            and tokenizer.pad_token is None
        ):
            if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"

        n_items = len(test_examples)
        clean_probs: list[np.ndarray] = [None] * n_items
        perm_aligned_probs: list[np.ndarray] = [None] * n_items
        targets: list[int] = [ex.get("target", 0) for ex in test_examples]
        latencies_ms: list[float] = []
        task_data: dict[str, dict[str, Any]] = {}

        # 1. Warmup
        if test_examples:
            first_ex = test_examples[0]
            for _ in range(warmup_runs):
                _ = self._infer_single(
                    model, first_ex["context"], first_ex["options"], tokenizer, device
                )

        # 2. Permutation draws, in index order. The old per-item loop drew them one per example in
        #    this order, and drift/flip are defined against those draws, so the draws are kept
        #    identical -- batched execution must not change a single number. `random.shuffle` is the
        #    seeded global stream; the latency sample below deliberately uses its own Random so it
        #    cannot perturb this stream.
        perms: list[list[int]] = []
        for ex in test_examples:
            perm = list(range(len(ex["options"])))
            random.shuffle(perm)
            perms.append(perm)

        # 3. Evaluation, batched inside strict same-K groups.
        #    A batch must share K or the [B, K, L] option tensor does not exist, so rows are grouped
        #    by K explicitly instead of being handed to a training sampler whose grouping contract is
        #    about padding cost, not about tensor shape. `batch_size=None` forces the old per-item
        #    path (used by the equivalence check).
        batch = self.batch_size or 1
        if hasattr(model, "predict_proba"):
            batch = 1  # that helper is single-example by contract
        by_k: dict[int, list[int]] = {}
        for i, ex in enumerate(test_examples):
            by_k.setdefault(len(ex["options"]), []).append(i)
        idx_batches = [
            group[s : s + batch]
            for k in sorted(by_k)
            for group in (by_k[k],)
            for s in range(0, len(group), batch)
        ]

        # Two forward passes per row; one monotone counter makes the phase legible from the log.
        total_work = 2 * n_items
        progress_every = max(1, (total_work + 199) // 200)
        done = 0
        t_loop = time.perf_counter()

        def _tick() -> None:
            nonlocal done
            done += 1
            if done % progress_every == 0 or done == total_work:
                elapsed = time.perf_counter() - t_loop
                rate = done / elapsed if elapsed > 0 else 0.0
                eta_min = (total_work - done) / rate / 60.0 if rate else 0.0
                print(
                    f"[bench] Step {done}/{total_work} | {rate:.1f} item/s | "
                    f"elapsed {elapsed:.0f}s | eta {eta_min:.1f} min | device {device}",
                    flush=True,
                )

        for group in idx_batches:
            contexts = [test_examples[i]["context"] for i in group]
            options = [test_examples[i]["options"] for i in group]
            if batch == 1:
                out = [self._infer_single(model, contexts[0], options[0], tokenizer, device)]
            else:
                out = self._infer_batch(model, contexts, options, tokenizer, device)
            for j, i in enumerate(group):
                clean_probs[i] = out[j]
                _tick()

        for group in idx_batches:
            contexts = [test_examples[i]["context"] for i in group]
            options = [[test_examples[i]["options"][t] for t in perms[i]] for i in group]
            if batch == 1:
                out = [self._infer_single(model, contexts[0], options[0], tokenizer, device)]
            else:
                out = self._infer_batch(model, contexts, options, tokenizer, device)
            for j, i in enumerate(group):
                # out[j][t] is the probability of options[perms[i][t]], so it belongs back at that index
                aligned = np.zeros(len(perms[i]), dtype=np.float32)
                for t, orig_i in enumerate(perms[i]):
                    aligned[orig_i] = out[j][t]
                perm_aligned_probs[i] = aligned
                _tick()

        # 4. Latency, sampled and strictly one example at a time: p50/p95/QPS are single-stream
        #    numbers and batching them would measure something else. The sample is capped at
        #    `num_latency_runs` and spread over the whole gate with a private RNG.
        if n_items:
            sample_rng = random.Random(1234)
            n_lat = min(num_latency_runs, n_items)
            for i in sorted(sample_rng.sample(range(n_items), n_lat)):
                ex = test_examples[i]
                t_lat = time.perf_counter()
                _ = self._infer_single(model, ex["context"], ex["options"], tokenizer, device)
                latencies_ms.append((time.perf_counter() - t_lat) * 1000.0)

        for i, ex in enumerate(test_examples):
            task = ex.get("task", "general")
            if task not in task_data:
                task_data[task] = {"probs": [], "targets": []}
            task_data[task]["probs"].append(clean_probs[i])
            task_data[task]["targets"].append(targets[i])

        # 3. Compute Metrics
        clean_metrics = compute_classification_and_calibration(clean_probs, targets)
        mean_drift, flip_rate = compute_permutation_drift(clean_probs, perm_aligned_probs)
        perm_metrics = {
            "mean_max_drift": round(mean_drift, 8),
            "flip_rate": round(flip_rate, 4),
        }
        latency_metrics = compute_latency_percentiles(latencies_ms)

        # Task breakdown
        task_breakdown = {}
        for t_name, t_val in task_data.items():
            t_res = compute_classification_and_calibration(t_val["probs"], t_val["targets"])
            task_breakdown[t_name] = {
                "top1_accuracy": t_res["top1_accuracy"],
                "top3_accuracy": t_res.get("top3_accuracy", 0.0),
                "ece": t_res["ece"],
                "n": t_res["n"],
            }

        return {
            "model_name": self.model_name,
            "clean_metrics": clean_metrics,
            "permutation_metrics": perm_metrics,
            "latency_metrics": latency_metrics,
            "task_breakdown": task_breakdown,
        }

    def _infer_batch(
        self,
        model: Any,
        contexts: list[str],
        options_list: list[list[str]],
        tokenizer: Any,
        device: str,
    ) -> list[np.ndarray]:
        """Clean probabilities for a batch of examples that all share K.

        Same tokenisation contract as `_infer_single`; the only difference is that the options are
        padded to the batch's longest option rather than to each row's own longest, so a probability
        can move by ~1e-7. Padding raises the *cost*, never the semantics -- but it is exactly why the
        equivalence check exists before any published number uses this path.
        """
        k = len(options_list[0])
        if any(len(o) != k for o in options_list):
            raise ValueError("_infer_batch needs a uniform option count across the batch")
        ctx_inputs = tokenizer(
            contexts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        opt_inputs = tokenizer(
            [o for opts in options_list for o in opts],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        b = len(contexts)
        l_opt = opt_inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model(
                context_input_ids=ctx_inputs["input_ids"].to(device),
                context_attention_mask=ctx_inputs["attention_mask"].to(device),
                option_input_ids=opt_inputs["input_ids"].view(b, k, l_opt).to(device),
                option_attention_mask=opt_inputs["attention_mask"].view(b, k, l_opt).to(device),
            )
            probs = torch.softmax(outputs["logits"], dim=-1).float().cpu().numpy()
        return [probs[i] for i in range(b)]

    def _infer_single(
        self,
        model: Any,
        context: str,
        options: list[str],
        tokenizer: Any,
        device: str,
    ) -> np.ndarray:
        """Runs a single example through model and returns probability array."""
        if hasattr(model, "predict_proba"):
            # Model has built-in prediction helper
            return model.predict_proba(context, options, tokenizer, device)

        # Fallback to direct tokenization and forward
        if tokenizer is None:
            raise ValueError("Tokenizer required for local PyTorch evaluation")

        k = len(options)
        ctx_inputs = tokenizer(
            [context],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        opt_inputs = tokenizer(
            options,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        ctx_ids = ctx_inputs["input_ids"].to(device)
        ctx_mask = ctx_inputs["attention_mask"].to(device)
        opt_ids = opt_inputs["input_ids"].unsqueeze(0).to(device)
        opt_mask = opt_inputs["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(
                context_input_ids=ctx_ids,
                context_attention_mask=ctx_mask,
                option_input_ids=opt_ids,
                option_attention_mask=opt_mask,
            )
            logits = outputs["logits"][0, :k]
            return torch.softmax(logits, dim=-1).float().cpu().numpy()

    def save_reports(
        self,
        report_data: dict[str, Any],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Saves report.json and scorecard.md into output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_file = generate_json_report(report_data, out / "report.json")
        md_file = generate_markdown_scorecard(report_data, out / "scorecard.md")
        return {"json": json_file, "scorecard": md_file}
