# decision_engine/training/trainer.py
"""
Modular Trainer for TAMEV Decision Engine.
Coordinates data loading, multi-objective optimization, validation,
and post-hoc temperature calibration.
"""

import contextlib
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from decision_engine.benchmark.metrics import compute_classification_and_calibration
from decision_engine.config.base import TamevConfig
from decision_engine.data.collator import TamevCollator
from decision_engine.data.dataset import KGroupedBatchSampler, TamevDataset
from decision_engine.training.calibrator import TemperatureCalibrator
from decision_engine.training.losses import CompositeLoss


def cuda_amp_dtype(capability_major: int) -> torch.dtype:
    """AMP dtype for a CUDA device of the given compute-capability major version.

    bf16 kernels exist only on sm_80+. Below that, ``torch.cuda.is_bf16_supported()`` returns
    True *by emulation* and autocast then emulates bf16 at a fraction of fp16 throughput
    (measured 3.7% of T4 fp16 peak), so pre-Ampere takes the fp16 + GradScaler path instead.
    """
    return torch.bfloat16 if capability_major >= 8 else torch.float16


def compute_composite_score(
    top1: float, top3: float, w_top1: float = 0.7, w_top3: float = 0.3
) -> float:
    """
    Computes weighted multi-metric composite score for model selection and checkpointing.
    Defaults to 0.7 Top-1 + 0.3 Top-3 as specified for production decision engines.
    """
    return (w_top1 * top1) + (w_top3 * top3)


def aggregate_domain_metrics(
    all_probs: list[Any],
    all_targets: list[int],
    all_tasks: list[str],
) -> dict[str, dict[str, float]]:
    """
    Computes per-domain / per-task metrics (Top-1, Top-3, ECE) to detect domain distribution shifts.
    """
    task_groups: dict[str, dict[str, list[Any]]] = {}
    for p, t, task in zip(all_probs, all_targets, all_tasks):
        if task not in task_groups:
            task_groups[task] = {"probs": [], "targets": []}
        task_groups[task]["probs"].append(p)
        task_groups[task]["targets"].append(t)

    breakdown: dict[str, dict[str, float]] = {}
    for task, data in task_groups.items():
        if data["targets"]:
            breakdown[task] = compute_classification_and_calibration(data["probs"], data["targets"])
    return breakdown


class TamevTrainer:
    """
    Standardized Trainer for all TAMEV model architectures.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: TamevConfig,
        train_dataset: TamevDataset,
        val_dataset: TamevDataset,
        tokenizer: Any,
        test_dataset: TamevDataset | None = None,
        ref_model: torch.nn.Module | None = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.ref_model = ref_model

        # Determine compute device
        if config.training.device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.training.device)

        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        # Mixed precision (AMP) setup for macOS Apple Silicon (MPS) and CUDA
        self.use_amp = getattr(config.training, "amp", True)
        self.grad_accum_steps = max(1, getattr(config.training, "gradient_accumulation_steps", 1))

        if self.device.type == "mps":
            self.autocast_device = "mps"
            self.amp_dtype = torch.bfloat16
        elif self.device.type == "cuda":
            self.autocast_device = "cuda"
            self.amp_dtype = cuda_amp_dtype(torch.cuda.get_device_capability()[0])
        else:
            self.autocast_device = "cpu"
            self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        # fp16 grad scaling is required on the fp16 path and must not run for bf16/fp32.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self.use_amp and self.amp_dtype == torch.float16,
        )

        # Loss function
        self.loss_fn = CompositeLoss(
            ce_weight=1.0,
            brier_weight=self.config.training.brier_weight,
            margin_weight=self.config.training.margin_weight,
            kd_weight=self.config.training.kd_weight,
            margin_gamma=self.config.training.margin_gamma,
            kd_temp=self.config.training.kd_temperature,
            ordinal_weight=getattr(self.config.training, "ordinal_weight", 0.0),
            cell_weights=getattr(self.config.training, "cell_weights", None),
            rdrop_weight=getattr(self.config.training, "rdrop_weight", 0.0),
            sce_weight=getattr(self.config.training, "sce_weight", 0.0),
            sce_alpha=getattr(self.config.training, "sce_alpha", 0.1),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Training collator with negative option subsampling capped by dataset.max_options
        self.train_collator = TamevCollator(
            tokenizer=tokenizer,
            max_ctx_len=config.dataset.max_ctx_len,
            max_opt_len=config.dataset.max_opt_len,
            max_options=config.dataset.max_options,
            device=self.device,
            seed=config.training.seed,
            hard_negatives_path=config.dataset.hard_negatives_path,
        )
        # Evaluation collator with full candidate set (or capped via eval_max_options)
        self.eval_collator = TamevCollator(
            tokenizer=tokenizer,
            max_ctx_len=config.dataset.max_ctx_len,
            max_opt_len=config.dataset.max_opt_len,
            max_options=config.dataset.eval_max_options,
            device=self.device,
            seed=config.training.seed,
        )

        # Padding-aware batching: the collator pads every row to the batch's max K, so uniform
        # shuffling wastes most of the forward pass on placeholder options (see the sampler doc).
        self.train_batch_sampler: KGroupedBatchSampler | None = None
        if self.config.training.batch_sampler == "k_grouped":
            items = getattr(self.train_dataset, "items", None)
            if items:
                self.train_batch_sampler = KGroupedBatchSampler(
                    cardinalities=[len(item.options) for item in items],
                    batch_size=self.config.training.batch_size,
                    seed=self.config.training.seed,
                    shuffle=True,
                )
        if self.train_batch_sampler is not None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                collate_fn=self.train_collator,
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                collate_fn=self.train_collator,
            )
        self.val_loader = self._build_eval_loader(self.val_dataset)
        self.test_loader = self._build_eval_loader(self.test_dataset) if self.test_dataset else None

        # Learning rate scheduler
        total_steps = max(1, self.config.training.epochs * len(self.train_loader))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )

        self.output_dir = Path(self.config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_eval_loader(self, dataset: TamevDataset) -> DataLoader:
        """Eval loader grouped by option cardinality.

        ``TamevCollator`` pads every row to the batch ``max_K``, so a mixed-K eval batch spends
        most of its forward pass on placeholder options: measured 3.159M vs 1.055M padded tokens
        (3.0x). Every eval metric is computed per row from ``batch["num_options"]`` and is
        order-independent, so grouping changes batch composition only. ``drop_last`` stays False:
        no row is skipped.
        """
        items = getattr(dataset, "items", None)
        if items:
            return DataLoader(
                dataset,
                batch_sampler=KGroupedBatchSampler(
                    cardinalities=[len(item.options) for item in items],
                    batch_size=self.config.training.batch_size,
                    seed=self.config.training.seed,
                    shuffle=False,
                ),
                collate_fn=self.eval_collator,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=self.eval_collator,
        )

    def _get_autocast_ctx(self):
        """Returns the appropriate autocast context manager for mixed-precision acceleration."""
        if self.use_amp and self.device.type in ("mps", "cuda"):
            return torch.autocast(device_type=self.autocast_device, dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def _extract_state_dict(self) -> dict[str, Any]:
        """Extracts model weights. For models with frozen backbones, extracts only trainable/pointer weights. Uses bfloat16 to conserve 50% disk."""
        if getattr(self.model, "freeze_backbone", False):
            return {
                k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
                if "pointer_head" in k or v.requires_grad
            }
        return {
            k: v.cpu().to(torch.bfloat16) if v.is_floating_point() else v.cpu().clone()
            for k, v in self.model.state_dict().items()
        }

    def _save_resume(self, path: Path, epoch: int, step: int, best_val_score: float) -> None:
        """Write the resume checkpoint atomically: a preemption must not leave a truncated file."""
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "model_state_dict": self._extract_state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "step": step,
                "best_val_score": best_val_score,
            },
            tmp,
        )
        tmp.replace(path)

    def _load_resume(self, path: Path) -> tuple[int, int, float]:
        """Restore model/optimizer/scheduler and return (epoch, step, best_val_score)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return int(ckpt["epoch"]), int(ckpt["step"]), float(ckpt.get("best_val_score") or 0.0)

    def train(self) -> dict[str, Any]:
        """Runs the training and calibration pipeline."""
        epochs = self.config.training.epochs
        best_val_score = float("-inf")
        best_ckpt_path = self.output_dir / "tamev_model_best.pt"
        resume_path = self.output_dir / "tamev_model_resume.pt"
        start_epoch, start_step = 1, 0
        if self.config.training.resume and resume_path.exists():
            start_epoch, start_step, best_val_score = self._load_resume(resume_path)
            print(
                f"▶️  Resumed from {resume_path} at epoch {start_epoch}, step {start_step}",
                flush=True,
            )
        patience = getattr(self.config.training, "early_stopping_patience", 3)
        monitor = getattr(self.config.training, "early_stopping_metric", "val_top1")
        patience_counter = 0

        val_metrics: dict[str, Any] = {}
        val_logits: list[Any] = []
        val_targets: list[Any] = []
        best_val_logits: list[Any] = []
        best_val_targets: list[Any] = []

        print(
            f"🚀 Starting TAMEV Training: {epochs} epochs on {self.device} "
            f"(amp={self.use_amp}, grad_accum={self.grad_accum_steps}, patience={patience}, monitor={monitor})"
        )
        resume_every = int(getattr(self.config.training, "resume_every", 0) or 0)
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            if self.train_batch_sampler is not None:
                self.train_batch_sampler.set_epoch(epoch)
            self.model.train()
            # Kept as a device tensor: nothing reads the running sum until the epoch boundary, so the
            # old per-step `.item()` only forced a device sync on every step for no reader.
            total_loss = torch.zeros((), device=self.device)
            # Mid-epoch resume: the K-grouped sampler is deterministic per (seed, epoch), so the
            # batches already consumed are re-iterated and skipped. Resumes always land on a
            # gradient-accumulation boundary, so the partial accumulation state never matters.
            skip = start_step if epoch == start_epoch else 0
            steps = skip
            step_in_epoch = skip

            self.optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx < skip:
                    continue
                with self._get_autocast_ctx():
                    out = self.model(
                        context_input_ids=batch["ctx_input_ids"],
                        context_attention_mask=batch["ctx_attention_mask"],
                        option_input_ids=batch["opt_input_ids"],
                        option_attention_mask=batch["opt_attention_mask"],
                        num_options=batch["num_options"],
                    )

                    ref_logits = None
                    if self.ref_model is not None:
                        with torch.no_grad():
                            ref_out = self.ref_model(
                                context_input_ids=batch["ctx_input_ids"],
                                context_attention_mask=batch["ctx_attention_mask"],
                                option_input_ids=batch["opt_input_ids"],
                                option_attention_mask=batch["opt_attention_mask"],
                                num_options=batch["num_options"],
                            )
                            ref_logits = ref_out["logits"]

                    # R-Drop: a second pass over the same batch under a fresh dropout mask, so the
                    # KL term below sees two genuine samples of the same distribution. Skipped
                    # entirely (no second forward) when `training.rdrop_weight` is 0, which keeps
                    # every other arm's step cost and numerics untouched.
                    logits2 = None
                    if self.loss_fn.rdrop_weight > 0:
                        out2 = self.model(
                            context_input_ids=batch["ctx_input_ids"],
                            context_attention_mask=batch["ctx_attention_mask"],
                            option_input_ids=batch["opt_input_ids"],
                            option_attention_mask=batch["opt_attention_mask"],
                            num_options=batch["num_options"],
                        )
                        logits2 = out2["logits"]

                    loss_dict = self.loss_fn(
                        logits=out["logits"],
                        targets=batch["labels"],
                        teacher_probs=batch.get("teacher_probs"),
                        num_options=batch["num_options"],
                        has_teacher=batch.get("has_teacher"),
                        ref_logits=ref_logits,
                        ordinal=batch.get("ordinal"),
                        cells=batch.get("cells"),
                        logits2=logits2,
                    )
                    loss = loss_dict["loss"] / self.grad_accum_steps

                # GradScaler is a no-op when disabled (bf16/fp32/MPS/CPU).
                self.scaler.scale(loss).backward()
                step_in_epoch += 1

                if step_in_epoch % self.grad_accum_steps == 0 or step_in_epoch == len(
                    self.train_loader
                ):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if resume_every and steps and steps % resume_every == 0:
                        self._save_resume(resume_path, epoch, steps, best_val_score)

                # Detach so the running sum never holds the autograd graph; no sync until the print.
                total_loss = total_loss + loss_dict["loss"].detach()
                steps += 1
                if steps % 50 == 0 or steps == len(self.train_loader):
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    print(
                        f"  [Epoch {epoch}/{epochs}] Step {steps}/{len(self.train_loader)} | "
                        f"Loss: {loss_dict['loss'].item():.4f} (CE: {loss_dict['ce_loss'].item():.4f})",
                        flush=True,
                    )

            if resume_every:
                self._save_resume(resume_path, epoch + 1, 0, best_val_score)

            # Validation
            val_metrics, val_logits, val_targets = self._evaluate_loader(self.val_loader)
            elapsed = time.time() - t0
            avg_loss = (total_loss / max(steps, 1)).item()

            # Score = 0.7 * Top-1 + 0.3 * Top-3
            composite_score = compute_composite_score(
                top1=val_metrics["top1_accuracy"],
                top3=val_metrics.get("top3_accuracy", 0.0),
            )
            min_delta = getattr(self.config.training, "early_stopping_min_delta", 0.001)

            print(
                f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val Acc: {val_metrics['top1_accuracy'] * 100:.2f}% | "
                f"Val Top-3: {val_metrics.get('top3_accuracy', 0.0) * 100:.2f}% | "
                f"Val Composite: {composite_score * 100:.2f}% | "
                f"Val ECE: {val_metrics['ece']:.4f}",
                flush=True,
            )

            if monitor == "val_top1":
                curr_metric = val_metrics["top1_accuracy"]
            elif monitor == "composite":
                curr_metric = composite_score
            else:
                curr_metric = val_metrics.get(monitor, val_metrics["top1_accuracy"])

            if curr_metric > (best_val_score + min_delta):
                best_val_score = curr_metric
                patience_counter = 0
                best_val_logits = val_logits
                best_val_targets = val_targets
                torch.save(
                    {
                        "model_state_dict": self._extract_state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    best_ckpt_path,
                )
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    print(
                        f"🛑 Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs, best was {best_val_score * 100:.2f}%)"
                    )
                    break

        # Post-hoc Temperature Calibration
        if not best_val_logits and best_ckpt_path.exists():
            best_data = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(best_data["model_state_dict"], strict=False)
            self.model.to(self.device)
            val_metrics, best_val_logits, best_val_targets = self._evaluate_loader(self.val_loader)

        print("🎯 Running post-hoc temperature calibration...")
        calibrator = TemperatureCalibrator()
        calibrated_T = calibrator.calibrate(best_val_logits, best_val_targets)
        print(f"Optimal Temperature T* = {calibrated_T}")

        # Update model temperature
        if hasattr(self.model, "set_temperature"):
            self.model.set_temperature(calibrated_T)

        calibrated_ckpt = self.output_dir / "tamev_model_calibrated.pt"
        torch.save(
            {
                "model_state_dict": self._extract_state_dict(),
                "temperature": calibrated_T,
                "val_metrics": val_metrics,
            },
            calibrated_ckpt,
        )

        if best_ckpt_path.exists() and best_ckpt_path != calibrated_ckpt:
            with contextlib.suppress(OSError):
                best_ckpt_path.unlink()

        if resume_path.exists():
            with contextlib.suppress(OSError):
                resume_path.unlink()

        results = {
            "best_checkpoint": str(calibrated_ckpt),
            "calibrated_checkpoint": str(calibrated_ckpt),
            "calibrated_temperature": calibrated_T,
            "final_val_metrics": val_metrics,
        }

        # Evaluate on Test dataset if provided
        if self.test_loader is not None:
            print("🧪 Evaluating on independent test dataset...")
            test_metrics, _, _ = self._evaluate_loader(self.test_loader)
            results["test_metrics"] = test_metrics
            print(
                f"   Test Top-1 Acc: {test_metrics['top1_accuracy'] * 100:.2f}% | "
                f"Test Top-3: {test_metrics.get('top3_accuracy', 0.0) * 100:.2f}% | "
                f"Test ECE: {test_metrics['ece']:.4f}",
                flush=True,
            )

        return results

    def _evaluate_loader(self, loader: DataLoader):
        if self.device.type == "mps":
            torch.mps.empty_cache()
        self.model.eval()
        all_probs = []
        all_logits = []
        all_targets = []
        all_tasks = []

        with torch.no_grad(), self._get_autocast_ctx():
            for batch in loader:
                out = self.model(
                    context_input_ids=batch["ctx_input_ids"],
                    context_attention_mask=batch["ctx_attention_mask"],
                    option_input_ids=batch["opt_input_ids"],
                    option_attention_mask=batch["opt_attention_mask"],
                    num_options=batch["num_options"],
                )
                logits_cpu = out["logits"].float().cpu()
                probs_cpu = out["probs"].float().cpu()

                for i, k in enumerate(batch["num_options"]):
                    item_p = probs_cpu[i, :k].numpy()
                    item_l = logits_cpu[i, :k].numpy()
                    all_probs.append(item_p)
                    all_logits.append(item_l)

                all_targets.extend(batch["labels"].cpu().numpy().tolist())
                if "tasks" in batch:
                    all_tasks.extend(batch["tasks"])
                else:
                    all_tasks.extend(["general"] * len(batch["labels"]))

        if self.device.type == "mps":
            torch.mps.empty_cache()

        metrics = compute_classification_and_calibration(all_probs, all_targets)
        metrics["domain_breakdown"] = aggregate_domain_metrics(all_probs, all_targets, all_tasks)
        return metrics, all_logits, all_targets
