#!/usr/bin/env python3
"""
TAMEV Master Pluggable Training CLI.

Supports seamless plug-and-play swapping of:
- Root Decision Models (Encoder, Causal LM, Bi-Encoder, TinyBERT)
- Teacher Distillation Models (HuggingFace, API/vLLM, Disk-Cached, Mock)
- Hyperparameter configurations and presets (Nano, Edge, Server)
- Arbitrary datasets (JSONL, Kev v7)
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

from decision_engine.config.base import TamevConfig
from decision_engine.data.dataset import TamevDataset
from decision_engine.models.base import ModelFactory
from decision_engine.teacher.base import TeacherFactory
from decision_engine.training.trainer import TamevTrainer


def build_config_from_args(args: argparse.Namespace) -> TamevConfig:
    """Builds and resolves TamevConfig from config file, preset, and CLI flags."""
    if args.config:
        cfg = TamevConfig.from_yaml(args.config)
    elif args.preset:
        cfg = TamevConfig.from_preset(args.preset)
    else:
        cfg = TamevConfig()

    # CLI overrides for Model
    if args.model_type:
        cfg.model.model_type = args.model_type
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.projection_dim:
        cfg.model.projection_dim = args.projection_dim
    if args.temperature:
        cfg.model.temperature = args.temperature

    # CLI overrides for Teacher
    if args.teacher_provider:
        cfg.teacher.provider = args.teacher_provider
        if args.teacher_provider == "agent":
            from decision_engine.teacher.agent_teacher import resolve_agent_identity

            agent_model = resolve_agent_identity(args.teacher_model)
            cfg.teacher.name = agent_model
            cfg.teacher.model_path = agent_model
    if args.teacher_model and (not args.teacher_provider or args.teacher_provider != "agent"):
        cfg.teacher.model_path = args.teacher_model
        cfg.teacher.name = (
            Path(args.teacher_model).name if "/" in args.teacher_model else args.teacher_model
        )
    if args.teacher_temp:
        cfg.teacher.temperature = args.teacher_temp
    if args.teacher_device:
        cfg.teacher.device = args.teacher_device
    if args.api_key:
        cfg.teacher.api_key = args.api_key

    # CLI overrides for Dataset
    if args.train_path:
        cfg.dataset.train_path = args.train_path
    if args.val_path:
        cfg.dataset.val_path = args.val_path
    if args.test_path:
        cfg.dataset.test_path = args.test_path

    # CLI overrides for Training
    if args.epochs:
        cfg.training.epochs = args.epochs
    if getattr(args, "resume", False):
        cfg.training.resume = True
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.device:
        cfg.training.device = args.device
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.kd_weight is not None:
        cfg.training.kd_weight = args.kd_weight

    return cfg


def main():
    parser = argparse.ArgumentParser(description="TAMEV Universal Pluggable Training Engine")

    # Config source
    parser.add_argument("--config", type=str, default=None, help="Path to Tamev YAML config")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["nano", "edge", "server"],
        help="Configuration preset",
    )

    # Model parameters
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Root model type (encoder, causal, bi_encoder, tinybert)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone HuggingFace repo ID or local checkpoint path",
    )
    parser.add_argument(
        "--projection-dim", type=int, default=None, help="Pointer head projection dimension"
    )
    parser.add_argument("--temperature", type=float, default=None, help="Model initial temperature")

    # Teacher parameters
    parser.add_argument(
        "--teacher-provider",
        type=str,
        default=None,
        choices=["mock", "agent", "huggingface", "api", "cached", "vllm"],
        help="Teacher distillation provider",
    )
    parser.add_argument(
        "--teacher-model", type=str, default=None, help="Teacher model repo or path"
    )
    parser.add_argument(
        "--teacher-temp", type=float, default=None, help="Teacher softening temperature"
    )
    parser.add_argument("--teacher-device", type=str, default=None, help="Teacher compute device")
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key for remote teacher endpoints"
    )

    # Dataset parameters
    parser.add_argument("--train-path", type=str, default=None, help="Path to training JSONL")
    parser.add_argument("--val-path", type=str, default=None, help="Path to validation JSONL")
    parser.add_argument("--test-path", type=str, default=None, help="Path to test JSONL")
    parser.add_argument(
        "--annotate-if-missing",
        action="store_true",
        help="Annotate training samples on the fly if teacher_probs missing",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from <output_dir>/tamev_model_resume.pt if present (preemption recovery)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="AdamW learning rate")
    parser.add_argument(
        "--device", type=str, default=None, help="Training device (auto, mps, cuda, cpu)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save run checkpoints and metrics"
    )
    parser.add_argument(
        "--kd-weight", type=float, default=None, help="Knowledge distillation loss weight"
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Path to initial model checkpoint for warm-starting or distillation",
    )

    args = parser.parse_args()

    cfg = build_config_from_args(args)

    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from decision_engine.utils.console import print_banner, print_kv_list

    init_info = [
        ("Root Model", f"{cfg.model.model_type} ({cfg.model.backbone})"),
        ("Teacher", f"{cfg.teacher.provider} ({cfg.teacher.model_path})"),
        ("Train Path", str(cfg.dataset.train_path)),
        ("Val Path", str(cfg.dataset.val_path)),
        (
            "Epochs / Batch / LR",
            f"{cfg.training.epochs} | {cfg.training.batch_size} | {cfg.training.learning_rate}",
        ),
    ]
    if args.init_checkpoint:
        init_info.append(("Init Checkpoint", str(args.init_checkpoint)))
    init_info.append(("Output Dir", str(out_dir)))

    print_banner("🚀 TAMEV Pluggable Decision Training Pipeline")
    print_kv_list(init_info)

    print(f"[INIT] Loading tokenizer for backbone: {cfg.model.backbone}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone)
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "[PAD]"

    print("[DATA] Loading datasets...")
    train_path = Path(cfg.dataset.train_path)
    if not train_path.exists():
        fallback_train = Path("data/processed/train_merged.jsonl")
        if not fallback_train.exists():
            fallback_train = Path("data/processed/train.jsonl")
        if fallback_train.exists():
            print(f"  Note: {train_path} not found, falling back to {fallback_train}")
            train_path = fallback_train
        else:
            raise FileNotFoundError(f"Training dataset not found at {train_path}")

    train_ds = TamevDataset(train_path)
    val_ds = TamevDataset(cfg.dataset.val_path)
    test_ds = TamevDataset(cfg.dataset.test_path) if Path(cfg.dataset.test_path).exists() else None
    print(
        f"[DATA] Loaded {len(train_ds)} train, {len(val_ds)} val, {len(test_ds) if test_ds else 0} test samples."
    )

    if args.annotate_if_missing:
        print("[TEACHER] Checking for missing teacher probabilities...")
        teacher = TeacherFactory.create(cfg.teacher)
        missing_count = sum(1 for item in train_ds.items if not item.teacher_probs)
        if missing_count > 0:
            print(f"  Annotating {missing_count} samples with {teacher.model_name}...")
            for item in train_ds.items:
                if not item.teacher_probs:
                    ctx = f"{item.state}\n{item.question}".strip() if item.question else item.state
                    opt_texts = [o.text for o in item.options]
                    probs = teacher.get_soft_targets(
                        ctx, opt_texts, temperature=cfg.teacher.temperature
                    )
                    item.teacher_probs = {
                        item.options[i].id: round(float(probs[i]), 4)
                        for i in range(len(item.options))
                    }
            print("  Annotation complete.")

    print(f"[MODEL] Building {cfg.model.model_type} decision model...")
    model = ModelFactory.create(cfg.model)
    ref_model = None

    if args.init_checkpoint:
        ckpt_p = Path(args.init_checkpoint)
        if ckpt_p.exists():
            print(f"[INIT] Loading initial weights from {ckpt_p}...")
            ckpt_data = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            model.load_state_dict(state_dict)

            print("[INIT] Creating frozen reference policy anchor...")
            ref_model = ModelFactory.create(cfg.model)
            ref_model.load_state_dict(state_dict)
        else:
            print(f"[WARN] Initial checkpoint {ckpt_p} not found. Training from scratch.")

    trainer = TamevTrainer(
        model=model,
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        tokenizer=tokenizer,
        ref_model=ref_model,
    )

    t_start = time.time()
    results = trainer.train()
    elapsed = time.time() - t_start

    cfg_yaml_path = out_dir / "resolved_config.yaml"
    cfg.to_yaml(cfg_yaml_path)

    summary = {
        "model_type": cfg.model.model_type,
        "backbone": cfg.model.backbone,
        "teacher_provider": cfg.teacher.provider,
        "teacher_model": cfg.teacher.model_path,
        "calibrated_temperature": results.get("calibrated_temperature"),
        "best_checkpoint": results.get("best_checkpoint"),
        "calibrated_checkpoint": results.get("calibrated_checkpoint"),
        "final_val_metrics": results.get("final_val_metrics"),
        "training_time_sec": elapsed,
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_banner("🏆 TRAINING AND CALIBRATION COMPLETE!")
    print_kv_list(
        [
            ("Val Top-1 Accuracy", f"{results['final_val_metrics']['top1_accuracy'] * 100:.2f}%"),
            ("Val ECE", f"{results['final_val_metrics']['ece']:.4f}"),
            ("Calibrated Temp T*", f"{results.get('calibrated_temperature'):.4f}"),
            ("Calibrated Model", str(results.get("calibrated_checkpoint"))),
            ("Configuration saved", str(cfg_yaml_path)),
        ]
    )


if __name__ == "__main__":
    main()
