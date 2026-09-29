"""
Unified Kev-like CLI interface for TAMEV.
Provides rich-styled commands for fine-tuning, serving, benchmarking, exporting, and auditing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from decision_engine.utils.console import (
    create_table,
    get_console,
    print_banner,
    print_error,
    print_info,
    print_metrics_summary,
    print_success,
    print_warning,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tamev",
        description="⚡️ TAMEV: Real-Time Edge AI Decision Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tamev finetune --data dataset.jsonl --tier micro --epochs 3
  tamev finetune --data dataset.jsonl --plan-size
  tamev serve --port 8008 --device mps
  tamev benchmark --suite data/processed/test.jsonl --out runs/benchmarks
  tamev export --checkpoint runs/model_best.pt --out models/exported
  tamev audit --config configs/default.yaml
        """,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    p_finetune = subparsers.add_parser(
        "finetune",
        help="1-command fine-tuning pipeline & dataset size planner",
    )
    p_finetune.add_argument(
        "--data", type=str, required=False, help="Path to decision dataset (.jsonl or .json)"
    )
    p_finetune.add_argument(
        "--val-data", type=str, default=None, help="Optional separate validation dataset"
    )
    p_finetune.add_argument(
        "--plan-size",
        action="store_true",
        help="Analyze dataset and plan required volume without training",
    )
    p_finetune.add_argument(
        "--tier",
        type=str,
        default="nano",
        help="Model tier (nano, micro, small, medium, large)",
    )
    p_finetune.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p_finetune.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    p_finetune.add_argument("--lr", type=float, default=None, help="Learning rate")
    p_finetune.add_argument(
        "--device", type=str, default="auto", help="Compute device (auto, cpu, mps, cuda)"
    )
    p_finetune.add_argument(
        "--output-dir",
        type=str,
        default="runs/finetuned_model",
        help="Directory to save model and exports",
    )
    p_finetune.add_argument(
        "--export-formats",
        "--formats",
        dest="export_formats",
        type=str,
        default=None,
        help="Formats to export after training (comma-separated: onnx_int8,coreml,mps,mlx,torchscript or 'all')",
    )

    p_serve = subparsers.add_parser(
        "serve",
        help="Starts TypeSafe-compatible /v1/systemone high-throughput decision server",
    )
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    p_serve.add_argument("--port", type=int, default=8008, help="Port to listen on")
    p_serve.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model weights checkpoint"
    )
    p_serve.add_argument(
        "--device", type=str, default=None, help="Hardware device (cpu, mps, cuda)"
    )
    p_serve.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Root model type (encoder, causal, bi_encoder, tinybert)",
    )
    p_serve.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="HuggingFace model backbone repository or local path",
    )
    p_serve.add_argument(
        "--config", type=str, default=None, help="Path to Tamev YAML configuration file"
    )

    p_benchmark = subparsers.add_parser(
        "benchmark",
        help="Runs latency, accuracy, ECE calibration, and exact permutation invariance benchmarks",
    )
    p_benchmark.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model display name, tier, or checkpoint path to benchmark",
    )
    p_benchmark.add_argument(
        "--suite",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to evaluation benchmark dataset (.jsonl)",
    )
    p_benchmark.add_argument(
        "--out",
        type=str,
        default="runs/benchmarks",
        help="Directory to save report.json and scorecard.md",
    )
    p_benchmark.add_argument("--config", type=str, default=None, help="Path to Tamev YAML config")
    p_benchmark.add_argument(
        "--device", type=str, default="cpu", help="Device for profiling (cpu, mps, cuda)"
    )
    p_benchmark.add_argument(
        "--limit", type=int, default=0, help="Maximum samples to evaluate (0 for all)"
    )

    p_export = subparsers.add_parser(
        "export",
        help="Exports trained models to PyTorch, ONNX, Apple CoreML, Apple MLX, and GGUF",
    )
    p_export.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face model repository ID or tier name (e.g. Tamkimd/tamev-nano-tinybert)",
    )
    p_export.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint (.pt), Safetensors (.safetensors), or model directory",
    )
    p_export.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML (e.g. configs/nano.yaml)",
    )
    p_export.add_argument(
        "--tier",
        type=str,
        default="nano",
        help="Architecture tier (nano, micro, small, medium, large)",
    )
    p_export.add_argument(
        "--out",
        "--output-dir",
        dest="out",
        type=str,
        default="models/exported",
        help="Output directory for exported artifacts",
    )
    p_export.add_argument(
        "--formats",
        type=str,
        default="all",
        help="Formats to export (comma-separated: pytorch_fp32,pytorch_bf16,pytorch_int8,pytorch_mps,onnx_fp32,onnx_int8,coreml,mlx,torchscript,gguf or 'all')",
    )
    p_export.add_argument(
        "--device", type=str, default="cpu", help="Device for model loading (cpu, mps, cuda)"
    )
    p_export.add_argument(
        "--dealias-int8-weights",
        action="store_true",
        help=(
            "Opt-in: strip Identity weight aliases before INT8 quantization so ORT can quantize "
            "every encoder MatMul. Nano 31.5 MB -> 14.2 MB, at the cost of more INT8 drift."
        ),
    )

    p_audit = subparsers.add_parser(
        "audit",
        help="Runs production SLA verification (exact permutation equivariance, sub-10ms latency, games)",
    )
    p_audit.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to audit config YAML"
    )

    return parser


def run_finetune_command(args: argparse.Namespace) -> int:
    from decision_engine.training.finetune import finetune, plan_dataset_size

    if args.plan_size:
        if not args.data:
            print_error("--data path is required for --plan-size")
            return 1
        plan_dataset_size(args.data)
        return 0

    if not args.data:
        print_error("--data path is required to train. Use --plan-size to inspect dataset first.")
        return 1

    finetune(
        data=args.data,
        val_data=args.val_data,
        tier=args.tier,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        export_formats=args.export_formats,
    )
    return 0


def run_serve_command(args: argparse.Namespace) -> int:
    from tamev.serve import run_server

    print_banner(
        "🚀 Starting TAMEV Edge Decision Server",
        subtitle=f"Listening on http://{args.host}:{args.port} | TypeSafe /v1/systemone",
    )
    run_server(
        host=args.host,
        port=args.port,
        checkpoint=args.checkpoint,
        device=args.device,
        model_type=args.model_type,
        backbone=args.backbone,
        config=args.config,
    )
    return 0


def run_benchmark_command(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer

    from decision_engine.benchmark import BenchmarkRunner
    from decision_engine.config.base import TamevConfig
    from decision_engine.config.presets import get_preset_config
    from decision_engine.data.dataset import TamevDataset
    from decision_engine.models.base import ModelFactory

    if args.config:
        cfg = TamevConfig.from_yaml(args.config)
    elif args.model in ("nano", "micro", "small_modernbert", "medium_qwen05b", "large_qwen4b"):
        cfg = get_preset_config(args.model)
    else:
        cfg = TamevConfig()

    ckpt_path = None
    if args.model and Path(args.model).exists():
        ckpt_path = args.model
    else:
        # Only the configured tier's own training output is a safe fallback: other
        # checkpoints belong to different backbones, so a strict load against them
        # fails with a shape mismatch (384 vs 312).
        candidates = [
            Path(cfg.training.output_dir) / "tamev_model_calibrated.pt",
            Path(cfg.training.output_dir) / "tamev_model_best.pt",
        ]
        for c in candidates:
            if c.exists():
                ckpt_path = str(c)
                break

    model_name = args.model or cfg.model.name or "TAMEV-Model"
    print_banner("📊 TAMEV Automated Benchmark Suite", subtitle=f"Model: {model_name}")

    suite_path = Path(args.suite)
    if not suite_path.exists():
        print_error(f"Benchmark suite file not found: {suite_path}")
        return 1

    ds = TamevDataset(str(suite_path))
    items = ds.items if args.limit <= 0 else ds.items[: args.limit]
    test_examples = [
        {
            "context": f"{item.state}\n{item.question}".strip() if item.question else item.state,
            "options": [o.text for o in item.options],
            "target": item.target_index,
            "task": item.task,
        }
        for item in items
    ]

    print_info(f"Prepared {len(test_examples)} evaluation samples from {suite_path}")

    tok = AutoTokenizer.from_pretrained(cfg.model.backbone)
    if hasattr(tok, "pad_token") and tok.pad_token is None:
        if hasattr(tok, "eos_token") and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.pad_token = "[PAD]"

    model = ModelFactory.create(cfg.model)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as err:
            if args.model and Path(args.model).exists():
                print_error(f"Checkpoint {ckpt_path} does not match {cfg.model.backbone}: {err}")
                return 1
            # Auto-discovery may pick up another backbone's checkpoint.
            print_warning(
                f"Auto-discovered checkpoint {ckpt_path} does not match {cfg.model.backbone} "
                f"({err}). Evaluating baseline initialization instead; pass "
                "--model <matching-checkpoint.pt> or --config for the tier you meant."
            )
        else:
            print_info(f"Loaded weights from checkpoint: {ckpt_path}")
    else:
        print_warning("No checkpoint weights loaded (evaluating baseline initialization)")

    runner = BenchmarkRunner(model_name=model_name)
    report = runner.evaluate_local_model(
        model=model,
        test_examples=test_examples,
        tokenizer=tok,
        device=args.device,
    )
    out_paths = runner.save_reports(report, args.out)

    clean = report["clean_metrics"]
    perm = report["permutation_metrics"]
    lat = report["latency_metrics"]

    print_metrics_summary(
        f"🏆 Benchmark Results: {model_name}",
        {
            "Top-1 Accuracy": clean["top1_accuracy"],
            "Top-3 Accuracy": clean["top3_accuracy"],
            "ECE Calibration": clean["ece"],
            "Brier Score": clean["brier_score"],
            "Permutation Drift": perm["mean_max_drift"],
            "Flip Rate": perm["flip_rate"],
            "CPU Latency p50": f"{lat['p50']:.2f} ms",
            "CPU Latency p95": f"{lat['p95']:.2f} ms",
            "Throughput QPS": f"{lat['qps']:.0f} req/s",
        },
    )
    print_success(f"Reports saved to: {out_paths['scorecard']}")
    return 0


def run_export_command(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer

    from decision_engine.config.base import TamevConfig
    from decision_engine.config.presets import get_preset_config
    from decision_engine.export.multi_format_exporter import MultiFormatExporter
    from decision_engine.models.base import ModelFactory

    model_arg = getattr(args, "model", None)
    ckpt_path = args.checkpoint or model_arg
    tier = args.tier

    if model_arg and not args.config:
        m_lower = model_arg.lower()
        if "nano" in m_lower or "tinybert" in m_lower:
            tier = "nano"
        elif "micro" in m_lower or "minilm" in m_lower:
            tier = "micro"
        elif "modernbert" in m_lower or "small" in m_lower:
            tier = "small"
        elif "0.8b" in m_lower or "medium" in m_lower:
            tier = "medium"
        elif "4b" in m_lower or "large" in m_lower:
            tier = "large"

    if args.config:
        cfg = TamevConfig.from_yaml(args.config)
        tier = cfg.model.size_class or tier
    else:
        cfg = get_preset_config(tier)

    if not ckpt_path:
        candidates = [
            Path(f"models/hf/tamev-{tier}-tinybert/model.safetensors"),
            Path(f"models/hf/tamev-{tier}-minilm/model.safetensors"),
            Path(f"models/hf/tamev-{tier}-modernbert/model.safetensors"),
            Path(f"models/hf/tamev-{tier}-qwen3.5-0.8b/model.safetensors"),
            Path(f"models/hf/tamev-{tier}-qwen3.5-4b/model.safetensors"),
            Path(cfg.training.output_dir) / "tamev_model_calibrated.pt",
            Path(cfg.training.output_dir) / "tamev_model_best.pt",
            Path(f"runs/tamev_{tier}_large/tamev_model_calibrated.pt"),
        ]
        if tier in ("medium", "medium_qwen35_08b"):
            candidates.append(Path("runs/tamev_medium_qwen35_08b_large/tamev_model_calibrated.pt"))
        elif tier in ("large", "large_qwen35_4b", "large_qwen4b"):
            candidates.append(Path("runs/tamev_large_qwen4b_large/tamev_model_calibrated.pt"))
        elif tier == "nano":
            candidates.extend(
                [
                    Path("runs/tamev_nano_large/tamev_model_calibrated.pt"),
                    Path("runs/tamev_distill_rl/tamev_tinybert_distill_rl_best.pt"),
                    Path("runs/tamev_super_tiny_tinybert/tamev_tinybert_best.pt"),
                ]
            )
        elif tier == "micro":
            candidates.append(Path("runs/tamev_micro_large/tamev_model_calibrated.pt"))
        elif tier in ("small", "small_modernbert"):
            candidates.extend(
                [
                    Path("runs/tamev_small_modernbert/tamev_model_calibrated.pt"),
                    Path("runs/tamev_small_modernbert_large/tamev_model_calibrated.pt"),
                ]
            )

        for c in candidates:
            if c.exists():
                ckpt_path = str(c)
                break

    print_banner(
        "📦 TAMEV Multi-Format & Edge Runtime Export Engine", subtitle=f"Target Tier: {tier}"
    )

    model = ModelFactory.create(cfg.model)
    if ckpt_path:
        p = Path(ckpt_path)
        if p.is_dir() and (p / "model.safetensors").exists():
            from safetensors.torch import load_file

            state_dict = load_file(p / "model.safetensors")
            model.load_state_dict(state_dict, strict=False)
            print_info(f"Loaded Safetensors directory: {p / 'model.safetensors'}")
        elif p.is_file() and p.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(p)
            model.load_state_dict(state_dict, strict=False)
            print_info(f"Loaded Safetensors weights from: {p}")
        elif p.is_file():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            if "temperature" in ckpt and hasattr(model, "set_temperature"):
                model.set_temperature(float(ckpt["temperature"]))
            print_info(f"Loaded PyTorch checkpoint from: {p}")
        elif "/" in ckpt_path and not p.exists():
            try:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file

                print_info(f"Downloading model.safetensors from Hugging Face: {ckpt_path}...")
                sf_file = hf_hub_download(repo_id=ckpt_path, filename="model.safetensors")
                state_dict = load_file(sf_file)
                model.load_state_dict(state_dict, strict=False)
                print_info(f"Loaded Hugging Face weights from: {ckpt_path}")
            except Exception as e:
                print_warning(f"Could not load Hugging Face repo {ckpt_path}: {e}")
        else:
            print_warning(
                f"Could not resolve checkpoint '{ckpt_path}'; exporting base architecture weights."
            )
    else:
        print_warning("No checkpoint found; exporting base architecture weights.")

    tok = AutoTokenizer.from_pretrained(cfg.model.backbone)

    is_incontext = getattr(cfg.model, "model_type", "") in ("incontext_causal", "incontext", "kev")

    if args.formats == "all":
        if is_incontext:
            formats = [
                "pytorch_fp32",
                "pytorch_bf16",
                "pytorch_int8",
                "pytorch_mps",
                "mlx",
            ]
        else:
            formats = [
                "pytorch_fp32",
                "pytorch_bf16",
                "pytorch_int8",
                "onnx_fp32",
                "onnx_int8",
            ]
    else:
        formats = [
            ("pytorch_mps" if f.strip() == "mps" else f.strip())
            for f in args.formats.split(",")
            if f.strip()
        ]

    exporter = MultiFormatExporter(
        model=model,
        model_name=f"tamev_{tier}",
        size_class=tier,
        tokenizer=tok,
        dealias_int8_weights=args.dealias_int8_weights,
    )

    out_dir = Path(args.out)
    if out_dir == Path("models/exported") or out_dir.name == "exported":
        out_dir = out_dir / tier
    manifest = exporter.export_all(output_dir=out_dir, formats=formats)

    table = create_table(
        title="Artifact Deployment Manifest",
        headers=["Format", "Artifact File", "Size (MB)", "Status"],
    )
    for fmt, meta in manifest.get("artifacts", {}).items():
        table.add_row(
            fmt,
            meta.get("file", ""),
            str(meta.get("size_mb", "N/A")),
            "[bold green]Ready for Deployment[/]",
        )

    get_console().print(table)
    print_success(f"Deployment manifest written to {out_dir / 'deployment_manifest.json'}")
    return 0


def run_audit_command(args: argparse.Namespace) -> int:
    from scripts.verify_production_audit import run_production_audit

    try:
        run_production_audit(config_path=args.config)
        return 0
    except AssertionError as err:
        print_error(f"Audit verification gate failed: {err}")
        return 1
    except Exception as e:
        print_error(f"Unexpected error during audit: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print_banner(
            "⚡️ TAMEV: Real-Time Edge AI Decision Engine",
            subtitle="TypeSafe SDK Compatible | Sub-10ms Latency | Exact 0.0000 Drift",
        )
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.subcommand == "finetune":
        return run_finetune_command(args)
    if args.subcommand == "serve":
        return run_serve_command(args)
    if args.subcommand == "benchmark":
        return run_benchmark_command(args)
    if args.subcommand == "export":
        return run_export_command(args)
    if args.subcommand == "audit":
        return run_audit_command(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
