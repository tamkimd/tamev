# decision_engine/export/multi_format_exporter.py
"""
Multi-Format & Multi-Size Export Engine for TAMEV.
Applies Strategy Pattern & Template Method Pattern via `ExportStrategyRegistry`.
Exports models to:
- PyTorch (FP32, BF16, Dynamic INT8, Apple Silicon MPS)
- ONNX (FP32, Dynamic INT8)
- Apple Ecosystem (CoreML .mlpackage on ANE, Apple MLX)
- Embedded C++ (TorchScript .torchscript)
Generates comprehensive deployment manifest with checksums, parameter counts, and SLAs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.nn as nn

from decision_engine.export.strategies import (
    ExportContext,
    ExportStrategyRegistry,
    compute_file_sha256,
)

__all__ = ["MultiFormatExporter", "compute_file_sha256"]


class MultiFormatExporter:
    """
    Exports any BaseDecisionModel to multiple precision formats and hardware runtimes.
    Delegates concrete compilation and serialization to registered export strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "tamev_model",
        size_class: str = "micro",
        tokenizer: Any = None,
        *,
        dealias_int8_weights: bool = True,
    ):
        self.model = model
        self.model_name = model_name
        self.size_class = size_class
        self.tokenizer = tokenizer
        self.dealias_int8_weights = dealias_int8_weights

        # Calculate parameter counts
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _get_context(self, output_dir: str | Path) -> ExportContext:
        return ExportContext(
            model=self.model,
            model_name=self.model_name,
            size_class=self.size_class,
            output_dir=Path(output_dir),
            tokenizer=self.tokenizer,
            dealias_int8_weights=self.dealias_int8_weights,
        )

    def export_pytorch_fp32(self, output_path: str | Path) -> dict[str, Any]:
        """Saves PyTorch state dict in float32."""
        dest = Path(output_path)
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("pytorch_fp32").execute(ctx, destination=dest)

    def export_pytorch_bf16(self, output_path: str | Path) -> dict[str, Any]:
        """Saves PyTorch state dict converted to bfloat16."""
        dest = Path(output_path)
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("pytorch_bf16").execute(ctx, destination=dest)

    def export_pytorch_int8(self, output_path: str | Path) -> dict[str, Any]:
        """Applies dynamic INT8 quantization to linear layers and saves state dict."""
        dest = Path(output_path)
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("pytorch_int8").execute(ctx, destination=dest)

    def export_pytorch_mps(self, output_path: str | Path) -> dict[str, Any]:
        """Exports model state dict optimized for Apple Silicon MPS (Metal Performance Shaders)."""
        dest = Path(output_path)
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("pytorch_mps").execute(ctx, destination=dest)

    def export_onnx_fp32(self, output_path: str | Path, opset_version: int = 17) -> dict[str, Any]:
        """Exports model to standard ONNX FP32 with dynamic axes."""
        dest = Path(output_path)
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("onnx_fp32").execute(ctx, destination=dest)

    def export_onnx_int8(
        self, input_onnx_path: str | Path, output_path: str | Path
    ) -> dict[str, Any]:
        """Quantizes an existing ONNX model dynamically using onnxruntime."""
        dest = Path(output_path)
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("onnx_int8").execute(ctx, destination=dest)

    def export_coreml(
        self, output_path: str | Path | None = None, precision: str = "fp16"
    ) -> dict[str, Any]:
        """Exports model to Apple CoreML (.mlpackage) format."""
        dest = (
            Path(output_path)
            if output_path is not None
            else Path("exported_models") / f"{self.model_name}.mlpackage"
        )
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("coreml").execute(ctx, destination=dest)

    def export_mlx(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Exports model to Apple MLX format (weights.npz / weights.safetensors + tamev_mlx_config.json)."""
        dest = (
            Path(output_dir)
            if output_dir is not None
            else Path("exported_models") / f"{self.model_name}_mlx"
        )
        ctx = self._get_context(dest)
        return ExportStrategyRegistry.get("mlx").execute(ctx, destination=dest)

    def export_torchscript(self, output_path: str | Path) -> dict[str, Any]:
        """Exports model to TorchScript (.torchscript) for C++ (libtorch) and embedded runtimes."""
        dest = Path(output_path)
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("torchscript").execute(ctx, destination=dest)

    def export_gguf(
        self, output_path: str | Path | None = None, precision: str = "fp16"
    ) -> dict[str, Any]:
        """Exports model to GGUF format (.gguf) for llama.cpp, Ollama, and GGML runtimes."""
        dest = (
            Path(output_path)
            if output_path is not None
            else Path("exported_models") / f"{self.model_name}_{precision}.gguf"
        )
        ctx = self._get_context(dest.parent)
        return ExportStrategyRegistry.get("gguf").execute(ctx, destination=dest)

    def export_all(
        self, output_dir: str | Path, formats: list[str] | str | None = None
    ) -> dict[str, Any]:
        """
        Executes exports for specified formats and generates deployment_manifest.json.
        Applies Strategy Pattern: dynamically resolves strategies from `ExportStrategyRegistry`.
        Supported formats: ['pytorch_fp32', 'pytorch_bf16', 'pytorch_int8', 'pytorch_mps',
                            'onnx_fp32', 'onnx_int8', 'coreml', 'mlx', 'torchscript', 'gguf']
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ctx = self._get_context(out_dir)
        strategies = ExportStrategyRegistry.resolve_formats(formats)

        artifacts: dict[str, Any] = {}
        for strategy in strategies:
            artifacts[strategy.format_name] = strategy.execute(ctx)

        # In-context causal tiers read a packed, order-sensitive sequence; only the
        # encoder path scores options independently, so it is invariant by construction.
        model_type = str(getattr(self.model, "model_type", "") or "")
        is_causal = model_type.startswith(("incontext", "causal", "kev"))

        manifest = {
            "model_name": self.model_name,
            "size_class": self.size_class,
            "parameters": {
                # Counts the model instance this export run loaded, which for a head-only
                # adapter tier includes the (frozen) upstream backbone. The released
                # artifact is much smaller - see the model card for released parameters.
                "total_in_memory": self.total_params,
                "trainable_in_memory": self.trainable_params,
                "note": (
                    "Parameter counts of the in-memory model used for this export, not of the released files. "
                    "Head-only tiers load their upstream backbone here, so this is larger than what ships."
                ),
            },
            "permutation_invariance": (
                "order_sensitive: the packed in-context sequence is not permutation-invariant"
                if is_causal
                else "exact: each option is scored independently, so results are permutation-invariant by construction"
            ),
            # Design targets from the export config - never measured here. Measured p50/p95
            # latencies per tier are published in the Hugging Face model cards.
            "latency_targets_ms": {
                "nano": {"p50_ms": 2.5, "p95_ms": 5.0},
                "micro": {"p50_ms": 4.5, "p95_ms": 8.0},
                "small": {"p50_ms": 12.0, "p95_ms": 20.0},
                "small_modernbert": {"p50_ms": 12.0, "p95_ms": 20.0},
                "medium": {"p50_ms": 25.0, "p95_ms": 45.0},
                "medium_qwen35_08b": {"p50_ms": 25.0, "p95_ms": 45.0},
                "large": {"p50_ms": 45.0, "p95_ms": 90.0},
                "large_qwen35_4b": {"p50_ms": 45.0, "p95_ms": 90.0},
            }.get(self.size_class, {"p50_ms": 5.0, "p95_ms": 10.0}),
            "latency_targets_note": (
                "Unvalidated design targets from the export config, not measured benchmarks. Measured p50 on the "
                "published artifacts is roughly 2x-13x higher (see the model card)."
            ),
            "artifacts": artifacts,
        }

        manifest_path = out_dir / "deployment_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
