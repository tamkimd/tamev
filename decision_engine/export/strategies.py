# decision_engine/export/strategies.py
"""
Export Strategy & Registry Pattern for TAMEV Multi-Format Exporters.
Applies:
1. Strategy Pattern: Each format has an isolated, testable export strategy.
2. Template Method Pattern: Common workflow (path resolution, directory creation,
   artifact export, size calculation, SHA-256 computation, error logging) is unified.
3. Registry Pattern: Format lookup with alias normalization (e.g. 'mps' -> 'pytorch_mps').
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from decision_engine.export.onnx_exporter import ExportableTamevWrapper, get_dynamic_axes_config
from decision_engine.export.quantizer import quantize_onnx_dynamic_int8, quantize_tamev_model_int8


def compute_file_sha256(filepath: str | Path) -> str:
    """Computes SHA-256 hash of a file or directory recursively."""
    path = Path(filepath)
    sha256_hash = hashlib.sha256()
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_file():
                sha256_hash.update(file.relative_to(path).as_posix().encode())
                with open(file, "rb") as f:
                    for byte_block in iter(lambda: f.read(65536), b""):
                        sha256_hash.update(byte_block)
    else:
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@dataclass
class ExportContext:
    """Carries model instance, metadata, output directory, and optional tokenizer."""

    model: nn.Module
    model_name: str
    size_class: str
    output_dir: Path
    tokenizer: Any = None
    # ON by default: the PyTorch exporter emits Identity nodes that alias a shared parameter, and
    # ONNX Runtime's dynamic quantizer only converts a MatMul whose B is a constant, so without
    # de-aliasing those weights stay FP32. On the shipped nano tier, de-aliased INT8 is 14.18 MiB
    # (0.26x fp32) at 97.1% argmax agreement vs 31.52 MiB (0.58x fp32) at 97.5% -- 17 MiB shipped
    # for two decisions in 512.
    dealias_int8_weights: bool = True


class BaseExportStrategy(ABC):
    """
    Template Method Pattern base class for format export strategies.
    Unifies artifact sizing, hash verification, directory management, and status logging.
    """

    format_name: str

    def _compute_size_mb(self, path: Path) -> float:
        if path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            total = path.stat().st_size
        return round(total / (1024 * 1024), 2)

    @abstractmethod
    def get_default_path(self, context: ExportContext) -> Path:
        """Determines the canonical default export destination path."""

    @abstractmethod
    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        """Executes format-specific compilation and saving."""

    def execute(self, context: ExportContext, destination: Path | None = None) -> dict[str, Any]:
        """
        Template method executing the unified export lifecycle:
        1. Resolve output path
        2. Ensure parent directory exists
        3. Execute concrete export logic
        4. Calculate artifact size and SHA-256 checksum
        5. Return standardized deployment metadata
        """
        target = destination if destination is not None else self.get_default_path(context)
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)

        context.model.eval()
        artifact_path = self.export_artifact(context, target)
        size_mb = self._compute_size_mb(artifact_path)
        sha256 = compute_file_sha256(artifact_path)

        return {
            "file": artifact_path.name,
            "format": self.format_name,
            "size_mb": size_mb,
            "sha256": sha256,
        }


# =====================================================================
# 1. PyTorch Strategies (FP32, BF16, INT8, MPS)
# =====================================================================


def _extract_model_state_for_export(
    model: nn.Module, dtype: torch.dtype | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Extracts weights and metadata for export.
    For incontext causal models (Kev-style), decouples the trained pointer head and calibration
    temperature from frozen LLM backbones to prevent multi-gigabyte bloat and memory exhaustion.
    """
    model_type = getattr(model, "model_type", "")
    if not model_type and hasattr(model, "config"):
        model_type = getattr(model.config, "model_type", "")

    extra_meta: dict[str, Any] = {}
    if model_type in ("incontext_causal", "incontext", "kev"):
        extra_meta["model_type"] = model_type
        if hasattr(model, "config") and hasattr(model.config, "backbone"):
            extra_meta["backbone"] = model.config.backbone
        temp = 1.0
        if hasattr(model, "temperature"):
            t_val = model.temperature
            temp = float(t_val.detach().cpu().item()) if hasattr(t_val, "detach") else float(t_val)
        extra_meta["temperature"] = temp

        ph = getattr(model, "pointer_head", model)
        sd: dict[str, Any] = {}
        for k, v in ph.state_dict().items():
            tensor = v.detach().cpu()
            sd[k] = tensor.to(dtype) if dtype is not None else tensor
        if hasattr(model, "q_proj"):
            w = model.q_proj.weight.detach().cpu()
            sd["q_proj.weight"] = w.to(dtype) if dtype is not None else w
            if model.q_proj.bias is not None:
                b = model.q_proj.bias.detach().cpu()
                sd["q_proj.bias"] = b.to(dtype) if dtype is not None else b
        if hasattr(model, "k_proj"):
            w = model.k_proj.weight.detach().cpu()
            sd["k_proj.weight"] = w.to(dtype) if dtype is not None else w
            if model.k_proj.bias is not None:
                b = model.k_proj.bias.detach().cpu()
                sd["k_proj.bias"] = b.to(dtype) if dtype is not None else b
        return sd, extra_meta

    sd = {}
    for k, v in model.state_dict().items():
        tensor = v.detach().cpu()
        sd[k] = tensor.to(dtype) if dtype is not None else tensor
    return sd, extra_meta


class PyTorchFP32Strategy(BaseExportStrategy):
    format_name = "pytorch_fp32"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}_fp32.pt"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        state_dict, extra = _extract_model_state_for_export(context.model, torch.float32)
        payload = {"model_state_dict": state_dict, "size_class": context.size_class, **extra}
        torch.save(payload, destination)
        return destination


class PyTorchBF16Strategy(BaseExportStrategy):
    format_name = "pytorch_bf16"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}_bf16.pt"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        state_dict, extra = _extract_model_state_for_export(context.model, torch.bfloat16)
        payload = {"model_state_dict": state_dict, "size_class": context.size_class, **extra}
        torch.save(payload, destination)
        return destination


class PyTorchINT8Strategy(BaseExportStrategy):
    format_name = "pytorch_int8"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}_int8.pt"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        model_type = getattr(context.model, "model_type", "")
        if not model_type and hasattr(context.model, "config"):
            model_type = getattr(context.model.config, "model_type", "")
        if model_type in ("incontext_causal", "incontext", "kev"):
            ph = getattr(context.model, "pointer_head", context.model)
            quant_ph, _ = quantize_tamev_model_int8(ph)
            state_dict = quant_ph.state_dict()
            temp = 1.0
            if hasattr(context.model, "temperature"):
                t_val = context.model.temperature
                temp = (
                    float(t_val.detach().cpu().item()) if hasattr(t_val, "detach") else float(t_val)
                )
            payload = {
                "model_state_dict": state_dict,
                "size_class": context.size_class,
                "temperature": temp,
                "model_type": model_type,
            }
            torch.save(payload, destination)
            return destination

        quant_model, _ = quantize_tamev_model_int8(context.model)
        torch.save(
            {
                "model_state_dict": quant_model.state_dict(),
                "size_class": context.size_class,
            },
            destination,
        )
        return destination


class PyTorchMPSStrategy(BaseExportStrategy):
    format_name = "pytorch_mps"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}_mps_fp16.pt"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        state_dict, extra = _extract_model_state_for_export(context.model, torch.float16)
        payload = {
            "model_state_dict": state_dict,
            "size_class": context.size_class,
            "target_device": "mps",
            "precision": "float16",
            **extra,
        }
        torch.save(payload, destination)
        return destination


# =====================================================================
# 2. ONNX Strategies (FP32, Dynamic INT8)
# =====================================================================


class ONNXFP32Strategy(BaseExportStrategy):
    format_name = "onnx_fp32"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}.onnx"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        wrapped = ExportableTamevWrapper(context.model)
        wrapped.eval()
        dummy_ctx_ids = torch.ones((1, 32), dtype=torch.long)
        dummy_ctx_mask = torch.ones((1, 32), dtype=torch.long)
        dummy_opt_ids = torch.ones((1, 4, 16), dtype=torch.long)
        dummy_opt_mask = torch.ones((1, 4, 16), dtype=torch.long)

        input_names = [
            "ctx_input_ids",
            "ctx_attention_mask",
            "opt_input_ids",
            "opt_attention_mask",
        ]
        output_names = ["logits", "probs"]
        dynamic_axes = {
            k: v for k, v in get_dynamic_axes_config().items() if k in (input_names + output_names)
        }

        torch.onnx.export(
            wrapped,
            (dummy_ctx_ids, dummy_ctx_mask, dummy_opt_ids, dummy_opt_mask),
            str(destination),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        return destination


class ONNXINT8Strategy(BaseExportStrategy):
    format_name = "onnx_int8"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}_int8.onnx"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        base_onnx = context.output_dir / f"{context.model_name}.onnx"
        if not base_onnx.exists():
            ONNXFP32Strategy().execute(context, destination=base_onnx)
        _ = quantize_onnx_dynamic_int8(
            str(base_onnx), str(destination), dealias_weights=context.dealias_int8_weights
        )
        return destination


# =====================================================================
# 3. Apple Ecosystem Strategies (CoreML, MLX)
# =====================================================================


class AppleCoreMLStrategy(BaseExportStrategy):
    format_name = "coreml"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}.mlpackage"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        from decision_engine.export.apple_exporter import export_coreml

        return export_coreml(
            model=context.model,
            tokenizer=context.tokenizer,
            output_dir=destination.parent if destination.suffix == ".mlpackage" else destination,
            model_name=context.model_name,
            precision="fp16",
        )


class AppleMLXStrategy(BaseExportStrategy):
    format_name = "mlx"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}_mlx"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        from decision_engine.export.apple_exporter import export_mlx

        mlx_path = export_mlx(
            model=context.model,
            tokenizer=context.tokenizer,
            output_dir=destination,
            model_name=context.model_name,
        )
        # Also ensure root output_dir has weights.npz and tamev_mlx_config.json if distinct
        if destination != context.output_dir:
            export_mlx(
                model=context.model,
                tokenizer=context.tokenizer,
                output_dir=context.output_dir,
                model_name=context.model_name,
            )
        return mlx_path


# =====================================================================
# 4. Embedded & C++ Strategy (TorchScript)
# =====================================================================


class TorchScriptStrategy(BaseExportStrategy):
    format_name = "torchscript"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}.torchscript"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        wrapped = ExportableTamevWrapper(context.model)
        dummy_ctx_ids = torch.ones((1, 32), dtype=torch.long)
        dummy_ctx_mask = torch.ones((1, 32), dtype=torch.long)
        dummy_opt_ids = torch.ones((1, 4, 16), dtype=torch.long)
        dummy_opt_mask = torch.ones((1, 4, 16), dtype=torch.long)

        traced = torch.jit.trace(
            wrapped,
            (dummy_ctx_ids, dummy_ctx_mask, dummy_opt_ids, dummy_opt_mask),
            strict=False,
        )
        traced.save(str(destination))
        return destination


# =====================================================================
# 5. GGUF Strategy (llama.cpp / Ollama / GGML)
# =====================================================================


class GGUFStrategy(BaseExportStrategy):
    format_name = "gguf"

    def get_default_path(self, context: ExportContext) -> Path:
        return context.output_dir / f"{context.model_name}.gguf"

    def export_artifact(self, context: ExportContext, destination: Path) -> Path:
        from decision_engine.export.gguf_exporter import export_gguf

        return export_gguf(
            model=context.model,
            output_path=destination,
            model_name=context.model_name,
            size_class=context.size_class,
            precision="fp16",
            tokenizer=context.tokenizer,
        )


# =====================================================================
# 6. Export Strategy Registry
# =====================================================================


class ExportStrategyRegistry:
    """Registry managing available format export strategies and friendly aliases."""

    _strategies: dict[str, BaseExportStrategy] = {}
    _aliases: dict[str, str] = {
        "mps": "pytorch_mps",
        "metal": "pytorch_mps",
        "onnx": "onnx_int8",
        "ts": "torchscript",
        "fp32": "pytorch_fp32",
        "bf16": "pytorch_bf16",
        "int8": "pytorch_int8",
        "apple": "coreml",
        "llama.cpp": "gguf",
        "ggml": "gguf",
    }

    @classmethod
    def register(cls, strategy: BaseExportStrategy) -> None:
        cls._strategies[strategy.format_name.lower()] = strategy

    @classmethod
    def get(cls, format_name: str) -> BaseExportStrategy:
        fmt = format_name.strip().lower()
        resolved = cls._aliases.get(fmt, fmt)
        if resolved not in cls._strategies:
            available = list(cls._strategies.keys()) + list(cls._aliases.keys())
            raise ValueError(
                f"Unknown export format: '{format_name}'. Available formats: {available}"
            )
        return cls._strategies[resolved]

    @classmethod
    def resolve_formats(cls, formats: list[str] | str | None) -> list[BaseExportStrategy]:
        if formats is None or formats == "all":
            # Return all primary registered strategies in canonical order
            order = [
                "pytorch_fp32",
                "pytorch_bf16",
                "pytorch_int8",
                "pytorch_mps",
                "onnx_fp32",
                "onnx_int8",
                "coreml",
                "mlx",
                "torchscript",
                "gguf",
            ]
            return [cls._strategies[k] for k in order if k in cls._strategies]

        if isinstance(formats, str):
            names = [f.strip() for f in formats.split(",") if f.strip()]
        else:
            names = list(formats)

        strategies: list[BaseExportStrategy] = []
        seen = set()
        for name in names:
            strat = cls.get(name)
            if strat.format_name not in seen:
                seen.add(strat.format_name)
                strategies.append(strat)
        return strategies


# Register all 10 primary strategies
for strat in [
    PyTorchFP32Strategy(),
    PyTorchBF16Strategy(),
    PyTorchINT8Strategy(),
    PyTorchMPSStrategy(),
    ONNXFP32Strategy(),
    ONNXINT8Strategy(),
    AppleCoreMLStrategy(),
    AppleMLXStrategy(),
    TorchScriptStrategy(),
    GGUFStrategy(),
]:
    ExportStrategyRegistry.register(strat)
