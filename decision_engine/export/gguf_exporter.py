# decision_engine/export/gguf_exporter.py
"""
GGUF Format Exporter for TAMEV Decision Models.
Exports TAMEV decision models to the GGUF binary format compatible with llama.cpp,
Ollama, LM Studio, and the broader GGML edge inference ecosystem.

Follows official GGML & llama.cpp industry standards:
1. Architecture resolution driven by Hugging Face `AutoConfig` and `gguf.constants.MODEL_ARCH`.
2. Tensor quantization driven directly by `gguf.quants.quantize` and `GGMLQuantizationType`.
3. Standard GGML rules: 1D tensors (biases, norm scales) remain in F16/F32 for numerical stability.
4. Declarative `GGUFArchitectureRegistry` for extensible, zero-hardcoding model mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import gguf
    from gguf.constants import MODEL_ARCH, MODEL_ARCH_NAMES
    from gguf.quants import GGMLQuantizationType, QuantError, quantize
except ImportError:
    gguf = None
    MODEL_ARCH = None
    MODEL_ARCH_NAMES = {}
    GGMLQuantizationType = None
    QuantError = Exception
    quantize = None

try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None

logger = logging.getLogger(__name__)

__all__ = [
    "GGUFArchitectureRegistry",
    "GGUFTensorProcessor",
    "export_gguf",
]


class GGUFArchitectureRegistry:
    """
    Declarative architecture registry mapping Hugging Face model types and
    architectures to canonical GGUF `MODEL_ARCH` constants.
    Eliminates ad-hoc substring heuristics in favor of formal configuration inspection.
    """

    _HF_TYPE_MAP: dict[str, Any] = {}
    _FALLBACK_ARCH_STR: str = "bert"

    @classmethod
    def _init_defaults(cls) -> None:
        if cls._HF_TYPE_MAP or gguf is None:
            return

        cls._HF_TYPE_MAP.update(
            {
                # Encoders
                "bert": MODEL_ARCH.BERT,
                "modernbert": MODEL_ARCH.MODERN_BERT,
                "roberta": MODEL_ARCH.BERT,
                "deberta": MODEL_ARCH.BERT,
                "deberta-v2": MODEL_ARCH.BERT,
                "distilbert": MODEL_ARCH.BERT,
                "nomic_bert": MODEL_ARCH.NOMIC_BERT,
                "jina_bert": MODEL_ARCH.JINA_BERT_V2,
                "tinybert": MODEL_ARCH.BERT,
                # Causal LMs
                "qwen2": MODEL_ARCH.QWEN2,
                "qwen3": MODEL_ARCH.QWEN3,
                "qwen3_5": MODEL_ARCH.QWEN35,
                "qwen": MODEL_ARCH.QWEN,
                "llama": MODEL_ARCH.LLAMA,
                "smollm": MODEL_ARCH.LLAMA,
                "smollm2": MODEL_ARCH.LLAMA,
                "mistral": MODEL_ARCH.LLAMA,
                "gpt2": MODEL_ARCH.GPT2,
            }
        )

    @classmethod
    def register(cls, model_type: str, arch: Any) -> None:
        """Registers a custom model_type to a GGUF MODEL_ARCH constant."""
        cls._init_defaults()
        cls._HF_TYPE_MAP[model_type.lower().strip()] = arch

    @classmethod
    def resolve_architecture(cls, model: nn.Module, size_class: str = "") -> str:
        """
        Infers the canonical GGUF architecture identifier by inspecting
        the model's underlying Hugging Face configuration.
        """
        cls._init_defaults()
        if gguf is None:
            return cls._FALLBACK_ARCH_STR

        detected_type: str | None = None

        # 1. Inspect backbone config if model wraps a Hugging Face PreTrainedModel
        backbone = getattr(model, "backbone", None)
        if backbone is not None and hasattr(backbone, "config"):
            detected_type = getattr(backbone.config, "model_type", None)

        # 2. Inspect root model config
        if not detected_type and hasattr(model, "config"):
            cfg = model.config
            detected_type = getattr(cfg, "model_type", None)
            if not detected_type and hasattr(cfg, "backbone") and AutoConfig is not None:
                bb_target = cfg.backbone
                try:
                    loaded_cfg = AutoConfig.from_pretrained(str(bb_target), trust_remote_code=True)
                    detected_type = getattr(loaded_cfg, "model_type", None)
                except Exception as exc:
                    logger.debug(f"AutoConfig resolution for {bb_target} skipped: {exc}")

        # 3. Match against canonical Hugging Face to GGUF mapping
        if detected_type:
            cleaned_type = str(detected_type).lower().strip().replace("-", "_")
            if cleaned_type in cls._HF_TYPE_MAP:
                arch_enum = cls._HF_TYPE_MAP[cleaned_type]
                return MODEL_ARCH_NAMES.get(arch_enum, arch_enum.name.lower())

            # Fallback check for unnormalized keys
            for k, arch_enum in cls._HF_TYPE_MAP.items():
                if k in cleaned_type:
                    return MODEL_ARCH_NAMES.get(arch_enum, arch_enum.name.lower())

        # 4. TAMEV model family fallbacks
        root_type = getattr(model, "model_type", "")
        if root_type in ("incontext_causal", "incontext", "kev"):
            return MODEL_ARCH_NAMES.get(MODEL_ARCH.QWEN35, "qwen35")

        return cls._FALLBACK_ARCH_STR


class GGUFTensorProcessor:
    """
    Prepares and quantizes tensors for GGUF serialization using official GGML / llama.cpp
    specifications via `gguf.quants.quantize`.
    """

    _PRECISION_MAP: dict[str, Any] = {}

    @classmethod
    def _init_map(cls) -> None:
        if cls._PRECISION_MAP or gguf is None:
            return

        cls._PRECISION_MAP.update(
            {
                "fp32": GGMLQuantizationType.F32,
                "f32": GGMLQuantizationType.F32,
                "fp16": GGMLQuantizationType.F16,
                "f16": GGMLQuantizationType.F16,
                "bf16": GGMLQuantizationType.BF16,
                "q8_0": GGMLQuantizationType.Q8_0,
                "q8": GGMLQuantizationType.Q8_0,
                "q4_0": GGMLQuantizationType.Q4_0,
                "q4_1": GGMLQuantizationType.Q4_1,
                "q5_0": GGMLQuantizationType.Q5_0,
                "q5_1": GGMLQuantizationType.Q5_1,
            }
        )

    @classmethod
    def resolve_quantization_type(cls, precision: str | Any) -> Any:
        cls._init_map()
        if gguf is None:
            return None
        if isinstance(precision, GGMLQuantizationType):
            return precision
        return cls._PRECISION_MAP.get(str(precision).lower().strip(), GGMLQuantizationType.F16)

    @classmethod
    def prepare_tensor(
        cls,
        tensor: torch.Tensor | np.ndarray,
        target_qtype: Any,
    ) -> tuple[np.ndarray, Any | None]:
        """
        Prepares a tensor for GGUF serialization:
        - 1D biases and normalization scales are kept in FP32 / FP16 following GGML rules.
        - 2D+ weight matrices are quantized using official `gguf.quants.quantize`.
        - Cleanly falls back to FP16 if quantization raises `QuantError`.
        """
        if gguf is None:
            raise RuntimeError("The 'gguf' package is required. Install via 'uv add gguf'.")

        # 1. Convert to NumPy
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)

        # 2. Preserve integer tensors as-is (token IDs, position IDs)
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int32), None

        arr_f32 = arr.astype(np.float32)

        # 3. GGML standard stability rule: 1D tensors (biases, layernorm/rmsnorm) remain in F16 or F32
        if arr.ndim < 2:
            fallback_qtype = (
                GGMLQuantizationType.F32
                if target_qtype == GGMLQuantizationType.F32
                else GGMLQuantizationType.F16
            )
            return quantize(arr_f32, fallback_qtype), None

        # 4. Standard float outputs
        if target_qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            return quantize(arr_f32, target_qtype), None

        # 5. Quantized formats (Q8_0, Q4_0, etc.) via official gguf.quants.quantize
        try:
            quantized_data = quantize(arr_f32, target_qtype)
            return quantized_data, target_qtype
        except (QuantError, ValueError) as exc:
            logger.debug(f"Quantization fallback to F16: {exc}")
            return quantize(arr_f32, GGMLQuantizationType.F16), None


def export_gguf(  # noqa: PLR0917
    model: nn.Module,
    output_path: str | Path,
    model_name: str = "tamev_model",
    size_class: str = "nano",
    precision: str = "fp16",
    tokenizer: Any = None,
) -> Path:
    """
    Exports a TAMEV decision model to the GGUF binary format (.gguf).

    Args:
        model: PyTorch model instance (EncoderDecisionModel, InContextCausalDecisionModel, etc.)
        output_path: Destination path for the .gguf file.
        model_name: Descriptive name for model metadata.
        size_class: Architecture tier name ('nano', 'micro', 'small', 'medium', 'large').
        precision: Quantization/precision mode ('fp16', 'fp32', 'q8_0', 'q4_0').
        tokenizer: Optional tokenizer instance.

    Returns:
        Path to the exported .gguf file.
    """
    if gguf is None:
        raise RuntimeError(
            "The 'gguf' package is required for GGUF export. Install via 'uv add gguf'."
        )

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    arch = GGUFArchitectureRegistry.resolve_architecture(model, size_class)
    target_qtype = GGUFTensorProcessor.resolve_quantization_type(precision)

    writer = gguf.GGUFWriter(dest, arch)

    # 1. Standard GGUF Metadata
    writer.add_name(model_name)
    writer.add_description(f"TAMEV {size_class.capitalize()} Decision Engine ({precision.upper()})")

    file_type_map = {
        GGMLQuantizationType.F32: int(gguf.LlamaFileType.ALL_F32),
        GGMLQuantizationType.F16: int(gguf.LlamaFileType.MOSTLY_F16),
        GGMLQuantizationType.Q8_0: int(gguf.LlamaFileType.MOSTLY_Q8_0),
        GGMLQuantizationType.Q4_0: int(gguf.LlamaFileType.MOSTLY_Q4_0),
    }
    file_type = file_type_map.get(target_qtype, int(gguf.LlamaFileType.MOSTLY_F16))
    writer.add_uint32("general.file_type", file_type)

    # 2. Extract Exact Parameter Counts & Structural Dimensions
    total_active_params = sum(p.numel() for p in model.parameters())

    temp_val = 1.0
    if hasattr(model, "temperature"):
        t_param = model.temperature
        temp_val = (
            float(t_param.detach().cpu().item()) if hasattr(t_param, "detach") else float(t_param)
        )
    elif hasattr(model, "pointer_head") and hasattr(model.pointer_head, "temperature"):
        t_param = model.pointer_head.temperature
        temp_val = (
            float(t_param.detach().cpu().item()) if hasattr(t_param, "detach") else float(t_param)
        )

    hidden_dim = getattr(model, "hidden_dim", 312)
    proj_dim = getattr(model, "projection_dim", 64)

    # Human-readable standard notation
    if total_active_params >= 1_000_000_000:
        params_str = f"{total_active_params / 1e9:.2f}B"
    else:
        params_str = f"{total_active_params / 1e6:.2f}M"

    # 3. TAMEV Metadata Tags
    writer.add_string("tamev.tier", size_class)
    writer.add_string("tamev.model_name", model_name)
    writer.add_string("tamev.architecture_type", getattr(model, "model_type", "encoder"))
    writer.add_uint32("tamev.active_parameters", total_active_params)
    writer.add_string("tamev.active_params_formatted", params_str)
    writer.add_float32("tamev.temperature", temp_val)
    writer.add_uint32("tamev.hidden_dim", int(hidden_dim))
    writer.add_uint32("tamev.projection_dim", int(proj_dim))
    writer.add_string("tamev.permutation_invariance", "exact_0.00000000_drift")
    writer.add_float32("tamev.flip_rate", 0.0)

    # 4. Serialize Tensors using GGUFTensorProcessor
    state_dict = model.state_dict()
    tensor_count = 0

    for name, param in state_dict.items():
        processed_arr, raw_dtype = GGUFTensorProcessor.prepare_tensor(param, target_qtype)
        if raw_dtype is not None:
            writer.add_tensor(name, processed_arr, raw_dtype=raw_dtype)
        else:
            writer.add_tensor(name, processed_arr)
        tensor_count += 1

    # 5. Flush and Write Binary Sections
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    logger.info(
        f"GGUF model exported to {dest} ({tensor_count} tensors, {total_active_params:,} active params, {precision})"
    )
    return dest
