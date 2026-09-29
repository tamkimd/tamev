# decision_engine/config/presets.py
"""
Predefined Architectural Presets and Size Profiles for TAMEV.
"""

from typing import Any

from decision_engine.config.base import (
    TamevConfig,
)

PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    # 1. Nano Tier (14.5M params, sub-5ms CPU, ~15MB INT8)
    "nano": {
        "model": {
            "name": "tamev_nano_tinybert",
            "size_profile": "nano",
            "model_type": "encoder",
            "backbone": "huawei-noah/TinyBERT_General_4L_312D",
            "projection_dim": 64,
            "temperature": 2.2,
            "pooling": "cls",
            # Must mirror the shipped artifact's `config.json`: exported runtimes are traced from
            # THIS preset, so a `bilinear` here while the shipped weights were trained `cross`
            # produces runtimes that score the wrong branch.
            "option_fusion": "cross",
        },
        "export": {
            "precisions": ["fp32", "fp16", "int8"],
            "targets": ["onnx", "pytorch"],
            "output_dir": "models/exported/nano",
        },
    },
    # 2. Micro Tier (22.7M params, sub-8ms CPU, ~25MB INT8)
    "micro": {
        "model": {
            "name": "tamev_micro_minilm",
            "size_profile": "micro",
            "model_type": "encoder",
            "backbone": "sentence-transformers/all-MiniLM-L6-v2",
            "projection_dim": 64,
            "temperature": 2.0,
            "pooling": "mean",
        },
        "export": {
            "precisions": ["fp32", "fp16", "int8"],
            "targets": ["onnx", "pytorch"],
            "output_dir": "models/exported/micro",
        },
    },
    # 3. Small Tier - gte-ModernBERT (149M params, high-accuracy bidirectional). The backbone is
    # Alibaba-NLP/gte-modernbert-base, not answerdotai/ModernBERT-base: the shipped weights were
    # trained on a local fp32 copy of gte's checkpoint, and the tokenizer hash and config keys
    # confirm it.
    "small_modernbert": {
        "model": {
            "name": "tamev_small_modernbert",
            "size_profile": "small",
            "model_type": "encoder",
            "backbone": "Alibaba-NLP/gte-modernbert-base",
            "projection_dim": 256,
            "temperature": 2.0,
            "pooling": "mean",
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 4e-5,
            "output_dir": "runs/tamev_small_modernbert",
        },
        "export": {
            "precisions": ["fp32", "bf16", "int8"],
            "targets": ["onnx", "pytorch"],
            "output_dir": "models/exported/small_modernbert",
        },
    },
    # 4. Medium Tier - Qwen 3.5 0.8B (800M params, in-context decision model)
    "medium_qwen35_08b": {
        "model": {
            "name": "tamev_medium_qwen35_08b",
            "size_profile": "medium",
            "model_type": "incontext_causal",
            "backbone": "Qwen/Qwen3.5-0.8B-Base",
            "projection_dim": 256,
            "temperature": 2.351,
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 2e-5,
            "output_dir": "runs/tamev_medium_qwen35_08b",
        },
        "export": {
            "precisions": ["bf16", "int8"],
            "targets": ["pytorch", "coreml", "mlx"],
            "output_dir": "models/exported/medium_qwen35_08b",
        },
    },
    # 5. Medium Tier - Qwen 0.5B (490M params, workstation/server)
    "medium_qwen05b": {
        "model": {
            "name": "tamev_medium_qwen05b",
            "size_profile": "medium",
            "model_type": "causal",
            "backbone": "Qwen/Qwen2.5-0.5B",
            "projection_dim": 256,
            "temperature": 1.5,
            "pooling": "last",
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-5,
            "output_dir": "runs/tamev_medium_qwen05b",
        },
        "export": {
            "precisions": ["bf16", "int8"],
            "targets": ["onnx", "pytorch"],
            "output_dir": "models/exported/medium_qwen05b",
        },
    },
    # 6. Large Tier - Qwen 3.5 4B (4.4B params, server grade, matches Kev-4B)
    "large_qwen35_4b": {
        "model": {
            "name": "tamev_large_qwen35_4b",
            "size_profile": "large",
            "model_type": "incontext_causal",
            "backbone": "Qwen/Qwen3.5-4B",
            "projection_dim": 256,
            "temperature": 2.406,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "output_dir": "runs/tamev_large_qwen35_4b",
        },
        "export": {
            "precisions": ["bf16", "int8"],
            "targets": ["pytorch", "coreml", "mlx"],
            "output_dir": "models/exported/large_qwen35_4b",
        },
    },
    "large_qwen4b": {
        "model": {
            "name": "tamev_large_qwen4b",
            "size_profile": "large",
            "model_type": "causal",
            "backbone": "Qwen/Qwen3.5-4B",
            "projection_dim": 512,
            "temperature": 1.5,
            "pooling": "last",
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-5,
            "output_dir": "runs/tamev_large_qwen4b",
        },
        "export": {
            "precisions": ["bf16", "int8"],
            "targets": ["onnx", "pytorch"],
            "output_dir": "models/exported/large_qwen4b",
        },
    },
}

# Aliases for convenience
PRESET_CONFIGS["small"] = PRESET_CONFIGS["small_modernbert"]
PRESET_CONFIGS["medium"] = PRESET_CONFIGS["medium_qwen35_08b"]
PRESET_CONFIGS["large"] = PRESET_CONFIGS["large_qwen35_4b"]


def get_preset_config(name: str) -> TamevConfig:
    """Retrieve pre-tuned TamevConfig by preset name."""
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: '{name}'. Available: {list(PRESET_CONFIGS.keys())}")
    base = TamevConfig()
    preset_dict = PRESET_CONFIGS[name]

    # Merge preset values over default config
    merged = base.to_dict()
    for section, values in preset_dict.items():
        if section in merged and isinstance(values, dict):
            merged[section].update(values)
        else:
            merged[section] = values

    return TamevConfig.from_dict(merged)
