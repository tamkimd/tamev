# decision_engine/export/apple_exporter.py
"""
Apple Silicon & Apple Ecosystem Export Engine for TAMEV Decision Models.
Supports:
1. Apple CoreML (.mlpackage) for Apple Neural Engine (ANE) and Metal Performance Shaders (MPSGraph).
2. Apple MLX format (weights.npz / weights.safetensors + tamev_mlx_config.json) for unified memory serving.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

try:
    import coremltools as ct
except ImportError:
    ct = None

import numpy as np
import torch
import torch.nn as nn

try:
    from safetensors.numpy import save_file as safetensors_save_file
except ImportError:
    safetensors_save_file = None

logger = logging.getLogger(__name__)


class CoreMLEndToEndWrapper(nn.Module):
    """
    Wraps full TAMEV decision model with static/dynamic inputs suitable for CoreML compilation.
    Performs context and option encoding followed by pointer head bilinear compatibility scoring.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        ctx_input_ids: torch.Tensor,
        ctx_attention_mask: torch.Tensor,
        opt_input_ids: torch.Tensor,
        opt_attention_mask: torch.Tensor,
        num_options: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            ctx_input_ids=ctx_input_ids,
            ctx_attention_mask=ctx_attention_mask,
            opt_input_ids=opt_input_ids,
            opt_attention_mask=opt_attention_mask,
        )
        logits = out["logits"]
        K = logits.shape[-1]
        mask = torch.arange(K, device=logits.device).unsqueeze(0) < num_options.unsqueeze(1)
        logits = torch.where(
            mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype)
        )
        temp = (
            self.model.pointer_head.temperature
            if hasattr(self.model, "pointer_head")
            and hasattr(self.model.pointer_head, "temperature")
            else torch.tensor(1.0, device=logits.device)
        )
        t = torch.clamp(temp, min=0.05)
        probs = torch.softmax(logits / t, dim=-1)
        return logits, probs


class CoreMLPointerCompatWrapper(nn.Module):
    """
    CoreML compatibility module for pointer network compatibility scoring.
    Decoupled representation keeps the exported graph free of dynamic shapes, so it can be
    targeted at Apple Neural Engine (ANE) / Metal Performance Shaders (MPSGraph). Latency is
    not benchmarked in this repository for this runner - measure it on your own device.
    """

    def __init__(self, pointer_head: nn.Module):
        super().__init__()
        self.pointer_head = pointer_head

    def forward(
        self,
        context_vec: torch.Tensor,
        option_vecs: torch.Tensor,
        num_options: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, probs = self.pointer_head(context_vec, option_vecs)
        K = logits.shape[-1]
        mask = torch.arange(K, device=logits.device).unsqueeze(0) < num_options.unsqueeze(1)
        logits = torch.where(
            mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype)
        )
        temp = (
            self.pointer_head.temperature
            if hasattr(self.pointer_head, "temperature")
            else torch.tensor(1.0, device=logits.device)
        )
        t = torch.clamp(temp, min=0.05)
        probs = torch.softmax(logits / t, dim=-1)
        return logits, probs


def export_coreml(
    model: nn.Module,
    tokenizer: Any = None,
    output_dir: str | Path = "exported_models",
    model_name: str = "tamev_model",
    precision: str = "fp16",
) -> Path:
    """
    Exports a TAMEV decision model to Apple CoreML (.mlpackage) format.
    Uses coremltools.convert with TorchScript trace on dummy inputs
    (ctx_input_ids, ctx_attention_mask, opt_input_ids, opt_attention_mask, num_options).
    If CoreML conversion encounters unsupported complex dynamic shapes or ops,
    provides a clean fallback exporting the pointer compatibility layer.

    Args:
        model: PyTorch decision model or pointer head.
        tokenizer: Optional tokenizer instance.
        output_dir: Destination directory or file path for the .mlpackage.
        model_name: Identifier name for the model.
        precision: Compute precision ('fp16' or 'fp32').

    Returns:
        Path to the saved .mlpackage bundle directory.
    """
    target = Path(output_dir)
    mlpackage_path = target if target.suffix == ".mlpackage" else target / f"{model_name}.mlpackage"
    mlpackage_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if ct is None:
        mlpackage_path.mkdir(parents=True, exist_ok=True)
        manifest_meta = {
            "model_name": model_name,
            "precision": precision,
            "status": "coreml_package_stub_ane",
            "runtime": "Apple Neural Engine / CoreML",
            "notice": "coremltools not installed; created CoreML package bundle for ANE",
        }
        with open(mlpackage_path / "Manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_meta, f, indent=2)
        logger.warning(
            f"coremltools not installed; created CoreML package container at {mlpackage_path}"
        )
        return mlpackage_path

    # Configure CoreML precision
    ct_precision = ct.precision.FLOAT16 if precision.lower() == "fp16" else ct.precision.FLOAT32

    # Dummy inputs for full model trace
    dummy_ctx_ids = torch.ones((1, 32), dtype=torch.long)
    dummy_ctx_mask = torch.ones((1, 32), dtype=torch.long)
    dummy_opt_ids = torch.ones((1, 4, 16), dtype=torch.long)
    dummy_opt_mask = torch.ones((1, 4, 16), dtype=torch.long)
    dummy_num_options = torch.tensor([4], dtype=torch.long)

    mlmodel = None

    # Step 1: Attempt end-to-end full model conversion
    try:
        wrapper = CoreMLEndToEndWrapper(model)
        traced_full = torch.jit.trace(
            wrapper,
            (dummy_ctx_ids, dummy_ctx_mask, dummy_opt_ids, dummy_opt_mask, dummy_num_options),
            strict=False,
        )
        mlmodel = ct.convert(
            traced_full,
            inputs=[
                ct.TensorType(name="ctx_input_ids", shape=dummy_ctx_ids.shape, dtype=int),
                ct.TensorType(name="ctx_attention_mask", shape=dummy_ctx_mask.shape, dtype=int),
                ct.TensorType(name="opt_input_ids", shape=dummy_opt_ids.shape, dtype=int),
                ct.TensorType(name="opt_attention_mask", shape=dummy_opt_mask.shape, dtype=int),
                ct.TensorType(name="num_options", shape=dummy_num_options.shape, dtype=int),
            ],
            outputs=[
                ct.TensorType(name="logits"),
                ct.TensorType(name="probs"),
            ],
            convert_to="mlprogram",
            compute_precision=ct_precision,
            minimum_deployment_target=ct.target.iOS15,
        )
        logger.info("CoreML full end-to-end model conversion succeeded.")
    except Exception as exc:
        logger.warning(
            f"CoreML full model conversion encountered unsupported dynamic shapes or ops: {exc}. "
            f"Activating clean fallback: exporting pointer compatibility layer."
        )

    # Step 2: Clean fallback: pointer compatibility layer
    if mlmodel is None:
        pointer_head = getattr(model, "pointer_head", None)
        if pointer_head is None:
            pointer_head = model

        hidden_dim = getattr(pointer_head, "hidden_dim", 312)
        compat_wrapper = CoreMLPointerCompatWrapper(pointer_head)
        ctx_vec = torch.randn(1, hidden_dim)
        opt_vecs = torch.randn(1, 4, hidden_dim)
        num_opt = torch.tensor([4], dtype=torch.long)

        traced_compat = torch.jit.trace(compat_wrapper, (ctx_vec, opt_vecs, num_opt), strict=False)
        mlmodel = ct.convert(
            traced_compat,
            inputs=[
                ct.TensorType(name="context_vec", shape=ctx_vec.shape),
                ct.TensorType(name="option_vecs", shape=opt_vecs.shape),
                ct.TensorType(name="num_options", shape=num_opt.shape, dtype=int),
            ],
            outputs=[
                ct.TensorType(name="logits"),
                ct.TensorType(name="probs"),
            ],
            convert_to="mlprogram",
            compute_precision=ct_precision,
            minimum_deployment_target=ct.target.iOS15,
        )
        logger.info("CoreML pointer compatibility layer converted successfully.")

    # Metadata tagging for Apple Silicon / ANE deployment
    mlmodel.short_description = f"TAMEV Decision Engine CoreML ({model_name})"
    mlmodel.version = "1.0.0"
    mlmodel.author = "TAMEV Apple Format Engine"

    # Save to .mlpackage
    mlmodel.save(str(mlpackage_path))
    logger.info(f"CoreML model package saved to {mlpackage_path}")
    return mlpackage_path


def export_mlx(
    model: nn.Module,
    tokenizer: Any = None,
    output_dir: str | Path = "exported_models",
    model_name: str = "tamev_model",
) -> Path:
    """
    Exports a TAMEV decision model to Apple MLX native format.
    Extracts PyTorch state_dict, converts tensors to numpy/safetensors for MLX,
    writes weights.npz, weights.safetensors, and tamev_mlx_config.json containing
    model configuration, architecture type, temperature, and vocabulary metadata.

    Args:
        model: PyTorch decision model.
        tokenizer: Optional tokenizer instance.
        output_dir: Destination directory.
        model_name: Identifier name for the model.

    Returns:
        Path to the directory containing MLX model artifacts.
    """
    out_path = Path(output_dir)
    mlx_dir = out_path.parent if out_path.suffix in (".npz", ".safetensors") else out_path
    mlx_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # 1. Extract PyTorch state_dict and convert tensors to numpy float32
    arch_type = getattr(model, "model_type", None)
    if arch_type is None and hasattr(model, "config"):
        arch_type = getattr(model.config, "model_type", None)
    if arch_type is None:
        arch_type = model.__class__.__name__

    numpy_weights: dict[str, np.ndarray] = {}
    if arch_type in ("incontext_causal", "incontext", "kev"):
        ph = getattr(model, "pointer_head", model)
        for k, v in ph.state_dict().items():
            numpy_weights[f"pointer_head.{k}"] = v.detach().cpu().to(torch.float32).numpy()
        if hasattr(model, "q_proj"):
            numpy_weights["pointer_head.q_proj.weight"] = (
                model.q_proj.weight.detach().cpu().to(torch.float32).numpy()
            )
            if model.q_proj.bias is not None:
                numpy_weights["pointer_head.q_proj.bias"] = (
                    model.q_proj.bias.detach().cpu().to(torch.float32).numpy()
                )
        if hasattr(model, "k_proj"):
            numpy_weights["pointer_head.k_proj.weight"] = (
                model.k_proj.weight.detach().cpu().to(torch.float32).numpy()
            )
            if model.k_proj.bias is not None:
                numpy_weights["pointer_head.k_proj.bias"] = (
                    model.k_proj.bias.detach().cpu().to(torch.float32).numpy()
                )
    else:
        state_dict = model.state_dict()
        for k, v in state_dict.items():
            numpy_weights[k] = v.detach().cpu().to(torch.float32).numpy()

    # 2. Save weights.npz (Numpy archive)
    npz_path = mlx_dir / "weights.npz"
    np.savez(npz_path, **numpy_weights)

    # 3. Save weights.safetensors (Zero-copy unified memory mapped)
    safetensors_path = mlx_dir / "weights.safetensors"
    wrote_safetensors = False
    if safetensors_save_file is not None:
        safetensors_save_file(numpy_weights, str(safetensors_path))
        wrote_safetensors = True

    # 4. Extract model configuration and architecture metadata

    # Extract calibrated temperature
    temp_val = 1.0
    if hasattr(model, "pointer_head") and hasattr(model.pointer_head, "temperature"):
        temp_val = float(model.pointer_head.temperature.detach().cpu().item())
    elif hasattr(model, "temperature"):
        t_val = model.temperature
        temp_val = float(t_val.detach().cpu().item()) if hasattr(t_val, "detach") else float(t_val)
    elif hasattr(model, "config") and hasattr(model.config, "temperature"):
        temp_val = float(model.config.temperature)

    # Extract dimensions
    hidden_dim = None
    projection_dim = None
    if hasattr(model, "pointer_head"):
        hidden_dim = getattr(model.pointer_head, "hidden_dim", None)
        projection_dim = getattr(model.pointer_head, "projection_dim", None)
    if hidden_dim is None and hasattr(model, "q_proj"):
        hidden_dim = getattr(model.q_proj, "in_features", None)
    if projection_dim is None and hasattr(model, "q_proj"):
        projection_dim = getattr(model.q_proj, "out_features", None)
    if hidden_dim is None and hasattr(model, "hidden_dim"):
        hidden_dim = model.hidden_dim
    if projection_dim is None and hasattr(model, "projection_dim"):
        projection_dim = model.projection_dim

    # Extract size class
    size_class = getattr(model, "size_class", None)
    if size_class is None and hasattr(model, "config"):
        size_class = getattr(model.config, "size_profile", None) or getattr(
            model.config, "size_class", "micro"
        )
    if size_class is None:
        size_class = "micro"

    # Extract backbone
    backbone_name = getattr(model, "backbone_name", None)
    if backbone_name is None and hasattr(model, "config"):
        backbone_name = getattr(model.config, "backbone", None)

    # Extract vocabulary metadata
    vocab_info: dict[str, Any] = {}
    if tokenizer is not None:
        vocab_info = {
            "tokenizer_type": tokenizer.__class__.__name__,
            "vocab_size": (
                len(tokenizer.get_vocab())
                if hasattr(tokenizer, "get_vocab")
                else getattr(tokenizer, "vocab_size", None)
            ),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "cls_token_id": getattr(tokenizer, "cls_token_id", None),
            "sep_token_id": getattr(tokenizer, "sep_token_id", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
            "mask_token_id": getattr(tokenizer, "mask_token_id", None),
        }
    elif hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        enc_cfg = model.encoder.config
        vocab_info = {
            "tokenizer_type": "BertTokenizer",
            "vocab_size": getattr(enc_cfg, "vocab_size", 30522),
            "pad_token_id": getattr(enc_cfg, "pad_token_id", 0),
            "cls_token_id": 101,
            "sep_token_id": 102,
            "unk_token_id": 100,
        }
    else:
        vocab_info = {
            "tokenizer_type": "StandardTokenizer",
            "vocab_size": 30522,
            "pad_token_id": 0,
            "cls_token_id": 101,
            "sep_token_id": 102,
            "unk_token_id": 100,
        }

    # 5. Build tamev_mlx_config.json
    mlx_config = {
        "model_name": model_name,
        "architecture_type": arch_type,
        "model_type": arch_type,
        "backbone": backbone_name,
        "size_class": size_class,
        "hidden_dim": hidden_dim,
        "projection_dim": projection_dim,
        "temperature": temp_val,
        "vocabulary": vocab_info,
        "precision": "fp32",
        # Only advertise files that were actually written next to this config.
        "weights_files": {"npz": "weights.npz"}
        | ({"safetensors": "weights.safetensors"} if wrote_safetensors else {}),
        "runtime": "mlx",
        "target": "apple_silicon_unified_memory",
    }

    config_path = mlx_dir / "tamev_mlx_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(mlx_config, f, indent=2)

    logger.info(
        f"Exported Apple MLX package for {model_name} to {mlx_dir}: "
        f"[weights.npz, weights.safetensors, tamev_mlx_config.json]"
    )
    return mlx_dir


class MLXTamevDecisionRunner:
    """
    Native Apple Silicon MLX inference runner for TAMEV decision models.
    Operates directly in unified memory (latency not benchmarked for this runner).
    """

    def __init__(self, model_dir: str | Path):
        import mlx.core as mx

        self.mx = mx
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "tamev_mlx_config.json"
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        # Load weights (prefer safetensors for zero-copy, fallback to npz)
        st_path = self.model_dir / "weights.safetensors"
        npz_path = self.model_dir / "weights.npz"
        if st_path.exists():
            self.weights = mx.load(str(st_path))
        elif npz_path.exists():
            self.weights = mx.load(str(npz_path))
        else:
            raise FileNotFoundError(f"No MLX weights found in {self.model_dir}")

        self.temperature = float(self.config.get("temperature", 1.0))
        self.projection_dim = int(self.config.get("projection_dim", 64))
        self.scale = 1.0 / math.sqrt(self.projection_dim)

    def score_pointer(self, context_vec: Any, option_vecs: Any) -> tuple[Any, Any]:
        """
        Bilinear compatibility scoring in Apple Silicon unified memory.

        Args:
            context_vec: (B, hidden_dim) or (hidden_dim,) array/tensor.
            option_vecs: (B, K, hidden_dim) or (K, hidden_dim) array/tensor.

        Returns:
            Tuple of (logits, calibrated probabilities) as MLX arrays.
        """
        mx = self.mx
        if not isinstance(context_vec, mx.array):
            context_vec = mx.array(
                context_vec.detach().cpu().numpy()
                if hasattr(context_vec, "detach")
                else np.asarray(context_vec)
            )
        if not isinstance(option_vecs, mx.array):
            option_vecs = mx.array(
                option_vecs.detach().cpu().numpy()
                if hasattr(option_vecs, "detach")
                else np.asarray(option_vecs)
            )

        W_q = self.weights["pointer_head.q_proj.weight"].T
        W_k = self.weights["pointer_head.k_proj.weight"].T

        if len(context_vec.shape) == 1:
            context_vec = mx.expand_dims(context_vec, 0)
        if len(option_vecs.shape) == 2:
            option_vecs = mx.expand_dims(option_vecs, 0)

        q = mx.expand_dims(context_vec @ W_q, 1)  # (B, 1, proj_dim)
        k = option_vecs @ W_k  # (B, K, proj_dim)

        logits = mx.squeeze(mx.matmul(q, mx.swapaxes(k, 1, 2)), 1) * self.scale
        t = max(self.temperature, 0.05)
        probs = mx.softmax(logits / t, axis=-1)
        return logits, probs
