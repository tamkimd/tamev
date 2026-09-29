# decision_engine/export/quantizer.py
"""
INT8 Quantizer for TAMEV Decision Models.
Applies dynamic INT8 quantization to linear layers, reducing memory footprint
by ~4x and optimizing integer operations for mobile CPUs and NPUs.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from decision_engine.models.tinybert_model import TamevTinyBertDecisionModel


def get_quantization_spec() -> dict[str, Any]:
    """
    Standard quantization profile specifications for mobile NPU/CPU deployment.
    """
    return {
        "precision_candidates": ["BF16", "FP16", "INT8", "INT4"],
        "primary_mobile_target": "INT8",
        "calibration_dataset": "data/processed/val.jsonl",
        "accuracy_drop_tolerance": 0.015,  # max 1.5% accuracy drop
        "ece_increase_tolerance": 0.02,  # max 0.02 ECE degradation
    }


def quantize_tamev_model_int8(
    model: nn.Module, output_path: str | None = None
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Performs dynamic INT8 quantization on all Linear layers of a TAMEV decision model.
    """
    model.eval()
    model.cpu()

    # Configure quantization engine for Apple Silicon / ARM mobile
    if (
        hasattr(torch.backends, "quantized")
        and "qnnpack" in torch.backends.quantized.supported_engines
    ):
        torch.backends.quantized.engine = "qnnpack"

    # Dynamic quantization of Linear layers to qint8
    quantized_model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    stats = {
        "device": "cpu",
        "quant_type": "torch.qint8_dynamic",
        "quantized_modules": ["nn.Linear"],
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model.state_dict(), out_p)
        stats["saved_path"] = str(out_p)
        stats["file_size_mb"] = out_p.stat().st_size / (1024 * 1024)

    return quantized_model, stats


def _dealias_identity_weights(model: Any) -> int:
    """
    Rewires `Identity(weight)` aliases back onto the weight initializer itself.

    The PyTorch exporter emits `Identity` nodes that alias a shared parameter, so many MatMuls
    take their B operand from an Identity output instead of an initializer. ONNX Runtime's
    dynamic quantizer only converts a MatMul whose B is a constant, so those weights silently
    stayed FP32: on Nano that was 17.4 MiB of a 31.5 MiB INT8 file (56%). Removing the aliases
    rewires consumers to the same values (an Identity is a no-op) and makes them quantizable.

    Returns the number of aliases removed.
    """
    initializers = {t.name for t in model.graph.initializer}
    alias: dict[str, str] = {}
    for node in model.graph.node:
        if (
            node.op_type == "Identity"
            and len(node.input) == 1
            and (node.input[0] in initializers or node.input[0] in alias)
        ):
            alias[node.output[0]] = alias.get(node.input[0], node.input[0])

    if not alias:
        return 0

    for node in model.graph.node:
        for i in range(len(node.input)):
            if node.input[i] in alias:
                node.input[i] = alias[node.input[i]]

    kept = [n for n in model.graph.node if not (n.op_type == "Identity" and n.output[0] in alias)]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return len(alias)


def quantize_onnx_dynamic_int8(
    onnx_input_path: str, onnx_output_path: str, *, dealias_weights: bool = False
) -> dict[str, Any]:
    """
    Quantizes an ONNX model dynamically using ONNX Runtime for mobile deployment.

    The pointer head is excluded from quantization. Its logits are small and nearly uniform by
    design (a bilinear scorer over 64-d pooled vectors), so INT8 rounding on those few MatMuls
    moves the argmax far more than the encoder error it saves: measured 64-item gate sample,
    head-quantized INT8 agreed with FP32 on 81.25% of decisions with a 5.1 max logit delta, while
    the head in FP32 agrees on 100% (see scripts/verify_export_parity.py). The head holds 0.1% of
    the parameters, so the size win is unchanged.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    in_p = Path(onnx_input_path)
    out_p = Path(onnx_output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    loaded = onnx.load(str(in_p))
    removed = _dealias_identity_weights(loaded) if dealias_weights else 0
    if removed:
        in_p = out_p.with_suffix(".dealias.onnx")
        onnx.save(loaded, str(in_p))
    graph = loaded.graph
    excluded = [node.name for node in graph.node if "pointer_head" in node.name]
    print(
        f"[QUANT-ONNX] Quantizing {in_p} -> {out_p} "
        f"(QInt8, {len(excluded)} head nodes kept FP32, {removed} weight aliases de-aliased)..."
    )
    # per_channel weights: the encoder's attention projections lose too much to a single
    # per-tensor scale. Measured on a 64-item gate sample, nano keeps 81.25% argmax agreement
    # per-tensor vs 96.9% per-channel; the head lands in FP32 either way.
    quantize_dynamic(
        str(in_p),
        str(out_p),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=excluded,
        per_channel=True,
    )

    if removed:
        in_p.unlink(missing_ok=True)

    orig_mb = Path(onnx_input_path).stat().st_size / (1024 * 1024)
    quant_mb = out_p.stat().st_size / (1024 * 1024)
    print(
        f"✅ [QUANT-ONNX] Saved: {out_p} ({quant_mb:.2f} MB, {orig_mb / max(quant_mb, 0.01):.2f}x compression)"
    )

    return {"original_onnx_mb": orig_mb, "quantized_onnx_mb": quant_mb, "output_path": str(out_p)}


def quantize_saved_checkpoint(
    checkpoint_path: str,
    output_path: str,
    backbone_name: str = "huawei-noah/TinyBERT_General_4L_312D",
    model_type: str = "tinybert",
) -> dict[str, Any]:
    """
    Loads a trained float model, applies dynamic INT8 quantization, and saves the quantized weights.
    """
    from decision_engine.models import ModelFactory

    print(f"[QUANT] Loading float model ({model_type}) from {checkpoint_path}...")
    try:
        model = ModelFactory.create_from_name(model_type, backbone=backbone_name)
    except Exception:
        model = TamevTinyBertDecisionModel(backbone_name_or_path=backbone_name)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    orig_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    print(f"[QUANT] Original checkpoint size: {orig_size_mb:.2f} MB")

    _quantized_model, stats = quantize_tamev_model_int8(model, output_path=output_path)
    quant_size_mb = stats.get("file_size_mb", 0.0)

    print("✅ [QUANT] Dynamic INT8 Quantization completed!")
    print(f"   Original Size:  {orig_size_mb:.2f} MB")
    print(f"   Quantized Size: {quant_size_mb:.2f} MB")
    if orig_size_mb > 0:
        print(f"   Compression:    {orig_size_mb / max(quant_size_mb, 0.01):.2f}x reduction")

    return {
        "original_size_mb": orig_size_mb,
        "quantized_size_mb": quant_size_mb,
        "output_path": output_path,
    }
