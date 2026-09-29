# decision_engine/export/onnx_exporter.py
"""
ONNX Exporter for TAMEV Decision Models.
Supports dynamic candidate option count (K), dynamic sequence lengths,
and dynamic batch sizes for mobile deployment (CoreML / ONNX Runtime / LiteRT).
"""

from pathlib import Path

import torch
import torch.nn as nn

from decision_engine.models.tinybert_model import TamevTinyBertDecisionModel


class ExportableTamevWrapper(nn.Module):
    """
    Wraps TAMEV model so its forward output is a clean Tuple[torch.Tensor, torch.Tensor]
    (logits, probabilities) instead of a dictionary, ensuring seamless ONNX trace compatibility.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            ctx_input_ids=ctx_input_ids,
            ctx_attention_mask=ctx_attention_mask,
            opt_input_ids=opt_input_ids,
            opt_attention_mask=opt_attention_mask,
        )
        return out["logits"], out["probs"]


def get_dynamic_axes_config() -> dict[str, dict[int, str]]:
    """
    Returns dynamic axes configuration for ONNX export supporting
    dynamic batch sizes, dynamic sequence lengths, and dynamic option cardinalities.
    """
    return {
        "context_input_ids": {0: "batch_size", 1: "ctx_len"},
        "context_attention_mask": {0: "batch_size", 1: "ctx_len"},
        "option_input_ids": {0: "batch_size", 1: "num_options", 2: "opt_len"},
        "option_attention_mask": {0: "batch_size", 1: "num_options", 2: "opt_len"},
        "ctx_input_ids": {0: "batch_size", 1: "ctx_len"},
        "ctx_attention_mask": {0: "batch_size", 1: "ctx_len"},
        "opt_input_ids": {0: "batch_size", 1: "num_options", 2: "opt_len"},
        "opt_attention_mask": {0: "batch_size", 1: "num_options", 2: "opt_len"},
        "logits": {0: "batch_size", 1: "num_options"},
        "probs": {0: "batch_size", 1: "num_options"},
    }


def export_tinybert_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str,
    backbone_name: str = "huawei-noah/TinyBERT_General_4L_312D",
    opset_version: int = 17,
    model_type: str = "tinybert",
) -> Path:
    """
    Exports a trained decision model checkpoint to ONNX format.
    """
    from decision_engine.models import ModelFactory

    output_path = Path(output_onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Loading model architecture ({model_type}) with backbone {backbone_name}...")
    try:
        model = ModelFactory.create_from_name(model_type, backbone=backbone_name)
    except Exception:
        model = TamevTinyBertDecisionModel(backbone_name_or_path=backbone_name)

    if Path(checkpoint_path).exists():
        print(f"[EXPORT] Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        print(f"[WARN] Checkpoint {checkpoint_path} not found, exporting base initialized model.")

    model.eval()
    wrapped_model = ExportableTamevWrapper(model)

    # Dummy inputs for tracing: batch_size=1, ctx_len=32, num_options=4, opt_len=16
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

    print(f"[EXPORT] Exporting to ONNX at {output_path} (opset {opset_version})...")
    torch.onnx.export(
        wrapped_model,
        (dummy_ctx_ids, dummy_ctx_mask, dummy_opt_ids, dummy_opt_mask),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ [EXPORT] Success! ONNX model saved: {output_path} ({size_mb:.2f} MB)")
    return output_path
