"""Export: ONNX dynamic-axes/quantization specs, GGUF (fp16/fp32/q8_0 + metadata), the Apple
CoreML/MLX engines, the multi-format exporter (PyTorch/ONNX/TorchScript), the strategy registry,
and the honesty of the deployment manifest.

Optional deps (gguf, coremltools, mlx) skip cleanly rather than failing collection.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from decision_engine.config import ModelConfig
from decision_engine.export.apple_exporter import (
    MLXTamevDecisionRunner,
    export_coreml,
    export_mlx,
)
from decision_engine.export.gguf_exporter import export_gguf
from decision_engine.export.multi_format_exporter import MultiFormatExporter
from decision_engine.export.onnx_exporter import get_dynamic_axes_config
from decision_engine.export.quantizer import get_quantization_spec
from decision_engine.export.strategies import ExportStrategyRegistry
from decision_engine.models import ModelFactory

TINYBERT = "huawei-noah/TinyBERT_General_4L_312D"


def _model(size_class: str, temperature: float):
    return ModelFactory.create(
        ModelConfig(
            model_type="encoder", backbone=TINYBERT, size_class=size_class, temperature=temperature
        )
    )


@pytest.fixture(scope="module")
def gguf():
    return pytest.importorskip("gguf", reason="gguf is optional")


@pytest.fixture(scope="module")
def nano_model():
    return _model("nano", 1.05)


@pytest.fixture(scope="module")
def micro_model():
    return _model("micro", 1.0)


@pytest.fixture(scope="module")
def apple_model():
    return _model("micro", 1.8)


@pytest.fixture(scope="module")
def mlx():
    pytest.importorskip("coremltools", reason="coremltools is optional")
    return pytest.importorskip("mlx.core", reason="mlx is optional")


# --------------------------------------------------------------------------------------
# ONNX specs
# --------------------------------------------------------------------------------------


def test_onnx_dynamic_axes_config():
    axes = get_dynamic_axes_config()
    assert axes["context_input_ids"][0] == "batch_size"
    assert axes["option_input_ids"][1] == "num_options"
    assert axes["probs"][1] == "num_options"


def test_onnx_quantization_spec():
    spec = get_quantization_spec()
    assert spec["primary_mobile_target"] == "INT8"
    assert "INT8" in spec["precision_candidates"]


# --------------------------------------------------------------------------------------
# GGUF
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["fp16", "fp32", "q8_0"])
def test_export_gguf_precisions(gguf, nano_model, tmp_path, precision):
    out_file = tmp_path / f"tinybert_{precision}.gguf"
    exported = export_gguf(
        model=nano_model,
        output_path=out_file,
        model_name=f"test_nano_{precision}",
        size_class="nano",
        precision=precision,
    )
    assert exported.exists()
    assert exported.suffix == ".gguf"
    assert exported.stat().st_size > 0
    assert len(gguf.GGUFReader(exported).tensors) > 0


def test_export_gguf_metadata(gguf, nano_model, tmp_path):
    exported = export_gguf(
        model=nano_model,
        output_path=tmp_path / "tinybert_fp16.gguf",
        model_name="test_nano_fp16",
        size_class="nano",
        precision="fp16",
    )
    reader = gguf.GGUFReader(exported)
    fields = dict(reader.fields)
    assert {
        "general.architecture",
        "tamev.active_parameters",
        "tamev.permutation_invariance",
    } <= set(fields)
    active = reader.fields["tamev.active_parameters"].parts[-1][0]
    assert active == sum(p.numel() for p in nano_model.parameters())


def test_multi_format_exporter_gguf_integration(gguf, nano_model, tmp_path):
    exporter = MultiFormatExporter(model=nano_model, model_name="test_tinybert", size_class="nano")
    out_file = tmp_path / "model.gguf"
    res = exporter.export_gguf(output_path=out_file, precision="fp16")

    assert res["format"] == "gguf"
    assert res["file"] == "model.gguf"
    assert res["size_mb"] > 0
    assert "sha256" in res
    assert out_file.exists()


@pytest.mark.parametrize("alias", ["gguf", "llama.cpp", "ggml"])
def test_export_strategy_registry_resolves_gguf_aliases(alias):
    assert ExportStrategyRegistry.get(alias).format_name == "gguf"


def test_gguf_architecture_registry(gguf):
    from decision_engine.export.gguf_exporter import GGUFArchitectureRegistry

    GGUFArchitectureRegistry.register("custom_dec_engine", gguf.constants.MODEL_ARCH.LLAMA)
    assert (
        GGUFArchitectureRegistry._HF_TYPE_MAP["custom_dec_engine"]
        == gguf.constants.MODEL_ARCH.LLAMA
    )

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"model_type": "modernbert"})()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DummyBackbone()

    assert GGUFArchitectureRegistry.resolve_architecture(DummyModel()) == "modern-bert"


def test_gguf_tensor_processor_quantisation_rules(gguf):
    from decision_engine.export.gguf_exporter import GGUFTensorProcessor

    q8 = gguf.quants.GGMLQuantizationType.Q8_0

    # 1D bias stays F16 (unquantized) under a Q8_0 target.
    q_bias, raw_dt_bias = GGUFTensorProcessor.prepare_tensor(
        np.random.randn(64).astype(np.float32), q8
    )
    assert raw_dt_bias is None
    assert q_bias.dtype == np.float16

    # 2D weight matrix is quantized.
    q_weight, raw_dt_weight = GGUFTensorProcessor.prepare_tensor(
        np.random.randn(32, 64).astype(np.float32), q8
    )
    assert raw_dt_weight == q8
    assert q_weight.shape == (32, 68)

    # Integer tensors are preserved, widened to int32.
    q_int, dt_int = GGUFTensorProcessor.prepare_tensor(np.array([1, 2, 3, 4], dtype=np.int64), q8)
    assert dt_int is None
    assert q_int.dtype == np.int32


# --------------------------------------------------------------------------------------
# Apple: CoreML + MLX
# --------------------------------------------------------------------------------------


def test_export_coreml_routine(mlx, apple_model, tmp_path):
    import coremltools as ct

    out_dir = tmp_path / "coreml_export"
    mlpackage_path = export_coreml(
        model=apple_model,
        output_dir=out_dir,
        model_name="test_tinybert_coreml",
        precision="fp16",
    )
    assert mlpackage_path.exists()
    assert mlpackage_path.suffix == ".mlpackage"
    assert mlpackage_path.name == "test_tinybert_coreml.mlpackage"
    assert (mlpackage_path / "Manifest.json").exists() or (mlpackage_path / "Data").exists()

    mlmodel = ct.models.MLModel(str(mlpackage_path))
    assert len(mlmodel.input_description) > 0
    assert "logits" in mlmodel.output_description
    assert "probs" in mlmodel.output_description

    if "context_vec" in mlmodel.input_description:
        hidden_dim = apple_model.pointer_head.hidden_dim
        inputs = {
            "context_vec": np.random.randn(1, hidden_dim).astype(np.float32),
            "option_vecs": np.random.randn(1, 4, hidden_dim).astype(np.float32),
        }
        if "num_options" in mlmodel.input_description:
            inputs["num_options"] = np.array([4], dtype=np.int32)
        preds = mlmodel.predict(inputs)
        assert preds["logits"].shape == (1, 4)
        assert np.isclose(np.sum(preds["probs"]), 1.0, atol=1e-3)


def test_export_mlx_routine(mlx, apple_model, tmp_path):
    mx = mlx
    mlx_dir = export_mlx(
        model=apple_model, output_dir=tmp_path / "mlx_export", model_name="test_tinybert_mlx"
    )

    npz_file = mlx_dir / "weights.npz"
    st_file = mlx_dir / "weights.safetensors"
    cfg_file = mlx_dir / "tamev_mlx_config.json"
    assert npz_file.exists() and st_file.exists() and cfg_file.exists()

    with open(cfg_file, encoding="utf-8") as f:
        mlx_cfg = json.load(f)
    assert mlx_cfg["model_name"] == "test_tinybert_mlx"
    assert mlx_cfg["architecture_type"] in ("encoder", "tinybert", "EncoderDecisionModel")
    assert mlx_cfg["temperature"] == pytest.approx(1.8, abs=0.01)
    assert "vocab_size" in mlx_cfg["vocabulary"]
    assert mlx_cfg["runtime"] == "mlx"
    assert mlx_cfg["target"] == "apple_silicon_unified_memory"

    loaded_npz = mx.load(str(npz_file))
    assert "pointer_head.q_proj.weight" in loaded_npz
    assert "pointer_head.k_proj.weight" in loaded_npz
    loaded_st = mx.load(str(st_file))
    assert len(loaded_st) == len(loaded_npz)
    assert "pointer_head.q_proj.weight" in loaded_st

    runner = MLXTamevDecisionRunner(mlx_dir)
    assert runner.temperature == pytest.approx(1.8, abs=0.01)

    hidden_dim = apple_model.pointer_head.hidden_dim
    logits, probs = runner.score_pointer(
        mx.random.normal((1, hidden_dim)), mx.random.normal((1, 4, hidden_dim))
    )
    mx.eval(logits, probs)
    assert logits.shape == (1, 4)
    assert probs.shape == (1, 4)
    assert np.isclose(float(mx.sum(probs).item()), 1.0, atol=1e-5)


def test_multi_format_exporter_apple_integration(mlx, apple_model, tmp_path):
    out_dir = tmp_path / "multi_apple_exported"
    exporter = MultiFormatExporter(
        model=apple_model, model_name="test_multi_apple", size_class="micro"
    )
    manifest = exporter.export_all(output_dir=out_dir, formats=["coreml", "mlx"])

    coreml_art = manifest["artifacts"]["coreml"]
    assert coreml_art["format"] == "coreml"
    assert coreml_art["file"] == "test_multi_apple.mlpackage"
    assert len(coreml_art["sha256"]) == 64
    assert coreml_art["size_mb"] > 0

    mlx_art = manifest["artifacts"]["mlx"]
    assert mlx_art["format"] == "mlx"
    assert len(mlx_art["sha256"]) == 64
    assert mlx_art["size_mb"] > 0

    assert (out_dir / "test_multi_apple.mlpackage").exists()
    assert (out_dir / "weights.npz").exists()
    assert (out_dir / "weights.safetensors").exists()
    assert (out_dir / "tamev_mlx_config.json").exists()
    assert (out_dir / "deployment_manifest.json").exists()


# --------------------------------------------------------------------------------------
# Multi-format: PyTorch, ONNX, TorchScript, manifest honesty
# --------------------------------------------------------------------------------------


def test_export_pytorch_formats(micro_model, tmp_path):
    out_dir = tmp_path / "exported"
    exporter = MultiFormatExporter(
        micro_model, model_name="test_micro_tinybert", size_class="micro"
    )
    manifest = exporter.export_all(
        output_dir=out_dir, formats=["pytorch_fp32", "pytorch_bf16", "pytorch_int8", "pytorch_mps"]
    )
    assert "artifacts" in manifest
    assert (out_dir / "test_micro_tinybert_fp32.pt").exists()
    assert (out_dir / "test_micro_tinybert_bf16.pt").exists()
    assert (out_dir / "test_micro_tinybert_int8.pt").exists()
    assert (out_dir / "test_micro_tinybert_mps_fp16.pt").exists()

    with open(out_dir / "deployment_manifest.json") as f:
        data = json.load(f)
    assert data["model_name"] == "test_micro_tinybert"
    assert data["size_class"] == "micro"
    assert {"pytorch_fp32", "pytorch_bf16", "pytorch_int8", "pytorch_mps"} <= set(data["artifacts"])
    assert data["artifacts"]["pytorch_mps"]["sha256"] != ""


def test_export_onnx_formats(micro_model, tmp_path):
    out_dir = tmp_path / "exported_onnx"
    exporter = MultiFormatExporter(micro_model, model_name="test_micro_onnx", size_class="micro")
    manifest = exporter.export_all(output_dir=out_dir, formats=["onnx_fp32", "onnx_int8"])
    assert "artifacts" in manifest
    assert (out_dir / "test_micro_onnx.onnx").exists()

    with open(out_dir / "deployment_manifest.json") as f:
        data = json.load(f)
    assert {"onnx_fp32", "onnx_int8"} <= set(data["artifacts"])


def test_export_torchscript_format(nano_model, tmp_path):
    out_dir = tmp_path / "exported_ts"
    exporter = MultiFormatExporter(nano_model, model_name="test_nano_ts", size_class="nano")
    manifest = exporter.export_all(output_dir=out_dir, formats=["torchscript"])
    assert "artifacts" in manifest

    loaded = torch.jit.load(str(out_dir / "test_nano_ts.torchscript"))
    logits, probs = loaded(
        torch.ones((1, 32), dtype=torch.long),
        torch.ones((1, 32), dtype=torch.long),
        torch.ones((1, 4, 16), dtype=torch.long),
        torch.ones((1, 4, 16), dtype=torch.long),
    )
    assert logits.shape == (1, 4)
    assert probs.shape == (1, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-4)


def test_manifest_labels_targets_and_invariance_honestly(micro_model, tmp_path):
    """The manifest must not publish unvalidated targets as measured SLAs, and must describe
    permutation behaviour per architecture instead of a global guarantee."""
    exporter = MultiFormatExporter(micro_model, model_name="test_manifest_meta", size_class="micro")
    manifest = exporter.export_all(output_dir=tmp_path / "manifest", formats=["pytorch_fp32"])

    assert "latency_sla" not in manifest
    assert manifest["latency_targets_ms"] == {"p50_ms": 4.5, "p95_ms": 8.0}
    assert "not measured" in manifest["latency_targets_note"]
    assert manifest["permutation_invariance"].startswith("exact")

    class _StubCausal(torch.nn.Module):
        model_type = "incontext_causal"

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

    causal = MultiFormatExporter(
        _StubCausal(), model_name="test_manifest_causal", size_class="medium_qwen35_08b"
    )
    causal_manifest = causal.export_all(output_dir=tmp_path / "causal", formats=["pytorch_fp32"])
    assert causal_manifest["permutation_invariance"].startswith("order_sensitive")
