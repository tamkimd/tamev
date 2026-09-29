"""Architecture: the decoupled pointer head, exact permutation invariance (zero drift) across
backbones, dynamic option-count cardinality, option-label bookkeeping under permutation, model
registry/factory, and the in-context causal option-isolation path.

This file holds the project's flagship guarantee: permuting the option order yields exactly the
(preserved) permutation of probabilities, at any option count K.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from decision_engine.config.base import ModelConfig
from decision_engine.models.base import BaseDecisionModel, ModelFactory, ModelRegistry
from decision_engine.models.pointer_head import DynamicPointerHeadRef, PointerHead
from decision_engine.models.tinybert_model import TamevTinyBertDecisionModel

TINYBERT = "huawei-noah/TinyBERT_General_4L_312D"
SMOLLM = "HuggingFaceTB/SmolLM2-135M"


# --------------------------------------------------------------------------------------
# Pointer head: shapes, normalisation and option-count cardinality
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 10, 20, 50, 77, 100, 255])
def test_numpy_pointer_head_shapes_and_normalisation(k):
    head = DynamicPointerHeadRef(hidden_dim=32, projection_dim=16)
    rng = np.random.default_rng(k)
    logits, probs = head.forward(rng.standard_normal((2, 32)), rng.standard_normal((2, k, 32)))

    assert logits.shape == (2, k)
    assert probs.shape == (2, k)
    assert not np.isnan(probs).any(), f"NaN detected at cardinality {k}"
    assert not np.isinf(probs).any(), f"Inf detected at cardinality {k}"
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(2), atol=1e-5)


@pytest.mark.parametrize("hidden,proj,temperature", [(64, 32, 1.0), (312, 64, 2.0)])
def test_torch_pointer_head_shapes_and_normalisation(hidden, proj, temperature):
    head = PointerHead(hidden_dim=hidden, projection_dim=proj, temperature=temperature)
    logits, probs = head(torch.randn(2, hidden), torch.randn(2, 4, hidden))

    assert logits.shape == (2, 4)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


# --------------------------------------------------------------------------------------
# Registry / factory across root architectures
# --------------------------------------------------------------------------------------


def test_model_registry_lists_all_root_models():
    assert {"encoder", "causal", "bi_encoder", "tinybert"} <= set(ModelRegistry.list_models())


@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("tinybert", ("tinybert", "encoder")),
        ("encoder", "encoder"),
        ("bi_encoder", "bi_encoder"),
        ("causal", "causal"),
    ],
)
def test_model_factory_create_from_name(name, expected_type):
    backbone = SMOLLM if name == "causal" else TINYBERT
    model = ModelFactory.create_from_name(name, backbone=backbone)
    assert isinstance(model, BaseDecisionModel)
    if isinstance(expected_type, tuple):
        assert model.model_type in expected_type
    else:
        assert model.model_type == expected_type


def test_encoder_model_instantiation_and_forward():
    cfg = ModelConfig(
        name="test_nano",
        model_type="encoder",
        backbone=TINYBERT,
        projection_dim=64,
        temperature=2.2,
    )
    model = ModelFactory.create(cfg)
    assert isinstance(model, BaseDecisionModel)
    assert model.hidden_dim == 312
    assert model.projection_dim == 64

    out = model(
        torch.ones(1, 16, dtype=torch.long),
        torch.ones(1, 16, dtype=torch.long),
        torch.ones(1, 4, 8, dtype=torch.long),
        torch.ones(1, 4, 8, dtype=torch.long),
        num_options=[4],
    )
    assert out["probs"].shape == (1, 4)
    assert torch.allclose(out["probs"].sum(), torch.tensor(1.0), atol=1e-4)


def test_temperature_calibration_interface():
    model = ModelFactory.create_from_name("tinybert")
    assert hasattr(model, "set_temperature")
    model.set_temperature(1.75)
    assert torch.isclose(model.temperature, torch.tensor(1.75))


# --------------------------------------------------------------------------------------
# TinyBERT decision model
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tinybert_model():
    model = TamevTinyBertDecisionModel(backbone_name_or_path=TINYBERT)
    model.eval()
    return model


def test_tinybert_backward_compatible_interface(tinybert_model):
    assert hasattr(tinybert_model, "pointer_head")
    assert hasattr(tinybert_model, "encode_context")
    assert hasattr(tinybert_model, "encode_options")


def test_tinybert_forward_shapes_and_normalisation(tinybert_model):
    b, l_ctx, k, l_opt = 2, 16, 4, 8
    out = tinybert_model(
        torch.randint(100, 2000, (b, l_ctx)),
        torch.ones((b, l_ctx), dtype=torch.long),
        torch.randint(100, 2000, (b, k, l_opt)),
        torch.ones((b, k, l_opt), dtype=torch.long),
    )
    assert out["logits"].shape == (b, k)
    assert out["probs"].shape == (b, k)
    torch.testing.assert_close(out["probs"].sum(dim=-1), torch.ones(b), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("k", [2, 3, 10, 77])
def test_tinybert_dynamic_option_cardinality(tinybert_model, k):
    with torch.no_grad():
        out = tinybert_model(
            torch.randint(100, 2000, (1, 10)),
            torch.ones((1, 10), dtype=torch.long),
            torch.randint(100, 2000, (1, k, 6)),
            torch.ones((1, k, 6), dtype=torch.long),
        )
    assert out["logits"].shape == (1, k)


def test_tinybert_permutation_equivariance(tinybert_model):
    b, l_ctx, k, l_opt = 1, 12, 4, 6
    ctx_ids = torch.randint(100, 2000, (b, l_ctx))
    ctx_mask = torch.ones((b, l_ctx), dtype=torch.long)
    opt_ids = torch.randint(100, 2000, (b, k, l_opt))
    opt_mask = torch.ones((b, k, l_opt), dtype=torch.long)

    with torch.no_grad():
        probs1 = tinybert_model(ctx_ids, ctx_mask, opt_ids, opt_mask)["probs"][0]

    perm = [2, 0, 3, 1]
    with torch.no_grad():
        probs2 = tinybert_model(ctx_ids, ctx_mask, opt_ids[:, perm, :], opt_mask[:, perm, :])[
            "probs"
        ][0]

    torch.testing.assert_close(probs2, probs1[perm], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------------------
# Flagship guard: exact zero-drift permutation invariance, any backbone
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["tinybert", "encoder", "bi_encoder"])
def test_exact_permutation_invariance_across_backbones(model_type):
    model = ModelFactory.create_from_name(model_type, backbone=TINYBERT)
    model.eval()

    ctx_ids = torch.randint(100, 1000, (1, 16))
    ctx_mask = torch.ones(1, 16, dtype=torch.long)
    opt_ids = torch.randint(100, 1000, (1, 4, 8))
    opt_mask = torch.ones(1, 4, 8, dtype=torch.long)

    with torch.no_grad():
        orig = model(ctx_ids, ctx_mask, opt_ids, opt_mask, num_options=[4])["probs"][0].numpy()
        perm = [2, 0, 3, 1]
        permuted = model(
            ctx_ids, ctx_mask, opt_ids[:, perm, :], opt_mask[:, perm, :], num_options=[4]
        )["probs"][0].numpy()

    # Align the permuted order back to the original: drift must be exactly zero.
    realigned = permuted[[perm.index(i) for i in range(4)]]
    assert np.max(np.abs(orig - realigned)) < 1e-7


# --------------------------------------------------------------------------------------
# In-context causal model + option isolation
# --------------------------------------------------------------------------------------


def test_incontext_causal_model_registration():
    from decision_engine.models.incontext_causal_model import InContextCausalDecisionModel

    assert ModelRegistry.get("incontext_causal") is InContextCausalDecisionModel


def test_incontext_causal_forward_pass():
    cfg = ModelConfig(
        name="test_incontext_smollm",
        model_type="incontext_causal",
        backbone=SMOLLM,
        projection_dim=64,
        temperature=2.0,
        freeze_backbone=True,
    )
    model = ModelFactory.create(cfg)
    result = model.predict_decision(
        "The user wants to book a flight from SFO to JFK on Friday.",
        "Which action should the system execute next?",
        [
            "Search flights from SFO to JFK on Friday",
            "Cancel the booking",
            "Send an email confirmation",
            "Ask user for passport details",
        ],
    )
    assert result["logits"].shape == (1, 4)
    assert result["probs"].shape == (1, 4)
    assert torch.isclose(
        result["probs"].sum(), torch.tensor(1.0, dtype=result["probs"].dtype), atol=1e-3
    )
    assert len(result["ranked"]) == 4


def test_incontext_causal_option_isolation_permutation_invariance():
    cfg = ModelConfig(
        name="test_incontext_isolation",
        model_type="incontext_causal",
        backbone=SMOLLM,
        projection_dim=64,
        temperature=2.0,
        freeze_backbone=True,
        option_isolation=True,
    )
    model = ModelFactory.create(cfg)

    state = "Current health is 15%, player has 1 health potion in inventory."
    question = "What is the optimal move?"
    res1 = model.predict_decision(
        state, question, ["Use health potion", "Attack boss with sword", "Flee combat"]
    )
    res2 = model.predict_decision(
        state, question, ["Attack boss with sword", "Use health potion", "Flee combat"]
    )

    prob_potion_1 = res1["probs"][0, 0].item()
    prob_potion_2 = res2["probs"][0, 1].item()
    tol = 5e-3 if model.load_dtype == torch.bfloat16 else 1e-4
    assert abs(prob_potion_1 - prob_potion_2) <= tol, (
        f"Option isolation drift detected: {prob_potion_1} vs {prob_potion_2} "
        f"(diff: {abs(prob_potion_1 - prob_potion_2)})"
    )


def test_incontext_causal_predict_proba():
    import numpy as np

    cfg = ModelConfig(
        name="test_incontext_proba",
        model_type="incontext_causal",
        backbone=SMOLLM,
        freeze_backbone=True,
    )
    model = ModelFactory.create(cfg)
    probs = model.predict_proba(
        "The robot is at position (3, 4) facing North.\nWhat should the robot do?",
        ["Move Forward", "Turn Left", "Turn Right", "Stop"],
    )
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 4
    assert np.isclose(probs.sum(), 1.0, atol=1e-3)


# --------------------------------------------------------------------------------------
# Option-label bookkeeping under permutation
# --------------------------------------------------------------------------------------


def test_permute_options_preserves_label_and_teacher_probs():
    from decision_engine.data.augmentor import permute_options
    from decision_engine.schema.decision import DecisionExample

    example = DecisionExample.from_dict(
        {
            "id": "1",
            "state": "Customer asks for billing change",
            "question": "Action?",
            "decision_type": "choice",
            "options": [
                {"id": "A", "text": "Option 1"},
                {"id": "B", "text": "Option 2"},
                {"id": "C", "text": "Option 3"},
            ],
            "label": "B",
            "teacher_probs": {"A": 0.1, "B": 0.7, "C": 0.2},
        }
    )

    # Force permutation [2, 0, 1] -> (C, A, B)
    permuted, _order = permute_options(example, permutation=[2, 0, 1])
    assert [o.id for o in permuted.options] == ["C", "A", "B"]
    assert permuted.label == "B"
    assert permuted.teacher_probs["B"] == 0.7
