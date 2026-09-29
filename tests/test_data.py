"""Data: the canonical DecisionItem schema and the legacy DecisionExample schema, the Kev/Jev v7
adapter, dataset parsing, the collator (context/teacher masks, cell labels), the quality filter,
and per-domain metric aggregation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from decision_engine.data.collator import TamevCollator, _cell_label
from decision_engine.data.dataset import TamevDataset
from decision_engine.data.quality_filter import DataQualityFilter
from decision_engine.data.schema import DecisionItem, OptionItem, build_context
from decision_engine.data.typesafe_adapter import parse_raw_sample_to_items
from decision_engine.schema.decision import DecisionExample, DecisionType
from decision_engine.training.trainer import aggregate_domain_metrics

TINYBERT = "huawei-noah/TinyBERT_General_4L_312D"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TINYBERT)


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------


def test_decision_item_schema_round_trip():
    item = DecisionItem(
        id="sample_01",
        state="Tetris column 9 is empty",
        question="Which move?",
        options=[
            OptionItem(id="drop", text="Drop I-bar"),
            OptionItem(id="shift", text="Shift left"),
        ],
        label="drop",
        teacher_probs={"drop": 0.8, "shift": 0.2},
    )
    assert item.target_index == 0
    assert len(item.options) == 2
    dumped = item.to_dict()
    assert dumped["label"] == "drop"
    assert DecisionItem.from_dict(dumped).id == "sample_01"


def test_decision_example_choice_round_trip():
    payload = {
        "id": "dec_001",
        "state": "User requested account refund due to service downtime.",
        "question": "What is the recommended support action?",
        "decision_type": "choice",
        "options": [
            {"id": "A", "text": "Issue full refund immediately"},
            {"id": "B", "text": "Investigate telemetry logs first"},
            {"id": "C", "text": "Reject request as out of policy"},
        ],
        "label": "B",
        "teacher_probs": {"A": 0.15, "B": 0.80, "C": 0.05},
        "metadata": {"source": "native", "domain": "support", "num_options": 3},
    }
    example = DecisionExample.from_dict(payload)
    assert example.id == "dec_001"
    assert example.decision_type is DecisionType.CHOICE
    assert len(example.options) == 3
    assert example.label == "B"
    assert example.teacher_probs["B"] == 0.80

    rebuilt = DecisionExample.from_dict(example.to_dict())
    assert rebuilt.id == example.id
    assert rebuilt.label == example.label


def test_decision_example_rejects_unknown_label():
    payload = {
        "id": "dec_002",
        "state": "Order pending delivery.",
        "question": "Select status",
        "decision_type": "choice",
        "options": [{"id": "A", "text": "Shipped"}, {"id": "B", "text": "Delayed"}],
        "label": "Z",
    }
    with pytest.raises(ValueError):
        DecisionExample.from_dict(payload)


@pytest.mark.parametrize(
    "payload,expected_type,expected_label",
    [
        pytest.param(
            {
                "id": "dec_003",
                "state": "Customer feedback sentiment",
                "question": "Rate satisfaction from 1 to 5",
                "decision_type": "score",
                "scale": {"min": 1, "max": 5},
                "label": 4,
            },
            DecisionType.SCORE,
            4,
            id="score",
        ),
        pytest.param(
            {
                "id": "dec_004",
                "state": "Should the system execute autonomous trade?",
                "question": "Risk within threshold?",
                "decision_type": "noul",
                "label": True,
            },
            DecisionType.NOUL,
            True,
            id="noul",
        ),
    ],
)
def test_decision_example_score_and_noul(payload, expected_type, expected_label):
    example = DecisionExample.from_dict(payload)
    assert example.decision_type is expected_type
    assert example.label == expected_label


# --------------------------------------------------------------------------------------
# Quality filter
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "teacher_probs,passed,to_hard_pool",
    [
        pytest.param({"A": 0.85, "B": 0.15}, True, False, id="confident-passes"),
        pytest.param({"A": 0.35, "B": 0.35, "C": 0.30}, False, True, id="ambiguous-hard-pool"),
    ],
)
def test_quality_filter_flags_ambiguous_examples(teacher_probs, passed, to_hard_pool):
    qfilter = DataQualityFilter(min_teacher_confidence=0.5, max_context_chars=1000)
    example = DecisionExample.from_dict(
        {
            "id": "ex",
            "state": "Short context description",
            "question": "What next?",
            "decision_type": "choice",
            "options": [{"id": oid, "text": f"Option {oid}"} for oid in teacher_probs],
            "label": "A",
            "teacher_probs": teacher_probs,
        }
    )
    res = qfilter.evaluate(example)
    assert res.passed is passed
    assert res.to_hard_pool is to_hard_pool


# --------------------------------------------------------------------------------------
# v7 adapter + explicit option precedence
# --------------------------------------------------------------------------------------


def test_typesafe_v7_adapter():
    v7_raw = {
        "state": "User requested account refund.",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Determine user intent",
                "criteria": {"refund": "Refund request", "inquiry": "General question"},
                "label": "refund",
            }
        },
    }
    items = parse_raw_sample_to_items(v7_raw)
    assert len(items) == 1
    assert items[0].label == "refund"
    assert len(items[0].options) == 2
    assert items[0].target_index == 0


def test_score_rows_prefer_explicit_ordinal_options():
    """A `score` row carrying its own options keeps them; the `Rating i` template is only a fallback."""
    grounded = {
        "id": "test/sst5/abc_sentiment",
        "state": "trivial where it should be profound .",
        "question": "What is the sentiment of this review sentence?",
        "decision_type": "score",
        "scale": {"min": 0, "max": 4},
        "label": 0,
        "options": [
            {"id": "0", "text": "very negative"},
            {"id": "1", "text": "negative"},
            {"id": "2", "text": "neutral"},
            {"id": "3", "text": "positive"},
            {"id": "4", "text": "very positive"},
        ],
    }
    item = DecisionItem.from_dict(grounded)
    assert [o.text for o in item.options] == [
        "very negative",
        "negative",
        "neutral",
        "positive",
        "very positive",
    ]
    assert item.target_index == 0 and item.label == "0"

    legacy = {k: v for k, v in grounded.items() if k != "options"} | {"options": None}
    legacy_item = DecisionItem.from_dict(legacy)
    assert [o.text for o in legacy_item.options] == [f"Rating {i}" for i in range(5)]
    assert legacy_item.target_index == 0


def test_noul_rows_prefer_explicit_options():
    """A `noul` row carrying its own options keeps them; `no:/yes:` synthesis is only a fallback."""
    explicit = {
        "id": "test/agnews/abc_is_world",
        "state": "Wall St rallies on jobs data .",
        "question": "Is this news article about world politics?",
        "decision_type": "noul",
        "label": "false",
        "options": [{"id": "false", "text": "No / False"}, {"id": "true", "text": "Yes / True"}],
    }
    item = DecisionItem.from_dict(explicit)
    assert [o.text for o in item.options] == ["No / False", "Yes / True"]
    assert item.target_index == 0

    legacy = {k: v for k, v in explicit.items() if k != "options"} | {"options": None}
    legacy_item = DecisionItem.from_dict(legacy)
    assert [o.text for o in legacy_item.options] == [
        "no: Is this news article about world politics?",
        "yes: Is this news article about world politics?",
    ]
    assert legacy_item.target_index == 0


# --------------------------------------------------------------------------------------
# Dataset + collator
# --------------------------------------------------------------------------------------


def test_tamev_dataset_and_collator(tmp_path, tokenizer):
    sample_file = tmp_path / "data.jsonl"
    data = [
        {
            "id": "1",
            "state": "State 1",
            "question": "Question 1",
            "options": [{"id": "a", "text": "Opt A"}, {"id": "b", "text": "Opt B"}],
            "label": "a",
            "teacher_probs": {"a": 0.9, "b": 0.1},
        },
        {
            "id": "2",
            "state": "State 2",
            "question": "Question 2",
            "options": [
                {"id": "x", "text": "Opt X"},
                {"id": "y", "text": "Opt Y"},
                {"id": "z", "text": "Opt Z"},
            ],
            "label": "z",
            "teacher_probs": {"x": 0.1, "y": 0.2, "z": 0.7},
        },
    ]
    with open(sample_file, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

    dataset = TamevDataset(sample_file)
    assert len(dataset) == 2

    batch = TamevCollator(tokenizer=tokenizer, max_ctx_len=32, max_opt_len=16)(
        [dataset[0], dataset[1]]
    )
    assert {"ctx_input_ids", "ctx_attention_mask", "opt_input_ids", "opt_attention_mask"} <= set(
        batch
    )
    assert "labels" in batch and "teacher_probs" in batch
    assert batch["ctx_input_ids"].shape[0] == 2
    assert batch["opt_input_ids"].shape[1] == 3  # padded to max K in batch
    assert batch["labels"].tolist() == [0, 2]


def test_noul_and_score_dataset_ingestion(tmp_path):
    sample_file = tmp_path / "noul_score.jsonl"
    data = [
        {
            "id": "imdb/1",
            "state": "Great movie with fantastic acting.",
            "question": "Is this movie review positive?",
            "decision_type": "noul",
            "options": None,
            "scale": None,
            "label": True,
            "teacher_probs": None,
        },
        {
            "id": "imdb/2",
            "state": "Terrible acting and boring plot.",
            "question": "Is this movie review positive?",
            "decision_type": "noul",
            "options": None,
            "scale": None,
            "label": False,
            "teacher_probs": None,
        },
        {
            "id": "sst5/1",
            "state": "The cinematography was average.",
            "question": "What is the sentiment rating?",
            "decision_type": "score",
            "options": None,
            "scale": {"min": 0, "max": 4},
            "label": 2,
            "teacher_probs": None,
        },
    ]
    with open(sample_file, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

    dataset = TamevDataset(sample_file)
    assert len(dataset) == 3
    assert [o.id for o in dataset[0].options] == ["false", "true"]
    assert dataset[0].label == "true" and dataset[0].target_index == 1
    assert dataset[1].label == "false" and dataset[1].target_index == 0
    assert [o.id for o in dataset[2].options] == ["0", "1", "2", "3", "4"]
    assert dataset[2].label == "2" and dataset[2].target_index == 2


def test_collator_context_and_teacher_mask(tokenizer):
    collator = TamevCollator(tokenizer=tokenizer, max_ctx_len=64, max_opt_len=16)

    # Word-shaped, not "A" * n: a char run tokenizes to a single [UNK] and would prove nothing.
    long_state = "alpha beta gamma delta " * 40
    items = [
        DecisionItem(
            id="1",
            state=long_state,
            question="Target Question?",
            options=[OptionItem("a", "Yes"), OptionItem("b", "No")],
            label="a",
            teacher_probs=None,
        ),
        DecisionItem(
            id="2",
            state="Short state",
            question="Is this real?",
            options=[OptionItem("x", "True"), OptionItem("y", "False")],
            label="x",
            teacher_probs={"x": 0.85, "y": 0.15},
        ),
    ]

    batch = collator(items)
    assert "is this real" in tokenizer.decode(batch["ctx_input_ids"][1]).lower()
    assert "target question" not in tokenizer.decode(batch["ctx_input_ids"][0]).lower()

    # `max_ctx_len` alone must bound the context, not the collision-time `state[:200]` cap.
    wide = TamevCollator(tokenizer=tokenizer, max_ctx_len=512, max_opt_len=16)([items[0]])
    reached = int(wide["ctx_attention_mask"][0].sum())
    capped = len(
        tokenizer(build_context(long_state[:200], items[0].question), truncation=True)["input_ids"]
    )
    assert reached > capped, f"context still limited to the first 200 chars ({reached} <= {capped})"

    # Byte-for-byte the serving string, per row.
    for idx, item in enumerate(items):
        ref = tokenizer(build_context(item.state, item.question), max_length=64, truncation=True)
        assert tokenizer.convert_ids_to_tokens(ref["input_ids"]) == tokenizer.convert_ids_to_tokens(
            batch["ctx_input_ids"][idx][: len(ref["input_ids"])].tolist()
        )

    assert batch["has_teacher"][0].item() is False
    assert batch["has_teacher"][1].item() is True


def test_cell_label_survives_a_metadata_free_row(tokenizer):
    """Corpora carry `source`/`qid` at the top level and no `metadata` key; if the parse drops
    them, every "<source>/<qid>"-keyed feature (cell_weights, per-cell breakdown) matches nothing."""
    raw = {
        "id": "train/yelp/abc_rating",
        "source": "yelp",
        "qid": "rating",
        "state": "the food was great",
        "question": "What is the rating?",
        "label": "4",
        "decision_type": "score",
        "scale": {"min": 0, "max": 4},
        "options": None,
    }
    item = parse_raw_sample_to_items(raw)[0]
    assert _cell_label(item) == "yelp/rating"
    assert TamevCollator(tokenizer=tokenizer, max_ctx_len=16, max_opt_len=8)([item])["cells"] == [
        "yelp/rating"
    ]

    # An explicit metadata block still wins for the keys it defines.
    item = parse_raw_sample_to_items({**raw, "metadata": {"source": "renamed"}})[0]
    assert _cell_label(item) == "renamed/rating"


# --------------------------------------------------------------------------------------
# Per-domain metrics
# --------------------------------------------------------------------------------------


def test_aggregate_domain_metrics():
    probs = [np.array([0.8, 0.2]), np.array([0.1, 0.9]), np.array([0.3, 0.7])]
    targets = [0, 1, 0]
    tasks = ["banking77", "banking77", "ag_news"]

    breakdown = aggregate_domain_metrics(probs, targets, tasks)
    assert breakdown["banking77"]["top1_accuracy"] == 1.0
    assert breakdown["banking77"]["n"] == 2
    assert breakdown["ag_news"]["top1_accuracy"] == 0.0
    assert breakdown["ag_news"]["n"] == 1
