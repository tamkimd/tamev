"""Serving: the TypeSafe decision engine and Flyweight cache, question handlers, the SDK client,
the CLI + console-output surface, and end-to-end real-world decision quality / permutation
invariance."""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest
import torch

from decision_engine.server.cache import LRUEmbeddingCache
from decision_engine.server.engine import TamevEngine
from decision_engine.server.handlers import (
    BaseQuestionHandler,
    ChoiceQuestionHandler,
    NoulQuestionHandler,
    QuestionHandlerRegistry,
    ScoreQuestionHandler,
)
from decision_engine.server.typesafe_server import TamevEngine as TypeSafeEngine
from tamev import Choice, Noul, Score, SystemOneResponse
from tamev.cli import create_parser, main
from tamev.typesafe import (
    TypeSafeDirectClient,
    choice_confidence,
    score_confidence,
    validate_choice_criteria,
)
from tests.real_cases_data import REAL_CASES

# --------------------------------------------------------------------------------------
# Flyweight embedding cache
# --------------------------------------------------------------------------------------


def test_lru_cache_put_get_hit_miss_and_detach():
    cache = LRUEmbeddingCache(maxsize=3)
    assert cache.hits == 0 and cache.misses == 0
    assert cache.hit_rate == 0.0

    t1 = torch.randn(1, 16)
    assert cache.get("key1") is None
    assert cache.misses == 1

    cache.put("key1", t1)
    res1 = cache.get("key1")
    assert res1 is not None
    assert torch.allclose(res1, t1)
    assert cache.hits == 1
    assert cache.hit_rate == 0.5  # 1 hit out of 2 accesses

    # Cached value is a detached clone: mutating the source must not change it.
    t1[0, 0] += 10.0
    assert not torch.allclose(cache.get("key1"), t1)


def test_lru_cache_evicts_least_recently_used():
    cache = LRUEmbeddingCache(maxsize=2)
    cache.put("k1", torch.tensor([1.0]))
    cache.put("k2", torch.tensor([2.0]))
    _ = cache.get("k1")  # k1 becomes most recently used

    cache.put("k3", torch.tensor([3.0]))
    assert cache.get("k2") is None  # evicted
    assert cache.get("k1") is not None
    assert cache.get("k3") is not None


def test_lru_cache_clear_resets_counters():
    cache = LRUEmbeddingCache(maxsize=10)
    cache.put("k1", torch.randn(2, 2))
    _ = cache.get("k1")
    assert cache.hits == 1
    assert len(cache._cache) == 1

    cache.clear()
    assert cache.hits == 0 and cache.misses == 0
    assert len(cache._cache) == 0


# --------------------------------------------------------------------------------------
# Question handler strategy / registry
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qtype,handler_cls",
    [
        pytest.param("choice", ChoiceQuestionHandler, id="choice"),
        pytest.param("noul", NoulQuestionHandler, id="noul"),
        pytest.param("score", ScoreQuestionHandler, id="score"),
        pytest.param("CHOICE", ChoiceQuestionHandler, id="case-insensitive"),
        pytest.param("NoUl", NoulQuestionHandler, id="case-insensitive-noul"),
        pytest.param("unknown_future_type", ChoiceQuestionHandler, id="fallback"),
    ],
)
def test_question_handler_registry(qtype, handler_cls):
    handler = QuestionHandlerRegistry.get(qtype)
    assert isinstance(handler, handler_cls)
    if qtype in ("choice", "noul", "score"):
        assert handler.question_type == qtype


def test_custom_handler_registration():
    class PriorityQuestionHandler(BaseQuestionHandler):
        question_type = "priority"

        def handle(self, engine, state_str, instructions, criteria):
            return {"type": "priority", "selected_level": "P0", "confidence": 0.99}

    QuestionHandlerRegistry.register(PriorityQuestionHandler())
    handler = QuestionHandlerRegistry.get("priority")
    assert isinstance(handler, PriorityQuestionHandler)

    res = handler.handle(None, "Database is down", "Assign priority", {})
    assert res["type"] == "priority"
    assert res["selected_level"] == "P0"


# --------------------------------------------------------------------------------------
# Engine Flyweight caching
# --------------------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = TamevEngine(
        model_type="encoder",
        backbone="huawei-noah/TinyBERT_General_4L_312D",
        device="cpu",
        enable_cache=True,
    )
    eng.clear_cache()
    return eng


def test_engine_option_cache_hit_and_zero_drift(engine):
    criteria = {
        "attack": "Direct physical assault",
        "defend": "Fortify current position",
        "retreat": "Tactical backward withdrawal",
    }
    state, instructions = (
        "Enemy forces approaching base from the east",
        "Select optimal squad action",
    )

    initial_misses = engine.option_cache.misses
    res1 = engine.predict_choice(state, instructions, criteria)
    assert engine.option_cache.misses > initial_misses
    hits_after_first = engine.option_cache.hits

    _ = engine.predict_choice("A different battle scenario state", instructions, criteria)
    assert engine.option_cache.hits > hits_after_first

    res1_again = engine.predict_choice(state, instructions, criteria)
    for k in criteria:
        assert np.isclose(res1["probabilities"][k], res1_again["probabilities"][k], atol=1e-5)


def test_engine_context_cache_shared_across_batch(engine):
    engine.clear_cache()
    state = "User requests account cancellation due to pricing"
    questions = {
        "action": {
            "type": "choice",
            "instructions": "Determine retention strategy",
            "criteria": {"discount": "Offer 20% discount", "cancel": "Proceed with cancellation"},
        },
        "is_churn_risk": {
            "type": "noul",
            "instructions": "Is the user an immediate churn risk?",
        },
        "satisfaction_level": {
            "type": "score",
            "instructions": "Estimate customer sentiment on 3-point scale",
            "criteria": ["Extremely unhappy", "Neutral / indifferent", "Satisfied"],
        },
    }

    res = engine.predict_batch(state, questions)
    assert res["answers"]["action"]["type"] == "choice"
    assert res["answers"]["is_churn_risk"]["type"] == "noul"
    assert res["answers"]["satisfaction_level"]["type"] == "score"

    stats = engine.cache_stats
    assert stats["option_cache"]["size"] >= 3
    assert stats["context_cache"]["size"] >= 1


def test_engine_cache_stats_observability(engine):
    stats = engine.cache_stats
    assert {"option_cache", "context_cache"} <= set(stats)
    assert {"hits", "misses", "hit_rate", "size", "maxsize"} <= set(stats["option_cache"])


# --------------------------------------------------------------------------------------
# TypeSafe SDK / client
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probs,expected",
    [
        pytest.param([1 / 3, 1 / 3, 1 / 3], 0.0, id="uniform-three"),
        pytest.param([1.0, 0.0, 0.0], 1.0, id="certain"),
        pytest.param([1.0], 1.0, id="single-option"),
    ],
)
def test_choice_confidence(probs, expected):
    assert choice_confidence(probs) == expected


def test_score_confidence():
    assert score_confidence([0.0, 1.0, 0.0]) == 1.0


def test_validate_choice_criteria():
    assert validate_choice_criteria({"a": "alpha", "b": "beta"}) is True
    with pytest.raises(ValueError):
        validate_choice_criteria({})


def test_typesafe_direct_client_choice_noul_score():
    client = TypeSafeDirectClient(backbone="huawei-noah/TinyBERT_General_4L_312D")

    res = client.system_one(
        state="Tetris board with open space at column 0",
        questions={
            "next_move": Choice(
                instructions="Select best move",
                criteria={"left": "Shift left", "drop": "Hard drop"},
            )
        },
    )
    assert isinstance(res, SystemOneResponse)
    ans = res.answers["next_move"]
    assert ans.type == "choice"
    assert ans.choice in ["left", "drop"]
    assert 0.0 <= ans.confidence <= 1.0
    assert sum(ans.probabilities.values()) == pytest.approx(1.0, abs=1e-3)

    res_noul = client.system_one(
        state="The stack height is 18 out of 20.",
        questions={
            "danger": Noul(instructions="Is the player in immediate danger of topping out?")
        },
    )
    ans_noul = res_noul.answers["danger"]
    assert ans_noul.type == "noul"
    assert 0.0 <= ans_noul.noul <= 1.0

    res_score = client.system_one(
        state="Current score is 12000, lines cleared 40.",
        questions={
            "performance": Score(
                instructions="Rate game performance on 3-tier scale",
                criteria=["Novice", "Intermediate", "Master"],
            )
        },
    )
    ans_score = res_score.answers["performance"]
    assert ans_score.type == "score"
    assert 0.0 <= ans_score.score <= 2.0
    assert 0.0 <= ans_score.confidence <= 1.0
    assert len(ans_score.legend) == 3


# --------------------------------------------------------------------------------------
# Engine-backed TypeSafe server (model injection + permutation check)
# --------------------------------------------------------------------------------------


def test_typesafe_engine_predict_choice_and_permutation_check():
    engine = TypeSafeEngine(
        model_type="encoder", backbone="huawei-noah/TinyBERT_General_4L_312D", device="cpu"
    )
    assert engine.model_type == "encoder"

    res = engine.predict_choice(
        state_str="User asking about credit card limit",
        instructions="Route to appropriate department",
        criteria={"billing": "Credit and billing", "support": "General tech support"},
    )
    assert res["type"] == "choice"
    assert res["choice"] in ("billing", "support")
    assert "billing" in res["probabilities"]

    perm_res = engine.check_permutation(
        state={"query": "Need help"},
        question={
            "type": "choice",
            "instructions": "Select one",
            "criteria": {"a": "Option A", "b": "Option B"},
        },
    )
    assert perm_res["is_invariant"] is True
    assert perm_res["max_drift"] < 1e-4


def test_engine_dependency_injection():
    """TamevEngine accepts any injected BaseDecisionModel."""
    from decision_engine.config import ModelConfig
    from decision_engine.models import ModelFactory

    cfg = ModelConfig(
        model_type="encoder",
        backbone="huawei-noah/TinyBERT_General_4L_312D",
        projection_dim=64,
        temperature=1.8,
    )
    engine = TamevEngine(
        model=ModelFactory.create(cfg),
        backbone="huawei-noah/TinyBERT_General_4L_312D",
        device="cpu",
    )
    resp = engine.predict_choice(
        state_str="Customer account suspended",
        instructions="Determine next action",
        criteria={"investigate": "Open security log", "unblock": "Lift suspension"},
    )
    assert resp["type"] == "choice"
    assert resp["choice"] in ["investigate", "unblock"]
    assert 0.0 <= resp["confidence"] <= 1.0


# --------------------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------------------


def test_cli_parser_creation():
    parser = create_parser()
    assert parser.prog == "tamev"

    args_finetune = parser.parse_args(
        ["finetune", "--data", "test.jsonl", "--tier", "micro", "--formats", "onnx_int8,mps,mlx"]
    )
    assert args_finetune.subcommand == "finetune"
    assert args_finetune.data == "test.jsonl"
    assert args_finetune.tier == "micro"
    assert args_finetune.export_formats == "onnx_int8,mps,mlx"

    args_serve = parser.parse_args(["serve", "--port", "9000", "--host", "127.0.0.1"])
    assert args_serve.subcommand == "serve"
    assert args_serve.port == 9000
    assert args_serve.host == "127.0.0.1"

    args_benchmark = parser.parse_args(["benchmark", "--suite", "bench.jsonl", "--out", "reports"])
    assert args_benchmark.subcommand == "benchmark"
    assert args_benchmark.suite == "bench.jsonl"
    assert args_benchmark.out == "reports"

    args_export = parser.parse_args(["export", "--tier", "nano", "--formats", "onnx,pytorch"])
    assert args_export.subcommand == "export"
    assert args_export.tier == "nano"
    assert args_export.formats == "onnx,pytorch"

    args_audit = parser.parse_args(["audit", "--config", "configs/custom.yaml"])
    assert args_audit.subcommand == "audit"
    assert args_audit.config == "configs/custom.yaml"


def test_cli_no_args_returns_zero():
    assert main([]) == 0


@pytest.mark.parametrize(
    "subcmd",
    [[], ["finetune"], ["serve"], ["benchmark"], ["export"], ["audit"]],
)
def test_cli_help_in_process(subcmd):
    with pytest.raises(SystemExit) as excinfo:
        main([*subcmd, "--help"])
    assert excinfo.value.code == 0


@pytest.mark.parametrize(
    "cmd_args",
    [
        ["--help"],
        ["finetune", "--help"],
        ["serve", "--help"],
        ["benchmark", "--help"],
        ["export", "--help"],
        ["audit", "--help"],
    ],
)
def test_cli_help_subprocess(cmd_args):
    proc = subprocess.run(
        [sys.executable, "-m", "tamev", *cmd_args], capture_output=True, text=True
    )
    assert proc.returncode == 0
    stdout = proc.stdout.lower()
    assert "usage:" in stdout or "options:" in stdout or "tamev" in stdout


# --------------------------------------------------------------------------------------
# End-to-end real-world decision quality
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_case_engine():
    return TypeSafeEngine(device="cpu")


@pytest.mark.parametrize("case", REAL_CASES, ids=[c["domain"] for c in REAL_CASES])
def test_real_case_decision_quality(real_case_engine, case):
    resp = real_case_engine.predict_batch(
        case["state"], case["questions"], model_name="tamev-latest"
    )
    assert resp["model"] == "tamev-latest"
    assert len(resp["answers"]) == len(case["questions"])

    for ans in resp["answers"].values():
        atype = ans["type"]
        if atype == "choice":
            assert ans["choice"] is not None
            assert 0.0 <= ans["confidence"] <= 1.0
            assert abs(sum(ans["probabilities"].values()) - 1.0) < 0.02
        elif atype == "noul":
            assert 0.0 <= ans["noul"] <= 1.0
        elif atype == "score":
            assert 0.0 <= ans["score"] <= 5.0
            assert 0.0 <= ans["confidence"] <= 1.0
            assert abs(sum(ans["probabilities"].values()) - 1.0) < 0.02


@pytest.mark.parametrize("case", REAL_CASES, ids=[c["domain"] for c in REAL_CASES])
def test_real_case_permutation_invariance(real_case_engine, case):
    choice_qid = next(
        qid
        for qid, q in case["questions"].items()
        if (q.type if hasattr(q, "type") else q.get("type")) == "choice"
    )
    choice_q = case["questions"][choice_qid]
    choice_dict = (
        choice_q
        if isinstance(choice_q, dict)
        else {
            "type": choice_q.type,
            "instructions": choice_q.instructions,
            "criteria": choice_q.criteria,
        }
    )

    perm_res = real_case_engine.check_permutation(case["state"], choice_dict)
    assert perm_res["permutation_invariant"] is True
    assert perm_res["max_drift"] < 1e-5


# --------------------------------------------------------------------------------------
# Console output helpers
# --------------------------------------------------------------------------------------


def test_format_metric():
    from decision_engine.utils.console import format_metric

    assert format_metric(0.75123, is_pct=True) == "75.12%"
    assert format_metric(0.04567, is_pct=False, decimals=4) == "0.0457"
    assert format_metric(None) == "N/A"


def test_print_banner_and_section_output():
    from decision_engine.utils.console import print_banner, print_metrics_summary, print_section

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_banner("TEST TITLE", subtitle="Sub-info", width=40)
        print_section("SECTION", details={"key": "value"}, width=40)
        print_metrics_summary("METRICS", {"top1_accuracy": 0.54, "drift": 0.0, "latency_p50": 4.59})

    output = buf.getvalue()
    assert "TEST TITLE" in output
    assert "Sub-info" in output
    assert "SECTION" in output
    assert "key" in output
    assert "54.00%" in output
    assert "0.00000000" in output
    assert "4.59 ms" in output


def test_create_table_and_scorecard():
    from decision_engine.utils.console import (
        create_table,
        print_error,
        print_info,
        print_scorecard_table,
        print_success,
        print_warning,
    )

    tbl = create_table(title="Test Table", headers=["Col A", "Col B"])
    assert tbl.title == "[bold cyan]Test Table[/]"
    assert len(tbl.columns) == 2

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_scorecard_table("Scorecard", ["H1", "H2"], [["R1C1", "R1C2"]])
        print_success("Operation completed")
        print_warning("Check warning")
        print_info("Informational notice")
        print_error("Failed task")

    out = buf.getvalue()
    assert "Scorecard" in out
    assert "R1C1" in out
    assert "Operation completed" in out
    assert "Check warning" in out
    assert "Informational notice" in out
