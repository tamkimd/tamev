"""Teacher system: Mock/Cached/API providers, TeacherRegistry/Factory, and the agentic teacher.

One concern: producing calibrated soft targets. Covers provider contracts (probability
distribution, persistence/caching, example annotation), API transport, and the domain-rubric
behaviour of the autonomous AgentTeacher.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from decision_engine.config import TeacherConfig
from decision_engine.teacher.agent_teacher import AgentTeacher
from decision_engine.teacher.api_teacher import APITeacher
from decision_engine.teacher.base import BaseTeacherModel, TeacherFactory, TeacherRegistry
from decision_engine.teacher.cached_teacher import CachedTeacher
from decision_engine.teacher.mock_teacher import MockTeacher


def test_teacher_registry_lists_all_providers():
    providers = TeacherRegistry.list_providers()
    assert {"mock", "cached", "huggingface", "api"} <= set(providers)


def test_teacher_factory_creation_and_temperature():
    cfg = TeacherConfig(name="test_mock", provider="mock", temperature=1.5)
    teacher = TeacherFactory.create(cfg)
    assert isinstance(teacher, BaseTeacherModel)
    assert teacher.temperature == 1.5


def test_mock_teacher_soft_targets():
    teacher = TeacherFactory.create_from_name("mock", temperature=2.0)
    assert isinstance(teacher, MockTeacher)
    assert isinstance(teacher, BaseTeacherModel)

    probs = teacher.get_soft_targets(
        "Tetris current piece is I-bar", ["drop col 0", "drop col 5", "drop col 9"], temperature=2.0
    )
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
    assert (probs > 0.0).all()


def test_cached_teacher_hits_misses_and_persistence(tmp_path):
    cache_file = tmp_path / "teacher_cache.json"
    cached = TeacherFactory.create_from_name("cached", cache_dir=str(cache_file), temperature=2.0)
    assert isinstance(cached, CachedTeacher)

    ctx, opts = "Question about banking transfer", ["transfer", "cancel", "check balance"]
    p1 = cached.get_soft_targets(ctx, opts)
    assert cache_file.exists()
    assert cached.cache_misses == 1
    assert cached.cache_hits == 0

    p2 = cached.get_soft_targets(ctx, opts)
    assert np.allclose(p1, p2)
    assert cached.cache_hits == 1


def test_mock_teacher_annotates_examples_with_normalised_soft_targets():
    mock = MockTeacher(model_name="mock_qwen")
    examples = [
        {
            "id": "ex1",
            "state": "Game state 1",
            "question": "What to do?",
            "options": [{"id": "opt1", "text": "Move left"}, {"id": "opt2", "text": "Move right"}],
        }
    ]
    annotated = mock.annotate_examples(examples, temperature=2.0)
    probs = annotated[0]["teacher_probs"]
    assert set(probs) == {"opt1", "opt2"}
    assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0


def test_api_teacher_falls_back_uniformly_without_a_key():
    teacher = APITeacher(endpoint_url="https://api.openai.com/v1/chat/completions", api_key="")
    probs = teacher.get_soft_targets("Some state", ["A", "B", "C"])
    assert len(probs) == 3
    assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3])
    teacher.close()


def test_api_teacher_parses_a_mock_transport_response():
    import httpx

    def mock_handler(request: httpx.Request) -> httpx.Response:
        data = {"choices": [{"message": {"content": json.dumps({"0": 0.8, "1": 0.2})}}]}
        return httpx.Response(200, json=data)

    teacher = APITeacher(
        endpoint_url="http://mock-llm:8000/v1/chat/completions", api_key="test-key", temperature=1.0
    )
    teacher._client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    probs = teacher.get_soft_targets("context", ["opt0", "opt1"])
    assert len(probs) == 2
    assert probs[0] > probs[1]
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
    teacher.close()


# --------------------------------------------------------------------------------------
# Agentic teacher (AgentTeacher)
# --------------------------------------------------------------------------------------


def test_agent_teacher_registry_and_factory():
    assert TeacherRegistry.get("agent") is AgentTeacher

    teacher_agent = TeacherFactory.create_from_name("agent", temperature=1.5)
    assert isinstance(teacher_agent, AgentTeacher)
    assert teacher_agent.temperature == 1.5


def test_agent_teacher_model_name_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_AS_TEACHER", "claude-3.7-sonnet")
    assert AgentTeacher().model_name == "claude-3.7-sonnet"

    monkeypatch.delenv("AGENT_MODEL_AS_TEACHER", raising=False)
    monkeypatch.setenv("AGENT_MODEL_AS_TECHER", "gemini-3.8-flash")
    assert AgentTeacher().model_name == "gemini-3.8-flash"


def test_agent_teacher_probability_distribution_contract(tmp_path):
    teacher = AgentTeacher(cache_path=tmp_path / "agent_cache.json", temperature=2.0)
    ctx = "Cluster experiencing high CPU load and thread exhaustion."

    probs = teacher.get_soft_targets(ctx, ["scale_nodes", "kill_tasks", "ignore"])
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0)
    assert all(p >= 0.0 for p in probs)

    # Single option boundary
    probs_single = teacher.get_soft_targets(ctx, ["scale_nodes"])
    assert len(probs_single) == 1
    assert probs_single[0] == 1.0

    # Empty options boundary
    assert len(teacher.get_soft_targets(ctx, [])) == 0


@pytest.mark.parametrize(
    "ctx,opts,floor",
    [
        pytest.param(
            "Postgres connection pool exhausted, thread lock in worker service, DB connection outage",
            ["remediate_and_scale_pool", "wait_and_see", "shutdown_database"],
            None,
            id="sre",
        ),
        pytest.param(
            "Snake game state: snake head at (5, 5), wall at (5, 6), food at (4, 5)",
            ["move_towards_food_safe", "move_into_wall_hazard"],
            0.60,
            id="game_ai",
        ),
        pytest.param(
            "Customer reports debit card declined at ATM during international travel",
            ["verify_identity_and_unblock", "close_account_permanently"],
            None,
            id="fintech",
        ),
    ],
)
def test_agent_teacher_domain_rubric_prefers_the_safe_action(tmp_path, ctx, opts, floor):
    teacher = AgentTeacher(cache_path=tmp_path / "rubric_cache.json", temperature=1.0)
    probs = teacher.get_soft_targets(ctx, opts)

    best_idx = int(np.argmax(probs))
    assert best_idx == 0, f"expected {opts[0]} to be top choice, got {opts[best_idx]}"
    assert probs[0] > max(probs[1:])
    if floor is not None:
        assert probs[0] > floor


def test_agent_teacher_persistent_caching(tmp_path):
    cache_file = tmp_path / "test_cache.json"
    teacher1 = AgentTeacher(cache_path=cache_file, temperature=2.0)
    ctx, opts = "Test context for persistent caching", ["opt_a", "opt_b"]

    probs1 = teacher1.get_soft_targets(ctx, opts)
    assert cache_file.exists()

    teacher2 = AgentTeacher(cache_path=cache_file, temperature=2.0)
    assert np.allclose(probs1, teacher2.get_soft_targets(ctx, opts))


def test_agent_teacher_manual_decision_injection(tmp_path):
    teacher = AgentTeacher(cache_path=tmp_path / "manual_cache.json", temperature=2.0)
    ctx = "Custom agent architectural review decision"
    opts = ["monolith", "microservices", "modular_monolith"]

    teacher.register_manual_agent_decision(
        ctx, opts, [0.10, 0.20, 0.70], rationale="High cohesion with decoupled pointer head"
    )

    probs = teacher.get_soft_targets(ctx, opts)
    assert np.isclose(probs[2], 0.70, atol=1e-3)
    assert np.isclose(probs[1], 0.20, atol=1e-3)
    assert np.isclose(probs[0], 0.10, atol=1e-3)


def test_agent_teacher_annotates_examples(tmp_path):
    teacher = AgentTeacher(cache_path=tmp_path / "ann_cache.json", temperature=2.0)
    examples = [
        {
            "id": "ex_1",
            "state": "High latency detected on API gateway",
            "question": "Choose routing action",
            "options": [
                {"id": "route_canary", "text": "Route to healthy canary cluster"},
                {"id": "drop_traffic", "text": "Drop all incoming traffic"},
            ],
        }
    ]
    t_probs = teacher.annotate_examples(examples)[0]["teacher_probs"]
    assert {"route_canary", "drop_traffic"} <= set(t_probs)
    assert t_probs["route_canary"] > t_probs["drop_traffic"]


def test_agent_teacher_build_prompt(tmp_path):
    teacher = AgentTeacher(domain="sre", cache_path=tmp_path / "p_cache.json")
    p_data = teacher.build_prompt(
        "Cluster OOMKilled worker on pod-9", ["scale_memory", "restart_pod"], domain="sre"
    )
    assert "decision distillation intelligence" in p_data["system_prompt"]
    assert "Site Reliability Engineering" in p_data["user_prompt"]
    assert "Cluster OOMKilled worker on pod-9" in p_data["user_prompt"]
    assert "[1] scale_memory" in p_data["user_prompt"]


def test_agent_teacher_custom_prompt_file(tmp_path):
    custom_file = tmp_path / "custom_prompt.md"
    custom_file.write_text("Custom distillation instructions for testing.", encoding="utf-8")
    teacher = AgentTeacher(prompt_file=str(custom_file), cache_path=tmp_path / "custom_cache.json")
    p_data = teacher.build_prompt("Context", ["opt1", "opt2"])
    assert p_data["system_prompt"] == "Custom distillation instructions for testing."


def test_agent_teacher_parses_fenced_json_response(tmp_path):
    teacher = AgentTeacher(cache_path=tmp_path / "parse_cache.json")
    agent_markdown = """
    Here is my expert analysis as Master Coding Agent:
    ```json
    {
      "rationale": "Memory limit must be increased to prevent cascading OOM kills",
      "probabilities": {"increase_limits": 0.88, "ignore": 0.12}
    }
    ```
    """
    parsed = teacher.parse_agent_response(agent_markdown, ["increase_limits", "ignore"])
    probs = parsed["probabilities"]
    assert len(probs) == 2
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > 0.80
    assert "cascading OOM" in parsed["rationale"]
