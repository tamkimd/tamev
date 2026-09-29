# decision_engine/teacher/base.py
"""
Base Teacher Interface and Factory for Knowledge Distillation in TAMEV.
Provides pluggable access to high-capacity teachers (Qwen, DeepSeek, Llama, API endpoints).
"""

import contextlib
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from decision_engine.config.base import TeacherConfig

# OpenCode Zen gateway (OpenAI-compatible). Default model: space-bunny-free (free + unlimited;
# requires "Allow free endpoints" in the workspace privacy settings). Override with OPENCODE_MODEL.
DEFAULT_OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_OPENCODE_MODEL = "space-bunny-free"


class BaseTeacherModel(ABC):
    """Abstract base class for all teacher models."""

    def __init__(self, model_name: str = "base_teacher", temperature: float = 2.0):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def get_soft_targets(
        self, context: str, options: list[str], temperature: float | None = None
    ) -> np.ndarray:
        """
        Computes soft probability distribution over candidate options.
        Returns a 1D numpy array of probabilities summing to 1.0.
        """

    def get_batch_soft_targets(
        self, contexts: list[str], options_list: list[list[str]], temperature: float | None = None
    ) -> list[np.ndarray]:
        """Default batch implementation calls get_soft_targets sequentially."""
        return [
            self.get_soft_targets(ctx, opts, temperature)
            for ctx, opts in zip(contexts, options_list)
        ]

    def annotate_examples(
        self, examples: list[dict[str, Any]], temperature: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Annotates a list of decision examples in-place with teacher soft probabilities.
        """
        T = temperature if temperature is not None else self.temperature
        for ex in examples:
            opts = ex.get("options", [])
            if len(opts) < 2:
                continue

            state = ex.get("state", "")
            question = ex.get("question", "")
            context = f"{state}\n{question}".strip()

            opt_texts = [o.get("text", str(o.get("id", ""))) for o in opts]
            probs = self.get_soft_targets(context, opt_texts, temperature=T)

            ex["teacher_probs"] = {
                opts[i]["id"]: round(float(probs[i]), 4) for i in range(len(opts))
            }
        return examples


class TeacherRegistry:
    """Registry mapping provider names to BaseTeacherModel subclasses."""

    _registry: dict[str, type[BaseTeacherModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[BaseTeacherModel]):
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseTeacherModel]:
        key = name.lower()
        if key == "vllm":
            key = "api" if "api" in cls._registry else "huggingface"
        elif key == "opencode":
            key = "api"
        elif key in ("hf", "transformer", "causal"):
            key = "huggingface"
        if key not in cls._registry:
            raise KeyError(
                f"Unknown teacher provider: '{name}'. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[key]

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._registry.keys())


class TeacherFactory:
    """Factory creating teacher instances from TeacherConfig."""

    @staticmethod
    def create(config: TeacherConfig) -> BaseTeacherModel:
        with contextlib.suppress(ImportError):
            pass
        with contextlib.suppress(ImportError):
            pass

        provider_cls = TeacherRegistry.get(config.provider)

        if config.provider == "mock":
            return provider_cls(model_name=config.name, temperature=config.temperature)
        if config.provider == "agent":
            cache_path = config.cache_dir or "runs/agent_teacher_cache.json"
            model_name = config.name
            if not model_name or "qwen" in model_name.lower():
                model_name = (
                    os.environ.get("AGENT_MODEL_AS_TEACHER")
                    or os.environ.get("AGENT_MODEL_AS_TECHER")
                    or os.environ.get("AGENT_MODEL")
                    or os.environ.get("TAMEV_AGENT_MODEL")
                    or os.environ.get("TAMEV_AGENT_NAME")
                    or "agent-teacher"
                )
            return provider_cls(
                model_name=model_name,
                temperature=config.temperature,
                cache_path=cache_path,
                system_prompt=getattr(config, "system_prompt", None),
                prompt_template=getattr(config, "prompt_template", None),
                domain=getattr(config, "domain", "general"),
            )
        if config.provider in ("huggingface", "hf", "transformer"):
            return provider_cls(
                model_name_or_path=config.model_path,
                device=config.device,
                temperature=config.temperature,
            )
        if config.provider in ("api", "vllm", "opencode"):
            endpoint_url = config.endpoint_url
            model_name = config.name
            if config.provider == "opencode":
                endpoint_url = endpoint_url or os.environ.get(
                    "OPENCODE_BASE_URL", DEFAULT_OPENCODE_BASE_URL
                )
                if "/chat/completions" not in endpoint_url:
                    endpoint_url = endpoint_url.rstrip("/") + "/chat/completions"
                if model_name in ("qwen3.5-4b", "Qwen/Qwen3.5-4B"):
                    model_name = os.environ.get("OPENCODE_MODEL") or DEFAULT_OPENCODE_MODEL
            return provider_cls(
                model_name=model_name,
                endpoint_url=endpoint_url,
                api_key=config.api_key,
                temperature=config.temperature,
                **({"reasoning_effort": "low"} if config.provider == "opencode" else {}),
            )
        if config.provider == "cached":
            cache_path = config.cache_dir or "runs/teacher_cache.json"
            return provider_cls(
                cache_path=cache_path, model_name=config.name, temperature=config.temperature
            )
        return provider_cls(model_name=config.name, temperature=config.temperature)

    @classmethod
    def create_from_name(
        cls,
        provider: str = "mock",
        model_name_or_path: str | None = None,
        temperature: float = 2.0,
        device: str = "auto",
        api_key: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> BaseTeacherModel:
        """Instantiate a Teacher Model directly from provider name and options."""
        prov = provider.lower()
        default_model = "agent-teacher" if prov == "agent" else "Qwen/Qwen3.5-4B"
        cfg = TeacherConfig(
            name=model_name_or_path or prov,
            provider=prov,
            model_path=model_name_or_path or default_model,
            temperature=temperature,
            device=device,
            api_key=api_key,
            cache_dir=cache_dir,
            **kwargs,
        )
        return cls.create(cfg)
