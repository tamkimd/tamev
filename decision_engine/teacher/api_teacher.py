# decision_engine/teacher/api_teacher.py
"""
API Teacher Model for TAMEV Distillation.
Connects to vLLM, Ollama, or OpenAI-compatible endpoints using httpx and tenacity.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import uuid
from typing import Any

import httpx
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from decision_engine.teacher.base import (
    DEFAULT_OPENCODE_BASE_URL,
    BaseTeacherModel,
    TeacherRegistry,
)

logger = logging.getLogger("tamev.teacher.api")


class TeacherAPIError(RuntimeError):
    """Raised when a configured remote teacher cannot produce soft targets.

    Silently substituting a uniform distribution poisons distillation supervision, so a
    configured-but-failing teacher raises by default. Pass ``strict=False`` for bulk
    annotation where skipping is preferable to aborting.
    """


def _default_endpoint() -> str:
    """Resolve the endpoint from the environment, OpenCode Zen first."""
    explicit = os.environ.get("TAMEV_TEACHER_ENDPOINT")
    if explicit:
        return explicit
    oc = os.environ.get("OPENCODE_BASE_URL")
    if oc:
        return oc.rstrip("/") + "/chat/completions"
    if os.environ.get("OPENCODE_API_KEY") and not os.environ.get("OPENAI_BASE_URL"):
        return DEFAULT_OPENCODE_BASE_URL + "/chat/completions"
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")


def _default_api_key() -> str:
    for var in ("TAMEV_TEACHER_API_KEY", "OPENCODE_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    return ""


@TeacherRegistry.register("api")
class APITeacher(BaseTeacherModel):
    """
    Calls a remote HTTP server (vLLM, Ollama, OpenAI) to obtain soft distributions.
    Uses httpx for high-performance connection pooling and tenacity for exponential backoff retries.
    """

    def __init__(
        self,
        model_name: str = "gpt-6-mini",
        endpoint_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 2.0,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_tokens: int = 4096,
        json_mode: bool = True,
        strict: bool = True,
        reasoning_effort: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.endpoint_url = endpoint_url or _default_endpoint()
        self.api_key = api_key or _default_api_key()
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.strict = strict
        self.reasoning_effort = reasoning_effort or os.environ.get("TAMEV_TEACHER_REASONING_EFFORT")
        self.transport = transport
        self.calls = 0
        self.failures = 0
        #: opencode's Zen gateway routes on `x-opencode-session` and answers 400 MissingSessionID
        #: without it. One id per teacher instance keeps a run's calls in one session bucket.
        self._session_id = uuid.uuid4().hex
        # Token accounting, appended per *API* call (list.append is atomic under the GIL,
        # so concurrent annotation needs no lock). Cache hits add nothing. [] means the
        # endpoint returned no usage block.
        self.usage_log: list[dict[str, Any]] = []
        self._no_key_logged = False
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None
        #: Guards `calls`/`failures` and the lazy client. `usage_log.append` needs no lock (list
        #: append is atomic under the GIL) but `self.calls += 1` is load-add-store, so under a
        #: thread pool two workers can read the same value and one increment is lost. `failures`
        #: drives the reported `api_failures`, so losing counts would understate a broken teacher.
        self._lock = threading.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if "opencode.ai" in self.endpoint_url:
            headers["x-opencode-session"] = self._session_id
        return headers

    @property
    def client(self) -> httpx.Client:
        """Lazily initialize thread-safe persistent connection pool."""
        if self._client is None or self._client.is_closed:
            with self._lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.Client(
                        headers=self._headers(),
                        timeout=httpx.Timeout(self.timeout),
                        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                        transport=self.transport,
                    )
        return self._client

    @property
    def aclient(self) -> httpx.AsyncClient:
        """Async twin of `client`, for `aget_soft_targets` on hundreds of concurrent rows.

        An `AsyncClient`'s connections belong to the event loop that opened them, so one instance
        serves one loop -- call `aclose` before starting a second `asyncio.run`.
        """
        if self._aclient is None or self._aclient.is_closed:
            with self._lock:
                if self._aclient is None or self._aclient.is_closed:
                    self._aclient = httpx.AsyncClient(
                        headers=self._headers(),
                        timeout=httpx.Timeout(self.timeout),
                        limits=httpx.Limits(max_keepalive_connections=20, max_connections=64),
                        transport=self.transport,
                    )
        return self._aclient

    def close(self) -> None:
        """Closes the underlying HTTP connection pool."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    async def aclose(self) -> None:
        """Closes the async pool. Must be awaited on the loop that opened it."""
        if self._aclient is not None and not self._aclient.is_closed:
            await self._aclient.aclose()

    def __del__(self) -> None:
        # Interpreter shutdown tears down httpx's module globals before this runs, so `close()` can
        # raise `TypeError: 'NoneType' object is not callable` from deep inside httpcore. A raising
        # `__del__` prints a traceback from a class that has nothing left to say.
        with contextlib.suppress(Exception):
            self.close()

    def _execute_api_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Executes API request with tenacity exponential backoff retries."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        )
        def _call() -> dict[str, Any]:
            resp = self.client.post(self.endpoint_url, json=payload)
            resp.raise_for_status()
            return resp.json()

        return _call()

    async def _aexecute_api_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """`_execute_api_call` over the async client: same tenacity policy, awaited."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        )
        async def _call() -> dict[str, Any]:
            resp = await self.aclient.post(self.endpoint_url, json=payload)
            resp.raise_for_status()
            return resp.json()

        return await _call()

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Parse JSON out of a chat reply that may be fenced or wrapped in prose."""
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("```")[1] if len(t.split("```")) > 1 else t
            t = t.removeprefix("json").strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            return json.loads(t[start : end + 1])
        raise ValueError(f"teacher reply is not JSON: {text[:120]!r}")

    @staticmethod
    def _weights_from(content: Any, k: int) -> list[float]:
        """Accept {"scores": {...}}, {"0": w, ...}, or {"weights": [...]}."""
        obj = APITeacher._extract_json(content) if isinstance(content, str) else content
        if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
            obj = obj["scores"]
        elif isinstance(obj, dict) and isinstance(obj.get("weights"), list):
            obj = dict(enumerate(obj["weights"]))
        if not isinstance(obj, dict):
            raise ValueError(f"teacher reply has no score object: {obj!r}")
        return [float(obj.get(str(i), obj.get(i, 0.0)) or 0.0) for i in range(k)]

    def get_soft_targets(
        self, context: str, options: list[str], temperature: float | None = None
    ) -> np.ndarray:
        k = len(options)
        T = temperature if temperature is not None else self.temperature
        if (early := self._early_return(k)) is not None:
            return early
        with self._lock:
            self.calls += 1
        try:
            return self._finish(self._execute_api_call(self._payload(context, options)), k, T)
        except Exception as e:
            return self._failed(e, k)

    async def aget_soft_targets(
        self, context: str, options: list[str], temperature: float | None = None
    ) -> np.ndarray:
        """`get_soft_targets` over `httpx.AsyncClient`, for hundreds of rows in flight at once.

        Identical prompt, parse and failure semantics: `_payload`, `_finish` and `_failed` are the
        same functions the sync path calls, so the two cannot drift into measuring different tasks.
        Concurrency is one event loop rather than N threads because none of this is CPU-bound -- the
        wall time is the gateway thinking, so the GIL is not the limit and thread stacks are pure
        overhead. Each call still retries on the same schedule as the sync one.
        """
        k = len(options)
        T = temperature if temperature is not None else self.temperature
        if (early := self._early_return(k)) is not None:
            return early
        with self._lock:
            self.calls += 1
        try:
            data = await self._aexecute_api_call(self._payload(context, options))
            return self._finish(data, k, T)
        except Exception as e:
            return self._failed(e, k)

    # -- shared by the sync and async paths: one implementation, so they cannot diverge --

    def _early_return(self, k: int) -> np.ndarray | None:
        """`k < 2` -> ones (a 1-option choice has no decision to make).

        Offline/local endpoints need no key; a remote endpoint without one cannot work, and saying
        so once is better than 40k silent uniform rows.
        """
        if k < 2:
            return np.ones(k, dtype=np.float32)
        if (
            not self.api_key
            and "localhost" not in self.endpoint_url
            and "127.0.0.1" not in self.endpoint_url
        ):
            if not self._no_key_logged:
                self._no_key_logged = True
                logger.error(
                    "No API key for remote teacher endpoint %s -- returning uniform until set.",
                    self.endpoint_url,
                )
            return np.ones(k, dtype=np.float32) / k
        return None

    def _payload(self, context: str, options: list[str]) -> dict[str, Any]:
        # Only dataset fields (passage + question + option texts) are ever sent; never repo code.
        prompt = (
            "Rate how well each option fits the context. Use a 0-10 scale "
            "(0 = clearly wrong, 10 = clearly right); use the full range and avoid ties unless "
            "the options are genuinely indistinguishable.\n"
            f"Context: {context[:2000]}\n"
            "Options:\n"
            + "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))
            + '\nReply with ONLY {"scores": {"<index>": <number>, ...}}, no prose.'
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            # Reasoning teachers otherwise burn the whole token budget on chain-of-thought
            # (measured: 6000 reasoning tokens vs 131 at effort=low for the same rating item).
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _finish(self, data: dict[str, Any], k: int, T: float) -> np.ndarray:
        """Usage accounting, truncation check, score parse, temperature softmax."""
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise TeacherAPIError(
                f"reply truncated (finish_reason=length, max_tokens={self.max_tokens}); "
                "reasoning models need a larger budget -- raise max_tokens"
            )
        usage = data.get("usage") or {}
        self.usage_log.append(
            {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "reasoning_tokens": int(
                    (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                ),
                "finish_reason": choice.get("finish_reason"),
                "k": k,
                "has_usage": bool(usage),
            }
        )
        weights = self._weights_from(choice["message"].get("content"), k)
        arr = np.array(weights, dtype=np.float32) / T
        arr = arr - np.max(arr)
        probs = np.exp(arr) / np.sum(np.exp(arr))
        return probs.astype(np.float32)

    def _failed(self, e: Exception, k: int) -> np.ndarray:
        with self._lock:
            self.failures += 1
        msg = f"Teacher API call failed ({type(e).__name__}: {e})"
        if self.strict:
            raise TeacherAPIError(
                f"{msg}; endpoint={self.endpoint_url} model={self.model_name}"
            ) from e
        logger.error("%s; falling back to uniform distribution.", msg)
        return np.ones(k, dtype=np.float32) / k
