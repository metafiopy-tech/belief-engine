"""LLM Client — the single path to any language model.

All agents call through this. Handles:
  - Model routing via ModelRouter
  - Structured output parsing via Pydantic
  - Token usage tracking
  - Graceful fallback on connection errors

Source: forge/llm.py + belief_call.py covenant gate concept
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextvars import ContextVar
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from belief.config.models import Backend, ModelRole, ModelRouter
from belief.config.local_cost_tracker import LocalCostTracker
from belief.models.artifacts import TokenUsage, _cost_usd

logger = logging.getLogger("belief.llm")

T = TypeVar("T", bound=BaseModel)

# Context variable for per-agent token accumulation
# BaseAgent.__call__ sets this before run() and reads it after
_usage_ctx: ContextVar[TokenUsage | None] = ContextVar("_usage_ctx", default=None)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# ---------------------------------------------------------------------------
# Session 6: Ollama backend
# ---------------------------------------------------------------------------


# Shared tracker so callers can read fallback count + per-model breakdown.
# Exposed module-level so the CLI `belief models` command can inspect it.
LOCAL_TRACKER = LocalCostTracker()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — Ollama's /api/chat doesn't return usage by default.

    Anthropic tokenization averages ~4 chars/token for English. We use the
    same heuristic for Ollama outputs so the fallback-counter and efficiency
    metrics stay comparable between backends.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class AsyncOllamaClient:
    """Async Ollama backend. Matches the path LLMClient already uses.

    Session 6 Task 1 shows a sync OllamaClient — we keep the same public
    surface (generate / is_available) but use httpx.AsyncClient because
    the rest of belief.llm is already async. LLMClient._call_with_role
    is the dispatch point.

    Validation-phase Session 1 additions:
      - keep_alive: tell Ollama to keep the model resident between calls.
        Avoids re-loading the 14B weights on every agent hop (saves
        ~3-5s per call on MacBook Air M2).
      - num_ctx: pin the context window so Ollama doesn't re-allocate
        KV cache between calls with different prompt sizes.
      - Identical system prompts across calls in a build get prefix-
        cached by Ollama automatically; we make sure that actually
        happens by keeping system/user separation stable.

    Env var overrides (read on construction):
      BELIEF_OLLAMA_KEEP_ALIVE  — e.g. "30m", "1h", "-1" (forever).
                                  Default: "30m".
      BELIEF_OLLAMA_NUM_CTX     — integer token window. Default: 8192.
    """

    _DEFAULT_KEEP_ALIVE = "30m"
    _DEFAULT_NUM_CTX = 8192

    def __init__(
        self,
        *,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        keep_alive: str | None = None,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

        # Resolution order: explicit arg > env var > default.
        env_keep_alive = os.environ.get("BELIEF_OLLAMA_KEEP_ALIVE", "").strip()
        self.keep_alive = (
            keep_alive
            if keep_alive is not None
            else (env_keep_alive or self._DEFAULT_KEEP_ALIVE)
        )

        env_num_ctx = os.environ.get("BELIEF_OLLAMA_NUM_CTX", "").strip()
        if num_ctx is not None:
            self.num_ctx = int(num_ctx)
        elif env_num_ctx:
            try:
                self.num_ctx = int(env_num_ctx)
            except ValueError:
                self.num_ctx = self._DEFAULT_NUM_CTX
        else:
            self.num_ctx = self._DEFAULT_NUM_CTX

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self._timeout
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> dict[str, Any]:
        """POST /api/chat. Returns an Anthropic-shaped response dict.

        Shape matches what LLMClient._call returns, so
        generate_text / generate_structured can unwrap uniformly:

          {"content": [{"text": "..."}],
           "usage": {"input_tokens": int, "output_tokens": int,
                     "cache_read_input_tokens": 0,
                     "cache_creation_input_tokens": 0}}
        """
        client = self._get_client()
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # keep_alive keeps the model resident between calls within a
            # build — same 14B weights across PLAN/BUILD/DEBUG instead of
            # a cold reload each time.
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                # Pinning num_ctx stabilises Ollama's KV-cache so the
                # system-prompt prefix is reused across calls in a build.
                "num_ctx": int(self.num_ctx),
            },
        }
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"message": {"role": "assistant", "content": "..."}}
        text = (data.get("message") or {}).get("content", "") or ""

        # Ollama sends prompt_eval_count + eval_count when available; fall
        # back to character-based estimates.
        in_tokens = int(data.get("prompt_eval_count") or _estimate_tokens(f"{system} {user}"))
        out_tokens = int(data.get("eval_count") or _estimate_tokens(text))
        return {
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            "_backend": "ollama",
        }

    async def is_available(self) -> bool:
        """True iff Ollama's HTTP endpoint responds to /api/tags."""
        try:
            client = self._get_client()
            r = await client.get("/api/tags")
            return r.status_code == 200
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException):
            return False
        except Exception:
            return False


class LLMClient:
    """Unified LLM client for the Belief Engine.

    Usage:
        llm = LLMClient(router)
        text = await llm.generate_text(role, system, prompt)
        obj = await llm.generate_structured(role, system, prompt, MyModel)
        await llm.close()
    """

    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self._client: httpx.AsyncClient | None = None
        # Session 6: lazy Ollama client. Built only when a call actually
        # needs it, so cloud-mode callers never pay the connection cost.
        self._ollama: AsyncOllamaClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            from belief.config.settings import settings
            self._client = httpx.AsyncClient(
                timeout=120.0,
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": API_VERSION,
                    "anthropic-beta": "prompt-caching-2024-07-31",
                    "content-type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._ollama is not None:
            await self._ollama.close()

    def _get_ollama(self) -> AsyncOllamaClient:
        if self._ollama is None:
            self._ollama = AsyncOllamaClient(
                model=self.router.local_model,
                base_url=self.router.ollama_base_url,
            )
        return self._ollama

    async def _call_with_role(
        self,
        role: ModelRole | str,
        system: str,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        complexity: int,
    ) -> tuple[dict, str, str]:
        """Dispatch to Ollama or Anthropic based on the router's mode.

        Returns (response_dict, model_name_used, backend_used).
        Falls back to cloud with a logged warning if the local backend
        is chosen but isn't available.
        """
        role_str = role.value if isinstance(role, ModelRole) else role
        backend = self.router.backend_for(role)
        cloud_model = self.router.get_model(role, complexity)

        logger.info(
            "Dispatch: role=%s backend=%s mode=%s",
            role_str,
            backend.value,
            self.router.mode.value,
        )

        if backend is Backend.LOCAL:
            ollama = self._get_ollama()
            if not await ollama.is_available():
                logger.warning(
                    "Ollama not available for role=%s; falling back to cloud",
                    role_str,
                )
                self.router.record_fallback()
                LOCAL_TRACKER.record_fallback()
                backend = Backend.CLOUD  # fall through to cloud below
            else:
                user_prompt = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                        break
                data = await ollama.generate(
                    system=system,
                    user=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                usage = data.get("usage", {})
                LOCAL_TRACKER.record_call(
                    model=self.router.local_model,
                    prompt_tokens=int(usage.get("input_tokens", 0)),
                    completion_tokens=int(usage.get("output_tokens", 0)),
                    role=role_str,
                )
                return data, self.router.local_model, "local"

        # Cloud path (default or fallback)
        data = await self._call(
            model=cloud_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return data, cloud_model, "cloud"

    def _record_usage(self, role: str, model: str,
                      prompt_tokens: int, completion_tokens: int,
                      cache_read_tokens: int = 0, cache_create_tokens: int = 0) -> None:
        cost = _cost_usd(model, prompt_tokens, completion_tokens,
                         cache_read_tokens, cache_create_tokens)
        usage = _usage_ctx.get()
        if usage is not None:
            usage.add_call(role, prompt_tokens, completion_tokens, cost)
        tier = "haiku" if "haiku" in model else "sonnet" if "sonnet" in model else "opus"
        cache_info = ""
        if cache_read_tokens > 0:
            cache_info = f" cache_hit={cache_read_tokens}"
        logger.debug(
            f"LLM call: role={role} tier={tier} "
            f"tokens={prompt_tokens}+{completion_tokens}{cache_info} cost=${cost:.4f}"
        )

    async def _call(self, model: str, system: str, messages: list[dict],
                    max_tokens: int = 4096, temperature: float = 0.3) -> dict:
        """Raw API call. Returns the full response dict.

        Uses prompt caching on system prompts for 90% input cost reduction.
        Cache-eligible system prompts must be >1024 tokens (~4000 chars).
        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        # Prompt caching: wrap system prompt in cache_control block
        # Only cache if system prompt is long enough (>1024 tokens ≈ 4000 chars)
        if system:
            if len(system) > 2000:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system

        try:
            resp = await client.post(API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Log cache performance
            usage = data.get("usage", {})
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            if cache_read > 0:
                logger.debug(f"Prompt cache HIT: {cache_read} tokens read from cache")
            elif cache_create > 0:
                logger.debug(f"Prompt cache MISS: {cache_create} tokens written to cache")

            return data
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot reach Anthropic API: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
            raise

    async def generate_text(
        self,
        role: ModelRole | str,
        system: str,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        complexity: int = 1,
    ) -> str:
        """Generate free-form text. Returns the response string."""
        role_str = role.value if isinstance(role, ModelRole) else role

        data, model, backend = await self._call_with_role(
            role, system,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            complexity=complexity,
        )

        text = data["content"][0]["text"]

        # Local-backend responses are already booked against LOCAL_TRACKER
        # inside _call_with_role; only cloud responses go through
        # the Anthropic-cost accounting path.
        if backend == "cloud":
            usage = data.get("usage", {})
            self._record_usage(
                role_str, model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
                usage.get("cache_creation_input_tokens", 0),
            )

        return text

    async def generate_structured(
        self,
        role: ModelRole | str,
        system: str,
        prompt: str,
        response_schema: Type[T],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        complexity: int = 1,
    ) -> T:
        """Generate a structured response parsed into a Pydantic model.

        The system prompt is augmented with JSON schema instructions.
        The response is parsed from the first JSON object found.
        """
        role_str = role.value if isinstance(role, ModelRole) else role

        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        augmented_system = (
            f"{system}\n\n"
            f"Respond ONLY with a valid JSON object matching this schema. "
            f"No markdown fences. No explanation. Just the JSON.\n\n"
            f"Schema:\n{schema_json}"
        )

        data, model, backend = await self._call_with_role(
            role, augmented_system,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            complexity=complexity,
        )

        raw_text = data["content"][0]["text"]

        if backend == "cloud":
            usage = data.get("usage", {})
            self._record_usage(
                role_str, model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
                usage.get("cache_creation_input_tokens", 0),
            )

        return _parse_structured(raw_text, response_schema)


def _parse_structured(raw: str, schema: Type[T]) -> T:
    """Extract and parse JSON from LLM response text.
    
    Handles truncated JSON by attempting repair (closing unclosed brackets/braces).
    This is critical for the decomposer and validator whose outputs frequently
    exceed the max_tokens limit.
    """
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Find the JSON object
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    # Try 1: Find matching closing brace (clean JSON)
    depth = 0
    json_str = None
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                json_str = text[brace_start: i + 1]
                break

    if json_str:
        try:
            data = json.loads(json_str)
            return schema.model_validate(data)
        except Exception as e:
            logger.debug(f"Initial JSON parse failed, attempting repair: {e}")

    # Try 2: Repair truncated JSON
    truncated = text[brace_start:]
    repaired = _repair_json(truncated)
    if repaired:
        try:
            data = json.loads(repaired)
            logger.info(f"Parsed repaired JSON ({len(repaired)} chars)")
            return schema.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"JSON repair failed for {schema.__name__}: {e}\n"
                f"Repaired text: {repaired[:300]}"
            )

    raise ValueError(f"Unclosed JSON object in response: {text[:200]}")


def _repair_json(text: str) -> str | None:
    """Repair truncated JSON by closing unclosed brackets and braces.
    
    Strategy: walk the JSON tracking the stack of open delimiters.
    When we find the text is truncated, roll back to the last position
    where the JSON was structurally valid, then close the remaining stack.
    """
    s = text.rstrip()
    if not s:
        return None

    # Walk the string tracking state
    stack: list[str] = []  # Stack of open delimiters: '{' or '['
    in_string = False
    escape = False
    last_complete = 0  # Last position where we completed a value

    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if c == "\\":
            if in_string:
                escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        # Outside strings
        if c == '{':
            stack.append('}')
        elif c == '[':
            stack.append(']')
        elif c in ('}', ']'):
            if stack and stack[-1] == c:
                stack.pop()
            last_complete = i + 1
        elif c == ',':
            last_complete = i
        elif c == ':':
            pass  # After a key, before a value

    # If we ended inside a string, roll back to last complete position
    if in_string and last_complete > 0:
        s = s[:last_complete]
        # Recalculate stack from clean text
        stack = []
        in_str = False
        esc = False
        for c in s:
            if esc:
                esc = False
                continue
            if c == '\\':
                if in_str:
                    esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                stack.append('}')
            elif c == '[':
                stack.append(']')
            elif c in ('}', ']'):
                if stack and stack[-1] == c:
                    stack.pop()

    # Remove trailing comma
    s = s.rstrip()
    if s and s[-1] == ',':
        s = s[:-1]

    # Close the stack in reverse order (innermost first)
    s += ''.join(reversed(stack))

    return s if s else None
