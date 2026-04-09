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
import re
from contextvars import ContextVar
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from belief.config.models import ModelRole, ModelRouter
from belief.models.artifacts import TokenUsage, _cost_usd

logger = logging.getLogger("belief.llm")

T = TypeVar("T", bound=BaseModel)

# Context variable for per-agent token accumulation
# BaseAgent.__call__ sets this before run() and reads it after
_usage_ctx: ContextVar[TokenUsage | None] = ContextVar("_usage_ctx", default=None)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


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
        model = self.router.get_model(role, complexity)

        data = await self._call(
            model=model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = data["content"][0]["text"]

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
        model = self.router.get_model(role, complexity)

        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        augmented_system = (
            f"{system}\n\n"
            f"Respond ONLY with a valid JSON object matching this schema. "
            f"No markdown fences. No explanation. Just the JSON.\n\n"
            f"Schema:\n{schema_json}"
        )

        data = await self._call(
            model=model,
            system=augmented_system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        raw_text = data["content"][0]["text"]

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
        except Exception:
            pass  # Fall through to repair

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
