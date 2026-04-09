"""Unified LLM client using OpenAI-compatible API.

All providers are accessed through a single OpenAI-compatible endpoint.
Configure via environment variables:

  CLAWBENCH_LLM_API_KEY      API key for the provider
  CLAWBENCH_LLM_BASE_URL     Base URL (e.g. https://open.bigmodel.cn/api/paas/v4)
  CLAWBENCH_DEFAULT_MODEL        Model name (e.g. glm-5.1)

For Anthropic models, the native SDK is used instead of the OpenAI
compatibility layer.
"""

import asyncio
import functools
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# Reasoning models may need more tokens since they spend tokens on
# thinking before producing the actual answer.
MAX_REASONING_RETRIES = 1
REASONING_RETRY_MULTIPLIER = 4


def resolve_api_key(api_key: str = "") -> str:
    """Resolve API key.

    Priority: explicit arg > CLAWBENCH_LLM_API_KEY env var.
    """
    if api_key:
        return api_key
    return os.environ.get("CLAWBENCH_LLM_API_KEY", "")


def has_api_key() -> bool:
    """Return True if an LLM API key is available."""
    return bool(os.environ.get("CLAWBENCH_LLM_API_KEY"))


def _generate(
    model: str = "",
    system: str = "",
    user_message: str = "",
    max_tokens: int = 8192,
    api_key: str = "",
    base_url: str = "",
    temperature: Optional[float] = None,
) -> str:
    """Synchronous LLM generation (internal).

    Prefer ``async_generate`` in async contexts — this function blocks
    the calling thread until the HTTP request completes.
    """
    model = model or os.environ.get("CLAWBENCH_DEFAULT_MODEL", "glm-5.1")
    if "/" in model:
        model = model.split("/", 1)[1]
    key = resolve_api_key(api_key)
    if not key:
        raise ValueError("No API key. Set CLAWBENCH_LLM_API_KEY or pass api_key argument.")

    url = base_url or os.environ.get("CLAWBENCH_LLM_BASE_URL", DEFAULT_BASE_URL)
    logger.info("LLM generate: model=%s, base_url=%s", model, url)

    return _generate_openai_compat(model, system, user_message, max_tokens, key, url, temperature=temperature)


LLM_CALL_TIMEOUT = 180  # seconds; prevents a single hung LLM call from
                        # blocking the async event loop indefinitely.


async def async_generate(
    model: str = "",
    system: str = "",
    user_message: str = "",
    max_tokens: int = 8192,
    api_key: str = "",
    base_url: str = "",
    temperature: Optional[float] = None,
    timeout: float = LLM_CALL_TIMEOUT,
) -> str:
    """Generate text using an OpenAI-compatible LLM endpoint (async).

    Runs the synchronous OpenAI call in a thread-pool executor so it
    does not block the asyncio event loop, and applies an explicit
    timeout to prevent hung API calls from stalling the validator.
    """
    loop = asyncio.get_running_loop()
    func = functools.partial(
        _generate,
        model=model,
        system=system,
        user_message=user_message,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
    return await asyncio.wait_for(
        loop.run_in_executor(None, func),
        timeout=timeout,
    )


def _generate_openai_compat(
    model: str,
    system: str,
    user_message: str,
    max_tokens: int,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    current_max_tokens = max_tokens

    for attempt in range(1 + MAX_REASONING_RETRIES):
        create_kwargs = dict(
            model=model,
            max_tokens=current_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        if temperature is not None:
            create_kwargs["temperature"] = temperature

        response = client.chat.completions.create(**create_kwargs)

        choice = response.choices[0]
        message = choice.message
        content = message.content
        reasoning = getattr(message, "reasoning_content", None)
        finish_reason = choice.finish_reason

        # Happy path: model returned content
        if content:
            return content

        # Reasoning model detection: has reasoning but no content
        if reasoning and not content:
            if finish_reason == "length" and attempt < MAX_REASONING_RETRIES:
                # Token budget exhausted by reasoning — retry with more tokens
                new_max = current_max_tokens * REASONING_RETRY_MULTIPLIER
                logger.warning(
                    "Reasoning model used all %d tokens on thinking "
                    "(finish_reason=length, reasoning=%d chars, content=empty). "
                    "Retrying with max_tokens=%d.",
                    current_max_tokens, len(reasoning), new_max,
                )
                current_max_tokens = new_max
                continue
            else:
                # Either retry exhausted or finish_reason != length
                # Fall back to reasoning_content
                logger.warning(
                    "LLM returned empty content but has reasoning_content "
                    "(%d chars, finish_reason=%s). This model may require "
                    "higher max_tokens or a non-reasoning model. "
                    "Falling back to reasoning_content.",
                    len(reasoning), finish_reason,
                )
                return reasoning

        # No content and no reasoning
        return content or ""
