"""Unified LLM client using OpenAI-compatible API.

All providers are accessed through a single OpenAI-compatible endpoint.
Configure via environment variables:

  CLAWBENCH_LLM_API_KEY      API key for the provider
  CLAWBENCH_LLM_BASE_URL     Base URL (e.g. https://open.bigmodel.cn/api/paas/v4)
  CLAWBENCH_DEFAULT_MODEL        Model name (e.g. glm-5)

For Anthropic models, the native SDK is used instead of the OpenAI
compatibility layer.
"""

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


def generate(
    model: str = "",
    system: str = "",
    user_message: str = "",
    max_tokens: int = 8192,
    api_key: str = "",
    base_url: str = "",
) -> str:
    """Generate text using an OpenAI-compatible LLM endpoint.

    Args:
        model: Model name (e.g. ``glm-5``). Falls back to ``CLAWBENCH_DEFAULT_MODEL`` env var.
        system: System prompt.
        user_message: User message.
        max_tokens: Maximum tokens in the response.
        api_key: Explicit API key (overrides ``LLM_API_KEY`` env var).
        base_url: Explicit base URL (overrides ``LLM_BASE_URL`` env var).

    Returns:
        Generated text content.
    """
    model = model or os.environ.get("CLAWBENCH_DEFAULT_MODEL", "glm-5")
    # Strip provider prefix (e.g. "zhipu/glm-5" -> "glm-5")
    if "/" in model:
        model = model.split("/", 1)[1]
    key = resolve_api_key(api_key)
    if not key:
        raise ValueError("No API key. Set CLAWBENCH_LLM_API_KEY or pass api_key argument.")

    url = base_url or os.environ.get("CLAWBENCH_LLM_BASE_URL", DEFAULT_BASE_URL)
    logger.info("LLM generate: model=%s, base_url=%s", model, url)

    return _generate_openai_compat(model, system, user_message, max_tokens, key, url)


def _generate_openai_compat(
    model: str,
    system: str,
    user_message: str,
    max_tokens: int,
    api_key: str,
    base_url: Optional[str] = None,
) -> str:
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    current_max_tokens = max_tokens

    for attempt in range(1 + MAX_REASONING_RETRIES):
        response = client.chat.completions.create(
            model=model,
            max_tokens=current_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )

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
