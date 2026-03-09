"""Unified LLM client using OpenAI-compatible API.

All providers are accessed through a single OpenAI-compatible endpoint.
Configure via environment variables:

  LLM_API_KEY      API key for the provider
  LLM_BASE_URL     Base URL (e.g. https://open.bigmodel.cn/api/paas/v4)
  LLM_MODEL        Model name (e.g. glm-5)

For Anthropic models, the native SDK is used instead of the OpenAI
compatibility layer.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def resolve_api_key(api_key: str = "") -> str:
    """Resolve API key.

    Priority: explicit arg > LLM_API_KEY env var.
    """
    if api_key:
        return api_key
    return os.environ.get("LLM_API_KEY", "")


def has_api_key() -> bool:
    """Return True if an LLM API key is available."""
    return bool(os.environ.get("LLM_API_KEY"))


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
        model: Model name (e.g. ``glm-5``). Falls back to ``LLM_MODEL`` env var.
        system: System prompt.
        user_message: User message.
        max_tokens: Maximum tokens in the response.
        api_key: Explicit API key (overrides ``LLM_API_KEY`` env var).
        base_url: Explicit base URL (overrides ``LLM_BASE_URL`` env var).

    Returns:
        Generated text content.
    """
    model = model or os.environ.get("LLM_MODEL", "glm-5")
    key = resolve_api_key(api_key)
    if not key:
        raise ValueError("No API key. Set LLM_API_KEY or pass api_key argument.")

    url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
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
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
