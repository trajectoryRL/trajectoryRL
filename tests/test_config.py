"""Validator LLM config is code-locked, not operator-overridable.

The testee model and inference endpoint that miners are scored against must
be identical across the whole validator fleet, otherwise scores are not
comparable. So ``ValidatorConfig.from_env`` pins the model + base URL (and the
judge's model + base URL, which default to the primary) to the code constants
and ignores any ``LLM_MODEL`` / ``LLM_BASE_URL`` / ``JUDGE_MODEL`` /
``JUDGE_BASE_URL`` set in the operator environment. Only the secrets
(``LLM_API_KEY`` / ``JUDGE_API_KEY``) remain env-driven — they cannot be
committed to a public repo.
"""
from __future__ import annotations

import logging
from pathlib import Path

from trajectoryrl.utils.config import (
    ValidatorConfig,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_BASE_URL,
)

# A dotenv path that does not exist, so ``load_dotenv`` is a no-op and the test
# is driven purely by ``monkeypatch.setenv`` — never the live .env.validator.
_NO_DOTENV = Path("/nonexistent/.env.validator")


def test_llm_model_override_is_ignored(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "operator-picked-other-model")
    cfg = ValidatorConfig.from_env(dotenv_path=_NO_DOTENV)
    assert cfg.llm_model == DEFAULT_LLM_MODEL


def test_llm_base_url_override_is_ignored(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    cfg = ValidatorConfig.from_env(dotenv_path=_NO_DOTENV)
    assert cfg.llm_base_url == DEFAULT_LLM_BASE_URL


def test_judge_model_override_is_ignored(monkeypatch):
    # The judge is locked to a fixed model distinct from the testee (keeps judge
    # bias uncorrelated), so the env override is ignored — NOT collapsed onto
    # the testee model.
    monkeypatch.setenv("JUDGE_MODEL", "operator-picked-judge-model")
    cfg = ValidatorConfig.from_env(dotenv_path=_NO_DOTENV)
    assert cfg.judge_model == DEFAULT_JUDGE_MODEL
    assert cfg.judge_model != cfg.llm_model  # judge stays decorrelated from testee


def test_judge_base_url_override_is_ignored(monkeypatch):
    monkeypatch.setenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
    cfg = ValidatorConfig.from_env(dotenv_path=_NO_DOTENV)
    assert cfg.judge_base_url == DEFAULT_JUDGE_BASE_URL


def test_llm_api_key_still_read_from_env(monkeypatch):
    # The one exception: the API key is a secret and stays operator-supplied.
    monkeypatch.setenv("LLM_API_KEY", "sk-operator-secret")
    cfg = ValidatorConfig.from_env(dotenv_path=_NO_DOTENV)
    assert cfg.llm_api_key == "sk-operator-secret"


def test_warns_when_a_locked_var_is_overridden(monkeypatch, caplog):
    monkeypatch.setenv("LLM_MODEL", "operator-picked-other-model")
    with caplog.at_level(logging.WARNING, logger="trajectoryrl.utils.config"):
        ValidatorConfig.from_env(dotenv_path=_NO_DOTENV)
    assert any("LLM_MODEL" in r.getMessage() and "ignored" in r.getMessage().lower()
               for r in caplog.records)
