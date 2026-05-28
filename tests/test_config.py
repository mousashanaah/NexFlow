"""Smoke tests for config loading."""

import os

import pytest

from nexflow.config import get_config, NexFlowConfig


def test_get_config_returns_nexflow_config() -> None:
    cfg = get_config()
    assert isinstance(cfg, NexFlowConfig)


def test_default_exchange_url() -> None:
    cfg = get_config()
    assert cfg.exchange.ws_url.startswith("wss://")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear the lru_cache so the override is picked up
    get_config.cache_clear()
    monkeypatch.setenv("NEXFLOW_APP__LOG_LEVEL", "DEBUG")
    cfg = get_config()
    assert cfg.app.log_level == "DEBUG"
    get_config.cache_clear()
