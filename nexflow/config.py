"""Config loading: YAML defaults layered with environment variable overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import json as _json

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict to dotted keys, e.g. {'app': {'env': 'dev'}} → {'APP__ENV': 'dev'}."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        full = f"{prefix}__{k}".upper() if prefix else k.upper()
        if isinstance(v, dict):
            out.update(_flatten(v, full))
        else:
            out[full] = v
    return out


class AppConfig(BaseSettings):
    env: str = "development"
    log_level: str = "INFO"


class ExchangeConfig(BaseSettings):
    name: str = "bitget"
    ws_url: str = "wss://ws.bitget.com/v2/ws/public"
    ws_ping_interval: int = 20
    ws_reconnect_delay: int = 5
    ws_max_reconnect_attempts: int = 10
    ws_connect_timeout: int = 10


class MarketDataConfig(BaseSettings):
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT"])
    product_type: str = "USDT-FUTURES"
    orderbook_depth: int = 20
    max_trade_history: int = 100

    @field_validator("symbols", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class NexFlowConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="NEXFLOW_",
        case_sensitive=False,
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)


@lru_cache(maxsize=1)
def get_config() -> NexFlowConfig:
    """Return the singleton config, loading YAML then applying env-var overrides."""
    yaml_data = _load_yaml(_DEFAULT_CONFIG_PATH)

    # Resolve optional user-specified config path
    user_path = os.environ.get("NEXFLOW_CONFIG_PATH")
    if user_path:
        yaml_data.update(_load_yaml(Path(user_path)))

    # Inject YAML values as env vars so Pydantic picks them up at lower priority than real env vars
    flat = _flatten(yaml_data)
    for key, value in flat.items():
        env_key = f"NEXFLOW_{key}"
        if env_key not in os.environ:
            os.environ[env_key] = _json.dumps(value) if isinstance(value, (list, dict)) else str(value)

    return NexFlowConfig()
