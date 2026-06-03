"""News and sentiment fetcher for crypto market signals.

Sources:
  1. Alternative.me Fear & Greed Index (free, no key)
  2. CryptoPanic news API (free tier — set CRYPTOPANIC_API_KEY env var,
     or leave unset to use public feed without key)
  3. Bitget funding rate snapshot (already in codebase)

Used by the news_signal module to produce a NexFlow MarketSentiment object.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


_FEAR_GREED_URL  = "https://api.alternative.me/fng/?limit=3&format=json"
_CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
_TIMEOUT_S       = 8


@dataclass
class FearGreed:
    value: int          # 0–100
    label: str          # "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
    timestamp: int      # unix seconds


@dataclass
class NewsItem:
    title: str
    published_at: str   # ISO 8601
    source: str
    url: str
    votes_positive: int = 0
    votes_negative: int = 0
    currencies: list[str] = field(default_factory=list)


def fetch_fear_greed() -> Optional[FearGreed]:
    """Return current Fear & Greed index. Returns None on network failure."""
    try:
        req = urllib.request.Request(
            _FEAR_GREED_URL,
            headers={"User-Agent": "NexFlow/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        entry = data["data"][0]
        return FearGreed(
            value     = int(entry["value"]),
            label     = entry["value_classification"],
            timestamp = int(entry["timestamp"]),
        )
    except Exception as exc:
        print(f"  [news] Fear&Greed fetch failed: {exc}")
        return None


def fetch_crypto_news(
    currencies: str = "BTC,ETH",
    limit: int = 20,
) -> list[NewsItem]:
    """Fetch recent crypto news from CryptoPanic.

    Uses CRYPTOPANIC_API_KEY env var if set (higher rate limit).
    Falls back to public (unauthenticated) endpoint.
    """
    api_key = os.getenv("CRYPTOPANIC_API_KEY", "")
    params  = f"currencies={currencies}&limit={limit}&public=true"
    if api_key:
        params = f"auth_token={api_key}&{params}"
    url = f"{_CRYPTOPANIC_URL}?{params}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NexFlow/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        items = []
        for post in data.get("results", []):
            currencies_list = [c["code"] for c in post.get("currencies", [])]
            votes = post.get("votes", {})
            items.append(NewsItem(
                title          = post.get("title", ""),
                published_at   = post.get("published_at", ""),
                source         = post.get("source", {}).get("title", ""),
                url            = post.get("url", ""),
                votes_positive = votes.get("positive", 0),
                votes_negative = votes.get("negative", 0),
                currencies     = currencies_list,
            ))
        return items
    except Exception as exc:
        print(f"  [news] CryptoPanic fetch failed: {exc}")
        return []
