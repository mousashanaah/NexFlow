"""News and sentiment fetcher for crypto trading signals.

Sources (all free, no API keys required):
  1. Alternative.me Fear & Greed Index
  2. Google News RSS (crypto headlines, no key needed)
  3. CryptoPanic (optional — set CRYPTOPANIC_API_KEY for higher rate limits)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from xml.etree import ElementTree


_FEAR_GREED_URL  = "https://api.alternative.me/fng/?limit=3&format=json"
_CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
_TIMEOUT_S       = 8

# Google News RSS — free, no key, no registration
_GOOGLE_NEWS_QUERIES = [
    "Bitcoin cryptocurrency",
    "Ethereum crypto market",
    "crypto regulation SEC",
]


@dataclass
class FearGreed:
    value: int          # 0–100
    label: str          # "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
    timestamp: int      # unix seconds


@dataclass
class NewsItem:
    title: str
    published_at: str
    source: str
    url: str
    votes_positive: int = 0
    votes_negative: int = 0
    currencies: list[str] = field(default_factory=list)


def fetch_fear_greed() -> Optional[FearGreed]:
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


def fetch_google_news(max_items: int = 20) -> list[NewsItem]:
    """Fetch crypto headlines from Google News RSS — completely free."""
    items: list[NewsItem] = []
    seen: set[str] = set()

    for query in _GOOGLE_NEWS_QUERIES:
        if len(items) >= max_items:
            break
        encoded = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NexFlow/1.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                xml_data = resp.read()
            root = ElementTree.fromstring(xml_data)
            channel = root.find("channel")
            if channel is None:
                continue
            for item in channel.findall("item"):
                title = item.findtext("title", "").strip()
                # Google News appends " - Source Name" to every title
                source = ""
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title  = parts[0].strip()
                    source = parts[1].strip()
                pub = item.findtext("pubDate", "")
                link = item.findtext("link", "")
                if title and title not in seen:
                    seen.add(title)
                    items.append(NewsItem(
                        title        = title,
                        published_at = pub,
                        source       = source,
                        url          = link,
                    ))
        except Exception as exc:
            print(f"  [news] Google News fetch failed for '{query}': {exc}")

    return items[:max_items]


def fetch_crypto_news(currencies: str = "BTC,ETH", limit: int = 20) -> list[NewsItem]:
    """Fetch from CryptoPanic (optional key) or fall back to Google News."""
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
    except Exception:
        # Fall back to Google News silently
        return fetch_google_news(max_items=limit)
