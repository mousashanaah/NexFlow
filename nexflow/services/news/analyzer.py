"""Claude-powered news sentiment analyzer for crypto trading signals.

Sends recent headlines + Fear & Greed to Claude API and gets back a
structured MarketSentiment object with:
  - overall_bias: "BULLISH" | "BEARISH" | "NEUTRAL"
  - confidence: 0.0–1.0
  - suspend_new_longs: bool  (true on extreme negative events)
  - key_events: list of the 3 most market-moving items
  - reasoning: one-sentence summary

Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from .fetcher import FearGreed, NewsItem


@dataclass
class MarketSentiment:
    overall_bias: str           # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence: float           # 0.0–1.0
    suspend_new_longs: bool     # True = hold off opening new positions
    fear_greed_value: int       # 0–100 raw index
    fear_greed_label: str       # e.g. "Extreme Fear"
    key_events: list[str]       # top 3 market-moving headlines
    reasoning: str              # one-line summary
    raw_response: str = ""


def analyze_sentiment(
    news_items: list[NewsItem],
    fear_greed: Optional[FearGreed],
    model: str = "claude-haiku-4-5-20251001",
) -> Optional[MarketSentiment]:
    """Call Claude API to analyze news sentiment. Returns None if API unavailable."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  [news] ANTHROPIC_API_KEY not set — skipping AI analysis")
        return None

    fg_text = ""
    if fear_greed:
        fg_text = f"\nFear & Greed Index: {fear_greed.value}/100 ({fear_greed.label})"

    headlines = "\n".join(
        f"- [{item.source}] {item.title} (+{item.votes_positive}/-{item.votes_negative})"
        for item in news_items[:15]
    )

    prompt = f"""You are a crypto market analyst. Analyze the following recent news and sentiment data for Bitcoin and Ethereum trading signals.
{fg_text}

Recent headlines:
{headlines}

Respond with a JSON object (and nothing else) with these exact fields:
{{
  "overall_bias": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <0.0 to 1.0>,
  "suspend_new_longs": <true if there is an extreme negative event like exchange hack, regulatory ban, major crash — false otherwise>,
  "key_events": ["<most important headline 1>", "<headline 2>", "<headline 3>"],
  "reasoning": "<one sentence explaining the overall assessment>"
}}

Be conservative: only say BULLISH/BEARISH with confidence > 0.6 if there is clear directional news. Default to NEUTRAL for routine market commentary."""

    import urllib.request
    try:
        payload = json.dumps({
            "model": model,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data    = payload,
            headers = {
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        raw = result["content"][0]["text"].strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())

        fg_val   = fear_greed.value if fear_greed else 50
        fg_label = fear_greed.label if fear_greed else "Unknown"

        return MarketSentiment(
            overall_bias      = parsed.get("overall_bias", "NEUTRAL"),
            confidence        = float(parsed.get("confidence", 0.5)),
            suspend_new_longs = bool(parsed.get("suspend_new_longs", False)),
            fear_greed_value  = fg_val,
            fear_greed_label  = fg_label,
            key_events        = parsed.get("key_events", []),
            reasoning         = parsed.get("reasoning", ""),
            raw_response      = raw,
        )
    except Exception as exc:
        print(f"  [news] Claude analysis failed: {exc}")
        return None


def sentiment_from_fear_greed_only(fg: FearGreed) -> MarketSentiment:
    """Fallback: derive simple sentiment from Fear & Greed alone (no API needed)."""
    v = fg.value
    if v <= 20:
        bias  = "BEARISH"; conf = 0.7; suspend = True
    elif v <= 35:
        bias  = "BEARISH"; conf = 0.5; suspend = False
    elif v <= 60:
        bias  = "NEUTRAL"; conf = 0.4; suspend = False
    elif v <= 80:
        bias  = "BULLISH"; conf = 0.5; suspend = False
    else:
        bias  = "BULLISH"; conf = 0.65; suspend = False

    return MarketSentiment(
        overall_bias      = bias,
        confidence        = conf,
        suspend_new_longs = suspend,
        fear_greed_value  = fg.value,
        fear_greed_label  = fg.label,
        key_events        = [],
        reasoning         = f"Fear & Greed {fg.value} ({fg.label}) — no news API",
    )
