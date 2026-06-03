#!/usr/bin/env python3
"""Standalone news & sentiment check for NexFlow.

Fetches Fear & Greed index + recent CryptoPanic headlines, analyzes
them with Claude, and prints a structured market sentiment report.

Run manually:
    python scripts/run_news_check.py

Or integrated into the duo/trio runner — called once per daily check.

Environment variables:
    ANTHROPIC_API_KEY    — Claude API key (for AI analysis)
    CRYPTOPANIC_API_KEY  — CryptoPanic API key (optional, improves rate limit)

Without ANTHROPIC_API_KEY: falls back to Fear & Greed only.
Without CRYPTOPANIC_API_KEY: uses public CryptoPanic feed (may be rate-limited).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nexflow.services.news.fetcher import fetch_fear_greed, fetch_crypto_news
from nexflow.services.news.analyzer import (
    analyze_sentiment,
    sentiment_from_fear_greed_only,
    MarketSentiment,
)


def run_news_check(verbose: bool = True) -> MarketSentiment:
    """Fetch and analyze current market sentiment. Returns MarketSentiment."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if verbose:
        print(f"[{ts}] Market sentiment check")
        print()

    # 1. Fear & Greed
    fg = fetch_fear_greed()
    if fg and verbose:
        bar = "█" * (fg.value // 5) + "░" * (20 - fg.value // 5)
        print(f"  Fear & Greed : {fg.value:>3}/100  {fg.label}")
        print(f"  [{bar}]")
        print()

    # 2. News headlines
    news = fetch_crypto_news(currencies="BTC,ETH", limit=20)
    if verbose:
        if news:
            print(f"  Latest headlines ({len(news)} fetched):")
            for item in news[:8]:
                age = ""
                try:
                    pub = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
                    mins_ago = int((datetime.now(timezone.utc) - pub).total_seconds() / 60)
                    age = f" ({mins_ago}m ago)" if mins_ago < 60 else f" ({mins_ago//60}h ago)"
                except Exception:
                    pass
                votes = f"+{item.votes_positive}/-{item.votes_negative}" if item.votes_positive + item.votes_negative > 0 else ""
                print(f"    [{item.source}]{age} {item.title} {votes}")
            print()
        else:
            print("  No news fetched (CryptoPanic unavailable or rate-limited)")
            print()

    # 3. Claude analysis
    sentiment = None
    if fg or news:
        sentiment = analyze_sentiment(news, fg)

    if sentiment is None and fg:
        sentiment = sentiment_from_fear_greed_only(fg)

    if sentiment is None:
        # Total fallback
        from nexflow.services.news.analyzer import MarketSentiment
        sentiment = MarketSentiment(
            overall_bias="NEUTRAL", confidence=0.0,
            suspend_new_longs=False, fear_greed_value=50,
            fear_greed_label="Unknown", key_events=[],
            reasoning="No data available",
        )

    if verbose:
        bias_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(sentiment.overall_bias, "⚪")
        print(f"  {bias_icon} Sentiment    : {sentiment.overall_bias}  (confidence {sentiment.confidence:.0%})")
        print(f"  Suspend longs : {'YES ⚠' if sentiment.suspend_new_longs else 'no'}")
        print(f"  Reasoning     : {sentiment.reasoning}")
        if sentiment.key_events:
            print(f"  Key events:")
            for ev in sentiment.key_events:
                print(f"    • {ev}")
        print()

        # Trading implication
        if sentiment.suspend_new_longs:
            print("  ⚠  TRADING IMPACT: Suspending new long entries this session")
            print("     (existing positions remain open)")
        elif sentiment.overall_bias == "BEARISH" and sentiment.confidence >= 0.6:
            print("  ↓  TRADING IMPACT: Bearish news — heightened caution on new longs")
        elif sentiment.overall_bias == "BULLISH" and sentiment.confidence >= 0.6:
            print("  ↑  TRADING IMPACT: Bullish news — normal operation, trend signals trusted")
        else:
            print("  →  TRADING IMPACT: Neutral — relying on price signals only")

    return sentiment


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()
    run_news_check(verbose=not args.quiet)


if __name__ == "__main__":
    main()
