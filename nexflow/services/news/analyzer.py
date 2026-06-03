"""Crypto news sentiment analyzer.

Default: free rule-based keyword scorer — no API key, no cost, deterministic.
Optional upgrade: set ANTHROPIC_API_KEY to enable Claude analysis on top.

The rule-based engine is tuned specifically for crypto trading signals:
  - Detects extreme events (hacks, bans, collapses) → suspend_new_longs
  - Scores headline sentiment from known bullish/bearish crypto keywords
  - Combines with Fear & Greed index for final bias
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .fetcher import FearGreed, NewsItem


@dataclass
class MarketSentiment:
    overall_bias: str           # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence: float           # 0.0–1.0
    suspend_new_longs: bool     # True = extreme negative event detected
    fear_greed_value: int       # 0–100
    fear_greed_label: str
    key_events: list[str]       # top headlines driving the signal
    reasoning: str
    source: str = "rule-based"  # "rule-based" | "claude"


# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

# Extreme events → always suspend new longs regardless of other signals
_EXTREME_NEGATIVE = [
    r"\bhack(ed|ing)?\b", r"\bexploit(ed)?\b", r"\bbreached?\b",
    r"\bban(ned|ning)?\b", r"\billegal\b", r"\bshutdown\b", r"\bshut down\b",
    r"\binsolven(t|cy)\b", r"\bbankrupt\b", r"\bcollaps(e|ed|ing)\b",
    r"\brug.?pull\b", r"\bfraud\b", r"\bponzi\b",
    r"\bexchange.{0,20}(down|offline|halt)\b",
    r"\b(ftx|celsius|luna|terra|3ac|genesis).{0,30}(fail|collaps|bankrupt|insolven)\b",
    r"\bsec.{0,20}(ban|halt|emergency|crackdown)\b",
    r"\bgovernment.{0,20}(ban|seize|confiscat)\b",
    r"\bmass.{0,20}liquidat\b",
    r"\bflash.?crash\b",
]

# Strong bearish signals (don't suspend, but bias BEARISH)
_STRONG_BEARISH = [
    r"\bsec.{0,30}(lawsuit|charge|sue|enforcement|action)\b",
    r"\bregulat(ion|ory).{0,20}(crackdown|tighten|restrict)\b",
    r"\bcbdc.{0,20}(ban|replac)\b",
    r"\b(whale|institutional).{0,20}sell(ing|off)\b",
    r"\b(bitcoin|btc|eth).{0,20}(plunge|plummet|crash|dump|tank)\b",
    r"\bbear.?market\b",
    r"\bovervalued\b",
    r"\bbubble.{0,20}burst\b",
    r"\bmining.{0,20}ban\b",
    r"\benergy.{0,20}ban\b",
    r"\bcrypto.{0,20}winter\b",
]

# Strong bullish signals
_STRONG_BULLISH = [
    r"\betf.{0,20}approv(ed|al)\b",
    r"\bspot.{0,20}etf\b",
    r"\ball.?time.?high\b", r"\bath\b",
    r"\bhalving\b",
    r"\binstitutional.{0,20}(buy|invest|adopt)\b",
    r"\b(blackrock|fidelity|vanguard|jpmorgan|goldman).{0,30}(bitcoin|crypto|btc)\b",
    r"\bnational.{0,20}(reserve|adoption|legal.tender)\b",
    r"\blegal.?tender\b",
    r"\bmass.{0,20}adoption\b",
    r"\bretail.{0,20}(surge|flood|rush)\b",
    r"\b(bitcoin|btc).{0,20}(surge|soar|rally|moon|breakout)\b",
    r"\bcrypto.{0,20}(rally|bull|surge|breakout)\b",
]

# Mild bearish
_MILD_BEARISH = [
    r"\bregulat(ion|ory)\b", r"\bsec\b", r"\bcftc\b",
    r"\brate.{0,15}hike\b", r"\binterest.{0,15}rate\b",
    r"\binflation\b", r"\brecession\b",
    r"\bprofit.?taking\b", r"\bcorrect(ion|ing)\b",
    r"\bsell.?off\b", r"\bdeclin(e|ing)\b", r"\bdrops?\b",
    r"\bconcern(s|ed)?\b", r"\bwarning\b", r"\brisk(s|y)?\b",
]

# Mild bullish
_MILD_BULLISH = [
    r"\badopt(ion|ed|ing)\b", r"\bpartnership\b", r"\bintegrat(ion|ed)\b",
    r"\blaunch(ed|ing)\b", r"\bupgrad(e|ed|ing)\b", r"\bimprov(e|ed|ing)\b",
    r"\bgrow(th|ing)\b", r"\bexpand(ing|ed)?\b", r"\bpositive\b",
    r"\bstron(g|ger)\b", r"\brecov(er|ery|ering)\b",
    r"\bbullish\b", r"\boptimis(m|tic)\b",
    r"\binvest(ment|or|ing)\b",
]


def _score_headline(title: str) -> tuple[float, bool]:
    """
    Returns (score, is_extreme).
    score: negative = bearish, positive = bullish, range roughly -3 to +3
    is_extreme: True if this headline alone warrants suspending longs
    """
    t = title.lower()
    score = 0.0
    is_extreme = False

    for pattern in _EXTREME_NEGATIVE:
        if re.search(pattern, t, re.IGNORECASE):
            is_extreme = True
            score -= 3.0
            break

    if not is_extreme:
        for pattern in _STRONG_BEARISH:
            if re.search(pattern, t, re.IGNORECASE):
                score -= 1.5
        for pattern in _STRONG_BULLISH:
            if re.search(pattern, t, re.IGNORECASE):
                score += 1.5
        for pattern in _MILD_BEARISH:
            if re.search(pattern, t, re.IGNORECASE):
                score -= 0.4
        for pattern in _MILD_BULLISH:
            if re.search(pattern, t, re.IGNORECASE):
                score += 0.4

    return score, is_extreme


def analyze_with_rules(
    news_items: list[NewsItem],
    fear_greed: Optional[FearGreed],
) -> MarketSentiment:
    """
    Free rule-based sentiment analysis. No API key needed.

    Algorithm:
      1. Score each headline with keyword patterns
      2. Detect any extreme events (hack/ban/collapse) → suspend_new_longs
      3. Combine headline score with Fear & Greed index
      4. Emit BULLISH/BEARISH/NEUTRAL with confidence
    """
    fg_val   = fear_greed.value if fear_greed else 50
    fg_label = fear_greed.label if fear_greed else "Unknown"

    # Score all headlines
    scored: list[tuple[float, bool, str]] = []  # (score, extreme, title)
    any_extreme = False
    for item in news_items:
        score, extreme = _score_headline(item.title)
        scored.append((score, extreme, item.title))
        if extreme:
            any_extreme = True

    scored.sort(key=lambda x: x[0])  # most negative first

    # Aggregate headline score (trim outliers: drop top/bottom 10%)
    all_scores = [s for s, _, _ in scored]
    if len(all_scores) >= 5:
        trim = max(1, len(all_scores) // 10)
        trimmed = all_scores[trim:-trim]
    else:
        trimmed = all_scores
    headline_score = sum(trimmed) / len(trimmed) if trimmed else 0.0

    # Fear & Greed contribution: map 0–100 → -1 to +1
    fg_score = (fg_val - 50) / 50.0  # -1 = extreme fear, +1 = extreme greed

    # Combined score: headlines 60%, F&G 40%
    combined = 0.6 * headline_score + 0.4 * fg_score * 1.5

    # Determine bias and confidence
    abs_combined = abs(combined)
    if abs_combined < 0.20:
        bias = "NEUTRAL"
        conf = 0.3
    elif abs_combined < 0.50:
        bias = "BEARISH" if combined < 0 else "BULLISH"
        conf = 0.45 + abs_combined * 0.3
    else:
        bias = "BEARISH" if combined < 0 else "BULLISH"
        conf = min(0.85, 0.6 + abs_combined * 0.25)

    # Override: extreme fear in F&G always at least BEARISH
    if fear_greed and fear_greed.value <= 20 and bias == "NEUTRAL":
        bias = "BEARISH"
        conf = 0.55

    # Key events: most negative headlines (for display)
    extreme_headlines = [t for _, ext, t in scored if ext]
    negative_headlines = [t for s, _, t in scored if s <= -1.0][:3]
    positive_headlines = [t for s, _, t in sorted(scored, key=lambda x: -x[0]) if s >= 1.0][:2]
    key_events = (extreme_headlines + negative_headlines + positive_headlines)[:3]

    # Reasoning
    fg_str = f"F&G={fg_val}({fg_label})"
    hl_str = f"headlines score {headline_score:+.2f}" if all_scores else "no headlines"
    reasoning = f"{fg_str}, {hl_str} → {bias} ({conf:.0%} conf)"
    if any_extreme:
        reasoning = "EXTREME EVENT detected — " + reasoning

    return MarketSentiment(
        overall_bias      = bias,
        confidence        = conf,
        suspend_new_longs = any_extreme or (fg_val <= 15),
        fear_greed_value  = fg_val,
        fear_greed_label  = fg_label,
        key_events        = key_events,
        reasoning         = reasoning,
        source            = "rule-based",
    )


def analyze_with_claude(
    news_items: list[NewsItem],
    fear_greed: Optional[FearGreed],
    model: str = "claude-haiku-4-5-20251001",
) -> Optional[MarketSentiment]:
    """
    Optional Claude API upgrade. Returns None if ANTHROPIC_API_KEY not set.
    When available, Claude cross-checks and can catch nuanced context the
    keyword engine misses (e.g., sarcastic headlines, complex regulatory news).
    """
    import json, urllib.request
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    fg_text = f"\nFear & Greed Index: {fear_greed.value}/100 ({fear_greed.label})" if fear_greed else ""
    headlines = "\n".join(
        f"- [{item.source}] {item.title}"
        for item in news_items[:15]
    )

    prompt = f"""Analyze these crypto news headlines for short-term BTC/ETH trading impact.{fg_text}

Headlines:
{headlines}

Reply with JSON only:
{{
  "overall_bias": "BULLISH"|"BEARISH"|"NEUTRAL",
  "confidence": <0.0-1.0>,
  "suspend_new_longs": <true only for exchange hack/collapse, government ban, or >15% crash>,
  "key_events": ["<headline 1>","<headline 2>","<headline 3>"],
  "reasoning": "<one sentence>"
}}"""

    try:
        payload = json.dumps({
            "model": model, "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        raw = result["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)

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
            source            = "claude",
        )
    except Exception as exc:
        print(f"  [news] Claude analysis failed: {exc}")
        return None


def analyze_sentiment(
    news_items: list[NewsItem],
    fear_greed: Optional[FearGreed],
) -> MarketSentiment:
    """
    Main entry point. Uses Claude if ANTHROPIC_API_KEY is set, otherwise rule-based.
    Rule-based is free and works well for crypto — no degradation without the key.
    """
    claude_result = analyze_with_claude(news_items, fear_greed)
    if claude_result is not None:
        return claude_result
    return analyze_with_rules(news_items, fear_greed)


# Keep for backward compat
def sentiment_from_fear_greed_only(fg: FearGreed) -> MarketSentiment:
    return analyze_with_rules([], fg)
