"""
Narrative Categorizer — Week 3

Assigns a primary narrative category to each discovered token using
rule-based pattern matching on token name, symbol, and DexScreener
metadata.  No external API calls, no ML.

Categories are designed to track macro-narrative cycles, not individual
hype events.  The question this module answers:

    "Is this token participating in a narrative that produces winners?"

As Alpha Memory accumulates outcomes, win rates per narrative become
meaningful signal for early discovery.

Categories
----------
AI_AGENTS    — AI, agents, LLM-adjacent tokens (historically early-mover)
GAMING       — GameFi, play-to-earn, metaverse
DEFI         — Yield, liquidity, DEX, lending protocols
RWA          — Real-world assets, tokenized finance
INFRA        — L1/L2 infrastructure, bridges, oracles, wallets
MEME         — Meme coins with no stated utility
ANIMAL       — Animal-themed memes (distinct sub-category, historically high vol)
SOCIAL       — Social-fi, creator economy, SocialFi
SPORTS       — Sports betting, fantasy, athlete tokens
SCAM_SIGNAL  — Name/symbol patterns strongly correlated with low-effort launches
OTHER        — Does not match any above

Each token receives one primary category plus a confidence score (0–1)
indicating how many signals matched vs how ambiguous the classification was.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NarrativeResult:
    token_address:  str
    token_name:     str
    token_symbol:   str
    category:       str          # primary category
    confidence:     float        # 0.0–1.0
    matched_signals: list[str]   = field(default_factory=list)
    secondary:      Optional[str] = None


# ── Keyword rules ─────────────────────────────────────────────────────────────
# Each entry: (category, weight, patterns_list)
# Patterns are matched against the lowercased name+symbol string.
# Weight sums drive primary vs secondary category selection.

_RULES: list[tuple[str, float, list[str]]] = [

    # AI Agents — strongest signal category (historically correlated with early wins)
    ("AI_AGENTS", 1.0, [
        r"\bai\b", r"\bagent", r"\bgpt\b", r"\bllm\b", r"\bneural",
        r"\bartificial", r"\bmachine.?learn", r"\bdeep.?learn",
        r"\bautonomous", r"\bchat\b", r"\bclaude\b", r"\bgemini\b",
        r"\bintelligent", r"\bsentient", r"\bcognitive", r"\bcompute\b",
        r"\boracle.?ai", r"\bai.?agent", r"\bsmart.?agent",
    ]),

    # Gaming
    ("GAMING", 0.9, [
        r"\bgame\b", r"\bgaming", r"\bplay", r"\bquest\b", r"\bquest",
        r"\bnft.?game", r"\bmetaverse", r"\bguild\b", r"\bitem\b",
        r"\bloot\b", r"\bhero\b", r"\bwarrior", r"\blegend\b",
        r"\bworld\b.*\bgame", r"\bcraft\b", r"\bcraft$",
        r"\barcade\b", r"\bpixel\b.*\bgame", r"\bsaga\b",
    ]),

    # DeFi — yield, liquidity, lending
    ("DEFI", 0.9, [
        r"\byield\b", r"\bliquid", r"\bstake\b", r"\bstaking",
        r"\bvault\b", r"\bpool\b", r"\bswap\b", r"\bdex\b",
        r"\blend\b", r"\bborrow\b", r"\bprotocol\b", r"\bfinance\b",
        r"\bdefi\b", r"\bamm\b", r"\bfarm\b", r"\bfarm$",
        r"\brebase\b", r"\bsynthetic\b", r"\bperp\b", r"\bperps\b",
        r"\bderivative", r"\bfutures\b", r"\blp\b", r"\bvote\b",
        r"\bgovernance\b", r"\bdao\b",
    ]),

    # Real-World Assets
    ("RWA", 0.95, [
        r"\brwa\b", r"\breal.?world", r"\btokenized", r"\breal.?estate",
        r"\bcommodit", r"\bgold\b", r"\bsilver\b", r"\btreasur",
        r"\bbond\b", r"\bequit", r"\bstock\b.*\btoken", r"\bsecurit",
        r"\bcredit\b", r"\binvoice\b", r"\btrade.?financ",
    ]),

    # Infrastructure
    ("INFRA", 0.85, [
        r"\bbridge\b", r"\boracle\b", r"\bwallet\b", r"\bl2\b",
        r"\blayer.?2", r"\brollup\b", r"\bzk\b", r"\bzero.?know",
        r"\bproof\b", r"\bscal", r"\bnode\b", r"\bvalidat",
        r"\binfra\b", r"\bprotocol\b.*\binfra", r"\bstorage\b",
        r"\bcompute\b", r"\brpc\b", r"\bnetwork\b",
    ]),

    # Social-fi
    ("SOCIAL", 0.85, [
        r"\bsocial\b", r"\bcreator\b", r"\binfluenc", r"\bfan\b",
        r"\bfriend\b", r"\bcommunity\b", r"\breputation\b",
        r"\bidentity\b", r"\bprofile\b", r"\bpost\b",
    ]),

    # Sports
    ("SPORTS", 0.9, [
        r"\bsport\b", r"\bfootball\b", r"\bsoccer\b", r"\bbasketball\b",
        r"\bnfl\b", r"\bnba\b", r"\bmlb\b", r"\bnhl\b", r"\bufc\b",
        r"\bfantasy\b", r"\bbet\b", r"\bwager\b", r"\bpredict\b",
        r"\bchampion\b", r"\bseason\b", r"\bteam\b.*\btoken",
    ]),

    # Animal memes (high-volume sub-category, track separately from general meme)
    ("ANIMAL", 0.95, [
        r"\bdog\b", r"\bdoge\b", r"\bcat\b", r"\bkitty\b", r"\bfrog\b",
        r"\bpepe\b", r"\bpanda\b", r"\bbear\b", r"\bbull\b", r"\bape\b",
        r"\bmonkey\b", r"\bbird\b", r"\bduck\b", r"\bwolf\b", r"\bfox\b",
        r"\bsheep\b", r"\belephant\b", r"\blion\b", r"\btiger\b",
        r"\bbunny\b", r"\bhamster\b", r"\bshib\b", r"\binu\b",
        r"\bfloki\b", r"\bcorgi\b", r"\bkong\b", r"\bbonk\b",
        r"\bcapy\b", r"\bcapybara\b", r"\bgoat\b", r"\bcow\b",
        r"\bpig\b", r"\bhorse\b", r"\bchicken\b", r"\bsnake\b",
    ]),

    # Meme (general — lower confidence than ANIMAL since much broader)
    ("MEME", 0.7, [
        r"\bmeme\b", r"\bfun\b$", r"\blol\b", r"\bwgmi\b", r"\bngmi\b",
        r"\bmoon\b", r"\brocket\b", r"\blambо\b", r"\bvibes\b",
        r"\bsers\b", r"\bfrens\b", r"\bwoof\b", r"\bmeow\b",
        r"\bpump\b", r"\bdump\b", r"\bstonk\b", r"\bChad\b",
        r"\bwagmi\b", r"\bshill\b",
    ]),

    # Scam signals — patterns correlated with low-effort/rug launches
    ("SCAM_SIGNAL", 1.0, [
        r"\b100x\b", r"\b1000x\b", r"\bguaranteed\b", r"\bsafe\b.*\bmoon",
        r"\blegit\b", r"\bnotrug\b", r"\bnot.?a.?rug", r"\bsafu\b.*\bmoon",
        r"\bfirst\b.*\bever\b", r"\bfastest\b.*\bgrow",
    ]),
]


# ── Core logic ────────────────────────────────────────────────────────────────

def categorize(
    token_name:    str,
    token_symbol:  str,
    token_address: str = "",
    description:   str = "",
) -> NarrativeResult:
    """
    Assign a narrative category to a token.
    Matches against name + symbol + description (all lowercased).
    """
    text = " ".join([
        (token_name    or "").lower(),
        (token_symbol  or "").lower(),
        (description   or "").lower(),
    ])

    scores: dict[str, float] = {}
    signals: dict[str, list[str]] = {}

    for category, weight, patterns in _RULES:
        hit_patterns: list[str] = []
        for pat in patterns:
            if re.search(pat, text):
                hit_patterns.append(pat)
        if hit_patterns:
            scores[category]  = scores.get(category, 0.0) + weight * len(hit_patterns)
            signals[category] = signals.get(category, []) + hit_patterns

    if not scores:
        return NarrativeResult(
            token_address    = token_address,
            token_name       = token_name,
            token_symbol     = token_symbol,
            category         = "OTHER",
            confidence       = 0.0,
            matched_signals  = [],
        )

    # Primary = highest-scoring category
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_cat, primary_score = ranked[0]
    secondary_cat = ranked[1][0] if len(ranked) > 1 else None

    # Confidence: ratio of primary score to sum of all scores (0–1)
    total_score = sum(scores.values())
    confidence  = min(1.0, primary_score / total_score) if total_score else 0.0

    return NarrativeResult(
        token_address   = token_address,
        token_name      = token_name,
        token_symbol    = token_symbol,
        category        = primary_cat,
        confidence      = round(confidence, 2),
        matched_signals = signals.get(primary_cat, []),
        secondary       = secondary_cat if secondary_cat != primary_cat else None,
    )


def categorize_batch(tokens: list[dict]) -> list[NarrativeResult]:
    """
    Categorize a list of token dicts.
    Each dict should have: token_address, token_name, token_symbol.
    Optional: description.
    """
    return [
        categorize(
            token_name    = t.get("token_name", ""),
            token_symbol  = t.get("token_symbol", ""),
            token_address = t.get("token_address", ""),
            description   = t.get("description", ""),
        )
        for t in tokens
    ]
