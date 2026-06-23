"""
Signal Attribution Framework — Week 3

Records every signal value at the moment of discovery so we can later
measure which signals actually predicted outcomes.

The fundamental question this module will eventually answer:

    "Of risk_score, wallet_score, narrative, liquidity, volume_ratio,
     and age — which correlated most strongly with 7d/30d winners?"

Design principle: capture now, measure later.
Every signal is frozen at discovery.  Outcome columns are filled by the
tracker.  Correlation analysis runs when the dataset is large enough to
be statistically meaningful (~50+ classified outcomes per bucket).

Schema
------
signal_snapshots  — one row per discovery, all signals at time of capture
signal_correlations — computed correlation coefficients (filled by analysis)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_DDL = """
CREATE TABLE IF NOT EXISTS signal_snapshots (
    snapshot_id         TEXT PRIMARY KEY,   -- pair_address + ':' + discovery_ts
    pair_address        TEXT NOT NULL,
    token_address       TEXT NOT NULL,
    token_symbol        TEXT,
    chain_id            TEXT NOT NULL,
    discovery_ts        TEXT NOT NULL,

    -- Risk signals
    risk_score          INTEGER,
    risk_label          TEXT,
    risk_passed         INTEGER,

    -- Liquidity / volume signals (at discovery)
    liquidity_usd       REAL,
    volume_24h          REAL,
    volume_liq_ratio    REAL,   -- volume_24h / liquidity_usd (momentum proxy)
    market_cap          REAL,
    age_hours           REAL,

    -- Wallet signals
    wallet_score        INTEGER,    -- None until outcome-backed
    wallet_outcome_backed INTEGER DEFAULT 0,
    wallets_tracked     INTEGER DEFAULT 0,

    -- Narrative signals
    narrative_category  TEXT,
    narrative_confidence REAL,

    -- FOMO tradability
    fomo_available      INTEGER DEFAULT 0,
    fomo_listed_ts      TEXT,
    fomo_delay_hours    REAL,   -- hours between discovery and FOMO listing

    -- Opportunity score (computed at snapshot time)
    opportunity_score   INTEGER,

    -- Outcomes (filled by tracker, mirrors alpha_memory)
    return_1d           REAL,
    return_7d           REAL,
    return_30d          REAL,
    classification      TEXT,   -- Winner | Neutral | Failure | Rug

    last_updated_ts     TEXT
);

CREATE INDEX IF NOT EXISTS idx_ss_pair      ON signal_snapshots(pair_address);
CREATE INDEX IF NOT EXISTS idx_ss_ts        ON signal_snapshots(discovery_ts);
CREATE INDEX IF NOT EXISTS idx_ss_class     ON signal_snapshots(classification);
CREATE INDEX IF NOT EXISTS idx_ss_narrative ON signal_snapshots(narrative_category);

-- Correlation coefficients computed by analysis script
CREATE TABLE IF NOT EXISTS signal_correlations (
    computed_ts         TEXT NOT NULL,
    outcome_period      TEXT NOT NULL,  -- "7d" | "30d"
    signal_name         TEXT NOT NULL,
    pearson_r           REAL,
    sample_size         INTEGER,
    PRIMARY KEY (outcome_period, signal_name)
);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_attribution(path: Path) -> None:
    with _connect(path) as conn:
        conn.executescript(_DDL)


# ── Opportunity Score ─────────────────────────────────────────────────────────

def compute_opportunity_score(
    risk_passed:        bool,
    risk_score:         Optional[int],
    liquidity_usd:      Optional[float],
    volume_24h:         Optional[float],
    age_hours:          Optional[float],
    wallet_score:       Optional[int],
    wallet_outcome_backed: bool,
    narrative_category: Optional[str],
    narrative_win_rate: Optional[float],
    fomo_available:     bool = False,
) -> int:
    """
    Proposed opportunity score formula (0–105).

    Design rationale:
    - Risk gate is binary (blocked = 0, no exceptions)
    - Freshness matters more than size for early discovery
    - Volume/liquidity ratio is a better momentum signal than raw volume
    - Wallet and narrative scores only apply when evidence-backed
    - FOMO availability adds 5 pts to surface actionable opportunities

    This formula is a PRIOR, not a posterior.  It will be revised once
    the attribution dataset has enough classified outcomes to measure
    actual signal quality.  Do not optimize it until then.
    """
    if not risk_passed:
        return 0

    score = 0

    # ── Liquidity layer (0–20 pts) ────────────────────────────────────────────
    if liquidity_usd is not None:
        if liquidity_usd >= 1_000_000:
            score += 20
        elif liquidity_usd >= 100_000:
            score += 15
        elif liquidity_usd >= 10_000:
            score += 10
        else:
            score += 5

    # ── Momentum: volume/liquidity ratio (0–25 pts) ───────────────────────────
    if liquidity_usd and volume_24h and liquidity_usd > 0:
        ratio = volume_24h / liquidity_usd
        if ratio >= 3.0:
            score += 25
        elif ratio >= 1.0:
            score += 20
        elif ratio >= 0.5:
            score += 12
        elif ratio >= 0.1:
            score += 6
        # else: 0 — no volume activity

    # ── Freshness (0–25 pts) — this is the core early-discovery signal ────────
    if age_hours is not None:
        if age_hours <= 1:
            score += 25
        elif age_hours <= 6:
            score += 20
        elif age_hours <= 12:
            score += 15
        elif age_hours <= 24:
            score += 10
        elif age_hours <= 48:
            score += 5

    # ── Risk quality (0–15 pts, only when checked) ────────────────────────────
    if risk_score is not None:
        if risk_score >= 18:
            score += 15
        elif risk_score >= 14:
            score += 10
        elif risk_score >= 10:
            score += 5
        # < 10 = adds nothing (barely passed)

    # ── Wallet score (0–15 pts, ONLY when outcome-backed) ─────────────────────
    if wallet_score is not None and wallet_outcome_backed:
        if wallet_score >= 80:
            score += 15
        elif wallet_score >= 65:
            score += 10
        elif wallet_score >= 50:
            score += 5

    # ── Narrative (0–10 pts, ONLY when win rate is evidence-backed) ───────────
    if narrative_category == "SCAM_SIGNAL":
        score -= 15
    elif narrative_win_rate is not None:
        if narrative_win_rate >= 0.6:
            score += 10
        elif narrative_win_rate >= 0.4:
            score += 5

    # ── FOMO actionability bonus (+5) ─────────────────────────────────────────
    if fomo_available:
        score += 5

    return max(0, min(105, score))


# ── Write ──────────────────────────────────────────────────────────────────────

def record_snapshot(
    pair_address:        str,
    token_address:       str,
    token_symbol:        str,
    chain_id:            str,
    discovery_ts:        str,
    risk_score:          Optional[int],
    risk_label:          Optional[str],
    risk_passed:         bool,
    liquidity_usd:       Optional[float],
    volume_24h:          Optional[float],
    market_cap:          Optional[float],
    age_hours:           Optional[float],
    wallet_score:        Optional[int],
    wallet_outcome_backed: bool,
    wallets_tracked:     int,
    narrative_category:  Optional[str],
    narrative_confidence: Optional[float],
    narrative_win_rate:  Optional[float],
    fomo_available:      bool,
    path:                Path,
) -> str:
    """Record a signal snapshot. Returns snapshot_id."""
    now = datetime.now(timezone.utc).isoformat()
    snapshot_id = f"{pair_address}:{discovery_ts}"

    vol_liq_ratio = (
        (volume_24h / liquidity_usd)
        if (volume_24h and liquidity_usd and liquidity_usd > 0) else None
    )

    opp_score = compute_opportunity_score(
        risk_passed         = risk_passed,
        risk_score          = risk_score,
        liquidity_usd       = liquidity_usd,
        volume_24h          = volume_24h,
        age_hours           = age_hours,
        wallet_score        = wallet_score,
        wallet_outcome_backed = wallet_outcome_backed,
        narrative_category  = narrative_category,
        narrative_win_rate  = narrative_win_rate,
        fomo_available      = fomo_available,
    )

    with _connect(path) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO signal_snapshots (
                snapshot_id, pair_address, token_address, token_symbol, chain_id,
                discovery_ts, risk_score, risk_label, risk_passed,
                liquidity_usd, volume_24h, volume_liq_ratio, market_cap, age_hours,
                wallet_score, wallet_outcome_backed, wallets_tracked,
                narrative_category, narrative_confidence,
                fomo_available, opportunity_score, last_updated_ts
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            snapshot_id, pair_address, token_address, token_symbol, chain_id,
            discovery_ts, risk_score, risk_label, int(risk_passed),
            liquidity_usd, volume_24h, vol_liq_ratio, market_cap, age_hours,
            wallet_score, int(wallet_outcome_backed), wallets_tracked,
            narrative_category, narrative_confidence,
            int(fomo_available), opp_score, now,
        ))
    return snapshot_id


def update_snapshot_outcome(
    pair_address:   str,
    return_1d:      Optional[float],
    return_7d:      Optional[float],
    return_30d:     Optional[float],
    classification: Optional[str],
    path:           Path,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        conn.execute("""
            UPDATE signal_snapshots
            SET return_1d=?, return_7d=?, return_30d=?,
                classification=?, last_updated_ts=?
            WHERE pair_address=? AND classification IS NULL
        """, (return_1d, return_7d, return_30d, classification, now, pair_address))


def update_fomo_availability(
    token_address:  str,
    fomo_available: bool,
    fomo_listed_ts: Optional[str],
    path:           Path,
) -> None:
    """Update FOMO tradability fields when listing is detected."""
    with _connect(path) as conn:
        # Compute delay from discovery to FOMO listing
        row = conn.execute(
            "SELECT discovery_ts FROM signal_snapshots WHERE token_address=? "
            "ORDER BY discovery_ts LIMIT 1",
            (token_address,)
        ).fetchone()
        delay = None
        if row and fomo_listed_ts:
            try:
                disc  = datetime.fromisoformat(row["discovery_ts"].replace("Z", "+00:00"))
                fomo  = datetime.fromisoformat(fomo_listed_ts.replace("Z", "+00:00"))
                delay = (fomo - disc).total_seconds() / 3600
            except Exception:
                pass

        conn.execute("""
            UPDATE signal_snapshots
            SET fomo_available=?, fomo_listed_ts=?, fomo_delay_hours=?,
                opportunity_score = opportunity_score + CASE WHEN ? THEN 5 ELSE 0 END
            WHERE token_address=?
        """, (int(fomo_available), fomo_listed_ts, delay, int(fomo_available), token_address))


# ── Read / Analysis ────────────────────────────────────────────────────────────

def load_for_analysis(path: Path, min_classified: int = 10) -> list[dict]:
    """Return snapshots that have both signals and classified outcomes."""
    with _connect(path) as conn:
        rows = conn.execute("""
            SELECT * FROM signal_snapshots
            WHERE classification IS NOT NULL
            ORDER BY discovery_ts DESC
        """).fetchall()
    return [dict(r) for r in rows]


def signal_quality_report(path: Path) -> dict:
    """
    Compute per-signal correlation with 7d outcomes.
    Returns a dict keyed by signal name with correlation stats.
    Requires at least 10 classified outcomes to produce meaningful results.
    """
    rows = load_for_analysis(path)
    if len(rows) < 10:
        return {"error": f"only {len(rows)} classified outcomes — need at least 10"}

    # Simple metric: average return_7d for top-half vs bottom-half of each signal
    signals = [
        ("risk_score",       lambda r: r.get("risk_score")),
        ("liquidity_usd",    lambda r: r.get("liquidity_usd")),
        ("volume_liq_ratio", lambda r: r.get("volume_liq_ratio")),
        ("age_hours",        lambda r: r.get("age_hours")),
        ("opportunity_score",lambda r: r.get("opportunity_score")),
    ]

    report = {}
    for sig_name, getter in signals:
        valid = [(getter(r), r.get("return_7d")) for r in rows
                 if getter(r) is not None and r.get("return_7d") is not None]
        if len(valid) < 5:
            report[sig_name] = {"n": len(valid), "status": "insufficient_data"}
            continue

        valid.sort(key=lambda x: x[0])
        mid = len(valid) // 2
        bottom_half = [v[1] for v in valid[:mid]]
        top_half    = [v[1] for v in valid[mid:]]

        avg_bottom = sum(bottom_half) / len(bottom_half)
        avg_top    = sum(top_half)    / len(top_half)

        report[sig_name] = {
            "n":           len(valid),
            "avg_ret_top_half":    round(avg_top,    3),
            "avg_ret_bottom_half": round(avg_bottom, 3),
            "top_minus_bottom":    round(avg_top - avg_bottom, 3),
            "status":      "ok",
        }
    return report


def attribution_stats(path: Path) -> dict:
    """Quick summary for board footer."""
    with _connect(path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM signal_snapshots").fetchone()[0]
        with_outcome = conn.execute(
            "SELECT COUNT(*) FROM signal_snapshots WHERE classification IS NOT NULL"
        ).fetchone()[0]
        fomo_count = conn.execute(
            "SELECT COUNT(*) FROM signal_snapshots WHERE fomo_available=1"
        ).fetchone()[0]
    return {
        "total_snapshots": total,
        "classified":      with_outcome,
        "fomo_available":  fomo_count,
    }
