"""
Wallet Intelligence Registry — Week 2

Records every wallet that appears as a top holder when a Solana token is
discovered.  As Alpha Memory classifies tokens (Winner/Neutral/Failure/Rug),
those outcomes flow back to update each wallet's track record.

A wallet that repeatedly appears early in winning tokens earns a high score.
A wallet that appears in rugs gets penalised.  A wallet that appears in 10+
tokens in 7 days with <0.5% average stake is flagged as a farm cluster.

Schema
------
wallet_appearances  — one row per (wallet, token) observation
wallet_scores       — one row per wallet, recomputed on demand

Score formula (0–100, clamped)
-------------------------------
Base 50
  +12 per Win   (max +36 from 3 wins)
  - 6 per Fail  (max -18)
  -20 per Rug   (max -40)
  + 5 if appearances >= 3
  +10 if win_rate >= 0.5
  + 5 if win_rate >= 0.7  (stacks with above)
  -30 if FARM_CLUSTER flag set
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


_DDL = """
CREATE TABLE IF NOT EXISTS wallet_appearances (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address  TEXT NOT NULL,
    token_address   TEXT NOT NULL,
    pair_address    TEXT,
    chain_id        TEXT NOT NULL,
    pct_held        REAL,       -- % of supply (0-100 scale, from RugCheck)
    holder_rank     INTEGER,    -- 1 = largest holder
    first_seen_ts   TEXT NOT NULL,
    outcome         TEXT,       -- Winner | Neutral | Failure | Rug (filled later)
    return_7d       REAL,
    UNIQUE(wallet_address, token_address)
);

CREATE TABLE IF NOT EXISTS wallet_scores (
    wallet_address  TEXT PRIMARY KEY,
    first_seen_ts   TEXT,
    last_seen_ts    TEXT,
    appearances     INTEGER DEFAULT 0,
    outcomes_known  INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    neutrals        INTEGER DEFAULT 0,
    failures        INTEGER DEFAULT 0,
    rugs            INTEGER DEFAULT 0,
    win_rate        REAL,
    avg_return_7d   REAL,
    score           INTEGER DEFAULT 50,
    flags           TEXT    -- JSON list e.g. ["FARM_CLUSTER", "REPEAT_WINNER"]
);

CREATE INDEX IF NOT EXISTS idx_wa_wallet  ON wallet_appearances(wallet_address);
CREATE INDEX IF NOT EXISTS idx_wa_token   ON wallet_appearances(token_address);
CREATE INDEX IF NOT EXISTS idx_wa_outcome ON wallet_appearances(outcome);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_wallet_registry(path: Path) -> None:
    with _connect(path) as conn:
        conn.executescript(_DDL)


# ── Write ──────────────────────────────────────────────────────────────────────

def record_appearances(
    token_address: str,
    pair_address:  str,
    chain_id:      str,
    top_holders:   list,   # raw list from RugCheck topHolders
    path:          Path,
) -> int:
    """
    Persist wallet appearances for a token's top holders.
    Returns the number of new wallet rows inserted.
    """
    if not top_holders:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    inserted = 0

    with _connect(path) as conn:
        for rank, h in enumerate(top_holders, 1):
            addr = h.get("address") or h.get("owner") or ""
            if not addr or len(addr) < 10:
                continue
            pct = h.get("pct")
            try:
                pct = float(pct) if pct is not None else None
            except (TypeError, ValueError):
                pct = None

            cur = conn.execute("""
                INSERT OR IGNORE INTO wallet_appearances (
                    wallet_address, token_address, pair_address, chain_id,
                    pct_held, holder_rank, first_seen_ts
                ) VALUES (?,?,?,?,?,?,?)
            """, (addr, token_address, pair_address, chain_id, pct, rank, now))
            inserted += cur.rowcount

    return inserted


def update_wallet_outcomes(
    token_address:  str,
    outcome:        str,          # Winner | Neutral | Failure | Rug
    return_7d:      Optional[float],
    path:           Path,
) -> None:
    """Back-fill outcome onto all wallet_appearances rows for this token."""
    with _connect(path) as conn:
        conn.execute("""
            UPDATE wallet_appearances
            SET outcome = ?, return_7d = ?
            WHERE token_address = ? AND outcome IS NULL
        """, (outcome, return_7d, token_address))


# ── Score computation ──────────────────────────────────────────────────────────

def _is_farm_cluster(wallet_address: str, conn: sqlite3.Connection) -> bool:
    """
    A wallet is a farm cluster if it appeared in 10+ distinct tokens
    within any rolling 7-day window and its average stake was < 0.5%.
    """
    rows = conn.execute("""
        SELECT first_seen_ts, pct_held
        FROM wallet_appearances
        WHERE wallet_address = ?
        ORDER BY first_seen_ts
    """, (wallet_address,)).fetchall()

    if len(rows) < 10:
        return False

    # Check any 7-day window with >= 10 appearances
    from datetime import datetime, timezone, timedelta
    timestamps = []
    pcts = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["first_seen_ts"].replace("Z", "+00:00"))
            timestamps.append(ts)
            pcts.append(r["pct_held"] or 0.0)
        except Exception:
            continue

    window = timedelta(days=7)
    for i, ts in enumerate(timestamps):
        count = sum(1 for t in timestamps if ts <= t <= ts + window)
        if count >= 10:
            avg_pct = sum(pcts) / len(pcts) if pcts else 0
            if avg_pct < 0.5:
                return True
    return False


def _compute_score(
    wins:         int,
    failures:     int,
    rugs:         int,
    outcomes:     int,
    appearances:  int,
    is_farm:      bool,
) -> tuple[int, list[str]]:
    """Return (score 0-100, flags list)."""
    flags: list[str] = []
    score = 50

    # Outcome-based adjustments
    score += min(wins * 12,    36)
    score -= min(failures * 6, 18)
    score -= min(rugs * 20,    40)

    # Experience bonus
    if appearances >= 3:
        score += 5

    # Win-rate bonus
    if outcomes >= 2:
        win_rate = wins / outcomes
        if win_rate >= 0.7:
            score += 15
            flags.append("REPEAT_WINNER")
        elif win_rate >= 0.5:
            score += 10

    # Farm penalty
    if is_farm:
        score -= 30
        flags.append("FARM_CLUSTER")

    if wins >= 2 and "REPEAT_WINNER" not in flags:
        flags.append("REPEAT_WINNER")

    return max(0, min(100, score)), flags


def recompute_wallet_score(wallet_address: str, path: Path) -> Optional[dict]:
    """Recompute and persist score for one wallet. Returns the score row or None."""
    with _connect(path) as conn:
        rows = conn.execute("""
            SELECT outcome, return_7d, first_seen_ts
            FROM wallet_appearances
            WHERE wallet_address = ?
            ORDER BY first_seen_ts
        """, (wallet_address,)).fetchall()

        if not rows:
            return None

        appearances   = len(rows)
        outcomes_data = [r for r in rows if r["outcome"]]
        wins      = sum(1 for r in outcomes_data if r["outcome"] == "Winner")
        neutrals  = sum(1 for r in outcomes_data if r["outcome"] == "Neutral")
        failures  = sum(1 for r in outcomes_data if r["outcome"] == "Failure")
        rugs      = sum(1 for r in outcomes_data if r["outcome"] == "Rug")
        outcomes_known = len(outcomes_data)

        win_rate = (wins / outcomes_known) if outcomes_known else None
        returns  = [r["return_7d"] for r in outcomes_data if r["return_7d"] is not None]
        avg_ret  = sum(returns) / len(returns) if returns else None

        first_seen = rows[0]["first_seen_ts"]
        last_seen  = rows[-1]["first_seen_ts"]

        is_farm = _is_farm_cluster(wallet_address, conn)
        score, flags = _compute_score(wins, failures, rugs, outcomes_known, appearances, is_farm)

        conn.execute("""
            INSERT INTO wallet_scores (
                wallet_address, first_seen_ts, last_seen_ts,
                appearances, outcomes_known, wins, neutrals, failures, rugs,
                win_rate, avg_return_7d, score, flags
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(wallet_address) DO UPDATE SET
                last_seen_ts    = excluded.last_seen_ts,
                appearances     = excluded.appearances,
                outcomes_known  = excluded.outcomes_known,
                wins            = excluded.wins,
                neutrals        = excluded.neutrals,
                failures        = excluded.failures,
                rugs            = excluded.rugs,
                win_rate        = excluded.win_rate,
                avg_return_7d   = excluded.avg_return_7d,
                score           = excluded.score,
                flags           = excluded.flags
        """, (
            wallet_address, first_seen, last_seen,
            appearances, outcomes_known, wins, neutrals, failures, rugs,
            win_rate, avg_ret, score, json.dumps(flags),
        ))

        return {
            "wallet_address": wallet_address,
            "score": score,
            "appearances": appearances,
            "wins": wins,
            "failures": failures,
            "rugs": rugs,
            "win_rate": win_rate,
            "flags": flags,
        }


def recompute_all_scores(path: Path) -> int:
    """Recompute scores for every wallet in the registry. Returns count updated."""
    with _connect(path) as conn:
        addrs = [r[0] for r in conn.execute(
            "SELECT DISTINCT wallet_address FROM wallet_appearances"
        ).fetchall()]

    count = 0
    for addr in addrs:
        if recompute_wallet_score(addr, path):
            count += 1
    return count


# ── Read ───────────────────────────────────────────────────────────────────────

def token_wallet_summary(token_address: str, path: Path) -> dict:
    """
    Return wallet intelligence summary for a single token.
    Used by the Alpha Board to show wallet score + explanation.

    wallet_score is None and outcome_backed=False when no outcome data exists —
    the board must NOT display a numeric score in that case.
    """
    with _connect(path) as conn:
        appearances = conn.execute("""
            SELECT wa.wallet_address, wa.pct_held, wa.holder_rank,
                   ws.score, ws.wins, ws.rugs, ws.appearances,
                   ws.outcomes_known, ws.flags
            FROM wallet_appearances wa
            LEFT JOIN wallet_scores ws ON wa.wallet_address = ws.wallet_address
            WHERE wa.token_address = ?
            ORDER BY wa.holder_rank
        """, (token_address,)).fetchall()

    if not appearances:
        return {
            "token_address":  token_address,
            "wallet_score":   None,
            "outcome_backed": False,
            "explanation":    "no wallet data",
        }

    rows = [dict(r) for r in appearances]

    scored = [r for r in rows if r["score"] is not None]
    farm_count   = sum(1 for r in scored if "FARM_CLUSTER" in (r.get("flags") or ""))
    repeat_count = sum(1 for r in scored if "REPEAT_WINNER" in (r.get("flags") or ""))
    known_wallets = len(scored)

    # Only wallets that have at least one resolved outcome contribute to the score
    outcome_wallets = [r for r in scored if (r.get("outcomes_known") or 0) > 0]
    outcome_backed  = len(outcome_wallets) > 0

    if outcome_backed:
        top_scores = sorted(
            [r["score"] for r in outcome_wallets if r["score"] is not None],
            reverse=True,
        )[:5]
        token_wallet_score = int(sum(top_scores) / len(top_scores)) if top_scores else None
    else:
        token_wallet_score = None   # suppress — not evidence-backed

    # Build explanation
    parts: list[str] = []
    if known_wallets:
        parts.append(f"{known_wallets} wallet{'s' if known_wallets != 1 else ''} tracked")
    if repeat_count:
        parts.append(f"{repeat_count} repeat winner{'s' if repeat_count != 1 else ''}")
    if farm_count:
        parts.append(f"{farm_count} farm cluster{'s' if farm_count != 1 else ''}")
    if not outcome_backed:
        parts.append("awaiting outcomes")

    return {
        "token_address":  token_address,
        "wallet_score":   token_wallet_score,
        "outcome_backed": outcome_backed,
        "known_wallets":  known_wallets,
        "repeat_winners": repeat_count,
        "farm_clusters":  farm_count,
        "explanation":    " | ".join(parts) if parts else "insufficient data",
    }


def registry_stats(path: Path) -> dict:
    """High-level registry statistics."""
    with _connect(path) as conn:
        total_wallets = conn.execute(
            "SELECT COUNT(DISTINCT wallet_address) FROM wallet_appearances"
        ).fetchone()[0]
        total_appearances = conn.execute(
            "SELECT COUNT(*) FROM wallet_appearances"
        ).fetchone()[0]
        scored = conn.execute(
            "SELECT COUNT(*) FROM wallet_scores WHERE outcomes_known > 0"
        ).fetchone()[0]
        repeat_winners = conn.execute(
            "SELECT COUNT(*) FROM wallet_scores WHERE flags LIKE '%REPEAT_WINNER%'"
        ).fetchone()[0]
        farm_clusters = conn.execute(
            "SELECT COUNT(*) FROM wallet_scores WHERE flags LIKE '%FARM_CLUSTER%'"
        ).fetchone()[0]

    return {
        "total_wallets":     total_wallets,
        "total_appearances": total_appearances,
        "wallets_with_outcomes": scored,
        "repeat_winners":    repeat_winners,
        "farm_clusters":     farm_clusters,
    }
