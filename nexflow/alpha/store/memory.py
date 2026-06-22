"""
Alpha Memory — permanent discovery and outcome tracking.

Every token discovered by NexFlow Alpha is recorded here.
Outcomes (price, liquidity, survival) are updated over time.
This dataset answers: what did winners look like before they won?
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Schema ────────────────────────────────────────────────────────────────────

_MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS alpha_memory (
    discovery_id      TEXT PRIMARY KEY,   -- pair_address + discovery_ts
    pair_address      TEXT NOT NULL,
    token_address     TEXT NOT NULL,
    token_symbol      TEXT,
    token_name        TEXT,
    chain_id          TEXT NOT NULL,
    dex_id            TEXT,

    -- Discovery context
    discovery_ts      TEXT NOT NULL,      -- ISO UTC
    source_signal     TEXT,               -- "profiles" | "boosts"

    -- Snapshot at discovery
    initial_price     REAL,
    initial_liquidity REAL,
    initial_volume    REAL,
    initial_mcap      REAL,
    age_hours_at_discovery REAL,

    -- Risk at discovery
    risk_score        INTEGER,
    risk_label        TEXT,
    risk_flags        TEXT,               -- JSON list

    -- Scores (filled as system matures)
    wallet_score      REAL,               -- Week 2
    narrative_score   REAL,               -- Week 3

    -- Outcomes (filled by tracker)
    price_1d          REAL,
    price_7d          REAL,
    price_30d         REAL,
    price_90d          REAL,
    liq_1d            REAL,
    liq_7d            REAL,
    liq_30d           REAL,
    volume_1d         REAL,
    volume_7d         REAL,

    -- Return multiples (price_Nd / initial_price)
    return_1d         REAL,
    return_7d         REAL,
    return_30d        REAL,
    return_90d        REAL,

    -- Classification (assigned after observation period)
    classification    TEXT,               -- Winner | Neutral | Failure | Rug

    -- Tracking state
    last_checked_ts   TEXT,
    check_count       INTEGER DEFAULT 0,
    url               TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_chain     ON alpha_memory(chain_id);
CREATE INDEX IF NOT EXISTS idx_memory_ts        ON alpha_memory(discovery_ts);
CREATE INDEX IF NOT EXISTS idx_memory_class     ON alpha_memory(classification);
CREATE INDEX IF NOT EXISTS idx_memory_pair      ON alpha_memory(pair_address);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_memory(path: Path) -> None:
    with _connect(path) as conn:
        conn.executescript(_MEMORY_DDL)


# ── Write ─────────────────────────────────────────────────────────────────────

def record_discovery(pool, risk_result, path: Path) -> str:
    """
    Record a newly discovered pool in Alpha Memory.
    Returns the discovery_id.
    """
    ts           = datetime.now(timezone.utc).isoformat()
    discovery_id = f"{pool.pair_address}:{int(time.time())}"

    flags = json.dumps(risk_result.risk_flags) if risk_result else "[]"
    r_score = risk_result.risk_score if risk_result else None
    r_label = risk_result.risk_label if risk_result else "UNVERIFIED"

    with _connect(path) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO alpha_memory (
                discovery_id, pair_address, token_address, token_symbol,
                token_name, chain_id, dex_id,
                discovery_ts, source_signal,
                initial_price, initial_liquidity, initial_volume, initial_mcap,
                age_hours_at_discovery,
                risk_score, risk_label, risk_flags,
                url, last_checked_ts, check_count
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
        """, (
            discovery_id,
            pool.pair_address,
            pool.token_address,
            pool.token_symbol,
            pool.token_name,
            pool.chain_id,
            pool.dex_id,
            ts,
            pool.source,
            pool.price_usd,
            pool.liquidity_usd,
            pool.volume_24h,
            pool.market_cap,
            pool.age_hours,
            r_score,
            r_label,
            flags,
            pool.url,
            ts,
        ))
    return discovery_id


def update_outcome(
    pair_address:  str,
    path:          Path,
    price:         Optional[float] = None,
    liquidity:     Optional[float] = None,
    volume:        Optional[float] = None,
    period:        str             = "1d",   # "1d" | "7d" | "30d" | "90d"
    classification: Optional[str] = None,
) -> None:
    """Update price/liquidity outcome for a tracked token."""
    col_map = {
        "1d":  ("price_1d",  "liq_1d",  "volume_1d",  "return_1d"),
        "7d":  ("price_7d",  "liq_7d",  "volume_7d",  "return_7d"),
        "30d": ("price_30d", "liq_30d", None,          "return_30d"),
        "90d": ("price_90d", None,       None,          "return_90d"),
    }
    cols = col_map.get(period)
    if not cols:
        return

    price_col, liq_col, vol_col, ret_col = cols
    now = datetime.now(timezone.utc).isoformat()

    with _connect(path) as conn:
        row = conn.execute(
            "SELECT initial_price FROM alpha_memory WHERE pair_address=? "
            "ORDER BY discovery_ts DESC LIMIT 1",
            (pair_address,)
        ).fetchone()

        if not row:
            return

        initial_price = row["initial_price"]
        ret = (price / initial_price) if (price and initial_price) else None

        updates: dict = {
            price_col:     price,
            liq_col:       liquidity if liq_col else None,
            ret_col:       ret,
            "last_checked_ts": now,
        }
        if vol_col:
            updates[vol_col] = volume
        if classification:
            updates["classification"] = classification

        # Remove None-keyed entries
        updates = {k: v for k, v in updates.items() if k and v is not None}

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE alpha_memory SET {set_clause}, check_count = check_count + 1 "
            f"WHERE pair_address = ?",
            list(updates.values()) + [pair_address],
        )


# ── Read ──────────────────────────────────────────────────────────────────────

def load_untracked(
    path:           Path,
    min_age_hours:  float = 24.0,
    max_age_days:   float = 90.0,
) -> list[dict]:
    """Return discoveries that need outcome updates."""
    with _connect(path) as conn:
        rows = conn.execute("""
            SELECT * FROM alpha_memory
            WHERE
                discovery_ts <= datetime('now', ? || ' hours')
            AND discovery_ts >= datetime('now', ? || ' days')
            ORDER BY discovery_ts DESC
        """, (f"-{min_age_hours}", f"-{max_age_days}")).fetchall()
    return [dict(r) for r in rows]


def load_all(path: Path, limit: int = 500) -> list[dict]:
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM alpha_memory ORDER BY discovery_ts DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def summary_stats(path: Path) -> dict:
    """Return high-level stats about the memory dataset."""
    with _connect(path) as conn:
        total    = conn.execute("SELECT COUNT(*) FROM alpha_memory").fetchone()[0]
        by_class = conn.execute(
            "SELECT classification, COUNT(*) as n FROM alpha_memory "
            "GROUP BY classification"
        ).fetchall()
        avg_ret_1d = conn.execute(
            "SELECT AVG(return_1d) FROM alpha_memory WHERE return_1d IS NOT NULL"
        ).fetchone()[0]
        avg_ret_7d = conn.execute(
            "SELECT AVG(return_7d) FROM alpha_memory WHERE return_7d IS NOT NULL"
        ).fetchone()[0]

    return {
        "total_discoveries": total,
        "by_classification": {r[0] or "unclassified": r[1] for r in by_class},
        "avg_return_1d":     avg_ret_1d,
        "avg_return_7d":     avg_ret_7d,
    }
