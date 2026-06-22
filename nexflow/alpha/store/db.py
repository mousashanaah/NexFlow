"""
Lightweight SQLite store for Alpha Board state.

Stores discovered pools and their risk results so the board
accumulates data between runs without re-fetching everything.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path = _DEFAULT_PATH) -> None:
    with _connect(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pools (
                pair_address    TEXT PRIMARY KEY,
                chain_id        TEXT NOT NULL,
                token_address   TEXT NOT NULL,
                token_name      TEXT,
                token_symbol    TEXT,
                dex_id          TEXT,
                price_usd       REAL,
                liquidity_usd   REAL,
                volume_24h      REAL,
                market_cap      REAL,
                pair_created_at INTEGER,
                age_hours       REAL,
                url             TEXT,
                source          TEXT,
                first_seen_at   TEXT NOT NULL,
                last_seen_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS risk_results (
                token_address       TEXT PRIMARY KEY,
                chain_id            TEXT NOT NULL,
                passed              INTEGER NOT NULL,
                risk_score          INTEGER,
                risk_label          TEXT,
                risk_flags          TEXT,
                is_honeypot         INTEGER,
                has_mint_function   INTEGER,
                owner_not_renounced INTEGER,
                has_blacklist       INTEGER,
                buy_tax             REAL,
                sell_tax            REAL,
                creator_percent     REAL,
                source_errors       TEXT,
                checked_at          TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pools_chain   ON pools(chain_id);
            CREATE INDEX IF NOT EXISTS idx_pools_created ON pools(pair_created_at);
            CREATE INDEX IF NOT EXISTS idx_risk_passed   ON risk_results(passed);
        """)


def upsert_pool(pool, path: Path = _DEFAULT_PATH) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        conn.execute("""
            INSERT INTO pools (
                pair_address, chain_id, token_address, token_name, token_symbol,
                dex_id, price_usd, liquidity_usd, volume_24h, market_cap,
                pair_created_at, age_hours, url, source, first_seen_at, last_seen_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(pair_address) DO UPDATE SET
                price_usd       = excluded.price_usd,
                liquidity_usd   = excluded.liquidity_usd,
                volume_24h      = excluded.volume_24h,
                market_cap      = excluded.market_cap,
                age_hours       = excluded.age_hours,
                last_seen_at    = excluded.last_seen_at
        """, (
            pool.pair_address, pool.chain_id, pool.token_address,
            pool.token_name, pool.token_symbol, pool.dex_id,
            pool.price_usd, pool.liquidity_usd, pool.volume_24h,
            pool.market_cap, pool.pair_created_at, pool.age_hours,
            pool.url, pool.source, now, now,
        ))


def upsert_risk(result, path: Path = _DEFAULT_PATH) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        conn.execute("""
            INSERT INTO risk_results (
                token_address, chain_id, passed, risk_score, risk_label,
                risk_flags, is_honeypot, has_mint_function, owner_not_renounced,
                has_blacklist, buy_tax, sell_tax, creator_percent,
                source_errors, checked_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(token_address) DO UPDATE SET
                passed              = excluded.passed,
                risk_score          = excluded.risk_score,
                risk_label          = excluded.risk_label,
                risk_flags          = excluded.risk_flags,
                is_honeypot         = excluded.is_honeypot,
                has_mint_function   = excluded.has_mint_function,
                owner_not_renounced = excluded.owner_not_renounced,
                has_blacklist       = excluded.has_blacklist,
                buy_tax             = excluded.buy_tax,
                sell_tax            = excluded.sell_tax,
                creator_percent     = excluded.creator_percent,
                source_errors       = excluded.source_errors,
                checked_at          = excluded.checked_at
        """, (
            result.token_address, result.chain_id, int(result.passed),
            result.risk_score, result.risk_label,
            json.dumps(result.risk_flags),
            int(result.is_honeypot) if result.is_honeypot is not None else None,
            int(result.has_mint_function) if result.has_mint_function is not None else None,
            int(result.owner_not_renounced) if result.owner_not_renounced is not None else None,
            int(result.has_blacklist) if result.has_blacklist is not None else None,
            result.buy_tax, result.sell_tax, result.creator_percent,
            json.dumps(result.source_errors), now,
        ))


def load_board(
    path:          Path  = _DEFAULT_PATH,
    passed_only:   bool  = False,
    max_age_hours: float = 48.0,
    limit:         int   = 100,
) -> list[dict]:
    """
    Load pools joined with risk results for display.
    Returns list of dicts sorted by liquidity desc.
    """
    with _connect(path) as conn:
        where_clauses = []
        params: list = []

        if max_age_hours:
            where_clauses.append("p.age_hours <= ?")
            params.append(max_age_hours)

        if passed_only:
            where_clauses.append("r.passed = 1")

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        rows = conn.execute(f"""
            SELECT
                p.pair_address, p.chain_id, p.token_address,
                p.token_name, p.token_symbol, p.dex_id,
                p.price_usd, p.liquidity_usd, p.volume_24h,
                p.market_cap, p.age_hours, p.url, p.source,
                r.passed, r.risk_score, r.risk_label, r.risk_flags,
                r.is_honeypot, r.has_mint_function, r.sell_tax,
                r.creator_percent, r.checked_at
            FROM pools p
            LEFT JOIN risk_results r ON p.token_address = r.token_address
            {where}
            ORDER BY p.liquidity_usd DESC NULLS LAST
            LIMIT ?
        """, params + [limit]).fetchall()

    return [dict(r) for r in rows]
