"""
Narrative Store — persists category assignments and tracks win rates.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


_DDL = """
CREATE TABLE IF NOT EXISTS narrative_tags (
    token_address   TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    confidence      REAL,
    matched_signals TEXT,   -- JSON list
    secondary       TEXT,
    tagged_ts       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_narrative_cat ON narrative_tags(category);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_narrative_store(path: Path) -> None:
    with _connect(path) as conn:
        conn.executescript(_DDL)


def upsert_narrative(result, path: Path) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        conn.execute("""
            INSERT INTO narrative_tags
                (token_address, category, confidence, matched_signals, secondary, tagged_ts)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(token_address) DO UPDATE SET
                category        = excluded.category,
                confidence      = excluded.confidence,
                matched_signals = excluded.matched_signals,
                secondary       = excluded.secondary,
                tagged_ts       = excluded.tagged_ts
        """, (
            result.token_address,
            result.category,
            result.confidence,
            json.dumps(result.matched_signals),
            result.secondary,
            now,
        ))


def load_narrative(token_address: str, path: Path) -> dict | None:
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM narrative_tags WHERE token_address = ?",
            (token_address,)
        ).fetchone()
    return dict(row) if row else None


def narrative_win_rates(path: Path) -> list[dict]:
    """
    Return win rate statistics per narrative category.
    Joins narrative_tags with alpha_memory outcomes.
    Only counts tokens that have been classified.
    """
    with _connect(path) as conn:
        # Check alpha_memory exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_memory'"
        ).fetchone()
        if not exists:
            return []

        rows = conn.execute("""
            SELECT
                nt.category,
                COUNT(*)                                                    AS total,
                SUM(CASE WHEN am.classification IS NOT NULL THEN 1 ELSE 0 END) AS classified,
                SUM(CASE WHEN am.classification = 'Winner'  THEN 1 ELSE 0 END) AS winners,
                SUM(CASE WHEN am.classification = 'Failure' THEN 1 ELSE 0 END) AS failures,
                SUM(CASE WHEN am.classification = 'Rug'     THEN 1 ELSE 0 END) AS rugs,
                AVG(CASE WHEN am.return_7d IS NOT NULL THEN am.return_7d END) AS avg_ret_7d,
                AVG(CASE WHEN am.return_30d IS NOT NULL THEN am.return_30d END) AS avg_ret_30d
            FROM narrative_tags nt
            LEFT JOIN alpha_memory am ON nt.token_address = am.token_address
            GROUP BY nt.category
            ORDER BY winners DESC, total DESC
        """).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        classified = d["classified"] or 0
        winners    = d["winners"]    or 0
        d["win_rate"] = (winners / classified) if classified > 0 else None
        result.append(d)
    return result


def narrative_stats(path: Path) -> dict:
    """Summary counts by category."""
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT category, COUNT(*) as n FROM narrative_tags GROUP BY category ORDER BY n DESC"
        ).fetchall()
    return {r["category"]: r["n"] for r in rows}
