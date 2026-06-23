#!/usr/bin/env python3
"""
NexFlow Alpha — Weekly Outcome Report

Prints a performance summary for all tokens in Alpha Memory.
Breaks results down by signal source (profiles vs boosts) and
by chain.  As wallet intelligence matures, breaks down by wallet
score quartile.

Usage:
  python scripts/alpha_weekly_report.py
  python scripts/alpha_weekly_report.py [--weeks 4] [--csv]

  --weeks N  : Look back N weeks (default: all time)
  --csv      : Also write report to alpha_report_YYYYMMDD.csv

Config:
  NEXFLOW_ALPHA_DB : SQLite path (default: /var/nexflow/alpha.db)
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median, mean

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.alpha.store.memory import init_memory
from nexflow.alpha.wallets.registry import init_wallet_registry
from nexflow.alpha.narrative.store import init_narrative_store, narrative_win_rates

_DB_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _fmt_pct(val) -> str:
    if val is None:
        return "  —"
    return f"{val*100:5.1f}%"


def _fmt_ret(val) -> str:
    if val is None:
        return "    —"
    return f"{val:5.2f}x"


def _section(rows: list[dict]) -> dict:
    """Compute stats for a slice of rows."""
    total   = len(rows)
    classified = [r for r in rows if r.get("classification")]
    winners = [r for r in classified if r["classification"] == "Winner"]
    failures= [r for r in classified if r["classification"] == "Failure"]
    rugs    = [r for r in classified if r["classification"] == "Rug"]
    neutral = [r for r in classified if r["classification"] == "Neutral"]

    returns_7d  = [r["return_7d"]  for r in rows if r.get("return_7d")  is not None]
    returns_30d = [r["return_30d"] for r in rows if r.get("return_30d") is not None]

    best_7d  = max(returns_7d,  default=None)
    worst_7d = min(returns_7d,  default=None)

    # Find best/worst symbols
    best_row  = next((r for r in rows if r.get("return_7d") == best_7d),  None)
    worst_row = next((r for r in rows if r.get("return_7d") == worst_7d), None)

    return {
        "total":          total,
        "classified":     len(classified),
        "winners":        len(winners),
        "failures":       len(failures),
        "rugs":           len(rugs),
        "neutral":        len(neutral),
        "win_rate":       len(winners) / len(classified) if classified else None,
        "avg_ret_7d":     mean(returns_7d)    if returns_7d  else None,
        "median_ret_7d":  median(returns_7d)  if returns_7d  else None,
        "avg_ret_30d":    mean(returns_30d)   if returns_30d else None,
        "best_7d":        best_7d,
        "worst_7d":       worst_7d,
        "best_sym":       (best_row  or {}).get("token_symbol", "?"),
        "worst_sym":      (worst_row or {}).get("token_symbol", "?"),
    }


def _print_section(title: str, s: dict) -> None:
    print(f"\n  {title}")
    print(f"    Discoveries : {s['total']}  (classified: {s['classified']})")
    if not s["classified"]:
        print(f"    No classified outcomes yet (need 7d of data)")
        return
    print(f"    Winners     : {s['winners']}  ({_fmt_pct(s['win_rate'])} win rate)")
    print(f"    Failures    : {s['failures']}")
    print(f"    Rugs        : {s['rugs']}")
    print(f"    Neutral     : {s['neutral']}")
    if s["avg_ret_7d"] is not None:
        print(f"    Avg ret 7d  : {_fmt_ret(s['avg_ret_7d'])}")
        print(f"    Median ret  : {_fmt_ret(s['median_ret_7d'])}")
    if s["best_7d"] is not None:
        print(f"    Best        : {s['best_sym']:<12}  {_fmt_ret(s['best_7d'])}")
    if s["worst_7d"] is not None:
        print(f"    Worst       : {s['worst_sym']:<12}  {_fmt_ret(s['worst_7d'])}")


def run_report(weeks_back: int | None, write_csv: bool) -> None:
    with _connect(_DB_PATH) as conn:
        where = ""
        params: list = []
        if weeks_back:
            cutoff = (datetime.now(timezone.utc) - timedelta(weeks=weeks_back)).isoformat()
            where  = "WHERE discovery_ts >= ?"
            params = [cutoff]

        rows = conn.execute(
            f"SELECT * FROM alpha_memory {where} ORDER BY discovery_ts DESC",
            params,
        ).fetchall()

    rows = [dict(r) for r in rows]

    if not rows:
        print("No data in Alpha Memory yet.")
        return

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    period  = f"last {weeks_back} week{'s' if weeks_back != 1 else ''}" if weeks_back else "all time"
    w = 65

    print(f"\n{'═'*w}")
    print(f"{'NEXFLOW ALPHA — WEEKLY OUTCOME REPORT':^{w}}")
    print(f"{now_str:^{w}}")
    print(f"{'Period: ' + period:^{w}}")
    print(f"{'═'*w}")

    # ── Overall ───────────────────────────────────────────────────────────────
    overall = _section(rows)
    print(f"\n{'─'*w}")
    print("  OVERALL")
    print(f"{'─'*w}")
    _print_section("All discoveries", overall)

    # ── By Signal Source ──────────────────────────────────────────────────────
    sources = sorted({r.get("source_signal") or "unknown" for r in rows})
    if len(sources) > 1:
        print(f"\n{'─'*w}")
        print("  BY SIGNAL SOURCE")
        print(f"{'─'*w}")
        for src in sources:
            subset = [r for r in rows if (r.get("source_signal") or "unknown") == src]
            _print_section(f"Source: {src}  ({len(subset)} pools)", _section(subset))

    # ── By Chain ──────────────────────────────────────────────────────────────
    chains = sorted({r.get("chain_id") or "unknown" for r in rows})
    if len(chains) > 1:
        print(f"\n{'─'*w}")
        print("  BY CHAIN")
        print(f"{'─'*w}")
        for chain in chains:
            subset = [r for r in rows if (r.get("chain_id") or "unknown") == chain]
            _print_section(f"Chain: {chain}  ({len(subset)} pools)", _section(subset))

    # ── Weekly Discovery Growth ───────────────────────────────────────────────
    print(f"\n{'─'*w}")
    print("  DISCOVERY GROWTH (by week)")
    print(f"{'─'*w}")
    with _connect(_DB_PATH) as conn:
        weekly = conn.execute("""
            SELECT
                strftime('%Y-W%W', discovery_ts) as week,
                COUNT(*) as total,
                SUM(CASE WHEN classification='Winner'  THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN classification='Rug'     THEN 1 ELSE 0 END) as rugs,
                SUM(CASE WHEN classification='Failure' THEN 1 ELSE 0 END) as failures
            FROM alpha_memory
            GROUP BY week
            ORDER BY week DESC
            LIMIT 12
        """).fetchall()

    print(f"  {'WEEK':<12}  {'DISCOVERED':>10}  {'WINNERS':>8}  {'RUGS':>6}  {'FAILURES':>9}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*9}")
    for w_row in weekly:
        print(
            f"  {w_row['week']:<12}  {w_row['total']:>10}  "
            f"{w_row['wins'] or 0:>8}  {w_row['rugs'] or 0:>6}  "
            f"{w_row['failures'] or 0:>9}"
        )

    # ── Narrative Win Rates ───────────────────────────────────────────────────
    try:
        nar_rows = narrative_win_rates(_DB_PATH)
        if nar_rows:
            print(f"\n{'─'*w}")
            print("  BY NARRATIVE")
            print(f"{'─'*w}")
            print(f"  {'CATEGORY':<14}  {'TOTAL':>6}  {'CLASSED':>8}  {'WIN%':>6}  {'AVG 7D':>8}  {'WINS':>5}  {'RUGS':>5}")
            print(f"  {'─'*14}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*5}")
            for nr in nar_rows:
                win_pct = f"{nr['win_rate']*100:5.1f}%" if nr.get("win_rate") is not None else "    —"
                avg_7d  = _fmt_ret(nr.get("avg_ret_7d"))
                print(
                    f"  {nr['category']:<14}  {nr['total']:>6}  {nr['classified'] or 0:>8}  "
                    f"{win_pct:>6}  {avg_7d:>8}  {nr['winners'] or 0:>5}  {nr['rugs'] or 0:>5}"
                )
    except Exception:
        pass

    # ── Wallet Registry ───────────────────────────────────────────────────────
    try:
        with _connect(_DB_PATH) as conn:
            wstats = conn.execute("""
                SELECT
                    COUNT(DISTINCT wallet_address) as total_wallets,
                    COUNT(*) as total_appearances,
                    SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as with_outcomes
                FROM wallet_appearances
            """).fetchone()
            top_wallets = conn.execute("""
                SELECT ws.wallet_address, ws.score, ws.wins, ws.rugs,
                       ws.appearances, ws.win_rate, ws.flags
                FROM wallet_scores ws
                WHERE ws.outcomes_known > 0
                ORDER BY ws.score DESC
                LIMIT 5
            """).fetchall()

        if wstats and wstats["total_wallets"]:
            print(f"\n{'─'*w}")
            print("  WALLET REGISTRY")
            print(f"{'─'*w}")
            print(
                f"  Total wallets tracked: {wstats['total_wallets']}"
                f"  |  appearances: {wstats['total_appearances']}"
                f"  |  with outcomes: {wstats['with_outcomes']}"
            )
            if top_wallets:
                print(f"\n  Top wallets by score:")
                print(f"  {'ADDRESS':<20}  {'SCORE':>6}  {'WINS':>5}  {'RUGS':>5}  {'SEEN':>5}  FLAGS")
                for tw in top_wallets:
                    flags = ""
                    try:
                        import json
                        flags = ", ".join(json.loads(tw["flags"] or "[]"))
                    except Exception:
                        pass
                    print(
                        f"  {tw['wallet_address'][:18]:<20}  {tw['score']:>6}  "
                        f"{tw['wins'] or 0:>5}  {tw['rugs'] or 0:>5}  "
                        f"{tw['appearances']:>5}  {flags}"
                    )
    except Exception:
        pass

    print(f"\n{'═'*w}\n")

    # ── CSV export ────────────────────────────────────────────────────────────
    if write_csv:
        fname = f"alpha_report_{datetime.now().strftime('%Y%m%d')}.csv"
        fieldnames = [
            "token_symbol", "chain_id", "discovery_ts", "source_signal",
            "initial_price", "initial_liquidity", "risk_label",
            "return_1d", "return_7d", "return_30d", "classification",
        ]
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV written: {fname}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="NexFlow Alpha Weekly Report")
    parser.add_argument("--weeks", type=int, default=None,
                        help="Look back N weeks (default: all time)")
    parser.add_argument("--csv", action="store_true",
                        help="Write CSV export")
    args = parser.parse_args()

    init_memory(_DB_PATH)
    init_wallet_registry(_DB_PATH)
    init_narrative_store(_DB_PATH)
    run_report(weeks_back=args.weeks, write_csv=args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
