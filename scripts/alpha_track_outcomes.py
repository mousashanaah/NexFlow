#!/usr/bin/env python3
"""
NexFlow Alpha — Outcome Tracker

Fetches current price/liquidity for all tracked pools in Alpha Memory,
updates outcome columns, and classifies tokens as Winner/Neutral/Failure/Rug.

Usage:
  python scripts/alpha_track_outcomes.py [--verbose] [--dry-run]
  python scripts/alpha_track_outcomes.py [--min-age 24] [--max-age 90]

  --min-age  : Only update tokens at least N hours old (default: 24)
  --max-age  : Skip tokens older than N days (default: 90)
  --dry-run  : Print what would be updated without writing
  --verbose  : Show detailed fetch progress

Classification rules:
  Winner  — return_7d >= 2.0x  (doubled within a week)
  Failure — return_7d < 0.5x   (lost more than half)
  Rug     — liquidity_usd < 100 AND initial_liquidity >= 1000
           OR pool no longer exists on DexScreener
  Neutral — everything else

Config:
  NEXFLOW_ALPHA_DB : SQLite path (default: /var/nexflow/alpha.db)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.alpha.store.memory import (
    init_memory, load_untracked, update_outcome, summary_stats,
)

_DB_PATH = Path(os.environ.get("NEXFLOW_ALPHA_DB", "/var/nexflow/alpha.db"))

# DexScreener
import json as _json
import urllib.request

_BASE    = "https://api.dexscreener.com"
_TIMEOUT = 15
_HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}


def _get(path: str) -> dict | list | None:
    url = f"{_BASE}{path}" if path.startswith("/") else path
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return _json.loads(resp.read().decode())
    except Exception:
        return None


def _fetch_pair(chain_id: str, pair_address: str) -> dict | None:
    """Fetch current data for a single pair."""
    data = _get(f"/latest/dex/pairs/{chain_id}/{pair_address}")
    if not data:
        return None
    pairs = data.get("pairs") or (data.get("pair") and [data["pair"]]) or []
    return pairs[0] if pairs else None


def _classify(
    initial_price:     float | None,
    initial_liquidity: float | None,
    current_price:     float | None,
    current_liquidity: float | None,
    pool_exists:       bool,
    hours_since_discovery: float,
) -> str | None:
    """Return classification string or None if not enough data yet."""
    if not pool_exists:
        return "Rug"

    # Rug: liquidity collapsed
    if (initial_liquidity and initial_liquidity >= 1_000
            and current_liquidity is not None
            and current_liquidity < 100):
        return "Rug"

    # Need at least 7d of data before classifying
    if hours_since_discovery < 168:
        return None

    if not initial_price or not current_price:
        return None

    ret = current_price / initial_price
    if ret >= 2.0:
        return "Winner"
    if ret < 0.5:
        return "Failure"
    return "Neutral"


def _determine_period(hours: float) -> list[str]:
    """Return which outcome periods to update based on hours since discovery."""
    periods = ["1d"]
    if hours >= 168:
        periods.append("7d")
    if hours >= 720:
        periods.append("30d")
    if hours >= 2160:
        periods.append("90d")
    return periods


def _fmt_usd(val) -> str:
    if val is None:
        return "—"
    if val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"${val/1_000:.0f}K"
    return f"${val:.0f}"


def _fmt_ret(ret: float | None) -> str:
    if ret is None:
        return "—"
    return f"{ret:.2f}x"


def run_tracker(
    min_age_hours: float,
    max_age_days:  float,
    dry_run:       bool,
    verbose:       bool,
) -> None:
    rows = load_untracked(_DB_PATH, min_age_hours=min_age_hours, max_age_days=max_age_days)

    if not rows:
        print(f"No pools to update (min_age={min_age_hours}h, max_age={max_age_days}d).")
        return

    print(f"Tracking {len(rows)} pool(s)...")
    now_ts = datetime.now(timezone.utc)

    updated = 0
    rugs    = 0
    winners = 0
    errors  = 0

    for i, row in enumerate(rows, 1):
        pair_addr   = row["pair_address"]
        chain_id    = row["chain_id"]
        symbol      = (row.get("token_symbol") or pair_addr[:8])[:12]
        disc_ts_str = row.get("discovery_ts") or ""

        try:
            disc_ts = datetime.fromisoformat(disc_ts_str.replace("Z", "+00:00"))
        except Exception:
            disc_ts = now_ts
        hours_since = (now_ts - disc_ts).total_seconds() / 3600

        if verbose:
            print(f"  [{i}/{len(rows)}] {symbol} ({chain_id})  "
                  f"age={hours_since:.0f}h  pair={pair_addr[:12]}...")

        pair = _fetch_pair(chain_id, pair_address=pair_addr)
        time.sleep(0.3)

        if pair is None:
            pool_exists   = False
            current_price = None
            current_liq   = None
            current_vol   = None
        else:
            pool_exists   = True
            price_str     = pair.get("priceUsd")
            current_price = float(price_str) if price_str else None
            liq_data      = pair.get("liquidity") or {}
            vol_data      = pair.get("volume") or {}
            current_liq   = liq_data.get("usd")
            current_vol   = vol_data.get("h24")

        initial_price = row.get("initial_price")
        initial_liq   = row.get("initial_liquidity")
        ret           = (current_price / initial_price) if (current_price and initial_price) else None

        classification = _classify(
            initial_price     = initial_price,
            initial_liquidity = initial_liq,
            current_price     = current_price,
            current_liquidity = current_liq,
            pool_exists       = pool_exists,
            hours_since_discovery = hours_since,
        )

        periods = _determine_period(hours_since)

        if verbose:
            print(f"         price: {_fmt_usd(current_price)} (was {_fmt_usd(initial_price)})  "
                  f"ret={_fmt_ret(ret)}  liq={_fmt_usd(current_liq)}  "
                  f"class={classification or '—'}  periods={periods}")

        if not dry_run:
            for period in periods:
                update_outcome(
                    pair_address   = pair_addr,
                    path           = _DB_PATH,
                    price          = current_price,
                    liquidity      = current_liq,
                    volume         = current_vol,
                    period         = period,
                    classification = classification,
                )

        updated += 1
        if classification == "Rug":
            rugs += 1
        elif classification == "Winner":
            winners += 1

        if not pool_exists:
            print(f"  RUG/DEAD  {symbol} ({chain_id}) — pair no longer on DexScreener")
        elif ret is not None and ret >= 2.0:
            print(f"  WINNER    {symbol}  {_fmt_ret(ret)}  liq={_fmt_usd(current_liq)}")
        elif ret is not None and ret < 0.5:
            print(f"  FAILURE   {symbol}  {_fmt_ret(ret)}  liq={_fmt_usd(current_liq)}")

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Updated {updated} pools | {winners} winners | {rugs} rugs | {errors} errors")

    if not dry_run:
        stats = summary_stats(_DB_PATH)
        total = stats["total_discoveries"]
        by_c  = stats["by_classification"]
        print(f"Alpha Memory: {total} total  |  ", end="")
        parts = [f"{k}={v}" for k, v in by_c.items() if k != "unclassified"]
        if parts:
            print(", ".join(parts), end="")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="NexFlow Alpha Outcome Tracker")
    parser.add_argument("--min-age",  type=float, default=24.0,
                        help="Min hours since discovery (default: 24)")
    parser.add_argument("--max-age",  type=float, default=90.0,
                        help="Max days since discovery (default: 90)")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    init_memory(_DB_PATH)
    run_tracker(
        min_age_hours = args.min_age,
        max_age_days  = args.max_age,
        dry_run       = args.dry_run,
        verbose       = args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
