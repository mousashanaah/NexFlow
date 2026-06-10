#!/usr/bin/env python3
"""Burst-scalp mode test — can V8.63 profit from violent 1H moves?

Idea: when a coin makes an extreme 1H move (rare, strict threshold), enter a
short-lived trade to catch the follow-through (continuation) or the snap-back
(fade). Tight stop, fixed max hold, full taker fees both sides.

Variants tested per (threshold, direction, hold):
  - CONT: enter in direction of the burst, exit after H hours or on stop
  - FADE: enter against the burst, exit after H hours or on stop

Strict accounting: 0.06% taker each side (0.12% round trip).
Data: clean Bitget 1H candles, BTC + ETH (only coins with 1H history).
Sizing: fixed $1,000 notional per trade (results scale linearly).
Stop: 1.5% against entry, checked on hourly closes (optimistic for stop —
      real intra-hour wicks would hit it more often, so live would be WORSE).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
_FEE = 0.0006          # taker, per side
_NOTIONAL = 1_000.0
_STOP = 0.015          # 1.5% hard stop


def _load(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(_REPO_ROOT / "data" / "candles" / f"{sym}_1H.parquet")
    df = df.sort_values("open_time").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["year"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.year
    return df


def _run(df: pd.DataFrame, thresh: float, direction: str, hold: int) -> dict:
    """Simulate burst trades. direction: 'cont' or 'fade'."""
    closes = df["close"].values
    rets   = df["ret"].values
    years  = df["year"].values
    n = len(df)

    trades = 0
    wins = 0
    total_pnl = 0.0
    year_pnl: dict[int, float] = {}
    i = 1
    while i < n - hold - 1:
        r = rets[i]
        if np.isnan(r) or abs(r) < thresh:
            i += 1
            continue
        burst_dir = 1 if r > 0 else -1
        side = burst_dir if direction == "cont" else -burst_dir
        entry = closes[i]            # enter at close of the burst bar
        exit_px = None
        for j in range(i + 1, min(i + 1 + hold, n)):
            move = (closes[j] - entry) / entry * side
            if move <= -_STOP:
                exit_px = closes[j]
                break
        if exit_px is None:
            exit_px = closes[min(i + hold, n - 1)]
        raw = (exit_px - entry) / entry * side
        net = (raw - 2 * _FEE) * _NOTIONAL
        total_pnl += net
        trades += 1
        wins += net > 0
        yr = int(years[i])
        year_pnl[yr] = year_pnl.get(yr, 0.0) + net
        i = min(i + hold, n - 1) + 1   # no overlapping trades

    return {"trades": trades, "wins": wins, "pnl": total_pnl, "year_pnl": year_pnl,
            "win_rate": wins / trades if trades else 0.0,
            "avg": total_pnl / trades if trades else 0.0}


def main():
    print("=" * 110)
    print("  BURST-SCALP TEST — extreme 1H moves, continuation vs fade  "
          "($1,000/trade, 0.12% round-trip fees, 1.5% stop)")
    print("=" * 110)

    data = {s: _load(s) for s in ["BTCUSDT", "ETHUSDT"]}
    for s, df in data.items():
        t0 = pd.to_datetime(df['open_time'].iloc[0], unit='ms').date()
        t1 = pd.to_datetime(df['open_time'].iloc[-1], unit='ms').date()
        print(f"  {s}: {len(df):,} bars  {t0} → {t1}")

    grid_thresh = [0.02, 0.03, 0.04]
    grid_hold   = [2, 4, 8, 24]

    for direction in ["cont", "fade"]:
        label = "CONTINUATION (ride the burst)" if direction == "cont" else "FADE (bet on snap-back)"
        print(f"\n  ── {label} " + "─" * (88 - len(label)))
        print(f"  {'thresh':>7} {'hold':>5} | " +
              " | ".join(f"{s[:3]:>26}" for s in data) + " |")
        print(f"  {'':>7} {'':>5} | " +
              " | ".join(f"{'trades':>7} {'win%':>5} {'PnL':>11}" for _ in data) + " |")
        for th in grid_thresh:
            for h in grid_hold:
                cells = []
                for s, df in data.items():
                    r = _run(df, th, direction, h)
                    cells.append(f"{r['trades']:>7} {r['win_rate']*100:>4.0f}% ${r['pnl']:>+9,.0f}")
                print(f"  {th*100:>6.1f}% {h:>4}h | " + " | ".join(cells) + " |")

    # Year-by-year for the single best-looking cell of each direction
    print("\n  Year-by-year for selected cells (BTC + ETH combined):")
    for direction, th, h in [("cont", 0.03, 8), ("fade", 0.03, 8),
                              ("cont", 0.04, 24), ("fade", 0.04, 4)]:
        combined: dict[int, float] = {}
        tot = 0.0
        ntr = 0
        for s, df in data.items():
            r = _run(df, th, direction, h)
            tot += r["pnl"]; ntr += r["trades"]
            for yr, p in r["year_pnl"].items():
                combined[yr] = combined.get(yr, 0.0) + p
        yrs = " ".join(f"{yr}:{p:>+7,.0f}" for yr, p in sorted(combined.items()))
        print(f"    {direction.upper():<5} {th*100:.0f}%/{h}h  trades={ntr:>4}  "
              f"total=${tot:>+8,.0f}   {yrs}")

    print("\n" + "=" * 110)
    print("  NOTE: stops checked on hourly closes — intra-hour wicks would hit stops")
    print("  more often in live trading, so real results would be WORSE than shown.")
    print("=" * 110)


if __name__ == "__main__":
    main()
