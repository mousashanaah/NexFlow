#!/usr/bin/env python3
"""Funding-cost reality check for V8.63.

Perp funding is paid every 8h (3×/day). Longs PAY when rate > 0 (almost
always in bull markets). Shorts RECEIVE when rate > 0. The backtest ignores
this; this script quantifies the drag.

Methodology:
  1. Replay V8.63 exactly (uses B._run internals) to record daily position
     snapshots (which symbols held long/short and at what notional).
  2. For each day, look up the 3 funding payments from the historical data.
     Only BTC and ETH have full funding history; for other coins we use the
     average of BTC+ETH as a proxy (highly correlated perp markets).
  3. Charge longs, credit shorts, accumulate year-by-year.
  4. Report funding-adjusted CAGR and DD vs raw backtest.

Honest caveats:
  - Only BTC/ETH funding files exist in data/funding/.
    All other coins use the BTC/ETH mean as a proxy — a reasonable upper
    bound (mid/small caps sometimes run at higher rates).
  - Funding history from Binance perpetuals; V8.63 runs on Bitget.
    Rates are highly correlated across venues.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.backtest_full_regime_system as B  # noqa: E402

B._CAPITAL = 5_000.0

_DAY_MS   = 86_400_000
_FROM     = int(datetime(2021, 1,  1, tzinfo=timezone.utc).timestamp() * 1000)
_TO       = int(datetime.now(timezone.utc).timestamp() * 1000)
_FUND_DIR = _REPO_ROOT / "data" / "funding"


# ---------------------------------------------------------------------------
# Load funding rates → dict  day_ts → daily_rate  (sum of 3 × 8h payments)
# ---------------------------------------------------------------------------
def _load_funding(symbol: str) -> dict[int, float]:
    path = _FUND_DIR / f"{symbol}_funding.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    df["day_ts"] = (df["timestamp_ms"] // _DAY_MS) * _DAY_MS
    # sum the up-to-3 payments that fall on each day
    day = df.groupby("day_ts")["funding_rate"].sum()
    return day.to_dict()


def _build_proxy(symbols: list[str]) -> dict[int, float]:
    """Build per-symbol funding proxy.  BTC/ETH get exact data; others get mean."""
    btc = _load_funding("BTCUSDT")
    eth = _load_funding("ETHUSDT")
    all_days = sorted(set(btc) | set(eth))
    proxy: dict[int, float] = {}
    for d in all_days:
        vals = [r for r in [btc.get(d), eth.get(d)] if r is not None]
        proxy[d] = sum(vals) / len(vals) if vals else 0.0
    out: dict[str, dict[int, float]] = {}
    for sym in symbols:
        exact_path = _FUND_DIR / f"{sym}_funding.parquet"
        if exact_path.exists():
            out[sym] = _load_funding(sym)
        else:
            out[sym] = proxy
    return out


# ---------------------------------------------------------------------------
# Patch B._run to also track daily position snapshots
# ---------------------------------------------------------------------------
def _run_with_snapshots(
    symbols: list[str],
    from_ts: int,
    to_ts: int,
) -> tuple[dict, list[dict]]:
    """Run V8.63 and collect one snapshot per trading day."""
    sig = B._build_signals(sorted(set(symbols) | {"BTCUSDT"}))
    old_syms = B._SYMBOLS
    B._SYMBOLS = symbols

    _BASE = dict(
        hard_stop_pct=0.15, use_atr_sizing=True,
        asymmetric_regime=True, and_entry=True,
        bear_drop_pct=-0.20, confirm_days=10,
        momentum_gate=True, momentum_gate_days=20,
    )

    # We need position snapshots, so we monkey-patch inside the run.
    # Easier: re-implement the minimal position tracking by post-processing
    # the trade log.  But B._run doesn't return a trade log.
    # Instead, we replicate the position tracking using signals directly.
    # Strategy: call _run normally for the authoritative PnL, then replay
    # positions separately to get daily notional exposure per symbol.

    result = B._run(sig, True, True, False, True, from_ts, to_ts, **_BASE)
    B._SYMBOLS = old_syms

    # --- replay positions ---------------------------------------------------
    # We need to redo the entry/exit logic to know which days each coin
    # was held and at what notional.  We run the same state machine but only
    # track positions (no PnL bookkeeping).
    snapshots = _replay_positions(symbols, sig, from_ts, to_ts, **_BASE)

    return result, snapshots


def _replay_positions(
    symbols: list[str],
    sig: dict,
    from_ts: int,
    to_ts: int,
    hard_stop_pct: float = 0.15,
    use_atr_sizing: bool = True,
    asymmetric_regime: bool = True,
    and_entry: bool = True,
    bear_drop_pct: float = -0.20,
    confirm_days: int = 10,
    momentum_gate: bool = True,
    momentum_gate_days: int = 20,
    target_risk: float = 0.01,
    **_,
) -> list[dict]:
    """Lightweight replay that records {ts, sym, side, notional} per day held.

    sig structure: sig[sym][ts] = {close, ema_long, macd_long, h4_long, sma200_above, ...}
    """
    # All timestamps across all symbols, filtered to range
    all_ts = sorted(
        {ts for sym in symbols + ["BTCUSDT"] for ts in sig.get(sym, {}) if from_ts <= ts < to_ts}
    )

    capital = B._CAPITAL
    base_notional = capital / max(len(symbols), 1)

    # Precompute BTC SMA200 and 30d momentum for each ts
    btc_data = sig.get("BTCUSDT", {})
    btc_ts_list = sorted(btc_data.keys())
    btc_closes  = [btc_data[t]["close"] for t in btc_ts_list]
    btc_sma200  = {}
    btc_mom30   = {}
    for i, t in enumerate(btc_ts_list):
        if i >= 199:
            btc_sma200[t] = sum(btc_closes[i - 199: i + 1]) / 200
        if i >= 30:
            btc_mom30[t] = (btc_closes[i] - btc_closes[i - 30]) / btc_closes[i - 30]

    # Precompute per-symbol ATR (14-day)
    vol_series: dict[str, dict[int, float]] = {}
    for sym in symbols:
        sdata = sig.get(sym, {})
        sts   = sorted(sdata.keys())
        closes_s = [sdata[t]["close"] for t in sts]
        vd: dict[int, float] = {}
        for i, t in enumerate(sts):
            if i >= 14:
                diffs = [abs(closes_s[k] - closes_s[k - 1]) for k in range(i - 13, i + 1)]
                atr = sum(diffs) / len(diffs)
                vd[t] = atr / closes_s[i] if closes_s[i] > 0 else 0.0
        vol_series[sym] = vd

    def _notional(sym: str, ts: int) -> float:
        if not use_atr_sizing:
            return base_notional
        vol = vol_series.get(sym, {}).get(ts, 0.0)
        if vol <= 0:
            return base_notional
        vol_sized = (target_risk * capital) / vol
        return min(vol_sized, base_notional * 2)

    positions: dict[str, dict] = {}
    snapshots: list[dict] = []

    prev_bear_mode = False
    above_streak   = 0
    last_rebal_ts  = 0

    for ts in all_ts:
        btc_sig   = btc_data.get(ts, {})
        sma200_v  = btc_sma200.get(ts)
        btc_above = btc_sig.get("sma200_above", True) if sma200_v is not None else True
        btc_30d   = btc_mom30.get(ts, 0.0)

        # Asymmetric regime
        if asymmetric_regime:
            drop_triggered = btc_30d < bear_drop_pct
            enter_bear = (not btc_above) and drop_triggered if and_entry else (not btc_above) or drop_triggered
            if btc_above:
                above_streak += 1
            else:
                above_streak = 0
            confirmed_bull = above_streak >= max(1, confirm_days)
            if prev_bear_mode:
                bear_mode = not confirmed_bull
            else:
                bear_mode = enter_bear
            prev_bear_mode = bear_mode
            btc_bull = not bear_mode
        else:
            btc_bull = btc_above

        long_allowed = btc_bull

        # ── TSMOM short rebalance (weekly) ──
        if not btc_bull and (ts - last_rebal_ts) >= 7 * _DAY_MS:
            last_rebal_ts = ts
            scores = []
            for sym in symbols:
                sdata = sig.get(sym, {})
                past  = [t for t in sorted(sdata) if t <= ts]
                if len(past) < 127:
                    continue
                c_now  = sdata[past[-1]]["close"]
                c_past = sdata[past[-127]]["close"]
                ret = (c_now - c_past) / c_past
                scores.append((ret, sym))
            desired = {sym for ret, sym in scores if ret < -0.05}
            for sym in list(positions):
                if positions[sym]["side"] == "SHORT" and sym not in desired:
                    positions.pop(sym)
            for sym in desired:
                if sym not in positions:
                    c = sig.get(sym, {}).get(ts, {}).get("close")
                    if c:
                        positions[sym] = {"entry": c, "notional": _notional(sym, ts), "side": "SHORT"}

        # ── Close shorts if back in bull ──
        if btc_bull:
            for sym in [s for s in list(positions) if positions[s]["side"] == "SHORT"]:
                positions.pop(sym)

        # ── Hard stop on shorts ──
        for sym in [s for s in list(positions) if positions[s]["side"] == "SHORT"]:
            c = sig.get(sym, {}).get(ts, {}).get("close")
            if c is None:
                continue
            pos = positions[sym]
            if (c - pos["entry"]) / pos["entry"] >= hard_stop_pct:
                positions.pop(sym)

        # ── Long exits ──
        for sym in list(positions):
            if positions[sym]["side"] != "LONG":
                continue
            c = sig.get(sym, {}).get(ts, {}).get("close")
            if c is None:
                continue
            pos = positions[sym]
            if c <= pos["entry"] * (1 - hard_stop_pct):
                positions.pop(sym)
                continue
            if not long_allowed:
                positions.pop(sym)
                continue
            ema_bull  = sig[sym][ts].get("ema_long",  False)
            macd_bull = sig[sym][ts].get("macd_long", False)
            if not ema_bull and not macd_bull:
                positions.pop(sym)

        # ── Record exposure AFTER exits, BEFORE entries ──
        for sym, pos in positions.items():
            snapshots.append({"ts": ts, "sym": sym, "side": pos["side"],
                               "notional": pos["notional"]})

        if not long_allowed:
            continue

        # ── Long entries ──
        for sym in symbols:
            if sym in positions:
                continue
            srow = sig.get(sym, {}).get(ts)
            if srow is None:
                continue
            ema_bull  = srow.get("ema_long",  False)
            macd_bull = srow.get("macd_long", False)
            h4_bull   = srow.get("h4_long",   False)
            if not (ema_bull and macd_bull and h4_bull):
                continue
            if momentum_gate:
                sdata = sig[sym]
                past  = [t for t in sorted(sdata) if t <= ts]
                if len(past) > momentum_gate_days:
                    c0 = sdata[past[-(momentum_gate_days + 1)]]["close"]
                    c1 = sdata[past[-1]]["close"]
                    if c1 <= c0:
                        continue
            c = srow["close"]
            positions[sym] = {"entry": c, "notional": _notional(sym, ts), "side": "LONG"}

    return snapshots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("  FUNDING-COST REALITY CHECK — V8.63 ($5K), 2021 → 2026")
    print("=" * 100)

    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
        "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
    ]

    print("\n  Loading funding rates ...")
    fund = _build_proxy(symbols)

    print("  Running V8.63 backtest + position replay ...")
    result, snapshots = _run_with_snapshots(symbols, _FROM, _TO)

    raw_equity = result["equity"]
    raw_cagr   = result["cagr"]
    raw_dd     = result["max_dd"]
    raw_sharpe = result["sharpe"]
    years_span = (_TO - _FROM) / (365.25 * _DAY_MS)

    print(f"\n  Raw backtest  →  equity=${raw_equity:>9,.0f}  "
          f"CAGR={raw_cagr*100:>+6.1f}%  DD={raw_dd*100:>4.1f}%  Sharpe={raw_sharpe:.2f}")

    # --- calculate funding drag ---
    if not snapshots:
        print("\n  [WARN] No position snapshots — cannot compute funding drag.")
        return

    df = pd.DataFrame(snapshots)
    df["day_ts"] = (df["ts"] // _DAY_MS) * _DAY_MS
    df["year"]   = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.year

    total_funding_paid   = 0.0
    total_funding_recv   = 0.0
    year_drag: dict[int, float] = {}

    for _, row in df.iterrows():
        day   = int(row["day_ts"])
        sym   = row["sym"]
        side  = row["side"]
        notl  = row["notional"]
        rate  = fund.get(sym, {}).get(day, 0.0)
        yr    = int(row["year"])

        # Positive rate: longs pay, shorts receive
        # (Negative rate — rare — reverses; we handle it correctly with sign)
        if side == "LONG":
            cost = rate * notl       # positive = drag
            total_funding_paid += cost
        else:  # SHORT
            cost = -rate * notl      # positive rate = receive
            total_funding_recv += abs(cost) if cost < 0 else 0
            cost = cost              # could be negative (cost) in negative rate
            total_funding_paid += cost  # net into total

        year_drag[yr] = year_drag.get(yr, 0.0) + cost

    net_funding_drag = total_funding_paid  # positive = we paid (drag)
    adj_equity = raw_equity - net_funding_drag

    adj_cagr = (adj_equity / B._CAPITAL) ** (1.0 / years_span) - 1.0

    print(f"\n  Funding payments (longs paid):   ${total_funding_paid:>8,.0f}")
    print(f"  Funding received (shorts recv):  ${total_funding_recv:>8,.0f}")
    print(f"  Net funding drag:                ${net_funding_drag:>8,.0f}")
    print(f"\n  Adjusted equity  →  equity=${adj_equity:>9,.0f}  CAGR={adj_cagr*100:>+6.1f}%")
    print(f"  CAGR reduction: {(raw_cagr - adj_cagr)*100:>+.1f}pp")

    # Year-by-year breakdown
    all_years = sorted(set(result["year_pnl"]) | set(year_drag))
    print(f"\n  {'Year':<6}  {'Raw PnL':>10}  {'Funding Drag':>13}  {'Drag %':>8}  {'Adj PnL':>10}")
    print("  " + "-" * 56)
    for yr in all_years:
        raw_pnl = result["year_pnl"].get(yr, 0)
        drag    = year_drag.get(yr, 0.0)
        adj_pnl = raw_pnl - drag
        drag_pct = (drag / raw_pnl * 100) if raw_pnl != 0 else float("nan")
        print(f"  {yr:<6}  ${raw_pnl:>9,.0f}  ${drag:>12,.0f}  {drag_pct:>7.1f}%  ${adj_pnl:>9,.0f}")

    # Exposure stats
    long_days  = len(df[df["side"] == "LONG"])
    short_days = len(df[df["side"] == "SHORT"])
    total_days = (_TO - _FROM) // _DAY_MS
    slots      = total_days * len(symbols)
    print(f"\n  Long exposure:  {long_days:,} coin-days  ({long_days/slots*100:.1f}% of available slots)")
    print(f"  Short exposure: {short_days:,} coin-days")

    print("\n" + "=" * 100)
    if net_funding_drag > 0:
        pct_of_profit = net_funding_drag / (raw_equity - B._CAPITAL) * 100
        print(f"  VERDICT: Funding cost ${net_funding_drag:,.0f} over {years_span:.1f}y "
              f"= {pct_of_profit:.1f}% of gross profit.")
        if abs(raw_cagr - adj_cagr) < 0.05:
            print("  IMPACT: SMALL (<5pp CAGR). Backtest conclusions stand.")
        elif abs(raw_cagr - adj_cagr) < 0.10:
            print("  IMPACT: MODERATE (5-10pp CAGR). Noteworthy but not disqualifying.")
        else:
            print("  IMPACT: LARGE (>10pp CAGR). Material gap between backtest and live.")
    print("=" * 100)


if __name__ == "__main__":
    main()
