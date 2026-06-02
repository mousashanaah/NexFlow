#!/usr/bin/env python3
"""Higher-Timeframe Trend-Following Backtest — 4H timeframe, trailing exit.

Strategic rationale: every fast, fade/mean-reversion, taker-fee, ~24h-hold
mechanism tested so far (compression, lead-lag, funding extremes, OI confirm,
and the cascade fade) has died — largely because a ~0.12% round-trip fee buries
a tiny edge when you trade often. This mechanism changes THREE axes at once:
  1. Direction of edge : TREND-following (not mean-reversion) — a different
     return source (positive skew, lets winners run).
  2. Speed             : 4H bars with a TRAILING exit → multi-day holds, so far
     fewer trades and far less fee drag per unit of move captured.
  3. Exit              : chandelier trailing stop (let winners run, cut losers).

Mechanism hypothesis: crypto perps trend. A breakout above the prior N-bar high
(4H) continues often/far enough that a trailing stop banks more than the losers
+ fees cost. This is the classic Donchian/turtle structure.

Two-section output
------------------
SECTION A  Mechanism Validity — pre-fee: after an N-bar breakout, what is the
           forward MFE/MAE and the share of breakouts that run >= 2R before
           giving back 1R? Does trend continuation exist at this horizon?
SECTION B  Monetization — the pre-committed breakout + trailing-stop rule after
           fees, shared-equity, OOS split, kill/go verdict.

Parameters (all pre-committed):
  timeframe : 4H (aggregated from 1H candles).
  entry     : close > highest high of prior 30 bars (5 days)  → LONG
              close < lowest  low  of prior 30 bars (5 days)  → SHORT
              entered at the NEXT bar open (no lookahead).
  initial stop : 3.0 × ATR(14) from entry.
  trailing  : chandelier — long stop = max(stop, highest_close_since_entry
              - 3.0×ATR); short symmetric. Only ratchets favorably.
  risk      : 1% account equity per trade;  fee 0.06% taker per side.
  one position at a time per symbol; new breakouts while in a position ignored.

Kill criteria:
  PF < 1.10 | max DD > 40% | n_trades < 60 | OOS PF (2023+) < 0.85 × full PF

Self-contained: loads 1H candles from data/candles/{SYMBOL}_1H.parquet and
aggregates them to 4H in-process. No new downloads, no keys.

Usage:
  python scripts/backtest_htf_trend.py
  python scripts/backtest_htf_trend.py --capital 50000 --lookback 30
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants (pre-committed)
# ---------------------------------------------------------------------------
_TAKER_FEE   = 0.0006
_RISK_PCT    = 0.01
_ATR_PERIOD  = 14
_LOOKBACK    = 30        # prior 4H bars for the breakout channel (5 days)
_STOP_MULT   = 3.0       # initial + trailing stop distance, in ATR
_BARS_PER_4H = 4         # 1H → 4H aggregation factor

_KILL_PF, _KILL_TRADES, _KILL_MAX_DD = 1.10, 60, 0.40
_KILL_OOS_RATIO, _OOS_START_YEAR = 0.85, 2023
_GO_PF, _GO_CAGR, _GO_DD = 1.30, 0.20, 0.40

_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_BAR_MS_4H = 4 * 3_600_000


# ---------------------------------------------------------------------------
# Data: load 1H, aggregate to 4H
# ---------------------------------------------------------------------------

def _load_1h(symbol: str, data_dir: Path) -> dict:
    path = data_dir / f"{symbol}_1H.parquet"
    if not path.exists():
        print(f"[ERROR] Missing: {path}")
        sys.exit(1)
    d = pq.read_table(path).to_pydict()
    idx = sorted(range(len(d["open_time"])), key=lambda i: d["open_time"][i])
    return {c: [d[c][i] for i in idx] for c in d}


def _aggregate_4h(c1h: dict) -> dict:
    """Aggregate 1H bars into 4H bars aligned to 00/04/08/12/16/20 UTC."""
    ot = c1h["open_time"]; op = c1h["open"]; hi = c1h["high"]
    lo = c1h["low"]; cl = c1h["close"]; vo = c1h.get("volume", [0.0]*len(ot))
    out = {k: [] for k in ("open_time", "open", "high", "low", "close", "volume")}
    bucket = None
    for i in range(len(ot)):
        b_start = (ot[i] // _BAR_MS_4H) * _BAR_MS_4H
        if bucket is None or b_start != bucket["open_time"]:
            if bucket is not None:
                for k in out:
                    out[k].append(bucket[k])
            bucket = {"open_time": b_start, "open": op[i], "high": hi[i],
                      "low": lo[i], "close": cl[i], "volume": vo[i]}
        else:
            bucket["high"] = max(bucket["high"], hi[i])
            bucket["low"] = min(bucket["low"], lo[i])
            bucket["close"] = cl[i]
            bucket["volume"] += vo[i]
    if bucket is not None:
        for k in out:
            out[k].append(bucket[k])
    return out


def _wilder_atr_series(highs, lows, closes, period) -> list[Optional[float]]:
    n = len(closes)
    atrs: list[Optional[float]] = [None] * n
    atr_val: Optional[float] = None
    seed: list[float] = []
    for i in range(n):
        if i == 0:
            continue
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        if atr_val is None:
            seed.append(tr)
            if len(seed) == period:
                atr_val = sum(seed) / period
        else:
            atr_val = (atr_val*(period-1) + tr) / period
        atrs[i] = atr_val
    return atrs


# ---------------------------------------------------------------------------
# Breakout signal construction
# ---------------------------------------------------------------------------

@dataclass
class _Signal:
    symbol: str; side: str; signal_idx: int; entry_idx: int; entry_ms: int


def _build_signals(symbol, candles, lookback) -> list[_Signal]:
    highs, lows, closes, ot = candles["high"], candles["low"], candles["close"], candles["open_time"]
    n = len(closes)
    out: list[_Signal] = []
    for i in range(lookback, n - 1):
        prior_high = max(highs[i-lookback:i])
        prior_low = min(lows[i-lookback:i])
        if closes[i] > prior_high:
            out.append(_Signal(symbol, "LONG", i, i+1, ot[i+1]))
        elif closes[i] < prior_low:
            out.append(_Signal(symbol, "SHORT", i, i+1, ot[i+1]))
    return out


# ---------------------------------------------------------------------------
# SECTION A — trend continuation (MFE/MAE in R units)
# ---------------------------------------------------------------------------

def _section_a(signals_by, candles_by, atr_by):
    print("\n" + "=" * 70)
    print("  SECTION A — MECHANISM VALIDITY (pre-fee, event-level analysis)")
    print("=" * 70)
    print(f"  Breakout = close beyond prior {_LOOKBACK}-bar (4H) channel high/low.")
    print(f"  R = {_STOP_MULT}×ATR(14) at entry. We measure how far breakouts run.")
    print(f"  Forward MFE/MAE over the next 30 bars (5 days) from next-bar open.")

    horizon = 30
    agg = {"LONG": [], "SHORT": []}
    run2R = {"LONG": [0, 0], "SHORT": [0, 0]}   # [reached 2R before -1R, total]

    for symbol in _SYMBOLS:
        candles = candles_by[symbol]; atrs = atr_by[symbol]
        highs, lows, opens = candles["high"], candles["low"], candles["open"]
        n = len(opens)
        for s in signals_by.get(symbol, []):
            ei = s.entry_idx
            atr = atrs[ei] if ei < len(atrs) else None
            if atr is None or atr <= 0:
                continue
            entry = opens[ei]
            R = _STOP_MULT * atr
            best_mfe = 0.0; worst_mae = 0.0
            hit2R = False; stopped = False
            for f in range(1, horizon + 1):
                j = ei + f
                if j >= n:
                    break
                if s.side == "LONG":
                    fav = (highs[j] - entry) / R
                    adv = (entry - lows[j]) / R
                else:
                    fav = (entry - lows[j]) / R
                    adv = (highs[j] - entry) / R
                best_mfe = max(best_mfe, fav); worst_mae = max(worst_mae, adv)
                if not stopped and not hit2R:
                    if fav >= 2.0:
                        hit2R = True
                    elif adv >= 1.0:
                        stopped = True
            agg[s.side].append((best_mfe, worst_mae))
            run2R[s.side][1] += 1
            if hit2R:
                run2R[s.side][0] += 1

    for side, label in (("LONG", "LONG breakouts"), ("SHORT", "SHORT breakouts")):
        data = agg[side]
        n = len(data)
        print(f"\n  {label}: {n:,} events")
        if not n:
            continue
        mfes = sorted(d[0] for d in data); maes = sorted(d[1] for d in data)
        def med(x): return x[len(x)//2]
        print(f"    Median MFE: {med(mfes):.2f} R    Median MAE: {med(maes):.2f} R")
        print(f"    Mean   MFE: {sum(mfes)/n:.2f} R    Mean   MAE: {sum(maes)/n:.2f} R")
        hit, tot = run2R[side]
        print(f"    Reached +2R before -1R: {hit/tot*100:.1f}%  ({hit}/{tot})")

    # Mechanism holds if breakouts reach 2R-before-1R at a rate that, with the
    # asymmetric payoff, is profitable pre-fee: p*2 - (1-p)*1 > 0  → p > 0.333.
    def rate(side):
        h, t = run2R[side]
        return h / t if t else float("nan")
    lr, sr = rate("LONG"), rate("SHORT")
    long_ok = not math.isnan(lr) and lr > 0.333
    short_ok = not math.isnan(sr) and sr > 0.333
    assessment = "STRONG" if (long_ok and short_ok) else ("WEAK" if (long_ok or short_ok) else "NO EDGE")
    print(f"\n  Mechanism assessment: {assessment}")
    print(f"    (breakeven needs >33.3% reaching +2R-before-1R given the 2:1 payoff)")
    if assessment == "NO EDGE":
        print("    Breakouts do not run far enough often enough to pay the 2:1 structure.")
    elif assessment == "WEAK":
        print("    Trend continuation works in only one direction.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# SECTION B — shared-equity sim with chandelier trailing stop
# ---------------------------------------------------------------------------

@dataclass
class _Trade:
    symbol: str; direction: str; entry_price: float; exit_price: float
    entry_ms: int; exit_ms: int; size: float; pnl: float; fees: float
    bars_held: int; exit_reason: str

@dataclass
class _OpenPos:
    symbol: str; direction: str; entry_price: float; entry_ms: int
    entry_idx: int; size: float; stop: float; entry_fee: float
    best_close: float; atr_at_entry: float; bars_held: int


def _year(ms): return datetime.fromtimestamp(ms/1000, tz=timezone.utc).year

def _max_dd(eq):
    peak = eq[0][1] if eq else 0.0; mdd = 0.0
    for _, e in eq:
        if e > peak: peak = e
        if peak > 0:
            dd = (peak-e)/peak
            if dd > mdd: mdd = dd
    return mdd

def _pf(ts):
    gp = sum(t.pnl for t in ts if t.pnl > 0)
    gl = abs(sum(t.pnl for t in ts if t.pnl <= 0))
    return gp/gl if gl > 0 else float("inf")


def _section_b(signals_by, candles_by, atr_by, initial_equity):
    sig_at = {}
    for symbol in _SYMBOLS:
        m = {}
        for s in signals_by.get(symbol, []):
            if s.entry_idx not in m:
                m[s.entry_idx] = s
        sig_at[symbol] = m

    timeline = []
    for symbol in _SYMBOLS:
        for idx, ts in enumerate(candles_by[symbol]["open_time"]):
            timeline.append((ts, symbol, idx))
    rank = {s: r for r, s in enumerate(_SYMBOLS)}
    timeline.sort(key=lambda x: (x[0], rank[x[1]]))

    equity = initial_equity
    eq_curve = [(timeline[0][0], equity)] if timeline else []
    trades: list[_Trade] = []
    open_pos = {s: None for s in _SYMBOLS}

    for ts, symbol, idx in timeline:
        c = candles_by[symbol]
        c_open, c_high, c_low, c_close = c["open"][idx], c["high"][idx], c["low"][idx], c["close"][idx]
        atr = atr_by[symbol][idx]
        pos = open_pos[symbol]

        if pos is not None and idx > pos.entry_idx:
            pos.bars_held += 1
            # update trailing stop using current ATR and best close since entry
            cur_atr = atr if (atr is not None and atr > 0) else pos.atr_at_entry
            if pos.direction == "LONG":
                pos.best_close = max(pos.best_close, c_close)
                pos.stop = max(pos.stop, pos.best_close - _STOP_MULT * cur_atr)
            else:
                pos.best_close = min(pos.best_close, c_close)
                pos.stop = min(pos.stop, pos.best_close + _STOP_MULT * cur_atr)
            # check stop hit intrabar
            stop_hit = False; exit_price = None
            if pos.direction == "LONG" and c_low <= pos.stop:
                stop_hit = True; exit_price = pos.stop
            elif pos.direction == "SHORT" and c_high >= pos.stop:
                stop_hit = True; exit_price = pos.stop
            if stop_hit:
                exit_fee = exit_price * pos.size * _TAKER_FEE
                raw = ((exit_price-pos.entry_price) if pos.direction == "LONG"
                       else (pos.entry_price-exit_price)) * pos.size
                equity += raw - exit_fee
                trades.append(_Trade(symbol, pos.direction, pos.entry_price, exit_price,
                                     pos.entry_ms, ts, pos.size, raw-exit_fee,
                                     pos.entry_fee+exit_fee, pos.bars_held, "TRAIL"))
                open_pos[symbol] = None; pos = None

        sig = sig_at[symbol].get(idx)
        if sig is not None and open_pos[symbol] is None and atr is not None and atr > 0:
            entry_price = c_open
            stop_dist = _STOP_MULT * atr
            stop_price = (entry_price-stop_dist if sig.side == "LONG" else entry_price+stop_dist)
            size = (equity*_RISK_PCT)/stop_dist
            if size > 0 and entry_price > 0:
                entry_fee = entry_price*size*_TAKER_FEE
                equity -= entry_fee
                open_pos[symbol] = _OpenPos(symbol, sig.side, entry_price, ts, idx, size,
                                            stop_price, entry_fee, c_close, atr, 0)
        eq_curve.append((ts, equity))

    for symbol in _SYMBOLS:
        pos = open_pos[symbol]
        if pos is None: continue
        c = candles_by[symbol]; li = len(c["close"]) - 1
        exit_price = c["close"][li]; exit_ms = c["open_time"][li]
        exit_fee = exit_price*pos.size*_TAKER_FEE
        raw = ((exit_price-pos.entry_price) if pos.direction == "LONG"
               else (pos.entry_price-exit_price)) * pos.size
        equity += raw - exit_fee
        trades.append(_Trade(symbol, pos.direction, pos.entry_price, exit_price,
                             pos.entry_ms, exit_ms, pos.size, raw-exit_fee,
                             pos.entry_fee+exit_fee, pos.bars_held, "END"))
    if eq_curve:
        eq_curve.append((eq_curve[-1][0], equity))
    trades.sort(key=lambda t: t.entry_ms)
    _print_section_b(trades, eq_curve, initial_equity, equity)


def _print_section_b(trades, eq_curve, initial_equity, final_equity):
    print("\n" + "=" * 70)
    print("  SECTION B — MONETIZATION BACKTEST (after fees)")
    print("=" * 70)
    if not trades:
        print("\n  [RESULT] No trades generated.")
        print("=" * 70)
        _print_verdict([], 0.0, 0.0, initial_equity, final_equity)
        return
    first_ms = min(t.entry_ms for t in trades); last_ms = max(t.exit_ms for t in trades)
    years = (last_ms-first_ms)/(1000*86400*365.25)
    cagr = (final_equity/initial_equity)**(1/years)-1 if years > 0 else 0.0
    mdd = _max_dd(eq_curve)
    wins = [t for t in trades if t.pnl > 0]
    pf_full = _pf(trades); wr = len(wins)/len(trades)*100
    total_fees = sum(t.fees for t in trades)
    n_l = sum(1 for t in trades if t.direction == "LONG")
    n_s = sum(1 for t in trades if t.direction == "SHORT")
    avg_bars = sum(t.bars_held for t in trades)/len(trades)

    print(f"  Parameters    : breakout={_LOOKBACK}bar(4H) channel  trail={_STOP_MULT}×ATR({_ATR_PERIOD})")
    print(f"  Period        : {datetime.fromtimestamp(first_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
          f" → {datetime.fromtimestamp(last_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${initial_equity:,.0f}")
    print(f"  Final equity  : ${final_equity:,.0f}")
    print(f"  Net PnL       : ${final_equity-initial_equity:+,.0f}  ({(final_equity-initial_equity)/initial_equity*100:+.1f}%)")
    print(f"  CAGR          : {cagr*100:.1f}%")
    print(f"  Max drawdown  : {mdd*100:.1f}%")
    print(f"  Profit factor : {pf_full:.2f}")
    print(f"  Total trades  : {len(trades)}  (L:{n_l}  S:{n_s})")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg hold      : {avg_bars:.1f} bars ({avg_bars*4:.0f}h)")
    print(f"  Total fees    : ${total_fees:,.0f}")

    print(f"\n  {'─'*70}")
    print(f"  {'Symbol':<10} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*12}")
    for symbol in _SYMBOLS:
        st = [t for t in trades if t.symbol == symbol]
        if not st:
            print(f"  {symbol:<10} {0:>7} {'n/a':>7} {'n/a':>7} {'$0':>12}"); continue
        sw = [t for t in st if t.pnl > 0]; spf = _pf(st)
        pf_str = f"{spf:.2f}" if not math.isinf(spf) else "inf"
        print(f"  {symbol:<10} {len(st):>7} {len(sw)/len(st)*100:>6.1f}% {pf_str:>7} ${sum(t.pnl for t in st):>+10,.0f}")

    print(f"\n  {'─'*70}")
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}  Status")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*12}  {'─'*10}")
    for yr in sorted({_year(t.entry_ms) for t in trades}):
        yt = [t for t in trades if _year(t.entry_ms) == yr]
        yw = [t for t in yt if t.pnl > 0]; ypf = _pf(yt)
        pf_str = f"{ypf:>7.2f}" if not math.isinf(ypf) else f"{'inf':>7}"
        status = "+" if (math.isinf(ypf) or ypf >= 1.0) else "-"
        print(f"  {yr:<8} {len(yt):>7} {len(yw)/len(yt)*100:>6.1f}% {pf_str} ${sum(t.pnl for t in yt):>+10,.0f}  {status}")

    sb = sorted(trades, key=lambda t: t.pnl, reverse=True)
    print(f"\n  {'─'*70}")
    print("  Top 5 trades by PnL:")
    for t in sb[:5]:
        dt = datetime.fromtimestamp(t.entry_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.symbol:<8} {t.direction:<5} entry {dt}  held={t.bars_held}b  ${t.pnl:>+,.0f}")
    print("  Bottom 5 trades by PnL:")
    for t in sb[-5:]:
        dt = datetime.fromtimestamp(t.entry_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.symbol:<8} {t.direction:<5} entry {dt}  held={t.bars_held}b  ${t.pnl:>+,.0f}")

    oos = [t for t in trades if _year(t.entry_ms) >= _OOS_START_YEAR]
    ins = [t for t in trades if _year(t.entry_ms) < _OOS_START_YEAR]
    pf_oos = _pf(oos)
    print(f"\n  {'─'*70}")
    print(f"  OOS split (IS = <{_OOS_START_YEAR}, OOS = {_OOS_START_YEAR}+):")
    print(f"    IS : {len(ins)} trades, PF {_pf(ins):.2f}" if ins else "    IS : no trades")
    print(f"    OOS: {len(oos)} trades, PF {pf_oos:.2f}" if oos else "    OOS: no trades")
    print("=" * 70)
    _print_verdict(trades, pf_full, mdd, initial_equity, final_equity, pf_oos=pf_oos, pf_full_ref=pf_full)


def _print_verdict(trades, pf_full, mdd, initial_equity, final_equity,
                   pf_oos=float("nan"), pf_full_ref=float("nan")):
    first_ms = min(t.entry_ms for t in trades) if trades else 0
    last_ms = max(t.exit_ms for t in trades) if trades else 0
    years = (last_ms-first_ms)/(1000*86400*365.25) if last_ms > first_ms else 0
    cagr = (final_equity/initial_equity)**(1/years)-1 if years > 0 and initial_equity > 0 else 0.0

    kill = []; n = len(trades)
    if pf_full < _KILL_PF: kill.append(f"Profit factor {pf_full:.2f} < kill threshold {_KILL_PF:.2f}")
    if mdd > _KILL_MAX_DD: kill.append(f"Max drawdown {mdd*100:.1f}% > kill threshold {_KILL_MAX_DD*100:.0f}%")
    if n < _KILL_TRADES: kill.append(f"Trade count {n} < minimum {_KILL_TRADES}")
    if not math.isnan(pf_oos) and not math.isnan(pf_full_ref):
        thr = _KILL_OOS_RATIO*pf_full_ref
        if pf_oos < thr:
            kill.append(f"OOS PF {pf_oos:.2f} < {_KILL_OOS_RATIO:.0%} × full-period PF {pf_full_ref:.2f} (threshold {thr:.2f}) — regime-specific")

    go = []
    if pf_full >= _GO_PF: go.append(f"PF {pf_full:.2f} >= {_GO_PF}")
    if cagr >= _GO_CAGR: go.append(f"CAGR {cagr*100:.1f}% >= {_GO_CAGR*100:.0f}%")
    if mdd <= _GO_DD: go.append(f"Max DD {mdd*100:.1f}% <= {_GO_DD*100:.0f}%")

    passes_core = pf_full >= _KILL_PF and mdd <= _KILL_MAX_DD and n >= _KILL_TRADES
    oos_weak = (not math.isnan(pf_oos) and not math.isnan(pf_full_ref)
                and pf_oos < _KILL_OOS_RATIO*pf_full_ref)

    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    if passes_core and oos_weak:
        print("\n  MARGINAL -- passes PF/DD/trade-count gates but OOS is weak:\n")
        print(f"     * PF {pf_full:.2f} >= {_KILL_PF:.2f} (kill gate)")
        print(f"     * Max DD {mdd*100:.1f}% <= {_KILL_MAX_DD*100:.0f}% (kill gate)")
        print(f"     * Trades {n} >= {_KILL_TRADES} (kill gate)")
        print(f"     * OOS PF {pf_oos:.2f} < {_KILL_OOS_RATIO:.0%} × full PF {pf_full_ref:.2f}")
        print("\n  Action: regime-sensitive — run paper trading before adding capital.")
    elif kill:
        print("\n  KILL -- Strategy fails one or more kill criteria:\n")
        for r in kill: print(f"     * {r}")
        print("\n  Action: do NOT deploy. Move to next hypothesis.")
    elif len(go) == 3:
        print("\n  GO -- All go criteria met:\n")
        for f in go: print(f"     * {f}")
        print("\n  Action: proceed to paper deployment.")
    else:
        print(f"\n  MARGINAL -- {len(go)}/3 go criteria met. Deploy on paper with caution.")
        print("  Passed  :", ", ".join(go) if go else "none")
    print("=" * 70)


def main():
    global _SYMBOLS
    parser = argparse.ArgumentParser(description="Higher-Timeframe Trend-Following Backtest (4H)")
    parser.add_argument("--data", default="data/candles")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--lookback", type=int, default=_LOOKBACK)
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override the symbol universe (default BTCUSDT ETHUSDT)")
    args = parser.parse_args()

    if args.symbols:
        _SYMBOLS = args.symbols

    data_dir = _REPO_ROOT / args.data
    lookback = args.lookback

    print(f"\n{'='*70}")
    print("  HIGHER-TIMEFRAME TREND-FOLLOWING BACKTEST — 4H timeframe")
    print(f"{'='*70}")
    print("  Hypothesis: crypto perps trend; breakouts run far enough that a")
    print("              trailing stop banks more than losers + fees cost.")
    print("  Pre-committed rule:")
    print(f"    entry = close beyond prior {lookback}-bar 4H channel (next-bar open)")
    print(f"    stop  = {_STOP_MULT}×ATR(14), chandelier trailing (ratchets favorably)")
    print(f"    fee   = {_TAKER_FEE*100:.4f}% taker/side  risk = {_RISK_PCT*100:.1f}%  capital = ${args.capital:,.0f}")
    print(f"    symbols = {', '.join(_SYMBOLS)}")

    print(f"\nLoading 1H candles and aggregating to 4H ...")
    candles_by = {}; atr_by = {}; signals_by = {}
    available = []
    for symbol in _SYMBOLS:
        if not (data_dir / f"{symbol}_1H.parquet").exists():
            print(f"  {symbol}: [SKIP] no 1H parquet (download may have failed)")
            continue
        available.append(symbol)
        c1h = _load_1h(symbol, data_dir)
        c4h = _aggregate_4h(c1h)
        candles_by[symbol] = c4h
        atr_by[symbol] = _wilder_atr_series(c4h["high"], c4h["low"], c4h["close"], _ATR_PERIOD)
        signals_by[symbol] = _build_signals(symbol, c4h, lookback)
        print(f"  {symbol}: {len(c1h['close']):,} 1H → {len(c4h['close']):,} 4H bars, "
              f"{len(signals_by[symbol]):,} breakout signals")

    _SYMBOLS = available
    if not _SYMBOLS:
        print("[ERROR] No symbols available. Aborting.")
        sys.exit(1)

    _section_a(signals_by, candles_by, atr_by)
    _section_b(signals_by, candles_by, atr_by, args.capital)

    print(f"\n{'='*70}")
    print("  Backtest complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
