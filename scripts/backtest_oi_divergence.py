#!/usr/bin/env python3
"""Open-Interest Divergence / Confirmation Backtest — 1H timeframe.

Mechanism hypothesis (OI-confirmed momentum):
  A strong price move backed by RISING open interest reflects fresh money taking
  conviction positions and tends to CONTINUE. A strong move on FALLING OI is just
  position-unwinding (short-covering / long-liquidation) and tends to stall.

  We therefore trade continuation only when OI confirms the move:
    strong price UP   + OI rising  → LONG  (new longs fueling the trend)
    strong price DOWN + OI rising  → SHORT (new shorts fueling the trend)

Two-section output
------------------
SECTION A  Mechanism Validity — does OI confirmation predict continuation
           (pre-fee, event-level)?  We bucket every bar into the four
           (price±, OI±) quadrants and measure forward returns. If the
           mechanism is real, the OI-confirmed quadrants should show
           continuation and the OI-diverging quadrants should not.
SECTION B  Monetization — can the pre-committed continuation rule be
           extracted after fees?

Parameters (all pre-committed):
  lookback   : 24 bars (24h) for both price change and OI change.
  strong move: price 24h-change in top/bottom 20% of a TRAILING 90-day window
               (per symbol, prior obs only — no lookahead; >= 30 obs required).
  oi confirm : OI 24h-change > 0  (rising).
  entry      : next bar OPEN after the signal bar (no lookahead).
  hold       : 24 bars (24h) OR 2.0×ATR(14) stop, whichever first.
  risk       : 1% account equity per trade;  fee 0.06% taker per side.
  one position at a time per symbol.

Kill criteria:
  PF < 1.10 | max DD > 40% | n_trades < 60 | OOS PF (2023+) < 0.85 × full PF

Self-contained: loads 1H candles from data/candles/{SYMBOL}_1H.parquet and 1H
open interest from data/oi/{SYMBOL}_OI_1H.parquet. The OI cache is produced by
scripts/download_open_interest_history.py (run locally once, commit the parquet).

Usage:
  python scripts/backtest_oi_divergence.py
  python scripts/backtest_oi_divergence.py --capital 50000 --threshold-pct 20
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
# Constants (pre-committed strategy parameters)
# ---------------------------------------------------------------------------
_TAKER_FEE     = 0.0006   # 0.06% per side — Bitget USDT-Perp taker
_RISK_PCT      = 0.01     # 1% account equity per trade
_ATR_PERIOD    = 14
_HOLD_HOURS    = 24
_STOP_MULT     = 2.0
_LOOKBACK      = 24       # bars for price-change and OI-change measurement

# Trailing-distribution window for the "strong move" percentile (per symbol)
_TRAIL_DAYS    = 90
_MIN_OBS       = 30       # need >= this many trailing price-change obs or skip
_HOUR_MS       = 3_600_000
_DAY_MS        = 86_400_000

# Kill thresholds
_KILL_PF        = 1.10
_KILL_TRADES    = 60
_KILL_MAX_DD    = 0.40
_KILL_OOS_RATIO = 0.85
_OOS_START_YEAR = 2023

# Go thresholds
_GO_PF   = 1.30
_GO_CAGR = 0.20
_GO_DD   = 0.40

_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_1h(symbol: str, data_dir: Path) -> dict:
    path = data_dir / f"{symbol}_1H.parquet"
    if not path.exists():
        print(f"[ERROR] Missing: {path}")
        sys.exit(1)
    d = pq.read_table(path).to_pydict()
    idx = sorted(range(len(d["open_time"])), key=lambda i: d["open_time"][i])
    return {col: [d[col][i] for i in idx] for col in d}


def _load_oi(symbol: str, oi_dir: Path) -> dict[int, float]:
    """Return {timestamp_ms -> open_interest}, or exit with guidance if missing."""
    path = oi_dir / f"{symbol}_OI_1H.parquet"
    if not path.exists():
        print(f"\n[ERROR] No open-interest cache found for {symbol}.")
        print(f"        Expected: {path}")
        print()
        print("        Run this on your LOCAL machine (no API key needed):")
        print("          python scripts\\download_open_interest_history.py")
        print("        Then commit and push:")
        print("          git add data/oi/")
        print("          git commit -m 'add open interest history cache'")
        print("          git push")
        sys.exit(1)
    d = pq.read_table(path).to_pydict()
    return {int(d["timestamp_ms"][i]): float(d["open_interest"][i])
            for i in range(len(d["timestamp_ms"]))}


def _align_oi(candles: dict, oi_map: dict[int, float]) -> list[Optional[float]]:
    """Align OI to each candle by open_time, carrying the last known value forward.

    Returns a per-candle list (None before the first OI observation).
    """
    open_times = candles["open_time"]
    aligned: list[Optional[float]] = [None] * len(open_times)
    last: Optional[float] = None
    matched = 0
    for i, ts in enumerate(open_times):
        v = oi_map.get(int(ts))
        if v is not None:
            last = v
            matched += 1
        aligned[i] = last
    return aligned, matched


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _wilder_atr_series(highs, lows, closes, period) -> list[Optional[float]]:
    n = len(closes)
    atrs: list[Optional[float]] = [None] * n
    atr_val: Optional[float] = None
    seed: list[float] = []
    for i in range(n):
        if i == 0:
            continue
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        if atr_val is None:
            seed.append(tr)
            if len(seed) == period:
                atr_val = sum(seed) / period
        else:
            atr_val = (atr_val * (period - 1) + tr) / period
        atrs[i] = atr_val
    return atrs


def _percentile(sorted_vals: list[float], pct: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(math.floor(rank)); hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------

@dataclass
class _Signal:
    symbol:    str
    side:      str       # "LONG" / "SHORT"
    signal_idx: int      # bar whose CLOSE produced the signal
    entry_idx: int       # next bar; its OPEN is the entry
    entry_ms:  int
    price_chg: float
    oi_chg:    float


def _compute_changes(candles: dict, oi: list[Optional[float]]) -> tuple[list[Optional[float]], list[Optional[float]]]:
    """Per-bar 24h price-change and OI-change (None where undefined)."""
    closes = candles["close"]
    n = len(closes)
    price_chg: list[Optional[float]] = [None] * n
    oi_chg:    list[Optional[float]] = [None] * n
    for i in range(_LOOKBACK, n):
        c0 = closes[i - _LOOKBACK]
        if c0 and c0 > 0:
            price_chg[i] = (closes[i] - c0) / c0
        o_now = oi[i]
        o_prev = oi[i - _LOOKBACK]
        if o_now is not None and o_prev is not None and o_prev > 0:
            oi_chg[i] = (o_now - o_prev) / o_prev
    return price_chg, oi_chg


def _build_signals(symbol, candles, price_chg, oi_chg, low_pct, high_pct) -> list[_Signal]:
    """Continuation signals: strong move (trailing-pct) confirmed by rising OI."""
    open_times = candles["open_time"]
    n = len(open_times)
    signals: list[_Signal] = []

    # Trailing window of prior price-change observations (sliding by time).
    pc_times: list[int] = []
    pc_vals:  list[float] = []
    win_start = 0

    for i in range(_LOOKBACK, n):
        pc = price_chg[i]
        oc = oi_chg[i]
        # First, evaluate the signal at bar i using the trailing window built from
        # observations strictly BEFORE i (no lookahead), then append bar i.
        if pc is not None and oc is not None:
            lo_bound = open_times[i] - _TRAIL_DAYS * _DAY_MS
            while win_start < len(pc_times) and pc_times[win_start] < lo_bound:
                win_start += 1
            window = pc_vals[win_start:]
            if len(window) >= _MIN_OBS:
                sorted_win = sorted(window)
                hi_thr = _percentile(sorted_win, high_pct)
                lo_thr = _percentile(sorted_win, low_pct)
                side: Optional[str] = None
                if pc >= hi_thr and oc > 0:
                    side = "LONG"
                elif pc <= lo_thr and oc > 0:
                    side = "SHORT"
                if side is not None and (i + 1) < n:
                    signals.append(_Signal(
                        symbol=symbol, side=side,
                        signal_idx=i, entry_idx=i + 1,
                        entry_ms=open_times[i + 1],
                        price_chg=pc, oi_chg=oc,
                    ))
        # Append current obs to the trailing window for future bars.
        if pc is not None:
            pc_times.append(open_times[i])
            pc_vals.append(pc)

    return signals


# ---------------------------------------------------------------------------
# SECTION A — Mechanism validity (quadrant forward returns)
# ---------------------------------------------------------------------------

def _section_a(candles_by_symbol, price_chg_by, oi_chg_by, low_pct, high_pct):
    print("\n" + "=" * 70)
    print("  SECTION A — MECHANISM VALIDITY (pre-fee, event-level analysis)")
    print("=" * 70)
    print(f"  Strong move = price 24h-change in top/bottom {100-high_pct:.0f}%/{low_pct:.0f}% "
          f"of TRAILING {_TRAIL_DAYS}d window")
    print(f"  Quadrants split every eligible bar by sign of (price 24h-chg, OI 24h-chg).")
    print(f"  Continuation hypothesis: P+/OI+ → keeps rising; P-/OI+ → keeps falling.")

    # Quadrant aggregates: forward returns from NEXT bar open.
    quads = ["P+/OI+", "P+/OI-", "P-/OI+", "P-/OI-"]
    agg = {q: {"n": 0, "r8": [], "r24": [], "r48": [], "cont_hit": 0, "cont_n": 0} for q in quads}

    for symbol in _SYMBOLS:
        candles = candles_by_symbol[symbol]
        opens = candles["open"]; closes = candles["close"]
        n = len(closes)
        pc = price_chg_by[symbol]; oc = oi_chg_by[symbol]

        def fwd(entry_idx, hours):
            tgt = entry_idx + hours
            eo = opens[entry_idx]
            if eo <= 0 or tgt >= n:
                return None
            return (closes[tgt] - eo) / eo

        for i in range(_LOOKBACK, n - 1):
            if pc[i] is None or oc[i] is None:
                continue
            psign = "+" if pc[i] >= 0 else "-"
            osign = "+" if oc[i] >= 0 else "-"
            q = f"P{psign}/OI{osign}"
            entry_idx = i + 1            # forward measured from next bar open
            r8 = fwd(entry_idx, 8); r24 = fwd(entry_idx, 24); r48 = fwd(entry_idx, 48)
            a = agg[q]
            a["n"] += 1
            if r8  is not None: a["r8"].append(r8)
            if r24 is not None: a["r24"].append(r24)
            if r48 is not None: a["r48"].append(r48)
            # Continuation hit = price keeps moving in the prior move's direction.
            if r24 is not None:
                a["cont_n"] += 1
                if psign == "+" and r24 > 0:
                    a["cont_hit"] += 1
                elif psign == "-" and r24 < 0:
                    a["cont_hit"] += 1

    def _mean(lst):
        return f"{sum(lst)/len(lst)*100:+.4f}%" if lst else "    n/a"

    print(f"\n  Forward price return by quadrant (from next-bar open):")
    print(f"  {'Quadrant':<10} {'N':>8} {'+8h':>12} {'+24h':>12} {'+48h':>12} {'Cont24%':>9}")
    print(f"  {'─'*10} {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*9}")
    for q in quads:
        a = agg[q]
        cont = f"{a['cont_hit']/a['cont_n']*100:.1f}%" if a["cont_n"] else "n/a"
        print(f"  {q:<10} {a['n']:>8,} {_mean(a['r8']):>12} {_mean(a['r24']):>12} "
              f"{_mean(a['r48']):>12} {cont:>9}")

    # Assessment: is OI confirmation better than OI divergence at continuation?
    def _contrate(q):
        a = agg[q]
        return a["cont_hit"] / a["cont_n"] if a["cont_n"] else float("nan")
    up_conf, up_div = _contrate("P+/OI+"), _contrate("P+/OI-")
    dn_conf, dn_div = _contrate("P-/OI+"), _contrate("P-/OI-")

    def _r24(q):
        a = agg[q]["r24"]
        return sum(a)/len(a) if a else float("nan")

    print(f"\n  OI-confirmation test (continuation hit-rate, +24h):")
    print(f"    UP   moves:  OI-rising {up_conf*100:.1f}%   vs  OI-falling {up_div*100:.1f}%")
    print(f"    DOWN moves:  OI-rising {dn_conf*100:.1f}%   vs  OI-falling {dn_div*100:.1f}%")

    # Mechanism holds if confirmed quadrants both (a) beat 50% continuation AND
    # (b) beat their diverging counterpart, with forward-return signs aligned.
    signs_ok = (not math.isnan(_r24("P+/OI+")) and not math.isnan(_r24("P-/OI+"))
                and _r24("P+/OI+") > 0 and _r24("P-/OI+") < 0)
    hit_ok = (not math.isnan(up_conf) and not math.isnan(dn_conf)
              and up_conf > 0.50 and dn_conf > 0.50)
    better_than_div = (not math.isnan(up_div) and not math.isnan(dn_div)
                       and up_conf > up_div and dn_conf > dn_div)

    if hit_ok and signs_ok and better_than_div:
        assessment = "STRONG"
    elif (hit_ok or signs_ok) and better_than_div:
        assessment = "WEAK"
    else:
        assessment = "NO EDGE"
    print(f"\n  Mechanism assessment: {assessment}")
    if assessment == "NO EDGE":
        print("    OI-confirmed moves do not continue more than OI-diverging moves.")
    elif assessment == "WEAK":
        print("    OI confirmation helps in only one direction or weakly.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# SECTION B — Monetization (shared equity sim)
# ---------------------------------------------------------------------------

@dataclass
class _Trade:
    symbol: str; direction: str; entry_price: float; exit_price: float
    entry_ms: int; exit_ms: int; size: float; pnl: float; fees: float
    stop_price: float; exit_reason: str


@dataclass
class _OpenPos:
    symbol: str; direction: str; entry_price: float; entry_ms: int
    entry_idx: int; size: float; stop: float; entry_fee: float; bars_held: int


def _year(ms): return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year


def _max_dd(eq):
    peak = eq[0][1] if eq else 0.0; mdd = 0.0
    for _, e in eq:
        if e > peak: peak = e
        if peak > 0:
            dd = (peak - e) / peak
            if dd > mdd: mdd = dd
    return mdd


def _pf(ts):
    gp = sum(t.pnl for t in ts if t.pnl > 0)
    gl = abs(sum(t.pnl for t in ts if t.pnl <= 0))
    return gp / gl if gl > 0 else float("inf")


def _section_b(signals_by_symbol, candles_by_symbol, atr_by_symbol,
               hold_bars, initial_equity, low_pct, high_pct):
    # entry_idx -> signal map (first signal at a bar wins)
    sig_at = {}
    for symbol in _SYMBOLS:
        m = {}
        for s in signals_by_symbol.get(symbol, []):
            if s.entry_idx not in m:
                m[s.entry_idx] = s
        sig_at[symbol] = m

    timeline = []
    for symbol in _SYMBOLS:
        for idx, ts in enumerate(candles_by_symbol[symbol]["open_time"]):
            timeline.append((ts, symbol, idx))
    rank = {s: r for r, s in enumerate(_SYMBOLS)}
    timeline.sort(key=lambda x: (x[0], rank[x[1]]))

    equity = initial_equity
    eq_curve = [(timeline[0][0], equity)] if timeline else []
    trades: list[_Trade] = []
    open_pos: dict[str, Optional[_OpenPos]] = {s: None for s in _SYMBOLS}

    for ts, symbol, idx in timeline:
        c = candles_by_symbol[symbol]
        c_open, c_high, c_low, c_close = c["open"][idx], c["high"][idx], c["low"][idx], c["close"][idx]

        pos = open_pos[symbol]
        if pos is not None and idx > pos.entry_idx:
            pos.bars_held += 1
            stop_hit = False; exit_price = None
            if pos.direction == "LONG" and c_low <= pos.stop:
                stop_hit = True; exit_price = pos.stop
            elif pos.direction == "SHORT" and c_high >= pos.stop:
                stop_hit = True; exit_price = pos.stop
            if stop_hit or pos.bars_held >= hold_bars:
                if exit_price is None: exit_price = c_close
                reason = "STOP" if stop_hit else "HOLD"
                exit_fee = exit_price * pos.size * _TAKER_FEE
                raw = ((exit_price - pos.entry_price) if pos.direction == "LONG"
                       else (pos.entry_price - exit_price)) * pos.size
                equity += raw - exit_fee
                trades.append(_Trade(symbol, pos.direction, pos.entry_price, exit_price,
                                     pos.entry_ms, ts, pos.size, raw - exit_fee,
                                     pos.entry_fee + exit_fee, pos.stop, reason))
                open_pos[symbol] = None; pos = None

        sig = sig_at[symbol].get(idx)
        if sig is not None and open_pos[symbol] is None:
            atr = atr_by_symbol[symbol][idx]
            entry_price = c_open
            if atr is not None and atr > 0 and entry_price > 0:
                stop_dist = _STOP_MULT * atr
                stop_price = (entry_price - stop_dist if sig.side == "LONG"
                              else entry_price + stop_dist)
                size = (equity * _RISK_PCT) / stop_dist
                if size > 0:
                    entry_fee = entry_price * size * _TAKER_FEE
                    equity -= entry_fee
                    open_pos[symbol] = _OpenPos(symbol, sig.side, entry_price, ts,
                                                idx, size, stop_price, entry_fee, 0)
        eq_curve.append((ts, equity))

    # force-close
    for symbol in _SYMBOLS:
        pos = open_pos[symbol]
        if pos is None: continue
        c = candles_by_symbol[symbol]
        li = len(c["close"]) - 1
        exit_price = c["close"][li]; exit_ms = c["open_time"][li]
        exit_fee = exit_price * pos.size * _TAKER_FEE
        raw = ((exit_price - pos.entry_price) if pos.direction == "LONG"
               else (pos.entry_price - exit_price)) * pos.size
        equity += raw - exit_fee
        trades.append(_Trade(symbol, pos.direction, pos.entry_price, exit_price,
                             pos.entry_ms, exit_ms, pos.size, raw - exit_fee,
                             pos.entry_fee + exit_fee, pos.stop, "HOLD"))
    if eq_curve:
        eq_curve.append((eq_curve[-1][0], equity))
    trades.sort(key=lambda t: t.entry_ms)
    _print_section_b(trades, eq_curve, initial_equity, equity, hold_bars, low_pct, high_pct)


def _print_section_b(trades, eq_curve, initial_equity, final_equity, hold_bars, low_pct, high_pct):
    print("\n" + "=" * 70)
    print("  SECTION B — MONETIZATION BACKTEST (after fees)")
    print("=" * 70)
    if not trades:
        print("\n  [RESULT] No trades generated.")
        print("=" * 70)
        _print_verdict([], 0.0, 0.0, initial_equity, final_equity)
        return

    first_ms = min(t.entry_ms for t in trades); last_ms = max(t.exit_ms for t in trades)
    years = (last_ms - first_ms) / (1000 * 86400 * 365.25)
    cagr = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0
    mdd = _max_dd(eq_curve)
    wins = [t for t in trades if t.pnl > 0]
    pf_full = _pf(trades); wr = len(wins) / len(trades) * 100
    total_fees = sum(t.fees for t in trades)
    n_l = sum(1 for t in trades if t.direction == "LONG")
    n_s = sum(1 for t in trades if t.direction == "SHORT")
    n_stop = sum(1 for t in trades if t.exit_reason == "STOP")
    n_hold = sum(1 for t in trades if t.exit_reason == "HOLD")

    print(f"  Parameters    : strong={low_pct:.0f}/{high_pct:.0f}pct trailing-{_TRAIL_DAYS}d  "
          f"lookback={_LOOKBACK}h  hold={hold_bars}bars  stop={_STOP_MULT}×ATR({_ATR_PERIOD})")
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
    print(f"  Exit reasons  : HOLD={n_hold}  STOP={n_stop}")
    print(f"  Total fees    : ${total_fees:,.0f}")

    print(f"\n  {'─'*70}")
    print(f"  {'Symbol':<10} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*12}")
    for symbol in _SYMBOLS:
        st = [t for t in trades if t.symbol == symbol]
        if not st:
            print(f"  {symbol:<10} {0:>7} {'n/a':>7} {'n/a':>7} {'$0':>12}"); continue
        sw = [t for t in st if t.pnl > 0]
        spf = _pf(st); pf_str = f"{spf:.2f}" if not math.isinf(spf) else "inf"
        print(f"  {symbol:<10} {len(st):>7} {len(sw)/len(st)*100:>6.1f}% {pf_str:>7} ${sum(t.pnl for t in st):>+10,.0f}")

    print(f"\n  {'─'*70}")
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}  Status")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*12}  {'─'*10}")
    for yr in sorted({_year(t.entry_ms) for t in trades}):
        yt = [t for t in trades if _year(t.entry_ms) == yr]
        yw = [t for t in yt if t.pnl > 0]
        ypf = _pf(yt); pf_str = f"{ypf:>7.2f}" if not math.isinf(ypf) else f"{'inf':>7}"
        status = "+" if (math.isinf(ypf) or ypf >= 1.0) else "-"
        print(f"  {yr:<8} {len(yt):>7} {len(yw)/len(yt)*100:>6.1f}% {pf_str} ${sum(t.pnl for t in yt):>+10,.0f}  {status}")

    sb = sorted(trades, key=lambda t: t.pnl, reverse=True)
    print(f"\n  {'─'*70}")
    print("  Top 5 trades by PnL:")
    for t in sb[:5]:
        dt = datetime.fromtimestamp(t.entry_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.symbol:<8} {t.direction:<5} entry {dt}  exit={t.exit_reason}  ${t.pnl:>+,.0f}")
    print("  Bottom 5 trades by PnL:")
    for t in sb[-5:]:
        dt = datetime.fromtimestamp(t.entry_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.symbol:<8} {t.direction:<5} entry {dt}  exit={t.exit_reason}  ${t.pnl:>+,.0f}")

    oos = [t for t in trades if _year(t.entry_ms) >= _OOS_START_YEAR]
    ins = [t for t in trades if _year(t.entry_ms) < _OOS_START_YEAR]
    pf_oos = _pf(oos)
    print(f"\n  {'─'*70}")
    print(f"  OOS split (IS = <{_OOS_START_YEAR}, OOS = {_OOS_START_YEAR}+):")
    print(f"    IS : {len(ins)} trades, PF {_pf(ins):.2f}" if ins else "    IS : no trades")
    print(f"    OOS: {len(oos)} trades, PF {pf_oos:.2f}" if oos else "    OOS: no trades")
    print("=" * 70)
    _print_verdict(trades, pf_full, mdd, initial_equity, final_equity,
                   pf_oos=pf_oos, pf_full_ref=pf_full)


def _print_verdict(trades, pf_full, mdd, initial_equity, final_equity,
                   pf_oos=float("nan"), pf_full_ref=float("nan")):
    first_ms = min(t.entry_ms for t in trades) if trades else 0
    last_ms = max(t.exit_ms for t in trades) if trades else 0
    years = (last_ms - first_ms) / (1000 * 86400 * 365.25) if last_ms > first_ms else 0
    cagr = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 and initial_equity > 0 else 0.0

    kill = []
    n = len(trades)
    if pf_full < _KILL_PF: kill.append(f"Profit factor {pf_full:.2f} < kill threshold {_KILL_PF:.2f}")
    if mdd > _KILL_MAX_DD: kill.append(f"Max drawdown {mdd*100:.1f}% > kill threshold {_KILL_MAX_DD*100:.0f}%")
    if n < _KILL_TRADES: kill.append(f"Trade count {n} < minimum {_KILL_TRADES}")
    if not math.isnan(pf_oos) and not math.isnan(pf_full_ref):
        thr = _KILL_OOS_RATIO * pf_full_ref
        if pf_oos < thr:
            kill.append(f"OOS PF {pf_oos:.2f} < {_KILL_OOS_RATIO:.0%} × full-period PF {pf_full_ref:.2f} "
                        f"(threshold {thr:.2f}) — regime-specific")

    go = []
    if pf_full >= _GO_PF: go.append(f"PF {pf_full:.2f} >= {_GO_PF}")
    if cagr >= _GO_CAGR: go.append(f"CAGR {cagr*100:.1f}% >= {_GO_CAGR*100:.0f}%")
    if mdd <= _GO_DD: go.append(f"Max DD {mdd*100:.1f}% <= {_GO_DD*100:.0f}%")

    passes_core = pf_full >= _KILL_PF and mdd <= _KILL_MAX_DD and n >= _KILL_TRADES
    oos_weak = (not math.isnan(pf_oos) and not math.isnan(pf_full_ref)
                and pf_oos < _KILL_OOS_RATIO * pf_full_ref)

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Open-Interest Divergence/Confirmation Backtest (1H)")
    parser.add_argument("--data", default="data/candles")
    parser.add_argument("--oi-dir", default="data/oi")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--threshold-pct", type=float, default=20.0,
                        help="Percentile for the strong-move split (default 20 = 20/80)")
    parser.add_argument("--hold-hours", type=int, default=_HOLD_HOURS)
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data
    oi_dir = _REPO_ROOT / args.oi_dir
    low_pct = args.threshold_pct
    high_pct = 100.0 - args.threshold_pct
    hold_bars = args.hold_hours

    print(f"\n{'='*70}")
    print("  OPEN-INTEREST DIVERGENCE / CONFIRMATION BACKTEST — 1H timeframe")
    print(f"{'='*70}")
    print("  Hypothesis: a strong price move CONFIRMED by rising OI continues;")
    print("              a strong move on falling OI (unwind) stalls.")
    print("  Pre-committed rule:")
    print(f"    strong move = price {_LOOKBACK}h-change in top/bottom {high_pct:.0f}/{low_pct:.0f} pct of trailing {_TRAIL_DAYS}d")
    print(f"    confirm     = OI {_LOOKBACK}h-change > 0")
    print(f"    LONG  = strong UP + OI rising;  SHORT = strong DOWN + OI rising")
    print(f"    entry = next bar open (no lookahead)")
    print(f"    hold  = {hold_bars} bars OR {_STOP_MULT}×ATR({_ATR_PERIOD}) stop")
    print(f"    fee   = {_TAKER_FEE*100:.4f}% taker/side  risk = {_RISK_PCT*100:.1f}%  capital = ${args.capital:,.0f}")
    print(f"    symbols = {', '.join(_SYMBOLS)}")

    print(f"\nLoading 1H candles from {data_dir} and OI from {oi_dir} ...")
    candles_by_symbol = {}
    price_chg_by = {}
    oi_chg_by = {}
    atr_by_symbol = {}
    signals_by_symbol = {}

    for symbol in _SYMBOLS:
        candles = _load_1h(symbol, data_dir)
        oi_map = _load_oi(symbol, oi_dir)
        oi_aligned, matched = _align_oi(candles, oi_map)
        n_bars = len(candles["close"])
        print(f"  {symbol}: {n_bars:,} candles, {len(oi_map):,} OI rows, "
              f"{matched:,} bars matched OI ({matched/n_bars*100:.0f}%)")
        if matched < n_bars * 0.30:
            print(f"  [WARN] {symbol}: OI covers <30% of candles — results limited to overlap.")
        pc, oc = _compute_changes(candles, oi_aligned)
        candles_by_symbol[symbol] = candles
        price_chg_by[symbol] = pc
        oi_chg_by[symbol] = oc
        atr_by_symbol[symbol] = _wilder_atr_series(candles["high"], candles["low"], candles["close"], _ATR_PERIOD)
        signals_by_symbol[symbol] = _build_signals(symbol, candles, pc, oc, low_pct, high_pct)
        print(f"  {symbol}: {len(signals_by_symbol[symbol]):,} continuation signals")

    _section_a(candles_by_symbol, price_chg_by, oi_chg_by, low_pct, high_pct)
    _section_b(signals_by_symbol, candles_by_symbol, atr_by_symbol,
               hold_bars, args.capital, low_pct, high_pct)

    print(f"\n{'='*70}")
    print("  Backtest complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
