#!/usr/bin/env python3
"""BTC→ETH Lead-Lag Backtest — 1H timeframe.

Mechanism hypothesis: a strong BTC 1H close return predicts an ETH 1H
return in the same direction on the FOLLOWING bar.

Two-section output
------------------
SECTION A  Mechanism Validity — does the correlation exist (pre-fee)?
SECTION B  Monetization      — can it be extracted after fees?

Parameters (all pre-committed):
  threshold : abs(BTC 1H return) > 0.5%  to generate a signal
  hold      : 4 bars from entry
  stop_mult : 1.5 × ATR(14) on ETH 1H from entry price

Kill criteria:
  PF < 1.10
  n_trades < 100
  max DD > 40%
  OOS PF (2023+) < 0.85 × full-period PF

Usage:
  python scripts/backtest_lead_lag.py
  python scripts/backtest_lead_lag.py --capital 50000 --threshold 0.75
  python scripts/backtest_lead_lag.py --data data/candles
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
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
_TAKER_FEE    = 0.0006   # 0.06% per side — Bitget USDT-Perp taker
_RISK_PCT     = 0.01     # 1% account equity per trade
_ATR_PERIOD   = 14
_HOLD_BARS    = 4
_STOP_MULT    = 1.5

# Kill thresholds
_KILL_PF       = 1.10
_KILL_TRADES   = 100
_KILL_MAX_DD   = 0.40
_KILL_OOS_RATIO= 0.85    # OOS PF must be >= 85% of full-period PF
_OOS_START_YEAR= 2023

# Go thresholds
_GO_PF   = 1.30
_GO_CAGR = 0.20
_GO_DD   = 0.40

# Signal strength buckets (absolute BTC return)
_BUCKETS = [(0.005, 0.01), (0.01, 0.02), (0.02, float("inf"))]
_BUCKET_LABELS = ["0.5-1%", "1-2%", ">2%"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_1h(symbol: str, data_dir: Path) -> dict:
    """Return dict of lists keyed by column name, sorted ascending by open_time."""
    path = data_dir / f"{symbol}_1H.parquet"
    if not path.exists():
        print(f"[ERROR] Missing: {path}")
        sys.exit(1)
    tbl = pq.read_table(path)
    d   = tbl.to_pydict()
    # Sort ascending
    idx = sorted(range(len(d["open_time"])), key=lambda i: d["open_time"][i])
    return {col: [d[col][i] for i in idx] for col in d}


def _align(btc: dict, eth: dict) -> tuple[list, list, list]:
    """Align BTC and ETH bars by open_time.  Returns (timestamps, btc_rows, eth_rows)."""
    btc_map = {t: i for i, t in enumerate(btc["open_time"])}
    eth_map = {t: i for i, t in enumerate(eth["open_time"])}
    common  = sorted(set(btc_map.keys()) & set(eth_map.keys()))

    def row(d: dict, i: int) -> dict:
        return {col: d[col][i] for col in d}

    ts_list  = common
    btc_rows = [row(btc, btc_map[t]) for t in common]
    eth_rows = [row(eth, eth_map[t]) for t in common]
    return ts_list, btc_rows, eth_rows


# ---------------------------------------------------------------------------
# ATR — Wilder smoothing (same as engine1)
# ---------------------------------------------------------------------------

def _wilder_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> Optional[float]:
    n = len(closes)
    if n < period + 1:
        return None
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


# ---------------------------------------------------------------------------
# SECTION A — Mechanism Validity
# ---------------------------------------------------------------------------

@dataclass
class _BucketStats:
    label:      str
    n_signals:  int = 0
    n_correct:  int = 0          # ETH next-bar same direction
    sum_eth_ret: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.n_correct / self.n_signals if self.n_signals else float("nan")

    @property
    def mean_eth_ret(self) -> float:
        return self.sum_eth_ret / self.n_signals if self.n_signals else float("nan")


def _section_a(
    ts_list: list,
    btc_rows: list[dict],
    eth_rows: list[dict],
    threshold: float,
) -> None:
    """Print SECTION A: mechanism validity analysis."""

    n = len(ts_list)

    # Aggregate stats
    up_rets:   list[float] = []   # ETH next-bar ret when BTC up signal
    dn_rets:   list[float] = []   # ETH next-bar ret when BTC down signal
    no_rets:   list[float] = []   # ETH next-bar ret when no signal

    bucket_up = [_BucketStats(lbl) for lbl in _BUCKET_LABELS]
    bucket_dn = [_BucketStats(lbl) for lbl in _BUCKET_LABELS]

    # Forward returns: ETH at +1H, +4H, +8H after strong BTC move
    fwd_eth_1h: list[float] = []
    fwd_eth_4h: list[float] = []
    fwd_eth_8h: list[float] = []

    for i in range(n - 1):
        btc_close_prev = btc_rows[i - 1]["close"] if i > 0 else None
        btc_close_cur  = btc_rows[i]["close"]
        if btc_close_prev is None or btc_close_prev <= 0:
            continue

        btc_ret = (btc_close_cur - btc_close_prev) / btc_close_prev

        # ETH next-bar return (bar i+1)
        eth_open_next  = eth_rows[i + 1]["open"]
        eth_close_next = eth_rows[i + 1]["close"]
        if eth_open_next <= 0:
            continue
        eth_next_ret = (eth_close_next - eth_open_next) / eth_open_next

        abs_ret = abs(btc_ret)

        if abs_ret > threshold:
            is_up = btc_ret > 0
            is_correct = (is_up and eth_next_ret > 0) or (not is_up and eth_next_ret < 0)

            if is_up:
                up_rets.append(eth_next_ret)
            else:
                dn_rets.append(eth_next_ret)

            # Bucket
            for bi, (lo, hi) in enumerate(_BUCKETS):
                if lo <= abs_ret < hi:
                    b = bucket_up[bi] if is_up else bucket_dn[bi]
                    b.n_signals += 1
                    if is_correct:
                        b.n_correct += 1
                    b.sum_eth_ret += eth_next_ret
                    break

            # Forward ETH returns after strong BTC move
            if abs_ret > threshold:
                eth_entry = eth_rows[i + 1]["open"]
                if eth_entry > 0:
                    fwd_eth_1h.append((eth_rows[i + 1]["close"] - eth_entry) / eth_entry)
                    if i + 4 < n:
                        fwd_eth_4h.append((eth_rows[i + 4]["close"] - eth_entry) / eth_entry)
                    if i + 8 < n:
                        fwd_eth_8h.append((eth_rows[i + 8]["close"] - eth_entry) / eth_entry)
        else:
            no_rets.append(eth_next_ret)

    total_signals = len(up_rets) + len(dn_rets)
    all_signals   = up_rets + dn_rets

    def _mean(lst: list[float]) -> str:
        if not lst:
            return "  n/a"
        return f"{sum(lst)/len(lst)*100:+.4f}%"

    def _hr(correct: int, total: int) -> str:
        if total == 0:
            return "  n/a"
        return f"{correct/total*100:.1f}%"

    n_correct_up = sum(1 for r in up_rets if r > 0)
    n_correct_dn = sum(1 for r in dn_rets if r < 0)
    n_correct_all= n_correct_up + n_correct_dn

    print("\n" + "=" * 70)
    print("  SECTION A — MECHANISM VALIDITY (pre-fee, bar-level analysis)")
    print("=" * 70)
    print(f"  BTC signal threshold : abs(return) > {threshold*100:.2f}%")
    print(f"  Aligned bar count    : {n:,}")
    print(f"  Total BTC signals    : {total_signals:,}  (up: {len(up_rets)}, down: {len(dn_rets)})")
    print(f"  No-signal bars       : {len(no_rets):,}")
    print()
    print(f"  {'Condition':<28} {'N':>6} {'Hit-Rate':>9} {'Mean ETH next-bar ret':>22}")
    print(f"  {'─'*28} {'─'*6} {'─'*9} {'─'*22}")
    print(f"  {'BTC up signal':<28} {len(up_rets):>6} {_hr(n_correct_up, len(up_rets)):>9} {_mean(up_rets):>22}")
    print(f"  {'BTC down signal':<28} {len(dn_rets):>6} {_hr(n_correct_dn, len(dn_rets)):>9} {_mean(dn_rets):>22}")
    print(f"  {'All signals combined':<28} {total_signals:>6} {_hr(n_correct_all, total_signals):>9} {_mean(all_signals):>22}")
    print(f"  {'No signal (baseline)':<28} {len(no_rets):>6} {'─':>9} {_mean(no_rets):>22}")

    print(f"\n  Signal strength breakdown:")
    print(f"  {'Bucket':<10} {'Dir':<5} {'N':>6} {'Hit-Rate':>9} {'Mean ETH ret':>14}")
    print(f"  {'─'*10} {'─'*5} {'─'*6} {'─'*9} {'─'*14}")
    for bi, lbl in enumerate(_BUCKET_LABELS):
        bu = bucket_up[bi]
        bd = bucket_dn[bi]
        for b, direction in [(bu, "UP"), (bd, "DN")]:
            hr  = f"{b.hit_rate*100:.1f}%" if b.n_signals else "n/a"
            mr  = f"{b.mean_eth_ret*100:+.4f}%" if b.n_signals else "n/a"
            print(f"  {lbl:<10} {direction:<5} {b.n_signals:>6} {hr:>9} {mr:>14}")

    print(f"\n  Average ETH move AFTER strong BTC signal (entry = next-bar open):")
    def _fwd(lst: list[float], label: str) -> None:
        if lst:
            print(f"    {label:<8}  n={len(lst):>5}  mean={sum(lst)/len(lst)*100:+.4f}%")
        else:
            print(f"    {label:<8}  n/a")
    _fwd(fwd_eth_1h, "+1H")
    _fwd(fwd_eth_4h, "+4H")
    _fwd(fwd_eth_8h, "+8H")

    # Interpretation
    edge_exists = (
        total_signals >= 50 and
        n_correct_all / total_signals > 0.52 and
        (sum(up_rets) / len(up_rets) > 0 if up_rets else False) and
        (sum(dn_rets) / len(dn_rets) < 0 if dn_rets else False)
    )
    print(f"\n  Mechanism assessment: {'CORRELATION PRESENT' if edge_exists else 'WEAK / NO CORRELATION'}")
    if not edge_exists and total_signals >= 50:
        overall_hr = n_correct_all / total_signals if total_signals else 0
        print(f"    Hit rate {overall_hr*100:.1f}% — near-random; monetization unlikely to pass.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# SECTION B — Monetization backtest
# ---------------------------------------------------------------------------

@dataclass
class _Trade:
    direction:    str           # "LONG" or "SHORT"
    entry_price:  float
    exit_price:   float
    entry_ms:     int
    exit_ms:      int
    size:         float
    pnl:          float         # net of all fees
    fees:         float
    stop_price:   float
    exit_reason:  str           # "HOLD" or "STOP"
    equity_at_entry: float


def _year(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year


def _section_b(
    ts_list:  list,
    btc_rows: list[dict],
    eth_rows: list[dict],
    threshold: float,
    initial_equity: float,
) -> None:
    """Run bar-by-bar backtest and print SECTION B."""

    n = len(ts_list)

    # Rolling ATR window for ETH
    eth_highs:  list[float] = []
    eth_lows:   list[float] = []
    eth_closes: list[float] = []

    equity      = initial_equity
    eq_curve: list[tuple[int, float]] = [(ts_list[0], equity)]

    trades: list[_Trade] = []

    # Open position state
    pos_direction: Optional[str]   = None
    pos_entry_price: float         = 0.0
    pos_entry_ms: int              = 0
    pos_size: float                = 0.0
    pos_stop: float                = 0.0
    pos_entry_fee: float           = 0.0
    pos_bars_held: int             = 0
    pos_equity_at_entry: float     = 0.0

    for i in range(n):
        ts          = ts_list[i]
        eth         = eth_rows[i]
        btc_prev_cl = btc_rows[i - 1]["close"] if i > 0 else None
        btc_cur_cl  = btc_rows[i]["close"]

        eth_open  = eth["open"]
        eth_high  = eth["high"]
        eth_low   = eth["low"]
        eth_close = eth["close"]

        # ----------------------------------------------------------------
        # 1. If we have an open position, process this bar
        # ----------------------------------------------------------------
        if pos_direction is not None:
            pos_bars_held += 1
            stop_hit = False

            # Check stop (intrabar — use high/low)
            if pos_direction == "LONG"  and eth_low  <= pos_stop:
                stop_hit = True
                exit_price = pos_stop
            elif pos_direction == "SHORT" and eth_high >= pos_stop:
                stop_hit = True
                exit_price = pos_stop
            else:
                exit_price = None

            should_exit = stop_hit or (pos_bars_held >= _HOLD_BARS)

            if should_exit:
                if exit_price is None:
                    exit_price = eth_close   # exit at close on hold expiry

                exit_reason = "STOP" if stop_hit else "HOLD"
                exit_fee    = exit_price * pos_size * _TAKER_FEE
                if pos_direction == "LONG":
                    raw_pnl = (exit_price - pos_entry_price) * pos_size
                else:
                    raw_pnl = (pos_entry_price - exit_price) * pos_size
                net_pnl = raw_pnl - exit_fee
                equity += raw_pnl - exit_fee

                trades.append(_Trade(
                    direction       = pos_direction,
                    entry_price     = pos_entry_price,
                    exit_price      = exit_price,
                    entry_ms        = pos_entry_ms,
                    exit_ms         = ts,
                    size            = pos_size,
                    pnl             = net_pnl,
                    fees            = pos_entry_fee + exit_fee,
                    stop_price      = pos_stop,
                    exit_reason     = exit_reason,
                    equity_at_entry = pos_equity_at_entry,
                ))
                pos_direction = None

        # ----------------------------------------------------------------
        # 2. Update ATR window with this bar
        # ----------------------------------------------------------------
        eth_highs.append(eth_high)
        eth_lows.append(eth_low)
        eth_closes.append(eth_close)

        # ----------------------------------------------------------------
        # 3. Check for signal on this BTC bar (enter next bar open)
        # ----------------------------------------------------------------
        if btc_prev_cl is None or btc_prev_cl <= 0:
            eq_curve.append((ts, equity))
            continue

        btc_ret = (btc_cur_cl - btc_prev_cl) / btc_prev_cl
        abs_ret = abs(btc_ret)

        if abs_ret <= threshold:
            eq_curve.append((ts, equity))
            continue

        signal = "LONG" if btc_ret > 0 else "SHORT"

        # Skip if already in a position (same or opposite — only one at a time)
        if pos_direction is not None:
            eq_curve.append((ts, equity))
            continue

        # Need next bar to exist for entry
        if i + 1 >= n:
            eq_curve.append((ts, equity))
            continue

        # ATR at signal bar
        atr = _wilder_atr(eth_highs, eth_lows, eth_closes, _ATR_PERIOD)
        if atr is None or atr <= 0:
            eq_curve.append((ts, equity))
            continue

        # Entry at next bar open
        entry_price = eth_rows[i + 1]["open"]
        if entry_price <= 0:
            eq_curve.append((ts, equity))
            continue

        stop_dist = _STOP_MULT * atr
        if signal == "LONG":
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist

        # Size: 1% risk
        risk_dollars = equity * _RISK_PCT
        size = risk_dollars / stop_dist
        if size <= 0:
            eq_curve.append((ts, equity))
            continue

        entry_fee = entry_price * size * _TAKER_FEE
        equity   -= entry_fee

        pos_direction       = signal
        pos_entry_price     = entry_price
        pos_entry_ms        = ts_list[i + 1]
        pos_size            = size
        pos_stop            = stop_price
        pos_entry_fee       = entry_fee
        pos_bars_held       = 0
        pos_equity_at_entry = equity

        eq_curve.append((ts, equity))

    # Force-close any open position at last bar close
    if pos_direction is not None:
        last_eth   = eth_rows[-1]
        exit_price = last_eth["close"]
        exit_fee   = exit_price * pos_size * _TAKER_FEE
        if pos_direction == "LONG":
            raw_pnl = (exit_price - pos_entry_price) * pos_size
        else:
            raw_pnl = (pos_entry_price - exit_price) * pos_size
        net_pnl = raw_pnl - exit_fee
        equity += raw_pnl - exit_fee
        trades.append(_Trade(
            direction       = pos_direction,
            entry_price     = pos_entry_price,
            exit_price      = exit_price,
            entry_ms        = pos_entry_ms,
            exit_ms         = ts_list[-1],
            size            = pos_size,
            pnl             = net_pnl,
            fees            = pos_entry_fee + exit_fee,
            stop_price      = pos_stop,
            exit_reason     = "HOLD",
            equity_at_entry = pos_equity_at_entry,
        ))
        pos_direction = None

    eq_curve.append((ts_list[-1], equity))
    _print_section_b(trades, eq_curve, initial_equity, equity, threshold)


def _max_dd(eq_curve: list[tuple[int, float]]) -> float:
    peak   = eq_curve[0][1] if eq_curve else 0.0
    max_dd = 0.0
    for _, eq in eq_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _pf(trade_list: list[_Trade]) -> float:
    gp = sum(t.pnl for t in trade_list if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trade_list if t.pnl <= 0))
    return gp / gl if gl > 0 else float("inf")


def _print_section_b(
    trades:         list[_Trade],
    eq_curve:       list[tuple[int, float]],
    initial_equity: float,
    final_equity:   float,
    threshold:      float,
) -> None:
    print("\n" + "=" * 70)
    print("  SECTION B — MONETIZATION BACKTEST (after fees)")
    print("=" * 70)

    if not trades:
        print("\n  [RESULT] No trades generated.")
        print("=" * 70)
        _print_verdict([], 0.0, 0.0, initial_equity, final_equity)
        return

    first_ms = min(t.entry_ms for t in trades)
    last_ms  = max(t.exit_ms  for t in trades)
    years    = (last_ms - first_ms) / (1000 * 86400 * 365.25)
    cagr     = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0
    max_drawdown = _max_dd(eq_curve)

    wins      = [t for t in trades if t.pnl > 0]
    losses    = [t for t in trades if t.pnl <= 0]
    pf_full   = _pf(trades)
    wr        = len(wins) / len(trades) * 100
    total_fees= sum(t.fees for t in trades)
    n_longs   = sum(1 for t in trades if t.direction == "LONG")
    n_shorts  = sum(1 for t in trades if t.direction == "SHORT")
    n_stops   = sum(1 for t in trades if t.exit_reason == "STOP")
    n_holds   = sum(1 for t in trades if t.exit_reason == "HOLD")

    print(f"  Parameters    : threshold={threshold*100:.2f}%  hold={_HOLD_BARS}bars  stop={_STOP_MULT}×ATR({_ATR_PERIOD})")
    print(f"  Period        : {datetime.fromtimestamp(first_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}"
          f" → {datetime.fromtimestamp(last_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${initial_equity:,.0f}")
    print(f"  Final equity  : ${final_equity:,.0f}")
    print(f"  Net PnL       : ${final_equity - initial_equity:+,.0f}  ({(final_equity-initial_equity)/initial_equity*100:+.1f}%)")
    print(f"  CAGR          : {cagr*100:.1f}%")
    print(f"  Max drawdown  : {max_drawdown*100:.1f}%")
    print(f"  Profit factor : {pf_full:.2f}")
    print(f"  Total trades  : {len(trades)}  (L:{n_longs}  S:{n_shorts})")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Exit reasons  : HOLD={n_holds}  STOP={n_stops}")
    print(f"  Total fees    : ${total_fees:,.0f}")

    # Year-by-year breakdown
    years_seen = sorted({_year(t.entry_ms) for t in trades})
    print(f"\n  {'─'*70}")
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>12}  {'Status'}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*12}  {'─'*10}")
    for yr in years_seen:
        yr_trades = [t for t in trades if _year(t.entry_ms) == yr]
        if not yr_trades:
            continue
        yr_wins  = [t for t in yr_trades if t.pnl > 0]
        yr_gp    = sum(t.pnl for t in yr_wins)
        yr_gl    = abs(sum(t.pnl for t in yr_trades if t.pnl <= 0))
        yr_pf    = yr_gp / yr_gl if yr_gl > 0 else float("inf")
        yr_wr    = len(yr_wins) / len(yr_trades) * 100
        yr_pnl   = sum(t.pnl for t in yr_trades)
        status   = "+" if yr_pf >= 1.0 else "-"
        print(f"  {yr:<8} {len(yr_trades):>7} {yr_wr:>6.1f}% {yr_pf:>7.2f} ${yr_pnl:>+10,.0f}  {status}")

    # Top 5 / bottom 5 by PnL
    sorted_by_pnl = sorted(trades, key=lambda t: t.pnl, reverse=True)
    print(f"\n  {'─'*70}")
    print("  Top 5 trades by PnL:")
    for t in sorted_by_pnl[:5]:
        dt = datetime.fromtimestamp(t.entry_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.direction:<5} entry {dt}  exit={t.exit_reason}  ${t.pnl:>+,.0f}")
    print("  Bottom 5 trades by PnL:")
    for t in sorted_by_pnl[-5:]:
        dt = datetime.fromtimestamp(t.entry_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"    {t.direction:<5} entry {dt}  exit={t.exit_reason}  ${t.pnl:>+,.0f}")

    # OOS split
    oos_trades = [t for t in trades if _year(t.entry_ms) >= _OOS_START_YEAR]
    is_trades  = [t for t in trades if _year(t.entry_ms) <  _OOS_START_YEAR]
    pf_oos     = _pf(oos_trades)
    pf_is      = _pf(is_trades)

    print(f"\n  {'─'*70}")
    print(f"  OOS split (IS = <{_OOS_START_YEAR}, OOS = {_OOS_START_YEAR}+):")
    if is_trades:
        print(f"    IS  ({min(_year(t.entry_ms) for t in is_trades)}"
              f"–{max(_year(t.entry_ms) for t in is_trades)}): "
              f"{len(is_trades)} trades, PF {pf_is:.2f}")
    else:
        print(f"    IS : no trades")
    if oos_trades:
        print(f"    OOS ({min(_year(t.entry_ms) for t in oos_trades)}"
              f"–{max(_year(t.entry_ms) for t in oos_trades)}): "
              f"{len(oos_trades)} trades, PF {pf_oos:.2f}")
    else:
        print(f"    OOS: no trades")

    print("=" * 70)
    _print_verdict(trades, pf_full, max_drawdown, initial_equity, final_equity,
                   pf_oos=pf_oos, pf_full_ref=pf_full)


def _print_verdict(
    trades:         list[_Trade],
    pf_full:        float,
    max_drawdown:   float,
    initial_equity: float,
    final_equity:   float,
    pf_oos:         float = float("nan"),
    pf_full_ref:    float = float("nan"),
) -> None:
    first_ms = min(t.entry_ms for t in trades) if trades else 0
    last_ms  = max(t.exit_ms  for t in trades) if trades else 0
    years    = (last_ms - first_ms) / (1000 * 86400 * 365.25) if last_ms > first_ms else 0
    cagr     = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 and initial_equity > 0 else 0.0

    kill_reasons: list[str] = []
    go_flags:     list[str] = []

    n_trades = len(trades)

    if n_trades < _KILL_TRADES:
        kill_reasons.append(f"Trade count {n_trades} < minimum {_KILL_TRADES}")
    if pf_full < _KILL_PF:
        kill_reasons.append(f"Profit factor {pf_full:.2f} < kill threshold {_KILL_PF:.2f}")
    if max_drawdown > _KILL_MAX_DD:
        kill_reasons.append(f"Max drawdown {max_drawdown*100:.1f}% > kill threshold {_KILL_MAX_DD*100:.0f}%")
    if not math.isnan(pf_oos) and not math.isnan(pf_full_ref):
        oos_threshold = _KILL_OOS_RATIO * pf_full_ref
        if pf_oos < oos_threshold:
            kill_reasons.append(
                f"OOS PF {pf_oos:.2f} < {_KILL_OOS_RATIO:.0%} × full-period PF {pf_full_ref:.2f} "
                f"(threshold {oos_threshold:.2f})"
            )

    if pf_full >= _GO_PF:
        go_flags.append(f"PF {pf_full:.2f} >= {_GO_PF}")
    if cagr >= _GO_CAGR:
        go_flags.append(f"CAGR {cagr*100:.1f}% >= {_GO_CAGR*100:.0f}%")
    if max_drawdown <= _GO_DD:
        go_flags.append(f"Max DD {max_drawdown*100:.1f}% <= {_GO_DD*100:.0f}%")

    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    if kill_reasons:
        print("\n  KILL -- Strategy fails one or more kill criteria:\n")
        for r in kill_reasons:
            print(f"     * {r}")
        print("\n  Action: do NOT deploy. Move to next hypothesis.")
    elif len(go_flags) == 3:
        print("\n  GO -- All go criteria met:\n")
        for f in go_flags:
            print(f"     * {f}")
        print("\n  Action: proceed to paper deployment.")
    else:
        passed = len(go_flags)
        print(f"\n  MARGINAL -- {passed}/3 go criteria met. Deploy on paper with caution.")
        print("  Passed  :", ", ".join(go_flags) if go_flags else "none")
        unmet = []
        if pf_full < _GO_PF:
            unmet.append(f"PF {pf_full:.2f} < {_GO_PF}")
        if cagr < _GO_CAGR:
            unmet.append(f"CAGR {cagr*100:.1f}% < {_GO_CAGR*100:.0f}%")
        if max_drawdown > _GO_DD:
            unmet.append(f"Max DD {max_drawdown*100:.1f}% > {_GO_DD*100:.0f}%")
        print("  Unmet   :", ", ".join(unmet))
        print("\n  Action: run paper trading before adding capital.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BTC→ETH Lead-Lag Backtest (1H)")
    parser.add_argument("--data",      default="data/candles",
                        help="Directory containing BTCUSDT_1H.parquet and ETHUSDT_1H.parquet")
    parser.add_argument("--capital",   type=float, default=100_000.0,
                        help="Initial account equity in USDT (default 100000)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="BTC signal threshold in percent (default 0.5)")
    args = parser.parse_args()

    data_dir  = _REPO_ROOT / args.data
    threshold = args.threshold / 100.0   # convert percent to decimal

    print(f"\n{'='*70}")
    print("  BTC→ETH LEAD-LAG BACKTEST — 1H timeframe")
    print(f"{'='*70}")
    print(f"  Parameters used:")
    print(f"    threshold = {args.threshold:.2f}%  (abs BTC 1H return)")
    print(f"    hold      = {_HOLD_BARS} bars")
    print(f"    stop_mult = {_STOP_MULT}x ATR({_ATR_PERIOD})")
    print(f"    fee       = {_TAKER_FEE*100:.4f}% taker (entry + exit)")
    print(f"    risk_pct  = {_RISK_PCT*100:.1f}% per trade")
    print(f"    capital   = ${args.capital:,.0f}")

    print(f"\nLoading data from {data_dir} ...")
    btc = _load_1h("BTCUSDT", data_dir)
    eth = _load_1h("ETHUSDT", data_dir)

    print(f"  BTC: {len(btc['open_time']):,} hourly bars")
    print(f"  ETH: {len(eth['open_time']):,} hourly bars")

    ts_list, btc_rows, eth_rows = _align(btc, eth)
    print(f"  Aligned: {len(ts_list):,} common bars")

    if len(ts_list) < 200:
        print("[ERROR] Too few aligned bars. Check data files.")
        sys.exit(1)

    first_dt = datetime.fromtimestamp(ts_list[0]  / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    last_dt  = datetime.fromtimestamp(ts_list[-1] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  Date range: {first_dt} → {last_dt}")

    # SECTION A
    _section_a(ts_list, btc_rows, eth_rows, threshold)

    # SECTION B
    _section_b(ts_list, btc_rows, eth_rows, threshold, args.capital)


if __name__ == "__main__":
    main()
