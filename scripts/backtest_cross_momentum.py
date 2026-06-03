#!/usr/bin/env python3
"""Cross-sectional momentum backtest — Mechanism #8 candidate.

Specification (pre-committed):
  Universe:  12 Bitget USDT-Perp symbols (same as HTF trend universe).
  Signal:    Every REBAL_DAYS calendar days, rank symbols by trailing
             LOOKBACK_DAYS return (close-to-close).
  Position:  Long top TOP_N symbols, short bottom TOP_N symbols.
             Remaining symbols are flat.
  Size:      Equal dollar weight across the 2×TOP_N active positions;
             total exposure = LEVERAGE × capital.
  Entry/Exit: At next bar open after ranking date (taker fee both sides).
  Fee:       0.06% taker on each side of every NEW or CLOSED position.
             If a symbol stays in the same side (long or short) across two
             consecutive rebalances, it incurs a fee only if the notional
             changes (treated as close-and-reopen here for simplicity).
  Stop:      No per-trade stop. Each symbol is held until next rebalance
             or until the position is closed by rank change.

Kill criteria:
  PF < 1.10, maxDD > 40%, n_trades < 60,
  OOS PF (2023+) < 0.85 × full-period PF.

Output: year-by-year table + per-symbol contribution + verdict.

Usage:
  python scripts/backtest_cross_momentum.py
  python scripts/backtest_cross_momentum.py --symbols BTCUSDT ETHUSDT ...
  python scripts/backtest_cross_momentum.py --lookback 20 --rebal 7 --top 3
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
# Constants (pre-committed)
# ---------------------------------------------------------------------------
_TAKER_FEE     = 0.0006   # 0.06% per side
_CAPITAL       = 100_000.0
_LEVERAGE      = 1.0       # gross exposure = 1× capital (conservative)
_LOOKBACK_DAYS = 20        # trailing return window for ranking
_REBAL_DAYS    = 7         # rebalance every N calendar days
_TOP_N         = 3         # long top-N, short bottom-N

_KILL_PF          = 1.10
_KILL_MAX_DD      = 0.40
_KILL_MIN_TRADES  = 60
_KILL_OOS_RATIO   = 0.85
_OOS_START_YEAR   = 2023

_DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class _Trade:
    symbol: str
    side: str        # "LONG" or "SHORT"
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    notional: float  # dollar amount at entry

    @property
    def pnl(self) -> float:
        if self.side == "LONG":
            raw = self.notional * (self.exit_price / self.entry_price - 1.0)
        else:
            raw = self.notional * (1.0 - self.exit_price / self.entry_price)
        fee = self.notional * _TAKER_FEE * 2
        return raw - fee

    @property
    def year(self) -> int:
        return datetime.fromtimestamp(self.entry_ts / 1000, tz=timezone.utc).year


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_daily(symbol: str, data_dir: Path) -> tuple[list[int], list[float]]:
    """Load daily close prices. Returns (timestamps_ms, closes)."""
    path = data_dir / f"{symbol}_1D.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing daily candle file: {path}")
    tbl = pq.read_table(path, columns=["open_time", "open", "close"])
    ot = tbl.column("open_time").to_pylist()
    closes = tbl.column("close").to_pylist()
    opens = tbl.column("open").to_pylist()
    rows = sorted(zip(ot, opens, closes))
    timestamps = [r[0] for r in rows]
    open_prices = [r[1] for r in rows]
    close_prices = [r[2] for r in rows]
    return timestamps, open_prices, close_prices


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def _run_backtest(
    symbols: list[str],
    data_dir: Path,
    lookback: int,
    rebal_days: int,
    top_n: int,
) -> list[_Trade]:
    """Run cross-sectional momentum backtest. Returns list of trades."""

    # Load all daily data
    data: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for sym in symbols:
        try:
            ts, opens, closes = _load_daily(sym, data_dir)
            data[sym] = (ts, opens, closes)
            print(f"  {sym}: {len(ts)} daily bars")
        except FileNotFoundError as e:
            print(f"  [WARN] {e} — skipping")

    if not data:
        print("[ERROR] No data loaded")
        return []

    # Build a unified timeline using the intersection of all symbols' dates
    # Use timestamps; treat each date as a set
    all_ts_sets = [set(ts) for ts, _, _ in data.values()]
    common_ts = sorted(set.intersection(*all_ts_sets))
    if not common_ts:
        print("[ERROR] No common timestamps across symbols")
        return []

    # Build indexed lookup: symbol → {ts: (open, close)}
    price_map: dict[str, dict[int, tuple[float, float]]] = {}
    for sym, (ts_list, opens, closes) in data.items():
        price_map[sym] = {t: (o, c) for t, o, c in zip(ts_list, opens, closes)}

    # Main loop: iterate through common timeline, rebalance every rebal_days
    trades: list[_Trade] = []

    # active_positions: {symbol: ("LONG"|"SHORT", entry_ts, entry_price, notional)}
    active: dict[str, tuple[str, int, float, float]] = {}

    last_rebal_ts: Optional[int] = None
    _DAY_MS = 86_400_000

    for i, ts in enumerate(common_ts):
        # Only rebalance on schedule
        if last_rebal_ts is not None:
            elapsed_days = (ts - last_rebal_ts) / _DAY_MS
            if elapsed_days < rebal_days:
                continue

        # Need lookback bars of close data to compute return
        if i < lookback:
            continue

        # Compute trailing lookback-day return for each symbol
        returns: dict[str, float] = {}
        for sym in list(data.keys()):
            # Get close lookback bars ago and current close
            idx_ts = common_ts[i - lookback]
            if idx_ts not in price_map[sym] or ts not in price_map[sym]:
                continue
            past_close = price_map[sym][idx_ts][1]
            curr_close = price_map[sym][ts][1]
            if past_close <= 0:
                continue
            returns[sym] = (curr_close - past_close) / past_close

        if len(returns) < 2 * top_n:
            continue  # not enough symbols to form a portfolio

        # Rank symbols
        ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        longs_target = {sym for sym, _ in ranked[:top_n]}
        shorts_target = {sym for sym, _ in ranked[-top_n:]}

        # Determine notional per position
        n_positions = 2 * top_n
        notional_per = (_CAPITAL * _LEVERAGE) / n_positions

        # Close positions no longer in target
        to_close = {
            sym: pos for sym, pos in active.items()
            if (pos[0] == "LONG" and sym not in longs_target)
            or (pos[0] == "SHORT" and sym not in shorts_target)
        }

        # Need next-bar open prices for entry/exit (use next common_ts if available)
        if i + 1 >= len(common_ts):
            break
        next_ts = common_ts[i + 1]

        for sym, (side, entry_ts, entry_price, notional) in to_close.items():
            if next_ts not in price_map[sym]:
                continue
            exit_price = price_map[sym][next_ts][0]  # open of next bar
            trades.append(_Trade(sym, side, entry_ts, entry_price, next_ts, exit_price, notional))
            del active[sym]

        # Open new positions
        for sym in longs_target:
            if sym in active and active[sym][0] == "LONG":
                continue  # already long, keep position
            # Close if wrong side
            if sym in active:
                side, entry_ts, entry_price, notional = active[sym]
                if next_ts in price_map[sym]:
                    exit_price = price_map[sym][next_ts][0]
                    trades.append(_Trade(sym, side, entry_ts, entry_price, next_ts, exit_price, notional))
                del active[sym]
            # Enter long
            if next_ts in price_map[sym]:
                entry_price = price_map[sym][next_ts][0]
                active[sym] = ("LONG", next_ts, entry_price, notional_per)

        for sym in shorts_target:
            if sym in active and active[sym][0] == "SHORT":
                continue
            if sym in active:
                side, entry_ts, entry_price, notional = active[sym]
                if next_ts in price_map[sym]:
                    exit_price = price_map[sym][next_ts][0]
                    trades.append(_Trade(sym, side, entry_ts, entry_price, next_ts, exit_price, notional))
                del active[sym]
            if next_ts in price_map[sym]:
                entry_price = price_map[sym][next_ts][0]
                active[sym] = ("SHORT", next_ts, entry_price, notional_per)

        last_rebal_ts = ts

    # Force-close all open positions at last available bar
    last_ts = common_ts[-1]
    for sym, (side, entry_ts, entry_price, notional) in list(active.items()):
        if last_ts in price_map[sym]:
            exit_price = price_map[sym][last_ts][1]  # use close
            trades.append(_Trade(sym, side, entry_ts, entry_price, last_ts, exit_price, notional))
    active.clear()

    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _calc_stats(trades: list[_Trade]) -> dict:
    if not trades:
        return {}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    equity = _CAPITAL
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    total_fees = sum(t.notional * _TAKER_FEE * 2 for t in trades)
    n = len(trades)
    win_rate = len(wins) / n if n else 0.0
    net_pnl = sum(pnls)
    final_equity = _CAPITAL + net_pnl

    # Approximate CAGR
    if trades:
        first_ts = min(t.entry_ts for t in trades)
        last_ts = max(t.exit_ts for t in trades)
        years = (last_ts - first_ts) / (365.25 * 24 * 3600 * 1000)
        cagr = (final_equity / _CAPITAL) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    else:
        cagr = 0.0

    # OOS split
    is_trades = [t for t in trades if t.year < _OOS_START_YEAR]
    oos_trades = [t for t in trades if t.year >= _OOS_START_YEAR]

    def _pf(ts: list[_Trade]) -> float:
        if not ts:
            return 0.0
        w = sum(t.pnl for t in ts if t.pnl > 0)
        l = abs(sum(t.pnl for t in ts if t.pnl <= 0))
        return w / l if l > 0 else float("inf")

    return {
        "n": n, "win_rate": win_rate, "pf": pf,
        "net_pnl": net_pnl, "final_equity": final_equity,
        "cagr": cagr, "max_dd": max_dd, "total_fees": total_fees,
        "gross_win": gross_win, "gross_loss": gross_loss,
        "is_pf": _pf(is_trades), "oos_pf": _pf(oos_trades),
        "is_n": len(is_trades), "oos_n": len(oos_trades),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _print_report(trades: list[_Trade], stats: dict, symbols: list[str]) -> None:
    W = 70
    sep = "=" * W

    print(sep)
    print("  CROSS-SECTIONAL MOMENTUM BACKTEST — daily timeframe")
    print(sep)
    print(f"  Hypothesis: coins with highest recent return continue to outperform")
    print(f"  Pre-committed rule:")
    print(f"    ranking  = trailing {_LOOKBACK_DAYS}-day return")
    print(f"    rebal    = every {_REBAL_DAYS} calendar days (taker fee on change)")
    print(f"    position = long top {_TOP_N}, short bottom {_TOP_N}")
    print(f"    exposure = {_LEVERAGE}× capital  fee = {_TAKER_FEE*100:.3f}% taker/side")
    print(f"    symbols  = {', '.join(symbols)}")
    print()

    if not trades:
        print("  No trades generated.")
        print(sep)
        return

    # Per-year breakdown
    from collections import defaultdict
    by_year: dict[int, list[_Trade]] = defaultdict(list)
    for t in trades:
        by_year[t.year].append(t)

    print(sep)
    print("  SECTION B — MONETIZATION BACKTEST (after fees)")
    print(sep)
    s = stats
    print(f"  Period        : {datetime.fromtimestamp(min(t.entry_ts for t in trades)/1000, tz=timezone.utc).strftime('%Y-%m-%d')} → {datetime.fromtimestamp(max(t.exit_ts for t in trades)/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${_CAPITAL:,.0f}")
    print(f"  Final equity  : ${s['final_equity']:,.0f}")
    print(f"  Net PnL       : ${s['net_pnl']:,.0f}  ({s['net_pnl']/_CAPITAL*100:.1f}%)")
    print(f"  CAGR          : {s['cagr']*100:.1f}%")
    print(f"  Max drawdown  : {s['max_dd']*100:.1f}%")
    print(f"  Profit factor : {s['pf']:.2f}")
    print(f"  Total trades  : {s['n']}  (L:{sum(1 for t in trades if t.side=='LONG')}  S:{sum(1 for t in trades if t.side=='SHORT')})")
    print(f"  Win rate      : {s['win_rate']*100:.1f}%")
    print(f"  Total fees    : ${s['total_fees']:,.0f}")
    print()

    # Per-symbol
    print(f"  {'Symbol':<12} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>13}")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*13}")
    for sym in symbols:
        st = [t for t in trades if t.symbol == sym]
        if not st:
            continue
        sp = _calc_stats(st)
        print(f"  {sym:<12} {sp['n']:>7} {sp['win_rate']*100:>6.1f}% {sp['pf']:>7.2f} ${sp['net_pnl']:>12,.0f}")
    print()

    # Per-year
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>13}  {'Status'}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*13}  {'─'*10}")
    for yr in sorted(by_year):
        yt = by_year[yr]
        yp = _calc_stats(yt)
        status = "+" if yp['net_pnl'] > 0 else "-"
        print(f"  {yr:<8} {yp['n']:>7} {yp['win_rate']*100:>6.1f}% {yp['pf']:>7.2f} ${yp['net_pnl']:>12,.0f}  {status}")
    print()

    print(f"  OOS split (IS = <{_OOS_START_YEAR}, OOS = {_OOS_START_YEAR}+):")
    print(f"    IS : {s['is_n']} trades, PF {s['is_pf']:.2f}")
    print(f"    OOS: {s['oos_n']} trades, PF {s['oos_pf']:.2f}")
    print(sep)
    print()

    # Verdict
    print(sep)
    print("  VERDICT")
    print(sep)
    kills = []
    if s['pf'] < _KILL_PF:
        kills.append(f"Profit factor {s['pf']:.2f} < kill threshold {_KILL_PF}")
    if s['max_dd'] > _KILL_MAX_DD:
        kills.append(f"Max drawdown {s['max_dd']*100:.1f}% > kill threshold {_KILL_MAX_DD*100:.0f}%")
    if s['n'] < _KILL_MIN_TRADES:
        kills.append(f"Trade count {s['n']} < minimum {_KILL_MIN_TRADES}")
    if s['is_n'] > 0 and s['oos_n'] > 0 and s['oos_pf'] < _KILL_OOS_RATIO * s['pf']:
        kills.append(f"OOS PF {s['oos_pf']:.2f} < {_KILL_OOS_RATIO}×full PF {s['pf']:.2f}")

    if kills:
        print()
        print("  KILL -- Strategy fails one or more kill criteria:")
        print()
        for k in kills:
            print(f"     * {k}")
        print()
        print("  Action: do NOT deploy. Move to next hypothesis.")
    else:
        wealth = (s['pf'] - 1.0) * 20 + s['cagr'] * 40 - s['max_dd'] * 30
        print()
        print(f"  PROCEED -- All kill criteria passed.")
        print()
        print(f"    PF {s['pf']:.2f}  CAGR {s['cagr']*100:.1f}%  maxDD {s['max_dd']*100:.1f}%")
        print(f"    OOS PF {s['oos_pf']:.2f} (IS: {s['is_pf']:.2f})")
        print(f"    Wealth score: {wealth:.0f}/100")
    print(sep)
    print()
    print(sep)
    print("  Backtest complete.")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--lookback", type=int, default=_LOOKBACK_DAYS)
    parser.add_argument("--rebal",    type=int, default=_REBAL_DAYS)
    parser.add_argument("--top",      type=int, default=_TOP_N)
    parser.add_argument("--data",     default="data/candles")
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data
    print(f"Loading daily candles from {data_dir}")
    print()

    trades = _run_backtest(args.symbols, data_dir, args.lookback, args.rebal, args.top)
    stats = _calc_stats(trades)
    _print_report(trades, stats, args.symbols)


if __name__ == "__main__":
    main()
