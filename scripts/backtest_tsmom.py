#!/usr/bin/env python3
"""Section B backtest for mechanism #11: Time-series momentum (TSMOM).

Hypothesis: A coin trending strongly for 6 months continues to trend.
  Long  coins with 126-day return > +5%.
  Short coins with 126-day return < -5%.
  No trade when return is near-zero (ranging).
  Rebalance weekly.

Differences from mechanism #8 (cross-sectional, KILLED):
  - Absolute threshold (+/-5%), not forced relative ranking
  - 126-day lookback (6 months), not 20-day (avoids noise)
  - Number of positions varies 0-12 (no forced shorts in bull market)

Pre-committed parameters (locked before first run):
  instruments : 12-coin universe (full set)
  lookback    : 126 calendar days
  threshold   : ±5% absolute 126d return (flat if between -5% and +5%)
  rebal_days  : 7
  position    : capital / 12 equal-weight (fixed notional per active position)
  fee         : 0.06% taker/side (only on changed positions)

Usage:
    python scripts/backtest_tsmom.py
    python scripts/backtest_tsmom.py --from 2022-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow required: pip install pyarrow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Pre-committed constants
# ---------------------------------------------------------------------------
_DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
_INITIAL_EQUITY  = 100_000.0
_LOOKBACK_DAYS   = 126          # ≈ 6 months
_REBAL_DAYS      = 7
_THRESHOLD_PCT   = 0.05         # |return| must exceed 5% to take a position
_LEVERAGE        = 1.0
_TAKER_FEE       = 0.0006       # 0.06% per side
_DAY_MS          = 86_400_000
_CANDLE_DIR      = _REPO_ROOT / "data" / "candles"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_daily(symbols: list[str], from_ts: int, to_ts: int) -> dict[str, list[tuple[int, float]]]:
    result: dict[str, list[tuple[int, float]]] = {}
    for sym in symbols:
        path = _CANDLE_DIR / f"{sym}_1D.parquet"
        if not path.exists():
            print(f"  [WARN] No daily data for {sym}")
            result[sym] = []
            continue
        tbl = pq.read_table(path, columns=["open_time", "close"])
        rows = sorted(zip(
            tbl.column("open_time").to_pylist(),
            tbl.column("close").to_pylist(),
        ))
        result[sym] = [(ts, float(c)) for ts, c in rows if from_ts <= ts <= to_ts]
        print(f"  {sym}: {len(result[sym]):,} daily bars")
    return result


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------
def run_backtest(
    symbols: list[str],
    capital: float,
    from_ts: int,
    to_ts: int,
    lookback: int,
    rebal_days: int,
    threshold: float,
) -> None:
    history = _load_daily(symbols, from_ts, to_ts)
    print()

    # Build sorted event list
    events: list[tuple[int, str, float]] = []
    for sym, bars in history.items():
        for ts, c in bars:
            events.append((ts, sym, c))
    events.sort()

    # Per-symbol price history (ring buffer)
    price_hist: dict[str, list[tuple[int, float]]] = {s: [] for s in symbols}

    # Portfolio state
    positions: dict[str, str] = {}   # {symbol: "LONG" | "SHORT"}
    equity = capital
    last_rebal_ts: int | None = None

    # Tracking
    all_trades: list[dict] = []
    equity_curve: list[tuple[int, float]] = [(from_ts, capital)]

    def _notional() -> float:
        return (capital * _LEVERAGE) / len(symbols)

    def _close_pos(sym: str, side: str, close_price: float, ts: int, reason: str) -> float:
        """Returns net PnL."""
        notional = _notional()
        fee = _TAKER_FEE * notional * 2  # round-trip (already paid entry, now paying exit)
        # fee already subtracted on entry, so just pay exit here
        fee = _TAKER_FEE * notional
        # Find entry price from last matching open trade
        entry_price = None
        for t in reversed(all_trades):
            if t["symbol"] == sym and t["action"].startswith("OPEN") and "exit_price" not in t:
                entry_price = t["entry_price"]
                t["exit_price"] = close_price
                t["exit_ts"] = ts
                t["reason"] = reason
                break
        if entry_price is None:
            return 0.0
        if side == "LONG":
            pnl = (close_price - entry_price) / entry_price * notional
        else:
            pnl = (entry_price - close_price) / entry_price * notional
        return pnl - fee

    def _open_pos(sym: str, side: str, price: float, ts: int) -> float:
        """Returns entry fee paid."""
        notional = _notional()
        fee = _TAKER_FEE * notional
        all_trades.append({
            "symbol": sym, "action": f"OPEN_{side}",
            "entry_price": price, "entry_ts": ts, "notional": notional,
        })
        return fee

    # Feed events day by day
    def _rebalance(ts: int) -> None:
        nonlocal equity

        # Compute trailing returns for all symbols
        returns: dict[str, float] = {}
        for sym in symbols:
            hist = price_hist[sym]
            if len(hist) <= lookback:
                continue
            curr_c  = hist[-1][1]
            past_c  = hist[-(lookback + 1)][1]
            if past_c <= 0:
                continue
            returns[sym] = (curr_c - past_c) / past_c

        # Determine target positions
        target: dict[str, str] = {}   # symbol → "LONG" | "SHORT" | (absent=flat)
        for sym, ret in returns.items():
            if ret > threshold:
                target[sym] = "LONG"
            elif ret < -threshold:
                target[sym] = "SHORT"
            # else: flat

        # Close positions no longer in target or side changed
        for sym in list(positions.keys()):
            side = positions[sym]
            price = price_hist[sym][-1][1] if price_hist[sym] else None
            if price is None:
                continue
            if sym not in target or target[sym] != side:
                pnl = _close_pos(sym, side, price, ts, "rebalance")
                equity += pnl
                del positions[sym]

        # Open new positions
        for sym, new_side in target.items():
            if sym not in positions:
                price = price_hist[sym][-1][1] if price_hist[sym] else None
                if price is None:
                    continue
                fee = _open_pos(sym, new_side, price, ts)
                equity -= fee
                positions[sym] = new_side

        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        n_long  = sum(1 for s in positions.values() if s == "LONG")
        n_short = sum(1 for s in positions.values() if s == "SHORT")
        n_changed = len([t for t in all_trades if t.get("entry_ts") == ts])
        if n_changed > 0:
            print(f"  Rebal {dt}: L={n_long} S={n_short} positions, {n_changed} changes, equity=${equity:,.0f}")

    # Main loop: process each daily close event
    seen_ts: set[int] = set()
    current_day_events: list[tuple[str, float]] = []
    current_ts: int = 0

    for ts, sym, close in events:
        if ts not in seen_ts and current_day_events:
            # Process previous day's batch
            if last_rebal_ts is None or (current_ts - last_rebal_ts) / _DAY_MS >= rebal_days:
                _rebalance(current_ts)
                last_rebal_ts = current_ts
            seen_ts.add(current_ts)
            current_day_events = []

        # Update price history
        hist = price_hist[sym]
        if hist and hist[-1][0] == ts:
            hist[-1] = (ts, close)
        else:
            hist.append((ts, close))
            if len(hist) > lookback + 2:
                hist.pop(0)

        current_ts = ts
        current_day_events.append((sym, close))
        seen_ts.add(ts)

    # Final rebalance
    if current_day_events:
        if last_rebal_ts is None or (current_ts - last_rebal_ts) / _DAY_MS >= rebal_days:
            _rebalance(current_ts)

    # Close all open positions at end
    for sym in list(positions.keys()):
        side = positions[sym]
        price = price_hist[sym][-1][1] if price_hist[sym] else None
        if price:
            pnl = _close_pos(sym, side, price, current_ts, "eod")
            equity += pnl
        del positions[sym]

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------
    closed_trades = [t for t in all_trades if "exit_price" in t]
    if not closed_trades:
        print("No completed trades.")
        return

    net_pnl_per_trade: list[float] = []
    for t in closed_trades:
        notional = t["notional"]
        side = "LONG" if "LONG" in t["action"] else "SHORT"
        if side == "LONG":
            pnl = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * notional
        else:
            pnl = (t["entry_price"] - t["exit_price"]) / t["entry_price"] * notional
        fee = _TAKER_FEE * notional * 2  # entry + exit
        net_pnl_per_trade.append(pnl - fee)

    gross_profit = sum(p for p in net_pnl_per_trade if p > 0)
    gross_loss   = abs(sum(p for p in net_pnl_per_trade if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    n_trades = len(closed_trades)
    n_wins   = sum(1 for p in net_pnl_per_trade if p > 0)
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    total_fees = sum(_TAKER_FEE * t["notional"] * 2 for t in closed_trades)

    net_pnl_total = equity - capital
    years = (to_ts - from_ts) / (1000 * 86400 * 365.25)
    cagr  = (equity / capital) ** (1 / years) - 1 if years > 0 else 0.0

    # Drawdown (rough: equity doesn't track per-bar, track per-rebalance)
    # Recompute via cumulative PnL
    cum_equity = capital
    peak = capital
    max_dd = 0.0
    for p in net_pnl_per_trade:
        cum_equity += p
        peak = max(peak, cum_equity)
        dd = (peak - cum_equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    print(f"  Final equity  : ${equity:,.0f}")
    print(f"  Net PnL       : ${net_pnl_total:,.0f}  ({net_pnl_total/capital*100:.1f}%)")
    print(f"  CAGR          : {cagr*100:.1f}%")
    print(f"  Max drawdown  : {max_dd*100:.1f}%")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Total trades  : {n_trades}  (L:{sum(1 for t in closed_trades if 'LONG' in t['action'])}  S:{sum(1 for t in closed_trades if 'SHORT' in t['action'])})")
    print(f"  Win rate      : {win_rate*100:.1f}%")
    print(f"  Total fees    : ${total_fees:,.0f}")
    print()

    # Per-symbol breakdown
    print(f"  {'Symbol':<14} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>13}")
    print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*7} {'─'*13}")
    for sym in symbols:
        sym_trades = [(t, p) for t, p in zip(closed_trades, net_pnl_per_trade) if t["symbol"] == sym]
        if not sym_trades:
            continue
        s_pnls = [p for _, p in sym_trades]
        s_gp = sum(p for p in s_pnls if p > 0)
        s_gl = abs(sum(p for p in s_pnls if p < 0))
        s_pf = s_gp / s_gl if s_gl > 0 else float("inf")
        s_wr = sum(1 for p in s_pnls if p > 0) / len(s_pnls) * 100
        s_net = sum(s_pnls)
        print(f"  {sym:<14} {len(sym_trades):>7} {s_wr:>6.1f}% {s_pf:>7.2f} ${s_net:>12,.0f}")

    print()

    # Year breakdown
    years_data: dict[int, list[float]] = {}
    for t, p in zip(closed_trades, net_pnl_per_trade):
        yr = datetime.fromtimestamp(t.get("exit_ts", 0) / 1000, tz=timezone.utc).year
        years_data.setdefault(yr, []).append(p)
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>13}  Status")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*13}  {'─'*6}")
    for yr in sorted(years_data):
        yr_pnls = years_data[yr]
        yr_gp = sum(p for p in yr_pnls if p > 0)
        yr_gl = abs(sum(p for p in yr_pnls if p < 0))
        yr_pf = yr_gp / yr_gl if yr_gl > 0 else float("inf")
        yr_wr = sum(1 for p in yr_pnls if p > 0) / len(yr_pnls) * 100
        yr_net = sum(yr_pnls)
        print(f"  {yr:<8} {len(yr_pnls):>7} {yr_wr:>6.1f}% {yr_pf:>7.2f} ${yr_net:>12,.0f}  {'+'if yr_net>0 else '-'}")

    print()
    is_pnls  = [p for t, p in zip(closed_trades, net_pnl_per_trade)
                if datetime.fromtimestamp(t.get("exit_ts",0)/1000,tz=timezone.utc).year < 2023]
    oos_pnls = [p for t, p in zip(closed_trades, net_pnl_per_trade)
                if datetime.fromtimestamp(t.get("exit_ts",0)/1000,tz=timezone.utc).year >= 2023]
    print(f"  OOS split (IS = <2023, OOS = 2023+):")
    for label, pnls in [("IS ", is_pnls), ("OOS", oos_pnls)]:
        if not pnls:
            print(f"    {label}: 0 trades, PF 0.00")
            continue
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        tpf = gp / gl if gl > 0 else float("inf")
        print(f"    {label}: {len(pnls)} trades, PF {tpf:.2f}")
    print("=" * 70)

    # Verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    kill_reasons = []
    if pf < 1.10:
        kill_reasons.append(f"Profit factor {pf:.2f} < kill threshold 1.1")
    if max_dd > 0.40:
        kill_reasons.append(f"Max drawdown {max_dd*100:.1f}% > kill threshold 40%")
    if n_trades < 60:
        kill_reasons.append(f"Trade count {n_trades} < minimum 60")
    if is_pnls and oos_pnls:
        is_gp  = sum(p for p in is_pnls if p > 0)
        is_gl  = abs(sum(p for p in is_pnls if p < 0))
        is_pf  = is_gp / is_gl if is_gl > 0 else float("inf")
        oos_gp = sum(p for p in oos_pnls if p > 0)
        oos_gl = abs(sum(p for p in oos_pnls if p < 0))
        oos_pf = oos_gp / oos_gl if oos_gl > 0 else float("inf")
        if is_pf < float("inf") and oos_pf < 0.85 * is_pf:
            kill_reasons.append(
                f"OOS PF {oos_pf:.2f} < 0.85 × IS PF {is_pf:.2f} ({0.85*is_pf:.2f})"
            )

    if kill_reasons:
        print(f"  KILL -- Strategy fails one or more kill criteria:")
        print()
        for r in kill_reasons:
            print(f"     * {r}")
        print()
        print(f"  Action: do NOT deploy. Move to next hypothesis.")
    elif pf >= 1.30 and cagr >= 0.20 and max_dd <= 0.40:
        print(f"  GO -- All thresholds met:")
        print(f"     PF {pf:.2f} ≥ 1.30, CAGR {cagr*100:.1f}% ≥ 20%, DD {max_dd*100:.1f}% ≤ 40%")
        print()
        print(f"  Action: deploy to live paper trading immediately.")
    else:
        print(f"  MARGINAL -- Passes kill criteria but below GO threshold:")
        print(f"     PF {pf:.2f}, CAGR {cagr*100:.1f}%, DD {max_dd*100:.1f}%")
        print()
        print(f"  Action: consider in combined portfolio; do NOT deploy solo.")

    print("=" * 70)
    print()
    print("=" * 70)
    print("  Backtest complete.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",   nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--capital",   type=float, default=_INITIAL_EQUITY)
    parser.add_argument("--lookback",  type=int,   default=_LOOKBACK_DAYS)
    parser.add_argument("--rebal",     type=int,   default=_REBAL_DAYS)
    parser.add_argument("--threshold", type=float, default=_THRESHOLD_PCT)
    parser.add_argument("--from", dest="from_date", default="2021-01-01")
    parser.add_argument("--to",   dest="to_date",   default=None)
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (
        datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.to_date else datetime.now(timezone.utc)
    )
    from_ts = int(from_dt.timestamp() * 1000)
    to_ts   = int(to_dt.timestamp() * 1000)

    print("=" * 70)
    print("  TIME-SERIES MOMENTUM (TSMOM) BACKTEST — mechanism #11")
    print("=" * 70)
    print(f"  Hypothesis: absolute 6M momentum predicts continuation")
    print(f"  Pre-committed rule:")
    print(f"    lookback     = {args.lookback} calendar days  (≈6 months)")
    print(f"    long signal  = trailing return > +{args.threshold*100:.0f}%")
    print(f"    short signal = trailing return < -{args.threshold*100:.0f}%")
    print(f"    flat signal  = |return| ≤ {args.threshold*100:.0f}% (no position)")
    print(f"    rebal        = every {args.rebal} days")
    print(f"    position     = capital/{len(args.symbols)} per signal")
    print(f"    fee          = {_TAKER_FEE*100:.3f}% taker/side, only on changes")
    print(f"    symbols      = {', '.join(args.symbols)}")
    print()

    print("=" * 70)
    print("  SECTION B — MONETIZATION BACKTEST (after fees)")
    print("=" * 70)
    print(f"  Period        : {from_dt.strftime('%Y-%m-%d')} → {to_dt.strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${args.capital:,.0f}")
    print()

    print("Loading daily candles ...")
    run_backtest(
        args.symbols, args.capital, from_ts, to_ts,
        args.lookback, args.rebal, args.threshold,
    )


if __name__ == "__main__":
    main()
