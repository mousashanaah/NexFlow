#!/usr/bin/env python3
"""Section B backtest for mechanism #9: Funding-rate carry.

Hypothesis: Extreme perpetual funding rate signals crowded positioning.
  SHORT when funding is extreme-positive (over-leveraged longs) → receive carry + fade.
  LONG  when funding is extreme-negative (over-leveraged shorts) → receive carry + fade.

Pre-committed parameters (locked before first run):
  instruments  : BTCUSDT, ETHUSDT
  signal       : 3-period EMA of 8H funding rate
  short_entry  : EMA > +0.00025  (≈ p90 of BTC funding, 0.025%/8H)
  long_entry   : EMA < -0.00006  (≈ p5  of BTC funding, -0.006%/8H)
  exit_short   : EMA < +0.0001 (neutral) OR price +5% adverse
  exit_long    : EMA > -0.0001 (neutral) OR price -5% adverse
  position size: 25% of equity per trade, max 1 per symbol
  carry income : funding_rate × notional each 8H period held
  fee          : 0.06% taker on entry and exit
  universe     : BTC + ETH only (coins with reliable funding history)

Usage:
    python scripts/backtest_funding_carry.py
    python scripts/backtest_funding_carry.py --from 2022-01-01
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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
_DEFAULT_SYMBOLS  = ["BTCUSDT", "ETHUSDT"]
_INITIAL_EQUITY   = 100_000.0
_POSITION_FRAC    = 0.25       # 25% of equity per trade
_SHORT_ENTRY_EMA  = 0.00025    # EMA > this → SHORT (extreme positive funding)
_LONG_ENTRY_EMA   = -0.00006   # EMA < this → LONG  (extreme negative funding)
_SHORT_EXIT_EMA   = 0.0001     # EMA drops below this → exit SHORT
_LONG_EXIT_EMA    = -0.0001    # EMA rises above this → exit LONG
_STOP_PCT         = 0.05       # 5% adverse price move → stop out
_EMA_PERIOD       = 3          # periods for funding EMA
_TAKER_FEE        = 0.0006     # 0.06% per side
_CANDLE_DIR       = _REPO_ROOT / "data" / "candles"
_FUNDING_DIR      = _REPO_ROOT / "data" / "funding"
_HOUR_MS          = 3_600_000
_8H_MS            = 8 * _HOUR_MS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_8h_prices(symbol: str, from_ts: int, to_ts: int) -> dict[int, float]:
    """Aggregate 1H candles to 8H buckets. Returns {bucket_start_ms: close_price}."""
    path = _CANDLE_DIR / f"{symbol}_1H.parquet"
    if not path.exists():
        print(f"  [WARN] No 1H data for {symbol}")
        return {}
    tbl = pq.read_table(path, columns=["open_time", "close"])
    prices: dict[int, float] = {}
    for ot, c in sorted(zip(
        tbl.column("open_time").to_pylist(),
        tbl.column("close").to_pylist(),
    )):
        if ot < from_ts or ot > to_ts:
            continue
        bucket = (ot // _8H_MS) * _8H_MS
        prices[bucket] = float(c)  # last 1H close in this 8H bucket is the 8H close
    return prices


def _load_funding(symbol: str, from_ts: int, to_ts: int) -> list[tuple[int, float]]:
    """Load 8H funding rate data. Returns [(timestamp_ms, rate), ...] sorted."""
    path = _FUNDING_DIR / f"{symbol}_funding.parquet"
    if not path.exists():
        print(f"  [WARN] No funding data for {symbol}")
        return []
    tbl = pq.read_table(path, columns=["timestamp_ms", "funding_rate"])
    rows = sorted(zip(
        tbl.column("timestamp_ms").to_pylist(),
        tbl.column("funding_rate").to_pylist(),
    ))
    return [(ts, float(fr)) for ts, fr in rows if from_ts <= ts <= to_ts]


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    side: str           # "LONG" or "SHORT"
    entry_ts: int
    entry_price: float
    notional: float
    exit_ts: int = 0
    exit_price: float = 0.0
    price_pnl: float = 0.0
    carry_pnl: float = 0.0
    fees: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.price_pnl + self.carry_pnl - self.fees

    @property
    def is_open(self) -> bool:
        return self.exit_ts == 0


def _ema_update(prev_ema: float, value: float, alpha: float) -> float:
    return alpha * value + (1 - alpha) * prev_ema


# ---------------------------------------------------------------------------
# Per-symbol simulation
# ---------------------------------------------------------------------------
def _simulate_symbol(
    symbol: str,
    funding: list[tuple[int, float]],
    prices: dict[int, float],
    initial_equity: float,
) -> list[Trade]:
    alpha = 2.0 / (_EMA_PERIOD + 1)
    trades: list[Trade] = []
    open_trade: Trade | None = None
    ema = 0.0
    equity = initial_equity

    for ts, rate in funding:
        price = prices.get(ts)
        if price is None:
            # Try the nearest price bucket
            bucket = (ts // _8H_MS) * _8H_MS
            price = prices.get(bucket)
        if price is None:
            continue

        # Update EMA
        ema = _ema_update(ema, rate, alpha)

        # ---- Manage open position ----
        if open_trade is not None and open_trade.is_open:
            # Carry income this period
            if open_trade.side == "SHORT":
                carry = rate * open_trade.notional
            else:
                carry = -rate * open_trade.notional  # LONG pays when rate > 0
            open_trade.carry_pnl += carry

            # Check exit conditions
            adverse_pct = (
                (price - open_trade.entry_price) / open_trade.entry_price
                if open_trade.side == "SHORT"
                else (open_trade.entry_price - price) / open_trade.entry_price
            )
            stop_hit   = adverse_pct >= _STOP_PCT
            ema_exit   = (
                (ema < _SHORT_EXIT_EMA if open_trade.side == "SHORT" else ema > _LONG_EXIT_EMA)
            )

            if stop_hit or ema_exit:
                open_trade.exit_ts    = ts
                open_trade.exit_price = price
                if open_trade.side == "SHORT":
                    open_trade.price_pnl = (open_trade.entry_price - price) / open_trade.entry_price * open_trade.notional
                else:
                    open_trade.price_pnl = (price - open_trade.entry_price) / open_trade.entry_price * open_trade.notional
                open_trade.fees += _TAKER_FEE * open_trade.notional  # exit fee
                equity += open_trade.net_pnl
                open_trade = None

        # ---- Check entry (only if flat) ----
        if open_trade is None:
            notional = equity * _POSITION_FRAC
            if ema > _SHORT_ENTRY_EMA:
                t = Trade(
                    symbol=symbol, side="SHORT",
                    entry_ts=ts, entry_price=price, notional=notional,
                )
                t.fees += _TAKER_FEE * notional  # entry fee
                equity -= _TAKER_FEE * notional
                open_trade = t
                trades.append(t)
            elif ema < _LONG_ENTRY_EMA:
                t = Trade(
                    symbol=symbol, side="LONG",
                    entry_ts=ts, entry_price=price, notional=notional,
                )
                t.fees += _TAKER_FEE * notional  # entry fee
                equity -= _TAKER_FEE * notional
                open_trade = t
                trades.append(t)

    # Close any open position at end of period (mark-to-market at last price)
    if open_trade is not None and open_trade.is_open:
        if prices:
            last_ts    = max(prices.keys())
            last_price = prices[last_ts]
        else:
            last_ts    = open_trade.entry_ts
            last_price = open_trade.entry_price
        open_trade.exit_ts    = last_ts
        open_trade.exit_price = last_price
        if open_trade.side == "SHORT":
            open_trade.price_pnl = (open_trade.entry_price - last_price) / open_trade.entry_price * open_trade.notional
        else:
            open_trade.price_pnl = (last_price - open_trade.entry_price) / open_trade.entry_price * open_trade.notional
        open_trade.fees += _TAKER_FEE * open_trade.notional
        # trade was already appended to trades[] on entry

    return trades


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def _compute_equity_curve(
    all_trades: list[Trade],
    initial_equity: float,
    from_ts: int,
    to_ts: int,
) -> tuple[float, float, float, int, int, float, float]:
    """Returns (final_equity, cagr, max_dd, n_trades, n_wins, pf, total_fees)."""
    closed = [t for t in all_trades if t.exit_ts > 0]
    if not closed:
        return initial_equity, 0.0, 0.0, 0, 0, 0.0, 0.0

    # Sort by exit time
    closed.sort(key=lambda t: t.exit_ts)

    equity = initial_equity
    peak = equity
    max_dd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    total_fees = sum(t.fees for t in closed)

    for t in closed:
        equity += t.net_pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        net = t.net_pnl
        if net > 0:
            gross_profit += net
        else:
            gross_loss += abs(net)

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    n_wins = sum(1 for t in closed if t.net_pnl > 0)

    years = (to_ts - from_ts) / (1000 * 86400 * 365.25)
    cagr = (equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0

    return equity, cagr, max_dd, len(closed), n_wins, pf, total_fees


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--capital", type=float, default=_INITIAL_EQUITY)
    parser.add_argument("--from", dest="from_date", default="2020-01-01")
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
    print("  FUNDING-RATE CARRY BACKTEST — mechanism #9")
    print("=" * 70)
    print(f"  Hypothesis: Extreme perpetual funding → crowded positioning → fade")
    print(f"  Pre-committed rule:")
    print(f"    short_entry  = EMA_{_EMA_PERIOD}(funding) > {_SHORT_ENTRY_EMA*100:.4f}%/8H")
    print(f"    long_entry   = EMA_{_EMA_PERIOD}(funding) < {_LONG_ENTRY_EMA*100:.4f}%/8H")
    print(f"    exit_short   = EMA < {_SHORT_EXIT_EMA*100:.4f}% OR +{_STOP_PCT*100:.0f}% adverse")
    print(f"    exit_long    = EMA > {_LONG_EXIT_EMA*100:.4f}% OR -{_STOP_PCT*100:.0f}% adverse")
    print(f"    position     = {_POSITION_FRAC*100:.0f}% of equity, max 1 per symbol")
    print(f"    carry income = funding_rate × notional each 8H held")
    print(f"    fee          = {_TAKER_FEE*100:.3f}% taker/side")
    print(f"    symbols      = {', '.join(args.symbols)}")
    print()

    print("=" * 70)
    print("  SECTION B — MONETIZATION BACKTEST (after fees)")
    print("=" * 70)
    print(f"  Period        : {from_dt.strftime('%Y-%m-%d')} → {to_dt.strftime('%Y-%m-%d')}")
    print(f"  Initial equity: ${args.capital:,.0f}")
    print()

    all_trades: list[Trade] = []
    equity_per_symbol = args.capital / len(args.symbols)

    for sym in args.symbols:
        print(f"  Loading {sym} ...")
        funding = _load_funding(sym, from_ts, to_ts)
        prices  = _load_8h_prices(sym, from_ts, to_ts)
        if not funding:
            print(f"    [SKIP] No funding data")
            continue
        print(f"    {len(funding):,} funding periods, {len(prices):,} 8H price bars")
        trades = _simulate_symbol(sym, funding, prices, equity_per_symbol)
        all_trades.extend(trades)
        print(f"    {len(trades)} trades generated")

    print()

    # Combined portfolio stats
    equity, cagr, max_dd, n_trades, n_wins, pf, total_fees = _compute_equity_curve(
        all_trades, args.capital, from_ts, to_ts
    )
    net_pnl   = equity - args.capital
    win_rate  = n_wins / n_trades if n_trades > 0 else 0.0
    total_carry = sum(t.carry_pnl for t in all_trades if t.exit_ts > 0)
    total_price = sum(t.price_pnl for t in all_trades if t.exit_ts > 0)

    print(f"  Final equity  : ${equity:,.0f}")
    print(f"  Net PnL       : ${net_pnl:,.0f}  ({net_pnl/args.capital*100:.1f}%)")
    print(f"  CAGR          : {cagr*100:.1f}%")
    print(f"  Max drawdown  : {max_dd*100:.1f}%")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Total trades  : {n_trades}  (L:{sum(1 for t in all_trades if t.side=='LONG' and t.exit_ts>0)}  S:{sum(1 for t in all_trades if t.side=='SHORT' and t.exit_ts>0)})")
    print(f"  Win rate      : {win_rate*100:.1f}%")
    print(f"  Price PnL     : ${total_price:,.0f}")
    print(f"  Carry PnL     : ${total_carry:,.0f}")
    print(f"  Total fees    : ${total_fees:,.0f}")
    print()

    # Per-symbol breakdown
    print(f"  {'Symbol':<14} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>13}  {'Carry$':>10}")
    print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*7} {'─'*13}  {'─'*10}")
    for sym in args.symbols:
        sym_trades = [t for t in all_trades if t.symbol == sym and t.exit_ts > 0]
        if not sym_trades:
            continue
        sym_gp = sum(t.net_pnl for t in sym_trades if t.net_pnl > 0)
        sym_gl = abs(sum(t.net_pnl for t in sym_trades if t.net_pnl < 0))
        sym_pf = sym_gp / sym_gl if sym_gl > 0 else float("inf")
        sym_wr = sum(1 for t in sym_trades if t.net_pnl > 0) / len(sym_trades) * 100
        sym_net = sum(t.net_pnl for t in sym_trades)
        sym_carry = sum(t.carry_pnl for t in sym_trades)
        print(f"  {sym:<14} {len(sym_trades):>7} {sym_wr:>6.1f}% {sym_pf:>7.2f} ${sym_net:>12,.0f}  ${sym_carry:>9,.0f}")

    print()

    # Year-by-year breakdown
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>13}  {'Status'}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*13}  {'─'*10}")
    years_seen: dict[int, list[Trade]] = {}
    for t in all_trades:
        if t.exit_ts == 0:
            continue
        yr = datetime.fromtimestamp(t.exit_ts / 1000, tz=timezone.utc).year
        years_seen.setdefault(yr, []).append(t)
    for yr in sorted(years_seen):
        yr_trades = years_seen[yr]
        gp = sum(t.net_pnl for t in yr_trades if t.net_pnl > 0)
        gl = abs(sum(t.net_pnl for t in yr_trades if t.net_pnl < 0))
        yr_pf = gp / gl if gl > 0 else float("inf")
        yr_wr = sum(1 for t in yr_trades if t.net_pnl > 0) / len(yr_trades) * 100
        yr_net = sum(t.net_pnl for t in yr_trades)
        status = "+" if yr_net > 0 else "-"
        print(f"  {yr:<8} {len(yr_trades):>7} {yr_wr:>6.1f}% {yr_pf:>7.2f} ${yr_net:>12,.0f}  {status}")

    print()
    print(f"  OOS split (IS = <2023, OOS = 2023+):")
    is_trades  = [t for t in all_trades if t.exit_ts > 0 and
                  datetime.fromtimestamp(t.exit_ts/1000, tz=timezone.utc).year < 2023]
    oos_trades = [t for t in all_trades if t.exit_ts > 0 and
                  datetime.fromtimestamp(t.exit_ts/1000, tz=timezone.utc).year >= 2023]
    for label, tset in [("IS ", is_trades), ("OOS", oos_trades)]:
        if not tset:
            print(f"    {label}: 0 trades, PF 0.00")
            continue
        gp = sum(t.net_pnl for t in tset if t.net_pnl > 0)
        gl = abs(sum(t.net_pnl for t in tset if t.net_pnl < 0))
        tpf = gp / gl if gl > 0 else float("inf")
        print(f"    {label}: {len(tset)} trades, PF {tpf:.2f}")
    print("=" * 70)
    print()

    # Verdict
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
    if is_trades and oos_trades:
        is_gp = sum(t.net_pnl for t in is_trades if t.net_pnl > 0)
        is_gl = abs(sum(t.net_pnl for t in is_trades if t.net_pnl < 0))
        is_pf = is_gp / is_gl if is_gl > 0 else 0.0
        oos_gp = sum(t.net_pnl for t in oos_trades if t.net_pnl > 0)
        oos_gl = abs(sum(t.net_pnl for t in oos_trades if t.net_pnl < 0))
        oos_pf = oos_gp / oos_gl if oos_gl > 0 else 0.0
        if is_pf > 0 and oos_pf < 0.85 * is_pf:
            kill_reasons.append(f"OOS PF {oos_pf:.2f} < 0.85 × IS PF {is_pf:.2f} ({0.85*is_pf:.2f})")

    if kill_reasons:
        print(f"  KILL -- Strategy fails one or more kill criteria:")
        print()
        for r in kill_reasons:
            print(f"     * {r}")
        print()
        print(f"  Action: do NOT deploy. Move to next hypothesis.")
    elif pf >= 1.30 and cagr >= 0.20 and max_dd <= 0.40:
        print(f"  GO -- Strategy passes all thresholds:")
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


if __name__ == "__main__":
    main()
