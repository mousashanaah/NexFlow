#!/usr/bin/env python3
"""Section B backtest for mechanism #10: Daily RSI in-trend pullback.

Hypothesis: Price in a defined trend (SMA-200) + RSI pullback to extreme →
  high-probability mean-reversion entry that's aligned with the trend.

Pre-committed parameters (locked before first run):
  instruments  : BTCUSDT, ETHUSDT, SOLUSDT, TRXUSDT
  signal       : daily RSI(14) + daily SMA(200) regime filter
  long_entry   : RSI(14) < 30 AND close > SMA(200)   [oversold dip in uptrend]
  short_entry  : RSI(14) > 70 AND close < SMA(200)   [overbought bounce in downtrend]
  exit         : RSI crosses 50 (mean-reversion) OR 8% adverse price stop
  position     : 20% of equity per trade, max 1 per symbol
  fee          : 0.06% taker/side

Usage:
    python scripts/backtest_daily_rsi_pullback.py
    python scripts/backtest_daily_rsi_pullback.py --symbols BTCUSDT ETHUSDT
    python scripts/backtest_daily_rsi_pullback.py --from 2022-01-01
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT"]
_INITIAL_EQUITY  = 100_000.0
_POSITION_FRAC   = 0.20      # 20% of equity per trade
_RSI_PERIOD      = 14
_SMA_PERIOD      = 200
_RSI_LONG_ENTRY  = 30        # RSI < this → long entry (oversold)
_RSI_SHORT_ENTRY = 70        # RSI > this → short entry (overbought)
_RSI_EXIT        = 50        # RSI crosses 50 → exit (mean-reversion complete)
_STOP_PCT        = 0.08      # 8% adverse stop
_TAKER_FEE       = 0.0006    # 0.06% per side
_CANDLE_DIR      = _REPO_ROOT / "data" / "candles"
_DAY_MS          = 86_400_000


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _compute_rsi(closes: list[float], period: int) -> list[float]:
    """Wilder-smoothed RSI. Returns same-length list; first (period) entries are nan."""
    rsi = [float("nan")] * len(closes)
    if len(closes) <= period:
        return rsi

    gains = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - 100 / (1 + rs)

    for i in range(period + 1, len(closes)):
        g = gains[i - 1]
        lo = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + lo) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - 100 / (1 + rs)

    return rsi


def _compute_sma(closes: list[float], period: int) -> list[float]:
    """Simple moving average. First (period-1) entries are nan."""
    sma = [float("nan")] * len(closes)
    window_sum = 0.0
    for i, c in enumerate(closes):
        window_sum += c
        if i >= period:
            window_sum -= closes[i - period]
        if i >= period - 1:
            sma[i] = window_sum / period
    return sma


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_daily(symbol: str, from_ts: int, to_ts: int) -> list[tuple[int, float]]:
    """Returns [(ts_ms, close), ...] sorted, filtered to [from_ts, to_ts]."""
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    if not path.exists():
        print(f"  [WARN] No daily data for {symbol}")
        return []
    tbl = pq.read_table(path, columns=["open_time", "close"])
    rows = sorted(zip(
        tbl.column("open_time").to_pylist(),
        tbl.column("close").to_pylist(),
    ))
    return [(ts, float(c)) for ts, c in rows if from_ts <= ts <= to_ts]


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    side: str
    entry_ts: int
    entry_price: float
    notional: float
    exit_ts: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    price_pnl: float = 0.0
    fees: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.price_pnl - self.fees

    @property
    def ret_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == "LONG":
            return (self.exit_price - self.entry_price) / self.entry_price
        return (self.entry_price - self.exit_price) / self.entry_price


# ---------------------------------------------------------------------------
# Per-symbol simulation
# ---------------------------------------------------------------------------
def _simulate(
    symbol: str,
    bars: list[tuple[int, float]],
    initial_equity: float,
) -> list[Trade]:
    if len(bars) <= _SMA_PERIOD + _RSI_PERIOD:
        return []

    closes = [c for _, c in bars]
    timestamps = [ts for ts, _ in bars]
    rsi_vals = _compute_rsi(closes, _RSI_PERIOD)
    sma_vals = _compute_sma(closes, _SMA_PERIOD)

    trades: list[Trade] = []
    open_trade: Trade | None = None
    equity = initial_equity

    for i in range(_SMA_PERIOD, len(bars)):
        ts     = timestamps[i]
        close  = closes[i]
        rsi    = rsi_vals[i]
        sma    = sma_vals[i]
        prev_rsi = rsi_vals[i - 1]

        if rsi != rsi or sma != sma:  # NaN check
            continue

        # ---- Manage open position ----
        if open_trade is not None:
            # RSI exit condition (crosses 50)
            rsi_exit = (
                (open_trade.side == "LONG"  and prev_rsi < _RSI_EXIT and rsi >= _RSI_EXIT) or
                (open_trade.side == "SHORT" and prev_rsi > _RSI_EXIT and rsi <= _RSI_EXIT)
            )
            # Stop loss
            adverse = (
                (close - open_trade.entry_price) / open_trade.entry_price
                if open_trade.side == "SHORT"
                else (open_trade.entry_price - close) / open_trade.entry_price
            )
            stop_hit = adverse >= _STOP_PCT

            if rsi_exit or stop_hit:
                open_trade.exit_ts    = ts
                open_trade.exit_price = close
                open_trade.exit_reason = "stop" if stop_hit else "rsi50"
                if open_trade.side == "LONG":
                    open_trade.price_pnl = (close - open_trade.entry_price) / open_trade.entry_price * open_trade.notional
                else:
                    open_trade.price_pnl = (open_trade.entry_price - close) / open_trade.entry_price * open_trade.notional
                open_trade.fees += _TAKER_FEE * open_trade.notional  # exit fee
                equity += open_trade.net_pnl
                open_trade = None

        # ---- Entry signals (only if flat) ----
        if open_trade is None:
            notional = equity * _POSITION_FRAC
            if rsi < _RSI_LONG_ENTRY and close > sma:
                t = Trade(
                    symbol=symbol, side="LONG",
                    entry_ts=ts, entry_price=close, notional=notional,
                )
                t.fees += _TAKER_FEE * notional
                equity -= _TAKER_FEE * notional
                open_trade = t
                trades.append(t)
            elif rsi > _RSI_SHORT_ENTRY and close < sma:
                t = Trade(
                    symbol=symbol, side="SHORT",
                    entry_ts=ts, entry_price=close, notional=notional,
                )
                t.fees += _TAKER_FEE * notional
                equity -= _TAKER_FEE * notional
                open_trade = t
                trades.append(t)

    # Close open position at end (mark to last price)
    if open_trade is not None:
        last_ts, last_price = bars[-1]
        open_trade.exit_ts    = last_ts
        open_trade.exit_price = last_price
        open_trade.exit_reason = "eod"
        if open_trade.side == "LONG":
            open_trade.price_pnl = (last_price - open_trade.entry_price) / open_trade.entry_price * open_trade.notional
        else:
            open_trade.price_pnl = (open_trade.entry_price - last_price) / open_trade.entry_price * open_trade.notional
        open_trade.fees += _TAKER_FEE * open_trade.notional

    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _stats(
    all_trades: list[Trade],
    initial_equity: float,
    from_ts: int,
    to_ts: int,
) -> tuple[float, float, float, int, int, float, float]:
    closed = [t for t in all_trades if t.exit_ts > 0]
    if not closed:
        return initial_equity, 0.0, 0.0, 0, 0, 0.0, 0.0

    closed.sort(key=lambda t: t.exit_ts)
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    gross_profit = gross_loss = 0.0

    for t in closed:
        equity += t.net_pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        if t.net_pnl > 0:
            gross_profit += t.net_pnl
        else:
            gross_loss += abs(t.net_pnl)

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    n_wins = sum(1 for t in closed if t.net_pnl > 0)
    total_fees = sum(t.fees for t in closed)
    years = (to_ts - from_ts) / (1000 * 86400 * 365.25)
    cagr = (equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0
    return equity, cagr, max_dd, len(closed), n_wins, pf, total_fees


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    parser.add_argument("--capital", type=float, default=_INITIAL_EQUITY)
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
    print("  DAILY RSI PULLBACK BACKTEST — mechanism #10")
    print("=" * 70)
    print(f"  Hypothesis: RSI oversold/overbought in trend → high-prob reversion")
    print(f"  Pre-committed rule:")
    print(f"    long_entry   = RSI({_RSI_PERIOD}) < {_RSI_LONG_ENTRY} AND close > SMA({_SMA_PERIOD})")
    print(f"    short_entry  = RSI({_RSI_PERIOD}) > {_RSI_SHORT_ENTRY} AND close < SMA({_SMA_PERIOD})")
    print(f"    exit         = RSI crosses {_RSI_EXIT} OR {_STOP_PCT*100:.0f}% adverse stop")
    print(f"    position     = {_POSITION_FRAC*100:.0f}% of equity, max 1 per symbol")
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

    for sym in args.symbols:
        bars = _load_daily(sym, from_ts, to_ts)
        if not bars:
            print(f"  {sym}: no data")
            continue
        print(f"  {sym}: {len(bars):,} daily bars")
        sym_trades = _simulate(sym, bars, args.capital / len(args.symbols))
        all_trades.extend(sym_trades)
        if sym_trades:
            closed = [t for t in sym_trades if t.exit_ts > 0]
            n_long  = sum(1 for t in closed if t.side == "LONG")
            n_short = sum(1 for t in closed if t.side == "SHORT")
            print(f"    {len(closed)} trades (L:{n_long}  S:{n_short})")

    print()

    equity, cagr, max_dd, n_trades, n_wins, pf, total_fees = _stats(
        all_trades, args.capital, from_ts, to_ts
    )
    net_pnl  = equity - args.capital
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0

    print(f"  Final equity  : ${equity:,.0f}")
    print(f"  Net PnL       : ${net_pnl:,.0f}  ({net_pnl/args.capital*100:.1f}%)")
    print(f"  CAGR          : {cagr*100:.1f}%")
    print(f"  Max drawdown  : {max_dd*100:.1f}%")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Total trades  : {n_trades}  (L:{sum(1 for t in all_trades if t.side=='LONG' and t.exit_ts>0)}  S:{sum(1 for t in all_trades if t.side=='SHORT' and t.exit_ts>0)})")
    print(f"  Win rate      : {win_rate*100:.1f}%")
    print(f"  Total fees    : ${total_fees:,.0f}")
    print()

    # Per-symbol
    print(f"  {'Symbol':<14} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>13}")
    print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*7} {'─'*13}")
    for sym in args.symbols:
        sym_t = [t for t in all_trades if t.symbol == sym and t.exit_ts > 0]
        if not sym_t:
            continue
        gp = sum(t.net_pnl for t in sym_t if t.net_pnl > 0)
        gl = abs(sum(t.net_pnl for t in sym_t if t.net_pnl < 0))
        sym_pf = gp / gl if gl > 0 else float("inf")
        sym_wr = sum(1 for t in sym_t if t.net_pnl > 0) / len(sym_t) * 100
        sym_net = sum(t.net_pnl for t in sym_t)
        print(f"  {sym:<14} {len(sym_t):>7} {sym_wr:>6.1f}% {sym_pf:>7.2f} ${sym_net:>12,.0f}")

    print()

    # Year-by-year
    years_data: dict[int, list[Trade]] = {}
    for t in all_trades:
        if t.exit_ts == 0:
            continue
        yr = datetime.fromtimestamp(t.exit_ts / 1000, tz=timezone.utc).year
        years_data.setdefault(yr, []).append(t)
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net$':>13}  Status")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*13}  {'─'*6}")
    for yr in sorted(years_data):
        yr_t = years_data[yr]
        gp = sum(t.net_pnl for t in yr_t if t.net_pnl > 0)
        gl = abs(sum(t.net_pnl for t in yr_t if t.net_pnl < 0))
        yr_pf = gp / gl if gl > 0 else float("inf")
        yr_wr = sum(1 for t in yr_t if t.net_pnl > 0) / len(yr_t) * 100
        yr_net = sum(t.net_pnl for t in yr_t)
        print(f"  {yr:<8} {len(yr_t):>7} {yr_wr:>6.1f}% {yr_pf:>7.2f} ${yr_net:>12,.0f}  {'+'if yr_net>0 else '-'}")

    print()
    is_trades  = [t for t in all_trades if t.exit_ts > 0 and
                  datetime.fromtimestamp(t.exit_ts/1000, tz=timezone.utc).year < 2023]
    oos_trades = [t for t in all_trades if t.exit_ts > 0 and
                  datetime.fromtimestamp(t.exit_ts/1000, tz=timezone.utc).year >= 2023]
    print(f"  OOS split (IS = <2023, OOS = 2023+):")
    for label, tset in [("IS ", is_trades), ("OOS", oos_trades)]:
        if not tset:
            print(f"    {label}: 0 trades, PF 0.00")
            continue
        gp = sum(t.net_pnl for t in tset if t.net_pnl > 0)
        gl = abs(sum(t.net_pnl for t in tset if t.net_pnl < 0))
        tpf = gp / gl if gl > 0 else float("inf")
        print(f"    {label}: {len(tset)} trades, PF {tpf:.2f}")
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
    if is_trades and oos_trades:
        is_gp  = sum(t.net_pnl for t in is_trades if t.net_pnl > 0)
        is_gl  = abs(sum(t.net_pnl for t in is_trades if t.net_pnl < 0))
        is_pf  = is_gp / is_gl if is_gl > 0 else float("inf")
        oos_gp = sum(t.net_pnl for t in oos_trades if t.net_pnl > 0)
        oos_gl = abs(sum(t.net_pnl for t in oos_trades if t.net_pnl < 0))
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


if __name__ == "__main__":
    main()
