#!/usr/bin/env python3
"""Mechanism #13: 1H Volume-Momentum Breakout

Entry signal:
  - 1H bar range (high-low) > range_mult × ATR(14)
  - 1H bar volume > vol_mult × SMA(volume, 20)
  - Long: bar closed UP (close > open)
  - Short: bar closed DOWN (close < open)

Exit:
  - Take profit: entry ± tp_r × ATR at signal bar
  - Stop loss  : entry ∓ sl_r × ATR at signal bar  (so R:R = tp_r/sl_r)
  - Time stop  : close at end of max_bars bars if neither hit

Daily bias filter (optional):
  - Long entries only when daily EMA(8) > EMA(21)
  - Short entries only when daily EMA(8) < EMA(21)

Universe: 12 Bitget USDT-perp coins
Timeframe: 1H candles
Fee: 0.06% taker each side

Pre-committed parameters:
  range_mult=2.0, vol_mult=2.0, tp_r=2.0, sl_r=1.0, max_bars=4
  long_only=True, daily_filter=True (tested together)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pip install pyarrow")
    sys.exit(1)

_CANDLE_DIR = _REPO_ROOT / "data" / "candles"
_TAKER_FEE  = 0.0006
_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_1h(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1H.parquet"
    if not path.exists():
        return []
    tbl = pq.read_table(path, columns=["open_time", "open", "high", "low", "close", "volume"])
    rows = []
    for ot, o, h, l, c, v in zip(
        tbl.column("open_time").to_pylist(),
        tbl.column("open").to_pylist(),
        tbl.column("high").to_pylist(),
        tbl.column("low").to_pylist(),
        tbl.column("close").to_pylist(),
        tbl.column("volume").to_pylist(),
    ):
        rows.append({"ts": ot, "open": float(o), "high": float(h),
                     "low": float(l), "close": float(c), "volume": float(v)})
    rows.sort(key=lambda x: x["ts"])
    return rows


def _load_daily(symbol: str) -> list[tuple[int, float]]:
    path = _CANDLE_DIR / f"{symbol}_1D.parquet"
    if not path.exists():
        return []
    tbl = pq.read_table(path, columns=["open_time", "close"])
    rows = sorted(zip(
        tbl.column("open_time").to_pylist(),
        tbl.column("close").to_pylist(),
    ))
    return [(int(ts), float(c)) for ts, c in rows]


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _atr(bars: list[dict], period: int = 14) -> list[Optional[float]]:
    trs = []
    for i, b in enumerate(bars):
        if i == 0:
            trs.append(b["high"] - b["low"])
        else:
            prev_c = bars[i-1]["close"]
            trs.append(max(b["high"] - b["low"],
                           abs(b["high"] - prev_c),
                           abs(b["low"]  - prev_c)))
    atrs: list[Optional[float]] = [None] * len(bars)
    for i in range(period - 1, len(bars)):
        atrs[i] = sum(trs[i-period+1:i+1]) / period
    return atrs


def _vol_sma(bars: list[dict], period: int = 20) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(bars)
    for i in range(period - 1, len(bars)):
        result[i] = sum(b["volume"] for b in bars[i-period+1:i+1]) / period
    return result


def _ema_series(closes: list[float], period: int) -> list[Optional[float]]:
    alpha = 2.0 / (period + 1)
    result: list[Optional[float]] = [None] * len(closes)
    ema = None
    for i, c in enumerate(closes):
        if ema is None:
            ema = c
        else:
            ema = alpha * c + (1 - alpha) * ema
        if i >= period - 1:
            result[i] = ema
    return result


# ---------------------------------------------------------------------------
# Per-symbol backtest
# ---------------------------------------------------------------------------
def _backtest_symbol(
    symbol: str,
    notional: float,
    range_mult: float,
    vol_mult: float,
    tp_r: float,
    sl_r: float,
    max_bars: int,
    long_only: bool,
    daily_filter: bool,
    from_ts: int,
    to_ts: int,
) -> dict:
    bars_1h   = _load_1h(symbol)
    daily_raw = _load_daily(symbol)

    if not bars_1h:
        return {"symbol": symbol, "pnl": 0.0, "trades": [], "n": 0}

    # Precompute indicators
    atrs     = _atr(bars_1h)
    vol_smas = _vol_sma(bars_1h)

    # Build daily EMA lookup: ts_day_ms → (ema8, ema21)
    daily_closes = [c for _, c in daily_raw]
    ema8_d  = _ema_series(daily_closes, 8)
    ema21_d = _ema_series(daily_closes, 21)
    daily_ema: dict[int, tuple[float, float]] = {}
    for i, (ts, _) in enumerate(daily_raw):
        if ema8_d[i] is not None and ema21_d[i] is not None:
            daily_ema[ts] = (ema8_d[i], ema21_d[i])  # type: ignore[assignment]

    def _daily_trend(bar_ts: int) -> int:
        """1=bull, -1=bear, 0=unknown. Find most recent daily close <= bar_ts."""
        # bar_ts is 1H bar open time in ms; daily bar open is midnight UTC
        day_open = (bar_ts // 86_400_000) * 86_400_000
        # Walk back to find the latest daily bar at or before bar's day
        best_ts = -1
        best_val = 0
        for dts, (e8, e21) in daily_ema.items():
            if dts <= day_open and dts > best_ts:
                best_ts = dts
                best_val = 1 if e8 > e21 else -1
        return best_val

    total_pnl = 0.0
    trades: list[dict] = []

    # State: open position
    in_pos = False
    entry_price = 0.0
    entry_dir   = 0  # 1=long, -1=short
    entry_atr   = 0.0
    bars_in_pos = 0
    entry_ts    = 0

    for i in range(max(20, 14), len(bars_1h)):
        bar = bars_1h[i]
        if bar["ts"] < from_ts or bar["ts"] > to_ts:
            continue

        # --- Manage open position ---
        if in_pos:
            bars_in_pos += 1
            tp = entry_price + entry_dir * tp_r * entry_atr
            sl = entry_price - entry_dir * sl_r * entry_atr

            hit_tp = (entry_dir == 1  and bar["high"] >= tp) or \
                     (entry_dir == -1 and bar["low"]  <= tp)
            hit_sl = (entry_dir == 1  and bar["low"]  <= sl) or \
                     (entry_dir == -1 and bar["high"] >= sl)
            time_stop = bars_in_pos >= max_bars

            exit_price = None
            exit_reason = ""
            if hit_sl and not hit_tp:
                exit_price = sl
                exit_reason = "SL"
            elif hit_tp:
                exit_price = tp
                exit_reason = "TP"
            elif hit_sl and hit_tp:
                # Both triggered same bar — assume SL hit first (conservative)
                exit_price = sl
                exit_reason = "SL"
            elif time_stop:
                exit_price = bar["close"]
                exit_reason = "TIME"

            if exit_price is not None:
                raw_pnl = (exit_price - entry_price) / entry_price * entry_dir * notional
                fee = 2 * _TAKER_FEE * notional
                net = raw_pnl - fee
                total_pnl += net
                trades.append({
                    "ts_in": entry_ts, "ts_out": bar["ts"],
                    "dir": entry_dir, "entry": entry_price,
                    "exit": exit_price, "net": net, "reason": exit_reason,
                })
                in_pos = False

        # --- Look for new entry on THIS bar's close ---
        if in_pos:
            continue  # only one position at a time

        atr_val  = atrs[i]
        vol_sma  = vol_smas[i]
        if atr_val is None or vol_sma is None or vol_sma == 0:
            continue

        bar_range = bar["high"] - bar["low"]
        if bar_range < range_mult * atr_val:
            continue
        if bar["volume"] < vol_mult * vol_sma:
            continue

        bar_up = bar["close"] > bar["open"]
        signal_dir = 1 if bar_up else -1

        if long_only and signal_dir == -1:
            continue

        if daily_filter:
            trend = _daily_trend(bar["ts"])
            if trend != 0 and trend != signal_dir:
                continue

        # Entry on next bar's open (simulate 1-bar execution lag)
        if i + 1 >= len(bars_1h):
            continue
        next_bar = bars_1h[i + 1]

        in_pos      = True
        entry_price = next_bar["open"]
        entry_dir   = signal_dir
        entry_atr   = atr_val
        bars_in_pos = 0
        entry_ts    = next_bar["ts"]

    return {"symbol": symbol, "pnl": total_pnl, "trades": trades, "n": len(trades)}


# ---------------------------------------------------------------------------
# Portfolio backtest
# ---------------------------------------------------------------------------
def run(
    symbols: list[str],
    capital: float,
    range_mult: float,
    vol_mult: float,
    tp_r: float,
    sl_r: float,
    max_bars: int,
    long_only: bool,
    daily_filter: bool,
    from_ts: int,
    to_ts: int,
) -> None:
    notional = capital / len(symbols)
    print(f"Mechanism #13: 1H Volume-Momentum Breakout")
    print(f"  range_mult={range_mult}, vol_mult={vol_mult}, TP={tp_r}R, SL={sl_r}R, "
          f"max_bars={max_bars}, long_only={long_only}, daily_filter={daily_filter}")
    print(f"  Capital ${capital:,.0f}  |  {notional:,.0f}/coin  |  {len(symbols)} coins")
    print()

    all_trades: list[dict] = []
    per_symbol: list[tuple[str, float, int]] = []

    for sym in symbols:
        res = _backtest_symbol(
            sym, notional, range_mult, vol_mult, tp_r, sl_r,
            max_bars, long_only, daily_filter, from_ts, to_ts,
        )
        per_symbol.append((sym, res["pnl"], res["n"]))
        all_trades.extend(res["trades"])
        print(f"  {sym:<12}  pnl=${res['pnl']:>+10,.0f}  trades={res['n']}")

    all_trades.sort(key=lambda t: t["ts_in"])

    # Equity curve + drawdown
    equity = capital
    peak   = capital
    max_dd = 0.0
    wins = losses = 0
    gross_win = gross_loss = 0.0
    year_pnl: dict[int, float] = {}

    for t in all_trades:
        equity += t["net"]
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
        if t["net"] > 0:
            wins += 1
            gross_win += t["net"]
        else:
            losses += 1
            gross_loss += abs(t["net"])
        yr = datetime.fromtimestamp(t["ts_in"] / 1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0.0) + t["net"]

    n = len(all_trades)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    win_rate = wins / n * 100 if n > 0 else 0
    net_pnl = equity - capital

    # IS/OOS split at 2023-01-01
    oos_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    is_trades  = [t for t in all_trades if t["ts_in"] <  oos_ts]
    oos_trades = [t for t in all_trades if t["ts_in"] >= oos_ts]
    def _pf(ts):
        gw = sum(t["net"] for t in ts if t["net"] > 0)
        gl = sum(abs(t["net"]) for t in ts if t["net"] < 0)
        return gw / gl if gl > 0 else float("inf")
    is_pf  = _pf(is_trades)
    oos_pf = _pf(oos_trades)

    cagr_years = (to_ts - from_ts) / (1_000 * 86_400 * 365.25)
    cagr = (equity / capital) ** (1 / cagr_years) - 1 if cagr_years > 0 and equity > 0 else -1.0

    print()
    print("=" * 60)
    print(f"  Trades     : {n}  (wins={wins}, losses={losses}, WR={win_rate:.1f}%)")
    print(f"  Gross win  : ${gross_win:,.0f}")
    print(f"  Gross loss : ${gross_loss:,.0f}")
    print(f"  Profit fac : {pf:.2f}")
    print(f"  Net PnL    : ${net_pnl:+,.0f}  ({net_pnl/capital*100:.1f}%)")
    print(f"  Final eq   : ${equity:,.0f}")
    print(f"  CAGR       : {cagr*100:.1f}%")
    print(f"  Max DD     : {max_dd*100:.1f}%")
    print(f"  IS  PF (<2023)  : {is_pf:.2f}  ({len(is_trades)} trades)")
    print(f"  OOS PF (2023+)  : {oos_pf:.2f}  ({len(oos_trades)} trades)")
    print()
    print("Year-by-year:")
    for yr in sorted(year_pnl):
        print(f"  {yr}: ${year_pnl[yr]:>+10,.0f}")

    print()
    print("Per-symbol summary:")
    for sym, pnl, nt in sorted(per_symbol, key=lambda x: -x[1]):
        print(f"  {sym:<12}  ${pnl:>+10,.0f}  ({nt} trades)")

    # Verdict
    print()
    if pf >= 1.30 and max_dd <= 0.40 and n >= 60 and oos_pf >= 0.85 * is_pf and cagr >= 0.15:
        verdict = "✓ GO"
    elif pf >= 1.10 and max_dd <= 0.50 and n >= 60:
        verdict = "MARGINAL"
    else:
        verdict = "KILL"
    print(f"  VERDICT: {verdict}  (PF={pf:.2f}, CAGR={cagr*100:.1f}%, DD={max_dd*100:.1f}%, "
          f"n={n}, IS_PF={is_pf:.2f}, OOS_PF={oos_pf:.2f})")

    # Also test variant: short-only (useful to know)
    print()
    print("Exit reason breakdown:")
    reasons = {}
    for t in all_trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        pnl_r = sum(t["net"] for t in all_trades if t["reason"] == r)
        print(f"  {r:<8}  {cnt:>5} trades  ${pnl_r:>+10,.0f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",      nargs="+", default=_SYMBOLS)
    parser.add_argument("--capital",      type=float, default=100_000.0)
    parser.add_argument("--range-mult",   type=float, default=2.0)
    parser.add_argument("--vol-mult",     type=float, default=2.0)
    parser.add_argument("--tp-r",         type=float, default=2.0)
    parser.add_argument("--sl-r",         type=float, default=1.0)
    parser.add_argument("--max-bars",     type=int,   default=4)
    parser.add_argument("--no-long-only", action="store_true")
    parser.add_argument("--no-daily-filter", action="store_true")
    parser.add_argument("--from",         dest="from_date", default="2021-01-01")
    parser.add_argument("--to",           dest="to_date",   default=None)
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (
        datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.to_date else datetime.now(timezone.utc)
    )

    run(
        symbols      = args.symbols,
        capital      = args.capital,
        range_mult   = args.range_mult,
        vol_mult     = args.vol_mult,
        tp_r         = args.tp_r,
        sl_r         = args.sl_r,
        max_bars     = args.max_bars,
        long_only    = not args.no_long_only,
        daily_filter = not args.no_daily_filter,
        from_ts      = int(from_dt.timestamp() * 1000),
        to_ts        = int(to_dt.timestamp() * 1000),
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Variant B: Near-high breakout (price within N% of 20-bar high/low)
# ---------------------------------------------------------------------------
def run_variant_b(
    symbols: list[str],
    capital: float,
    range_mult: float,
    vol_mult: float,
    tp_r: float,
    sl_r: float,
    max_bars: int,
    proximity_pct: float,  # how close to N-bar high/low (0.02 = within 2%)
    lookback: int,         # bars to define the high/low level
    from_ts: int,
    to_ts: int,
) -> None:
    """Near-high breakout: only trade when price is near the recent range extreme."""
    notional = capital / len(symbols)
    print(f"\nVariant B: Near-high breakout filter")
    print(f"  range_mult={range_mult}, vol_mult={vol_mult}, TP={tp_r}R, SL={sl_r}R, "
          f"max_bars={max_bars}, proximity={proximity_pct*100:.0f}%, lookback={lookback}")

    all_trades: list[dict] = []
    per_symbol: list[tuple[str, float, int]] = []

    for symbol in symbols:
        bars_1h = _load_1h(symbol)
        daily_raw = _load_daily(symbol)
        if not bars_1h:
            per_symbol.append((symbol, 0.0, 0))
            continue

        atrs     = _atr(bars_1h)
        vol_smas = _vol_sma(bars_1h)

        daily_closes = [c for _, c in daily_raw]
        ema8_d  = _ema_series(daily_closes, 8)
        ema21_d = _ema_series(daily_closes, 21)
        daily_ema: dict[int, tuple[float, float]] = {}
        for i, (ts, _) in enumerate(daily_raw):
            if ema8_d[i] is not None and ema21_d[i] is not None:
                daily_ema[ts] = (ema8_d[i], ema21_d[i])  # type: ignore[assignment]

        def _daily_trend(bar_ts: int) -> int:
            day_open = (bar_ts // 86_400_000) * 86_400_000
            best_ts = -1; best_val = 0
            for dts, (e8, e21) in daily_ema.items():
                if dts <= day_open and dts > best_ts:
                    best_ts = dts; best_val = 1 if e8 > e21 else -1
            return best_val

        total_pnl = 0.0
        sym_trades: list[dict] = []
        in_pos = False; entry_price = 0.0; entry_dir = 0
        entry_atr = 0.0; bars_in_pos = 0; entry_ts = 0

        for i in range(max(lookback, 20), len(bars_1h)):
            bar = bars_1h[i]
            if bar["ts"] < from_ts or bar["ts"] > to_ts:
                continue

            if in_pos:
                bars_in_pos += 1
                tp = entry_price + entry_dir * tp_r * entry_atr
                sl = entry_price - entry_dir * sl_r * entry_atr
                hit_tp = (entry_dir == 1 and bar["high"] >= tp) or (entry_dir == -1 and bar["low"] <= tp)
                hit_sl = (entry_dir == 1 and bar["low"]  <= sl) or (entry_dir == -1 and bar["high"] >= sl)
                time_stop = bars_in_pos >= max_bars
                exit_price = None; exit_reason = ""
                if hit_sl and not hit_tp: exit_price = sl; exit_reason = "SL"
                elif hit_tp: exit_price = tp; exit_reason = "TP"
                elif hit_sl and hit_tp: exit_price = sl; exit_reason = "SL"
                elif time_stop: exit_price = bar["close"]; exit_reason = "TIME"
                if exit_price is not None:
                    raw_pnl = (exit_price - entry_price) / entry_price * entry_dir * notional
                    fee = 2 * _TAKER_FEE * notional
                    net = raw_pnl - fee
                    total_pnl += net
                    sym_trades.append({"ts_in": entry_ts, "ts_out": bar["ts"], "dir": entry_dir,
                                        "entry": entry_price, "exit": exit_price, "net": net, "reason": exit_reason})
                    in_pos = False

            if in_pos:
                continue

            atr_val = atrs[i]; vol_sma = vol_smas[i]
            if atr_val is None or vol_sma is None or vol_sma == 0:
                continue
            if (bar["high"] - bar["low"]) < range_mult * atr_val:
                continue
            if bar["volume"] < vol_mult * vol_sma:
                continue

            bar_up = bar["close"] > bar["open"]
            signal_dir = 1 if bar_up else -1

            # Daily trend filter (long-only in uptrend)
            trend = _daily_trend(bar["ts"])
            if trend != 0 and trend != signal_dir:
                continue
            if signal_dir == -1:
                continue  # long-only

            # Near-high filter: close must be within proximity_pct of lookback high
            recent_high = max(b["high"] for b in bars_1h[i-lookback:i])
            if bar["close"] < recent_high * (1 - proximity_pct):
                continue

            if i + 1 >= len(bars_1h):
                continue
            next_bar = bars_1h[i + 1]
            in_pos = True; entry_price = next_bar["open"]; entry_dir = signal_dir
            entry_atr = atr_val; bars_in_pos = 0; entry_ts = next_bar["ts"]

        per_symbol.append((symbol, total_pnl, len(sym_trades)))
        all_trades.extend(sym_trades)
        print(f"  {symbol:<12}  pnl=${total_pnl:>+10,.0f}  trades={len(sym_trades)}")

    all_trades.sort(key=lambda t: t["ts_in"])
    equity = capital; peak = capital; max_dd = 0.0
    wins = losses = 0; gross_win = gross_loss = 0.0; year_pnl: dict[int, float] = {}
    for t in all_trades:
        equity += t["net"]
        if equity > peak: peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd: max_dd = dd
        if t["net"] > 0: wins += 1; gross_win += t["net"]
        else: losses += 1; gross_loss += abs(t["net"])
        yr = datetime.fromtimestamp(t["ts_in"] / 1000, tz=timezone.utc).year
        year_pnl[yr] = year_pnl.get(yr, 0.0) + t["net"]
    n = len(all_trades)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    win_rate = wins / n * 100 if n > 0 else 0
    cagr_years = (to_ts - from_ts) / (1_000 * 86_400 * 365.25)
    cagr = (equity / capital) ** (1 / cagr_years) - 1 if cagr_years > 0 and equity > 0 else -1.0
    oos_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    def _pf(ts): gw = sum(t["net"] for t in ts if t["net"] > 0); gl = sum(abs(t["net"]) for t in ts if t["net"] < 0); return gw/gl if gl > 0 else float("inf")
    is_pf = _pf([t for t in all_trades if t["ts_in"] < oos_ts])
    oos_pf = _pf([t for t in all_trades if t["ts_in"] >= oos_ts])
    print(f"\n  Trades={n}, WR={win_rate:.1f}%, PF={pf:.2f}, CAGR={cagr*100:.1f}%, DD={max_dd*100:.1f}%")
    print(f"  IS PF={is_pf:.2f}, OOS PF={oos_pf:.2f}")
    for yr in sorted(year_pnl): print(f"  {yr}: ${year_pnl[yr]:>+10,.0f}")
    if pf >= 1.30 and max_dd <= 0.40 and n >= 60 and oos_pf >= 0.85 * is_pf and cagr >= 0.15:
        verdict = "✓ GO"
    elif pf >= 1.10 and max_dd <= 0.50 and n >= 60:
        verdict = "MARGINAL"
    else:
        verdict = "KILL"
    print(f"  VERDICT: {verdict}")
