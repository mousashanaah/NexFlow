#!/usr/bin/env python3
"""Intraday / short-hold candidate strategies for mechanism #13.

Strategy A: Asian Session Range Breakout
  - Range: highest high and lowest low of 00:00–07:59 UTC bars (8 bars)
  - Entry: on break above range high (long) or below range low (short) at 08:00+
  - TP: entry ± 1.5 × range_width
  - SL: entry ∓ 0.5 × range_width  (3:1 R:R)
  - Time exit: close of 19:00 UTC bar (end of London/NY session)
  - Long-only + daily EMA(8/21) filter

Strategy B: 4H EMA 5/13 Long-Only
  - Same mechanism as daily EMA 8/21 but on 4H candles
  - EMA(5) > EMA(13) → hold long; cross below → exit
  - Long-only, daily filter via daily EMA(8/21)
  - Hold duration: days (4-20 bars = 16-80H)

Universe: 12 Bitget USDT-perp coins
Fee: 0.06% taker each side
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
_DAY_MS  = 86_400_000
_HOUR_MS = 3_600_000


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_1h(symbol: str) -> list[dict]:
    path = _CANDLE_DIR / f"{symbol}_1H.parquet"
    if not path.exists():
        return []
    tbl = pq.read_table(path)
    cols = tbl.schema.names
    vol_col = "volume" if "volume" in cols else ("quote_volume" if "quote_volume" in cols else None)
    ots = tbl.column("open_time").to_pylist()
    opens  = tbl.column("open").to_pylist()
    highs  = tbl.column("high").to_pylist()
    lows   = tbl.column("low").to_pylist()
    closes = tbl.column("close").to_pylist()
    vols   = tbl.column(vol_col).to_pylist() if vol_col else [0.0] * tbl.num_rows
    rows = [
        {"ts": int(ots[i]), "open": float(opens[i]), "high": float(highs[i]),
         "low": float(lows[i]), "close": float(closes[i]), "volume": float(vols[i])}
        for i in range(tbl.num_rows)
    ]
    rows.sort(key=lambda x: x["ts"])
    return rows


def _load_4h(symbol: str) -> list[dict]:
    """Load 4H bars — resample from 1H cache if no 4H file exists."""
    path_4h = _CANDLE_DIR / f"{symbol}_4H.parquet"
    if path_4h.exists():
        tbl = pq.read_table(path_4h)
        rows = []
        for i in range(tbl.num_rows):
            rows.append({
                "ts":    int(tbl.column("open_time")[i].as_py()),
                "open":  float(tbl.column("open")[i].as_py()),
                "high":  float(tbl.column("high")[i].as_py()),
                "low":   float(tbl.column("low")[i].as_py()),
                "close": float(tbl.column("close")[i].as_py()),
            })
        rows.sort(key=lambda x: x["ts"])
        return rows

    # Resample 1H → 4H (each 4H bar starts at 00,04,08,12,16,20 UTC)
    bars_1h = _load_1h(symbol)
    if not bars_1h:
        return []
    buckets: dict[int, list[dict]] = {}
    for b in bars_1h:
        # bucket = floor to nearest 4H boundary
        hour_of_day = (b["ts"] % _DAY_MS) // _HOUR_MS
        boundary_hour = (hour_of_day // 4) * 4
        bucket_ts = (b["ts"] // _DAY_MS) * _DAY_MS + boundary_hour * _HOUR_MS
        buckets.setdefault(bucket_ts, []).append(b)
    bars_4h = []
    for bucket_ts in sorted(buckets):
        grp = buckets[bucket_ts]
        bars_4h.append({
            "ts":    bucket_ts,
            "open":  grp[0]["open"],
            "high":  max(b["high"] for b in grp),
            "low":   min(b["low"]  for b in grp),
            "close": grp[-1]["close"],
        })
    return bars_4h


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
def _ema_series(closes: list[float], period: int) -> list[Optional[float]]:
    alpha = 2.0 / (period + 1)
    result: list[Optional[float]] = [None] * len(closes)
    ema = None
    for i, c in enumerate(closes):
        ema = alpha * c + (1 - alpha) * ema if ema is not None else c
        if i >= period - 1:
            result[i] = ema
    return result


def _build_daily_trend(daily_raw: list[tuple[int, float]]) -> dict[int, int]:
    """Returns day_open_ms → trend (1=bull, -1=bear)."""
    closes = [c for _, c in daily_raw]
    e8  = _ema_series(closes, 8)
    e21 = _ema_series(closes, 21)
    out: dict[int, int] = {}
    for i, (ts, _) in enumerate(daily_raw):
        if e8[i] is not None and e21[i] is not None:
            out[ts] = 1 if e8[i] > e21[i] else -1
    return out


def _daily_trend_at(trend_map: dict[int, int], bar_ts_ms: int) -> int:
    day_open = (bar_ts_ms // _DAY_MS) * _DAY_MS
    best_ts = -1; best_val = 0
    for dts, val in trend_map.items():
        if dts <= day_open and dts > best_ts:
            best_ts = dts; best_val = val
    return best_val


# ---------------------------------------------------------------------------
# Strategy A: Asian session range breakout
# ---------------------------------------------------------------------------
def _asian_breakout_symbol(
    symbol: str,
    notional: float,
    tp_mult: float,
    sl_mult: float,
    from_ts: int,
    to_ts: int,
    trend_map: dict[int, int],
) -> tuple[float, list[dict]]:
    bars = _load_1h(symbol)
    if not bars:
        return 0.0, []

    # Group bars by calendar day (UTC)
    days: dict[int, list[dict]] = {}
    for b in bars:
        day = (b["ts"] // _DAY_MS) * _DAY_MS
        days.setdefault(day, []).append(b)

    total_pnl = 0.0
    trades: list[dict] = []

    for day_ts in sorted(days):
        if day_ts < from_ts or day_ts > to_ts:
            continue
        day_bars = sorted(days[day_ts], key=lambda x: x["ts"])

        # Asian range: bars with open_time in [00:00, 08:00) UTC
        asian = [b for b in day_bars if b["ts"] < day_ts + 8 * _HOUR_MS]
        if len(asian) < 4:
            continue
        range_high = max(b["high"] for b in asian)
        range_low  = min(b["low"]  for b in asian)
        range_width = range_high - range_low
        if range_width <= 0:
            continue

        # London/NY bars: 08:00–19:59 UTC
        london = [b for b in day_bars
                  if day_ts + 8 * _HOUR_MS <= b["ts"] < day_ts + 20 * _HOUR_MS]
        if not london:
            continue

        # Daily trend filter
        trend = _daily_trend_at(trend_map, day_ts)

        in_pos = False
        entry_price = 0.0; entry_dir = 0
        tp_price = 0.0; sl_price = 0.0
        entry_ts_val = 0

        for b in london:
            if in_pos:
                hit_tp = (entry_dir == 1 and b["high"] >= tp_price) or \
                         (entry_dir == -1 and b["low"]  <= tp_price)
                hit_sl = (entry_dir == 1 and b["low"]  <= sl_price) or \
                         (entry_dir == -1 and b["high"] >= sl_price)
                time_stop = (b["ts"] >= day_ts + 19 * _HOUR_MS)

                exit_price = None
                if hit_sl and not hit_tp: exit_price = sl_price; reason = "SL"
                elif hit_tp: exit_price = tp_price; reason = "TP"
                elif hit_sl and hit_tp: exit_price = sl_price; reason = "SL"
                elif time_stop: exit_price = b["close"]; reason = "TIME"

                if exit_price is not None:
                    raw_pnl = (exit_price - entry_price) / entry_price * entry_dir * notional
                    net = raw_pnl - 2 * _TAKER_FEE * notional
                    total_pnl += net
                    trades.append({"ts_in": entry_ts_val, "ts_out": b["ts"],
                                   "net": net, "reason": reason})
                    in_pos = False
                continue

            # Look for breakout
            breaks_high = b["high"] > range_high and b["close"] > range_high
            breaks_low  = b["low"]  < range_low  and b["close"] < range_low

            if breaks_high and (trend == 0 or trend == 1):
                # Enter long at close of breakout bar
                entry_price = b["close"]
                entry_dir   = 1
                tp_price    = entry_price + tp_mult * range_width
                sl_price    = entry_price - sl_mult * range_width
                entry_ts_val = b["ts"]
                in_pos = True
            elif breaks_low and (trend == 0 or trend == -1):
                # Long-only: skip short entries
                pass  # entry_dir = -1 would go here for L+S version

    return total_pnl, trades


def run_asian_breakout(
    symbols: list[str],
    capital: float,
    tp_mult: float,
    sl_mult: float,
    from_ts: int,
    to_ts: int,
) -> None:
    notional = capital / len(symbols)
    print("=" * 60)
    print(f"Strategy A: Asian Session Range Breakout (LONG-ONLY)")
    print(f"  TP={tp_mult}×range  SL={sl_mult}×range  time-exit=19:00UTC")
    print(f"  Capital ${capital:,.0f}  |  ${notional:,.0f}/coin  |  daily EMA filter")
    print()

    all_trades: list[dict] = []
    per_symbol: list[tuple[str, float, int]] = []

    for sym in symbols:
        daily_raw = _load_daily(sym)
        trend_map = _build_daily_trend(daily_raw)
        pnl, trades = _asian_breakout_symbol(sym, notional, tp_mult, sl_mult,
                                              from_ts, to_ts, trend_map)
        per_symbol.append((sym, pnl, len(trades)))
        all_trades.extend(trades)
        print(f"  {sym:<12}  pnl=${pnl:>+10,.0f}  trades={len(trades)}")

    _report(all_trades, capital, from_ts, to_ts, per_symbol, label="Asian Breakout")


# ---------------------------------------------------------------------------
# Strategy B: 4H EMA 5/13 Long-Only
# ---------------------------------------------------------------------------
def _4h_ema_symbol(
    symbol: str,
    notional: float,
    fast: int,
    slow: int,
    from_ts: int,
    to_ts: int,
    trend_map: dict[int, int],
    use_daily_filter: bool,
) -> tuple[float, list[dict]]:
    bars = _load_4h(symbol)
    if not bars:
        return 0.0, []

    closes = [b["close"] for b in bars]
    ema_fast = _ema_series(closes, fast)
    ema_slow = _ema_series(closes, slow)

    total_pnl = 0.0
    trades: list[dict] = []
    prev_above: Optional[bool] = None
    in_pos = False
    entry_price = 0.0
    entry_ts_val = 0

    for i, b in enumerate(bars):
        if b["ts"] < from_ts or b["ts"] > to_ts:
            if ema_fast[i] is not None and ema_slow[i] is not None:
                prev_above = ema_fast[i] > ema_slow[i]
            continue

        if ema_fast[i] is None or ema_slow[i] is None:
            continue

        above = ema_fast[i] > ema_slow[i]

        if prev_above is not None and above != prev_above:
            if above:
                # Bullish cross
                trend = _daily_trend_at(trend_map, b["ts"]) if use_daily_filter else 1
                if (trend == 0 or trend == 1) and not in_pos:
                    entry_price = b["close"]
                    entry_ts_val = b["ts"]
                    in_pos = True
                    fee = _TAKER_FEE * notional
                    total_pnl -= fee
            else:
                # Bearish cross — exit long
                if in_pos:
                    pnl = (b["close"] - entry_price) / entry_price * notional
                    fee = _TAKER_FEE * notional
                    net = pnl - fee
                    total_pnl += net
                    trades.append({"ts_in": entry_ts_val, "ts_out": b["ts"],
                                   "net": net + fee,  # store gross so we can recalculate
                                   "entry": entry_price, "exit": b["close"]})
                    # fix: store actual net
                    trades[-1]["net"] = net
                    in_pos = False

        prev_above = above

    # Close any open position at last bar
    if in_pos and bars:
        last = bars[-1]
        if from_ts <= last["ts"] <= to_ts:
            pnl = (last["close"] - entry_price) / entry_price * notional
            fee = _TAKER_FEE * notional
            net = pnl - fee
            total_pnl += net
            trades.append({"ts_in": entry_ts_val, "ts_out": last["ts"],
                           "net": net, "entry": entry_price, "exit": last["close"]})

    return total_pnl, trades


def run_4h_ema(
    symbols: list[str],
    capital: float,
    fast: int,
    slow: int,
    use_daily_filter: bool,
    from_ts: int,
    to_ts: int,
) -> None:
    notional = capital / len(symbols)
    print("=" * 60)
    print(f"Strategy B: 4H EMA {fast}/{slow} Long-Only")
    print(f"  Capital ${capital:,.0f}  |  ${notional:,.0f}/coin  |  "
          f"daily_filter={use_daily_filter}")
    print()

    all_trades: list[dict] = []
    per_symbol: list[tuple[str, float, int]] = []

    for sym in symbols:
        daily_raw = _load_daily(sym)
        trend_map = _build_daily_trend(daily_raw) if use_daily_filter else {}
        pnl, trades = _4h_ema_symbol(sym, notional, fast, slow,
                                      from_ts, to_ts, trend_map, use_daily_filter)
        per_symbol.append((sym, pnl, len(trades)))
        all_trades.extend(trades)
        print(f"  {sym:<12}  pnl=${pnl:>+10,.0f}  trades={len(trades)}")

    _report(all_trades, capital, from_ts, to_ts, per_symbol, label=f"4H EMA {fast}/{slow}")


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------
def _report(
    all_trades: list[dict],
    capital: float,
    from_ts: int,
    to_ts: int,
    per_symbol: list[tuple[str, float, int]],
    label: str,
) -> None:
    all_trades.sort(key=lambda t: t["ts_in"])
    equity = capital; peak = capital; max_dd = 0.0
    wins = losses = 0; gross_win = gross_loss = 0.0
    year_pnl: dict[int, float] = {}

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
    net_pnl = equity - capital
    cagr_years = (to_ts - from_ts) / (1_000 * 86_400 * 365.25)
    cagr = (equity / capital) ** (1 / cagr_years) - 1 if cagr_years > 0 and equity > 0 else -1.0

    oos_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    def _pf(ts):
        gw = sum(t["net"] for t in ts if t["net"] > 0)
        gl = sum(abs(t["net"]) for t in ts if t["net"] < 0)
        return gw / gl if gl > 0 else float("inf")
    is_pf  = _pf([t for t in all_trades if t["ts_in"] <  oos_ts])
    oos_pf = _pf([t for t in all_trades if t["ts_in"] >= oos_ts])

    print()
    print(f"--- {label} Results ---")
    print(f"  Trades     : {n}  (WR={win_rate:.1f}%,  wins={wins}, losses={losses})")
    print(f"  Profit fac : {pf:.2f}")
    print(f"  Net PnL    : ${net_pnl:+,.0f}  ({net_pnl/capital*100:.1f}%)")
    print(f"  Final eq   : ${equity:,.0f}")
    print(f"  CAGR       : {cagr*100:.1f}%")
    print(f"  Max DD     : {max_dd*100:.1f}%")
    print(f"  IS  PF (<2023) : {is_pf:.2f}  ({len([t for t in all_trades if t['ts_in']<oos_ts])} trades)")
    print(f"  OOS PF (2023+) : {oos_pf:.2f}  ({len([t for t in all_trades if t['ts_in']>=oos_ts])} trades)")
    print()
    print("  Year-by-year:")
    for yr in sorted(year_pnl):
        print(f"    {yr}: ${year_pnl[yr]:>+10,.0f}")
    print()
    print("  Per-symbol (sorted by PnL):")
    for sym, pnl, nt in sorted(per_symbol, key=lambda x: -x[1]):
        print(f"    {sym:<12}  ${pnl:>+10,.0f}  ({nt} trades)")

    if pf >= 1.30 and max_dd <= 0.40 and n >= 60 and cagr >= 0.15:
        verdict = "✓ GO"
    elif pf >= 1.10 and max_dd <= 0.50 and n >= 60:
        verdict = "MARGINAL"
    else:
        verdict = "KILL"
    print(f"\n  VERDICT: {verdict}  (PF={pf:.2f}, CAGR={cagr*100:.1f}%, DD={max_dd*100:.1f}%, n={n})")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",    nargs="+", default=_SYMBOLS)
    parser.add_argument("--capital",    type=float, default=100_000.0)
    parser.add_argument("--from",       dest="from_date", default="2021-01-01")
    parser.add_argument("--to",         dest="to_date",   default=None)
    parser.add_argument("--strategy",   choices=["A", "B", "both"], default="both")
    # Strategy A params
    parser.add_argument("--tp-mult",    type=float, default=1.5)
    parser.add_argument("--sl-mult",    type=float, default=0.5)
    # Strategy B params
    parser.add_argument("--fast",       type=int,   default=5)
    parser.add_argument("--slow",       type=int,   default=13)
    parser.add_argument("--no-daily-filter", action="store_true")
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = (
        datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.to_date else datetime.now(timezone.utc)
    )
    from_ts = int(from_dt.timestamp() * 1000)
    to_ts   = int(to_dt.timestamp() * 1000)

    if args.strategy in ("A", "both"):
        run_asian_breakout(args.symbols, args.capital, args.tp_mult, args.sl_mult,
                           from_ts, to_ts)

    if args.strategy in ("B", "both"):
        run_4h_ema(args.symbols, args.capital, args.fast, args.slow,
                   not args.no_daily_filter, from_ts, to_ts)

        # Also test without daily filter for comparison
        if args.strategy == "B":
            print("\n--- Comparison: without daily EMA filter ---")
            run_4h_ema(args.symbols, args.capital, args.fast, args.slow,
                       False, from_ts, to_ts)


if __name__ == "__main__":
    main()
