#!/usr/bin/env python3
"""Regime robustness study.

Runs the exact current strategy month-by-month across all available
historical data and reports per-month metrics.

Goal: determine whether the original 57% WR period was representative
or an outlier, and whether the strategy edge is regime-persistent.

Data source:
  Priority 1 — parquet candle files from the live candle engine
               (have real buy_volume + spread_avg)
  Priority 2 — Bitget REST API, cached locally on first fetch
               (buy_volume approximated from bar position;
                spread_avg = 0.0 → spread filter always passes)

The same approximation applies to ALL months uniformly, so the
month-to-month comparison is valid even though absolute numbers
may diverge slightly from live execution.

Usage:
  python scripts/regime_robustness.py
  python scripts/regime_robustness.py --symbol ETHUSDT --start 2023-01
  python scripts/regime_robustness.py --candle-dir data/candles
  python scripts/regime_robustness.py --no-rest           (parquet only)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.backtest_runner import BacktestConfig, BacktestRunner
from nexflow.services.strategy.momentum_strategy import MomentumConfig, MomentumStrategy
from nexflow.services.strategy.paper_execution import ExecutionConfig
from nexflow.services.strategy.risk_engine import RiskConfig

_W = 78

# Strategy warmup: need at least 22 1m bars + 7 5m bars warm.
# Use 400 1m bars (6h40m) of pre-month data to be safe.
_WARMUP_BARS = 400


# ---------------------------------------------------------------------------
# Raw bar (lightweight for caching)
# ---------------------------------------------------------------------------

@dataclass
class _Bar:
    ts_s: int      # epoch seconds = close_time
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# Local cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, symbol: str, year: int, month: int) -> Path:
    return cache_dir / f"{symbol}_1m_{year:04d}{month:02d}.csv"


def _save_cache(path: Path, bars: list[_Bar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_s", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.ts_s, b.open, b.high, b.low, b.close, b.volume])


def _load_cache(path: Path) -> list[_Bar] | None:
    if not path.exists():
        return None
    bars: list[_Bar] = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bars.append(_Bar(
                ts_s=int(row["ts_s"]),
                open=float(row["open"]),  high=float(row["high"]),
                low=float(row["low"]),   close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    return bars if bars else None


# ---------------------------------------------------------------------------
# REST fetch
# ---------------------------------------------------------------------------

def _fetch_rest_month(
    symbol: str, year: int, month: int, cache_dir: Path
) -> list[_Bar]:
    """Fetch one month of 1m bars. Returns cached data if available."""
    path = _cache_path(cache_dir, symbol, year, month)
    cached = _load_cache(path)
    if cached is not None:
        return cached

    # Compute start/end epoch ms for the month
    from calendar import monthrange
    days_in_month = monthrange(year, month)[1]
    start_dt = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_dt   = datetime(year, month, days_in_month, 23, 59, 59, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    bars: list[_Bar] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + 999 * 60_000, end_ms)
        url = (
            f"https://api.bitget.com/api/v2/mix/market/candles"
            f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1m"
            f"&startTime={cursor}&endTime={chunk_end}&limit=1000"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "NexFlow/1.0", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            if data.get("code") != "00000":
                break
            rows = data.get("data", [])
            if not rows:
                break
            for r in rows:
                ts_ms = int(r[0])
                bars.append(_Bar(
                    ts_s=ts_ms // 1000 + 60,   # close_time = open + 60s
                    open=float(r[1]),  high=float(r[2]),
                    low=float(r[3]),   close=float(r[4]),
                    volume=float(r[5]) if len(r) > 5 else 0.0,
                ))
            cursor = int(rows[-1][0]) + 60_000
            time.sleep(0.15)
        except Exception as exc:
            print(f"    [WARN] REST fetch error: {exc}")
            break

    bars.sort(key=lambda b: b.ts_s)
    if bars:
        _save_cache(path, bars)
    return bars


# ---------------------------------------------------------------------------
# Parquet load
# ---------------------------------------------------------------------------

def _load_parquet_range(
    candle_dir: Path, symbol: str, start_s: int, end_s: int
) -> list[_Bar]:
    if not _HAS_PARQUET:
        return []
    path = candle_dir / f"{symbol}_1m.parquet"
    if not path.exists():
        return []
    rows = pq.read_table(path).to_pylist()
    bars: list[_Bar] = []
    for r in rows:
        ct = r.get("close_time", 0)
        if start_s <= ct <= end_s and r.get("is_final"):
            bars.append(_Bar(
                ts_s=ct,
                open=r["open"],  high=r["high"],
                low=r["low"],    close=r["close"],
                volume=r["volume"],
            ))
    bars.sort(key=lambda b: b.ts_s)
    return bars


# ---------------------------------------------------------------------------
# Bar → Candle conversion (with field approximation for REST data)
# ---------------------------------------------------------------------------

def _bar_to_candle(b: _Bar, symbol: str) -> Candle:
    o, h, l, c = b.open, b.high, b.low, b.close
    vol = b.volume
    # Bar-position proxy for buy_volume:
    # price near high → buy-dominated; price near low → sell-dominated
    if h > l:
        buy_frac = (c - l) / (h - l)
    else:
        buy_frac = 0.5
    buy_vol  = vol * buy_frac
    sell_vol = vol - buy_vol
    vwap = (o + h + l + c) / 4

    return Candle(
        symbol=symbol, timeframe="1m",
        open_time=b.ts_s - 60,
        close_time=b.ts_s,
        open=o, high=h, low=l, close=c,
        volume=vol,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        trade_count=0,
        vwap=vwap,
        spread_avg=0.0,      # NOT available from REST → spread filter always passes
        spread_max=0.0,
        volatility_estimate=(h - l) / o if o > 0 else 0.0,
        is_final=True,
    )


def _resample_5m(candles_1m: list[Candle], symbol: str) -> list[Candle]:
    result: list[Candle] = []
    bucket: list[Candle] = []
    for c in candles_1m:
        bucket.append(c)
        if c.close_time % 300 == 0 and bucket:
            o  = bucket[0].open
            h  = max(b.high for b in bucket)
            l  = min(b.low  for b in bucket)
            cl = bucket[-1].close
            vol  = sum(b.volume for b in bucket)
            bvol = sum(b.buy_volume for b in bucket)
            pv   = sum(b.vwap * b.volume for b in bucket)
            result.append(Candle(
                symbol=symbol, timeframe="5m",
                open_time=bucket[0].open_time,
                close_time=bucket[-1].close_time,
                open=o, high=h, low=l, close=cl,
                volume=vol, buy_volume=bvol, sell_volume=vol-bvol,
                trade_count=0, vwap=pv/vol if vol > 0 else (o+h+l+cl)/4,
                spread_avg=0.0, spread_max=0.0,
                volatility_estimate=(h-l)/o if o > 0 else 0.0,
                is_final=True,
            ))
            bucket = []
    return result


def _resample_15m(candles_1m: list[Candle], symbol: str) -> list[Candle]:
    result: list[Candle] = []
    bucket: list[Candle] = []
    for c in candles_1m:
        bucket.append(c)
        if c.close_time % 900 == 0 and bucket:
            o  = bucket[0].open
            h  = max(b.high for b in bucket)
            l  = min(b.low  for b in bucket)
            cl = bucket[-1].close
            vol  = sum(b.volume for b in bucket)
            bvol = sum(b.buy_volume for b in bucket)
            pv   = sum(b.vwap * b.volume for b in bucket)
            result.append(Candle(
                symbol=symbol, timeframe="15m",
                open_time=bucket[0].open_time,
                close_time=bucket[-1].close_time,
                open=o, high=h, low=l, close=cl,
                volume=vol, buy_volume=bvol, sell_volume=vol-bvol,
                trade_count=0, vwap=pv/vol if vol > 0 else (o+h+l+cl)/4,
                spread_avg=0.0, spread_max=0.0,
                volatility_estimate=(h-l)/o if o > 0 else 0.0,
                is_final=True,
            ))
            bucket = []
    return result


# ---------------------------------------------------------------------------
# Monthly run
# ---------------------------------------------------------------------------

@dataclass
class MonthResult:
    year: int
    month: int
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy_usd: float
    net_pnl: float
    max_drawdown: float
    total_fees: float
    n_bars: int
    data_ok: bool


def _run_month(
    symbol: str,
    all_bars: list[_Bar],       # full history sorted by ts_s
    year: int, month: int,
) -> MonthResult:
    """Slice the month (+ warmup) from all_bars and run BacktestRunner."""
    from calendar import monthrange
    days_in_month = monthrange(year, month)[1]
    month_start = int(datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    month_end   = int(datetime(year, month, days_in_month, 23, 59, 59, tzinfo=timezone.utc).timestamp())

    # Find the index of the first bar in the month
    month_bars = [b for b in all_bars if month_start <= b.ts_s <= month_end + 60]
    if not month_bars:
        return MonthResult(year, month, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False)

    # Warmup: last _WARMUP_BARS from before month_start
    pre_bars = [b for b in all_bars if b.ts_s < month_start]
    warmup   = pre_bars[-_WARMUP_BARS:] if len(pre_bars) > _WARMUP_BARS else pre_bars

    run_bars   = warmup + month_bars
    candles_1m = [_bar_to_candle(b, symbol) for b in run_bars]
    candles_5m  = _resample_5m(candles_1m, symbol)
    candles_15m = _resample_15m(candles_1m, symbol)

    strategy = MomentumStrategy()
    bt_cfg   = BacktestConfig(
        initial_equity=100_000.0,
        risk=RiskConfig(),
        execution=ExecutionConfig(),
    )
    runner  = BacktestRunner(strategy, bt_cfg)
    metrics = runner.run({symbol: {"1m": candles_1m, "5m": candles_5m, "15m": candles_15m}})

    n      = metrics.total_trades
    wins   = sum(1 for p in metrics.pnl_distribution if p > 0)
    losses = n - wins
    wr     = wins / n if n > 0 else 0.0
    gw     = sum(p for p in metrics.pnl_distribution if p > 0)
    gl     = abs(sum(p for p in metrics.pnl_distribution if p <= 0))
    pf     = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
    exp    = metrics.net_pnl / n if n > 0 else 0.0

    return MonthResult(
        year=year, month=month,
        trades=n, wins=wins, losses=losses,
        win_rate=wr, profit_factor=pf,
        expectancy_usd=exp,
        net_pnl=metrics.net_pnl,
        max_drawdown=metrics.max_drawdown,
        total_fees=metrics.total_fees,
        n_bars=len(month_bars),
        data_ok=True,
    )


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _months_in_range(start_year: int, start_month: int,
                     end_year: int, end_month: int) -> list[tuple[int, int]]:
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1
    return months


def _gather_all_bars(
    symbol: str,
    months: list[tuple[int, int]],
    candle_dir: Path,
    cache_dir: Path,
    no_rest: bool,
) -> tuple[list[_Bar], str]:
    """Load all bars for all requested months (+ prev month for warmup of first month)."""
    # Expand range by one month backwards for warmup
    first_year, first_month = months[0]
    first_month -= 1
    if first_month == 0:
        first_month = 12; first_year -= 1
    warmup_months = [(first_year, first_month)] + months

    # Try parquet first (covers full range in one file)
    if _HAS_PARQUET and candle_dir.exists():
        path = candle_dir / f"{symbol}_1m.parquet"
        if path.exists():
            last_y, last_m = months[-1]
            from calendar import monthrange
            last_end = int(datetime(last_y, last_m, monthrange(last_y, last_m)[1],
                                    23, 59, 59, tzinfo=timezone.utc).timestamp())
            first_start = int(datetime(first_year, first_month, 1, 0, 0, 0,
                                        tzinfo=timezone.utc).timestamp())
            bars = _load_parquet_range(candle_dir, symbol, first_start, last_end)
            if bars:
                return bars, "parquet (real buy_volume + spread_avg)"

    if no_rest:
        return [], "none"

    # Fetch from REST, month by month (with cache)
    print(f"  Fetching {len(warmup_months)} months from Bitget REST (cached in {cache_dir}) …")
    all_bars: list[_Bar] = []
    seen_ts: set[int] = set()
    for i, (y, m) in enumerate(warmup_months):
        label = f"{y:04d}-{m:02d}"
        print(f"    [{i+1:3d}/{len(warmup_months)}] {label}", end=" ", flush=True)
        bars = _fetch_rest_month(symbol, y, m, cache_dir)
        added = 0
        for b in bars:
            if b.ts_s not in seen_ts:
                all_bars.append(b)
                seen_ts.add(b.ts_s)
                added += 1
        cached = _load_cache(_cache_path(cache_dir, symbol, y, m))
        from_cache = cached is not None and added == len(cached or [])
        print(f"  {added:5d} bars {'[cached]' if from_cache else '[fetched]'}")

    all_bars.sort(key=lambda b: b.ts_s)
    return all_bars, "Bitget REST (buy_volume=bar-proxy, spread_avg=0)"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pf_str(pf: float) -> str:
    if pf == float("inf"):
        return "   ∞  "
    return f"{pf:6.2f}"


def _print_report(results: list[MonthResult], symbol: str, data_source: str) -> None:
    valid = [r for r in results if r.data_ok and r.trades > 0]
    all_  = [r for r in results if r.data_ok]

    print()
    print("═" * _W)
    print(f"  REGIME ROBUSTNESS STUDY — {symbol}")
    print(f"  Data source  : {data_source}")
    if valid:
        first = valid[0];  last = valid[-1]
        print(f"  Period       : {first.year:04d}-{first.month:02d}  →  {last.year:04d}-{last.month:02d}")
        print(f"  Total months : {len(all_)}  ({len(valid)} with trades)")
    print("═" * _W)
    print()

    # Per-month table
    hdr = (f"  {'Month':<10} {'Trades':>6} {'WR%':>7} {'  PF':>6} "
           f"{'Exp(USD)':>10} {'Net PnL':>10} {'MaxDD%':>7} {'Fees':>8}")
    print(hdr)
    print("  " + "─" * (_W - 2))

    for r in results:
        if not r.data_ok:
            print(f"  {r.year:04d}-{r.month:02d}    [no data]")
            continue
        if r.trades == 0:
            print(f"  {r.year:04d}-{r.month:02d}  {'—':>6} {'—':>7} {'—':>6} "
                  f"{'—':>10} {'—':>10} {'—':>7} {'—':>8}")
            continue

        label   = f"{r.year:04d}-{r.month:02d}"
        wr_mark = "◀" if r.win_rate >= 0.55 else ("▼" if r.win_rate < 0.30 else " ")
        pnl_sign = "+" if r.net_pnl >= 0 else ""
        print(
            f"  {label:<10} {r.trades:>6d}  {r.win_rate*100:>6.1f}%{wr_mark}"
            f" {_pf_str(r.profit_factor)}"
            f" {r.expectancy_usd:>+10.1f}"
            f" {pnl_sign}{r.net_pnl:>9.1f}"
            f" {r.max_drawdown*100:>6.2f}%"
            f" {r.total_fees:>8.1f}"
        )

    print("  " + "─" * (_W - 2))
    print("  ◀ = WR ≥ 55%    ▼ = WR < 30%")
    print()

    if not valid:
        print("  No trades found in any month. Check data availability.")
        return

    # Summary statistics
    wrs     = [r.win_rate for r in valid]
    pfs     = [r.profit_factor for r in valid if r.profit_factor != float("inf")]
    nets    = [r.net_pnl for r in valid]
    dds     = [r.max_drawdown for r in valid]
    exps    = [r.expectancy_usd for r in valid]

    def median(lst: list[float]) -> float:
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2

    med_wr   = median(wrs)
    med_pf   = median(pfs) if pfs else 0.0
    med_exp  = median(exps)
    prof_months = sum(1 for r in valid if r.net_pnl > 0)
    pct_prof    = prof_months / len(valid)
    best  = max(valid, key=lambda r: r.net_pnl)
    worst = min(valid, key=lambda r: r.net_pnl)

    # WR distribution
    wr_buckets = {
        "WR ≥ 55% (trending)":   sum(1 for r in valid if r.win_rate >= 0.55),
        "WR 40–55% (neutral) ":  sum(1 for r in valid if 0.40 <= r.win_rate < 0.55),
        "WR 30–40% (choppy)  ":  sum(1 for r in valid if 0.30 <= r.win_rate < 0.40),
        "WR < 30%  (hostile) ":  sum(1 for r in valid if r.win_rate < 0.30),
    }

    print("═" * _W)
    print("  SUMMARY STATISTICS")
    print("═" * _W)
    print(f"  {'Months with trades':<35}: {len(valid)}")
    print(f"  {'Profitable months':<35}: {prof_months} / {len(valid)}  ({pct_prof*100:.1f}%)")
    print()
    print(f"  {'Median win rate':<35}: {med_wr*100:.1f}%")
    print(f"  {'Median profit factor':<35}: {med_pf:.2f}")
    print(f"  {'Median expectancy (USD/trade)':<35}: {med_exp:+.1f}")
    print()
    print(f"  {'Best month':<35}: {best.year:04d}-{best.month:02d}  "
          f"Net {best.net_pnl:+.1f}  WR {best.win_rate*100:.1f}%  PF {_pf_str(best.profit_factor).strip()}")
    print(f"  {'Worst month':<35}: {worst.year:04d}-{worst.month:02d}  "
          f"Net {worst.net_pnl:+.1f}  WR {worst.win_rate*100:.1f}%  PF {_pf_str(worst.profit_factor).strip()}")
    print()

    print("  Win-rate distribution across months:")
    for label, count in wr_buckets.items():
        bar_ = "█" * count
        print(f"    {label}: {count:3d}  {bar_}")
    print()

    # Was the 57% WR period representative?
    n_above55 = wr_buckets["WR ≥ 55% (trending)"]
    n_below30 = wr_buckets["WR < 30%  (hostile) "]
    pct_trending = n_above55 / len(valid) * 100
    pct_hostile  = n_below30 / len(valid) * 100

    total_net = sum(r.net_pnl for r in valid)
    total_fees = sum(r.total_fees for r in valid)
    total_trades = sum(r.trades for r in valid)

    print("─" * _W)
    print("  FULL-PERIOD AGGREGATE")
    print("─" * _W)
    print(f"  Total trades   : {total_trades}")
    print(f"  Total net PnL  : {total_net:+.2f} USD")
    print(f"  Total fee drag : {total_fees:.2f} USD  ({total_fees/total_net*100:.1f}% of gross)" if total_net != 0 else f"  Total fee drag : {total_fees:.2f} USD")
    print()

    print("─" * _W)
    print("  VERDICT")
    print("─" * _W)
    print(f"  Months with WR ≥ 55% (trending): {n_above55}/{len(valid)} ({pct_trending:.0f}%)")
    print(f"  Months with WR <  30% (hostile) : {n_below30}/{len(valid)} ({pct_hostile:.0f}%)")
    print()

    if pct_trending < 20:
        print("  ✗  High-WR (≥55%) months are RARE — they represent < 20% of the sample.")
        print("     The original 57% WR period was likely an outlier, not representative.")
        print("     The strategy does not have a persistent edge across regimes.")
    elif pct_trending >= 40:
        print("  ✓  High-WR months occur in ≥ 40% of the sample.")
        print("     The strategy has some regime persistence, but is not universally robust.")
    else:
        print(f"  ~  High-WR months occur in {pct_trending:.0f}% of the sample.")
        print("     Edge is moderate — depends on market conditions.")

    if pct_hostile > 30:
        print(f"  ✗  {pct_hostile:.0f}% of months are hostile (WR < 30%).")
        print("     The current live period likely falls in this hostile regime cluster.")
    print()

    if med_wr < 0.40:
        print("  DIAGNOSIS: The median WR across all regimes is below 40%.")
        print("  The strategy requires trending conditions to be profitable.")
        print("  It is not a market-neutral edge — it is a trend-following strategy")
        print("  that performs well in trending markets and poorly in choppy ones.")
    elif 0.40 <= med_wr < 0.50:
        print("  DIAGNOSIS: Median WR is in the 40–50% range.")
        print("  The strategy has a modest edge but depends on a positive risk/reward ratio")
        print("  (more won per win than lost per loss) to be net profitable.")
    else:
        print("  DIAGNOSIS: Median WR is above 50%. The edge appears relatively persistent.")

    print()
    print("  Next step:")
    if pct_trending < 25:
        print("  Regime identification (not optimization) — a simple filter to detect")
        print("  trending vs ranging conditions and pause trading during hostile months")
        print("  would materially improve the overall record without changing strategy logic.")
    else:
        print("  The edge is present. Focus on execution (maker entries/exits) to reduce")
        print("  fee drag and improve the net result in moderate-WR months.")
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _write_csv(results: list[MonthResult], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "month", "trades", "wins", "losses", "win_rate_pct",
            "profit_factor", "expectancy_usd", "net_pnl", "max_drawdown_pct",
            "total_fees", "n_bars",
        ])
        w.writeheader()
        for r in results:
            if not r.data_ok:
                continue
            pf = r.profit_factor
            w.writerow({
                "month":           f"{r.year:04d}-{r.month:02d}",
                "trades":          r.trades,
                "wins":            r.wins,
                "losses":          r.losses,
                "win_rate_pct":    round(r.win_rate * 100, 2),
                "profit_factor":   round(pf, 4) if pf != float("inf") else "inf",
                "expectancy_usd":  round(r.expectancy_usd, 2),
                "net_pnl":         round(r.net_pnl, 2),
                "max_drawdown_pct": round(r.max_drawdown * 100, 3),
                "total_fees":      round(r.total_fees, 2),
                "n_bars":          r.n_bars,
            })
    print(f"  CSV written → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Regime robustness study")
    p.add_argument("--symbol",     default="ETHUSDT")
    p.add_argument("--start",      default="2023-01",
                   help="First month to analyse (YYYY-MM)")
    p.add_argument("--end",        default=None,
                   help="Last month to analyse (YYYY-MM). Default: current month")
    p.add_argument("--candle-dir", default="data/candles",
                   help="Path to parquet candle directory (preferred)")
    p.add_argument("--cache-dir",  default=".cache/candles",
                   help="Directory for caching REST responses")
    p.add_argument("--no-rest",    action="store_true",
                   help="Use parquet only; fail if not available")
    p.add_argument("--csv",        default="regime_robustness.csv")
    args = p.parse_args()

    # Parse date range
    try:
        sy, sm = [int(x) for x in args.start.split("-")]
    except ValueError:
        print("--start must be YYYY-MM"); sys.exit(1)

    if args.end:
        try:
            ey, em = [int(x) for x in args.end.split("-")]
        except ValueError:
            print("--end must be YYYY-MM"); sys.exit(1)
    else:
        now = datetime.now(tz=timezone.utc)
        ey, em = now.year, now.month

    months = _months_in_range(sy, sm, ey, em)
    print(f"\nRegime robustness study")
    print(f"  Symbol    : {args.symbol}")
    print(f"  Range     : {sy:04d}-{sm:02d}  →  {ey:04d}-{em:02d}  ({len(months)} months)")

    candle_dir = _REPO_ROOT / args.candle_dir
    cache_dir  = _REPO_ROOT / args.cache_dir

    # Load all bar data
    print(f"\nLoading candle data …")
    all_bars, data_source = _gather_all_bars(
        args.symbol, months, candle_dir, cache_dir, args.no_rest
    )
    if not all_bars:
        print("No data available. Run with internet access for REST fetch,")
        print("or provide --candle-dir with parquet files.")
        sys.exit(1)
    print(f"  {len(all_bars):,} total 1m bars loaded  [{data_source}]")

    # Data quality note
    if "REST" in data_source:
        print()
        print("  ⚠  REST candle limitations:")
        print("     buy_volume = bar-position proxy  (real trade-flow unavailable)")
        print("     spread_avg = 0.0                 (spread filter always passes)")
        print("     These biases apply uniformly to ALL months — relative comparison is valid.")

    # Run per-month backtests
    print(f"\nRunning {len(months)} monthly backtests …")
    results: list[MonthResult] = []
    for i, (y, m) in enumerate(months):
        label = f"{y:04d}-{m:02d}"
        print(f"  [{i+1:3d}/{len(months)}] {label}", end=" ", flush=True)
        r = _run_month(args.symbol, all_bars, y, m)
        print(f"  {r.trades:3d} trades  WR={r.win_rate*100:.0f}%  Net={r.net_pnl:+.0f}")
        results.append(r)

    # Print report
    print_report = _print_report
    print_report(results, args.symbol, data_source)

    # Write CSV
    if args.csv:
        _write_csv(results, _REPO_ROOT / args.csv)


if __name__ == "__main__":
    main()
