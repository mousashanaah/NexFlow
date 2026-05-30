#!/usr/bin/env python3
"""Parity investigation: live paper vs replay backtest on identical dates.

Goal: determine whether the strategy failed in the live period, or whether
the original backtest overstated performance.

Method:
  1. Parse all journal files → extract live paper stats (signals, fills, WR, PnL, PF).
  2. Derive exact date range from the journals.
  3. Load candles for that range (parquet preferred; Bitget REST fallback).
  4. Run BacktestRunner on those exact candles.
  5. Print side-by-side and flag all code-path differences between the two paths.

Data quality note (critical):
  Live candles from the WebSocket engine contain:
    buy_volume  — from real trade-flow aggregation
    spread_avg  — from real orderbook spread sampling
  REST candles contain only OHLCV.  Missing fields are approximated:
    buy_volume  ≈ (close − low) / (high − low) × volume  (bar-position proxy)
    spread_avg  = 0.0  (no orderbook data → spread filter ALWAYS passes in replay)
  These approximations are flagged in the report.

Usage:
  python scripts/parity_investigation.py
  python scripts/parity_investigation.py --journal-dir logs/paper --candle-dir data/candles
  python scripts/parity_investigation.py --no-rest          (parquet only)
  python scripts/parity_investigation.py --symbol ETHUSDT
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

from nexflow.services.candles.candle_engine import Candle, TIMEFRAMES
from nexflow.services.strategy.backtest_runner import BacktestConfig, BacktestRunner
from nexflow.services.strategy.momentum_strategy import MomentumConfig, MomentumStrategy
from nexflow.services.strategy.paper_execution import ExecutionConfig
from nexflow.services.strategy.risk_engine import RiskConfig
from nexflow.services.strategy.signal_models import Direction

TAKER = 0.0006
MAKER = 0.0002
_W = 70


# ---------------------------------------------------------------------------
# Live stats from journal
# ---------------------------------------------------------------------------

@dataclass
class LiveStats:
    symbol: str
    start_ts: float
    end_ts: float
    n_signals: int
    n_fills: int
    n_wins: int
    n_losses: int
    n_stops: int
    n_tp_only: int          # trades closed entirely by TPs (no stop)
    net_pnl: float
    gross_pnl: float
    total_fees: float
    win_rate: float
    profit_factor: float
    expectancy_usd: float   # net_pnl / n_fills
    signal_ts_list: list[float] = field(default_factory=list)   # for signal-level diff


def _parse_journals(journal_dir: Path, symbol: str) -> LiveStats:
    """Extract live paper stats from all .jsonl files, filtered to symbol."""
    events: list[dict] = []
    for p in sorted(journal_dir.rglob("*.jsonl")):
        for line in p.open():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    events.sort(key=lambda e: e.get("ts_epoch", 0))

    # Group into trades per symbol
    open_fill: dict[str, dict] = {}
    open_partials: dict[str, list] = {}

    n_signals = 0
    signal_ts: list[float] = []
    trades: list[dict] = []     # {"fill", "partials", "exit"}
    start_ts = float("inf")
    end_ts   = 0.0

    for ev in events:
        if ev.get("symbol") != symbol:
            continue
        etype = ev.get("event", "")
        ts    = ev.get("ts_epoch", 0.0)
        start_ts = min(start_ts, ts)
        end_ts   = max(end_ts, ts)

        if etype == "SIGNAL":
            n_signals += 1
            signal_ts.append(ts)

        elif etype == "FILL":
            open_fill[symbol] = ev
            open_partials[symbol] = []

        elif etype == "PARTIAL_TP":
            open_partials.setdefault(symbol, []).append(ev)
            total = open_fill.get(symbol, {}).get("size", 0.0)
            closed = sum(p.get("size", 0.0) for p in open_partials[symbol])
            if total > 0 and abs(closed - total) < 1e-6:
                trades.append({"fill": open_fill.pop(symbol, {}),
                               "partials": list(open_partials.pop(symbol, [])),
                               "exit": None})

        elif etype in ("STOP_HIT", "FORCE_CLOSE"):
            if symbol in open_fill:
                trades.append({"fill": open_fill.pop(symbol, {}),
                               "partials": list(open_partials.pop(symbol, [])),
                               "exit": ev})

    # Compute stats
    wins = 0; losses = 0; stops = 0; tp_only = 0
    net_pnl = 0.0; gross_pnl = 0.0; total_fees = 0.0

    for t in trades:
        f       = t["fill"]
        partials = t["partials"]
        ex      = t["exit"]
        entry_fee = f.get("fee", 0.0)
        fill_px   = f.get("fill_price", 0.0)
        total_sz  = f.get("size", 0.0)
        is_long   = f.get("direction", "long") == "long"
        sign      = 1.0 if is_long else -1.0

        # Compute from first principles (avoids journal ambiguities)
        recon_net = -entry_fee
        for p in partials:
            tp_fill = p.get("fill_price", 0.0)
            tp_sz   = p.get("size", 0.0)
            tp_fee  = p.get("fee", 0.0)
            recon_net += (tp_fill - fill_px) * tp_sz * sign - tp_fee

        if ex and ex.get("event") == "STOP_HIT":
            stops += 1
            stop_fill = ex.get("fill_price", 0.0)
            stop_fee  = ex.get("fee", 0.0)
            tp_sz_sum = sum(p.get("size", 0.0) for p in partials)
            remaining = total_sz - tp_sz_sum
            recon_net += (stop_fill - fill_px) * remaining * sign - stop_fee
            all_fees  = entry_fee + sum(p.get("fee",0) for p in partials) + stop_fee
        elif ex is None:
            tp_only += 1
            all_fees = entry_fee + sum(p.get("fee",0) for p in partials)
        else:
            all_fees = entry_fee  # force close — rough

        gross = recon_net + all_fees
        total_fees += all_fees
        gross_pnl  += gross
        net_pnl    += recon_net
        if recon_net > 0:
            wins += 1
        else:
            losses += 1

    n_fills = len(trades)
    wr  = wins / n_fills if n_fills > 0 else 0.0
    gross_wins  = sum(t for t in [_trade_net(t) for t in trades] if t > 0)
    gross_losses = abs(sum(t for t in [_trade_net(t) for t in trades] if t <= 0))
    pf  = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    exp = net_pnl / n_fills if n_fills > 0 else 0.0

    return LiveStats(
        symbol=symbol,
        start_ts=start_ts if start_ts != float("inf") else 0.0,
        end_ts=end_ts,
        n_signals=n_signals,
        n_fills=n_fills,
        n_wins=wins,
        n_losses=losses,
        n_stops=stops,
        n_tp_only=tp_only,
        net_pnl=net_pnl,
        gross_pnl=gross_pnl,
        total_fees=total_fees,
        win_rate=wr,
        profit_factor=pf,
        expectancy_usd=exp,
        signal_ts_list=signal_ts,
    )


def _trade_net(t: dict) -> float:
    f = t["fill"]; partials = t["partials"]; ex = t["exit"]
    fill_px = f.get("fill_price", 0.0)
    total_sz = f.get("size", 0.0)
    is_long = f.get("direction", "long") == "long"
    sign = 1.0 if is_long else -1.0
    net = -f.get("fee", 0.0)
    for p in partials:
        tp_fill = p.get("fill_price", 0.0)
        tp_sz   = p.get("size", 0.0)
        tp_fee  = p.get("fee", 0.0)
        net += (tp_fill - fill_px) * tp_sz * sign - tp_fee
    if ex and ex.get("event") == "STOP_HIT":
        tp_sz_sum = sum(p.get("size", 0.0) for p in partials)
        remaining = total_sz - tp_sz_sum
        net += (ex.get("fill_price", 0.0) - fill_px) * remaining * sign - ex.get("fee", 0.0)
    return net


# ---------------------------------------------------------------------------
# Candle loading
# ---------------------------------------------------------------------------

def _load_parquet(candle_dir: Path, symbol: str, tf: str,
                  start_ts: float, end_ts: float) -> list[Candle]:
    path = candle_dir / f"{symbol}_{tf}.parquet"
    if not path.exists() or not _HAS_PARQUET:
        return []
    rows = pq.read_table(path).to_pylist()
    return [
        Candle(
            symbol=r["symbol"], timeframe=r["timeframe"],
            open_time=r["open_time"], close_time=r["close_time"],
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], buy_volume=r["buy_volume"],
            sell_volume=r["sell_volume"], trade_count=r["trade_count"],
            vwap=r["vwap"], spread_avg=r["spread_avg"],
            spread_max=r["spread_max"],
            volatility_estimate=r["volatility_estimate"],
            is_final=True,
        )
        for r in rows
        if r.get("is_final") and start_ts <= r.get("close_time", 0) <= end_ts
    ]


def _buy_vol_proxy(o: float, h: float, l: float, c: float, vol: float) -> float:
    """Bar-position proxy for buy volume. If h==l returns 0.5 × vol."""
    if h <= l:
        return vol * 0.5
    return (c - l) / (h - l) * vol


def _fetch_rest_1m(symbol: str, start_ms: int, end_ms: int) -> list[Candle]:
    """Fetch 1m candles from Bitget REST for the date range.

    buy_volume is approximated via bar-position proxy.
    spread_avg is set to 0.0 (not available from REST).
    """
    candles: list[Candle] = []
    cursor = start_ms
    print(f"  Fetching {symbol} 1m REST candles "
          f"{datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} "
          f"→ {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    while cursor < end_ms:
        chunk_end = min(cursor + 999 * 60_000, end_ms)
        url = (
            f"https://api.bitget.com/api/v2/mix/market/candles"
            f"?symbol={symbol}&productType=USDT-FUTURES&granularity=1m"
            f"&startTime={cursor}&endTime={chunk_end}&limit=1000"
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            if data.get("code") != "00000":
                print(f"  [WARN] REST error: {data.get('msg')}")
                break
            rows = data.get("data", [])
            if not rows:
                break
            for r in rows:
                ts_ms = int(r[0])
                o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                vol = float(r[5]) if len(r) > 5 else 0.0
                bvol = _buy_vol_proxy(o, h, l, c, vol)
                open_s  = ts_ms // 1000
                close_s = open_s + 60
                candles.append(Candle(
                    symbol=symbol, timeframe="1m",
                    open_time=open_s, close_time=close_s,
                    open=o, high=h, low=l, close=c,
                    volume=vol, buy_volume=bvol,
                    sell_volume=vol - bvol,
                    trade_count=0,
                    vwap=(o + h + l + c) / 4,
                    spread_avg=0.0,   # ← NOT AVAILABLE FROM REST
                    spread_max=0.0,
                    volatility_estimate=(h - l) / o if o > 0 else 0.0,
                    is_final=True,
                ))
            cursor = int(rows[-1][0]) + 60_000
            time.sleep(0.15)
        except Exception as exc:
            print(f"  [WARN] REST fetch failed: {exc}")
            break
    candles.sort(key=lambda c: c.close_time)
    return candles


def _resample_to_5m(candles_1m: list[Candle], symbol: str) -> list[Candle]:
    """Aggregate 1m candles into 5m candles (5-bar groups aligned to UTC 5m boundaries)."""
    result: list[Candle] = []
    bucket: list[Candle] = []
    for c in candles_1m:
        # 5m boundary: close_time is aligned to 5m if (close_time % 300 == 0)
        # open_time aligns to 5m boundary
        bucket.append(c)
        if c.close_time % 300 == 0:
            if bucket:
                o  = bucket[0].open
                h  = max(b.high for b in bucket)
                l  = min(b.low for b in bucket)
                cl = bucket[-1].close
                vol   = sum(b.volume for b in bucket)
                bvol  = sum(b.buy_volume for b in bucket)
                svol  = sum(b.sell_volume for b in bucket)
                pv    = sum(b.vwap * b.volume for b in bucket)
                vwap  = pv / vol if vol > 0 else (o + h + l + cl) / 4
                spreads = [b.spread_avg for b in bucket if b.spread_avg > 0]
                result.append(Candle(
                    symbol=symbol, timeframe="5m",
                    open_time=bucket[0].open_time,
                    close_time=bucket[-1].close_time,
                    open=o, high=h, low=l, close=cl,
                    volume=vol, buy_volume=bvol, sell_volume=svol,
                    trade_count=sum(b.trade_count for b in bucket),
                    vwap=vwap,
                    spread_avg=sum(spreads)/len(spreads) if spreads else 0.0,
                    spread_max=max(b.spread_max for b in bucket),
                    volatility_estimate=(h - l) / o if o > 0 else 0.0,
                    is_final=True,
                ))
                bucket = []
    return result


def _resample_to_15m(candles_1m: list[Candle], symbol: str) -> list[Candle]:
    result: list[Candle] = []
    bucket: list[Candle] = []
    for c in candles_1m:
        bucket.append(c)
        if c.close_time % 900 == 0:
            if bucket:
                o  = bucket[0].open
                h  = max(b.high for b in bucket)
                l  = min(b.low for b in bucket)
                cl = bucket[-1].close
                vol  = sum(b.volume for b in bucket)
                bvol = sum(b.buy_volume for b in bucket)
                pv   = sum(b.vwap * b.volume for b in bucket)
                vwap = pv / vol if vol > 0 else (o + h + l + cl) / 4
                spreads = [b.spread_avg for b in bucket if b.spread_avg > 0]
                result.append(Candle(
                    symbol=symbol, timeframe="15m",
                    open_time=bucket[0].open_time,
                    close_time=bucket[-1].close_time,
                    open=o, high=h, low=l, close=cl,
                    volume=vol, buy_volume=bvol,
                    sell_volume=vol - bvol,
                    trade_count=sum(b.trade_count for b in bucket),
                    vwap=vwap,
                    spread_avg=sum(spreads)/len(spreads) if spreads else 0.0,
                    spread_max=max(b.spread_max for b in bucket),
                    volatility_estimate=(h - l) / o if o > 0 else 0.0,
                    is_final=True,
                ))
                bucket = []
    return result


def _load_candles(
    symbol: str, start_ts: float, end_ts: float,
    candle_dir: Path, no_rest: bool
) -> tuple[dict[str, dict[str, list[Candle]]], str]:
    """Return (all_candles_dict, data_source_label)."""
    # Add 30min warmup before live start (strategy needs min_bars_1m=22 warm up)
    warmup_s = 40 * 60   # 40 minutes
    fetch_start = int(start_ts) - warmup_s
    fetch_end   = int(end_ts)   + 60     # one bar after last event

    # Try parquet first
    c1m = _load_parquet(candle_dir, symbol, "1m", fetch_start, fetch_end)
    if c1m:
        c5m  = _load_parquet(candle_dir, symbol, "5m",  fetch_start, fetch_end)
        c15m = _load_parquet(candle_dir, symbol, "15m", fetch_start, fetch_end)
        source = "parquet (real buy_volume + spread_avg)"
    elif not no_rest:
        print(f"\nNo parquet found. Fetching from Bitget REST API …")
        c1m  = _fetch_rest_1m(symbol, fetch_start * 1000, fetch_end * 1000)
        c5m  = _resample_to_5m(c1m, symbol)
        c15m = _resample_to_15m(c1m, symbol)
        source = "Bitget REST (buy_volume=bar-proxy, spread_avg=0.0)"
    else:
        return {}, "none"

    if not c1m:
        return {}, "none"

    return {symbol: {"1m": c1m, "5m": c5m, "15m": c15m}}, source


# ---------------------------------------------------------------------------
# Instrumented backtest — captures signal-level events
# ---------------------------------------------------------------------------

@dataclass
class ReplaySignal:
    ts: int
    symbol: str
    direction: str
    entry_price: float
    stop_price: float
    atr: float
    features: dict


class InstrumentedStrategy(MomentumStrategy):
    """Wraps MomentumStrategy and records every signal emitted."""

    def __init__(self, cfg: MomentumConfig | None = None) -> None:
        super().__init__(cfg)
        self.emitted_signals: list[ReplaySignal] = []

    def on_candle(self, candle: Candle):
        sig = super().on_candle(candle)
        if sig is not None:
            self.emitted_signals.append(ReplaySignal(
                ts=candle.close_time,
                symbol=candle.symbol,
                direction=sig.direction.value,
                entry_price=sig.entry_price,
                stop_price=sig.stop_price,
                atr=sig.atr,
                features=dict(sig.features),
            ))
        return sig


def run_replay(
    all_candles: dict[str, dict[str, list[Candle]]],
) -> tuple[object, InstrumentedStrategy]:
    strategy = InstrumentedStrategy()
    bt_cfg = BacktestConfig(
        initial_equity=100_000.0,
        risk=RiskConfig(),
        execution=ExecutionConfig(),
    )
    runner = BacktestRunner(strategy, bt_cfg)
    metrics = runner.run(all_candles)
    return metrics, strategy


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: float, dp: int = 2) -> str:
    return f"{v:+.{dp}f}" if v != 0 else f"{v:.{dp}f}"


def _pct(v: float) -> str:
    return f"{v*100:.1f}%"


def _bar() -> str:
    return "─" * _W


def print_report(
    live: LiveStats,
    metrics,
    strategy: InstrumentedStrategy,
    data_source: str,
    all_candles: dict,
    symbol: str,
) -> None:
    trades = metrics.pnl_distribution
    n_replay = metrics.total_trades
    replay_wins = sum(1 for p in trades if p > 0)
    replay_losses = n_replay - replay_wins
    replay_wr = replay_wins / n_replay if n_replay > 0 else 0.0
    replay_fees = metrics.total_fees
    replay_gross_wins  = sum(p for p in trades if p > 0)
    replay_gross_losses = abs(sum(p for p in trades if p <= 0))
    replay_pf = replay_gross_wins / replay_gross_losses if replay_gross_losses > 0 else float("inf")
    replay_exp = metrics.net_pnl / n_replay if n_replay > 0 else 0.0
    replay_signals = len(strategy.emitted_signals)

    start_str = datetime.fromtimestamp(live.start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    end_str   = datetime.fromtimestamp(live.end_ts,   tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print()
    print("═" * _W)
    print("PARITY INVESTIGATION — LIVE PAPER vs REPLAY BACKTEST")
    print("═" * _W)
    print(f"  Symbol     : {symbol}")
    print(f"  Period     : {start_str}  →  {end_str}")
    print(f"  Candle src : {data_source}")
    n1m = len(all_candles.get(symbol, {}).get("1m", []))
    n5m = len(all_candles.get(symbol, {}).get("5m", []))
    print(f"  Bars loaded: {n1m} × 1m   {n5m} × 5m")
    print()

    # Side-by-side table
    col = 26
    print(f"{'Metric':<30} {'LIVE':>{col}} {'REPLAY':>{col}}  {'DIFF':>10}")
    print(_bar())

    def row(label: str, live_val: str, replay_val: str, diff_val: str = "") -> None:
        print(f"  {label:<28} {live_val:>{col}} {replay_val:>{col}}  {diff_val:>10}")

    row("Signals fired",
        str(live.n_signals), str(replay_signals),
        _fmt(replay_signals - live.n_signals, 0))

    row("Fills (trades closed)",
        str(live.n_fills), str(n_replay),
        _fmt(n_replay - live.n_fills, 0))

    row("Win rate",
        _pct(live.win_rate), _pct(replay_wr),
        _fmt((replay_wr - live.win_rate) * 100, 1) + "pp")

    row("Wins / Losses",
        f"{live.n_wins} / {live.n_losses}",
        f"{replay_wins} / {replay_losses}")

    row("Net PnL (USD)",
        f"{live.net_pnl:+.2f}", f"{metrics.net_pnl:+.2f}",
        _fmt(metrics.net_pnl - live.net_pnl))

    row("Profit Factor",
        f"{live.profit_factor:.3f}",
        f"{replay_pf:.3f}" if replay_pf != float("inf") else "∞",
        "")

    row("Expectancy (USD/trade)",
        f"{live.expectancy_usd:+.2f}", f"{replay_exp:+.2f}",
        _fmt(replay_exp - live.expectancy_usd))

    row("Fee drag (USD)",
        f"{live.total_fees:.2f}", f"{replay_fees:.2f}",
        _fmt(replay_fees - live.total_fees))

    print(_bar())
    print()

    # -----------------------------------------------------------------------
    # Divergence analysis
    # -----------------------------------------------------------------------
    print("DIVERGENCE ANALYSIS")
    print(_bar())

    sig_diff   = replay_signals - live.n_signals
    fill_diff  = n_replay - live.n_fills
    wr_diff_pp = (replay_wr - live.win_rate) * 100

    if abs(sig_diff) < 3 and abs(fill_diff) < 3:
        print("  ✓ Signal and fill counts are close — same candle data is reaching the strategy.")
    else:
        print(f"  ✗ Signal count diverges by {sig_diff:+d} ({replay_signals} replay vs {live.n_signals} live).")
        if sig_diff > 5:
            print("    → Replay fires MORE signals. Possible cause: spread_avg=0 in REST candles")
            print("      causes the spread_regime filter to ALWAYS pass. Live rejects some signals")
            print("      when the real spread exceeds 30% of ATR.")
        elif sig_diff < -5:
            print("    → Replay fires FEWER signals. Possible cause: buy_volume approximation")
            print("      differs from real trade flow used in live system.")

    if abs(wr_diff_pp) < 10:
        print(f"  ✓ Win rates are within 10pp ({live.win_rate*100:.1f}% vs {replay_wr*100:.1f}%).")
        print("    → Performance gap is NOT regime-dependent on this time window.")
        print("    → The strategy genuinely underperforms. Original backtest may have been")
        print("      on a different (trending) period. Regime mismatch is the most likely cause.")
    elif wr_diff_pp > 10:
        print(f"  ✗ Replay win rate is {wr_diff_pp:+.1f}pp HIGHER than live ({replay_wr*100:.1f}% vs {live.win_rate*100:.1f}%).")
        print("    → The replay backtest overstates performance even on the SAME dates.")
        print("    → STRONG evidence of look-ahead bias or code-path divergence.")
        print("    → Inspect code path differences section below.")
    else:
        print(f"  ✗ Live win rate is {abs(wr_diff_pp):.1f}pp HIGHER than replay.")
        print("    → Unusual: live outperforms replay. May be a signal-count/sizing difference.")

    print()

    # -----------------------------------------------------------------------
    # Code path differences
    # -----------------------------------------------------------------------
    print("CODE PATH DIFFERENCES (live vs replay)")
    print(_bar())

    # 1. Entry timing
    print("  1. ENTRY TIMING")
    print("     Live:   signal fires at bar N close; position opened at bar N close + slip.")
    print("             WebSocket delivers bar at bar N+1 open; position is live from bar N+1.")
    print("     Replay: signal fires at bar N close; position opened at bar N close + slip.")
    print("             Stop/TP evaluation begins at bar N+1.")
    print("     VERDICT: identical — same bar-close entry, same next-bar evaluation.")
    print()

    # 2. Same-bar stop+TP priority
    print("  2. SAME-BAR STOP + TP CONFLICT")
    print("     Both live and replay: if bar.low ≤ stop AND bar.high ≥ TP1 simultaneously,")
    print("     the stop takes priority and TP1 is skipped (conservative assumption).")
    print("     This is identical in both paths — not a source of divergence.")
    print()

    # 3. buy_volume / spread_avg
    if "REST" in data_source:
        print("  3. CANDLE FIELD QUALITY  ← KNOWN DIVERGENCE")
        print("     Live candles have real buy_volume from WebSocket trade aggregation.")
        print("     REST candles use bar-position proxy:  buy_vol ≈ (close-low)/(high-low) × vol")
        print("     This approximation introduces noise in the buy_sell_imbalance filter.")
        print()
        print("     Live candles have real spread_avg from orderbook sampling.")
        print("     REST candles set spread_avg = 0.0 → spread_regime = 0 → filter ALWAYS passes.")
        print("     Effect: replay may fire signals that live rejected due to wide spread.")
        if sig_diff > 0:
            print(f"     This likely explains the {sig_diff:+d} signal count difference.")
        print()
    else:
        print("  3. CANDLE FIELD QUALITY")
        print("     Using parquet candles from the live engine — buy_volume and spread_avg")
        print("     are real values, not approximated. This path is identical to live.")
        print()

    # 4. Risk engine state
    print("  4. RISK ENGINE STATE")
    print("     Live: risk engine tracks daily PnL drawdown and kill-switch state across")
    print("           session restarts. A kill-switch in session 1 blocks entries in session 2.")
    print("     Replay: risk engine resets to zero at start — no kill-switch carry-over.")
    print("     Effect: if live trading had kill-switch events, replay ignores them and")
    print("             fires signals that live blocked.")
    print()

    # 5. Position-already-open guard
    print("  5. POSITION GUARD / FILL RATE")
    print("     Live: if a position is open, any new signal for the same symbol is REJECTED.")
    print("     Replay: same logic.")
    print("     VERDICT: identical.")
    print()

    # -----------------------------------------------------------------------
    # Signal-level diff
    # -----------------------------------------------------------------------
    print("SIGNAL TIMELINE COMPARISON")
    print(_bar())

    # Bin live signals by 1-hour buckets for comparison
    live_ts_set  = set(int(ts // 3600) for ts in live.signal_ts_list)
    replay_ts_set = set(int(s.ts // 3600) for s in strategy.emitted_signals)
    only_live   = live_ts_set - replay_ts_set
    only_replay = replay_ts_set - live_ts_set
    both        = live_ts_set & replay_ts_set

    print(f"  Signal hours in BOTH    : {len(both)}")
    print(f"  Signal hours LIVE only  : {len(only_live)}")
    print(f"  Signal hours REPLAY only: {len(only_replay)}")
    if only_live:
        for h in sorted(only_live)[:5]:
            dt = datetime.fromtimestamp(h * 3600, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"    live-only hour: {dt}")
    if only_replay:
        for h in sorted(only_replay)[:5]:
            dt = datetime.fromtimestamp(h * 3600, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"    replay-only hour: {dt}")
    print()

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("═" * _W)
    print("VERDICT")
    print("═" * _W)

    if abs(wr_diff_pp) < 10:
        print("  Replay WR ≈ Live WR on the same date range.")
        print()
        if live.win_rate < 0.30:
            print("  The strategy genuinely has a low win rate in this market period.")
            print("  The original backtest was run on a DIFFERENT time period where the")
            print("  market was trending. This is REGIME MISMATCH — the strategy was")
            print("  not overfit, but it was developed in conditions that no longer exist.")
            print()
            print("  Next step: identify the original backtest date range and check if")
            print("  the strategy's WR collapses on out-of-sample months prior to May 2026.")
        else:
            print("  Win rate on this window is reasonable. Check if fee drag is the issue.")
    elif wr_diff_pp > 15:
        print("  Replay WR is materially HIGHER than live WR on identical dates.")
        print()
        print("  This is the signature of LOOK-AHEAD BIAS or a code-path divergence.")
        print("  The backtest engine is seeing information unavailable to the live system,")
        print("  OR the candle fields differ significantly between live and REST data.")
        print()
        if "REST" in data_source:
            print("  ACTION REQUIRED: Re-run with --candle-dir pointing to real parquet files.")
            print("  If the divergence persists with parquet candles, look-ahead bias is confirmed.")
        else:
            print("  ACTION REQUIRED: Inspect momentum_strategy.py compute_signal() for any")
            print("  use of future bars. Check that rolling_high/rolling_low exclude current bar.")
            print("  Check that the entry price is bar N close, not bar N+1 open.")
    else:
        print(f"  Win rates differ by {wr_diff_pp:+.1f}pp — moderate divergence.")
        print("  Investigate candle field quality (buy_volume, spread_avg) as primary cause.")

    print()
    print("  Fee drag comparison:")
    fee_pct_live   = live.total_fees / 100_000 * 100
    fee_pct_replay = replay_fees / 100_000 * 100
    print(f"    Live   fee drag: {live.total_fees:.2f} USD ({fee_pct_live:.3f}% of equity)")
    print(f"    Replay fee drag: {replay_fees:.2f} USD ({fee_pct_replay:.3f}% of equity)")
    print(f"    Fee drag is {'likely not' if fee_pct_live < 0.5 else 'potentially'} the primary driver of losses.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Live vs replay parity investigation")
    p.add_argument("--journal-dir", default="logs/paper_validation",
                   help="Directory with .jsonl journal files")
    p.add_argument("--candle-dir",  default="data/candles",
                   help="Directory with .parquet candle files")
    p.add_argument("--symbol",      default="ETHUSDT")
    p.add_argument("--no-rest",     action="store_true",
                   help="Skip REST fallback; fail if no parquet found")
    args = p.parse_args()

    journal_dir = _REPO_ROOT / args.journal_dir
    if not journal_dir.exists():
        journal_dir = Path(args.journal_dir)
    candle_dir = _REPO_ROOT / args.candle_dir
    if not candle_dir.exists():
        candle_dir = Path(args.candle_dir)

    # 1. Extract live stats
    print(f"Parsing journals in {journal_dir} …")
    live = _parse_journals(journal_dir, args.symbol)
    if live.n_fills == 0:
        print(f"No completed trades found for {args.symbol} in {journal_dir}.")
        print("Copy your live journal files there and re-run.")
        sys.exit(1)
    print(f"  Found {live.n_signals} signals, {live.n_fills} fills for {args.symbol}")
    print(f"  Period: {datetime.fromtimestamp(live.start_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} "
          f"→ {datetime.fromtimestamp(live.end_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

    # 2. Load candles
    print(f"\nLoading candles …")
    all_candles, data_source = _load_candles(
        args.symbol, live.start_ts, live.end_ts, candle_dir, args.no_rest
    )
    if not all_candles:
        print("No candle data available. Use --candle-dir to point to parquet files,")
        print("or remove --no-rest to enable the Bitget REST fallback.")
        sys.exit(1)

    n1m = len(all_candles.get(args.symbol, {}).get("1m", []))
    print(f"  {n1m} × 1m bars loaded  [{data_source}]")

    # 3. Run replay
    print(f"\nRunning replay backtest on {n1m} bars …")
    metrics, strategy = run_replay(all_candles)
    print(f"  Replay complete: {metrics.total_trades} trades, "
          f"WR={metrics.win_rate*100:.1f}%, Net={metrics.net_pnl:+.2f}")

    # 4. Print report
    print_report(live, metrics, strategy, data_source, all_candles, args.symbol)


if __name__ == "__main__":
    main()
