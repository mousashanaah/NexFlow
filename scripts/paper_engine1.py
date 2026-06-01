#!/usr/bin/env python3
"""Engine #1 daemon — daily trend following.

Supports three execution modes (--mode flag or NEXFLOW_EXEC_MODE env var):
  LOCAL_PAPER   — local simulation, no exchange (default)
  BITGET_PAPER  — real orders to Bitget Demo Trading via REST API

Strategy logic is identical across all modes. Only the execution adapter
changes. BACKTEST mode is handled by scripts/run_trend_backtest.py.

Wakes at 00:05 UTC each day (5 min after Bitget daily candle close).
State persisted to data/engine1_state.json after every cycle.

Usage:
  python scripts/paper_engine1.py                           # LOCAL_PAPER, live
  python scripts/paper_engine1.py --mode BITGET_PAPER       # demo exchange
  python scripts/paper_engine1.py --run-now                 # one cycle, exit
  python scripts/paper_engine1.py --backfill                # replay parquet
  python scripts/paper_engine1.py --status                  # print state, exit
  python scripts/paper_engine1.py --equity 5000             # set starting equity

For BITGET_PAPER mode, set environment variables before running:
  BITGET_API_KEY=<your demo key>
  BITGET_API_SECRET=<your demo secret>
  BITGET_PASSPHRASE=<your demo passphrase>
  BITGET_PAPER=1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nexflow.execution.adapter import ExecMode, build_adapter, ExecutionAdapter
from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.signal_models import Direction
from nexflow.services.strategy.trend_strategy import TrendFollowingStrategy, _wilder_atr

# ---------------------------------------------------------------------------
# Configuration — mirrors run_trend_backtest.py exactly
# ---------------------------------------------------------------------------
_SYMBOLS        = ["BTCUSDT", "ETHUSDT"]
_LOOKBACK       = 20
_ATR_PERIOD     = 14
_INIT_MULT      = 2.5
_TRAIL_MULT     = 2.0
_RISK_PCT       = 0.01
_MAX_RISK_PCT   = 0.03
_TAKER_FEE      = 0.0006
_CANDLES_NEEDED = 60

_TRIGGER_HOUR_UTC   = 0
_TRIGGER_MINUTE_UTC = 5

_URL_HISTORY = "https://api.bitget.com/api/v2/mix/market/history-candles"
_HEADERS     = {"User-Agent": "NexFlow/1.0", "Accept": "application/json"}

_STATE_FILE   = _REPO_ROOT / "data" / "engine1_state.json"
_JOURNAL_FILE = _REPO_ROOT / "data" / "engine1_journal.csv"
_LOG_FILE     = _REPO_ROOT / "data" / "engine1_daemon.log"
_DAY_MS       = 86_400_000

_JOURNAL_FIELDS = [
    "date", "symbol", "event", "direction",
    "price", "size", "atr", "stop",
    "r_multiple", "pnl", "equity", "mode", "note",
]


# ---------------------------------------------------------------------------
# Logging / state
# ---------------------------------------------------------------------------

def _load_state(initial_equity: float) -> dict:
    if _STATE_FILE.exists():
        with open(_STATE_FILE) as f:
            return json.load(f)
    return {
        "equity":              initial_equity,
        "last_processed_date": None,
        "positions":           {},
        "pending_entries":     {},
    }


def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _journal_write(row: dict) -> None:
    _JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not _JOURNAL_FILE.exists()
    with open(_JOURNAL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_JOURNAL_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _JOURNAL_FIELDS})


def _log(msg: str) -> None:
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Candle fetching (public endpoint, no auth needed)
# ---------------------------------------------------------------------------

def _fetch_daily_candles(symbol: str, n: int = _CANDLES_NEEDED) -> list[Candle]:
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    url = (
        f"{_URL_HISTORY}?symbol={symbol}&productType=USDT-FUTURES"
        f"&granularity=1D&endTime={end_ms}&limit={n}"
    )
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)
    if data.get("code") != "00000":
        raise RuntimeError(f"API error for {symbol}: {data.get('msg')}")
    rows = sorted(data["data"], key=lambda r: int(r[0]))
    candles = []
    for r in rows:
        ts = int(r[0])
        candles.append(Candle(
            symbol=symbol, timeframe="1D",
            open_time=ts, close_time=ts + _DAY_MS - 1,
            open=float(r[1]), high=float(r[2]),
            low=float(r[3]),  close=float(r[4]),
            volume=float(r[5]) if len(r) > 5 else 0.0,
            buy_volume=0.0, sell_volume=0.0, trade_count=0,
            vwap=float(r[4]), spread_avg=0.0, spread_max=0.0,
            volatility_estimate=0.0, is_final=True,
        ))
    return candles


def _compute_atr(candles: list[Candle]) -> float | None:
    return _wilder_atr(
        [c.high  for c in candles],
        [c.low   for c in candles],
        [c.close for c in candles],
        _ATR_PERIOD,
    )


# ---------------------------------------------------------------------------
# Daily cycle
# ---------------------------------------------------------------------------

def _run_cycle(
    state: dict,
    today_str: str,
    adapter: ExecutionAdapter,
    exec_mode: ExecMode,
) -> None:
    if state.get("last_processed_date") == today_str:
        _log(f"  Cycle {today_str} already processed — skipping (idempotent)")
        return

    _log(f"-- Daily cycle: {today_str}  [{exec_mode.value}] ---------------")

    equity          = state["equity"]
    positions       = state["positions"]
    pending_entries = state["pending_entries"]
    mode_str        = exec_mode.value

    for symbol in _SYMBOLS:
        _log(f"  [{symbol}] fetching candles ...")
        try:
            candles = _fetch_daily_candles(symbol, _CANDLES_NEEDED)
        except Exception as exc:
            _log(f"  [{symbol}] ERROR fetching candles: {exc}")
            continue

        if len(candles) < _LOOKBACK + _ATR_PERIOD + 2:
            _log(f"  [{symbol}] insufficient history ({len(candles)} bars) — skipping")
            continue

        today_candle = candles[-1]
        today_date   = datetime.fromtimestamp(
            today_candle.open_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        # ── A. Execute pending entry from yesterday's signal ─────────────
        if symbol in pending_entries:
            pe       = pending_entries.pop(symbol)
            sig_dir  = Direction(pe["direction"])
            sig_atr  = pe["signal_atr"]
            entry_price = today_candle.open or today_candle.close

            initial_stop = (
                entry_price - _INIT_MULT * sig_atr if sig_dir is Direction.LONG
                else entry_price + _INIT_MULT * sig_atr
            )
            stop_dist = abs(entry_price - initial_stop)

            # Close existing position first (direction flip)
            if symbol in positions:
                old     = positions.pop(symbol)
                old_dir = Direction(old["direction"])
                adapter.on_close(symbol, old_dir.value, old["size"], entry_price, "flip")
                exit_fee = entry_price * old["size"] * _TAKER_FEE
                raw_pnl  = (entry_price - old["entry_price"]) * old["size"] * (
                    1 if old_dir is Direction.LONG else -1)
                net_pnl  = raw_pnl - exit_fee
                equity  += raw_pnl - exit_fee
                risk_amount = abs(old["entry_price"] - old["initial_stop"]) * old["size"]
                r_mult = net_pnl / risk_amount if risk_amount > 0 else 0.0
                _log(f"  [{symbol}] FLIP CLOSE {old_dir.value} @ ${entry_price:,.2f}  "
                     f"PnL ${net_pnl:+,.2f}  {r_mult:+.2f}R")
                _journal_write({
                    "date": today_date, "symbol": symbol, "event": "FLIP_CLOSE",
                    "direction": old_dir.value, "price": round(entry_price, 4),
                    "size": round(old["size"], 6), "atr": round(sig_atr, 4),
                    "stop": round(old["trail_stop"], 4),
                    "r_multiple": round(r_mult, 3), "pnl": round(net_pnl, 2),
                    "equity": round(equity, 2), "mode": mode_str,
                    "note": "direction flip",
                })

            if stop_dist > 0:
                risk_amount = equity * _RISK_PCT
                size        = risk_amount / stop_dist

                if stop_dist * max(size, 0.001) / equity > _MAX_RISK_PCT and size < 0.001:
                    _log(f"  [{symbol}] skip entry — position too large for account size")
                else:
                    result = adapter.on_entry(
                        symbol, sig_dir.value, size, entry_price, initial_stop, sig_atr
                    )
                    if not result.accepted:
                        _log(f"  [{symbol}] ENTRY REJECTED: {result.note}")
                    else:
                        fill_price = result.fill_price
                        fill_size  = result.fill_size
                        entry_fee  = fill_price * fill_size * _TAKER_FEE
                        equity    -= entry_fee
                        positions[symbol] = {
                            "direction":     sig_dir.value,
                            "entry_price":   fill_price,
                            "entry_time_ms": today_candle.open_time,
                            "size":          fill_size,
                            "initial_stop":  initial_stop,
                            "trail_stop":    initial_stop,
                            "best_close":    fill_price,
                            "fees_paid":     entry_fee,
                            "stop_order_id": result.stop_order_id,
                        }
                        _log(f"  [{symbol}] ENTER {sig_dir.value} @ ${fill_price:,.2f}  "
                             f"stop ${initial_stop:,.2f}  size {fill_size:.6f}  "
                             f"risk ${risk_amount:.2f}  [{result.note}]")
                        _journal_write({
                            "date": today_date, "symbol": symbol, "event": "ENTRY",
                            "direction": sig_dir.value, "price": round(fill_price, 4),
                            "size": round(fill_size, 6), "atr": round(sig_atr, 4),
                            "stop": round(initial_stop, 4),
                            "r_multiple": "", "pnl": round(-entry_fee, 2),
                            "equity": round(equity, 2), "mode": mode_str,
                            "note": f"signal {pe['signal_date']} | {result.note}",
                        })

        # ── B. Check if today's bar hit the trailing stop ────────────────
        if symbol in positions:
            pos     = positions[symbol]
            pos_dir = Direction(pos["direction"])
            stop_hit = (
                (pos_dir is Direction.LONG  and today_candle.low  <= pos["trail_stop"]) or
                (pos_dir is Direction.SHORT and today_candle.high >= pos["trail_stop"])
            )
            if stop_hit:
                exit_price = pos["trail_stop"]
                adapter.on_close(symbol, pos_dir.value, pos["size"], exit_price, "stop")
                exit_fee    = exit_price * pos["size"] * _TAKER_FEE
                raw_pnl     = (exit_price - pos["entry_price"]) * pos["size"] * (
                    1 if pos_dir is Direction.LONG else -1)
                net_pnl  = raw_pnl - exit_fee
                equity  += raw_pnl - exit_fee
                risk_amount = abs(pos["entry_price"] - pos["initial_stop"]) * pos["size"]
                r_mult = net_pnl / risk_amount if risk_amount > 0 else 0.0
                hold_d = (today_candle.open_time - pos["entry_time_ms"]) / _DAY_MS
                _log(f"  [{symbol}] STOP {pos_dir.value} @ ${exit_price:,.2f}  "
                     f"PnL ${net_pnl:+,.2f}  {r_mult:+.2f}R  held {hold_d:.0f}d")
                _journal_write({
                    "date": today_date, "symbol": symbol, "event": "STOP",
                    "direction": pos_dir.value, "price": round(exit_price, 4),
                    "size": round(pos["size"], 6), "atr": "",
                    "stop": round(pos["trail_stop"], 4),
                    "r_multiple": round(r_mult, 3), "pnl": round(net_pnl, 2),
                    "equity": round(equity, 2), "mode": mode_str,
                    "note": f"held {hold_d:.0f} days",
                })
                del positions[symbol]

        # ── C. Update trailing stop for still-open position ───────────────
        if symbol in positions:
            pos = positions[symbol]
            atr = _compute_atr(candles)
            if atr and atr > 0:
                new_trail, new_best = TrendFollowingStrategy.update_trail(
                    current_trail = pos["trail_stop"],
                    direction     = Direction(pos["direction"]),
                    bar_close     = today_candle.close,
                    best_close    = pos["best_close"],
                    atr           = atr,
                    trail_mult    = _TRAIL_MULT,
                )
                if new_trail != pos["trail_stop"]:
                    new_stop_id = adapter.on_stop_update(
                        symbol, pos["direction"], new_trail, pos.get("stop_order_id")
                    )
                    old_trail = pos["trail_stop"]
                    pos["trail_stop"]    = new_trail
                    pos["best_close"]    = new_best
                    if new_stop_id is not None:
                        pos["stop_order_id"] = new_stop_id
                    _log(f"  [{symbol}] trail updated: ${old_trail:,.2f} -> ${new_trail:,.2f}")
                else:
                    pos["best_close"] = new_best

        # ── D. Run strategy to detect new signal ─────────────────────────
        strategy = TrendFollowingStrategy(
            lookback=_LOOKBACK, atr_period=_ATR_PERIOD,
            initial_stop_mult=_INIT_MULT, trail_mult=_TRAIL_MULT,
        )
        signal = None
        for c in candles:
            signal = strategy.on_candle(c)

        if signal is not None:
            current_dir = Direction(positions[symbol]["direction"]) if symbol in positions else None
            if signal.direction != current_dir:
                _log(f"  [{symbol}] SIGNAL {signal.direction.value} — pending entry tomorrow "
                     f"@ open  ATR={signal.atr:.2f}")
                pending_entries[symbol] = {
                    "direction":  signal.direction.value,
                    "signal_atr": signal.atr,
                    "signal_date": today_date,
                }
                _journal_write({
                    "date": today_date, "symbol": symbol, "event": "SIGNAL",
                    "direction": signal.direction.value, "price": round(today_candle.close, 4),
                    "size": "", "atr": round(signal.atr, 4), "stop": round(signal.stop_price, 4),
                    "r_multiple": "", "pnl": "", "equity": round(equity, 2),
                    "mode": mode_str, "note": "entry tomorrow at open",
                })
        else:
            if symbol in positions:
                pos = positions[symbol]
                status = f"position {pos['direction']} stop ${pos['trail_stop']:,.2f}"
            else:
                status = "flat"
            _log(f"  [{symbol}] no signal  ({status})")

    # ── End of cycle ──────────────────────────────────────────────────────
    state["equity"]              = equity
    state["positions"]           = positions
    state["pending_entries"]     = pending_entries
    state["last_processed_date"] = today_str
    _save_state(state)
    _print_status(state)


# ---------------------------------------------------------------------------
# State-sync from exchange (BITGET_PAPER startup reconciliation)
# ---------------------------------------------------------------------------

def _sync_from_exchange(state: dict, adapter: ExecutionAdapter) -> None:
    """On startup in BITGET_PAPER mode, reconcile local state with exchange.

    If the exchange has a position that local state doesn't know about,
    local state is updated. If local state has a position but the exchange
    is flat (stop was triggered while daemon was offline), close it locally.
    """
    _log("  Syncing position state from exchange ...")
    for symbol in _SYMBOLS:
        try:
            ex_pos = adapter.sync_position(symbol)
        except Exception as exc:
            _log(f"  [{symbol}] sync error: {exc}")
            continue

        local_pos = state["positions"].get(symbol)

        if ex_pos is None and local_pos is not None:
            _log(f"  [{symbol}] exchange is flat but local state has position — "
                 f"assuming stop was triggered offline; closing local record")
            pos = state["positions"].pop(symbol)
            _journal_write({
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "symbol": symbol, "event": "SYNC_CLOSE",
                "direction": pos["direction"], "price": "",
                "size": round(pos["size"], 6), "atr": "", "stop": "",
                "r_multiple": "", "pnl": "", "equity": round(state["equity"], 2),
                "mode": "BITGET_PAPER", "note": "exchange flat at startup",
            })

        elif ex_pos is not None and local_pos is None:
            _log(f"  [{symbol}] exchange has {ex_pos.direction} position not in local state — "
                 f"adding with exchange data (trail/stop may be approximate)")
            state["positions"][symbol] = {
                "direction":     ex_pos.direction,
                "entry_price":   ex_pos.entry_price,
                "entry_time_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
                "size":          ex_pos.size,
                "initial_stop":  0.0,
                "trail_stop":    0.0,
                "best_close":    ex_pos.entry_price,
                "fees_paid":     0.0,
                "stop_order_id": ex_pos.stop_order_id,
            }

        elif ex_pos is not None and local_pos is not None:
            # Direction mismatch: local state and exchange disagree
            if ex_pos.direction != local_pos.get("direction"):
                _log(f"  [{symbol}] WARNING direction mismatch — "
                     f"local={local_pos.get('direction')} exchange={ex_pos.direction}. "
                     f"Trusting exchange; overwriting local state.")
                local_pos["direction"]   = ex_pos.direction
                local_pos["entry_price"] = ex_pos.entry_price
                local_pos["size"]        = ex_pos.size
            if ex_pos.stop_order_id and ex_pos.stop_order_id != local_pos.get("stop_order_id"):
                local_pos["stop_order_id"] = ex_pos.stop_order_id
                _log(f"  [{symbol}] updated stop_order_id from exchange: {ex_pos.stop_order_id}")
            _log(f"  [{symbol}] exchange position confirmed: {ex_pos.direction} "
                 f"size={ex_pos.size}  entry={ex_pos.entry_price:.2f}  "
                 f"unrealizedPnL={ex_pos.unrealized_pnl:+.2f}")
        else:
            _log(f"  [{symbol}] flat (confirmed)")

    _save_state(state)


# ---------------------------------------------------------------------------
# Status printer
# ---------------------------------------------------------------------------

def _print_status(state: dict) -> None:
    _log("  -- Status --")
    _log(f"  Equity   : ${state['equity']:>12,.2f}")
    if state["positions"]:
        for sym, pos in state["positions"].items():
            entry_dt = datetime.fromtimestamp(
                pos["entry_time_ms"] / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            hold = (datetime.now(timezone.utc).timestamp() * 1000 - pos["entry_time_ms"]) / _DAY_MS
            stop_id = pos.get("stop_order_id", "")
            stop_info = f" stop_id={stop_id}" if stop_id else ""
            _log(
                f"  {sym:<12} {pos['direction']:<6} entry ${pos['entry_price']:,.2f} "
                f"on {entry_dt}  trail ${pos['trail_stop']:,.2f}  held {hold:.0f}d{stop_info}"
            )
    else:
        _log("  Positions: none")
    if state["pending_entries"]:
        for sym, pe in state["pending_entries"].items():
            _log(f"  Pending  : {sym} {pe['direction']} enters tomorrow at open")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _next_trigger() -> datetime:
    now = datetime.now(timezone.utc)
    candidate = now.replace(
        hour=_TRIGGER_HOUR_UTC, minute=_TRIGGER_MINUTE_UTC,
        second=0, microsecond=0,
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _run_live(state: dict, adapter: ExecutionAdapter, exec_mode: ExecMode) -> None:
    _log(f"Engine #1 daemon starting [{exec_mode.value}]. "
         f"Waiting for next 00:05 UTC trigger.")
    _print_status(state)
    while True:
        trigger = _next_trigger()
        wait_s  = (trigger - datetime.now(timezone.utc)).total_seconds()
        _log(f"  Next trigger: {trigger.strftime('%Y-%m-%d %H:%M UTC')}  "
             f"(in {wait_s/3600:.1f}h)")
        time.sleep(max(wait_s - 30, 0))
        while datetime.now(timezone.utc) < trigger:
            time.sleep(1)
        today_str = trigger.strftime("%Y-%m-%d")
        try:
            _run_cycle(state, today_str, adapter, exec_mode)
        except Exception as exc:
            _log(f"  [ERROR] Cycle failed: {exc}")
            import traceback
            _log(traceback.format_exc())


# ---------------------------------------------------------------------------
# Backfill (LOCAL_PAPER only — replays parquet, no exchange calls)
# ---------------------------------------------------------------------------

def _run_backfill(state: dict, data_dir: Path, exec_mode: ExecMode) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[ERROR] pyarrow required for backfill mode")
        sys.exit(1)

    _log(f"Backfill mode [{exec_mode.value}]: replaying historical candles ...")
    all_candles: dict[str, list[Candle]] = {}
    for sym in _SYMBOLS:
        path = data_dir / f"{sym}_1D.parquet"
        if not path.exists():
            _log(f"  [WARN] No parquet file for {sym} — skipping")
            continue
        tbl = pq.read_table(path).to_pydict()
        candles = []
        for i in range(len(tbl["open_time"])):
            candles.append(Candle(
                symbol=sym, timeframe="1D",
                open_time=tbl["open_time"][i], close_time=tbl["close_time"][i],
                open=tbl["open"][i], high=tbl["high"][i],
                low=tbl["low"][i],   close=tbl["close"][i],
                volume=tbl["volume"][i],
                buy_volume=0.0, sell_volume=0.0, trade_count=0,
                vwap=tbl["close"][i], spread_avg=0.0, spread_max=0.0,
                volatility_estimate=0.0, is_final=True,
            ))
        candles.sort(key=lambda c: c.open_time)
        all_candles[sym] = candles
        _log(f"  {sym}: {len(candles)} historical bars")

    equity    = state["equity"]
    positions: dict = {}
    pending:   dict = {}

    all_dates = sorted({
        datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        for sym_candles in all_candles.values()
        for c in sym_candles
    })

    for date_str in all_dates:
        for sym in _SYMBOLS:
            sym_candles = all_candles.get(sym, [])
            today_c = next(
                (c for c in sym_candles
                 if datetime.fromtimestamp(c.open_time/1000, tz=timezone.utc).strftime("%Y-%m-%d") == date_str),
                None,
            )
            if today_c is None:
                continue
            subset = [c for c in sym_candles if c.open_time <= today_c.open_time]

            if sym in pending:
                pe      = pending.pop(sym)
                sig_dir = Direction(pe["direction"])
                sig_atr = pe["signal_atr"]
                entry_price = today_c.open or today_c.close
                initial_stop = (
                    entry_price - _INIT_MULT * sig_atr if sig_dir is Direction.LONG
                    else entry_price + _INIT_MULT * sig_atr
                )
                stop_dist = abs(entry_price - initial_stop)
                if sym in positions:
                    old = positions.pop(sym)
                    old_dir = Direction(old["direction"])
                    exit_fee = entry_price * old["size"] * _TAKER_FEE
                    raw_pnl  = (entry_price - old["entry_price"]) * old["size"] * (
                        1 if old_dir is Direction.LONG else -1)
                    equity += raw_pnl - exit_fee
                if stop_dist > 0:
                    risk_amount = equity * _RISK_PCT
                    size = risk_amount / stop_dist
                    entry_fee = entry_price * size * _TAKER_FEE
                    equity -= entry_fee
                    positions[sym] = {
                        "direction": sig_dir.value, "entry_price": entry_price,
                        "entry_time_ms": today_c.open_time, "size": size,
                        "initial_stop": initial_stop, "trail_stop": initial_stop,
                        "best_close": entry_price, "fees_paid": entry_fee,
                        "stop_order_id": None,
                    }

            if sym in positions:
                pos = positions[sym]
                pos_dir = Direction(pos["direction"])
                if ((pos_dir is Direction.LONG  and today_c.low  <= pos["trail_stop"]) or
                        (pos_dir is Direction.SHORT and today_c.high >= pos["trail_stop"])):
                    exit_price = pos["trail_stop"]
                    exit_fee   = exit_price * pos["size"] * _TAKER_FEE
                    raw_pnl    = (exit_price - pos["entry_price"]) * pos["size"] * (
                        1 if pos_dir is Direction.LONG else -1)
                    equity += raw_pnl - exit_fee
                    del positions[sym]

            if sym in positions:
                pos = positions[sym]
                atr = _compute_atr(subset)
                if atr and atr > 0:
                    new_trail, new_best = TrendFollowingStrategy.update_trail(
                        current_trail=pos["trail_stop"], direction=Direction(pos["direction"]),
                        bar_close=today_c.close, best_close=pos["best_close"],
                        atr=atr, trail_mult=_TRAIL_MULT,
                    )
                    pos["trail_stop"] = new_trail
                    pos["best_close"] = new_best

            strategy = TrendFollowingStrategy(
                lookback=_LOOKBACK, atr_period=_ATR_PERIOD,
                initial_stop_mult=_INIT_MULT, trail_mult=_TRAIL_MULT,
            )
            signal = None
            for c in subset:
                signal = strategy.on_candle(c)
            if signal is not None:
                current_dir = Direction(positions[sym]["direction"]) if sym in positions else None
                if signal.direction != current_dir:
                    pending[sym] = {
                        "direction": signal.direction.value,
                        "signal_atr": signal.atr,
                        "signal_date": date_str,
                    }

    state["equity"]              = equity
    state["positions"]           = positions
    state["pending_entries"]     = pending
    state["last_processed_date"] = all_dates[-1] if all_dates else None
    _save_state(state)
    _log("Backfill complete.")
    _print_status(state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Load .env if present (simple key=value format, no dependencies required)
    env_file = _REPO_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    parser = argparse.ArgumentParser(description="Engine #1 daemon")
    parser.add_argument(
        "--mode",
        default=None,
        choices=[m.value for m in ExecMode if m is not ExecMode.BACKTEST],
        help="Execution mode (overrides NEXFLOW_EXEC_MODE env var)",
    )
    parser.add_argument("--equity",   type=float, default=100_000.0)
    parser.add_argument("--run-now",  action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--status",   action="store_true")
    parser.add_argument("--data-dir", default="data/candles")
    args = parser.parse_args()

    # Resolve execution mode
    if args.mode:
        os.environ["NEXFLOW_EXEC_MODE"] = args.mode
    exec_mode = ExecMode.from_env()

    data_dir = _REPO_ROOT / args.data_dir
    state    = _load_state(args.equity)

    _log(f"Engine #1 startup  mode={exec_mode.value}")

    if args.status:
        _print_status(state)
        return

    # Build adapter (no adapter needed for status or backfill in LOCAL_PAPER)
    if exec_mode is not ExecMode.BACKTEST:
        adapter = build_adapter(exec_mode)
    else:
        _log("[ERROR] Use scripts/run_trend_backtest.py for BACKTEST mode")
        sys.exit(1)

    # Sync exchange state on startup (BITGET_PAPER only)
    if exec_mode is ExecMode.BITGET_PAPER and not args.backfill:
        _sync_from_exchange(state, adapter)

    if args.backfill:
        _run_backfill(state, data_dir, exec_mode)
        return

    if args.run_now:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state["last_processed_date"] = None
        _run_cycle(state, today_str, adapter, exec_mode)
        return

    _run_live(state, adapter, exec_mode)


if __name__ == "__main__":
    main()
