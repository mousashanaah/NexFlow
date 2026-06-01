#!/usr/bin/env python3
"""Engine #1 paper trading daemon — daily trend following.

Runs as an always-on process. Wakes at 00:05 UTC each day (5 minutes
after Bitget USDT-Futures daily candle close), fetches the latest candles,
evaluates stops, updates trails, generates signals, and records paper orders.

State is persisted to data/engine1_state.json after every cycle so the
process can be restarted without losing positions.

Architecture:
  - Fetch last 60 daily candles via REST (enough for ATR warmup + lookback)
  - For each open position: check if today's bar triggered the stop
  - Update trailing stops using today's close
  - Feed candles to strategy; detect new signals on today's bar
  - Pending entries execute at next day's open (correct no-lookahead behaviour)
  - All events written to data/engine1_journal.csv

Usage:
  python scripts/paper_engine1.py                  # live mode, waits for 00:05 UTC
  python scripts/paper_engine1.py --backfill        # process historical candles once
  python scripts/paper_engine1.py --run-now         # trigger one cycle immediately (testing)
  python scripts/paper_engine1.py --equity 5000     # set starting equity
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.strategy.signal_models import Direction
from nexflow.services.strategy.trend_strategy import TrendFollowingStrategy, _wilder_atr

# ---------------------------------------------------------------------------
# Configuration — mirrors run_trend_backtest.py exactly
# ---------------------------------------------------------------------------
_SYMBOLS      = ["BTCUSDT", "ETHUSDT"]
_LOOKBACK     = 20
_ATR_PERIOD   = 14
_INIT_MULT    = 2.5
_TRAIL_MULT   = 2.0
_RISK_PCT     = 0.01
_MAX_RISK_PCT = 0.03
_TAKER_FEE    = 0.0006
_CANDLES_NEEDED = 60   # warmup buffer for ATR + lookback

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
    "r_multiple", "pnl", "equity", "note",
]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state(initial_equity: float) -> dict:
    if _STATE_FILE.exists():
        with open(_STATE_FILE) as f:
            return json.load(f)
    return {
        "equity":             initial_equity,
        "last_processed_date": None,
        "positions":          {},   # symbol → position dict
        "pending_entries":    {},   # symbol → pending entry dict
    }


def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _journal_write(row: dict) -> None:
    _JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not _JOURNAL_FILE.exists()
    with open(_JOURNAL_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_JOURNAL_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _JOURNAL_FIELDS})


def _log(msg: str) -> None:
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Candle fetching
# ---------------------------------------------------------------------------

def _fetch_daily_candles(symbol: str, n: int = _CANDLES_NEEDED) -> list[Candle]:
    """Fetch the last N daily candles via history-candles (backward pagination)."""
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


# ---------------------------------------------------------------------------
# ATR helper (same as backtest)
# ---------------------------------------------------------------------------

def _compute_atr(candles: list[Candle]) -> float | None:
    highs  = [c.high  for c in candles]
    lows   = [c.low   for c in candles]
    closes = [c.close for c in candles]
    return _wilder_atr(highs, lows, closes, _ATR_PERIOD)


# ---------------------------------------------------------------------------
# Daily cycle
# ---------------------------------------------------------------------------

def _run_cycle(state: dict, today_str: str, dry_run: bool = False) -> None:
    """Process one daily cycle. Modifies state in-place."""
    if state.get("last_processed_date") == today_str:
        _log(f"  Cycle {today_str} already processed — skipping (idempotent)")
        return

    _log(f"── Daily cycle: {today_str} ──────────────────────────────────────")

    equity = state["equity"]
    positions      = state["positions"]
    pending_entries = state["pending_entries"]

    for symbol in _SYMBOLS:
        _log(f"  [{symbol}] fetching candles ...")
        try:
            candles = _fetch_daily_candles(symbol, _CANDLES_NEEDED)
        except Exception as exc:
            _log(f"  [{symbol}] ERROR fetching candles: {exc}")
            continue

        if len(candles) < _LOOKBACK + _ATR_PERIOD + 2:
            _log(f"  [{symbol}] insufficient candle history ({len(candles)} bars) — skipping")
            continue

        today_candle = candles[-1]
        today_date   = datetime.fromtimestamp(
            today_candle.open_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        # ----------------------------------------------------------------
        # A. Execute pending entry from yesterday's signal
        # ----------------------------------------------------------------
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
                old = positions.pop(symbol)
                old_dir    = Direction(old["direction"])
                exit_price = entry_price
                exit_fee   = exit_price * old["size"] * _TAKER_FEE
                raw_pnl    = (exit_price - old["entry_price"]) * old["size"] * (
                    1 if old_dir is Direction.LONG else -1)
                net_pnl = raw_pnl - exit_fee
                equity += raw_pnl - exit_fee
                risk_amount = abs(old["entry_price"] - old["initial_stop"]) * old["size"]
                r_mult = net_pnl / risk_amount if risk_amount > 0 else 0.0
                _log(f"  [{symbol}] FLIP CLOSE {old_dir.value} @ ${exit_price:,.2f}  "
                     f"PnL ${net_pnl:+,.2f}  {r_mult:+.2f}R")
                _journal_write({
                    "date": today_date, "symbol": symbol, "event": "FLIP_CLOSE",
                    "direction": old_dir.value, "price": round(exit_price, 4),
                    "size": round(old["size"], 6), "atr": round(sig_atr, 4),
                    "stop": round(old["trail_stop"], 4),
                    "r_multiple": round(r_mult, 3), "pnl": round(net_pnl, 2),
                    "equity": round(equity, 2), "note": "direction flip",
                })

            if stop_dist > 0:
                risk_amount = equity * _RISK_PCT
                size        = risk_amount / stop_dist
                # Skip if minimum size implies > max risk
                if stop_dist * max(size, 0.001) / equity > _MAX_RISK_PCT and size < 0.001:
                    _log(f"  [{symbol}] skip entry — position too large for account size")
                else:
                    entry_fee = entry_price * size * _TAKER_FEE
                    equity   -= entry_fee
                    positions[symbol] = {
                        "direction":    sig_dir.value,
                        "entry_price":  entry_price,
                        "entry_time_ms": today_candle.open_time,
                        "size":         size,
                        "initial_stop": initial_stop,
                        "trail_stop":   initial_stop,
                        "best_close":   entry_price,
                        "fees_paid":    entry_fee,
                    }
                    _log(f"  [{symbol}] ENTER {sig_dir.value} @ ${entry_price:,.2f}  "
                         f"stop ${initial_stop:,.2f}  size {size:.6f}  "
                         f"risk ${risk_amount:.2f} ({_RISK_PCT*100:.0f}%)")
                    _journal_write({
                        "date": today_date, "symbol": symbol, "event": "ENTRY",
                        "direction": sig_dir.value, "price": round(entry_price, 4),
                        "size": round(size, 6), "atr": round(sig_atr, 4),
                        "stop": round(initial_stop, 4),
                        "r_multiple": "", "pnl": round(-entry_fee, 2),
                        "equity": round(equity, 2), "note": f"signal from {pe['signal_date']}",
                    })

        # ----------------------------------------------------------------
        # B. Check if today's bar hit the trailing stop
        # ----------------------------------------------------------------
        if symbol in positions:
            pos     = positions[symbol]
            pos_dir = Direction(pos["direction"])
            stop_hit = (
                (pos_dir is Direction.LONG  and today_candle.low  <= pos["trail_stop"]) or
                (pos_dir is Direction.SHORT and today_candle.high >= pos["trail_stop"])
            )
            if stop_hit:
                exit_price = pos["trail_stop"]
                exit_fee   = exit_price * pos["size"] * _TAKER_FEE
                raw_pnl    = (exit_price - pos["entry_price"]) * pos["size"] * (
                    1 if pos_dir is Direction.LONG else -1)
                net_pnl  = raw_pnl - exit_fee
                equity  += raw_pnl - exit_fee
                risk_amount = abs(pos["entry_price"] - pos["initial_stop"]) * pos["size"]
                r_mult = net_pnl / risk_amount if risk_amount > 0 else 0.0
                hold_days = (today_candle.open_time - pos["entry_time_ms"]) / _DAY_MS
                _log(f"  [{symbol}] STOP {pos_dir.value} @ ${exit_price:,.2f}  "
                     f"PnL ${net_pnl:+,.2f}  {r_mult:+.2f}R  held {hold_days:.0f}d")
                _journal_write({
                    "date": today_date, "symbol": symbol, "event": "STOP",
                    "direction": pos_dir.value, "price": round(exit_price, 4),
                    "size": round(pos["size"], 6), "atr": "",
                    "stop": round(pos["trail_stop"], 4),
                    "r_multiple": round(r_mult, 3), "pnl": round(net_pnl, 2),
                    "equity": round(equity, 2), "note": f"held {hold_days:.0f} days",
                })
                del positions[symbol]

        # ----------------------------------------------------------------
        # C. Update trailing stop for still-open position
        # ----------------------------------------------------------------
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
                    _log(f"  [{symbol}] trail updated: ${pos['trail_stop']:,.2f} → ${new_trail:,.2f}")
                pos["trail_stop"] = new_trail
                pos["best_close"] = new_best

        # ----------------------------------------------------------------
        # D. Run strategy to detect new signal on today's bar
        # ----------------------------------------------------------------
        strategy = TrendFollowingStrategy(
            lookback=_LOOKBACK, atr_period=_ATR_PERIOD,
            initial_stop_mult=_INIT_MULT, trail_mult=_TRAIL_MULT,
        )
        signal = None
        for c in candles:
            signal = strategy.on_candle(c)
        # signal now reflects today's candle

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
                    "note": "entry tomorrow at open",
                })
        else:
            status = f"position {positions[symbol]['direction']} stop ${positions[symbol]['trail_stop']:,.2f}" \
                if symbol in positions else "flat"
            _log(f"  [{symbol}] no signal  ({status})")

    # ----------------------------------------------------------------
    # End of cycle: save state and print summary
    # ----------------------------------------------------------------
    state["equity"]              = equity
    state["positions"]           = positions
    state["pending_entries"]     = pending_entries
    state["last_processed_date"] = today_str

    _save_state(state)
    _print_status(state)


def _print_status(state: dict) -> None:
    _log("  ── Status ──")
    _log(f"  Equity   : ${state['equity']:>12,.2f}")
    if state["positions"]:
        for sym, pos in state["positions"].items():
            entry_dt = datetime.fromtimestamp(
                pos["entry_time_ms"] / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            hold = (datetime.now(timezone.utc).timestamp() * 1000 - pos["entry_time_ms"]) / _DAY_MS
            r_unrealised = ""  # would need current price — not fetched at end of cycle
            _log(
                f"  {sym:<12} {pos['direction']:<6} entry ${pos['entry_price']:,.2f} "
                f"on {entry_dt}  trail ${pos['trail_stop']:,.2f}  held {hold:.0f}d"
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
    """Return the next 00:05 UTC trigger time."""
    now = datetime.now(timezone.utc)
    candidate = now.replace(
        hour=_TRIGGER_HOUR_UTC, minute=_TRIGGER_MINUTE_UTC,
        second=0, microsecond=0,
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _run_live(state: dict) -> None:
    _log("Engine #1 paper daemon starting. Waiting for next 00:05 UTC trigger.")
    _print_status(state)
    while True:
        trigger = _next_trigger()
        wait_s  = (trigger - datetime.now(timezone.utc)).total_seconds()
        _log(f"  Next trigger: {trigger.strftime('%Y-%m-%d %H:%M UTC')}  "
             f"(in {wait_s/3600:.1f}h)")
        time.sleep(max(wait_s - 30, 0))  # wake up 30s early to be precise
        # Spin-wait for the exact trigger
        while datetime.now(timezone.utc) < trigger:
            time.sleep(1)
        today_str = trigger.strftime("%Y-%m-%d")
        try:
            _run_cycle(state, today_str)
        except Exception as exc:
            _log(f"  [ERROR] Cycle failed: {exc}")
            import traceback
            _log(traceback.format_exc())


def _run_backfill(state: dict, data_dir: Path) -> None:
    """Process historical parquet data to populate state as if running live."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[ERROR] pyarrow required for backfill mode")
        sys.exit(1)

    _log("Backfill mode: replaying historical candles from parquet files ...")
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

    # Run strategy over historical candles to establish current position state
    # (same logic as daily cycle but applied to parquet data instead of live API)
    _log("  Replaying to establish current position state ...")
    equity = state["equity"]
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
            # candles up to and including this date
            today_c = next(
                (c for c in sym_candles
                 if datetime.fromtimestamp(c.open_time/1000, tz=timezone.utc).strftime("%Y-%m-%d") == date_str),
                None
            )
            if today_c is None:
                continue
            # subset up to today
            subset = [c for c in sym_candles if c.open_time <= today_c.open_time]

            # Execute pending entry
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
                    }

            # Check stop
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

            # Update trail
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

            # Check signal
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

    state["equity"]           = equity
    state["positions"]        = positions
    state["pending_entries"]  = pending
    state["last_processed_date"] = all_dates[-1] if all_dates else None
    _save_state(state)
    _log("Backfill complete.")
    _print_status(state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Engine #1 paper trading daemon")
    parser.add_argument("--equity",   type=float, default=100_000.0,
                        help="Starting paper equity (only used for a fresh state file)")
    parser.add_argument("--run-now",  action="store_true",
                        help="Trigger one cycle immediately then exit")
    parser.add_argument("--backfill", action="store_true",
                        help="Replay historical parquet data to build current state")
    parser.add_argument("--status",   action="store_true",
                        help="Print current state and exit")
    parser.add_argument("--data-dir", default="data/candles")
    args = parser.parse_args()

    data_dir = _REPO_ROOT / args.data_dir

    state = _load_state(args.equity)

    if args.status:
        _print_status(state)
        return

    if args.backfill:
        _run_backfill(state, data_dir)
        return

    if args.run_now:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Allow reprocessing today for --run-now
        state["last_processed_date"] = None
        _run_cycle(state, today_str)
        return

    _run_live(state)


if __name__ == "__main__":
    main()
