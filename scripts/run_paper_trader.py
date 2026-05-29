#!/usr/bin/env python3
"""Run NexFlow paper trading engine.

Modes:
    live    — connect to Bitget WebSocket, stream live candles, simulate execution
    replay  — feed historical parquet candles through the live stack (deterministic)

Usage:
    python scripts/run_paper_trader.py --mode live
    python scripts/run_paper_trader.py --mode replay --data-dir data/candles
    python scripts/run_paper_trader.py --mode replay --data-dir data/candles --symbols BTCUSDT --equity 50000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

from nexflow.config import get_config
from nexflow.logging import configure_logging, get_logger
from nexflow.services.candles.candle_engine import Candle, TIMEFRAMES
from nexflow.services.paper_trading.paper_trader import PaperTrader, PaperTraderConfig
from nexflow.services.strategy.momentum_strategy import MomentumConfig
from nexflow.services.strategy.paper_execution import ExecutionConfig
from nexflow.services.strategy.risk_engine import RiskConfig
from nexflow.services.paper_trading.live_risk_monitor import LiveRiskConfig


def _load_candles(data_dir: Path, symbols: list[str]) -> dict[str, dict[str, list[Candle]]]:
    all_candles: dict[str, dict[str, list[Candle]]] = {}
    for symbol in symbols:
        all_candles[symbol] = {}
        for tf in TIMEFRAMES:
            path = data_dir / f"{symbol}_{tf}.parquet"
            if not path.exists():
                continue
            rows = pq.read_table(path).to_pylist()
            candles = [
                Candle(
                    symbol=r["symbol"], timeframe=r["timeframe"],
                    open_time=r["open_time"], close_time=r["close_time"],
                    open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                    volume=r["volume"], buy_volume=r["buy_volume"],
                    sell_volume=r["sell_volume"], trade_count=r["trade_count"],
                    vwap=r["vwap"], spread_avg=r["spread_avg"],
                    spread_max=r["spread_max"],
                    volatility_estimate=r["volatility_estimate"],
                    is_final=r["is_final"],
                )
                for r in rows if r.get("is_final")
            ]
            if candles:
                all_candles[symbol][tf] = sorted(candles, key=lambda c: c.close_time)
    return all_candles


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NexFlow paper trading engine")
    p.add_argument("--mode", choices=["live", "replay"], default="replay")
    p.add_argument("--data-dir", type=Path, default=Path("data/candles"),
                   help="Parquet directory (replay mode only)")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated, e.g. BTCUSDT,ETHUSDT")
    p.add_argument("--trade-symbols", type=str, default=None,
                   help="Subset of --symbols to actually trade. Others receive candles "
                        "but are blocked from entry. E.g. --symbols BTCUSDT,ETHUSDT "
                        "--trade-symbols ETHUSDT")
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--risk", type=float, default=0.005)
    p.add_argument("--rel-vol", type=float, default=1.5)
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper"))
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--log-level", type=str, default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)
    log = get_logger(__name__)

    cfg = get_config()
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else cfg.market_data.symbols
    )

    pt_cfg = PaperTraderConfig(
        initial_equity=args.equity,
        risk=RiskConfig(max_risk_per_trade=args.risk),
        execution=ExecutionConfig(),
        live_risk=LiveRiskConfig(),
        momentum=MomentumConfig(rel_vol_threshold=args.rel_vol),
        journal_dir=args.journal_dir,
        enable_dashboard=not args.no_dashboard,
    )

    trade_symbols = (
        [s.strip() for s in args.trade_symbols.split(",") if s.strip()]
        if args.trade_symbols else symbols
    )

    trader = PaperTrader(cfg=pt_cfg, symbols=symbols, trade_symbols=trade_symbols)

    if args.mode == "live":
        log.info("paper_trader.mode", mode="live", symbols=symbols)
        print(f"Starting live paper trading on: {', '.join(symbols)}")
        print(f"Journal: {args.journal_dir}")
        print("Press Ctrl+C to stop.")
        try:
            asyncio.run(trader.run_live())
        except KeyboardInterrupt:
            print("\nStopped by user.")

    else:  # replay
        log.info("paper_trader.mode", mode="replay", symbols=symbols)
        all_candles = _load_candles(args.data_dir, symbols)
        total_bars = sum(len(cs) for s in all_candles.values() for cs in s.values())

        if total_bars == 0:
            print(f"No candle data found in {args.data_dir}. Run run_candle_engine.py first.")
            sys.exit(1)

        print(f"Replay mode: {total_bars} bars across {len(symbols)} symbol(s)")
        print(f"Journal: {args.journal_dir}\n")

        state = trader.run_replay(all_candles)
        _print_final_state(state, args.equity)


def _print_final_state(state, initial_equity: float) -> None:
    bar = "=" * 55
    print(bar)
    print("PAPER TRADER — FINAL STATE")
    print(bar)
    print(f"  Equity           : {state.equity:>12,.2f} USDT")
    pnl = state.equity - initial_equity
    print(f"  Net PnL          : {pnl:>+12.2f} USDT  ({pnl/initial_equity*100:+.2f}%)")
    print(f"  Realized PnL     : {state.realized_pnl:>+12.2f}")
    print(f"  Open Positions   : {state.open_positions}")
    print(f"  Total Trades     : {state.total_trades}")
    print(f"  Win Rate         : {state.win_rate*100:.1f}%")
    print(f"  Max Drawdown obs.: {state.drawdown*100:.3f}%")
    print(f"  Rolling Sharpe   : {state.rolling_sharpe:.3f}")
    print(f"  Kill Switch      : {'ACTIVE' if state.is_killed else 'clear'}")
    if state.kill_reasons:
        print(f"  Kill Reasons     : {', '.join(state.kill_reasons)}")
    print(bar)


if __name__ == "__main__":
    main()
