#!/usr/bin/env python3
"""Run a backtest over historical candle parquet files.

Usage:
    python scripts/run_backtest.py --data-dir data/candles
    python scripts/run_backtest.py --data-dir data/candles --symbols BTCUSDT --equity 50000
    python scripts/run_backtest.py --data-dir data/candles --rel-vol 1.3 --risk 0.003
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq

from nexflow.config import get_config
from nexflow.logging import configure_logging, get_logger
from nexflow.services.candles.candle_engine import Candle, TIMEFRAMES
from nexflow.services.strategy.backtest_runner import BacktestConfig, BacktestRunner
from nexflow.services.strategy.momentum_strategy import MomentumConfig, MomentumStrategy
from nexflow.services.strategy.paper_execution import ExecutionConfig
from nexflow.services.strategy.risk_engine import RiskConfig
from nexflow.services.strategy.signal_models import BacktestMetrics


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


def _print_metrics(m: BacktestMetrics, initial_equity: float) -> None:
    bar = "=" * 60
    print(bar)
    print("BACKTEST RESULTS")
    print(bar)
    print(f"  Total trades       : {m.total_trades}")
    print(f"  Win rate           : {m.win_rate * 100:.1f}%")
    print(f"  Expectancy (R)     : {m.expectancy:.4f}")
    print(f"  Sharpe (annualised): {m.sharpe:.2f}")
    print(f"  Max drawdown       : {m.max_drawdown * 100:.2f}%")
    print(f"  Profit factor      : {m.profit_factor:.2f}")
    print(f"  Avg hold (bars)    : {m.avg_hold_bars:.1f}")
    print(f"  Net PnL            : {m.net_pnl:+.2f} USD")
    print(f"  Total fees         : {m.total_fees:.2f} USD")
    print(f"  Return             : {m.net_pnl / initial_equity * 100:+.2f}%")
    print(bar)
    print(f"  Long  trades       : {m.long_trades}  win rate: {m.long_win_rate*100:.1f}%")
    print(f"  Short trades       : {m.short_trades}  win rate: {m.short_win_rate*100:.1f}%")
    if m.pnl_distribution:
        sorted_pnl = sorted(m.pnl_distribution)
        n = len(sorted_pnl)
        p5  = sorted_pnl[max(0, int(n * 0.05))]
        p95 = sorted_pnl[min(n - 1, int(n * 0.95))]
        print(f"  PnL p5 / p95       : {p5:+.2f} / {p95:+.2f} USD")
    print(bar)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NexFlow backtester")
    p.add_argument("--data-dir", type=Path, default=Path("data/candles"))
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated, e.g. BTCUSDT,ETHUSDT")
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--risk", type=float, default=0.005,
                   help="Max risk per trade as fraction (default 0.005 = 0.5%%)")
    p.add_argument("--rel-vol", type=float, default=1.5,
                   help="Relative volume threshold (default 1.5)")
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

    log.info("backtest.loading", data_dir=str(args.data_dir), symbols=symbols)
    all_candles = _load_candles(args.data_dir, symbols)
    total_bars = sum(len(tf_c) for s in all_candles.values() for tf_c in s.values())

    if total_bars == 0:
        print(f"No candle data found in {args.data_dir}. Run run_candle_engine.py first.")
        sys.exit(1)

    log.info("backtest.start", total_bars=total_bars)

    strategy = MomentumStrategy(MomentumConfig(rel_vol_threshold=args.rel_vol))
    bt_cfg = BacktestConfig(
        initial_equity=args.equity,
        risk=RiskConfig(max_risk_per_trade=args.risk),
        execution=ExecutionConfig(),
    )
    runner = BacktestRunner(strategy, bt_cfg)
    metrics = runner.run(all_candles)

    _print_metrics(metrics, args.equity)


if __name__ == "__main__":
    main()
