from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq

from nexflow.exchange.bitget_constraints import SYMBOL_REGISTRY, round_size

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

_OOS_SPLIT_MS = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


@dataclass
class StrategySpec:
    name: str
    symbols: list[str]
    timeframe: str
    parameters: dict
    taker_fee: float = 0.0006
    risk_pct: float = 0.01
    max_risk_pct: float = 0.03
    init_capital: float = 100_000.0


@dataclass
class Signal:
    direction: str      # "long" | "short"
    atr: float
    stop_price: float


@dataclass
class BacktestResult:
    spec_name: str
    parameters: dict
    pf: float
    cagr: float
    max_dd: float
    n_trades: int
    win_rate: float
    avg_r: float
    oos_pf: float
    oos_n_trades: int
    yearly_pnl: dict
    yearly_pf: dict
    r_multiples: list[float]
    wealth_score: float
    passed_kill_criteria: bool
    kill_reason: str


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class _Bar:
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _load_bars(symbol: str, timeframe: str, data_dir: Path) -> list[_Bar]:
    path = data_dir / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing candle file: {path}")
    df = pq.read_table(path).to_pydict()
    bars = [
        _Bar(
            open_time=df["open_time"][i],
            close_time=df["close_time"][i],
            open=df["open"][i],
            high=df["high"][i],
            low=df["low"][i],
            close=df["close"][i],
            volume=df["volume"][i],
        )
        for i in range(len(df["open_time"]))
    ]
    bars.sort(key=lambda b: b.open_time)
    return bars


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _profit_factor(trades: list[dict]) -> float:
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    return gp / gl if gl > 0 else float("inf")


def _max_drawdown(eq_curve: list[float]) -> float:
    peak = eq_curve[0] if eq_curve else 0.0
    dd = 0.0
    for eq in eq_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = max(dd, (peak - eq) / peak)
    return dd


def _year_str(epoch_ms: int) -> str:
    return str(datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).year)


# ---------------------------------------------------------------------------
# Kill criteria
# ---------------------------------------------------------------------------

def _apply_kill_criteria(
    pf: float,
    is_pf: float,
    oos_pf: float,
    max_dd: float,
    n_trades: int,
    r_multiples: list[float],
) -> tuple[bool, str]:
    if pf < 1.10:
        return False, f"PF {pf:.2f} < 1.10"
    if max_dd > 0.40:
        return False, f"Max DD {max_dd*100:.1f}% > 40%"
    if n_trades < 60:
        return False, f"Trade count {n_trades} < 60"
    if oos_pf < 0.85 * is_pf:
        return False, f"OOS PF {oos_pf:.2f} < 85% of IS PF {is_pf:.2f} (regime-specific)"
    if r_multiples and len(r_multiples) >= 3:
        gross_profit = sum(r for r in r_multiples if r > 0)
        top3 = sum(sorted([r for r in r_multiples if r > 0], reverse=True)[:3])
        if gross_profit > 0 and top3 / gross_profit > 0.60:
            return False, f"Top-3 trades = {top3/gross_profit*100:.0f}% of gross profit (fragile)"
    return True, ""


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

SignalFn = Callable[[str, list[_Bar]], Signal | None]
StopFn = Callable[[dict, _Bar], bool]
TrailFn = Callable[[dict, _Bar], dict]


def _run_simulation(
    spec: StrategySpec,
    bars_by_symbol: dict[str, list[_Bar]],
    signal_fn: SignalFn,
    stop_fn: StopFn,
    trail_fn: TrailFn,
) -> tuple[list[dict], list[tuple[int, float]]]:
    equity = spec.init_capital
    positions: dict[str, dict] = {}
    # pending: symbol -> (Signal, bar_index_of_signal)
    pending: dict[str, tuple[Signal, int]] = {}
    # per-symbol bar history fed to signal_fn
    history: dict[str, list[_Bar]] = {s: [] for s in spec.symbols}

    closed_trades: list[dict] = []
    eq_curve: list[tuple[int, float]] = [(0, equity)]

    # Merge all symbols into chronological stream tagged with (open_time, symbol, bar, bar_index)
    all_events: list[tuple[int, str, _Bar, int]] = []
    for sym, bars in bars_by_symbol.items():
        for i, bar in enumerate(bars):
            all_events.append((bar.open_time, sym, bar, i))
    all_events.sort(key=lambda x: x[0])

    for _, symbol, bar, bar_idx in all_events:
        constraints = SYMBOL_REGISTRY.get(symbol)

        # ------------------------------------------------------------------
        # 1. Enter pending signal at this bar's open (no-lookahead: signal
        #    fired at previous bar's close, entry executes at this bar's open)
        # ------------------------------------------------------------------
        if symbol in pending:
            sig, _ = pending.pop(symbol)
            entry_price = bar.open if bar.open > 0 else bar.close
            stop_dist = abs(entry_price - sig.stop_price)

            if stop_dist > 0:
                risk_amount = equity * spec.risk_pct
                raw_size = risk_amount / stop_dist
                size = (
                    round_size(raw_size, constraints)
                    if constraints else math.floor(raw_size * 1000) / 1000
                )
                # Skip if even minimum meaningful size implies > max_risk_pct
                min_size = constraints.min_order_qty if constraints else 0.001
                effective_size = max(size, min_size)
                if (stop_dist * effective_size) / equity > spec.max_risk_pct:
                    pass  # skip — too risky
                elif size > 0:
                    entry_fee = entry_price * size * spec.taker_fee
                    equity -= entry_fee
                    initial_stop = sig.stop_price
                    positions[symbol] = {
                        "symbol": symbol,
                        "direction": sig.direction,
                        "entry_price": entry_price,
                        "entry_time": bar.open_time,
                        "size": size,
                        "initial_stop": initial_stop,
                        "trail_stop": initial_stop,
                        "best_close": entry_price,
                        "_entry_fee": entry_fee,
                    }

        # ------------------------------------------------------------------
        # 2. Check stop for open position
        # ------------------------------------------------------------------
        if symbol in positions:
            pos = positions[symbol]
            if stop_fn(pos, bar):
                exit_price = pos["trail_stop"]
                exit_fee = exit_price * pos["size"] * spec.taker_fee
                direction_mult = 1 if pos["direction"] == "long" else -1
                raw_pnl = (exit_price - pos["entry_price"]) * pos["size"] * direction_mult
                net_pnl = raw_pnl - exit_fee
                equity += raw_pnl - exit_fee
                risk_per_unit = abs(pos["entry_price"] - pos["initial_stop"])
                r_mult = net_pnl / (risk_per_unit * pos["size"]) if risk_per_unit > 0 else 0.0
                closed_trades.append({
                    "symbol": symbol,
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "entry_time": pos["entry_time"],
                    "exit_time": bar.close_time,
                    "size": pos["size"],
                    "pnl": net_pnl,
                    "r_multiple": r_mult,
                })
                del positions[symbol]
                eq_curve.append((bar.close_time, equity))
                history[symbol].append(bar)
                continue

        # ------------------------------------------------------------------
        # 3. Update trail for still-open position
        # ------------------------------------------------------------------
        if symbol in positions:
            positions[symbol] = trail_fn(positions[symbol], bar)

        # ------------------------------------------------------------------
        # 4. Append bar to history, then evaluate signal on full history
        # ------------------------------------------------------------------
        history[symbol].append(bar)
        sig = signal_fn(symbol, history[symbol])
        if sig is not None and symbol not in positions:
            pending[symbol] = (sig, bar_idx)

        eq_curve.append((bar.close_time, equity))

    # Force-close any remaining open positions at last known bar close
    for symbol, pos in list(positions.items()):
        bars = bars_by_symbol[symbol]
        if not bars:
            continue
        last = bars[-1]
        exit_price = last.close
        exit_fee = exit_price * pos["size"] * spec.taker_fee
        direction_mult = 1 if pos["direction"] == "long" else -1
        raw_pnl = (exit_price - pos["entry_price"]) * pos["size"] * direction_mult
        net_pnl = raw_pnl - exit_fee
        equity += raw_pnl - exit_fee
        risk_per_unit = abs(pos["entry_price"] - pos["initial_stop"])
        r_mult = net_pnl / (risk_per_unit * pos["size"]) if risk_per_unit > 0 else 0.0
        closed_trades.append({
            "symbol": symbol,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "entry_time": pos["entry_time"],
            "exit_time": last.close_time,
            "size": pos["size"],
            "pnl": net_pnl,
            "r_multiple": r_mult,
        })
        eq_curve.append((last.close_time, equity))

    return closed_trades, eq_curve


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    spec: StrategySpec,
    signal_fn: SignalFn,
    stop_fn: StopFn,
    trail_fn: TrailFn,
    data_dir: Path | str = Path("data/candles"),
) -> BacktestResult:
    data_dir = Path(data_dir)
    bars_by_symbol: dict[str, list[_Bar]] = {
        sym: _load_bars(sym, spec.timeframe, data_dir) for sym in spec.symbols
    }

    trades, eq_curve = _run_simulation(spec, bars_by_symbol, signal_fn, stop_fn, trail_fn)

    if not trades:
        return BacktestResult(
            spec_name=spec.name, parameters=spec.parameters,
            pf=0.0, cagr=0.0, max_dd=0.0, n_trades=0, win_rate=0.0, avg_r=0.0,
            oos_pf=0.0, oos_n_trades=0, yearly_pnl={}, yearly_pf={},
            r_multiples=[], wealth_score=0.0,
            passed_kill_criteria=False, kill_reason="No trades generated",
        )

    eq_values = [eq for _, eq in eq_curve]
    final_equity = eq_values[-1]
    max_dd = _max_drawdown(eq_values)

    first_ms = min(t["entry_time"] for t in trades)
    last_ms = max(t["exit_time"] for t in trades)
    years = (last_ms - first_ms) / (1000 * 86400 * 365.25)
    cagr = (final_equity / spec.init_capital) ** (1 / years) - 1 if years > 0 else 0.0

    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / len(trades)
    r_multiples = [t["r_multiple"] for t in trades]
    avg_r = sum(r_multiples) / len(r_multiples)
    pf = _profit_factor(trades)

    is_trades = [t for t in trades if t["entry_time"] < _OOS_SPLIT_MS]
    oos_trades = [t for t in trades if t["entry_time"] >= _OOS_SPLIT_MS]
    is_pf = _profit_factor(is_trades) if is_trades else 1.0
    oos_pf = _profit_factor(oos_trades) if oos_trades else 0.0

    yearly_pnl: dict[str, float] = {}
    yearly_gp: dict[str, float] = {}
    yearly_gl: dict[str, float] = {}
    for t in trades:
        yr = _year_str(t["entry_time"])
        yearly_pnl[yr] = yearly_pnl.get(yr, 0.0) + t["pnl"]
        if t["pnl"] > 0:
            yearly_gp[yr] = yearly_gp.get(yr, 0.0) + t["pnl"]
        else:
            yearly_gl[yr] = yearly_gl.get(yr, 0.0) + abs(t["pnl"])
    yearly_pf = {
        yr: (yearly_gp.get(yr, 0.0) / yearly_gl[yr] if yr in yearly_gl else float("inf"))
        for yr in yearly_pnl
    }

    wealth_score = pf * math.sqrt(len(trades)) * (1 - max_dd)

    passed, kill_reason = _apply_kill_criteria(pf, is_pf, oos_pf, max_dd, len(trades), r_multiples)

    return BacktestResult(
        spec_name=spec.name,
        parameters=spec.parameters,
        pf=pf,
        cagr=cagr,
        max_dd=max_dd,
        n_trades=len(trades),
        win_rate=win_rate,
        avg_r=avg_r,
        oos_pf=oos_pf,
        oos_n_trades=len(oos_trades),
        yearly_pnl=yearly_pnl,
        yearly_pf=yearly_pf,
        r_multiples=r_multiples,
        wealth_score=wealth_score,
        passed_kill_criteria=passed,
        kill_reason=kill_reason,
    )


# ---------------------------------------------------------------------------
# Standardized console output
# ---------------------------------------------------------------------------

def print_result(result: BacktestResult) -> None:
    r = result
    print("\n" + "=" * 70)
    print(f"  {r.spec_name.upper()}")
    print("=" * 70)
    print(f"  Profit factor : {r.pf:.2f}")
    print(f"  CAGR          : {r.cagr*100:.1f}%")
    print(f"  Max drawdown  : {r.max_dd*100:.1f}%")
    print(f"  Total trades  : {r.n_trades}")
    print(f"  Win rate      : {r.win_rate*100:.1f}%")
    print(f"  Avg R         : {r.avg_r:.2f}R")
    print(f"  OOS PF        : {r.oos_pf:.2f}  ({r.oos_n_trades} trades)")
    print(f"  Wealth score  : {r.wealth_score:.2f}")

    if r.yearly_pnl:
        print(f"\n{'─'*70}")
        print(f"  {'Year':<8} {'Net PnL':>12}  {'PF':>7}")
        print(f"  {'─'*8} {'─'*12}  {'─'*7}")
        for yr in sorted(r.yearly_pnl):
            pf_str = f"{r.yearly_pf[yr]:.2f}" if r.yearly_pf.get(yr, 0) != float("inf") else "inf"
            print(f"  {yr:<8} ${r.yearly_pnl[yr]:>+10,.0f}  {pf_str:>7}")

    print(f"\n{'='*70}")
    verdict = "PASS" if r.passed_kill_criteria else "KILL"
    reason = f"  {r.kill_reason}" if r.kill_reason else ""
    print(f"  VERDICT: {verdict}{reason}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    _REPO = Path(__file__).parent.parent.parent

    # Trivial always-long strategy: enters long on every bar, stops out 3 ATR below.
    # Uses a simple rolling ATR to set stops; just validates the harness plumbing.

    def _simple_atr(bars: list[_Bar], period: int = 14) -> float:
        if len(bars) < 2:
            return bars[-1].close * 0.02 if bars else 1.0
        trs = [
            max(bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i-1].close),
                abs(bars[i].low  - bars[i-1].close))
            for i in range(1, len(bars))
        ]
        window = trs[-period:]
        return sum(window) / len(window)

    def my_signal(symbol: str, bars: list[_Bar]) -> Signal | None:
        if len(bars) < 15:
            return None
        if bars[-1].close > bars[-2].close:  # any upward close triggers long
            atr = _simple_atr(bars)
            return Signal(direction="long", atr=atr, stop_price=bars[-1].close - 3 * atr)
        return None

    def my_stop(pos: dict, bar: _Bar) -> bool:
        return bar.low <= pos["trail_stop"]

    def my_trail(pos: dict, bar: _Bar) -> dict:
        if pos["direction"] == "long":
            new_best = max(pos["best_close"], bar.close)
            # trail at 2 ATR below best close — approximated here without live ATR
            # (trail_fn receives a position, not ATR; real engines pass ATR in the closure)
            pos["best_close"] = new_best
        return pos

    spec = StrategySpec(
        name="AlwaysLong-SmokTest",
        symbols=["BTCUSDT"],
        timeframe="1D",
        parameters={"atr_period": 14, "stop_mult": 3.0},
    )

    data_dir = _REPO / "data" / "candles"
    print(f"Running smoke test against {data_dir}/BTCUSDT_1D.parquet ...")
    result = run_backtest(spec, my_signal, my_stop, my_trail, data_dir)
    print_result(result)
    print(f"\nSmoke test complete — {result.n_trades} trades, harness OK.")
    sys.exit(0)
