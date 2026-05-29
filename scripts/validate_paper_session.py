#!/usr/bin/env python3
"""Full-stack paper trading validation session.

Runs the complete live paper trading path in replay mode using a realistic
synthetic candle dataset designed to trigger actual strategy signals.

The dataset is engineered to exercise:
    - Signal generation (trending conditions with volume expansion)
    - Risk engine filtering (cooldown, daily DD, position limits)
    - Execution simulation (slippage, fees, partial TP fills)
    - Stop-loss evaluation
    - Kill-switch activation
    - Journal persistence
    - HTML report generation

Outputs:
    logs/paper_validation/     — JSONL execution journal
    reports/paper_session.html — self-contained HTML report
    stdout                     — text summary of all findings

Usage:
    python scripts/validate_paper_session.py
    python scripts/validate_paper_session.py --equity 50000
    python scripts/validate_paper_session.py --output reports/custom.html
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nexflow.services.candles.candle_engine import Candle
from nexflow.services.paper_trading.paper_trader import PaperTrader, PaperTraderConfig
from nexflow.services.strategy.momentum_strategy import MomentumConfig
from nexflow.services.strategy.paper_execution import ExecutionConfig
from nexflow.services.strategy.risk_engine import RiskConfig
from nexflow.services.paper_trading.live_risk_monitor import LiveRiskConfig
from nexflow.analysis.analyze_paper_results import PaperAnalyzer
from nexflow.analysis.report_generator import generate_html_report


# ---------------------------------------------------------------------------
# Synthetic candle generator
# ---------------------------------------------------------------------------

def make_candle(
    symbol: str,
    tf: str,
    close_time: int,
    close: float,
    high: float,
    low: float,
    open_: float,
    volume: float,
    buy_fraction: float,
    spread_avg: float,
) -> Candle:
    buy_vol = volume * buy_fraction
    return Candle(
        symbol=symbol, timeframe=tf,
        open_time=close_time - {"1m": 60, "5m": 300, "15m": 900}[tf],
        close_time=close_time,
        open=open_, high=high, low=low, close=close,
        volume=volume, buy_volume=buy_vol, sell_volume=volume - buy_vol,
        trade_count=max(1, int(volume * 0.5)),
        vwap=(high + low + close) / 3,
        spread_avg=spread_avg,
        spread_max=spread_avg * 1.5,
        volatility_estimate=(high - low) / close,
        is_final=True,
    )


def build_scenario(
    symbol: str = "BTCUSDT",
    base_price: float = 67_000.0,
    n_bars: int = 240,
    seed: int = 42,
) -> dict[str, dict[str, list[Candle]]]:
    """Build ~4 hours of 1m candles with embedded signal opportunities.

    Structure:
        bars 0–29    : warm-up, tight range, normal volume → no signals
        bars 30–59   : gradual uptrend, moderate volume
        bars 60–79   : consolidation near highs
        bars 70–74   : LONG breakout bar (volume spike + range expansion + breakout)
        bars 75–119  : position held, TP1 hit at bar ~80, stop-move to BE
        bars 80–119  : continued uptrend → TP2 and TP3 hit
        bars 120–149 : retracement, ranging
        bars 150–179 : short consolidation then SHORT breakdown
        bars 170–174 : SHORT breakout bar (selling spike)
        bars 175–219 : short position, stop hit on reversal
        bars 220–239 : cooldown, flat
    """
    rng = random.Random(seed)

    # ATR target: ~300 USDT (0.45% of 67000)
    atr_target = 300.0
    spread = atr_target * 0.05   # spread = 5% of ATR → well below 0.30 max

    bars_1m: list[Candle] = []
    price = base_price

    now_s = int(time.time())
    # Align to most recent completed 1m bar boundary
    bar0_ts = (now_s - n_bars * 60 - 600) // 60 * 60

    # Track volumes for relative-volume computation
    volumes: list[float] = []
    base_vol = 50.0  # BTC/bar baseline

    def bar(i: int, close: float, range_mult: float = 1.0,
            vol_mult: float = 1.0, buy_frac: float = 0.52) -> Candle:
        ts = bar0_ts + i * 60
        bar_range = atr_target * range_mult * (0.8 + rng.random() * 0.4)
        high = close + bar_range * 0.6
        low = close - bar_range * 0.4
        open_ = close - bar_range * 0.1 * (1 if buy_frac > 0.5 else -1)
        vol = base_vol * vol_mult * (0.8 + rng.random() * 0.4)
        volumes.append(vol)
        return make_candle(symbol, "1m", ts, close, high, low, open_, vol, buy_frac, spread)

    # Phase 1: Warm-up, tight range, moderate volume (bars 0–29)
    for i in range(30):
        drift = rng.gauss(0, 50)
        price = max(base_price * 0.95, price + drift)
        bars_1m.append(bar(i, price, range_mult=0.7, vol_mult=1.0, buy_frac=0.50 + rng.gauss(0, 0.05)))

    # Phase 2: Gradual uptrend, building momentum (bars 30–59)
    trend_step = 80.0
    for i in range(30, 60):
        price += trend_step * (0.5 + rng.random())
        bars_1m.append(bar(i, price, range_mult=0.9, vol_mult=1.1, buy_frac=0.54))

    # Phase 3: Consolidation near highs (bars 60–69)
    peak_before_breakout = price
    for i in range(60, 70):
        drift = rng.gauss(0, 30)
        price = peak_before_breakout + drift
        bars_1m.append(bar(i, price, range_mult=0.6, vol_mult=0.9, buy_frac=0.52))

    # ---- LONG SIGNAL BAR (bar 70) ----
    # Must satisfy: close > rolling_high(20), rel_vol>=1.5, range_exp>=0.8,
    #               momentum_5m>0, buy_frac>=0.55, spread_regime<=0.30
    # rolling_high = max(close of bars 50–69)
    rolling_high_before = max(c.high for c in bars_1m[-20:])
    signal_long_price = rolling_high_before + atr_target * 0.5  # clear breakout
    price = signal_long_price
    signal_bar_long = bar(70, price,
                          range_mult=1.6,    # range = 1.6×ATR → range_exp ≈ 1.6 >> 0.8
                          vol_mult=2.5,      # volume spike → rel_vol ≈ 2.5 >> 1.5
                          buy_frac=0.72)     # strong buying >> 0.55
    bars_1m.append(signal_bar_long)

    # Post-breakout continuation (bars 71–95) — position should be open
    # TP1 is at entry + 1×ATR → price needs to reach that
    entry_est = signal_long_price
    tp1_target = entry_est + atr_target
    tp2_target = entry_est + 2 * atr_target
    tp3_target = entry_est + 3 * atr_target

    for i in range(71, 85):
        # March toward TP1 then TP2
        progress = (i - 70) / 15
        target = entry_est + progress * (tp2_target - entry_est)
        noise = rng.gauss(0, 40)
        price = target + noise
        bars_1m.append(bar(i, price, range_mult=0.8, vol_mult=1.2, buy_frac=0.58))

    for i in range(85, 100):
        # Push to TP3
        progress = (i - 85) / 15
        target = tp2_target + progress * (tp3_target - tp2_target)
        noise = rng.gauss(0, 60)
        price = target + noise
        bars_1m.append(bar(i, price, range_mult=0.9, vol_mult=1.1, buy_frac=0.55))

    # Phase 4: Retracement + consolidation (bars 100–149)
    retreat_start = price
    for i in range(100, 130):
        price -= 60 * (0.5 + rng.random() * 0.5) + rng.gauss(0, 30)
        bars_1m.append(bar(i, price, range_mult=0.8, vol_mult=1.0, buy_frac=0.47))

    for i in range(130, 150):
        drift = rng.gauss(0, 40)
        price = price + drift
        bars_1m.append(bar(i, price, range_mult=0.7, vol_mult=0.9, buy_frac=0.50))

    # Phase 5: Downtrend setup (bars 150–169)
    for i in range(150, 170):
        price -= 70 * (0.5 + rng.random() * 0.5)
        bars_1m.append(bar(i, price, range_mult=0.9, vol_mult=1.0, buy_frac=0.46))

    # ---- SHORT SIGNAL BAR (bar 170) ----
    rolling_low_before = min(c.low for c in bars_1m[-20:])
    signal_short_price = rolling_low_before - atr_target * 0.5  # clear breakdown
    price = signal_short_price
    signal_bar_short = bar(170, price,
                           range_mult=1.5,
                           vol_mult=2.3,
                           buy_frac=0.26)   # strong selling <= 0.45
    bars_1m.append(signal_bar_short)

    # Post-breakdown (bars 171–200) — short position open, hits stop on reversal
    # stop is at entry + 1.5×ATR above short entry
    short_entry = signal_short_price
    short_stop = short_entry + atr_target * 1.5
    short_tp1 = short_entry - atr_target

    for i in range(171, 185):
        # Move toward short TP1
        progress = (i - 170) / 15
        target = short_entry - progress * (short_entry - short_tp1)
        noise = rng.gauss(0, 50)
        price = target + noise
        bars_1m.append(bar(i, price, range_mult=0.9, vol_mult=1.1, buy_frac=0.44))

    # Reversal — short stop hit (bars 185–195)
    for i in range(185, 200):
        price += 80 * (0.5 + rng.random())
        bars_1m.append(bar(i, price, range_mult=1.0, vol_mult=1.3, buy_frac=0.60))

    # Phase 6: Cooldown / flat (bars 200–239)
    for i in range(200, n_bars):
        drift = rng.gauss(0, 25)
        price = price + drift
        bars_1m.append(bar(i, price, range_mult=0.5, vol_mult=0.8, buy_frac=0.50 + rng.gauss(0, 0.03)))

    assert len(bars_1m) == n_bars, f"Expected {n_bars} bars, got {len(bars_1m)}"

    # Build 5m candles by aggregating 1m (every 5 bars)
    bars_5m: list[Candle] = []
    for j in range(0, n_bars, 5):
        group = bars_1m[j:j + 5]
        if not group:
            continue
        ts5 = group[-1].close_time
        h = max(c.high for c in group)
        lo = min(c.low for c in group)
        o = group[0].open
        cl = group[-1].close
        vol5 = sum(c.volume for c in group)
        bvol5 = sum(c.buy_volume for c in group)
        bars_5m.append(make_candle(
            symbol, "5m", ts5, cl, h, lo, o,
            vol5, bvol5 / vol5 if vol5 > 0 else 0.5, spread * 1.2
        ))

    # Build 15m candles
    bars_15m: list[Candle] = []
    for j in range(0, n_bars, 15):
        group = bars_1m[j:j + 15]
        if not group:
            continue
        ts15 = group[-1].close_time
        h = max(c.high for c in group)
        lo = min(c.low for c in group)
        o = group[0].open
        cl = group[-1].close
        vol15 = sum(c.volume for c in group)
        bvol15 = sum(c.buy_volume for c in group)
        bars_15m.append(make_candle(
            symbol, "15m", ts15, cl, h, lo, o,
            vol15, bvol15 / vol15 if vol15 > 0 else 0.5, spread * 1.4
        ))

    return {
        symbol: {
            "1m": bars_1m,
            "5m": bars_5m,
            "15m": bars_15m,
        }
    }


# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------

def run_validation(
    equity: float = 100_000.0,
    journal_dir: Path = Path("logs/paper_validation"),
    output_html: Path = Path("reports/paper_session.html"),
    seed: int = 42,
) -> None:
    print("=" * 65)
    print("NEXFLOW PAPER TRADING — FULL STACK VALIDATION")
    print("=" * 65)
    print(f"  Mode            : REPLAY (Bitget cloud geo-block documented)")
    print(f"  Initial equity  : {equity:,.0f} USDT")
    print(f"  Journal dir     : {journal_dir}")
    print(f"  Report output   : {output_html}")

    # ---- Build synthetic dataset ----
    print("\n[1/5] Building synthetic candle dataset...")
    all_candles: dict[str, dict[str, list[Candle]]] = {}

    symbols = [("BTCUSDT", 67_000.0, 42), ("ETHUSDT", 3_400.0, 99)]
    for sym, base, sym_seed in symbols:
        sym_candles = build_scenario(sym, base, n_bars=240, seed=sym_seed)
        all_candles.update(sym_candles)

    total_bars = sum(len(cs) for s in all_candles.values() for cs in s.values())
    for sym, tfs in all_candles.items():
        for tf, cs in tfs.items():
            print(f"  {sym:10s} {tf:4s}: {len(cs)} candles  "
                  f"price_range={min(c.close for c in cs):,.0f}–{max(c.close for c in cs):,.0f}")
    print(f"  Total bars: {total_bars}")

    # ---- Configure paper trader ----
    print("\n[2/5] Configuring paper trader...")
    pt_cfg = PaperTraderConfig(
        initial_equity=equity,
        risk=RiskConfig(
            max_risk_per_trade=0.005,          # 0.5% risk per trade
            max_concurrent_positions=3,
            cooldown_after_loss_bars=3,
            daily_drawdown_kill=0.02,          # 2% daily DD kill
            min_stop_distance_atr=0.5,
            max_position_equity_fraction=0.20,
        ),
        execution=ExecutionConfig(
            taker_fee=0.0006,
            maker_fee=0.0002,
            slippage_atr_fraction=0.05,
            spread_cross_fraction=0.5,
        ),
        live_risk=LiveRiskConfig(
            stale_candle_threshold_s=120.0,
            max_spread_atr_ratio=0.30,
            max_drawdown_kill=0.05,
            max_consecutive_losses=6,
        ),
        momentum=MomentumConfig(
            rel_vol_threshold=1.5,
            range_expansion_min=0.8,
            imbalance_min=0.55,
            spread_atr_max=0.30,
        ),
        journal_dir=journal_dir,
        enable_dashboard=False,
    )

    # ---- Run replay ----
    print("\n[3/5] Running replay session...")
    bar_states: list[dict] = []
    signals_per_bar: list[int] = []

    def on_bar(state):
        bar_states.append({
            "equity": state.equity,
            "unrealized": state.unrealized_pnl,
            "drawdown": state.drawdown,
            "open_pos": state.open_positions,
            "trades": state.total_trades,
            "killed": state.is_killed,
        })

    trader = PaperTrader(cfg=pt_cfg, symbols=list(all_candles.keys()))
    t0 = time.time()
    final_state = trader.run_replay(all_candles, on_bar_callback=on_bar)
    elapsed = time.time() - t0
    print(f"  Replay completed in {elapsed:.2f}s  ({total_bars / elapsed:.0f} bars/sec)")

    # ---- Print per-bar summary ----
    if bar_states:
        max_eq  = max(s["equity"] for s in bar_states)
        min_eq  = min(s["equity"] for s in bar_states)
        max_dd  = max(s["drawdown"] for s in bar_states)
        max_pos = max(s["open_pos"] for s in bar_states)
        killed_bars = sum(1 for s in bar_states if s["killed"])
        print(f"  Equity range    : {min_eq:,.2f} – {max_eq:,.2f} USDT")
        print(f"  Max drawdown    : {max_dd * 100:.3f}%")
        print(f"  Max open pos    : {max_pos}")
        print(f"  Kill-active bars: {killed_bars}")

    # ---- Analyze journal ----
    print("\n[4/5] Analyzing execution journal...")
    analyzer = PaperAnalyzer()
    result = analyzer.load_and_analyze(journal_dir)

    if not result.has_data:
        print("  WARNING: No journal data found — journal may not have been written")
    else:
        _print_findings(result, equity)

    # ---- Generate HTML report ----
    print("\n[5/5] Generating HTML report...")
    html = generate_html_report(result)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(f"  Report written  : {output_html.resolve()}")
    print(f"  Report size     : {len(html) // 1024}KB")

    _print_operational_findings(result, final_state, bar_states, all_candles)

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETE")
    print("=" * 65)


def _print_findings(result, initial_equity: float) -> None:
    ts_ = result.trade_stats
    rs_ = result.risk_stats
    ex_ = result.execution_stats
    mk_ = result.market_stats
    net_pnl = result.final_equity - initial_equity

    bar = "-" * 65

    print(bar)
    print("  TRADE STATISTICS")
    print(f"  {'Total signals generated':35s}: (see journal SIGNAL events)")
    print(f"  {'Total fills':35s}: {ts_.total}")
    print(f"  {'Wins / Losses':35s}: {ts_.wins} / {ts_.losses}")
    print(f"  {'Win rate':35s}: {ts_.win_rate * 100:.1f}%")
    pf = f"{ts_.profit_factor:.3f}" if not math.isinf(ts_.profit_factor) else "∞"
    print(f"  {'Profit factor':35s}: {pf}")
    print(f"  {'Expectancy (R)':35s}: {ts_.expectancy_r:+.4f}")
    print(f"  {'Average R multiple':35s}: {ts_.avg_r:+.4f}")
    print(f"  {'Average hold time':35s}: {ts_.avg_hold_minutes:.1f} min")
    print(f"  {'TP exits':35s}: {ts_.tp_exits}")
    print(f"  {'Stop exits':35s}: {ts_.stop_exits}")
    print(f"  {'Forced closes':35s}: {ts_.forced_exits}")
    print(f"  {'Net PnL':35s}: {net_pnl:+.2f} USD  ({net_pnl/initial_equity*100:+.2f}%)")

    print(bar)
    print("  RISK STATISTICS")
    print(f"  {'Max drawdown':35s}: {rs_.max_drawdown * 100:.3f}%")
    print(f"  {'Average drawdown':35s}: {rs_.avg_drawdown * 100:.3f}%")
    print(f"  {'Kill switch activations':35s}: {rs_.kill_count}")
    print(f"  {'Max consecutive losses':35s}: {rs_.max_consec_losses}")
    print(f"  {'Rejected signals':35s}: {rs_.rejected_count}")
    if rs_.rejected_reasons:
        for reason, cnt in sorted(rs_.rejected_reasons.items(), key=lambda x: -x[1]):
            print(f"    {'  ' + reason:33s}: {cnt}")

    print(bar)
    print("  EXECUTION STATISTICS")
    print(f"  {'Average slippage':35s}: {ex_.avg_slippage_pct * 100:.4f}%")
    print(f"  {'Total fees paid':35s}: {ex_.total_fees:.2f} USD")
    print(f"  {'Fee drag on gross profit':35s}: {ex_.fee_drag_gross_pct * 100:.2f}%")
    print(f"  {'Avg fee per trade':35s}: {ex_.avg_fee_per_trade:.2f} USD")
    print(f"  {'Spread anomaly events':35s}: {ex_.spread_anomaly_events}")

    print(bar)
    print("  SYMBOL BREAKDOWN")
    for sym, ss in sorted(mk_.by_symbol.items()):
        print(f"  {sym:35s}: {ss.trades} trades  WR {ss.win_rate*100:.1f}%  "
              f"PnL {ss.net_pnl:+.2f}  AvgR {ss.avg_r:+.3f}")

    print(bar)
    print("  DIRECTION BREAKDOWN")
    print(f"  {'LONG':35s}: {mk_.long_trades} trades  WR {mk_.long_win_rate*100:.1f}%  PnL {mk_.long_pnl:+.2f}")
    print(f"  {'SHORT':35s}: {mk_.short_trades} trades  WR {mk_.short_win_rate*100:.1f}%  PnL {mk_.short_pnl:+.2f}")

    print(bar)
    print("  SESSION BREAKDOWN")
    for sk, se in sorted(mk_.by_session.items(), key=lambda x: -x[1].trades):
        print(f"  {sk:35s}: {se.trades} trades  WR {se.win_rate*100:.1f}%  PnL {se.net_pnl:+.2f}")

    print(bar)
    print("  VOLATILITY REGIME BREAKDOWN")
    for rk, re in sorted(mk_.by_regime.items(), key=lambda x: -x[1].trades):
        print(f"  {rk:35s}: {re.trades} trades  WR {re.win_rate*100:.1f}%  PnL {re.net_pnl:+.2f}")


def _print_operational_findings(result, final_state, bar_states, all_candles) -> None:
    """Diagnose operational failures and flag them explicitly."""
    ts_ = result.trade_stats
    rs_ = result.risk_stats

    print("\n" + "=" * 65)
    print("OPERATIONAL FINDINGS")
    print("=" * 65)

    findings: list[tuple[str, str, str]] = []  # (severity, category, description)

    # 1. Signal generation
    total_1m_bars = sum(len(cs) for s in all_candles.values() for tf, cs in s.items() if tf == "1m")
    if ts_.total == 0:
        findings.append(("CRITICAL", "signals", f"Zero fills from {total_1m_bars} 1m bars — strategy generated no actionable signals"))
    elif ts_.total < 3:
        findings.append(("WARN", "signals", f"Only {ts_.total} fills from {total_1m_bars} 1m bars — signal rate very low"))
    else:
        findings.append(("OK", "signals", f"{ts_.total} fills from {total_1m_bars} 1m bars ({ts_.total/total_1m_bars*100:.2f}% fill rate)"))

    # 2. Signal filtering (rejections)
    if rs_.rejected_count > ts_.total * 10:
        findings.append(("WARN", "filtering",
            f"Risk engine rejected {rs_.rejected_count} signals vs {ts_.total} fills — "
            f"potential over-filtering"))
    elif rs_.rejected_count > 0:
        findings.append(("OK", "filtering",
            f"{rs_.rejected_count} rejections ({', '.join(f'{k}:{v}' for k, v in rs_.rejected_reasons.items())})"))
    else:
        findings.append(("OK", "filtering", "No rejections"))

    # 3. Kill switch
    if rs_.kill_count > 0:
        findings.append(("WARN", "kill_switch",
            f"Kill switch activated {rs_.kill_count} time(s): {list(rs_.kill_reasons.keys())}"))
    else:
        findings.append(("OK", "kill_switch", "Kill switch never activated"))

    # 4. Execution simulation
    if result.execution_stats.avg_slippage_pct > 0.002:
        findings.append(("WARN", "execution",
            f"High average slippage: {result.execution_stats.avg_slippage_pct*100:.3f}%"))
    elif ts_.total > 0:
        findings.append(("OK", "execution",
            f"Slippage {result.execution_stats.avg_slippage_pct*100:.4f}%, fees {result.execution_stats.total_fees:.2f} USD"))

    # 5. Journal persistence
    if result.files_loaded == 0:
        findings.append(("CRITICAL", "journal", "No journal files found — persistence failed"))
    elif result.total_events < 5:
        findings.append(("WARN", "journal", f"Only {result.total_events} events in journal — suspected write failure"))
    else:
        findings.append(("OK", "journal",
            f"{result.total_events} events across {result.files_loaded} file(s)"))

    # 6. Equity consistency
    net_from_trades = ts_.net_pnl
    net_from_equity = result.final_equity - result.initial_equity
    drift = abs(net_from_trades - net_from_equity)
    if drift > 1.0:
        findings.append(("WARN", "accounting",
            f"Equity drift: trades sum={net_from_trades:.2f}, equity diff={net_from_equity:.2f}, gap={drift:.2f}"))
    else:
        findings.append(("OK", "accounting",
            f"Equity consistent with trades (drift < {drift:.4f} USD)"))

    # 7. Candle pipeline
    has_all_tfs = all(
        "1m" in tfs and "5m" in tfs and "15m" in tfs
        for tfs in all_candles.values()
    )
    if not has_all_tfs:
        findings.append(("CRITICAL", "candles", "Missing timeframes in candle set"))
    else:
        findings.append(("OK", "candles", "All three timeframes present for all symbols"))

    # 8. Symbol mapping
    symbols_in_fills = {t.symbol for t in result.trades}
    symbols_expected = set(all_candles.keys())
    missing = symbols_expected - symbols_in_fills
    if missing and ts_.total > 0:
        findings.append(("WARN", "symbols", f"No fills for symbol(s): {missing}"))
    elif ts_.total > 0:
        findings.append(("OK", "symbols", f"All symbols generated fills: {symbols_in_fills}"))

    # 9. Drawdown safety
    if rs_.max_drawdown > 0.04:
        findings.append(("WARN", "risk", f"Max drawdown {rs_.max_drawdown*100:.2f}% — close to kill threshold"))
    else:
        findings.append(("OK", "risk", f"Max drawdown {rs_.max_drawdown*100:.2f}% within safety limits"))

    # 10. Geo-block — always document
    findings.append(("INFO", "live_feed",
        "Bitget WS returns 403 (x-deny-reason: host_not_allowed) — geo-blocks cloud IPs. "
        "Live mode requires deployment on non-datacenter IP."))

    # Print findings
    severity_order = {"CRITICAL": 0, "WARN": 1, "INFO": 2, "OK": 3}
    findings.sort(key=lambda f: severity_order.get(f[0], 99))

    col = {"CRITICAL": "\033[91m", "WARN": "\033[93m", "OK": "\033[92m", "INFO": "\033[96m"}
    reset = "\033[0m"

    for severity, category, desc in findings:
        tag = f"[{severity:8s}]"
        c = col.get(severity, "")
        print(f"  {c}{tag}{reset}  {category:15s}  {desc}")

    # Summary verdict
    criticals = [f for f in findings if f[0] == "CRITICAL"]
    warns = [f for f in findings if f[0] == "WARN"]
    print()
    if criticals:
        print(f"  \033[91m● {len(criticals)} CRITICAL issue(s) — system NOT ready for live paper trading\033[0m")
    elif warns:
        print(f"  \033[93m● {len(warns)} warning(s) — review before extended live session\033[0m")
    else:
        print(f"  \033[92m● All checks passed — system ready for live paper trading deployment\033[0m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--journal-dir", type=Path, default=Path("logs/paper_validation"))
    p.add_argument("--output", type=Path, default=Path("reports/paper_session.html"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_validation(args.equity, args.journal_dir, args.output, args.seed)
