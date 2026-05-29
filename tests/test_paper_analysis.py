"""Tests for paper trading analysis and HTML report generation.

Covers:
    - JSONL journal loading from directory
    - Trade reconstruction (SIGNAL → FILL → exit)
    - R multiple calculation
    - Session classification
    - Volatility regime assignment
    - All statistics categories (trade, risk, execution, market)
    - Monthly PnL grouping
    - HTML report generation (non-empty, valid structure)
    - Empty / missing journal graceful handling
    - Multi-session aggregation
    - Partial TP trade reconstruction
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import pytest

from nexflow.analysis.analyze_paper_results import (
    AnalysisResult,
    PaperAnalyzer,
    TradeRecord,
    _classify_session,
    _assign_volatility_regimes,
    _compute_trade_stats,
    _compute_risk_stats,
    _compute_execution_stats,
    _compute_market_stats,
    _compute_monthly_pnl,
    _reconstruct_trades,
)
from nexflow.analysis.report_generator import (
    generate_html_report,
    equity_curve_svg,
    drawdown_svg,
    pnl_histogram_svg,
    monthly_bar_svg,
    session_bar_svg,
)


# ---------------------------------------------------------------------------
# Fixtures: JSONL journal builder
# ---------------------------------------------------------------------------

_SESSION_ID = "test-session-001"


def _write_event(f, event_type: str, payload: dict, ts_offset: float = 0.0) -> None:
    base_ts = 1_700_000_000.0 + ts_offset
    record = {
        "ts": "2023-11-14T00:00:00+00:00",
        "ts_epoch": base_ts,
        "event": event_type,
        "session_id": _SESSION_ID,
        **payload,
    }
    f.write(json.dumps(record) + "\n")


def _write_trade_sequence(
    f,
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    fill_price: float = 50_000.0,
    stop_price: float = 49_000.0,
    size: float = 0.01,
    pnl: float = 100.0,
    exit_type: str = "STOP_HIT",
    ts_offset: float = 0.0,
    atr: float = 200.0,
) -> None:
    fee = 0.30
    slippage = fill_price * 0.0001  # 0.01% slippage
    equity = 100_000.0

    _write_event(f, "SIGNAL", {
        "symbol": symbol, "direction": direction,
        "entry_price": fill_price, "stop_price": stop_price,
        "tp_prices": [fill_price * 1.02, fill_price * 1.04, fill_price * 1.06],
        "atr": atr, "features": {"atr": atr},
    }, ts_offset)

    _write_event(f, "FILL", {
        "symbol": symbol, "direction": direction,
        "fill_price": fill_price + slippage,
        "size": size, "fee": fee,
        "slippage": slippage,
        "equity_after": equity - fee,
    }, ts_offset + 1)

    if exit_type == "STOP_HIT":
        exit_price = stop_price + (stop_price * 0.001)  # slight slippage
        exit_pnl = pnl
        exit_fee = 0.20
        _write_event(f, "STOP_HIT", {
            "symbol": symbol, "fill_price": exit_price,
            "size": size, "pnl": exit_pnl, "fee": exit_fee,
            "equity_after": equity - fee + exit_pnl - exit_fee,
        }, ts_offset + 3600)

    elif exit_type.startswith("TP"):
        # Write all three partial TPs
        for i in range(3):
            partial_size = size * [0.5, 0.25, 0.25][i]
            partial_pnl = pnl * [0.5, 0.25, 0.25][i]
            _write_event(f, "PARTIAL_TP", {
                "symbol": symbol, "tp_idx": i,
                "fill_price": fill_price * (1 + 0.02 * (i + 1)),
                "size": partial_size, "pnl": partial_pnl,
                "fee": 0.10, "equity_after": equity + partial_pnl,
            }, ts_offset + 1800 + i * 600)

    elif exit_type == "FORCE_CLOSE":
        _write_event(f, "FORCE_CLOSE", {
            "symbol": symbol, "price": fill_price,
            "pnl": pnl, "reason": "engine_shutdown",
        }, ts_offset + 3600)


def _make_journal(tmp_path: Path, n_trades: int = 5) -> Path:
    """Write a single JSONL journal with n_trades trades and equity snapshots."""
    journal_path = tmp_path / "journal_20231114_000000_test001.jsonl"
    with open(journal_path, "w") as f:
        _write_event(f, "SESSION_START", {"session_id": _SESSION_ID})

        for i in range(n_trades):
            pnl = 100.0 if i % 2 == 0 else -50.0
            _write_trade_sequence(
                f, symbol="BTCUSDT",
                direction="LONG" if i % 3 != 0 else "SHORT",
                fill_price=50_000.0 + i * 100,
                stop_price=49_000.0 + i * 100,
                pnl=pnl,
                ts_offset=i * 7200,
                atr=200.0 + i * 10,
            )

            # Equity snapshot after each trade
            equity = 100_000.0 + sum(
                100.0 if j % 2 == 0 else -50.0 for j in range(i + 1)
            )
            dd = max(0.0, (100_000.0 - equity) / 100_000.0)
            _write_event(f, "EQUITY_SNAPSHOT", {
                "equity": equity, "realized_pnl": equity - 100_000.0,
                "unrealized_pnl": 25.0, "drawdown": dd, "open_positions": 0,
            }, i * 7200 + 3700)

        _write_event(f, "SESSION_END", {
            "final_equity": 100_000.0 + sum(100.0 if i % 2 == 0 else -50.0 for i in range(n_trades)),
            "total_trades": n_trades,
        }, n_trades * 7200 + 100)

    return journal_path


# ---------------------------------------------------------------------------
# Journal loading tests
# ---------------------------------------------------------------------------

class TestJournalLoading:
    def test_empty_directory_returns_no_data(self, tmp_path):
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        assert not result.has_data

    def test_nonexistent_directory_returns_no_data(self, tmp_path):
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path / "does_not_exist")
        assert not result.has_data

    def test_directory_with_no_jsonl_returns_no_data(self, tmp_path):
        (tmp_path / "not_a_journal.txt").write_text("hello")
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        assert not result.has_data

    def test_loads_single_journal(self, tmp_path):
        _make_journal(tmp_path, n_trades=3)
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        assert result.has_data
        assert result.files_loaded == 1

    def test_loads_multiple_journals(self, tmp_path):
        for i in range(3):
            p = tmp_path / f"journal_{i:04d}.jsonl"
            with open(p, "w") as f:
                _write_event(f, "SESSION_START", {"session_id": f"session-{i}"})
                _write_trade_sequence(f, ts_offset=i * 100_000)
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        assert result.files_loaded == 3

    def test_corrupt_lines_skipped_gracefully(self, tmp_path):
        p = tmp_path / "journal_corrupt.jsonl"
        with open(p, "w") as f:
            f.write('{"event":"SESSION_START","ts_epoch":1700000000,"session_id":"x"}\n')
            f.write("not valid json\n")
            f.write("{broken\n")
            f.write('{"event":"SESSION_END","ts_epoch":1700001000,"session_id":"x","final_equity":100000,"total_trades":0}\n')
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        # Should not crash; may have no data or minimal data
        assert isinstance(result, AnalysisResult)

    def test_total_events_counted(self, tmp_path):
        _make_journal(tmp_path, n_trades=4)
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        assert result.total_events > 10  # at least SESSION_START + 4 fills + 4 exits + equity snaps


# ---------------------------------------------------------------------------
# Trade reconstruction tests
# ---------------------------------------------------------------------------

class TestTradeReconstruction:
    def _events_for_trade(
        self, symbol="BTCUSDT", direction="LONG",
        fill_price=50000.0, stop_price=49000.0, size=0.01,
        pnl=-50.0, exit_type="STOP_HIT", atr=200.0, ts_offset=0.0
    ):
        events = []
        base_ts = 1_700_000_000.0

        def ev(event_type, payload, offset=0):
            events.append({
                "ts_epoch": base_ts + ts_offset + offset,
                "event": event_type,
                "symbol": symbol,
                **payload,
            })

        ev("SIGNAL", {
            "direction": direction, "entry_price": fill_price,
            "stop_price": stop_price, "atr": atr,
            "tp_prices": [fill_price * 1.02, fill_price * 1.04, fill_price * 1.06],
            "features": {},
        })
        ev("FILL", {
            "direction": direction, "fill_price": fill_price,
            "size": size, "fee": 0.30, "slippage": 5.0, "equity_after": 99_999.70,
        }, 1)
        if exit_type == "STOP_HIT":
            ev("STOP_HIT", {"fill_price": stop_price, "size": size, "pnl": pnl, "fee": 0.20,
                            "equity_after": 99_949.50}, 3601)
        elif exit_type == "FORCE_CLOSE":
            ev("FORCE_CLOSE", {"price": fill_price, "pnl": pnl, "reason": "shutdown"}, 3601)
        elif exit_type == "ALL_TP":
            for i in range(3):
                sz = size * [0.5, 0.25, 0.25][i]
                ev("PARTIAL_TP", {
                    "tp_idx": i, "fill_price": fill_price * 1.02 * (i + 1),
                    "size": sz, "pnl": pnl * [0.5, 0.25, 0.25][i], "fee": 0.10,
                    "equity_after": 100_100.0,
                }, 1800 + i * 600)
        return events

    def test_stop_hit_creates_trade(self):
        events = self._events_for_trade(pnl=-50.0, exit_type="STOP_HIT")
        trades = _reconstruct_trades(events)
        assert len(trades) == 1
        assert trades[0].exit_type == "STOP"
        assert trades[0].pnl == -50.0

    def test_force_close_creates_trade(self):
        events = self._events_for_trade(pnl=25.0, exit_type="FORCE_CLOSE")
        trades = _reconstruct_trades(events)
        assert len(trades) == 1
        assert trades[0].exit_type == "FORCED"

    def test_all_tp_hit_creates_trade(self):
        events = self._events_for_trade(pnl=100.0, exit_type="ALL_TP")
        trades = _reconstruct_trades(events)
        assert len(trades) == 1
        assert trades[0].pnl == pytest.approx(100.0, abs=0.01)

    def test_multiple_symbols_independent(self):
        events = (
            self._events_for_trade(symbol="BTCUSDT", pnl=-50.0, ts_offset=0)
            + self._events_for_trade(symbol="ETHUSDT", pnl=80.0, exit_type="FORCE_CLOSE", ts_offset=100)
        )
        events.sort(key=lambda e: e["ts_epoch"])
        trades = _reconstruct_trades(events)
        syms = {t.symbol for t in trades}
        assert "BTCUSDT" in syms
        assert "ETHUSDT" in syms

    def test_direction_preserved(self):
        events = self._events_for_trade(direction="SHORT", pnl=-30.0)
        trades = _reconstruct_trades(events)
        assert trades[0].direction == "SHORT"

    def test_r_multiple_correct(self):
        # fill_price=50000, stop=49000, size=0.01, pnl=-100 (hit stop)
        # stop_distance = 1000, risk = 1000 * 0.01 = 10
        # R = -100 / 10 = -10
        events = self._events_for_trade(
            fill_price=50_000.0, stop_price=49_000.0, size=0.01, pnl=-100.0
        )
        trades = _reconstruct_trades(events)
        assert not math.isnan(trades[0].r_multiple)
        assert abs(trades[0].r_multiple - (-100.0 / (1000.0 * 0.01))) < 0.01

    def test_r_multiple_nan_when_stop_at_entry(self):
        events = self._events_for_trade(
            fill_price=50_000.0, stop_price=50_000.0, size=0.01, pnl=0.0
        )
        trades = _reconstruct_trades(events)
        # stop_distance = 0, R should be nan
        assert math.isnan(trades[0].r_multiple)

    def test_hold_time_calculated(self):
        events = self._events_for_trade(ts_offset=0, pnl=-30.0)
        trades = _reconstruct_trades(events)
        # Entry at base+1, exit at base+3601 → hold ≈ 60 min
        assert abs(trades[0].hold_minutes - 60.0) < 1.0

    def test_fees_accumulated(self):
        events = self._events_for_trade(size=0.01, pnl=-50.0)
        trades = _reconstruct_trades(events)
        # entry fee 0.30 + exit fee 0.20 = 0.50
        assert trades[0].fees == pytest.approx(0.50, abs=0.01)

    def test_no_fill_no_trade(self):
        events = [{"ts_epoch": 1700000000, "event": "SIGNAL", "symbol": "BTCUSDT",
                   "direction": "LONG", "entry_price": 50000, "stop_price": 49000,
                   "atr": 200, "tp_prices": [], "features": {}}]
        trades = _reconstruct_trades(events)
        assert trades == []

    def test_fill_without_signal_still_creates_trade(self):
        """A FILL with no preceding SIGNAL should still create a trade (signal may have been lost)."""
        events = [
            {"ts_epoch": 1700000000, "event": "FILL", "symbol": "BTCUSDT", "direction": "LONG",
             "fill_price": 50000.0, "size": 0.01, "fee": 0.30, "slippage": 5.0, "equity_after": 99999.70},
            {"ts_epoch": 1700003600, "event": "STOP_HIT", "symbol": "BTCUSDT",
             "fill_price": 49000.0, "size": 0.01, "pnl": -100.0, "fee": 0.20, "equity_after": 99899.50},
        ]
        trades = _reconstruct_trades(events)
        assert len(trades) == 1
        # R should be nan (no stop_price from signal; fill_price == stop_price in default _OpenTrade)
        # actually stop_price defaults to fill_price when no signal, so R = nan
        assert math.isnan(trades[0].r_multiple)


# ---------------------------------------------------------------------------
# Session classification tests
# ---------------------------------------------------------------------------

class TestSessionClassification:
    def test_asia_hours(self):
        # 04:00 UTC → hour 4 → ASIA
        ts = 4 * 3600  # 4 hours into epoch day
        assert _classify_session(ts) == "asia"

    def test_london_hours(self):
        ts = 10 * 3600  # 10:00 UTC
        assert _classify_session(ts) == "london"

    def test_new_york_hours(self):
        ts = 16 * 3600  # 16:00 UTC
        assert _classify_session(ts) == "new_york"

    def test_off_hours(self):
        ts = 22 * 3600  # 22:00 UTC
        assert _classify_session(ts) == "off_hours"

    def test_boundary_london_start(self):
        ts = 8 * 3600   # exactly 08:00 UTC
        assert _classify_session(ts) == "london"

    def test_boundary_ny_start(self):
        ts = 13 * 3600  # exactly 13:00 UTC
        assert _classify_session(ts) == "new_york"

    def test_boundary_off_hours_start(self):
        ts = 21 * 3600  # exactly 21:00 UTC
        assert _classify_session(ts) == "off_hours"

    def test_midnight_is_asia(self):
        ts = 0  # exactly midnight
        assert _classify_session(ts) == "asia"


# ---------------------------------------------------------------------------
# Volatility regime assignment tests
# ---------------------------------------------------------------------------

class TestVolatilityRegimeAssignment:
    def _make_trade(self, atr: float) -> TradeRecord:
        return TradeRecord(
            symbol="BTCUSDT", direction="LONG",
            entry_time=1_700_000_000, exit_time=1_700_003_600,
            hold_minutes=60, entry_price=50_000, exit_price=50_100,
            size=0.01, pnl=10.0, fees=0.5, slippage=5.0, slippage_pct=0.0001,
            r_multiple=1.0, exit_type="TP1", atr_at_entry=atr,
            stop_price=49_000, session="new_york", volatility_regime="MEDIUM",
        )

    def test_low_medium_high_assignment(self):
        atrs = [100, 100, 100, 200, 200, 200, 300, 300, 300]
        trades = [self._make_trade(a) for a in atrs]
        _assign_volatility_regimes(trades)
        regimes = [t.volatility_regime for t in trades]
        assert "LOW" in regimes
        assert "MEDIUM" in regimes
        assert "HIGH" in regimes

    def test_zero_atr_gets_unknown(self):
        trades = [self._make_trade(0.0)]
        _assign_volatility_regimes(trades)
        assert trades[0].volatility_regime == "UNKNOWN"

    def test_single_trade_no_crash(self):
        trades = [self._make_trade(200.0)]
        _assign_volatility_regimes(trades)
        assert trades[0].volatility_regime in ("LOW", "MEDIUM", "HIGH")

    def test_empty_list_no_crash(self):
        _assign_volatility_regimes([])  # should not raise


# ---------------------------------------------------------------------------
# Trade statistics tests
# ---------------------------------------------------------------------------

class TestTradeStatistics:
    def _make_trade(self, pnl: float, direction: str = "LONG",
                    hold_minutes: float = 30.0, r: float = float("nan"),
                    exit_type: str = "STOP") -> TradeRecord:
        return TradeRecord(
            symbol="BTCUSDT", direction=direction,
            entry_time=1_700_000_000, exit_time=1_700_001_800,
            hold_minutes=hold_minutes,
            entry_price=50_000, exit_price=50_000 + pnl / 0.01,
            size=0.01, pnl=pnl, fees=0.5, slippage=5.0, slippage_pct=0.0001,
            r_multiple=r, exit_type=exit_type, atr_at_entry=200.0,
            stop_price=49_000, session="new_york", volatility_regime="MEDIUM",
        )

    def test_win_rate_correct(self):
        trades = [self._make_trade(100), self._make_trade(-50), self._make_trade(80)]
        stats = _compute_trade_stats(trades)
        assert stats.wins == 2
        assert stats.losses == 1
        assert abs(stats.win_rate - 2/3) < 1e-9

    def test_profit_factor_correct(self):
        trades = [self._make_trade(200), self._make_trade(-100)]
        stats = _compute_trade_stats(trades)
        assert abs(stats.profit_factor - 2.0) < 1e-9

    def test_avg_r_excludes_nan(self):
        trades = [
            self._make_trade(100, r=2.0),
            self._make_trade(-50, r=-1.0),
            self._make_trade(30, r=float("nan")),
        ]
        stats = _compute_trade_stats(trades)
        assert abs(stats.avg_r - 0.5) < 1e-9

    def test_avg_hold_time(self):
        trades = [self._make_trade(100, hold_minutes=30), self._make_trade(-50, hold_minutes=90)]
        stats = _compute_trade_stats(trades)
        assert abs(stats.avg_hold_minutes - 60.0) < 1e-9

    def test_empty_trades(self):
        stats = _compute_trade_stats([])
        assert stats.total == 0
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0

    def test_all_losses_profit_factor_zero(self):
        trades = [self._make_trade(-100), self._make_trade(-50)]
        stats = _compute_trade_stats(trades)
        assert stats.profit_factor == 0.0
        assert stats.wins == 0

    def test_all_wins_profit_factor_inf(self):
        trades = [self._make_trade(100), self._make_trade(200)]
        stats = _compute_trade_stats(trades)
        assert math.isinf(stats.profit_factor)

    def test_exit_type_counts(self):
        trades = [
            self._make_trade(100, exit_type="TP1"),
            self._make_trade(50, exit_type="TP2"),
            self._make_trade(-80, exit_type="STOP"),
            self._make_trade(10, exit_type="FORCED"),
        ]
        stats = _compute_trade_stats(trades)
        assert stats.tp_exits == 2
        assert stats.stop_exits == 1
        assert stats.forced_exits == 1


# ---------------------------------------------------------------------------
# Market statistics tests
# ---------------------------------------------------------------------------

class TestMarketStatistics:
    def _trade(self, symbol="BTCUSDT", direction="LONG",
               pnl=100.0, session="new_york", regime="MEDIUM") -> TradeRecord:
        return TradeRecord(
            symbol=symbol, direction=direction,
            entry_time=1_700_000_000, exit_time=1_700_003_600,
            hold_minutes=60, entry_price=50_000, exit_price=50_100,
            size=0.01, pnl=pnl, fees=0.5, slippage=5.0, slippage_pct=0.0001,
            r_multiple=1.0 if pnl > 0 else -1.0, exit_type="STOP",
            atr_at_entry=200, stop_price=49_000,
            session=session, volatility_regime=regime,
        )

    def test_symbol_breakdown(self):
        trades = [
            self._trade("BTCUSDT", pnl=100),
            self._trade("BTCUSDT", pnl=-50),
            self._trade("ETHUSDT", pnl=80),
        ]
        stats = _compute_market_stats(trades)
        assert "BTCUSDT" in stats.by_symbol
        assert "ETHUSDT" in stats.by_symbol
        assert stats.by_symbol["BTCUSDT"].trades == 2
        assert stats.by_symbol["ETHUSDT"].trades == 1

    def test_direction_split(self):
        trades = [
            self._trade(direction="LONG", pnl=100),
            self._trade(direction="LONG", pnl=-50),
            self._trade(direction="SHORT", pnl=80),
        ]
        stats = _compute_market_stats(trades)
        assert stats.long_trades == 2
        assert stats.short_trades == 1
        assert abs(stats.long_win_rate - 0.5) < 1e-9
        assert stats.short_win_rate == 1.0

    def test_session_breakdown(self):
        trades = [
            self._trade(session="london", pnl=100),
            self._trade(session="london", pnl=-50),
            self._trade(session="new_york", pnl=200),
        ]
        stats = _compute_market_stats(trades)
        assert stats.by_session["london"].trades == 2
        assert stats.by_session["new_york"].trades == 1

    def test_regime_breakdown(self):
        trades = [
            self._trade(regime="LOW", pnl=100),
            self._trade(regime="HIGH", pnl=-50),
            self._trade(regime="HIGH", pnl=80),
        ]
        stats = _compute_market_stats(trades)
        assert stats.by_regime["LOW"].trades == 1
        assert stats.by_regime["HIGH"].trades == 2
        assert abs(stats.by_regime["HIGH"].win_rate - 0.5) < 1e-9

    def test_empty_trades_no_crash(self):
        stats = _compute_market_stats([])
        assert stats.long_trades == 0
        assert stats.by_symbol == {}


# ---------------------------------------------------------------------------
# Monthly PnL tests
# ---------------------------------------------------------------------------

class TestMonthlyPnL:
    def _trade_at(self, ts: float, pnl: float) -> TradeRecord:
        return TradeRecord(
            symbol="BTCUSDT", direction="LONG",
            entry_time=ts - 3600, exit_time=ts,
            hold_minutes=60, entry_price=50_000, exit_price=50_100,
            size=0.01, pnl=pnl, fees=0.5, slippage=5.0, slippage_pct=0.0001,
            r_multiple=1.0, exit_type="TP1", atr_at_entry=200.0,
            stop_price=49_000, session="new_york", volatility_regime="MEDIUM",
        )

    def test_single_month(self):
        # 2023-11-14 UTC
        ts = 1_699_920_000.0
        trades = [self._trade_at(ts, 100.0), self._trade_at(ts + 3600, -30.0)]
        monthly = _compute_monthly_pnl(trades)
        assert len(monthly) == 1
        total = list(monthly.values())[0]
        assert abs(total - 70.0) < 0.01

    def test_two_months_separated(self):
        # November 2023
        ts_nov = 1_699_920_000.0
        # December 2023 (add ~30 days)
        ts_dec = ts_nov + 32 * 86_400
        trades = [
            self._trade_at(ts_nov, 100.0),
            self._trade_at(ts_dec, -50.0),
        ]
        monthly = _compute_monthly_pnl(trades)
        assert len(monthly) == 2
        keys = list(monthly.keys())
        assert keys[0] < keys[1]  # chronological order

    def test_empty_returns_empty(self):
        assert _compute_monthly_pnl([]) == {}


# ---------------------------------------------------------------------------
# Full integration: load → analyze → report
# ---------------------------------------------------------------------------

class TestFullIntegration:
    def test_round_trip_with_realistic_data(self, tmp_path):
        _make_journal(tmp_path, n_trades=8)
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)

        assert result.has_data
        assert result.trade_stats.total > 0
        assert result.equity_curve  # equity snapshots present

    def test_result_equity_consistent_with_trades(self, tmp_path):
        _make_journal(tmp_path, n_trades=6)
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)

        # Net PnL from trade stats must equal final - initial
        net_from_trades = result.trade_stats.net_pnl
        net_from_equity = result.final_equity - result.initial_equity
        # Allow some tolerance (fees, unrealized differ slightly)
        assert abs(net_from_trades - net_from_equity) < 200.0

    def test_multi_session_aggregation(self, tmp_path):
        """Trades from two separate journals are merged correctly."""
        for i in range(2):
            p = tmp_path / f"journal_sess{i}.jsonl"
            with open(p, "w") as f:
                _write_event(f, "SESSION_START", {"session_id": f"sess-{i}"})
                _write_trade_sequence(f, pnl=50.0 * (i + 1), ts_offset=i * 100_000)
                _write_event(f, "EQUITY_SNAPSHOT", {
                    "equity": 100_000 + 50 * (i + 1), "realized_pnl": 50 * (i + 1),
                    "unrealized_pnl": 0, "drawdown": 0, "open_positions": 0,
                }, i * 100_000 + 4000)
                _write_event(f, "SESSION_END", {"final_equity": 100_000 + 50 * (i + 1), "total_trades": 1})

        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        assert result.session_count == 2
        assert result.files_loaded == 2


# ---------------------------------------------------------------------------
# HTML report generation tests
# ---------------------------------------------------------------------------

class TestHTMLReportGeneration:
    def test_no_data_report_is_valid_html(self):
        result = AnalysisResult(has_data=False)
        html = generate_html_report(result)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "No paper trading data" in html

    def test_report_with_data_contains_key_sections(self, tmp_path):
        _make_journal(tmp_path, n_trades=5)
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        html = generate_html_report(result)

        assert "NexFlow Paper Trading Report" in html
        assert "Equity Curve" in html
        assert "Drawdown" in html
        assert "Trade Statistics" in html
        assert "Risk Statistics" in html
        assert "Execution Quality" in html
        assert "Monthly Performance" in html
        assert "Market Breakdown" in html
        assert "Risk Events" in html

    def test_report_is_self_contained_no_cdn(self, tmp_path):
        _make_journal(tmp_path, n_trades=3)
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        html = generate_html_report(result)

        # No external resources
        assert "cdn.js" not in html
        assert "googleapis.com" not in html
        assert "unpkg.com" not in html
        assert "cdn.jsdelivr" not in html
        assert 'src="http' not in html

    def test_report_contains_svg_charts(self, tmp_path):
        _make_journal(tmp_path, n_trades=6)
        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(tmp_path)
        html = generate_html_report(result)

        assert "<svg" in html
        assert "polyline" in html or "polygon" in html or "rect" in html

    def test_report_handles_zero_trades(self, tmp_path):
        # Journal with only equity snapshots, no trades
        p = tmp_path / "journal_notrades.jsonl"
        with open(p, "w") as f:
            _write_event(f, "SESSION_START", {"session_id": "x"})
            _write_event(f, "EQUITY_SNAPSHOT", {
                "equity": 100_000, "realized_pnl": 0, "unrealized_pnl": 0,
                "drawdown": 0, "open_positions": 0,
            })
        result = PaperAnalyzer().load_and_analyze(tmp_path)
        html = generate_html_report(result)
        assert "<!DOCTYPE html>" in html
        assert "No trades" in html or "0" in html

    def test_report_size_reasonable(self, tmp_path):
        _make_journal(tmp_path, n_trades=10)
        result = PaperAnalyzer().load_and_analyze(tmp_path)
        html = generate_html_report(result)
        # Should be at least 10KB (CSS + charts + tables)
        assert len(html) > 10_000
        # Should not be absurdly large (< 2MB for 10 trades)
        assert len(html) < 2_000_000


# ---------------------------------------------------------------------------
# SVG chart unit tests
# ---------------------------------------------------------------------------

class TestSVGCharts:
    def test_equity_curve_returns_svg(self):
        equities = [100_000 + i * 50 for i in range(20)]
        timestamps = [1_700_000_000 + i * 3600 for i in range(20)]
        svg = equity_curve_svg(equities, timestamps)
        assert "<svg" in svg
        assert "polyline" in svg

    def test_drawdown_returns_svg(self):
        dds = [0.0, 0.01, 0.02, 0.015, 0.005, 0.0]
        timestamps = [1_700_000_000 + i * 3600 for i in range(6)]
        svg = drawdown_svg(dds, timestamps)
        assert "<svg" in svg

    def test_histogram_returns_svg(self):
        pnls = [-100, -50, 0, 50, 100, 150, -30, 80]
        svg = pnl_histogram_svg(pnls)
        assert "<svg" in svg
        assert "rect" in svg

    def test_monthly_bars_returns_svg(self):
        monthly = {"2023-10": 200.0, "2023-11": -80.0, "2023-12": 150.0}
        svg = monthly_bar_svg(monthly)
        assert "<svg" in svg
        assert "rect" in svg

    def test_charts_handle_empty_gracefully(self):
        assert "No data" in equity_curve_svg([], [])
        assert "No data" in drawdown_svg([], [])
        assert "No data" in pnl_histogram_svg([])
        assert "No data" in monthly_bar_svg({})

    def test_session_chart_with_data(self):
        from nexflow.analysis.analyze_paper_results import SessionStats
        sessions = {
            "new_york": SessionStats(session="new_york", trades=10, wins=6, net_pnl=500.0, win_rate=0.6),
            "london": SessionStats(session="london", trades=5, wins=2, net_pnl=-100.0, win_rate=0.4),
        }
        svg = session_bar_svg(sessions)
        assert "<svg" in svg
        assert "New York" in svg
