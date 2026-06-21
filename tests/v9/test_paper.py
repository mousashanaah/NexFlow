"""
V9 Paper Trading — Module 6 tests.

Test classes:
  TestIntendedOrder        — order dataclass and side assignment
  TestExecutionReport      — serialisation, round-trip
  TestPositionSnapshot     — serialisation, round-trip
  TestPaperTraderOrders    — compute_orders() correctness
  TestPaperTraderExecution — execute() state transitions
  TestPaperTraderSnapshot  — snapshot() fields
  TestPaperTraderPersist   — save()/load() round-trip
  TestSessionHistory       — append/load execution and snapshot logs
  TestOperationalWorkflow  — end-to-end integration: signal → order → fill → snapshot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from nexflow.v9 import core
from nexflow.v9.paper import (
    CRYPTO_BOOK,
    STOCK_TICKERS,
    ExecutionReport,
    IntendedOrder,
    PaperFill,
    PaperTrader,
    PositionSnapshot,
    append_execution_report,
    append_snapshot,
    load_execution_reports,
    load_snapshots,
)
from nexflow.v9.signals import CryptoSignal, DailySignalRecord, StockSignal


# ── helpers ───────────────────────────────────────────────────────────────────

def _ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _make_signal(
    date: str = "2024-03-01",
    c_sc: float = 3.5,
    s_sc: float = 2.25,
    in_bear: bool = False,
    with_stocks: bool = True,
) -> DailySignalRecord:
    wc, ws   = core.allocate(c_sc, s_sc)
    regime   = core.allocation_regime_name(c_sc, s_sc)
    btc      = CryptoSignal(
        close=62000.0, sma200=40000.0, mom90=0.30, mom30=0.05,
        atr14=800.0, atr_avg=1200.0,
        pts_sma200=2.0, pts_mom90=1.0, pts_mom30=0.5, pts_vol=0.5,
        pts_bonus=0.0, raw=c_sc, score=c_sc,
    )
    stocks = []
    if with_stocks:
        for ticker, price in [("AMD", 180.0), ("GOOGL", 175.0), ("MSTR", 350.0), ("SPOT", 25.0)]:
            stocks.append(StockSignal(
                ticker=ticker, close=price, sma200=price * 0.85, mom90=0.12,
                ema_fast=price * 1.01, ema_slow=price * 0.99,
                pts_sma200=1.0, pts_mom90=1.0, pts_bonus=0.0, pts_ema=0.5,
                score=2.5,
            ))
    return DailySignalRecord(
        date=date, timestamp_ms=_ms(date), in_bear=in_bear,
        btc=btc, crypto_score=c_sc, stocks=stocks, stock_score=s_sc,
        allocation_regime=regime, wc=wc, ws=ws, cash=round(1.0 - wc - ws, 10),
    )


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def trader():
    return PaperTrader(portfolio_value=5_000.0)


@pytest.fixture
def signal():
    return _make_signal()


# ── IntendedOrder ─────────────────────────────────────────────────────────────

class TestIntendedOrder:
    def test_buy_when_delta_positive(self):
        order = IntendedOrder(
            instrument="AMD", side="BUY",
            target_weight=0.10, current_weight=0.0,
            delta_weight=0.10, notional_delta=500.0,
        )
        assert order.side == "BUY"

    def test_sell_when_delta_negative(self):
        order = IntendedOrder(
            instrument="AMD", side="SELL",
            target_weight=0.05, current_weight=0.10,
            delta_weight=-0.05, notional_delta=-250.0,
        )
        assert order.side == "SELL"

    def test_to_dict_has_all_fields(self):
        order = IntendedOrder(
            instrument="CRYPTO_BOOK", side="BUY",
            target_weight=0.65, current_weight=0.0,
            delta_weight=0.65, notional_delta=3250.0,
        )
        d = order.to_dict()
        assert set(d.keys()) == {
            "instrument", "side", "target_weight",
            "current_weight", "delta_weight", "notional_delta",
        }


# ── ExecutionReport ───────────────────────────────────────────────────────────

class TestExecutionReport:
    def _make_report(self) -> ExecutionReport:
        return ExecutionReport(
            date="2024-03-01", portfolio_value=5000.0,
            orders=[IntendedOrder("CRYPTO_BOOK", "BUY", 0.65, 0.50, 0.15, 750.0)],
            fills=[PaperFill("CRYPTO_BOOK", 0.65, 62000.0)],
            rebalance_reason="21-day rebalance — BOTH_HOT",
        )

    def test_status_is_paper_simulated(self):
        report = self._make_report()
        assert report.status == "PAPER_SIMULATED"

    def test_to_json_round_trip(self):
        report  = self._make_report()
        report2 = ExecutionReport.from_json(report.to_json())
        assert report2.date == report.date
        assert report2.orders[0].instrument == "CRYPTO_BOOK"
        assert report2.fills[0].reference_price == 62000.0

    def test_from_dict_preserves_orders(self):
        report = self._make_report()
        d      = report.to_dict()
        r2     = ExecutionReport.from_dict(d)
        assert len(r2.orders) == 1
        assert r2.orders[0].side == "BUY"

    def test_rebalance_reason_preserved(self):
        report = self._make_report()
        r2     = ExecutionReport.from_json(report.to_json())
        assert r2.rebalance_reason == "21-day rebalance — BOTH_HOT"


# ── PositionSnapshot ──────────────────────────────────────────────────────────

class TestPositionSnapshot:
    def _make_snap(self) -> PositionSnapshot:
        return PositionSnapshot(
            date="2024-03-01", portfolio_value=5000.0,
            positions={"CRYPTO_BOOK": 0.65, "AMD": 0.0875},
            cash_weight=0.0, cash_notional=0.0,
            allocation_regime="BOTH_HOT",
            crypto_score=3.5, stock_score=2.25, in_bear=False,
        )

    def test_to_json_round_trip(self):
        snap  = self._make_snap()
        snap2 = PositionSnapshot.from_json(snap.to_json())
        assert snap2.date == "2024-03-01"
        assert snap2.positions["CRYPTO_BOOK"] == 0.65
        assert snap2.in_bear is False

    def test_all_fields_preserved(self):
        snap  = self._make_snap()
        snap2 = PositionSnapshot.from_dict(snap.to_dict())
        assert snap2.crypto_score == 3.5
        assert snap2.allocation_regime == "BOTH_HOT"


# ── compute_orders() ─────────────────────────────────────────────────────────

class TestPaperTraderOrders:
    def test_first_rebalance_all_buys(self, trader, signal):
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        sides   = {o.instrument: o.side for o in orders}
        # All positions are new (from 0.0), so all must be BUY
        assert all(o.side == "BUY" for o in orders if o.target_weight > 0)

    def test_crypto_book_present_in_orders(self, trader, signal):
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        instr   = [o.instrument for o in orders]
        assert CRYPTO_BOOK in instr

    def test_stock_tickers_present_in_orders(self, trader, signal):
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        instr   = {o.instrument for o in orders}
        for ticker in ["AMD", "GOOGL", "MSTR", "SPOT"]:
            assert ticker in instr

    def test_stock_weights_sum_to_ws(self, trader, signal):
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        stock_w = sum(o.target_weight for o in orders if o.instrument in STOCK_TICKERS)
        assert abs(stock_w - ws) < 1e-9

    def test_crypto_weight_equals_wc(self, trader, signal):
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        crypto  = next(o for o in orders if o.instrument == CRYPTO_BOOK)
        assert abs(crypto.target_weight - wc) < 1e-9

    def test_notional_delta_equals_delta_times_value(self, trader, signal):
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        for o in orders:
            expected = round(o.delta_weight * 5_000.0, 2)
            assert abs(o.notional_delta - expected) < 0.01

    def test_hold_when_no_change(self, trader, signal):
        # Pre-fill positions to match target
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        # Execute to set positions
        trader.execute(orders, signal, rebalance_reason="test")
        # Call compute_orders again — all should be HOLD
        orders2 = trader.compute_orders(wc, ws, signal)
        assert all(o.side == "HOLD" for o in orders2)

    def test_sell_when_target_lower_than_current(self, trader, signal):
        # Set high crypto position first
        wc_high = 0.80
        orders1 = trader.compute_orders(wc_high, 0.20, signal)
        trader.execute(orders1, signal, rebalance_reason="initial")
        # Now target is lower
        orders2 = trader.compute_orders(0.40, 0.40, signal)
        crypto_order = next(o for o in orders2 if o.instrument == CRYPTO_BOOK)
        assert crypto_order.side == "SELL"
        assert crypto_order.delta_weight < 0

    def test_compute_orders_is_pure(self, trader, signal):
        """compute_orders must not modify internal state."""
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        before  = dict(trader.positions)
        trader.compute_orders(wc, ws, signal)
        after   = dict(trader.positions)
        assert before == after

    def test_equal_stock_weights(self, trader, signal):
        """All 4 stocks get equal weight within the stock book."""
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        stock_w = [o.target_weight for o in orders if o.instrument in STOCK_TICKERS]
        assert len(stock_w) == 4
        assert max(stock_w) - min(stock_w) < 1e-9


# ── execute() ────────────────────────────────────────────────────────────────

class TestPaperTraderExecution:
    def test_execute_updates_positions(self, trader, signal):
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        trader.execute(orders, signal, rebalance_reason="test")
        assert CRYPTO_BOOK in trader.positions
        assert abs(trader.positions[CRYPTO_BOOK] - wc) < 1e-9

    def test_execute_returns_report(self, trader, signal):
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        report  = trader.execute(orders, signal, rebalance_reason="21-day rebalance")
        assert isinstance(report, ExecutionReport)

    def test_report_status_is_paper(self, trader, signal):
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders = trader.compute_orders(wc, ws, signal)
        report = trader.execute(orders, signal, rebalance_reason="test")
        assert report.status == "PAPER_SIMULATED"

    def test_report_date_matches_signal(self, trader, signal):
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders = trader.compute_orders(wc, ws, signal)
        report = trader.execute(orders, signal, rebalance_reason="test")
        assert report.date == signal.date

    def test_fills_contain_reference_prices(self, trader, signal):
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders = trader.compute_orders(wc, ws, signal)
        report = trader.execute(orders, signal, rebalance_reason="test")
        fill_instr = {f.instrument for f in report.fills}
        # Crypto book and stocks should have reference prices
        for fill in report.fills:
            if fill.instrument == CRYPTO_BOOK:
                assert fill.reference_price == 62000.0
            elif fill.instrument == "AMD":
                assert fill.reference_price == 180.0

    def test_hold_orders_not_in_fills(self, trader, signal):
        """HOLD orders must not appear in fills — they generate no execution."""
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders = trader.compute_orders(wc, ws, signal)
        # Execute once to set positions
        trader.execute(orders, signal, rebalance_reason="initial")
        # Second execution: all HOLD
        orders2 = trader.compute_orders(wc, ws, signal)
        report2 = trader.execute(orders2, signal, rebalance_reason="no change")
        assert len(report2.fills) == 0

    def test_zero_positions_removed(self, trader, signal):
        """After selling to zero, instrument must not appear in positions."""
        # Buy everything
        wc, ws = 0.80, 0.20
        orders = trader.compute_orders(wc, ws, signal)
        trader.execute(orders, signal, rebalance_reason="initial")
        # Now go defensive (40/40/20 cash) — all stock weights change
        orders2 = trader.compute_orders(0.0, 0.0, signal)
        trader.execute(orders2, signal, rebalance_reason="defensive")
        # CRYPTO_BOOK should be gone (target was 0.0)
        assert CRYPTO_BOOK not in trader.positions

    def test_report_reason_preserved(self, trader, signal):
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders = trader.compute_orders(wc, ws, signal)
        report = trader.execute(orders, signal, rebalance_reason="21-day rebalance — BOTH_HOT")
        assert report.rebalance_reason == "21-day rebalance — BOTH_HOT"


# ── snapshot() ───────────────────────────────────────────────────────────────

class TestPaperTraderSnapshot:
    def test_snapshot_date_matches_signal(self, trader, signal):
        snap = trader.snapshot(signal, 0.65, 0.35)
        assert snap.date == signal.date

    def test_snapshot_cash_correct(self, trader, signal):
        snap = trader.snapshot(signal, 0.65, 0.35)
        assert abs(snap.cash_weight - 0.0) < 1e-9

    def test_snapshot_defensive_cash(self, trader, signal):
        snap = trader.snapshot(signal, 0.40, 0.40)
        assert abs(snap.cash_weight - 0.20) < 1e-9

    def test_snapshot_cash_notional(self, trader, signal):
        snap = trader.snapshot(signal, 0.40, 0.40)
        assert abs(snap.cash_notional - 1000.0) < 0.01  # 0.20 * 5000

    def test_snapshot_regime_from_signal(self, trader, signal):
        snap = trader.snapshot(signal, signal.wc, signal.ws)
        assert snap.allocation_regime == signal.allocation_regime

    def test_snapshot_scores_from_signal(self, trader, signal):
        snap = trader.snapshot(signal, signal.wc, signal.ws)
        assert snap.crypto_score == signal.crypto_score
        assert snap.stock_score  == signal.stock_score

    def test_snapshot_bear_flag_from_signal(self):
        sig  = _make_signal(in_bear=True)
        t    = PaperTrader()
        snap = t.snapshot(sig, 0.40, 0.40)
        assert snap.in_bear is True

    def test_snapshot_portfolio_value(self, trader, signal):
        snap = trader.snapshot(signal, 0.65, 0.35)
        assert snap.portfolio_value == 5_000.0


# ── PaperTrader persistence ───────────────────────────────────────────────────

class TestPaperTraderPersist:
    def test_save_and_load_round_trip(self, tmp_dir, signal):
        trader  = PaperTrader(portfolio_value=10_000.0)
        wc, ws  = core.allocate(signal.crypto_score, signal.stock_score)
        orders  = trader.compute_orders(wc, ws, signal)
        trader.execute(orders, signal, rebalance_reason="test")
        path    = tmp_dir / "paper.json"
        trader.save(path)
        loaded  = PaperTrader.load(path)
        assert loaded.portfolio_value == 10_000.0
        assert CRYPTO_BOOK in loaded.positions

    def test_positions_preserved_after_round_trip(self, tmp_dir, signal):
        trader = PaperTrader(portfolio_value=5_000.0)
        wc, ws = core.allocate(signal.crypto_score, signal.stock_score)
        orders = trader.compute_orders(wc, ws, signal)
        trader.execute(orders, signal, rebalance_reason="test")
        orig   = dict(trader.positions)
        path   = tmp_dir / "paper.json"
        trader.save(path)
        loaded = PaperTrader.load(path)
        assert loaded.positions == orig

    def test_empty_positions_round_trip(self, tmp_dir):
        trader = PaperTrader(portfolio_value=5_000.0)
        path   = tmp_dir / "paper.json"
        trader.save(path)
        loaded = PaperTrader.load(path)
        assert loaded.positions == {}

    def test_save_is_atomic(self, tmp_dir, signal):
        trader = PaperTrader()
        path   = tmp_dir / "paper.json"
        trader.save(path)
        assert list(tmp_dir.glob("*.tmp")) == []


# ── Session history ───────────────────────────────────────────────────────────

class TestSessionHistory:
    def _make_report(self, date: str = "2024-03-01") -> ExecutionReport:
        return ExecutionReport(
            date=date, portfolio_value=5000.0,
            orders=[IntendedOrder(CRYPTO_BOOK, "BUY", 0.65, 0.0, 0.65, 3250.0)],
            fills=[PaperFill(CRYPTO_BOOK, 0.65, 62000.0)],
            rebalance_reason="test",
        )

    def test_append_creates_file(self, tmp_dir):
        path = tmp_dir / "history.jsonl"
        append_execution_report(self._make_report(), path)
        assert path.exists()

    def test_load_empty_returns_empty_list(self, tmp_dir):
        path = tmp_dir / "history.jsonl"
        assert load_execution_reports(path) == []

    def test_append_and_load_single_report(self, tmp_dir):
        path = tmp_dir / "history.jsonl"
        append_execution_report(self._make_report("2024-03-01"), path)
        reports = load_execution_reports(path)
        assert len(reports) == 1
        assert reports[0].date == "2024-03-01"

    def test_multiple_reports_preserved_in_order(self, tmp_dir):
        path = tmp_dir / "history.jsonl"
        for d in ["2024-01-01", "2024-02-01", "2024-03-01"]:
            append_execution_report(self._make_report(d), path)
        reports = load_execution_reports(path)
        assert len(reports) == 3
        assert reports[0].date == "2024-01-01"
        assert reports[2].date == "2024-03-01"

    def test_snapshot_history_round_trip(self, tmp_dir, signal):
        path = tmp_dir / "snaps.jsonl"
        trader = PaperTrader()
        snap   = trader.snapshot(signal, signal.wc, signal.ws)
        append_snapshot(snap, path)
        snaps  = load_snapshots(path)
        assert len(snaps) == 1
        assert snaps[0].date == signal.date


# ── End-to-end operational workflow ──────────────────────────────────────────

class TestOperationalWorkflow:
    """
    Full operational cycle: signal → compute orders → execute → snapshot → persist.
    Tests the module as it will be called in production.
    """

    def test_full_rebalance_cycle(self, tmp_dir):
        """Signal arrives, rebalance computed, executed, snapshot taken."""
        sig    = _make_signal(date="2024-03-01", c_sc=3.5, s_sc=2.25)
        trader = PaperTrader(portfolio_value=5_000.0)
        wc, ws = core.allocate(sig.crypto_score, sig.stock_score)

        orders = trader.compute_orders(wc, ws, sig)
        report = trader.execute(orders, sig, rebalance_reason="21-day rebalance — BOTH_HOT")
        snap   = trader.snapshot(sig, wc, ws)

        assert report.status == "PAPER_SIMULATED"
        assert snap.allocation_regime == sig.allocation_regime
        assert CRYPTO_BOOK in trader.positions

    def test_two_rebalances_produce_two_reports(self, tmp_dir):
        hist_path = tmp_dir / "history.jsonl"
        snap_path = tmp_dir / "snaps.jsonl"
        trader    = PaperTrader(portfolio_value=5_000.0)

        for date, c_sc, s_sc in [("2024-02-01", 3.5, 2.25), ("2024-03-01", 4.0, 3.0)]:
            sig    = _make_signal(date=date, c_sc=c_sc, s_sc=s_sc)
            wc, ws = core.allocate(c_sc, s_sc)
            orders = trader.compute_orders(wc, ws, sig)
            report = trader.execute(orders, sig, rebalance_reason=f"rebalance {date}")
            snap   = trader.snapshot(sig, wc, ws)
            append_execution_report(report, hist_path)
            append_snapshot(snap, snap_path)

        reports = load_execution_reports(hist_path)
        snaps   = load_snapshots(snap_path)
        assert len(reports) == 2
        assert len(snaps)   == 2

    def test_portfolio_value_consistent_across_operations(self, tmp_dir):
        sig    = _make_signal()
        trader = PaperTrader(portfolio_value=5_000.0)
        wc, ws = core.allocate(sig.crypto_score, sig.stock_score)
        orders = trader.compute_orders(wc, ws, sig)
        report = trader.execute(orders, sig, rebalance_reason="test")
        snap   = trader.snapshot(sig, wc, ws)
        assert report.portfolio_value == 5_000.0
        assert snap.portfolio_value   == 5_000.0

    def test_session_persists_across_restart(self, tmp_dir):
        """Save trader state, reload, and continue — positions preserved."""
        sig    = _make_signal()
        trader = PaperTrader(portfolio_value=5_000.0)
        wc, ws = core.allocate(sig.crypto_score, sig.stock_score)
        orders = trader.compute_orders(wc, ws, sig)
        trader.execute(orders, sig, rebalance_reason="test")
        path   = tmp_dir / "paper.json"
        trader.save(path)

        # Simulate restart
        trader2 = PaperTrader.load(path)
        orders2 = trader2.compute_orders(wc, ws, sig)
        # After reload, all orders are HOLD (already at target)
        assert all(o.side == "HOLD" for o in orders2)

    def test_defensive_transition_cash_visible_in_snapshot(self):
        """When allocation goes defensive, cash weight appears in snapshot."""
        sig    = _make_signal(c_sc=0.5, s_sc=0.4)  # DEFENSIVE → 40/40/20
        trader = PaperTrader(portfolio_value=5_000.0)
        wc, ws = core.allocate(0.5, 0.4)            # → 0.40, 0.40
        orders = trader.compute_orders(wc, ws, sig)
        trader.execute(orders, sig, rebalance_reason="defensive transition")
        snap   = trader.snapshot(sig, wc, ws)
        assert abs(snap.cash_weight - 0.20) < 1e-9
        assert abs(snap.cash_notional - 1000.0) < 0.01

    def test_no_allocation_logic_in_paper_module(self):
        """Hard rule: paper.py must not contain allocation formulas."""
        import inspect
        import nexflow.v9.paper as paper_module
        src = inspect.getsource(paper_module)
        assert "BOTH_HOT_THRESHOLD" not in src
        assert "CRYPTO_SCORE_MAX"   not in src
        assert "REBALANCE_DAYS"     not in src
