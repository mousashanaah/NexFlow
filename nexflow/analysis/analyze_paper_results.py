"""Paper trading results analyzer.

Reads JSONL execution journals produced by ExecutionJournal and reconstructs:
    - Completed trade records with R multiples, session labels, regime labels
    - Equity curve from EQUITY_SNAPSHOT events
    - Kill-switch event log
    - Execution quality metrics (slippage, fees)

Trade reconstruction state machine (per symbol):
    SIGNAL  → remembers last signal data (stop_price, atr, tp_prices)
    FILL    → opens a trade, matches to last pending SIGNAL
    PARTIAL_TP / STOP_HIT / FORCE_CLOSE → accumulates PnL, closes trade
    Trades with open-but-not-closed positions at journal end are marked FORCED.

All statistics are computed from the reconstructed trade list.
Returns an AnalysisResult that the report generator consumes.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    symbol: str
    direction: str          # "LONG" or "SHORT"
    entry_time: float       # epoch seconds
    exit_time: float
    hold_minutes: float
    entry_price: float
    exit_price: float       # weighted average
    size: float
    pnl: float              # net, after fees
    fees: float
    slippage: float         # abs entry slippage in price units
    slippage_pct: float     # slippage / entry_price
    r_multiple: float       # pnl / (stop_distance * size); nan if unmeasurable
    exit_type: str          # STOP / TP1 / TP2 / TP3 / FORCED / PARTIAL
    atr_at_entry: float
    stop_price: float
    session: str            # asia / london / new_york / off_hours
    volatility_regime: str  # LOW / MEDIUM / HIGH (assigned post-hoc by percentile)


@dataclass
class EquityPoint:
    ts: float
    equity: float
    drawdown: float
    unrealized: float


@dataclass
class KillEvent:
    ts: float
    reason: str
    detail: str


@dataclass
class TradeStats:
    total: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0   # mean R multiple
    avg_r: float = 0.0
    avg_hold_minutes: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    tp_exits: int = 0
    stop_exits: int = 0
    forced_exits: int = 0


@dataclass
class RiskStats:
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_p50: float = 0.0
    drawdown_p95: float = 0.0
    kill_count: int = 0
    kill_reasons: dict[str, int] = field(default_factory=dict)
    max_consec_losses: int = 0
    rejected_count: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    stale_feed_events: int = 0
    latency_spike_events: int = 0


@dataclass
class ExecutionStats:
    avg_slippage_pct: float = 0.0
    total_slippage_cost: float = 0.0
    total_fees: float = 0.0
    fee_drag_gross_pct: float = 0.0   # fees / gross_profit
    avg_fee_per_trade: float = 0.0
    avg_spread_atr_ratio: float = 0.0
    spread_anomaly_events: int = 0


@dataclass
class SymbolStats:
    symbol: str
    trades: int = 0
    wins: int = 0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    avg_r: float = 0.0


@dataclass
class SessionStats:
    session: str
    trades: int = 0
    wins: int = 0
    net_pnl: float = 0.0
    win_rate: float = 0.0


@dataclass
class RegimeStats:
    regime: str
    trades: int = 0
    wins: int = 0
    net_pnl: float = 0.0
    win_rate: float = 0.0


@dataclass
class MarketStats:
    by_symbol: dict[str, SymbolStats] = field(default_factory=dict)
    long_trades: int = 0
    short_trades: int = 0
    long_wins: int = 0
    short_wins: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    by_session: dict[str, SessionStats] = field(default_factory=dict)
    by_regime: dict[str, RegimeStats] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    has_data: bool = False
    trade_stats: TradeStats = field(default_factory=TradeStats)
    risk_stats: RiskStats = field(default_factory=RiskStats)
    execution_stats: ExecutionStats = field(default_factory=ExecutionStats)
    market_stats: MarketStats = field(default_factory=MarketStats)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    monthly_pnl: dict[str, float] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    initial_equity: float = 100_000.0
    final_equity: float = 100_000.0
    session_count: int = 0
    files_loaded: int = 0
    date_range: tuple[str, str] = ("—", "—")
    total_events: int = 0


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class PaperAnalyzer:
    """Load JSONL journals and compute analysis statistics.

    Usage::

        analyzer = PaperAnalyzer()
        result = analyzer.load_and_analyze(Path("logs/paper"))
    """

    def load_and_analyze(self, journal_dir: Path) -> AnalysisResult:
        """Load all *.jsonl files in journal_dir and return AnalysisResult."""
        journal_dir = Path(journal_dir)
        if not journal_dir.exists():
            return AnalysisResult(has_data=False)

        files = sorted(journal_dir.glob("*.jsonl"))
        if not files:
            return AnalysisResult(has_data=False)

        all_events: list[dict[str, Any]] = []
        for path in files:
            try:
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if line:
                        try:
                            all_events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass

        if not all_events:
            return AnalysisResult(has_data=False, files_loaded=len(files))

        all_events.sort(key=lambda e: e.get("ts_epoch", 0))

        trades = _reconstruct_trades(all_events)
        equity_curve = _extract_equity_curve(all_events)
        kill_events, rejected, stale, latency_spikes, spread_anomalies = _extract_operational_events(all_events)
        session_count = _count_sessions(all_events)
        initial_equity = _infer_initial_equity(all_events, equity_curve, trades)
        final_equity = _infer_final_equity(all_events, equity_curve, trades, initial_equity)

        if trades:
            _assign_volatility_regimes(trades)

        result = AnalysisResult(
            has_data=bool(trades or equity_curve),
            trades=trades,
            equity_curve=equity_curve,
            initial_equity=initial_equity,
            final_equity=final_equity,
            session_count=session_count,
            files_loaded=len(files),
            total_events=len(all_events),
        )

        result.trade_stats = _compute_trade_stats(trades)
        result.risk_stats = _compute_risk_stats(
            equity_curve, kill_events, rejected, stale, latency_spikes, trades
        )
        result.execution_stats = _compute_execution_stats(trades, all_events, spread_anomalies)
        result.market_stats = _compute_market_stats(trades)
        result.monthly_pnl = _compute_monthly_pnl(trades)
        result.date_range = _compute_date_range(all_events)

        return result


# ---------------------------------------------------------------------------
# Trade reconstruction
# ---------------------------------------------------------------------------

@dataclass
class _OpenTrade:
    symbol: str
    direction: str
    fill_price: float
    fill_time: float
    size: float
    fee: float
    slippage: float
    equity_before: float
    # from matched SIGNAL
    stop_price: float
    atr: float
    tp_prices: list[float]
    # accumulated as partials arrive
    partial_pnl: float = 0.0
    partial_fees: float = 0.0
    partial_exit_count: int = 0
    last_tp_idx: int = -1
    last_partial_ts: float = 0.0


def _reconstruct_trades(events: list[dict]) -> list[TradeRecord]:
    # Per-symbol state
    pending_signals: dict[str, dict] = {}   # last SIGNAL event per symbol
    open_trades: dict[str, _OpenTrade] = {}
    completed: list[TradeRecord] = []

    for ev in events:
        etype = ev.get("event", "")
        sym = ev.get("symbol", "")

        if etype == "SIGNAL":
            pending_signals[sym] = ev

        elif etype == "FILL":
            # If there's already an open trade for this symbol (e.g., previous trade
            # was fully closed via 3 partial TPs but no STOP_HIT was logged), flush it.
            existing = open_trades.pop(sym, None)
            if existing and existing.partial_exit_count > 0:
                tp_labels = ["TP1", "TP2", "TP3"]
                exit_type = tp_labels[min(existing.last_tp_idx, 2)] if existing.partial_exit_count >= 3 else "PARTIAL"
                close_ts = existing.last_partial_ts or existing.fill_time
                completed.append(_make_record(
                    existing, existing.partial_pnl, existing.partial_fees + existing.fee,
                    close_ts, exit_type, existing.fill_price,
                ))
            sig = pending_signals.get(sym, {})
            fill_price = ev.get("fill_price", 0.0)
            equity_after = ev.get("equity_after", 0.0)
            fee = ev.get("fee", 0.0)
            # equity before fill ≈ equity_after + fee (entry fee is cost)
            equity_before = equity_after + fee
            open_trades[sym] = _OpenTrade(
                symbol=sym,
                direction=ev.get("direction", "long").upper(),
                fill_price=fill_price,
                fill_time=ev.get("ts_epoch", 0.0),
                size=ev.get("size", 0.0),
                fee=fee,
                slippage=ev.get("slippage", 0.0),
                equity_before=equity_before,
                stop_price=sig.get("stop_price", fill_price),
                atr=sig.get("atr", 0.0),
                tp_prices=sig.get("tp_prices", []),
            )

        elif etype == "PARTIAL_TP":
            t = open_trades.get(sym)
            if t:
                t.partial_pnl += ev.get("pnl", 0.0)
                t.partial_fees += ev.get("fee", 0.0)
                t.partial_exit_count += 1
                t.last_tp_idx = max(t.last_tp_idx, ev.get("tp_idx", 0))
                t.last_partial_ts = ev.get("ts_epoch", t.fill_time)
                # If the position is now fully closed (tp_idx==2 or all partials done)
                # we detect full closure when a follow-up event doesn't see the position
                # The router calls portfolio.close_position after all TPs → we rely on
                # the equity snapshot to track this; FORCE_CLOSE handles final close.
                # For PARTIAL_TP: don't close yet — wait for explicit close signal.

        elif etype in ("STOP_HIT", "FORCE_CLOSE"):
            t = open_trades.pop(sym, None)
            if t:
                close_pnl = ev.get("pnl", 0.0)
                close_fee = ev.get("fee", 0.0)
                total_pnl = t.partial_pnl + close_pnl
                total_fees = t.partial_fees + t.fee + close_fee
                exit_time = ev.get("ts_epoch", t.fill_time)
                exit_type = "STOP" if etype == "STOP_HIT" else "FORCED"
                exit_price = ev.get("fill_price", ev.get("price", t.fill_price))
                completed.append(_make_record(t, total_pnl, total_fees, exit_time, exit_type, exit_price))

    # Trades that accumulated partials and were closed via portfolio.close_position
    # are only signalled in the journal by the absence of further FILL events.
    # We detect "all TPs hit, position closed" if we see PARTIAL_TP events but no STOP_HIT.
    # In the router, close_position is called automatically when pos.is_closed().
    # We don't have an explicit "ALL_TP_CLOSED" event, so we emit the record for
    # trades that have partial_pnl > 0 and no further open state using a heuristic:
    # after processing all events, if an open_trade has partial_exit_count > 0
    # and remaining_size ≈ 0, we treat it as closed via TPs.
    # Since we don't have remaining_size in the journal, we use partial_exit_count == 3
    # (all three TP levels hit) as the heuristic.
    for sym, t in list(open_trades.items()):
        if t.partial_exit_count >= 3:
            tp_labels = ["TP1", "TP2", "TP3"]
            exit_type = tp_labels[min(t.last_tp_idx, 2)]
            close_ts = t.last_partial_ts or t.fill_time
            completed.append(_make_record(t, t.partial_pnl, t.partial_fees + t.fee,
                                          close_ts, exit_type, t.fill_price))
            open_trades.pop(sym)
        elif t.partial_exit_count > 0:
            # Partial TPs hit but not fully closed — include as PARTIAL record
            completed.append(_make_record(t, t.partial_pnl, t.partial_fees + t.fee,
                                          t.fill_time, "PARTIAL", t.fill_price))

    return completed


def _make_record(
    t: _OpenTrade,
    total_pnl: float,
    total_fees: float,
    exit_time: float,
    exit_type: str,
    exit_price: float,
) -> TradeRecord:
    hold_minutes = max(0.0, (exit_time - t.fill_time) / 60.0)
    stop_distance = abs(t.fill_price - t.stop_price)
    risk_amount = stop_distance * t.size
    r_multiple = (total_pnl / risk_amount) if risk_amount > 1e-12 else float("nan")
    slippage_pct = (t.slippage / t.fill_price) if t.fill_price > 0 else 0.0

    return TradeRecord(
        symbol=t.symbol,
        direction=t.direction,
        entry_time=t.fill_time,
        exit_time=exit_time,
        hold_minutes=hold_minutes,
        entry_price=t.fill_price,
        exit_price=exit_price,
        size=t.size,
        pnl=total_pnl,
        fees=total_fees,
        slippage=t.slippage,
        slippage_pct=slippage_pct,
        r_multiple=r_multiple,
        exit_type=exit_type,
        atr_at_entry=t.atr,
        stop_price=t.stop_price,
        session=_classify_session(t.fill_time),
        volatility_regime="MEDIUM",  # assigned later in _assign_volatility_regimes
    )


# ---------------------------------------------------------------------------
# Equity curve extraction
# ---------------------------------------------------------------------------

def _extract_equity_curve(events: list[dict]) -> list[EquityPoint]:
    points: list[EquityPoint] = []
    for ev in events:
        if ev.get("event") == "EQUITY_SNAPSHOT":
            points.append(EquityPoint(
                ts=ev.get("ts_epoch", 0.0),
                equity=ev.get("equity", 0.0),
                drawdown=ev.get("drawdown", 0.0),
                unrealized=ev.get("unrealized_pnl", 0.0),
            ))
    return points


# ---------------------------------------------------------------------------
# Operational event extraction
# ---------------------------------------------------------------------------

def _extract_operational_events(
    events: list[dict],
) -> tuple[list[KillEvent], list[dict], int, int, list[dict]]:
    kill_events: list[KillEvent] = []
    rejected: list[dict] = []
    stale_count = 0
    latency_count = 0
    spread_anomalies: list[dict] = []

    for ev in events:
        etype = ev.get("event", "")
        if etype == "KILL_SWITCH":
            kill_events.append(KillEvent(
                ts=ev.get("ts_epoch", 0.0),
                reason=ev.get("reason", ""),
                detail=ev.get("detail", ""),
            ))
        elif etype == "REJECTED":
            rejected.append(ev)
        elif etype == "FEED_STALE":
            stale_count += 1
        elif etype == "LATENCY_SPIKE":
            latency_count += 1
        elif etype == "SPREAD_ANOMALY":
            spread_anomalies.append(ev)

    return kill_events, rejected, stale_count, latency_count, spread_anomalies


def _count_sessions(events: list[dict]) -> int:
    return len({ev.get("session_id") for ev in events if ev.get("event") == "SESSION_START"})


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _compute_trade_stats(trades: list[TradeRecord]) -> TradeStats:
    s = TradeStats()
    if not trades:
        return s

    s.total = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    s.wins = len(wins)
    s.losses = len(losses)
    s.win_rate = s.wins / s.total
    s.gross_profit = sum(t.pnl for t in wins)
    s.gross_loss = abs(sum(t.pnl for t in losses))
    s.net_pnl = sum(t.pnl for t in trades)
    s.profit_factor = (s.gross_profit / s.gross_loss) if s.gross_loss > 0 else float("inf")
    s.avg_win = s.gross_profit / len(wins) if wins else 0.0
    s.avg_loss = s.gross_loss / len(losses) if losses else 0.0
    s.best_trade = max(t.pnl for t in trades)
    s.worst_trade = min(t.pnl for t in trades)

    # Expectancy in R
    valid_r = [t.r_multiple for t in trades if not math.isnan(t.r_multiple)]
    s.avg_r = sum(valid_r) / len(valid_r) if valid_r else 0.0

    avg_win_r = sum(r for r in valid_r if r > 0) / max(1, sum(1 for r in valid_r if r > 0))
    avg_loss_r = abs(sum(r for r in valid_r if r <= 0)) / max(1, sum(1 for r in valid_r if r <= 0))
    s.expectancy_r = (s.win_rate * avg_win_r - (1 - s.win_rate) * avg_loss_r)

    s.avg_hold_minutes = sum(t.hold_minutes for t in trades) / s.total

    s.tp_exits = sum(1 for t in trades if t.exit_type.startswith("TP"))
    s.stop_exits = sum(1 for t in trades if t.exit_type == "STOP")
    s.forced_exits = sum(1 for t in trades if t.exit_type in ("FORCED", "PARTIAL"))

    return s


def _compute_risk_stats(
    equity_curve: list[EquityPoint],
    kill_events: list[KillEvent],
    rejected: list[dict],
    stale: int,
    latency_spikes: int,
    trades: list[TradeRecord],
) -> RiskStats:
    s = RiskStats()
    s.stale_feed_events = stale
    s.latency_spike_events = latency_spikes

    if equity_curve:
        dds = [p.drawdown for p in equity_curve]
        s.max_drawdown = max(dds)
        nonzero = [d for d in dds if d > 0]
        s.avg_drawdown = sum(nonzero) / len(nonzero) if nonzero else 0.0
        sorted_dds = sorted(dds)
        n = len(sorted_dds)
        s.drawdown_p50 = sorted_dds[n // 2]
        s.drawdown_p95 = sorted_dds[min(n - 1, int(n * 0.95))]

    s.kill_count = len(kill_events)
    for ke in kill_events:
        for part in ke.reason.split(","):
            r = part.strip()
            if r:
                s.kill_reasons[r] = s.kill_reasons.get(r, 0) + 1

    s.rejected_count = len(rejected)
    for ev in rejected:
        reason = ev.get("reason", "unknown")
        # Normalize: strip kill_switch: prefix detail
        base = reason.split(":")[0].strip()
        s.rejected_reasons[base] = s.rejected_reasons.get(base, 0) + 1

    # Max consecutive losses
    if trades:
        streak = max_streak = 0
        for t in trades:
            if t.pnl < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        s.max_consec_losses = max_streak

    return s


def _compute_execution_stats(
    trades: list[TradeRecord],
    all_events: list[dict],
    spread_anomalies: list[dict],
) -> ExecutionStats:
    s = ExecutionStats()
    s.spread_anomaly_events = len(spread_anomalies)

    if not trades:
        return s

    slippage_pcts = [t.slippage_pct for t in trades if t.slippage_pct > 0]
    s.avg_slippage_pct = sum(slippage_pcts) / len(slippage_pcts) if slippage_pcts else 0.0
    s.total_slippage_cost = sum(t.slippage * t.size for t in trades)
    s.total_fees = sum(t.fees for t in trades)
    s.avg_fee_per_trade = s.total_fees / len(trades)

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    s.fee_drag_gross_pct = (s.total_fees / gross_profit) if gross_profit > 0 else 0.0

    ratios = [ev.get("spread_atr_ratio", 0.0) for ev in spread_anomalies if ev.get("spread_atr_ratio", 0)]
    s.avg_spread_atr_ratio = sum(ratios) / len(ratios) if ratios else 0.0

    return s


def _compute_market_stats(trades: list[TradeRecord]) -> MarketStats:
    s = MarketStats()
    if not trades:
        return s

    sym_map: dict[str, SymbolStats] = {}
    sess_map: dict[str, SessionStats] = {}
    reg_map: dict[str, RegimeStats] = {}

    for t in trades:
        # Symbol
        ss = sym_map.setdefault(t.symbol, SymbolStats(symbol=t.symbol))
        ss.trades += 1
        ss.net_pnl += t.pnl
        if t.pnl > 0:
            ss.wins += 1

        # Direction
        if t.direction == "LONG":
            s.long_trades += 1
            s.long_pnl += t.pnl
            if t.pnl > 0:
                s.long_wins += 1
        else:
            s.short_trades += 1
            s.short_pnl += t.pnl
            if t.pnl > 0:
                s.short_wins += 1

        # Session
        se = sess_map.setdefault(t.session, SessionStats(session=t.session))
        se.trades += 1
        se.net_pnl += t.pnl
        if t.pnl > 0:
            se.wins += 1

        # Volatility regime
        re = reg_map.setdefault(t.volatility_regime, RegimeStats(regime=t.volatility_regime))
        re.trades += 1
        re.net_pnl += t.pnl
        if t.pnl > 0:
            re.wins += 1

    # Compute win rates
    for ss in sym_map.values():
        ss.win_rate = ss.wins / ss.trades if ss.trades else 0.0
        r_vals = [t.r_multiple for t in trades if t.symbol == ss.symbol and not math.isnan(t.r_multiple)]
        ss.avg_r = sum(r_vals) / len(r_vals) if r_vals else 0.0

    for se in sess_map.values():
        se.win_rate = se.wins / se.trades if se.trades else 0.0

    for re in reg_map.values():
        re.win_rate = re.wins / re.trades if re.trades else 0.0

    s.long_win_rate = s.long_wins / s.long_trades if s.long_trades else 0.0
    s.short_win_rate = s.short_wins / s.short_trades if s.short_trades else 0.0
    s.by_symbol = sym_map
    s.by_session = sess_map
    s.by_regime = reg_map

    return s


def _compute_monthly_pnl(trades: list[TradeRecord]) -> dict[str, float]:
    monthly: dict[str, float] = {}
    for t in trades:
        import time as _t
        tm = _t.gmtime(t.exit_time)
        key = f"{tm.tm_year}-{tm.tm_mon:02d}"
        monthly[key] = monthly.get(key, 0.0) + t.pnl
    return dict(sorted(monthly.items()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_session(epoch_s: float) -> str:
    hour = int(epoch_s % 86_400) // 3_600
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "off_hours"


def _assign_volatility_regimes(trades: list[TradeRecord]) -> None:
    # Mark zero-ATR trades first (no reliable data)
    for t in trades:
        if t.atr_at_entry <= 0:
            t.volatility_regime = "UNKNOWN"

    atrs = [t.atr_at_entry for t in trades if t.atr_at_entry > 0]
    if not atrs:
        return
    sorted_atrs = sorted(atrs)
    n = len(sorted_atrs)
    p33 = sorted_atrs[max(0, int(n * 0.33) - 1)]
    p67 = sorted_atrs[min(n - 1, int(n * 0.67))]
    for t in trades:
        if t.atr_at_entry <= 0:
            continue  # already UNKNOWN
        if t.atr_at_entry <= p33:
            t.volatility_regime = "LOW"
        elif t.atr_at_entry >= p67:
            t.volatility_regime = "HIGH"
        else:
            t.volatility_regime = "MEDIUM"


def _infer_initial_equity(
    events: list[dict],
    equity_curve: list[EquityPoint],
    trades: list[TradeRecord],
) -> float:
    # Prefer SESSION_START data, else first equity snapshot, else default
    if equity_curve:
        return equity_curve[0].equity
    return 100_000.0


def _infer_final_equity(
    events: list[dict],
    equity_curve: list[EquityPoint],
    trades: list[TradeRecord],
    initial_equity: float,
) -> float:
    # Prefer SESSION_END final_equity, else last equity snapshot, else initial + pnl
    for ev in reversed(events):
        if ev.get("event") == "SESSION_END":
            fe = ev.get("final_equity", 0.0)
            if fe > 0:
                return fe
    if equity_curve:
        return equity_curve[-1].equity
    if trades:
        return initial_equity + sum(t.pnl for t in trades)
    return initial_equity


def _compute_date_range(events: list[dict]) -> tuple[str, str]:
    import time as _t
    timestamps = [ev.get("ts_epoch", 0.0) for ev in events if ev.get("ts_epoch", 0.0) > 0]
    if not timestamps:
        return ("—", "—")
    fmt = "%Y-%m-%d %H:%M UTC"
    start = _t.strftime(fmt, _t.gmtime(min(timestamps)))
    end = _t.strftime(fmt, _t.gmtime(max(timestamps)))
    return (start, end)
