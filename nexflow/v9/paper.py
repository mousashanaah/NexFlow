"""
V9 Confidence — Module 6: Paper Trading Execution Layer

Responsibilities:
  - Accept signals and allocation decisions from the runner
  - Compute delta orders from current to target weights
  - Simulate execution (immediate fill at signal prices — no capital at risk)
  - Produce execution reports and position snapshots
  - Record session history for operational audit

This module is the shim between Module 5 (allocation runner) and
Module 7 (live execution).  In Module 7, PaperTrader.execute() is
replaced by a real exchange client.  Everything else stays identical.

Design invariants:
  - No allocation formulas. Target weights come from the runner/signal.
  - No exchange connectivity. Orders are paper-filled at signal prices.
  - All outputs are serialisable. Every session state is reconstructable.
  - PaperTrader is stateful (tracks current positions); the caller persists it.

Portfolio model:
  Crypto book  → single synthetic instrument "CRYPTO_BOOK" with weight = wc
                 (individual coin splits are handled at the exchange level)
  Stock book   → 4 equal-weight instruments: AMD, GOOGL, MSTR, SPOT (ws/4 each)
  Cash         → 1 - wc - ws (held uninvested in paper mode)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from nexflow.v9.signals import DailySignalRecord


# ── Constants ─────────────────────────────────────────────────────────────────

CRYPTO_BOOK    = "CRYPTO_BOOK"
STOCK_TICKERS  = ["AMD", "GOOGL", "MSTR", "SPOT"]

# Minimum weight delta to generate an order (avoids noise trades)
ORDER_THRESHOLD = 1e-6


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class IntendedOrder:
    instrument:     str
    side:           str       # "BUY" | "SELL" | "HOLD"
    target_weight:  float     # fraction of portfolio
    current_weight: float
    delta_weight:   float     # target - current
    notional_delta: float     # delta * portfolio_value (informational)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperFill:
    instrument:     str
    filled_weight:  float
    reference_price: Optional[float]   # from signal if available, else None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionReport:
    date:              str
    portfolio_value:   float
    orders:            list[IntendedOrder]
    fills:             list[PaperFill]
    rebalance_reason:  str
    status:            str = "PAPER_SIMULATED"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionReport":
        d = dict(d)
        d["orders"] = [IntendedOrder(**o) for o in d["orders"]]
        d["fills"]  = [PaperFill(**f)    for f in d["fills"]]
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "ExecutionReport":
        return cls.from_dict(json.loads(s))


@dataclass
class PositionSnapshot:
    date:              str
    portfolio_value:   float
    positions:         dict    # instrument → weight
    cash_weight:       float
    cash_notional:     float
    allocation_regime: str
    crypto_score:      float
    stock_score:       float
    in_bear:           bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "PositionSnapshot":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "PositionSnapshot":
        return cls.from_dict(json.loads(s))


# ── Paper trader ──────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Stateful paper execution engine.

    Tracks current positions as target weights.  On rebalance, computes
    delta orders and applies them immediately (paper fill).

    State: _positions dict (instrument → current weight).  Caller is
    responsible for persisting this between sessions via save()/load().
    """

    def __init__(
        self,
        portfolio_value: float = 5_000.0,
        positions:       Optional[dict] = None,
    ) -> None:
        self._portfolio_value = portfolio_value
        self._positions: dict[str, float] = dict(positions or {})

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    # ── Pure helpers ──────────────────────────────────────────────────────────

    def _target_weights(
        self,
        target_wc: float,
        target_ws: float,
        signal:    DailySignalRecord,
    ) -> dict[str, float]:
        """
        Compute per-instrument target weights.

        Crypto book treated as a single unit.
        Stocks split equally within the stock book.
        """
        n_stocks   = len(signal.stocks) if signal.stocks else len(STOCK_TICKERS)
        per_stock  = target_ws / n_stocks if n_stocks > 0 else 0.0

        targets: dict[str, float] = {CRYPTO_BOOK: target_wc}
        if signal.stocks:
            for s in signal.stocks:
                targets[s.ticker] = per_stock
        else:
            for t in STOCK_TICKERS:
                targets[t] = per_stock
        return targets

    def _reference_price(self, instrument: str, signal: DailySignalRecord) -> Optional[float]:
        """Look up the closing price for an instrument from the signal record."""
        if instrument == CRYPTO_BOOK:
            return signal.btc.close
        for s in signal.stocks:
            if s.ticker == instrument:
                return s.close
        return None

    def compute_orders(
        self,
        target_wc: float,
        target_ws: float,
        signal:    DailySignalRecord,
    ) -> list[IntendedOrder]:
        """
        Pure: compute delta orders from current positions to target weights.
        Does not modify internal state.
        """
        targets      = self._target_weights(target_wc, target_ws, signal)
        all_instr    = sorted(set(list(targets.keys()) + list(self._positions.keys())))
        orders: list[IntendedOrder] = []

        for inst in all_instr:
            tgt   = targets.get(inst, 0.0)
            cur   = self._positions.get(inst, 0.0)
            delta = round(tgt - cur, 10)

            if abs(delta) < ORDER_THRESHOLD:
                side = "HOLD"
            elif delta > 0:
                side = "BUY"
            else:
                side = "SELL"

            orders.append(IntendedOrder(
                instrument     = inst,
                side           = side,
                target_weight  = tgt,
                current_weight = cur,
                delta_weight   = delta,
                notional_delta = round(delta * self._portfolio_value, 2),
            ))
        return orders

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(
        self,
        orders:           list[IntendedOrder],
        signal:           DailySignalRecord,
        rebalance_reason: str,
    ) -> ExecutionReport:
        """
        Paper fill: update internal positions immediately at signal prices.

        Returns an ExecutionReport documenting every order and fill.
        """
        fills: list[PaperFill] = []

        for order in orders:
            if order.side == "HOLD":
                continue
            self._positions[order.instrument] = order.target_weight
            fills.append(PaperFill(
                instrument    = order.instrument,
                filled_weight = order.target_weight,
                reference_price = self._reference_price(order.instrument, signal),
            ))

        # Drop zero (or near-zero) positions
        self._positions = {
            k: v for k, v in self._positions.items()
            if v > ORDER_THRESHOLD
        }

        return ExecutionReport(
            date             = signal.date,
            portfolio_value  = self._portfolio_value,
            orders           = orders,
            fills            = fills,
            rebalance_reason = rebalance_reason,
        )

    def snapshot(
        self,
        signal:    DailySignalRecord,
        target_wc: float,
        target_ws: float,
    ) -> PositionSnapshot:
        """
        Record the current intended position state as a point-in-time snapshot.
        """
        cash  = round(1.0 - target_wc - target_ws, 10)
        return PositionSnapshot(
            date             = signal.date,
            portfolio_value  = self._portfolio_value,
            positions        = dict(self._positions),
            cash_weight      = cash,
            cash_notional    = round(cash * self._portfolio_value, 2),
            allocation_regime = signal.allocation_regime,
            crypto_score     = signal.crypto_score,
            stock_score      = signal.stock_score,
            in_bear          = signal.in_bear,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist portfolio state (value + positions) for session continuity."""
        import os, tempfile
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "portfolio_value": self._portfolio_value,
            "positions":       self._positions,
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> "PaperTrader":
        with open(path) as f:
            d = json.load(f)
        return cls(
            portfolio_value = d["portfolio_value"],
            positions       = d.get("positions", {}),
        )


# ── Session history ───────────────────────────────────────────────────────────

def append_execution_report(report: ExecutionReport, history_path: Path) -> None:
    """Append one ExecutionReport to the session history (JSON Lines)."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(report.to_json().replace("\n", " ") + "\n")


def append_snapshot(snapshot: PositionSnapshot, history_path: Path) -> None:
    """Append one PositionSnapshot to the snapshot log (JSON Lines)."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(snapshot.to_json().replace("\n", " ") + "\n")


def load_execution_reports(history_path: Path) -> list[ExecutionReport]:
    if not history_path.exists():
        return []
    out = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ExecutionReport.from_json(line))
    return out


def load_snapshots(history_path: Path) -> list[PositionSnapshot]:
    if not history_path.exists():
        return []
    out = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(PositionSnapshot.from_json(line))
    return out
