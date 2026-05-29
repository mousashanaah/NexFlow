"""Real-time OHLCV candle engine.

Aggregates live trade and orderbook data from MarketState into multi-timeframe
candles. Finalized candles are emitted via async callbacks and persisted to
per-symbol/timeframe parquet files.

Supported timeframes: 1m, 5m, 15m (aligned to UTC epoch boundaries).

Usage:
    store = CandleStore(Path("data/candles"))
    engine = CandleEngine(config, store=store)
    engine.on_candle_close(my_async_callback)
    engine.attach(ws_client)
    await ws_client.start()           # engine fires automatically via callback
    await engine.stop()               # flushes partial candles on shutdown
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from nexflow.logging import get_logger
from nexflow.models.market_state import MarketState, Trade, TradeSide

if TYPE_CHECKING:
    from nexflow.config import NexFlowConfig
    from nexflow.services.market_data.bitget_ws import BitgetWSClient


_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timeframe registry  (name → seconds)
# ---------------------------------------------------------------------------
TIMEFRAMES: dict[str, int] = {"1m": 60, "5m": 300, "15m": 900}

# Trades older than this relative to now are discarded (handles reconnect replays
# of very old history and protects candles from backdated inserts).
_STALE_TRADE_THRESHOLD_S: int = 1_800   # 30 minutes

# Seen-ID set is pruned when it exceeds this size.
_SEEN_ID_CAP: int = 500

CandleCallback = Callable[["str", "str", "Candle"], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Candle — the public data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Candle:
    symbol: str
    timeframe: str
    open_time: int          # bar start, UTC epoch seconds
    close_time: int         # bar end (exclusive), UTC epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    trade_count: int
    vwap: float             # volume-weighted average price
    spread_avg: float       # mean of orderbook spread samples during bar
    spread_max: float       # max spread observed during bar
    volatility_estimate: float  # (high - low) / open, 0 if no trades
    is_final: bool          # True once the bar is closed and will not change


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------

class _LiveCandle:
    """Mutable accumulator for one in-progress bar. Not thread-safe (single event loop)."""

    __slots__ = (
        "symbol", "timeframe", "timeframe_s", "open_time",
        "_open", "_high", "_low", "_close",
        "_volume", "_buy_volume", "_sell_volume",
        "_trade_count", "_pv_sum", "_initialized",
        "_spreads",
    )

    def __init__(self, symbol: str, timeframe: str, timeframe_s: int, open_time: int) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_s = timeframe_s
        self.open_time = open_time

        self._open = 0.0
        self._high = 0.0
        self._low = math.inf
        self._close = 0.0
        self._volume = 0.0
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._trade_count = 0
        self._pv_sum = 0.0
        self._initialized = False
        self._spreads: list[float] = []

    def ingest_trade(self, trade: Trade) -> None:
        price = trade.price
        size = trade.size

        if not self._initialized:
            self._open = price
            self._high = price
            self._low = price
            self._initialized = True
        else:
            if price > self._high:
                self._high = price
            if price < self._low:
                self._low = price

        self._close = price
        self._volume += size
        self._pv_sum += price * size
        self._trade_count += 1

        if trade.side is TradeSide.BUY:
            self._buy_volume += size
        else:
            self._sell_volume += size

    def add_spread(self, spread: float) -> None:
        if spread > 0:
            self._spreads.append(spread)

    def to_candle(self, *, is_final: bool) -> Candle:
        if self._initialized:
            high = self._high
            low = self._low
            volatility = (high - low) / self._open if self._open > 0 else 0.0
            vwap = self._pv_sum / self._volume if self._volume > 0 else self._open
        else:
            high = low = 0.0
            volatility = 0.0
            vwap = 0.0

        spread_avg = sum(self._spreads) / len(self._spreads) if self._spreads else 0.0
        spread_max = max(self._spreads) if self._spreads else 0.0

        return Candle(
            symbol=self.symbol,
            timeframe=self.timeframe,
            open_time=self.open_time,
            close_time=self.open_time + self.timeframe_s,
            open=self._open,
            high=high,
            low=low,
            close=self._close,
            volume=round(self._volume, 8),
            buy_volume=round(self._buy_volume, 8),
            sell_volume=round(self._sell_volume, 8),
            trade_count=self._trade_count,
            vwap=round(vwap, 8),
            spread_avg=round(spread_avg, 8),
            spread_max=round(spread_max, 8),
            volatility_estimate=round(volatility, 8),
            is_final=is_final,
        )


# ---------------------------------------------------------------------------
# Single-symbol, single-timeframe aggregator
# ---------------------------------------------------------------------------

class _TimeframeAggregator:
    """Accumulates trades for one (symbol, timeframe) pair."""

    def __init__(self, symbol: str, timeframe: str, timeframe_s: int) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_s = timeframe_s
        self._live: _LiveCandle | None = None
        self._last_finalized_open: int = 0  # open_time of most recently closed bar

    # ------------------------------------------------------------------

    def ingest_trade(self, trade: Trade, spread: float | None) -> Candle | None:
        """Feed one trade. Returns a finalized Candle if the bar rolled over."""
        bar_open = _bar_open(trade.timestamp_ms // 1000, self.timeframe_s)

        # Discard late-arriving trades that belong to a bar already finalized.
        # Without this, a late trade re-opens a past bar which the wall-clock
        # check then immediately closes, producing micro-candles with 1-2 trades.
        if bar_open <= self._last_finalized_open:
            return None

        finalized: Candle | None = None
        if self._live is not None and bar_open > self._live.open_time:
            finalized = self._do_finalize()

        if self._live is None:
            self._live = _LiveCandle(self.symbol, self.timeframe, self.timeframe_s, bar_open)

        self._live.ingest_trade(trade)
        if spread is not None:
            self._live.add_spread(spread)

        return finalized

    def ingest_spread(self, spread: float) -> None:
        """Record a spread sample without a trade (orderbook-only update)."""
        if self._live is not None and spread > 0:
            self._live.add_spread(spread)

    def check_wall_clock_rollover(self) -> Candle | None:
        """Finalize if wall clock has passed the bar boundary (handles quiet markets)."""
        if self._live is None:
            return None
        if int(time.time()) >= self._live.open_time + self._live.timeframe_s:
            return self._do_finalize()
        return None

    def force_finalize(self) -> Candle | None:
        """Emit a partial candle on graceful shutdown (is_final=True)."""
        return self._do_finalize()

    def get_live_snapshot(self) -> Candle | None:
        return self._live.to_candle(is_final=False) if self._live else None

    # ------------------------------------------------------------------

    def _do_finalize(self) -> Candle | None:
        if self._live is None:
            return None
        candle = self._live.to_candle(is_final=True)
        self._last_finalized_open = self._live.open_time
        self._live = None
        return candle


# ---------------------------------------------------------------------------
# Parquet persistence
# ---------------------------------------------------------------------------

_PARQUET_SCHEMA = pa.schema([
    pa.field("symbol", pa.string()),
    pa.field("timeframe", pa.string()),
    pa.field("open_time", pa.int64()),
    pa.field("close_time", pa.int64()),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("volume", pa.float64()),
    pa.field("buy_volume", pa.float64()),
    pa.field("sell_volume", pa.float64()),
    pa.field("trade_count", pa.int64()),
    pa.field("vwap", pa.float64()),
    pa.field("spread_avg", pa.float64()),
    pa.field("spread_max", pa.float64()),
    pa.field("volatility_estimate", pa.float64()),
    pa.field("is_final", pa.bool_()),
])


class CandleStore:
    """Reads and appends finalized candles to per-symbol+timeframe parquet files.

    Files are rewritten on each close; this is acceptable because candle
    volumes are small (≤ 1440 rows/day for 1m) and correct persistence matters
    more than write throughput at this layer.
    """

    def __init__(self, base_dir: Path = Path("data/candles")) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        # In-memory buffer: key → list[dict]
        self._buffers: dict[str, list[dict[str, Any]]] = {}

    def append(self, candle: Candle) -> None:
        key = self._key(candle.symbol, candle.timeframe)
        buf = self._load_buffer(key, candle.symbol, candle.timeframe)
        buf.append(_candle_to_dict(candle))
        self._write(key, candle.symbol, candle.timeframe)

    def read_all(self, symbol: str, timeframe: str) -> list[Candle]:
        """Return all persisted candles for a symbol+timeframe."""
        key = self._key(symbol, timeframe)
        return [_dict_to_candle(row) for row in self._load_buffer(key, symbol, timeframe)]

    # ------------------------------------------------------------------

    def _load_buffer(self, key: str, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        if key not in self._buffers:
            path = self._path(symbol, timeframe)
            if path.exists():
                try:
                    self._buffers[key] = pq.read_table(path).to_pylist()
                except Exception as exc:
                    _log.warning("candle_store.read_failed", path=str(path), error=str(exc))
                    self._buffers[key] = []
            else:
                self._buffers[key] = []
        return self._buffers[key]

    def _write(self, key: str, symbol: str, timeframe: str) -> None:
        path = self._path(symbol, timeframe)
        try:
            table = pa.Table.from_pylist(self._buffers[key], schema=_PARQUET_SCHEMA)
            pq.write_table(table, path, compression="snappy")
        except Exception as exc:
            _log.error("candle_store.write_failed", path=str(path), error=str(exc))

    def _path(self, symbol: str, timeframe: str) -> Path:
        return self._base / f"{symbol}_{timeframe}.parquet"

    @staticmethod
    def _key(symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"


# ---------------------------------------------------------------------------
# Candle engine — main public class
# ---------------------------------------------------------------------------

class CandleEngine:
    """Multi-symbol, multi-timeframe real-time OHLCV candle aggregator.

    Attach to a BitgetWSClient before starting the client:

        engine = CandleEngine(config)
        engine.on_candle_close(my_callback)
        engine.attach(ws_client)
        await ws_client.start()
        await engine.stop()     # flush partials on shutdown
    """

    def __init__(self, config: NexFlowConfig, store: CandleStore | None = None) -> None:
        self._cfg = config
        self._store = store or CandleStore()
        self._callbacks: list[CandleCallback] = []

        # {symbol → {timeframe → aggregator}}
        self._aggregators: dict[str, dict[str, _TimeframeAggregator]] = {
            sym: {tf: _TimeframeAggregator(sym, tf, tf_s) for tf, tf_s in TIMEFRAMES.items()}
            for sym in config.market_data.symbols
        }

        # {symbol → set of already-processed trade_ids}
        self._seen_ids: dict[str, set[str]] = {
            sym: set() for sym in config.market_data.symbols
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_candle_close(self, callback: CandleCallback) -> None:
        """Register an async callback: callback(symbol, timeframe, candle)."""
        self._callbacks.append(callback)

    def attach(self, client: BitgetWSClient) -> None:
        """Hook into a BitgetWSClient. Must be called before client.start()."""
        client.on_update(self._on_update)

    def get_live_snapshot(self, symbol: str, timeframe: str) -> Candle | None:
        """Return the current in-progress candle (is_final=False), or None."""
        agg = self._aggregators.get(symbol, {}).get(timeframe)
        return agg.get_live_snapshot() if agg else None

    async def stop(self) -> None:
        """Flush all in-progress candles with is_final=True before shutdown."""
        for sym_aggs in self._aggregators.values():
            for agg in sym_aggs.values():
                candle = agg.force_finalize()
                if candle and candle.trade_count > 0:
                    await self._emit(candle)

    # ------------------------------------------------------------------
    # Internal update handler
    # ------------------------------------------------------------------

    async def _on_update(self, state: MarketState) -> None:
        symbol = state.symbol
        if symbol not in self._aggregators:
            return

        spread = state.spread
        new_trades = self._drain_new_trades(state)

        for trade in new_trades:
            for agg in self._aggregators[symbol].values():
                closed = agg.ingest_trade(trade, spread)
                if closed is not None:
                    await self._emit(closed)

        # Feed spread into live candles even when there are no new trades
        if spread is not None and not new_trades:
            for agg in self._aggregators[symbol].values():
                agg.ingest_spread(spread)

        # Wall-clock rollover: handles quiet markets at timeframe boundaries
        for agg in self._aggregators[symbol].values():
            closed = agg.check_wall_clock_rollover()
            if closed is not None:
                await self._emit(closed)

    def _drain_new_trades(self, state: MarketState) -> list[Trade]:
        """Return trades from state not yet processed, sorted by exchange timestamp.

        Filters:
          - Already-seen trade IDs (dedup on reconnect replay)
          - Trades older than _STALE_TRADE_THRESHOLD_S (backdated or replayed history)
        """
        seen = self._seen_ids[state.symbol]
        stale_cutoff_ms = int((time.time() - _STALE_TRADE_THRESHOLD_S) * 1000)

        new_trades: list[Trade] = []
        for t in state.trades:
            if t.trade_id in seen:
                continue
            seen.add(t.trade_id)
            if t.timestamp_ms < stale_cutoff_ms:
                _log.debug(
                    "candle_engine.stale_trade_ignored",
                    symbol=state.symbol,
                    trade_id=t.trade_id,
                    age_s=round((time.time() * 1000 - t.timestamp_ms) / 1000, 1),
                )
                continue
            new_trades.append(t)

        # Prune seen set to avoid unbounded growth
        if len(seen) > _SEEN_ID_CAP:
            current_ids = {t.trade_id for t in state.trades}
            seen.intersection_update(current_ids)

        return sorted(new_trades, key=lambda t: t.timestamp_ms)

    async def _emit(self, candle: Candle) -> None:
        _log.info(
            "candle_engine.candle_closed",
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            open_time=candle.open_time,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            trade_count=candle.trade_count,
            vwap=candle.vwap,
        )
        try:
            self._store.append(candle)
        except Exception as exc:
            _log.error("candle_engine.store_error", error=str(exc))

        for cb in self._callbacks:
            try:
                await cb(candle.symbol, candle.timeframe, candle)
            except Exception as exc:
                _log.error("candle_engine.callback_error", error=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_open(ts_s: int, timeframe_s: int) -> int:
    """Return the UTC epoch-second start of the bar containing ts_s."""
    return (ts_s // timeframe_s) * timeframe_s


def _candle_to_dict(c: Candle) -> dict[str, Any]:
    return {
        "symbol": c.symbol,
        "timeframe": c.timeframe,
        "open_time": c.open_time,
        "close_time": c.close_time,
        "open": c.open,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        "volume": c.volume,
        "buy_volume": c.buy_volume,
        "sell_volume": c.sell_volume,
        "trade_count": c.trade_count,
        "vwap": c.vwap,
        "spread_avg": c.spread_avg,
        "spread_max": c.spread_max,
        "volatility_estimate": c.volatility_estimate,
        "is_final": c.is_final,
    }


def _dict_to_candle(d: dict[str, Any]) -> Candle:
    return Candle(
        symbol=d["symbol"],
        timeframe=d["timeframe"],
        open_time=d["open_time"],
        close_time=d["close_time"],
        open=d["open"],
        high=d["high"],
        low=d["low"],
        close=d["close"],
        volume=d["volume"],
        buy_volume=d["buy_volume"],
        sell_volume=d["sell_volume"],
        trade_count=d["trade_count"],
        vwap=d["vwap"],
        spread_avg=d["spread_avg"],
        spread_max=d["spread_max"],
        volatility_estimate=d["volatility_estimate"],
        is_final=d["is_final"],
    )
