"""Tests for the CandleEngine — no network, fully deterministic."""

from __future__ import annotations

import tempfile
import time
from collections import deque
from pathlib import Path

import pytest

from nexflow.config import get_config
from nexflow.models.market_state import MarketState, OrderBookLevel, Trade, TradeSide
from nexflow.services.candles.candle_engine import (
    TIMEFRAMES,
    Candle,
    CandleEngine,
    CandleStore,
    _LiveCandle,
    _TimeframeAggregator,
    _bar_open,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _trade(
    trade_id: str,
    price: float,
    size: float,
    side: TradeSide,
    timestamp_ms: int | None = None,
) -> Trade:
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return Trade(trade_id=trade_id, price=price, size=size, side=side, timestamp_ms=timestamp_ms)


def _state(symbol: str = "BTCUSDT", trades: list[Trade] | None = None) -> MarketState:
    s = MarketState(symbol=symbol, product_type="USDT-FUTURES")
    s.bids = [OrderBookLevel(100.0, 1.0)]
    s.asks = [OrderBookLevel(100.5, 1.0)]
    s.trades = deque(trades or [])
    return s


def _engine(symbols: list[str] | None = None, store: CandleStore | None = None) -> CandleEngine:
    get_config.cache_clear()
    cfg = get_config()
    # Patch symbols without touching env permanently
    cfg.market_data.symbols = symbols or ["BTCUSDT"]
    eng = CandleEngine(cfg, store=store)
    return eng


def _ts(minute_offset: int = 0, second_offset: int = 0) -> int:
    """Return a timestamp_ms aligned to a whole minute + offsets."""
    now_s = int(time.time())
    aligned_s = (now_s // 60) * 60  # floor to current minute
    return (aligned_s + minute_offset * 60 + second_offset) * 1000


def _null_store() -> CandleStore:
    tmp = tempfile.mkdtemp()
    return CandleStore(Path(tmp))


# ---------------------------------------------------------------------------
# _bar_open
# ---------------------------------------------------------------------------

def test_bar_open_1m_alignment() -> None:
    assert _bar_open(61, 60) == 60
    assert _bar_open(60, 60) == 60
    assert _bar_open(59, 60) == 0
    assert _bar_open(0, 60) == 0
    assert _bar_open(3599, 3600) == 0
    assert _bar_open(3600, 3600) == 3600


def test_bar_open_5m_alignment() -> None:
    assert _bar_open(300, 300) == 300
    assert _bar_open(599, 300) == 300
    assert _bar_open(301, 300) == 300
    assert _bar_open(299, 300) == 0


def test_bar_open_15m_alignment() -> None:
    assert _bar_open(900, 900) == 900
    assert _bar_open(1799, 900) == 900
    assert _bar_open(899, 900) == 0


# ---------------------------------------------------------------------------
# _LiveCandle
# ---------------------------------------------------------------------------

def test_live_candle_single_trade() -> None:
    lc = _LiveCandle("BTCUSDT", "1m", 60, 1_000_000)
    t = _trade("1", 100.0, 2.0, TradeSide.BUY)
    lc.ingest_trade(t)

    c = lc.to_candle(is_final=False)
    assert c.open == 100.0
    assert c.high == 100.0
    assert c.low == 100.0
    assert c.close == 100.0
    assert c.volume == 2.0
    assert c.buy_volume == 2.0
    assert c.sell_volume == 0.0
    assert c.trade_count == 1
    assert c.vwap == 100.0


def test_live_candle_multiple_trades_ohlcv() -> None:
    lc = _LiveCandle("BTCUSDT", "1m", 60, 1_000_000)
    lc.ingest_trade(_trade("1", 100.0, 1.0, TradeSide.BUY))
    lc.ingest_trade(_trade("2", 105.0, 2.0, TradeSide.BUY))
    lc.ingest_trade(_trade("3",  98.0, 1.0, TradeSide.SELL))
    lc.ingest_trade(_trade("4", 102.0, 3.0, TradeSide.SELL))

    c = lc.to_candle(is_final=True)
    assert c.open == 100.0
    assert c.high == 105.0
    assert c.low == 98.0
    assert c.close == 102.0
    assert c.volume == pytest.approx(7.0)
    assert c.buy_volume == pytest.approx(3.0)
    assert c.sell_volume == pytest.approx(4.0)
    assert c.trade_count == 4


def test_live_candle_vwap_accuracy() -> None:
    lc = _LiveCandle("BTCUSDT", "1m", 60, 1_000_000)
    # VWAP = (100*1 + 200*3) / (1+3) = 700/4 = 175
    lc.ingest_trade(_trade("1", 100.0, 1.0, TradeSide.BUY))
    lc.ingest_trade(_trade("2", 200.0, 3.0, TradeSide.SELL))

    c = lc.to_candle(is_final=True)
    assert c.vwap == pytest.approx(175.0)


def test_live_candle_spread_avg_max() -> None:
    lc = _LiveCandle("BTCUSDT", "1m", 60, 1_000_000)
    lc.add_spread(1.0)
    lc.add_spread(3.0)
    lc.add_spread(2.0)

    c = lc.to_candle(is_final=True)
    assert c.spread_avg == pytest.approx(2.0)
    assert c.spread_max == pytest.approx(3.0)


def test_live_candle_volatility_estimate() -> None:
    lc = _LiveCandle("BTCUSDT", "1m", 60, 1_000_000)
    lc.ingest_trade(_trade("1", 100.0, 1.0, TradeSide.BUY))   # open = 100
    lc.ingest_trade(_trade("2", 110.0, 1.0, TradeSide.BUY))   # high = 110
    lc.ingest_trade(_trade("3",  90.0, 1.0, TradeSide.SELL))  # low  = 90

    c = lc.to_candle(is_final=True)
    # (110 - 90) / 100 = 0.2
    assert c.volatility_estimate == pytest.approx(0.2)


def test_live_candle_no_trades_zeros() -> None:
    lc = _LiveCandle("BTCUSDT", "1m", 60, 1_000_000)
    c = lc.to_candle(is_final=True)
    assert c.trade_count == 0
    assert c.volume == 0.0
    assert c.vwap == 0.0
    assert c.high == 0.0
    assert c.low == 0.0
    assert c.volatility_estimate == 0.0


def test_live_candle_negative_spread_ignored() -> None:
    lc = _LiveCandle("BTCUSDT", "1m", 60, 1_000_000)
    lc.add_spread(-1.0)
    lc.add_spread(0.0)
    lc.add_spread(2.0)
    c = lc.to_candle(is_final=True)
    assert c.spread_avg == pytest.approx(2.0)  # only the valid sample


# ---------------------------------------------------------------------------
# _TimeframeAggregator
# ---------------------------------------------------------------------------

def test_aggregator_no_rollover_same_bar() -> None:
    agg = _TimeframeAggregator("BTCUSDT", "1m", 60)
    bar_open = _bar_open(int(time.time()), 60)
    ts = bar_open * 1000 + 5_000   # 5s into current bar

    result = agg.ingest_trade(_trade("1", 100.0, 1.0, TradeSide.BUY, ts), spread=None)
    assert result is None   # no finalized candle yet


def test_aggregator_rollover_closes_previous_bar() -> None:
    agg = _TimeframeAggregator("BTCUSDT", "1m", 60)
    bar0 = 1_000 * 60   # minute 1000
    bar1 = 1_001 * 60   # minute 1001

    # Trade in bar0
    agg.ingest_trade(_trade("1", 100.0, 1.0, TradeSide.BUY, bar0 * 1000 + 100), spread=None)
    # Trade in bar1 → triggers finalization of bar0
    finalized = agg.ingest_trade(_trade("2", 101.0, 1.0, TradeSide.BUY, bar1 * 1000 + 100), spread=None)

    assert finalized is not None
    assert finalized.open_time == bar0
    assert finalized.close_time == bar1
    assert finalized.is_final is True
    assert finalized.close == 100.0   # last price of bar0


def test_aggregator_rollover_multiple_bars() -> None:
    """Skipping over an empty intermediate bar still produces one candle per non-empty bar."""
    agg = _TimeframeAggregator("BTCUSDT", "1m", 60)
    bar0 = 2_000 * 60
    bar2 = 2_002 * 60   # skip bar1

    agg.ingest_trade(_trade("1", 50.0, 1.0, TradeSide.BUY, bar0 * 1000 + 100), spread=None)
    finalized = agg.ingest_trade(_trade("2", 55.0, 1.0, TradeSide.BUY, bar2 * 1000 + 100), spread=None)

    assert finalized is not None
    assert finalized.open_time == bar0
    # bar1 is naturally empty; bar2 starts fresh
    live = agg.get_live_snapshot()
    assert live is not None
    assert live.open_time == bar2


def test_aggregator_spread_ingest_without_trade() -> None:
    agg = _TimeframeAggregator("BTCUSDT", "1m", 60)
    ts = (_bar_open(int(time.time()), 60)) * 1000 + 500
    agg.ingest_trade(_trade("1", 100.0, 1.0, TradeSide.BUY, ts), spread=1.0)
    agg.ingest_spread(2.0)
    agg.ingest_spread(3.0)

    live = agg.get_live_snapshot()
    assert live is not None
    assert live.spread_max == pytest.approx(3.0)
    assert live.spread_avg == pytest.approx(2.0)   # (1+2+3)/3


def test_aggregator_force_finalize() -> None:
    agg = _TimeframeAggregator("BTCUSDT", "1m", 60)
    ts = (_bar_open(int(time.time()), 60)) * 1000 + 500
    agg.ingest_trade(_trade("1", 99.0, 5.0, TradeSide.SELL, ts), spread=None)

    candle = agg.force_finalize()
    assert candle is not None
    assert candle.is_final is True
    assert candle.open == 99.0
    # Live candle is cleared
    assert agg.get_live_snapshot() is None


# ---------------------------------------------------------------------------
# CandleEngine — trade deduplication and stale filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_packet_dedup_on_reconnect() -> None:
    """Replayed trade IDs must not be double-counted."""
    store = _null_store()
    eng = _engine(store=store)

    t1 = _trade("tid-1", 100.0, 1.0, TradeSide.BUY)
    t2 = _trade("tid-2", 101.0, 2.0, TradeSide.SELL)

    state = _state(trades=[t1, t2])
    await eng._on_update(state)   # first pass — processes both

    state2 = _state(trades=[t1, t2])
    await eng._on_update(state2)  # replay — should be ignored

    live = eng.get_live_snapshot("BTCUSDT", "1m")
    assert live is not None
    assert live.trade_count == 2    # not 4
    assert live.volume == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_engine_stale_trade_ignored() -> None:
    """Trades older than 30 minutes must be discarded."""
    store = _null_store()
    eng = _engine(store=store)

    old_ms = int((time.time() - 3_600) * 1000)   # 1 hour ago
    stale = _trade("stale-1", 50_000.0, 10.0, TradeSide.BUY, old_ms)
    fresh = _trade("fresh-1", 60_000.0, 1.0, TradeSide.BUY)

    state = _state(trades=[stale, fresh])
    await eng._on_update(state)

    live = eng.get_live_snapshot("BTCUSDT", "1m")
    assert live is not None
    assert live.trade_count == 1
    assert live.open == pytest.approx(60_000.0)


@pytest.mark.asyncio
async def test_engine_incremental_updates() -> None:
    """New trades arriving across multiple updates accumulate correctly."""
    store = _null_store()
    eng = _engine(store=store)

    bar_start = _bar_open(int(time.time()), 60) * 1000

    state1 = _state(trades=[_trade("a", 100.0, 1.0, TradeSide.BUY, bar_start + 1_000)])
    await eng._on_update(state1)

    state2 = _state(trades=[
        _trade("a", 100.0, 1.0, TradeSide.BUY, bar_start + 1_000),  # already seen
        _trade("b", 105.0, 2.0, TradeSide.SELL, bar_start + 2_000), # new
    ])
    await eng._on_update(state2)

    live = eng.get_live_snapshot("BTCUSDT", "1m")
    assert live is not None
    assert live.trade_count == 2
    assert live.high == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# CandleEngine — timeframe rollover
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_1m_rollover_emits_callback() -> None:
    """Crossing a 1m boundary triggers on_candle_close for the closed bar."""
    store = _null_store()
    eng = _engine(store=store)

    closed: list[tuple[str, str, Candle]] = []

    async def capture(symbol: str, timeframe: str, candle: Candle) -> None:
        closed.append((symbol, timeframe, candle))

    eng.on_candle_close(capture)

    # bar0 is ~2 mins ago (will be closed by trade rollover).
    # bar1 is the CURRENT minute so the wall-clock check does not also close it.
    bar0 = _bar_open(int(time.time()) - 130, 60)
    bar1 = _bar_open(int(time.time()), 60)  # current in-progress minute

    t0 = _trade("x1", 100.0, 1.0, TradeSide.BUY, bar0 * 1000 + 500)
    t1 = _trade("x2", 101.0, 1.0, TradeSide.BUY, bar1 * 1000 + 500)

    await eng._on_update(_state(trades=[t0]))
    await eng._on_update(_state(trades=[t0, t1]))   # t1 causes rollover

    one_m_closes = [(s, tf, c) for s, tf, c in closed if tf == "1m"]
    assert len(one_m_closes) == 1
    s, tf, c = one_m_closes[0]
    assert s == "BTCUSDT"
    assert c.open_time == bar0
    assert c.close == 100.0
    assert c.is_final is True


@pytest.mark.asyncio
async def test_engine_5m_independent_of_1m() -> None:
    """1m candle closing should not prematurely close the 5m candle."""
    store = _null_store()
    eng = _engine(store=store)

    closed_tfs: list[str] = []

    async def capture(symbol: str, timeframe: str, candle: Candle) -> None:
        closed_tfs.append(timeframe)

    eng.on_candle_close(capture)

    # Two trades 1 minute apart — crosses 1m boundary but NOT 5m or 15m
    bar0 = (int(time.time()) // 300) * 300           # floor to current 5m bar
    t0 = _trade("m1", 100.0, 1.0, TradeSide.BUY, bar0 * 1000 + 1_000)
    t1 = _trade("m2", 101.0, 1.0, TradeSide.BUY, (bar0 + 61) * 1000)  # 61s later → new 1m

    await eng._on_update(_state(trades=[t0]))
    await eng._on_update(_state(trades=[t0, t1]))

    assert "1m" in closed_tfs
    assert "5m" not in closed_tfs
    assert "15m" not in closed_tfs


@pytest.mark.asyncio
async def test_engine_all_timeframes_roll_at_15m() -> None:
    """A 15-minute boundary should close all three timeframes."""
    store = _null_store()
    eng = _engine(store=store)

    closed_tfs: list[str] = []

    async def capture(symbol: str, timeframe: str, candle: Candle) -> None:
        closed_tfs.append(timeframe)

    eng.on_candle_close(capture)

    # Use recent timestamps: bar0 ~20 min ago, bar1 ~5 min ago — both within stale threshold
    bar0 = _bar_open(int(time.time()) - 1_200, 900)  # floor current time−20m to 15m boundary
    bar1 = bar0 + 900

    t0 = _trade("r1", 100.0, 1.0, TradeSide.BUY, bar0 * 1000 + 1_000)
    t1 = _trade("r2", 101.0, 1.0, TradeSide.BUY, bar1 * 1000 + 1_000)

    await eng._on_update(_state(trades=[t0]))
    await eng._on_update(_state(trades=[t0, t1]))

    assert set(closed_tfs) == {"1m", "5m", "15m"}


# ---------------------------------------------------------------------------
# CandleEngine — multi-symbol isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_symbols_are_isolated() -> None:
    """Trades for ETHUSDT must not affect BTCUSDT candles."""
    get_config.cache_clear()
    cfg = get_config()
    cfg.market_data.symbols = ["BTCUSDT", "ETHUSDT"]
    store = _null_store()
    eng = CandleEngine(cfg, store=store)

    ts = _bar_open(int(time.time()), 60) * 1000 + 500

    await eng._on_update(_state("BTCUSDT", [_trade("b1", 30_000.0, 1.0, TradeSide.BUY, ts)]))
    await eng._on_update(_state("ETHUSDT", [_trade("e1", 2_000.0, 5.0, TradeSide.SELL, ts)]))

    btc_live = eng.get_live_snapshot("BTCUSDT", "1m")
    eth_live = eng.get_live_snapshot("ETHUSDT", "1m")

    assert btc_live is not None and btc_live.open == pytest.approx(30_000.0)
    assert eth_live is not None and eth_live.open == pytest.approx(2_000.0)


# ---------------------------------------------------------------------------
# CandleEngine — VWAP and volume attribution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_vwap_across_updates() -> None:
    """VWAP must remain correct as trades arrive across multiple _on_update calls."""
    store = _null_store()
    eng = _engine(store=store)

    ts_base = _bar_open(int(time.time()), 60) * 1000

    # VWAP = (100*2 + 200*8) / 10 = (200 + 1600) / 10 = 180.0
    await eng._on_update(_state(trades=[_trade("v1", 100.0, 2.0, TradeSide.BUY, ts_base + 1_000)]))
    await eng._on_update(_state(trades=[
        _trade("v1", 100.0, 2.0, TradeSide.BUY, ts_base + 1_000),  # already seen
        _trade("v2", 200.0, 8.0, TradeSide.SELL, ts_base + 2_000),
    ]))

    live = eng.get_live_snapshot("BTCUSDT", "1m")
    assert live is not None
    assert live.vwap == pytest.approx(180.0)


@pytest.mark.asyncio
async def test_engine_volume_attribution() -> None:
    """buy_volume + sell_volume must equal total volume."""
    store = _null_store()
    eng = _engine(store=store)

    ts_base = _bar_open(int(time.time()), 60) * 1000
    trades = [
        _trade("va1", 100.0, 3.0, TradeSide.BUY, ts_base + 1_000),
        _trade("va2", 101.0, 2.0, TradeSide.SELL, ts_base + 2_000),
        _trade("va3", 100.5, 1.5, TradeSide.BUY, ts_base + 3_000),
    ]
    state = _state(trades=trades)
    await eng._on_update(state)

    live = eng.get_live_snapshot("BTCUSDT", "1m")
    assert live is not None
    assert live.volume == pytest.approx(6.5)
    assert live.buy_volume == pytest.approx(4.5)
    assert live.sell_volume == pytest.approx(2.0)
    assert live.buy_volume + live.sell_volume == pytest.approx(live.volume)


# ---------------------------------------------------------------------------
# CandleEngine — stop() flushes partials
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_stop_flushes_partial_candles() -> None:
    store = _null_store()
    eng = _engine(store=store)

    flushed: list[Candle] = []

    async def capture(symbol: str, timeframe: str, candle: Candle) -> None:
        flushed.append(candle)

    eng.on_candle_close(capture)

    ts = _bar_open(int(time.time()), 60) * 1000 + 500
    await eng._on_update(_state(trades=[_trade("s1", 99.0, 1.0, TradeSide.BUY, ts)]))
    await eng.stop()

    # Should have emitted partials for all 3 timeframes
    assert len(flushed) == len(TIMEFRAMES)
    for c in flushed:
        assert c.is_final is True
        assert c.trade_count == 1


# ---------------------------------------------------------------------------
# CandleStore — parquet persistence
# ---------------------------------------------------------------------------

def _sample_candle(symbol: str = "BTCUSDT", timeframe: str = "1m") -> Candle:
    return Candle(
        symbol=symbol, timeframe=timeframe,
        open_time=1_000_000, close_time=1_000_060,
        open=100.0, high=110.0, low=90.0, close=105.0,
        volume=10.0, buy_volume=6.0, sell_volume=4.0, trade_count=5,
        vwap=103.0, spread_avg=0.5, spread_max=1.0, volatility_estimate=0.2,
        is_final=True,
    )


def test_candle_store_write_and_read() -> None:
    tmp = Path(tempfile.mkdtemp())
    store = CandleStore(tmp)
    c = _sample_candle()
    store.append(c)

    results = store.read_all("BTCUSDT", "1m")
    assert len(results) == 1
    r = results[0]
    assert r.symbol == "BTCUSDT"
    assert r.open == pytest.approx(100.0)
    assert r.vwap == pytest.approx(103.0)
    assert r.is_final is True


def test_candle_store_appends_multiple_candles() -> None:
    tmp = Path(tempfile.mkdtemp())
    store = CandleStore(tmp)

    for i in range(3):
        c = Candle(
            symbol="BTCUSDT", timeframe="5m",
            open_time=1_000_000 + i * 300, close_time=1_000_300 + i * 300,
            open=float(100 + i), high=float(110 + i), low=float(90 + i),
            close=float(105 + i), volume=float(i + 1),
            buy_volume=float(i), sell_volume=1.0, trade_count=i + 1,
            vwap=float(103 + i), spread_avg=0.1, spread_max=0.2,
            volatility_estimate=0.1, is_final=True,
        )
        store.append(c)

    results = store.read_all("BTCUSDT", "5m")
    assert len(results) == 3
    assert results[2].open == pytest.approx(102.0)


def test_candle_store_survives_restart() -> None:
    """A new CandleStore instance reading the same dir should see existing candles."""
    tmp = Path(tempfile.mkdtemp())

    store1 = CandleStore(tmp)
    store1.append(_sample_candle())

    store2 = CandleStore(tmp)   # fresh instance, same dir
    results = store2.read_all("BTCUSDT", "1m")
    assert len(results) == 1
    assert results[0].close == pytest.approx(105.0)


def test_candle_store_separate_symbols_separate_files() -> None:
    tmp = Path(tempfile.mkdtemp())
    store = CandleStore(tmp)

    store.append(_sample_candle("BTCUSDT", "1m"))
    store.append(_sample_candle("ETHUSDT", "1m"))

    assert (tmp / "BTCUSDT_1m.parquet").exists()
    assert (tmp / "ETHUSDT_1m.parquet").exists()

    btc = store.read_all("BTCUSDT", "1m")
    eth = store.read_all("ETHUSDT", "1m")
    assert len(btc) == 1 and btc[0].symbol == "BTCUSDT"
    assert len(eth) == 1 and eth[0].symbol == "ETHUSDT"


def test_candle_store_missing_file_returns_empty() -> None:
    tmp = Path(tempfile.mkdtemp())
    store = CandleStore(tmp)
    assert store.read_all("XYZUSDT", "15m") == []
