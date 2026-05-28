"""Unit tests for MarketState mutations."""

from nexflow.models.market_state import MarketState, OrderBookLevel, Trade, TradeSide


def _state() -> MarketState:
    return MarketState(symbol="BTCUSDT", product_type="USDT-FUTURES")


def test_snapshot_sets_bids_asks() -> None:
    s = _state()
    s.apply_orderbook_snapshot(
        bids=[(30000.0, 1.0), (29999.0, 2.0)],
        asks=[(30001.0, 1.5), (30002.0, 0.5)],
    )
    assert s.best_bid is not None and s.best_bid.price == 30000.0
    assert s.best_ask is not None and s.best_ask.price == 30001.0
    assert s.mid_price == 30000.5
    assert s.spread == 1.0


def test_delta_removes_level() -> None:
    s = _state()
    s.apply_orderbook_snapshot(bids=[(100.0, 1.0), (99.0, 2.0)], asks=[(101.0, 1.0)])
    s.apply_orderbook_delta(bid_changes=[(100.0, 0.0)], ask_changes=[])
    assert len(s.bids) == 1
    assert s.bids[0].price == 99.0


def test_delta_adds_level() -> None:
    s = _state()
    s.apply_orderbook_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
    s.apply_orderbook_delta(bid_changes=[(100.5, 3.0)], ask_changes=[])
    prices = [lvl.price for lvl in s.bids]
    assert 100.5 in prices
    assert prices[0] == 100.5  # best bid first


def test_add_trade_caps_history() -> None:
    s = _state()
    for i in range(5):
        s.add_trade(
            Trade(trade_id=str(i), price=100.0, size=1.0, side=TradeSide.BUY, timestamp_ms=i),
            max_history=3,
        )
    assert len(s.trades) == 3
    assert s.last_trade is not None and s.last_trade.trade_id == "4"


def test_ticker_update() -> None:
    s = _state()
    s.apply_ticker({"open24h": 29000.0, "high24h": 31000.0, "low24h": 28000.0, "close24h": 30000.0})
    assert s.high_24h == 31000.0
    assert s.close_24h == 30000.0


def test_empty_state_properties() -> None:
    s = _state()
    assert s.best_bid is None
    assert s.best_ask is None
    assert s.mid_price is None
    assert s.spread is None
    assert s.last_trade is None
