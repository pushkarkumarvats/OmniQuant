"""
Exchange Emulator
Isolated order execution and market simulation engine.
Decoupled from data iteration - responds only to standardized Tick and Order events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .matching_engine import MatchingEngine, MarketConfig
from .orderbook import Order, Side, OrderType, Trade


@dataclass(frozen=True)
class Tick:
    """Immutable market data tick - the sole input the exchange understands."""

    timestamp: float
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int = 0


@dataclass(frozen=True)
class Fill:
    """Result of an order execution."""

    symbol: str
    side: Side
    quantity: int
    avg_price: float
    commission: float
    trades: Tuple[Trade, ...] = ()


class ExchangeEmulator:
    """
    Simulates an exchange matching engine.

    Responsibilities:
      - Maintain per-symbol order books
      - Accept and match orders
      - Apply realistic slippage, commissions, and price impact
      - Expose market state (best bid/ask, mid price)

    This class knows *nothing* about strategies, portfolios, or data iteration.
    """

    def __init__(
        self,
        market_config: Optional[MarketConfig] = None,
        commission_rate: float = 0.0002,
        slippage_bps: float = 1.0,
    ) -> None:
        self._config = market_config or MarketConfig()
        self._engine = MatchingEngine(self._config)
        self._engine.enable_latency = False  # No sleep in backtest
        self._commission_rate = commission_rate
        self._slippage_bps = slippage_bps

    # ------------------------------------------------------------------
    # Market state updates
    # ------------------------------------------------------------------

    def process_tick(self, tick: Tick) -> None:
        """Seed the order book with the tick's bid/ask so the book is never empty."""
        sym = tick.symbol
        if sym not in self._engine.orderbooks:
            self._engine.create_orderbook(sym)

        volume = max(tick.volume, 100)
        self._engine.submit_order(sym, Side.BID, tick.bid, volume)
        self._engine.submit_order(sym, Side.ASK, tick.ask, volume)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        limit_price: Optional[float] = None,
    ) -> Optional[Fill]:
        """Submit an order and return a Fill if any execution occurred."""
        book = self._engine.get_orderbook(symbol)
        if book is None:
            return None

        mid = book.get_mid_price()
        if mid is None:
            return None

        if limit_price is None:
            order_type = OrderType.MARKET
            price = mid
        else:
            order_type = OrderType.LIMIT
            price = limit_price

        order, trades = self._engine.submit_order(
            symbol, side, price, quantity, order_type
        )

        if not trades:
            return None

        total_qty = sum(t.quantity for t in trades)
        avg_price = sum(t.price * t.quantity for t in trades) / total_qty
        commission = total_qty * avg_price * self._commission_rate

        return Fill(
            symbol=symbol,
            side=side,
            quantity=total_qty,
            avg_price=avg_price,
            commission=commission,
            trades=tuple(trades),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_mid_price(self, symbol: str) -> Optional[float]:
        book = self._engine.get_orderbook(symbol)
        return book.get_mid_price() if book else None

    def get_best_bid(self, symbol: str) -> Optional[float]:
        book = self._engine.get_orderbook(symbol)
        return book.get_best_bid() if book else None

    def get_best_ask(self, symbol: str) -> Optional[float]:
        book = self._engine.get_orderbook(symbol)
        return book.get_best_ask() if book else None

    def reset(self) -> None:
        self._engine.reset()
