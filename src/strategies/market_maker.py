"""
Market Making Strategy

Provides liquidity and profits from the bid-ask spread using the
Avellaneda-Stoikov (2008) framework with inventory-aware quote skewing
and exponentially-weighted volatility estimation.
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, Any, Set
from loguru import logger

from .base_strategy import BaseStrategy


class MarketMakerStrategy(BaseStrategy):
    """Market maker strategy with inventory management and order tracking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("MarketMaker", config)

        # Strategy parameters
        self.spread_bps = self.config.get('spread_bps', 10)
        self.inventory_limit = self.config.get('inventory_limit', 1000)
        self.risk_aversion = self.config.get('risk_aversion', 0.5)
        self.quote_size = self.config.get('quote_size', 100)
        self.vol_ema_span = self.config.get('vol_ema_span', 20)

        # Live order tracking: order_id → {side, price, qty, symbol}
        self._active_bid_ids: Set[str] = set()
        self._active_ask_ids: Set[str] = set()

        # State
        self.inventory = 0
        self.mid_price = None
        self.volatility = 0.01
        self._return_buffer: deque = deque(maxlen=self.vol_ema_span)

    def initialize(self, simulator: Any):
        self.is_initialized = True
        self._active_bid_ids.clear()
        self._active_ask_ids.clear()
        self._return_buffer.clear()
        logger.info(f"Market Maker initialized - Spread: {self.spread_bps} bps, "
                     f"Quote Size: {self.quote_size}")

    def on_data(self, simulator: Any, symbol: str, data: pd.Series):
        self._update_state(data)
        bid_price, ask_price = self._calculate_quotes()

        if bid_price is None or ask_price is None:
            return

        # Cancel stale quotes before submitting new ones
        self._cancel_active_orders(simulator, symbol)

        bid_id = simulator.buy(symbol, self.quote_size, bid_price)
        ask_id = simulator.sell(symbol, self.quote_size, ask_price)

        if bid_id:
            self._active_bid_ids.add(bid_id)
        if ask_id:
            self._active_ask_ids.add(ask_id)

        self.inventory = simulator.get_position(symbol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel_active_orders(self, simulator: Any, symbol: str):
        """Cancel all live quotes so we can re-price at the new level."""
        cancel_fn = getattr(simulator, 'cancel_order', None)
        if cancel_fn is None:
            # Simulator does not support cancel — clear our tracking sets
            self._active_bid_ids.clear()
            self._active_ask_ids.clear()
            return

        for oid in list(self._active_bid_ids):
            try:
                cancel_fn(symbol, oid)
            except Exception:
                pass
        for oid in list(self._active_ask_ids):
            try:
                cancel_fn(symbol, oid)
            except Exception:
                pass
        self._active_bid_ids.clear()
        self._active_ask_ids.clear()

    def _update_state(self, data: pd.Series):
        """Update mid price and exponentially-weighted volatility."""
        if 'mid_price' in data:
            self.mid_price = data['mid_price']
        elif 'price' in data:
            self.mid_price = data['price']
        elif 'close' in data:
            self.mid_price = data['close']

        if 'volatility' in data:
            self.volatility = data['volatility']
        elif 'return' in data:
            self._return_buffer.append(data['return'])
            if len(self._return_buffer) >= 2:
                arr = np.array(self._return_buffer)
                # EMA variance with span = vol_ema_span
                alpha = 2.0 / (self.vol_ema_span + 1)
                weights = (1 - alpha) ** np.arange(len(arr) - 1, -1, -1)
                weights /= weights.sum()
                self.volatility = float(np.sqrt(np.average((arr - arr.mean()) ** 2, weights=weights)))

    def _calculate_quotes(self) -> tuple:
        """
        Calculate optimal bid/ask using the Avellaneda-Stoikov reservation
        price framework.  The reservation price offsets the mid by
        ``-q * gamma * sigma^2 * T`` and the optimal spread adds a
        ``gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/k)`` term,
        where *q* is inventory, *gamma* is risk aversion, and *T* is
        remaining session fraction (set to 1 here for simplicity).
        """
        if self.mid_price is None:
            return None, None

        sigma = max(self.volatility, 1e-8)
        q = self.inventory
        gamma = self.risk_aversion
        T = 1.0  # normalised remaining time
        k = 1.5  # order-arrival intensity parameter

        # Reservation price: shifts mid against inventory direction
        reservation = self.mid_price - q * gamma * (sigma ** 2) * T

        # Optimal spread around reservation price
        optimal_spread = (gamma * (sigma ** 2) * T
                          + (2.0 / gamma) * np.log(1 + gamma / k))

        # Enforce a minimum spread of spread_bps
        min_spread = self.mid_price * (self.spread_bps / 10_000)
        actual_spread = max(optimal_spread, min_spread)

        half = actual_spread / 2.0
        bid_price = round(reservation - half, 2)
        ask_price = round(reservation + half, 2)

        return bid_price, ask_price
