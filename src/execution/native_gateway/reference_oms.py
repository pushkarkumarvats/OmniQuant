"""
Reference OMS - Pure-Python implementation mirroring the native C++/Rust OMS API.

Used for development, testing, and as a behavioral specification.
NOT suitable for production trading (lacks deterministic latency guarantees).
"""

from __future__ import annotations

import time
import uuid
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .oms_bridge import (
    NativeFill,
    NativeOrder,
    NativeOrderSide,
    NativeOrderStatus,
    NativeOrderType,
    NativeTimeInForce,
    OMSConfig,
)


@dataclass
class _InternalOrder:
    """Mutable internal order state."""
    native_order: NativeOrder
    status: NativeOrderStatus
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    cumulative_commission: float = 0.0
    fills: List[NativeFill] = field(default_factory=list)
    created_at_ns: int = 0
    last_updated_ns: int = 0


class ReferenceOMS:
    """
    Pure-Python OMS reference implementation.
    
    Matches the native C++/Rust OMS API exactly so strategies can be
    developed and tested without compiling native code.
    
    Features:
      - Full order lifecycle (New -> PartialFill -> Fill / Cancel)
      - Commission model (configurable)
      - Fat-finger protection
      - Order rate limiting
      - Fill simulation with realistic slippage
      - Thread-safe operation
    """

    def __init__(self, config: Optional[OMSConfig] = None) -> None:
        self.config = config or OMSConfig()
        self._connected = False
        self._lock = threading.Lock()
        
        # Order storage
        self._orders: Dict[str, _InternalOrder] = {}
        self._order_counter = 0
        
        # Fill queue (SPSC-style)
        self._fill_queue: deque[NativeFill] = deque(maxlen=100_000)
        self._fill_callbacks: List[Callable[[NativeFill], None]] = []
        
        # Rate limiting
        self._order_timestamps: deque[float] = deque(maxlen=self.config.max_orders_per_second)
        
        # Market state (fed externally)
        self._market_prices: Dict[str, float] = {}
        self._market_bid_ask: Dict[str, Tuple[float, float]] = {}
        
        # Latency tracking
        self._submit_latencies_ns: List[int] = []
        
        # Position tracking (for risk checks)
        self._positions: Dict[str, int] = defaultdict(int)
        self._position_notional: Dict[str, float] = defaultdict(float)
        
        # Commission schedule
        self._commission_per_share = 0.001  # $0.001/share
        self._commission_min = 1.0          # $1 minimum
        
        logger.info("Reference OMS initialized (Python fallback)")

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Simulate connection initialization."""
        self._connected = True
        logger.info("Reference OMS connected")
        return True

    def disconnect(self) -> None:
        """Shutdown reference OMS."""
        self._connected = False
        logger.info("Reference OMS disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Market data feed (for fill simulation)
    # ------------------------------------------------------------------

    def update_market_price(self, symbol: str, price: float, bid: float = 0, ask: float = 0) -> None:
        """Update market price for fill simulation."""
        self._market_prices[symbol] = price
        if bid > 0 and ask > 0:
            self._market_bid_ask[symbol] = (bid, ask)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def submit_order(self, order: NativeOrder) -> Tuple[bool, str]:
        """Submit an order with pre-trade risk checks."""
        start_ns = time.time_ns()
        
        with self._lock:
            # Rate limit check
            now = time.time()
            while self._order_timestamps and now - self._order_timestamps[0] > 1.0:
                self._order_timestamps.popleft()
            
            if len(self._order_timestamps) >= self.config.max_orders_per_second:
                return False, f"Rate limit exceeded ({self.config.max_orders_per_second}/s)"
            
            # Max open orders check
            open_count = sum(
                1 for o in self._orders.values()
                if o.status in (NativeOrderStatus.NEW, NativeOrderStatus.PARTIALLY_FILLED, NativeOrderStatus.PENDING_NEW)
            )
            if open_count >= self.config.max_open_orders:
                return False, f"Max open orders exceeded ({self.config.max_open_orders})"
            
            # Fat-finger check
            if order.price > 0 and order.symbol in self._market_prices:
                mid = self._market_prices[order.symbol]
                deviation_pct = abs(order.price - mid) / mid * 100
                if deviation_pct > self.config.fat_finger_threshold_pct:
                    return False, f"Fat-finger check failed: {deviation_pct:.1f}% from mid ({mid})"
            
            # Max order value check
            order_value = order.quantity * order.price if order.price > 0 else 0
            if order_value > self.config.max_order_value:
                return False, f"Order value ${order_value:,.0f} exceeds max ${self.config.max_order_value:,.0f}"
            
            # Max quantity check
            if order.quantity > self.config.max_order_qty:
                return False, f"Quantity {order.quantity} exceeds max {self.config.max_order_qty}"
            
            # Accept order
            internal = _InternalOrder(
                native_order=order,
                status=NativeOrderStatus.NEW,
                created_at_ns=time.time_ns(),
                last_updated_ns=time.time_ns(),
            )
            self._orders[order.order_id] = internal
            self._order_timestamps.append(now)
            
            # Simulate immediate fill for market orders
            if order.order_type == NativeOrderType.MARKET:
                self._simulate_fill(internal)
            elif order.order_type in (NativeOrderType.IOC, NativeOrderType.FOK):
                self._simulate_fill(internal, immediate_only=True)
            
            latency_ns = time.time_ns() - start_ns
            self._submit_latencies_ns.append(latency_ns)
            
            return True, f"Accepted in {latency_ns/1000:.1f}µs"

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel an open order."""
        with self._lock:
            if order_id not in self._orders:
                return False, "Order not found"
            
            internal = self._orders[order_id]
            if internal.status in (NativeOrderStatus.FILLED, NativeOrderStatus.CANCELLED, NativeOrderStatus.REJECTED):
                return False, f"Cannot cancel order in {internal.status.name} state"
            
            internal.status = NativeOrderStatus.CANCELLED
            internal.last_updated_ns = time.time_ns()
            
            return True, "Cancelled"

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        cancelled = 0
        with self._lock:
            for oid, internal in self._orders.items():
                if internal.status in (NativeOrderStatus.NEW, NativeOrderStatus.PARTIALLY_FILLED):
                    if symbol is None or internal.native_order.symbol == symbol:
                        internal.status = NativeOrderStatus.CANCELLED
                        internal.last_updated_ns = time.time_ns()
                        cancelled += 1
        return cancelled

    def get_order_status(self, order_id: str) -> Optional[NativeOrderStatus]:
        """Get current order status."""
        internal = self._orders.get(order_id)
        return internal.status if internal else None

    def get_open_orders(self) -> Dict[str, NativeOrder]:
        """Get all open orders."""
        return {
            oid: internal.native_order
            for oid, internal in self._orders.items()
            if internal.status in (NativeOrderStatus.NEW, NativeOrderStatus.PARTIALLY_FILLED)
        }

    # ------------------------------------------------------------------
    # Fill simulation
    # ------------------------------------------------------------------

    def _simulate_fill(self, internal: _InternalOrder, immediate_only: bool = False) -> None:
        """Simulate order fill with realistic slippage and commission."""
        order = internal.native_order
        remaining = order.quantity - internal.filled_qty
        
        if remaining <= 0:
            return
        
        # Get market price
        if order.symbol in self._market_bid_ask:
            bid, ask = self._market_bid_ask[order.symbol]
        elif order.symbol in self._market_prices:
            mid = self._market_prices[order.symbol]
            spread = mid * 0.0002  # 2 bps synthetic spread
            bid = mid - spread / 2
            ask = mid + spread / 2
        else:
            # No market data - can't fill
            if immediate_only:
                internal.status = NativeOrderStatus.CANCELLED
            return
        
        # Determine fill price with slippage
        if order.side == NativeOrderSide.BUY:
            base_price = ask
            # Market impact: sqrt model
            impact = 0.0001 * np.sqrt(remaining / 1000) * base_price
            fill_price = base_price + impact
        else:
            base_price = bid
            impact = 0.0001 * np.sqrt(remaining / 1000) * base_price
            fill_price = base_price - impact
        
        # For limit orders, check if price is acceptable
        if order.order_type == NativeOrderType.LIMIT:
            if order.side == NativeOrderSide.BUY and fill_price > order.price:
                if immediate_only:
                    internal.status = NativeOrderStatus.CANCELLED
                return
            if order.side == NativeOrderSide.SELL and fill_price < order.price:
                if immediate_only:
                    internal.status = NativeOrderStatus.CANCELLED
                return
        
        # Calculate commission
        commission = max(
            self._commission_min,
            remaining * self._commission_per_share
        )
        
        # Create fill
        fill = NativeFill(
            fill_id=f"FILL_{uuid.uuid4().hex[:12]}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_qty=remaining,
            fill_price=round(fill_price, 6),
            commission=round(commission, 4),
            liquidity_flag="R",  # Removing liquidity
            exchange="SIMULATED",
            timestamp_ns=time.time_ns(),
            leaves_qty=0,
            cum_qty=order.quantity,
            avg_price=round(fill_price, 6),
        )
        
        # Update internal state
        internal.filled_qty = order.quantity
        internal.avg_fill_price = fill_price
        internal.cumulative_commission += commission
        internal.fills.append(fill)
        internal.status = NativeOrderStatus.FILLED
        internal.last_updated_ns = time.time_ns()
        
        # Update position
        qty_signed = remaining if order.side == NativeOrderSide.BUY else -remaining
        self._positions[order.symbol] += qty_signed
        self._position_notional[order.symbol] = abs(self._positions[order.symbol]) * fill_price
        
        # Enqueue fill
        self._fill_queue.append(fill)
        
        # Invoke callbacks
        for cb in self._fill_callbacks:
            try:
                cb(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    # ------------------------------------------------------------------
    # Fill polling
    # ------------------------------------------------------------------

    def poll_fills(self, max_count: int = 1000) -> List[NativeFill]:
        """Poll for new fills (non-blocking)."""
        fills = []
        for _ in range(min(max_count, len(self._fill_queue))):
            fills.append(self._fill_queue.popleft())
        return fills

    def register_fill_callback(self, callback: Callable[[NativeFill], None]) -> None:
        """Register a fill callback."""
        self._fill_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Latency / telemetry
    # ------------------------------------------------------------------

    def get_latency_stats(self) -> Dict[str, float]:
        """Get order submission latency statistics (microseconds)."""
        if not self._submit_latencies_ns:
            return {"min_us": 0, "max_us": 0, "avg_us": 0, "p50_us": 0, "p99_us": 0, "count": 0}

        arr = np.array(self._submit_latencies_ns) / 1000.0
        return {
            "min_us": float(np.min(arr)),
            "max_us": float(np.max(arr)),
            "avg_us": float(np.mean(arr)),
            "p50_us": float(np.percentile(arr, 50)),
            "p99_us": float(np.percentile(arr, 99)),
            "count": len(arr),
        }

    def get_positions(self) -> Dict[str, int]:
        """Get current positions."""
        return dict(self._positions)

    def get_position_notional(self) -> Dict[str, float]:
        """Get position notional values."""
        return dict(self._position_notional)
