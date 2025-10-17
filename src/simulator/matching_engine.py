"""
Matching Engine
Handles order matching, latency simulation, and market impact
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
import time

from .orderbook import OrderBook, Order, Trade, Side, OrderType


@dataclass
class MarketConfig:
    """Market configuration"""
    tick_size: float = 0.01
    lot_size: int = 100
    latency_mean_ms: float = 0.5
    latency_std_ms: float = 0.1
    impact_coefficient: float = 0.1
    impact_model: str = "sqrt"  # "linear", "sqrt", "power"


class MatchingEngine:
    """
    Central matching engine for order processing
    """
    
    def __init__(self, config: Optional[MarketConfig] = None):
        """
        Initialize matching engine
        
        Args:
            config: Market configuration
        """
        self.config = config or MarketConfig()
        self.orderbooks: Dict[str, OrderBook] = {}
        self.trade_history: List[Trade] = []
        self.order_history: List[Order] = []
        
        # Latency simulation
        self.enable_latency = True
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'total_trades': 0,
            'total_volume': 0,
            'total_cancels': 0
        }
    
    def create_orderbook(self, symbol: str) -> OrderBook:
        """
        Create order book for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            OrderBook instance
        """
        if symbol not in self.orderbooks:
            self.orderbooks[symbol] = OrderBook(symbol, self.config.tick_size)
            logger.info(f"Created order book for {symbol}")
        
        return self.orderbooks[symbol]
    
    def submit_order(
        self,
        symbol: str,
        side: Side,
        price: float,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        order_id: Optional[str] = None
    ) -> Tuple[Order, List[Trade]]:
        """
        Submit order to matching engine
        
        Args:
            symbol: Trading symbol
            side: BID or ASK
            price: Order price
            quantity: Order quantity
            order_type: LIMIT or MARKET
            order_id: Optional order ID
            
        Returns:
            Tuple of (Order, list of Trades)
        """
        # Create order book if needed
        if symbol not in self.orderbooks:
            self.create_orderbook(symbol)
        
        # Simulate latency
        if self.enable_latency:
            latency = self._simulate_latency()
            time.sleep(latency / 1000)  # Convert ms to seconds
        
        # Generate order ID
        if order_id is None:
            order_id = f"ORD_{self.stats['total_orders']}"
        
        # Create order
        order = Order(
            order_id=order_id,
            timestamp=time.time(),
            side=side,
            price=price,
            quantity=quantity,
            order_type=order_type
        )
        
        # Apply price impact for large orders
        if order_type == OrderType.MARKET:
            impact_price = self._calculate_price_impact(symbol, side, quantity)
            order.price = impact_price
        
        # Submit to order book
        trades = self.orderbooks[symbol].add_order(order)
        
        # Update statistics
        self.stats['total_orders'] += 1
        if trades:
            self.stats['total_trades'] += len(trades)
            self.stats['total_volume'] += sum(t.quantity for t in trades)
            self.trade_history.extend(trades)
        
        self.order_history.append(order)
        
        return order, trades
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel order
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            True if canceled, False otherwise
        """
        if symbol not in self.orderbooks:
            return False
        
        success = self.orderbooks[symbol].cancel_order(order_id)
        
        if success:
            self.stats['total_cancels'] += 1
        
        return success
    
    def _simulate_latency(self) -> float:
        """
        Simulate network latency
        
        Returns:
            Latency in milliseconds
        """
        latency = np.random.normal(
            self.config.latency_mean_ms,
            self.config.latency_std_ms
        )
        return max(0, latency)
    
    def _calculate_price_impact(
        self,
        symbol: str,
        side: Side,
        quantity: int
    ) -> float:
        """
        Calculate price impact for market orders
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            
        Returns:
            Effective execution price
        """
        book = self.orderbooks[symbol]
        mid_price = book.get_mid_price()
        
        if mid_price is None:
            return 0.0
        
        # Normalize quantity
        norm_quantity = quantity / 1000
        
        # Calculate impact based on model
        if self.config.impact_model == "linear":
            impact = self.config.impact_coefficient * norm_quantity
        elif self.config.impact_model == "sqrt":
            impact = self.config.impact_coefficient * np.sqrt(norm_quantity)
        elif self.config.impact_model == "power":
            impact = self.config.impact_coefficient * (norm_quantity ** 0.6)
        else:
            impact = 0.0
        
        # Apply impact to price
        if side == Side.BID:
            return mid_price * (1 + impact / 100)
        else:
            return mid_price * (1 - impact / 100)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """
        Get order book for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            OrderBook or None
        """
        return self.orderbooks.get(symbol)
    
    def get_market_data(self, symbol: str, levels: int = 5) -> Dict[str, Any]:
        """
        Get market data snapshot
        
        Args:
            symbol: Trading symbol
            levels: Number of levels
            
        Returns:
            Market data dictionary
        """
        if symbol not in self.orderbooks:
            return {}
        
        book = self.orderbooks[symbol]
        snapshot = book.get_snapshot(levels)
        
        return snapshot
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get matching engine statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            'num_symbols': len(self.orderbooks),
            'avg_trade_size': self.stats['total_volume'] / max(1, self.stats['total_trades'])
        }
    
    def reset(self):
        """Reset matching engine state"""
        self.orderbooks.clear()
        self.trade_history.clear()
        self.order_history.clear()
        self.stats = {
            'total_orders': 0,
            'total_trades': 0,
            'total_volume': 0,
            'total_cancels': 0
        }
        logger.info("Matching engine reset")


if __name__ == "__main__":
    # Example usage
    engine = MatchingEngine()
    
    # Create order book
    symbol = "AAPL"
    engine.create_orderbook(symbol)
    
    # Submit some orders
    logger.info("Submitting limit orders...")
    
    # Bids
    engine.submit_order(symbol, Side.BID, 100.00, 100)
    engine.submit_order(symbol, Side.BID, 99.99, 200)
    engine.submit_order(symbol, Side.BID, 99.98, 150)
    
    # Asks
    engine.submit_order(symbol, Side.ASK, 100.02, 100)
    engine.submit_order(symbol, Side.ASK, 100.03, 200)
    
    # Market data
    market_data = engine.get_market_data(symbol)
    logger.info(f"Market Data: {market_data}")
    
    # Submit aggressive order
    logger.info("\nSubmitting aggressive order...")
    order, trades = engine.submit_order(symbol, Side.BID, 100.03, 150)
    
    logger.info(f"Order: {order}")
    logger.info(f"Trades: {len(trades)}")
    for trade in trades:
        logger.info(f"  {trade}")
    
    # Statistics
    stats = engine.get_statistics()
    logger.info(f"\nEngine Statistics: {stats}")
