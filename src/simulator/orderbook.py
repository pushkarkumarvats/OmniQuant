"""
Order Book Implementation
Maintains bid and ask sides with price-time priority
"""

from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import bisect
from loguru import logger


class Side(Enum):
    """Order side"""
    BID = "BID"
    ASK = "ASK"


class OrderType(Enum):
    """Order type"""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    CANCEL = "CANCEL"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    timestamp: float
    side: Side
    price: float
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    filled_quantity: int = 0
    
    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled"""
        return self.filled_quantity >= self.quantity


@dataclass
class Trade:
    """Trade execution"""
    trade_id: str
    timestamp: float
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: int
    buyer_is_aggressor: bool


class PriceLevel:
    """Price level in order book"""
    
    def __init__(self, price: float):
        """
        Initialize price level
        
        Args:
            price: Price level
        """
        self.price = price
        self.orders: deque[Order] = deque()  # FIFO queue for time priority
        self.total_quantity = 0
    
    def add_order(self, order: Order):
        """Add order to this level"""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
    
    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove order from this level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                removed_order = self.orders[i]
                del self.orders[i]
                self.total_quantity -= removed_order.remaining_quantity
                return removed_order
        return None
    
    def match(self, quantity: int) -> List[Tuple[Order, int]]:
        """
        Match orders at this level
        
        Args:
            quantity: Quantity to match
            
        Returns:
            List of (order, matched_quantity) tuples
        """
        matches = []
        remaining = quantity
        
        while remaining > 0 and len(self.orders) > 0:
            order = self.orders[0]
            match_qty = min(remaining, order.remaining_quantity)
            
            matches.append((order, match_qty))
            order.filled_quantity += match_qty
            self.total_quantity -= match_qty
            remaining -= match_qty
            
            if order.is_filled:
                self.orders.popleft()
        
        return matches
    
    def is_empty(self) -> bool:
        """Check if level is empty"""
        return len(self.orders) == 0


class OrderBook:
    """
    Limit order book with price-time priority
    """
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        """
        Initialize order book
        
        Args:
            symbol: Trading symbol
            tick_size: Minimum price increment
        """
        self.symbol = symbol
        self.tick_size = tick_size
        
        # Price levels
        self.bids: Dict[float, PriceLevel] = {}  # Price -> PriceLevel
        self.asks: Dict[float, PriceLevel] = {}
        
        # Sorted price lists
        self.bid_prices: List[float] = []  # Descending order
        self.ask_prices: List[float] = []  # Ascending order
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        
        # Statistics
        self.total_trades = 0
        self.total_volume = 0
        
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add order to book
        
        Args:
            order: Order to add
            
        Returns:
            List of trades if order matches
        """
        trades = []
        
        # Store order
        self.orders[order.order_id] = order
        
        # Try to match
        if order.order_type == OrderType.MARKET or self._can_match(order):
            trades = self._match_order(order)
        
        # Add remaining to book if limit order
        if not order.is_filled and order.order_type == OrderType.LIMIT:
            self._add_to_book(order)
        
        return trades
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if canceled, False if not found
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Remove from book
        if order.side == Side.BID:
            if order.price in self.bids:
                self.bids[order.price].remove_order(order_id)
                if self.bids[order.price].is_empty():
                    del self.bids[order.price]
                    self.bid_prices.remove(order.price)
        else:
            if order.price in self.asks:
                self.asks[order.price].remove_order(order_id)
                if self.asks[order.price].is_empty():
                    del self.asks[order.price]
                    self.ask_prices.remove(order.price)
        
        # Remove from tracking
        del self.orders[order_id]
        
        return True
    
    def _can_match(self, order: Order) -> bool:
        """Check if order can match immediately"""
        if order.side == Side.BID:
            return len(self.ask_prices) > 0 and order.price >= self.ask_prices[0]
        else:
            return len(self.bid_prices) > 0 and order.price <= self.bid_prices[0]
    
    def _match_order(self, order: Order) -> List[Trade]:
        """
        Match order against book
        
        Args:
            order: Order to match
            
        Returns:
            List of trades
        """
        trades = []
        
        if order.side == Side.BID:
            # Match against asks
            while not order.is_filled and len(self.ask_prices) > 0:
                best_ask = self.ask_prices[0]
                
                # Check if can match
                if order.order_type == OrderType.LIMIT and order.price < best_ask:
                    break
                
                # Match at this level
                level = self.asks[best_ask]
                matches = level.match(order.remaining_quantity)
                
                for matched_order, qty in matches:
                    trade = Trade(
                        trade_id=f"{self.total_trades}",
                        timestamp=order.timestamp,
                        buy_order_id=order.order_id,
                        sell_order_id=matched_order.order_id,
                        price=best_ask,
                        quantity=qty,
                        buyer_is_aggressor=True
                    )
                    trades.append(trade)
                    order.filled_quantity += qty
                    self.total_trades += 1
                    self.total_volume += qty
                
                # Remove level if empty
                if level.is_empty():
                    del self.asks[best_ask]
                    self.ask_prices.pop(0)
        
        else:  # ASK
            # Match against bids
            while not order.is_filled and len(self.bid_prices) > 0:
                best_bid = self.bid_prices[0]
                
                # Check if can match
                if order.order_type == OrderType.LIMIT and order.price > best_bid:
                    break
                
                # Match at this level
                level = self.bids[best_bid]
                matches = level.match(order.remaining_quantity)
                
                for matched_order, qty in matches:
                    trade = Trade(
                        trade_id=f"{self.total_trades}",
                        timestamp=order.timestamp,
                        buy_order_id=matched_order.order_id,
                        sell_order_id=order.order_id,
                        price=best_bid,
                        quantity=qty,
                        buyer_is_aggressor=False
                    )
                    trades.append(trade)
                    order.filled_quantity += qty
                    self.total_trades += 1
                    self.total_volume += qty
                
                # Remove level if empty
                if level.is_empty():
                    del self.bids[best_bid]
                    self.bid_prices.pop(0)
        
        return trades
    
    def _add_to_book(self, order: Order):
        """Add remaining order quantity to book"""
        if order.side == Side.BID:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
                # Insert in descending order
                bisect.insort(self.bid_prices, order.price)
                self.bid_prices.reverse()
                self.bid_prices = sorted(self.bid_prices, reverse=True)
            
            self.bids[order.price].add_order(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
                # Insert in ascending order
                bisect.insort(self.ask_prices, order.price)
            
            self.asks[order.price].add_order(order)
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self.bid_prices[0] if len(self.bid_prices) > 0 else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self.ask_prices[0] if len(self.ask_prices) > 0 else None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def get_depth(self, side: Side, levels: int = 5) -> List[Tuple[float, int]]:
        """
        Get order book depth
        
        Args:
            side: BID or ASK
            levels: Number of levels
            
        Returns:
            List of (price, quantity) tuples
        """
        depth = []
        
        if side == Side.BID:
            for price in self.bid_prices[:levels]:
                level = self.bids[price]
                depth.append((price, level.total_quantity))
        else:
            for price in self.ask_prices[:levels]:
                level = self.asks[price]
                depth.append((price, level.total_quantity))
        
        return depth
    
    def get_snapshot(self, levels: int = 10) -> Dict:
        """
        Get order book snapshot
        
        Args:
            levels: Number of levels to include
            
        Returns:
            Dictionary with book state
        """
        return {
            'symbol': self.symbol,
            'bids': self.get_depth(Side.BID, levels),
            'asks': self.get_depth(Side.ASK, levels),
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'total_trades': self.total_trades,
            'total_volume': self.total_volume
        }


if __name__ == "__main__":
    # Example usage
    import time
    
    # Create order book
    book = OrderBook("AAPL", tick_size=0.01)
    
    # Add some limit orders
    timestamp = time.time()
    
    # Bids
    book.add_order(Order("1", timestamp, Side.BID, 100.00, 100))
    book.add_order(Order("2", timestamp, Side.BID, 99.99, 200))
    book.add_order(Order("3", timestamp, Side.BID, 99.98, 150))
    
    # Asks
    book.add_order(Order("4", timestamp, Side.ASK, 100.02, 100))
    book.add_order(Order("5", timestamp, Side.ASK, 100.03, 200))
    book.add_order(Order("6", timestamp, Side.ASK, 100.04, 150))
    
    # Print snapshot
    snapshot = book.get_snapshot()
    logger.info(f"Order Book Snapshot: {snapshot}")
    
    # Add aggressive order that crosses spread
    trades = book.add_order(Order("7", timestamp, Side.BID, 100.03, 150))
    logger.info(f"Generated {len(trades)} trades")
    
    for trade in trades:
        logger.info(f"Trade: {trade}")
