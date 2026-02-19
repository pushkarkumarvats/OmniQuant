"""
Unit tests for Order Book
"""

import unittest
import time
from src.simulator.orderbook import OrderBook, Order, Side, OrderType


class TestOrderBook(unittest.TestCase):
    """Test cases for OrderBook"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.book = OrderBook("TEST", tick_size=0.01)
        self.timestamp = time.time()
    
    def test_add_limit_order(self):
        """Test adding limit orders"""
        order = Order("1", self.timestamp, Side.BID, 100.00, 100)
        trades = self.book.add_order(order)
        
        self.assertEqual(len(trades), 0)  # No match
        self.assertEqual(self.book.get_best_bid(), 100.00)
    
    def test_price_time_priority(self):
        """Test price-time priority matching"""
        # Add orders at same price
        order1 = Order("1", self.timestamp, Side.BID, 100.00, 100)
        order2 = Order("2", self.timestamp + 1, Side.BID, 100.00, 100)
        
        self.book.add_order(order1)
        self.book.add_order(order2)
        
        # Sell should match first order first (time priority)
        sell_order = Order("3", self.timestamp + 2, Side.ASK, 100.00, 150)
        trades = self.book.add_order(sell_order)
        
        self.assertEqual(len(trades), 2)  # Two matches
        self.assertEqual(trades[0].buy_order_id, "1")  # First order matched first
    
    def test_spread_calculation(self):
        """Test bid-ask spread calculation"""
        self.book.add_order(Order("1", self.timestamp, Side.BID, 100.00, 100))
        self.book.add_order(Order("2", self.timestamp, Side.ASK, 100.05, 100))
        
        spread = self.book.get_spread()
        self.assertAlmostEqual(spread, 0.05, places=2)
    
    def test_mid_price(self):
        """Test mid price calculation"""
        self.book.add_order(Order("1", self.timestamp, Side.BID, 100.00, 100))
        self.book.add_order(Order("2", self.timestamp, Side.ASK, 100.10, 100))
        
        mid = self.book.get_mid_price()
        self.assertAlmostEqual(mid, 100.05, places=2)
    
    def test_market_order_execution(self):
        """Test market order execution"""
        # Add liquidity
        self.book.add_order(Order("1", self.timestamp, Side.ASK, 100.00, 100))
        self.book.add_order(Order("2", self.timestamp, Side.ASK, 100.05, 100))
        
        # Market buy
        market_order = Order("3", self.timestamp + 1, Side.BID, 100.10, 150, OrderType.MARKET)
        trades = self.book.add_order(market_order)
        
        self.assertEqual(len(trades), 2)  # Matched at two levels
        self.assertEqual(sum(t.quantity for t in trades), 150)
    
    def test_partial_fill(self):
        """Test partial order fills"""
        # Add small liquidity
        self.book.add_order(Order("1", self.timestamp, Side.ASK, 100.00, 50))
        
        # Large buy order
        buy_order = Order("2", self.timestamp + 1, Side.BID, 100.00, 100)
        trades = self.book.add_order(buy_order)
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].quantity, 50)  # Partial fill
        self.assertEqual(buy_order.filled_quantity, 50)
        self.assertEqual(buy_order.remaining_quantity, 50)
    
    def test_order_cancellation(self):
        """Test order cancellation"""
        order = Order("1", self.timestamp, Side.BID, 100.00, 100)
        self.book.add_order(order)
        
        # Cancel order
        result = self.book.cancel_order("1")
        
        self.assertTrue(result)
        self.assertIsNone(self.book.get_best_bid())
    
    def test_depth(self):
        """Test order book depth"""
        # Add multiple levels
        self.book.add_order(Order("1", self.timestamp, Side.BID, 100.00, 100))
        self.book.add_order(Order("2", self.timestamp, Side.BID, 99.99, 200))
        self.book.add_order(Order("3", self.timestamp, Side.BID, 99.98, 150))
        
        depth = self.book.get_depth(Side.BID, levels=2)
        
        self.assertEqual(len(depth), 2)
        self.assertEqual(depth[0][0], 100.00)  # Best bid
        self.assertEqual(depth[0][1], 100)     # Quantity
    
    def test_empty_book(self):
        """Test empty order book"""
        self.assertIsNone(self.book.get_best_bid())
        self.assertIsNone(self.book.get_best_ask())
        self.assertIsNone(self.book.get_mid_price())
        self.assertIsNone(self.book.get_spread())


if __name__ == '__main__':
    unittest.main()
