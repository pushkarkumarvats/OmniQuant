"""
Market Making Strategy
Provides liquidity and profits from bid-ask spread
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from loguru import logger

from .base_strategy import BaseStrategy


class MarketMakerStrategy(BaseStrategy):
    """
    Market maker strategy with inventory management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize market maker
        
        Config parameters:
            spread_bps: Target spread in basis points
            inventory_limit: Maximum inventory position
            risk_aversion: Risk aversion parameter
            quote_size: Size of each quote
        """
        super().__init__("MarketMaker", config)
        
        # Strategy parameters
        self.spread_bps = self.config.get('spread_bps', 10)
        self.inventory_limit = self.config.get('inventory_limit', 1000)
        self.risk_aversion = self.config.get('risk_aversion', 0.5)
        self.quote_size = self.config.get('quote_size', 100)
        
        # State
        self.inventory = 0
        self.mid_price = None
        self.volatility = 0.01
        
    def initialize(self, simulator: Any):
        """Initialize strategy"""
        self.is_initialized = True
        logger.info(f"Market Maker initialized - Spread: {self.spread_bps} bps, Quote Size: {self.quote_size}")
    
    def on_data(self, simulator: Any, symbol: str, data: pd.Series):
        """Process market data and update quotes"""
        # Update market state
        self._update_state(data)
        
        # Calculate optimal quotes
        bid_price, ask_price = self._calculate_quotes()
        
        # Submit quotes
        if bid_price and ask_price:
            # Cancel existing orders (simplified)
            # In practice, would track and cancel specific orders
            
            # Submit new quotes
            simulator.buy(symbol, self.quote_size, bid_price)
            simulator.sell(symbol, self.quote_size, ask_price)
            
            # Update inventory
            self.inventory = simulator.get_position(symbol)
    
    def _update_state(self, data: pd.Series):
        """Update market state estimates"""
        # Update mid price
        if 'mid_price' in data:
            self.mid_price = data['mid_price']
        elif 'price' in data:
            self.mid_price = data['price']
        elif 'close' in data:
            self.mid_price = data['close']
        
        # Update volatility (simplified)
        if 'volatility' in data:
            self.volatility = data['volatility']
        elif 'return' in data:
            self.volatility = abs(data['return'])
    
    def _calculate_quotes(self) -> tuple:
        """
        Calculate optimal bid and ask prices
        Uses Avellaneda-Stoikov framework
        
        Returns:
            Tuple of (bid_price, ask_price)
        """
        if self.mid_price is None:
            return None, None
        
        # Base spread
        base_spread = self.mid_price * (self.spread_bps / 10000)
        
        # Inventory skew (widen quotes when inventory is high)
        inventory_ratio = self.inventory / self.inventory_limit
        inventory_skew = self.risk_aversion * inventory_ratio * base_spread
        
        # Volatility adjustment
        vol_adjustment = self.volatility * self.mid_price * 0.5
        
        # Calculate quotes
        half_spread = (base_spread + vol_adjustment) / 2
        bid_price = self.mid_price - half_spread + inventory_skew
        ask_price = self.mid_price + half_spread + inventory_skew
        
        # Round to tick size
        bid_price = round(bid_price, 2)
        ask_price = round(ask_price, 2)
        
        return bid_price, ask_price
