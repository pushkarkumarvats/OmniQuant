"""
Momentum Strategy
Trend-following strategy based on statistical momentum
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from loguru import logger
from collections import deque

from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize momentum strategy
        
        Config parameters:
            lookback_period: Period for momentum calculation
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            position_size: Size of each position
            stop_loss: Stop loss percentage
        """
        super().__init__("Momentum", config)
        
        # Strategy parameters
        self.lookback_period = self.config.get('lookback_period', 20)
        self.entry_threshold = self.config.get('entry_threshold', 2.0)
        self.exit_threshold = self.config.get('exit_threshold', 0.5)
        self.position_size = self.config.get('position_size', 100)
        self.stop_loss = self.config.get('stop_loss', 0.05)
        
        # State
        self.prices = deque(maxlen=self.lookback_period)
        self.position = 0
        self.entry_price = None
        
    def initialize(self, simulator: Any):
        """Initialize strategy"""
        self.is_initialized = True
        logger.info(f"Momentum Strategy initialized - Lookback: {self.lookback_period}, Threshold: {self.entry_threshold}")
    
    def on_data(self, simulator: Any, symbol: str, data: pd.Series):
        """Process market data and generate signals"""
        # Get current price
        current_price = data.get('close', data.get('price'))
        if current_price is None:
            return
        
        # Update price history
        self.prices.append(current_price)
        
        # Need enough data
        if len(self.prices) < self.lookback_period:
            return
        
        # Calculate momentum signal
        signal = self._calculate_momentum()
        
        # Get current position
        self.position = simulator.get_position(symbol)
        
        # Trading logic
        if self.position == 0:
            # No position - look for entry
            if signal > self.entry_threshold:
                # Buy signal
                simulator.buy(symbol, self.position_size)
                self.entry_price = current_price
                self.log_signal('LONG_ENTRY', {'price': current_price, 'signal': signal})
                
            elif signal < -self.entry_threshold:
                # Short signal
                simulator.sell(symbol, self.position_size)
                self.entry_price = current_price
                self.log_signal('SHORT_ENTRY', {'price': current_price, 'signal': signal})
                
        elif self.position > 0:
            # Long position - look for exit
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            if signal < self.exit_threshold or pnl_pct < -self.stop_loss:
                # Exit signal or stop loss
                simulator.sell(symbol, abs(self.position))
                self.log_signal('LONG_EXIT', {'price': current_price, 'pnl_pct': pnl_pct})
                self.entry_price = None
                
        elif self.position < 0:
            # Short position - look for exit
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            
            if signal > -self.exit_threshold or pnl_pct < -self.stop_loss:
                # Exit signal or stop loss
                simulator.buy(symbol, abs(self.position))
                self.log_signal('SHORT_EXIT', {'price': current_price, 'pnl_pct': pnl_pct})
                self.entry_price = None
    
    def _calculate_momentum(self) -> float:
        """
        Calculate momentum signal using z-score
        
        Returns:
            Z-score of current momentum
        """
        prices = np.array(self.prices)
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Current return
        current_return = returns[-1]
        
        # Historical mean and std
        mean_return = np.mean(returns[:-1])
        std_return = np.std(returns[:-1])
        
        # Z-score
        if std_return > 0:
            z_score = (current_return - mean_return) / std_return
        else:
            z_score = 0.0
        
        return z_score
