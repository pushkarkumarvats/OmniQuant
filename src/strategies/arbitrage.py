"""
Arbitrage Strategy
Statistical arbitrage and pairs trading
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from loguru import logger
from collections import deque

from .base_strategy import BaseStrategy


class ArbitrageStrategy(BaseStrategy):
    """
    Statistical arbitrage / pairs trading strategy
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize arbitrage strategy
        
        Config parameters:
            lookback_period: Period for spread calculation
            entry_z_score: Z-score threshold for entry
            exit_z_score: Z-score threshold for exit
            position_size: Size of each leg
            hedge_ratio: Fixed hedge ratio (optional)
        """
        super().__init__("Arbitrage", config)
        
        # Strategy parameters
        self.lookback_period = self.config.get('lookback_period', 60)
        self.entry_z_score = self.config.get('entry_z_score', 2.0)
        self.exit_z_score = self.config.get('exit_z_score', 0.5)
        self.position_size = self.config.get('position_size', 100)
        self.hedge_ratio = self.config.get('hedge_ratio', 1.0)
        
        # State
        self.spreads = deque(maxlen=self.lookback_period)
        self.position = 0  # 1 = long spread, -1 = short spread, 0 = flat
        
    def initialize(self, simulator: Any):
        """Initialize strategy"""
        self.is_initialized = True
        logger.info(f"Arbitrage Strategy initialized - Z-score entry: {self.entry_z_score}")
    
    def on_data_pair(
        self,
        simulator: Any,
        symbol1: str,
        symbol2: str,
        price1: float,
        price2: float
    ):
        """
        Process data for pair trading
        
        Args:
            simulator: Event simulator
            symbol1: First symbol
            symbol2: Second symbol
            price1: Price of first symbol
            price2: Price of second symbol
        """
        # Calculate spread
        spread = price1 - self.hedge_ratio * price2
        self.spreads.append(spread)
        
        # Need enough data
        if len(self.spreads) < self.lookback_period:
            return
        
        # Calculate z-score of spread
        z_score = self._calculate_spread_zscore()
        
        # Trading logic
        if self.position == 0:
            # No position - look for entry
            if z_score < -self.entry_z_score:
                # Spread is too low - buy spread (long S1, short S2)
                simulator.buy(symbol1, self.position_size)
                simulator.sell(symbol2, int(self.position_size * self.hedge_ratio))
                self.position = 1
                self.log_signal('LONG_SPREAD', {'z_score': z_score, 'spread': spread})
                
            elif z_score > self.entry_z_score:
                # Spread is too high - short spread (short S1, long S2)
                simulator.sell(symbol1, self.position_size)
                simulator.buy(symbol2, int(self.position_size * self.hedge_ratio))
                self.position = -1
                self.log_signal('SHORT_SPREAD', {'z_score': z_score, 'spread': spread})
                
        elif self.position == 1:
            # Long spread - look for exit
            if z_score > -self.exit_z_score:
                # Close position
                simulator.sell(symbol1, self.position_size)
                simulator.buy(symbol2, int(self.position_size * self.hedge_ratio))
                self.position = 0
                self.log_signal('CLOSE_LONG_SPREAD', {'z_score': z_score})
                
        elif self.position == -1:
            # Short spread - look for exit
            if z_score < self.exit_z_score:
                # Close position
                simulator.buy(symbol1, self.position_size)
                simulator.sell(symbol2, int(self.position_size * self.hedge_ratio))
                self.position = 0
                self.log_signal('CLOSE_SHORT_SPREAD', {'z_score': z_score})
    
    def on_data(self, simulator: Any, symbol: str, data: pd.Series):
        """Single symbol - not applicable for pairs trading"""
        pass
    
    def _calculate_spread_zscore(self) -> float:
        """
        Calculate z-score of current spread
        
        Returns:
            Z-score
        """
        spreads = np.array(self.spreads)
        
        current_spread = spreads[-1]
        mean_spread = np.mean(spreads[:-1])
        std_spread = np.std(spreads[:-1])
        
        if std_spread > 0:
            z_score = (current_spread - mean_spread) / std_spread
        else:
            z_score = 0.0
        
        return z_score
