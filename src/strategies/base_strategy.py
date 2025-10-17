"""
Base Strategy Class
Foundation for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from loguru import logger


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        
        # Performance tracking
        self.trades = []
        self.signals = []
        self.metrics = {}
    
    @abstractmethod
    def initialize(self, simulator: Any):
        """
        Initialize strategy before backtest
        
        Args:
            simulator: Event simulator instance
        """
        pass
    
    @abstractmethod
    def on_data(self, simulator: Any, symbol: str, data: pd.Series):
        """
        Process new market data
        
        Args:
            simulator: Event simulator instance
            symbol: Trading symbol
            data: Market data
        """
        pass
    
    def finalize(self, simulator: Any):
        """
        Finalize strategy after backtest
        
        Args:
            simulator: Event simulator instance
        """
        logger.info(f"Strategy {self.name} finalized")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def log_signal(self, signal_type: str, details: Dict[str, Any]):
        """
        Log trading signal
        
        Args:
            signal_type: Type of signal
            details: Signal details
        """
        self.signals.append({
            'type': signal_type,
            'timestamp': details.get('timestamp'),
            **details
        })
    
    def log_trade(self, trade_details: Dict[str, Any]):
        """
        Log executed trade
        
        Args:
            trade_details: Trade information
        """
        self.trades.append(trade_details)
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name} Strategy"
