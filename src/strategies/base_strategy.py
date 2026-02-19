"""
Base Strategy Class
Foundation for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from loguru import logger


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        
        # Performance tracking
        self.trades = []
        self.signals = []
        self.metrics = {}
    
    @abstractmethod
    def initialize(self, simulator: Any):
        """Called once before backtest starts."""
        pass
    
    @abstractmethod
    def on_data(self, simulator: Any, symbol: str, data: pd.Series):
        """Called on each new bar / tick."""
        pass
    
    def finalize(self, simulator: Any):
        logger.info(f"Strategy {self.name} finalized")
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
    
    def log_signal(self, signal_type: str, details: Dict[str, Any]):
        self.signals.append({
            'type': signal_type,
            'timestamp': details.get('timestamp'),
            **details
        })
    
    def log_trade(self, trade_details: Dict[str, Any]):
        self.trades.append(trade_details)
    
    def __str__(self) -> str:
        return f"{self.name} Strategy"
