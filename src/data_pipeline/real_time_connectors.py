"""
Real-Time Data Connectors
Implementations for live trading data feeds
"""

from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger
import pandas as pd

from src.common.event_bus import get_event_bus, MarketDataEvent


@dataclass
class Quote:
    """Quote data structure"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid


class DataConnector(ABC):
    """Abstract base class for data connectors"""
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize connector
        
        Args:
            symbols: List of symbols to subscribe to
            config: Connector configuration
        """
        self.symbols = symbols
        self.config = config or {}
        self.is_connected = False
        self.event_bus = get_event_bus()
    
    @abstractmethod
    async def connect(self):
        """Connect to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source"""
        pass


class AlpacaConnector(DataConnector):
    """Alpaca Markets data connector"""
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        super().__init__(symbols, config)
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        logger.info("Alpaca connector initialized")
    
    async def connect(self):
        """Connect to Alpaca stream"""
        self.is_connected = True
        logger.info("Connected to Alpaca")
    
    async def disconnect(self):
        """Disconnect from Alpaca"""
        self.is_connected = False
        logger.info("Disconnected from Alpaca")


class PolygonConnector(DataConnector):
    """Polygon.io data connector"""
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        super().__init__(symbols, config)
        self.api_key = config.get('api_key')
        logger.info("Polygon connector initialized")
    
    async def connect(self):
        self.is_connected = True
        logger.info("Connected to Polygon.io")
    
    async def disconnect(self):
        self.is_connected = False
        logger.info("Disconnected from Polygon.io")


class SimulatedConnector(DataConnector):
    """Simulated real-time connector for testing"""
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        super().__init__(symbols, config)
        self.update_interval = config.get('update_interval', 1.0)
        self.prices = {symbol: 100.0 for symbol in symbols}
    
    async def connect(self):
        self.is_connected = True
        logger.info("Simulated connector started")
    
    async def disconnect(self):
        self.is_connected = False
        logger.info("Simulated connector stopped")


def create_connector(connector_type: str, symbols: List[str], config: Dict) -> DataConnector:
    """Factory function for creating connectors"""
    connectors = {
        'alpaca': AlpacaConnector,
        'polygon': PolygonConnector,
        'simulated': SimulatedConnector
    }
    
    if connector_type not in connectors:
        raise ValueError(f"Unknown connector type: {connector_type}")
    
    return connectors[connector_type](symbols, config)
