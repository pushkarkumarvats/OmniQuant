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
        self.symbols = symbols
        self.config = config or {}
        self.is_connected = False
        self.event_bus = get_event_bus()
    
    @abstractmethod
    async def connect(self):
        pass
    
    @abstractmethod
    async def disconnect(self):
        pass


class AlpacaConnector(DataConnector):
    """Alpaca Markets live data connector (requires alpaca-py)."""

    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        super().__init__(symbols, config)
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self._stream = None

    async def connect(self):
        if not self.api_key or not self.secret_key:
            raise ValueError("AlpacaConnector requires 'api_key' and 'secret_key' in config")

        try:
            from alpaca.data.live import StockDataStream
        except ImportError:
            raise ImportError(
                "alpaca-py is required for AlpacaConnector. "
                "Install it: pip install alpaca-py"
            )

        self._stream = StockDataStream(self.api_key, self.secret_key)

        async def _on_quote(data):
            quote = Quote(
                symbol=data.symbol,
                timestamp=data.timestamp,
                bid=data.bid_price,
                ask=data.ask_price,
                bid_size=data.bid_size,
                ask_size=data.ask_size,
            )
            self.event_bus.publish("market_data", MarketDataEvent(data={
                "symbol": quote.symbol,
                "bid": quote.bid,
                "ask": quote.ask,
                "mid": quote.mid_price,
                "spread": quote.spread,
            }))

        self._stream.subscribe_quotes(_on_quote, *self.symbols)
        self.is_connected = True
        logger.info(f"Alpaca stream connected for {self.symbols}")

    async def disconnect(self):
        if self._stream:
            await self._stream.close()
        self.is_connected = False
        logger.info("Disconnected from Alpaca")


class PolygonConnector(DataConnector):
    """Polygon.io WebSocket data connector."""

    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        super().__init__(symbols, config)
        self.api_key = config.get('api_key')
        self._ws = None
        self._task = None

    async def connect(self):
        if not self.api_key:
            raise ValueError("PolygonConnector requires 'api_key' in config")

        try:
            import websockets  # noqa: F401
        except ImportError:
            raise ImportError(
                "websockets is required for PolygonConnector. "
                "Install it: pip install websockets"
            )

        import websockets
        import json as _json

        url = "wss://socket.polygon.io/stocks"
        self._ws = await websockets.connect(url)
        # Authenticate
        await self._ws.send(_json.dumps({"action": "auth", "params": self.api_key}))

        # Subscribe
        tickers = ",".join(f"Q.{s}" for s in self.symbols)
        await self._ws.send(_json.dumps({"action": "subscribe", "params": tickers}))
        self.is_connected = True
        logger.info(f"Polygon.io WebSocket connected for {self.symbols}")

        self._task = asyncio.ensure_future(self._read_loop())

    async def _read_loop(self):
        import json as _json
        try:
            async for raw in self._ws:
                messages = _json.loads(raw)
                for msg in (messages if isinstance(messages, list) else [messages]):
                    if msg.get("ev") == "Q":
                        self.event_bus.publish("market_data", MarketDataEvent(data={
                            "symbol": msg.get("sym"),
                            "bid": msg.get("bp"),
                            "ask": msg.get("ap"),
                            "bid_size": msg.get("bs"),
                            "ask_size": msg.get("as"),
                        }))
        except Exception as e:
            logger.error(f"Polygon read loop error: {e}")

    async def disconnect(self):
        if self._task:
            self._task.cancel()
        if self._ws:
            await self._ws.close()
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
    """Build a connector by short name or fully-qualified class path."""
    _builtins = {
        'alpaca': AlpacaConnector,
        'polygon': PolygonConnector,
        'simulated': SimulatedConnector,
    }

    if connector_type in _builtins:
        return _builtins[connector_type](symbols, config)

    # Try dynamic import: "my_package.connectors:MyConnector"
    import importlib
    if ":" in connector_type:
        mod_path, cls_name = connector_type.rsplit(":", 1)
    elif "." in connector_type:
        mod_path, cls_name = connector_type.rsplit(".", 1)
    else:
        raise ValueError(
            f"Unknown connector type: '{connector_type}'. "
            f"Available built-ins: {list(_builtins)}, "
            f"or pass 'module.path:ClassName'."
        )

    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)
    return cls(symbols, config)
