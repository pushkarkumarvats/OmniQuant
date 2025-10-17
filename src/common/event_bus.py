"""
Event Bus / Message Queue
Implements event-driven architecture with pub/sub pattern
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from collections import defaultdict
from loguru import logger
import json


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Base event class"""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'priority': self.priority.value,
            'source': self.source
        }
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict())


# Trading-specific events
@dataclass
class MarketDataEvent(Event):
    """Market data update event"""
    symbol: str = ""
    price: float = 0.0
    volume: int = 0
    
    def __post_init__(self):
        self.event_type = "market_data"
        self.data.update({
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume
        })


@dataclass
class OrderEvent(Event):
    """Order event"""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0
    
    def __post_init__(self):
        self.event_type = "order"
        self.data.update({
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price
        })


@dataclass
class TradeEvent(Event):
    """Trade execution event"""
    trade_id: str = ""
    symbol: str = ""
    quantity: int = 0
    price: float = 0.0
    
    def __post_init__(self):
        self.event_type = "trade"
        self.data.update({
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price
        })


@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    symbol: str = ""
    signal: str = ""
    strength: float = 0.0
    
    def __post_init__(self):
        self.event_type = "signal"
        self.data.update({
            'symbol': self.symbol,
            'signal': self.signal,
            'strength': self.strength
        })


class EventBus:
    """
    Event Bus implementation with pub/sub pattern
    Supports synchronous and asynchronous handlers
    """
    
    def __init__(self):
        """Initialize event bus"""
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = 1000
        logger.info("Event Bus initialized")
    
    def subscribe(self, event_type: str, handler: Callable):
        """
        Subscribe to an event type
        
        Args:
            event_type: Type of event to subscribe to
            handler: Handler function (receives Event object)
        """
        if asyncio.iscoroutinefunction(handler):
            self._async_subscribers[event_type].append(handler)
            logger.debug(f"Async subscriber added for {event_type}: {handler.__name__}")
        else:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscriber added for {event_type}: {handler.__name__}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """
        Unsubscribe from an event type
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
        if handler in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].remove(handler)
        logger.debug(f"Unsubscribed {handler.__name__} from {event_type}")
    
    def publish(self, event: Event):
        """
        Publish an event synchronously
        
        Args:
            event: Event to publish
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Call synchronous subscribers
        for handler in self._subscribers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__name__}: {e}")
        
        # Also call wildcard subscribers
        for handler in self._subscribers['*']:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in wildcard handler {handler.__name__}: {e}")
        
        logger.debug(f"Published {event.event_type} from {event.source}")
    
    async def publish_async(self, event: Event):
        """
        Publish an event asynchronously
        
        Args:
            event: Event to publish
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Call async subscribers
        tasks = []
        for handler in self._async_subscribers[event.event_type]:
            tasks.append(handler(event))
        
        for handler in self._async_subscribers['*']:
            tasks.append(handler(event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Also call sync subscribers
        self.publish(event)
        
        logger.debug(f"Published async {event.event_type} from {event.source}")
    
    def get_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """
        Get event history
        
        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        if event_type:
            events = [e for e in self._event_history if e.event_type == event_type]
        else:
            events = self._event_history
        
        return events[-limit:]
    
    def clear_history(self):
        """Clear event history"""
        self._event_history.clear()
        logger.debug("Event history cleared")
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type"""
        return len(self._subscribers[event_type]) + len(self._async_subscribers[event_type])


# Global event bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get global event bus"""
    return _event_bus


class RedisEventBus(EventBus):
    """
    Redis-backed event bus for distributed systems
    Requires redis-py package
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis event bus
        
        Args:
            redis_url: Redis connection URL
        """
        super().__init__()
        try:
            import redis
            self.redis_client = redis.from_url(redis_url)
            self.pubsub = self.redis_client.pubsub()
            logger.info(f"Redis Event Bus connected to {redis_url}")
        except ImportError:
            logger.error("redis-py not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def publish(self, event: Event):
        """Publish event to Redis"""
        super().publish(event)
        
        # Publish to Redis channel
        channel = f"omniquant:{event.event_type}"
        self.redis_client.publish(channel, event.to_json())
    
    def subscribe_redis(self, event_type: str):
        """Subscribe to Redis channel"""
        channel = f"omniquant:{event_type}"
        self.pubsub.subscribe(channel)
        logger.info(f"Subscribed to Redis channel: {channel}")
    
    def listen(self):
        """Listen for Redis messages"""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    event = Event(**data)
                    super().publish(event)
                except Exception as e:
                    logger.error(f"Error processing Redis message: {e}")


if __name__ == "__main__":
    # Example usage
    bus = get_event_bus()
    
    # Subscribe to events
    def on_market_data(event: Event):
        print(f"Market data: {event.data}")
    
    def on_trade(event: Event):
        print(f"Trade executed: {event.data}")
    
    bus.subscribe("market_data", on_market_data)
    bus.subscribe("trade", on_trade)
    
    # Publish events
    market_event = MarketDataEvent(symbol="AAPL", price=150.0, volume=1000)
    bus.publish(market_event)
    
    trade_event = TradeEvent(trade_id="T1", symbol="AAPL", quantity=100, price=150.0)
    bus.publish(trade_event)
    
    # Get history
    history = bus.get_history()
    print(f"\nEvent history: {len(history)} events")
