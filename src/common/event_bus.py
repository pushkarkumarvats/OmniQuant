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
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'priority': self.priority.value,
            'source': self.source
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# Trading-specific events
@dataclass
class MarketDataEvent(Event):
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
    """Pub/sub event bus supporting sync and async handlers."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = 1000
        logger.info("Event Bus initialized")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Register a handler for the given event type."""
        if asyncio.iscoroutinefunction(handler):
            self._async_subscribers[event_type].append(handler)
            logger.debug(f"Async subscriber added for {event_type}: {handler.__name__}")
        else:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscriber added for {event_type}: {handler.__name__}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
        if handler in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].remove(handler)
        logger.debug(f"Unsubscribed {handler.__name__} from {event_type}")
    
    def publish(self, event: Event):
        """Dispatch event to all matching subscribers synchronously."""
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
        """Async version of publish — awaits async handlers then fires sync ones."""
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
        """Return recent events, optionally filtered by type."""
        if event_type:
            events = [e for e in self._event_history if e.event_type == event_type]
        else:
            events = self._event_history
        
        return events[-limit:]
    
    def clear_history(self):
        self._event_history.clear()
        logger.debug("Event history cleared")
    
    def get_subscriber_count(self, event_type: str) -> int:
        return len(self._subscribers[event_type]) + len(self._async_subscribers[event_type])


# Global event bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    return _event_bus


class RedisEventBus(EventBus):
    """Redis-backed event bus for distributed pub/sub."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
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


class ChannelBackedEventBus(EventBus):
    """
    EventBus backed by Phase 2.0 messaging channels.

    Publishes events through the new typed channel infrastructure while
    keeping the legacy EventBus API fully compatible.  Incoming channel
    messages are re-emitted as local events for old-style subscribers.

    Threading contract:
        * ``publish()`` is called from synchronous (non-async) code.
          It cannot ``await`` the channel directly, so it schedules the
          publish as a fire-and-forget task on the running loop.  If no
          loop is running it silently drops the channel-side publish
          (local subscribers still fire).
        * ``publish_async()`` is the preferred path when inside an async
          context - it awaits the channel publish directly.
    """

    # Map legacy event_type strings to channel names
    _EVENT_CHANNEL_MAP: Dict[str, str] = {
        "market_data": "market_data",
        "order": "orders",
        "trade": "fills",
        "signal": "signals",
        "risk": "risk",
    }

    def __init__(self, channel_manager):
        super().__init__()
        self._channel_manager = channel_manager
        logger.info("ChannelBackedEventBus initialized (Phase 2.0 messaging)")

    def publish(self, event: Event):
        """Publish event to local subscribers AND the matching channel.

        The channel publish is scheduled as a fire-and-forget coroutine on
        the running event loop.  This avoids the deadlock that would occur
        if we called ``loop.run_until_complete()`` inside an already-running
        loop.  If no loop is active the channel publish is skipped (local
        delivery still happens).
        """
        super().publish(event)  # local delivery first (always sync)

        channel_name = self._EVENT_CHANNEL_MAP.get(event.event_type)
        if not channel_name:
            return
        channel = self._channel_manager.get(channel_name)
        if not channel:
            return

        try:
            loop = asyncio.get_running_loop()
            # We are inside a running loop - schedule without blocking
            loop.create_task(channel.publish(event.to_dict()))
        except RuntimeError:
            # No running loop - cannot publish to async channel from sync
            # context.  Local subscribers already received the event above.
            pass

    async def publish_async(self, event: Event):
        """Async publish to local subscribers AND the matching channel."""
        await super().publish_async(event)

        channel_name = self._EVENT_CHANNEL_MAP.get(event.event_type)
        if channel_name:
            channel = self._channel_manager.get(channel_name)
            if channel:
                await channel.publish(event.to_dict())


def get_channel_backed_event_bus():
    """
    Create an event bus wired to Phase 2.0 messaging channels.
    Falls back to the plain in-memory EventBus if channels aren't available.
    """
    try:
        from src.messaging.transport import TransportConfig, TransportBackend, create_transport
        from src.messaging.channels import ChannelManager

        transport = create_transport(TransportConfig(backend=TransportBackend.IN_MEMORY))
        channels = ChannelManager.create_default(transport)
        return ChannelBackedEventBus(channels)
    except Exception as e:
        logger.warning(f"Channel-backed event bus unavailable, using in-memory: {e}")
        return EventBus()


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
