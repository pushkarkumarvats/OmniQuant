"""
Typed Message Channels

Strongly-typed pub/sub channels for each domain concern:
  - MarketDataChannel: Tick, bar, and order-book snapshot distribution
  - OrderChannel: Order lifecycle events
  - FillChannel: Execution reports
  - SignalChannel: Alpha / strategy signals
  - RiskChannel: Risk alerts and position limits
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from loguru import logger

from .serialization import MessageSerializer, MsgPackSerializer
from .transport import MessageEnvelope, MessageTransport


class ChannelPriority(Enum):
    CRITICAL = 0   # Risk breaches, kill switches
    HIGH = 1       # Order lifecycle, fills
    NORMAL = 2     # Market data, signals
    LOW = 3        # Analytics, logging


@dataclass
class ChannelConfig:
    name: str
    topic_prefix: str
    priority: ChannelPriority = ChannelPriority.NORMAL
    serializer: Optional[MessageSerializer] = None
    max_batch_size: int = 1
    flush_interval_ms: int = 0  # 0 = immediate
    enable_replay: bool = False
    replay_window_seconds: int = 3600
    compression: bool = False


class Channel:
    """Typed pub/sub channel with serialization and optional batching."""

    def __init__(
        self,
        config: ChannelConfig,
        transport: MessageTransport,
    ) -> None:
        self._config = config
        self._transport = transport
        self._serializer = config.serializer or MsgPackSerializer()
        self._subscribers: List[Callable] = []
        self._batch: List[Dict[str, Any]] = []
        self._last_flush = time.monotonic_ns()
        self._message_count = 0
        self._running = False

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def topic(self) -> str:
        return self._config.topic_prefix

    async def start(self) -> None:
        self._running = True
        await self._transport.subscribe(self.topic, self._on_message)
        logger.info(f"Channel '{self.name}' started on topic '{self.topic}'")

    async def stop(self) -> None:
        if self._batch:
            await self._flush_batch()
        self._running = False
        await self._transport.unsubscribe(self.topic)
        logger.info(f"Channel '{self.name}' stopped")

    async def publish(
        self,
        data: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        enriched_headers = {
            "channel": self.name,
            "priority": str(self._config.priority.value),
            "content-type": self._serializer.content_type,
            **(headers or {}),
        }

        if self._config.max_batch_size > 1:
            self._batch.append(data)
            if len(self._batch) >= self._config.max_batch_size:
                await self._flush_batch()
        else:
            payload = self._serializer.serialize(data)
            await self._transport.publish(
                topic=self.topic,
                key=key or "",
                payload=payload,
                headers=enriched_headers,
            )
            self._message_count += 1

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)

    async def _on_message(self, envelope: MessageEnvelope) -> None:
        data = self._serializer.deserialize(envelope.payload)
        for cb in self._subscribers:
            try:
                result = cb(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Channel '{self.name}' handler error: {e}")

    async def _flush_batch(self) -> None:
        if not self._batch:
            return
        batch_payload = self._serializer.serialize({"batch": self._batch})
        await self._transport.publish(
            topic=self.topic,
            key="batch",
            payload=batch_payload,
            headers={
                "channel": self.name,
                "batch_size": str(len(self._batch)),
            },
        )
        self._message_count += len(self._batch)
        self._batch.clear()
        self._last_flush = time.monotonic_ns()


# --------------------------------------------------------------------------- #
#  Domain-specific channels                                                    #
# --------------------------------------------------------------------------- #

class MarketDataChannel(Channel):
    """Tick, bar, and order-book data distribution."""

    def __init__(self, transport: MessageTransport, **kwargs) -> None:
        super().__init__(
            ChannelConfig(
                name="market_data",
                topic_prefix="md",
                priority=ChannelPriority.NORMAL,
                max_batch_size=kwargs.get("batch_size", 64),
                flush_interval_ms=kwargs.get("flush_ms", 5),
                compression=True,
            ),
            transport,
        )

    async def publish_tick(
        self, venue: str, symbol: str, price: float, size: float,
        side: str, timestamp_ns: int,
    ) -> None:
        await self.publish(
            {
                "type": "tick",
                "venue": venue,
                "symbol": symbol,
                "price": price,
                "size": size,
                "side": side,
                "ts": timestamp_ns,
            },
            key=f"{venue}.{symbol}",
        )

    async def publish_bar(
        self, venue: str, symbol: str, open_: float, high: float,
        low: float, close: float, volume: float, bar_ts: int,
    ) -> None:
        await self.publish(
            {
                "type": "bar",
                "venue": venue,
                "symbol": symbol,
                "o": open_, "h": high, "l": low, "c": close,
                "v": volume, "ts": bar_ts,
            },
            key=f"{venue}.{symbol}",
        )

    async def publish_book_snapshot(
        self, venue: str, symbol: str, bids: list, asks: list,
        timestamp_ns: int,
    ) -> None:
        await self.publish(
            {
                "type": "book",
                "venue": venue,
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "ts": timestamp_ns,
            },
            key=f"{venue}.{symbol}",
        )


class OrderChannel(Channel):
    """Order lifecycle events (new, ack, partial fill, cancel, reject)."""

    def __init__(self, transport: MessageTransport, **kwargs) -> None:
        super().__init__(
            ChannelConfig(
                name="orders",
                topic_prefix="order",
                priority=ChannelPriority.HIGH,
            ),
            transport,
        )

    async def publish_new_order(
        self, strategy_id: str, order_id: str, symbol: str,
        side: str, quantity: float, price: float, order_type: str,
    ) -> None:
        await self.publish(
            {
                "event": "new_order",
                "strategy_id": strategy_id,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "qty": quantity,
                "price": price,
                "order_type": order_type,
                "ts": time.time_ns(),
            },
            key=f"{strategy_id}.{order_id}",
        )

    async def publish_order_ack(
        self, strategy_id: str, order_id: str, exchange_order_id: str,
    ) -> None:
        await self.publish(
            {
                "event": "order_ack",
                "strategy_id": strategy_id,
                "order_id": order_id,
                "exchange_order_id": exchange_order_id,
                "ts": time.time_ns(),
            },
            key=f"{strategy_id}.{order_id}",
        )

    async def publish_order_reject(
        self, strategy_id: str, order_id: str, reason: str,
    ) -> None:
        await self.publish(
            {
                "event": "order_reject",
                "strategy_id": strategy_id,
                "order_id": order_id,
                "reason": reason,
                "ts": time.time_ns(),
            },
            key=f"{strategy_id}.{order_id}",
        )


class FillChannel(Channel):
    """Execution report distribution with replay support."""

    def __init__(self, transport: MessageTransport, **kwargs) -> None:
        super().__init__(
            ChannelConfig(
                name="fills",
                topic_prefix="fill",
                priority=ChannelPriority.HIGH,
                enable_replay=True,
                replay_window_seconds=86400,
            ),
            transport,
        )

    async def publish_fill(
        self, strategy_id: str, order_id: str, fill_price: float,
        fill_qty: float, commission: float, venue: str, is_partial: bool,
    ) -> None:
        await self.publish(
            {
                "event": "fill",
                "strategy_id": strategy_id,
                "order_id": order_id,
                "fill_price": fill_price,
                "fill_qty": fill_qty,
                "commission": commission,
                "venue": venue,
                "is_partial": is_partial,
                "ts": time.time_ns(),
            },
            key=f"{strategy_id}.{order_id}",
        )


class SignalChannel(Channel):
    """Alpha and strategy signal distribution."""

    def __init__(self, transport: MessageTransport, **kwargs) -> None:
        super().__init__(
            ChannelConfig(
                name="signals",
                topic_prefix="signal",
                priority=ChannelPriority.NORMAL,
            ),
            transport,
        )

    async def publish_signal(
        self, model_id: str, symbol: str, direction: float,
        strength: float, confidence: float, features: Optional[Dict] = None,
    ) -> None:
        await self.publish(
            {
                "event": "signal",
                "model_id": model_id,
                "symbol": symbol,
                "direction": direction,
                "strength": strength,
                "confidence": confidence,
                "features": features or {},
                "ts": time.time_ns(),
            },
            key=f"{model_id}.{symbol}",
        )


class RiskChannel(Channel):
    """Risk alerts, limit breaches, and kill switch commands."""

    def __init__(self, transport: MessageTransport, **kwargs) -> None:
        super().__init__(
            ChannelConfig(
                name="risk",
                topic_prefix="risk",
                priority=ChannelPriority.CRITICAL,
                enable_replay=True,
                replay_window_seconds=86400 * 7,
            ),
            transport,
        )

    async def publish_risk_alert(
        self, severity: str, entity: str, metric: str,
        current_value: float, limit_value: float, message: str,
    ) -> None:
        await self.publish(
            {
                "event": "risk_alert",
                "severity": severity,
                "entity": entity,
                "metric": metric,
                "current_value": current_value,
                "limit_value": limit_value,
                "message": message,
                "ts": time.time_ns(),
            },
            key=f"{severity}.{entity}",
        )

    async def publish_kill_switch(
        self, entity: str, reason: str, operator: str,
    ) -> None:
        await self.publish(
            {
                "event": "kill_switch",
                "entity": entity,
                "reason": reason,
                "operator": operator,
                "ts": time.time_ns(),
            },
            key=f"critical.{entity}",
        )


class ChannelManager:
    """Central registry and lifecycle manager for typed channels."""

    def __init__(self, transport: MessageTransport) -> None:
        self._transport = transport
        self._channels: Dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        self._channels[channel.name] = channel

    def get(self, name: str) -> Optional[Channel]:
        return self._channels.get(name)

    async def start_all(self) -> None:
        await self._transport.start()
        for ch in self._channels.values():
            await ch.start()
        logger.info(f"ChannelManager started {len(self._channels)} channels")

    async def stop_all(self) -> None:
        for ch in self._channels.values():
            await ch.stop()
        await self._transport.stop()
        logger.info("ChannelManager stopped")

    @classmethod
    def create_default(cls, transport: MessageTransport) -> "ChannelManager":
        """Create a manager with all standard institutional channels."""
        mgr = cls(transport)
        mgr.register(MarketDataChannel(transport))
        mgr.register(OrderChannel(transport))
        mgr.register(FillChannel(transport))
        mgr.register(SignalChannel(transport))
        mgr.register(RiskChannel(transport))
        return mgr
