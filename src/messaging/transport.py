"""
Message Transport Backends

Production-grade messaging transports for inter-process communication.
All backends implement the MessageTransport protocol for hot-swappability.

                           ┌─────────────────────────────────┐
                           │         MessageTransport        │
                           │  publish() / subscribe() API    │
                           └──────────┬──────────────────────┘
                                      │
              ┌───────────────────────┼──────────────────────────┐
              │                       │                          │
     ┌────────▼────────┐   ┌─────────▼──────────┐   ┌──────────▼──────────┐
     │  AeronTransport │   │  KafkaTransport    │   │ InMemoryTransport   │
     │  (Shared Memory │   │  (Persistent       │   │ (Testing/Dev)       │
     │   IPC, ~1µs)    │   │   Streaming)       │   │                     │
     └─────────────────┘   └────────────────────┘   └─────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TransportBackend(str, Enum):
    AERON = "aeron"
    KAFKA = "kafka"
    REDPANDA = "redpanda"
    IN_MEMORY = "in_memory"


@dataclass
class TransportConfig:
    backend: TransportBackend = TransportBackend.IN_MEMORY
    
    # Kafka / Redpanda
    bootstrap_servers: str = "localhost:9092"
    schema_registry_url: str = "http://localhost:8081"
    consumer_group: str = "omniquant"
    
    # Aeron
    aeron_directory: str = "/dev/shm/aeron"
    aeron_stream_id: int = 1001
    aeron_channel: str = "aeron:ipc"
    
    # Shared settings
    batch_size: int = 1000
    linger_ms: int = 0      # 0 = no batching delay (lowest latency)
    compression: str = "none"  # "none", "lz4", "zstd", "snappy"
    
    # Reliability
    acks: str = "all"        # "all", "1", "0"
    retries: int = 3
    idempotent: bool = True
    
    # Performance
    buffer_memory: int = 64 * 1024 * 1024  # 64 MB
    max_request_size: int = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MessageEnvelope:
    message_id: str
    topic: str
    key: str           # Partition key (e.g., symbol)
    payload: bytes     # Serialized message body
    timestamp_ns: int  # Nanosecond precision
    headers: Dict[str, str] = field(default_factory=dict)
    
    @property
    def timestamp_ms(self) -> int:
        return self.timestamp_ns // 1_000_000


# ---------------------------------------------------------------------------
# Transport interface
# ---------------------------------------------------------------------------

class MessageTransport(ABC):
    """Abstract message transport with hot-swappable backends."""
    
    @abstractmethod
    async def start(self) -> None:
        ...
    
    @abstractmethod
    async def stop(self) -> None:
        ...
    
    @abstractmethod
    async def publish(self, topic: str, key: str, payload: bytes, 
                      headers: Optional[Dict[str, str]] = None) -> str:
        ...
    
    @abstractmethod
    async def subscribe(self, topic: str, callback: Callable[[MessageEnvelope], None],
                        group: Optional[str] = None) -> None:
        ...
    
    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# In-Memory Transport (development / testing)
# ---------------------------------------------------------------------------

class InMemoryTransport(MessageTransport):
    """In-memory transport for development and testing."""
    
    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        self.config = config or TransportConfig(backend=TransportBackend.IN_MEMORY)
        self._subscribers: Dict[str, List[Callable[[MessageEnvelope], None]]] = defaultdict(list)
        self._message_log: List[MessageEnvelope] = []
        self._running = False
        self._publish_count = 0
        self._deliver_count = 0
        self._start_time = 0.0
        self._latencies_ns: List[int] = []
    
    async def start(self) -> None:
        self._running = True
        self._start_time = time.time()
        logger.info("InMemory transport started")
    
    async def stop(self) -> None:
        self._running = False
        logger.info("InMemory transport stopped")
    
    async def publish(self, topic: str, key: str, payload: bytes,
                      headers: Optional[Dict[str, str]] = None) -> str:
        msg_id = uuid.uuid4().hex[:16]
        publish_ts = time.time_ns()
        
        envelope = MessageEnvelope(
            message_id=msg_id,
            topic=topic,
            key=key,
            payload=payload,
            timestamp_ns=publish_ts,
            headers=headers or {},
        )
        
        self._message_log.append(envelope)
        self._publish_count += 1
        
        # Deliver to subscribers (supports both sync and async callbacks)
        for callback in self._subscribers.get(topic, []):
            try:
                result = callback(envelope)
                if asyncio.iscoroutine(result):
                    await result
                self._deliver_count += 1
                self._latencies_ns.append(time.time_ns() - publish_ts)
            except Exception as e:
                logger.error(f"Subscriber error on {topic}: {e}")
        
        # Wildcard subscribers
        for callback in self._subscribers.get("*", []):
            try:
                result = callback(envelope)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Wildcard subscriber error: {e}")
        
        return msg_id
    
    async def subscribe(self, topic: str, callback: Callable[[MessageEnvelope], None],
                        group: Optional[str] = None) -> None:
        self._subscribers[topic].append(callback)
        logger.debug(f"Subscribed to {topic}")
    
    async def unsubscribe(self, topic: str) -> None:
        self._subscribers.pop(topic, None)
    
    def get_metrics(self) -> Dict[str, Any]:
        import numpy as np
        latencies = np.array(self._latencies_ns) / 1000.0 if self._latencies_ns else []
        return {
            "backend": "in_memory",
            "published": self._publish_count,
            "delivered": self._deliver_count,
            "topics": list(self._subscribers.keys()),
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "avg_latency_us": float(np.mean(latencies)) if len(latencies) > 0 else 0,
            "p99_latency_us": float(np.percentile(latencies, 99)) if len(latencies) > 0 else 0,
        }


# ---------------------------------------------------------------------------
# Aeron Transport (ultra-low-latency shared memory IPC)
# ---------------------------------------------------------------------------

class AeronTransport(MessageTransport):
    """Aeron shared-memory IPC transport (~1us latency)."""
    
    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        self.config = config or TransportConfig(backend=TransportBackend.AERON)
        self._aeron = None
        self._publications: Dict[str, Any] = {}
        self._subscriptions: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._publish_count = 0
    
    async def start(self) -> None:
        try:
            import aeron
            self._aeron = aeron.Aeron(
                aeron.Context().aeron_directory(self.config.aeron_directory)
            )
            self._running = True
            logger.info(f"Aeron transport started (dir={self.config.aeron_directory})")
        except ImportError:
            raise ImportError(
                "aeron-python is required for AeronTransport. "
                "Install it: pip install aeron-python  — or switch to "
                "TransportBackend.IN_MEMORY for development."
            )
    
    async def stop(self) -> None:
        self._running = False
        if self._aeron:
            self._aeron.close()
        logger.info("Aeron transport stopped")
    
    async def publish(self, topic: str, key: str, payload: bytes,
                      headers: Optional[Dict[str, str]] = None) -> str:
        msg_id = uuid.uuid4().hex[:16]
        
        if self._aeron and topic in self._publications:
            pub = self._publications[topic]
            # Aeron uses direct buffer offers
            result = pub.offer(payload)
            if result < 0:
                logger.warning(f"Aeron publish backpressure on {topic}")
        
        self._publish_count += 1
        
        # Deliver to local callbacks (supports async)
        envelope = MessageEnvelope(
            message_id=msg_id, topic=topic, key=key,
            payload=payload, timestamp_ns=time.time_ns(),
            headers=headers or {},
        )
        for cb in self._callbacks.get(topic, []):
            try:
                result = cb(envelope)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Aeron callback error: {e}")
        
        return msg_id
    
    async def subscribe(self, topic: str, callback: Callable[[MessageEnvelope], None],
                        group: Optional[str] = None) -> None:
        self._callbacks[topic].append(callback)
        
        if self._aeron:
            channel = self.config.aeron_channel
            stream_id = self.config.aeron_stream_id + hash(topic) % 1000
            
            try:
                sub = self._aeron.add_subscription(channel, stream_id)
                self._subscriptions[topic] = sub
            except Exception as e:
                logger.warning(f"Aeron subscription setup failed: {e}")
    
    async def unsubscribe(self, topic: str) -> None:
        self._callbacks.pop(topic, None)
        sub = self._subscriptions.pop(topic, None)
        if sub:
            sub.close()
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "backend": "aeron",
            "published": self._publish_count,
            "topics": list(self._callbacks.keys()),
            "aeron_connected": self._aeron is not None,
        }


# ---------------------------------------------------------------------------
# Kafka Transport (high-throughput persistent streaming)
# ---------------------------------------------------------------------------

class KafkaTransport(MessageTransport):
    """Kafka transport for persistent high-throughput streaming."""
    
    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        self.config = config or TransportConfig(backend=TransportBackend.KAFKA)
        self._producer = None
        self._consumers: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._publish_count = 0
        self._consumer_tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        try:
            from confluent_kafka import Producer
            
            producer_config = {
                "bootstrap.servers": self.config.bootstrap_servers,
                "acks": self.config.acks,
                "retries": self.config.retries,
                "enable.idempotence": self.config.idempotent,
                "batch.size": self.config.batch_size,
                "linger.ms": self.config.linger_ms,
                "compression.type": self.config.compression,
                "buffer.memory": self.config.buffer_memory,
                "max.request.size": self.config.max_request_size,
            }
            
            self._producer = Producer(producer_config)
            self._running = True
            logger.info(f"Kafka transport started ({self.config.bootstrap_servers})")
            
        except ImportError:
            raise ImportError(
                "confluent-kafka is required for KafkaTransport. "
                "Install it: pip install confluent-kafka  — or switch to "
                "TransportBackend.IN_MEMORY for development."
            )
    
    async def stop(self) -> None:
        if self._producer:
            self._producer.flush(timeout=10)
        for task in self._consumer_tasks:
            task.cancel()
        self._running = False
        logger.info("Kafka transport stopped")
    
    async def publish(self, topic: str, key: str, payload: bytes,
                      headers: Optional[Dict[str, str]] = None) -> str:
        msg_id = uuid.uuid4().hex[:16]
        
        if self._producer:
            kafka_headers = [(k, v.encode()) for k, v in (headers or {}).items()]
            kafka_headers.append(("msg_id", msg_id.encode()))
            
            self._producer.produce(
                topic=topic,
                key=key.encode(),
                value=payload,
                headers=kafka_headers,
            )
            
            # Trigger delivery callbacks
            self._producer.poll(0)
        
        self._publish_count += 1
        
        # Local delivery (supports async callbacks)
        envelope = MessageEnvelope(
            message_id=msg_id, topic=topic, key=key,
            payload=payload, timestamp_ns=time.time_ns(),
            headers=headers or {},
        )
        for cb in self._callbacks.get(topic, []):
            try:
                result = cb(envelope)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Kafka callback error: {e}")
        
        return msg_id
    
    async def subscribe(self, topic: str, callback: Callable[[MessageEnvelope], None],
                        group: Optional[str] = None) -> None:
        self._callbacks[topic].append(callback)
        
        if self._producer is not None:  # Kafka is available
            try:
                from confluent_kafka import Consumer
                
                consumer_config = {
                    "bootstrap.servers": self.config.bootstrap_servers,
                    "group.id": group or self.config.consumer_group,
                    "auto.offset.reset": "latest",
                    "enable.auto.commit": True,
                }
                
                consumer = Consumer(consumer_config)
                consumer.subscribe([topic])
                self._consumers[topic] = consumer
                
                # Start polling in background
                task = asyncio.create_task(self._poll_loop(topic, consumer))
                self._consumer_tasks.append(task)
                
            except ImportError:
                pass
    
    async def _poll_loop(self, topic: str, consumer: Any) -> None:
        while self._running:
            try:
                msg = consumer.poll(timeout=0.01)
                if msg is None:
                    await asyncio.sleep(0.001)
                    continue
                if msg.error():
                    logger.error(f"Kafka consumer error: {msg.error()}")
                    continue
                
                headers_dict = {}
                if msg.headers():
                    headers_dict = {k: v.decode() for k, v in msg.headers()}
                
                envelope = MessageEnvelope(
                    message_id=headers_dict.get("msg_id", ""),
                    topic=topic,
                    key=msg.key().decode() if msg.key() else "",
                    payload=msg.value(),
                    timestamp_ns=time.time_ns(),
                    headers=headers_dict,
                )
                
                for cb in self._callbacks.get(topic, []):
                    cb(envelope)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Kafka poll error: {e}")
                await asyncio.sleep(1.0)
    
    async def unsubscribe(self, topic: str) -> None:
        self._callbacks.pop(topic, None)
        consumer = self._consumers.pop(topic, None)
        if consumer:
            consumer.close()
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "backend": "kafka",
            "published": self._publish_count,
            "bootstrap_servers": self.config.bootstrap_servers,
            "topics": list(self._callbacks.keys()),
            "kafka_connected": self._producer is not None,
        }


# ---------------------------------------------------------------------------
# Redpanda Transport (Kafka-compatible, simpler operations)
# ---------------------------------------------------------------------------

class RedpandaTransport(KafkaTransport):
    """Kafka-compatible Redpanda transport with optimized defaults."""
    
    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        super().__init__(config)
        # Redpanda-specific optimizations
        if self.config.compression == "none":
            self.config.compression = "lz4"  # Redpanda handles LZ4 very efficiently

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["backend"] = "redpanda"
        return metrics


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_transport(config: Optional[TransportConfig] = None) -> MessageTransport:
    """Create a message transport from the given config."""
    config = config or TransportConfig()
    
    transport_map = {
        TransportBackend.AERON: AeronTransport,
        TransportBackend.KAFKA: KafkaTransport,
        TransportBackend.REDPANDA: RedpandaTransport,
        TransportBackend.IN_MEMORY: InMemoryTransport,
    }
    
    transport_cls = transport_map.get(config.backend, InMemoryTransport)
    return transport_cls(config)
