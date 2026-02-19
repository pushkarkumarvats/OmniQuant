"""
High-Throughput Messaging Backbone

Replaces the in-memory Python EventBus with a production-grade,
low-latency messaging system for inter-process communication.

Supported backends:
  - Aeron (ultra-low-latency, shared memory IPC)
  - Kafka / Redpanda (high-throughput, persistent streaming)
  - In-memory (for testing, backward compatible)

Architecture:
    Producer -> Serializer -> Transport Backend -> Deserializer -> Consumer
    
All backends implement the same MessageTransport interface so they can
be swapped via configuration without code changes.
"""

from .transport import (
    MessageTransport,
    TransportConfig,
    AeronTransport,
    KafkaTransport,
    RedpandaTransport,
    InMemoryTransport,
    create_transport,
)
from .serialization import (
    MessageSerializer,
    FlatBufferSerializer,
    ProtobufSerializer,
    MsgPackSerializer,
    JsonSerializer,
)
from .channels import (
    Channel,
    ChannelConfig,
    MarketDataChannel,
    OrderChannel,
    FillChannel,
    SignalChannel,
    RiskChannel,
)

__all__ = [
    "MessageTransport",
    "TransportConfig",
    "AeronTransport",
    "KafkaTransport",
    "RedpandaTransport",
    "InMemoryTransport",
    "create_transport",
    "MessageSerializer",
    "FlatBufferSerializer",
    "ProtobufSerializer",
    "MsgPackSerializer",
    "JsonSerializer",
    "Channel",
    "ChannelConfig",
    "MarketDataChannel",
    "OrderChannel",
    "FillChannel",
    "SignalChannel",
    "RiskChannel",
]
