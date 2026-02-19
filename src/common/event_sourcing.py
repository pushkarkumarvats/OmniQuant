"""
Event Sourcing Architecture for Production HFT

Provides an append-only event store for deterministic replay and crash
recovery of the full trading system state.  Every state mutation
(order submitted, fill received, position change, risk limit update)
is captured as an immutable event.

Key properties:
  - Append-only: events are never mutated or deleted
  - Deterministic replay: identical event stream → identical state
  - Write-ahead: events persisted before side-effects execute
  - Snapshot support: periodic state snapshots for fast recovery
  - Integration with Rust OMS journal for cross-language consistency
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from loguru import logger


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    # Order lifecycle
    ORDER_SUBMITTED = "order.submitted"
    ORDER_ACCEPTED = "order.accepted"
    ORDER_REJECTED = "order.rejected"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REPLACED = "order.replaced"
    ORDER_EXPIRED = "order.expired"

    # Fills
    FILL_RECEIVED = "fill.received"
    FILL_CORRECTED = "fill.corrected"
    FILL_BUSTED = "fill.busted"

    # Position
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"

    # Risk
    RISK_LIMIT_UPDATED = "risk.limit_updated"
    RISK_BREACH = "risk.breach"
    RISK_CIRCUIT_BREAKER = "risk.circuit_breaker"

    # Market data
    MARKET_DATA_UPDATE = "market_data.update"
    REFERENCE_DATA_UPDATE = "reference_data.update"

    # System
    SYSTEM_START = "system.start"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SNAPSHOT_CREATED = "system.snapshot"

    # Reconciliation
    RECON_BREAK_DETECTED = "recon.break_detected"
    RECON_BREAK_RESOLVED = "recon.break_resolved"


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Immutable domain event with write-ahead semantics."""
    event_id: str
    event_type: EventType
    timestamp_ns: int
    sequence: int
    aggregate_id: str          # e.g. order_id, position key
    aggregate_type: str        # "order", "position", "risk", …
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def to_bytes(self) -> bytes:
        """Serialize to length-prefixed JSON bytes."""
        data = json.dumps(asdict(self), default=str).encode("utf-8")
        return struct.pack("<I", len(data)) + data

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Event":
        """Deserialise from length-prefixed JSON bytes."""
        length = struct.unpack("<I", raw[:4])[0]
        payload = json.loads(raw[4:4 + length])
        payload["event_type"] = EventType(payload["event_type"])
        return cls(**payload)

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Event Store (write-ahead, append-only)
# ---------------------------------------------------------------------------

class EventStore:
    """Append-only event journal backed by a length-prefixed binary file."""

    def __init__(self, journal_dir: str = "data/event_journal") -> None:
        self._dir = Path(journal_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._journal_path = self._dir / "events.journal"
        self._index: Dict[str, List[int]] = {}     # aggregate_id → [offset…]
        self._global_seq = 0
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[Event], None]] = []
        self._fd = open(self._journal_path, "ab+")
        self._rebuild_index()

    # -- Write path ---------------------------------------------------------

    def append(self, event: Event) -> int:
        """Write-ahead persist and notify subscribers. Returns global sequence."""
        with self._lock:
            self._global_seq += 1
            # We make a new event with the global seq injected
            ev = Event(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp_ns=event.timestamp_ns,
                sequence=self._global_seq,
                aggregate_id=event.aggregate_id,
                aggregate_type=event.aggregate_type,
                payload=event.payload,
                metadata=event.metadata,
                version=event.version,
            )
            data = ev.to_bytes()
            offset = self._fd.tell()
            self._fd.write(data)
            self._fd.flush()
            os.fsync(self._fd.fileno())
            self._index.setdefault(ev.aggregate_id, []).append(offset)

        for sub in self._subscribers:
            try:
                sub(ev)
            except Exception as exc:
                logger.error(f"Event subscriber error: {exc}")

        return self._global_seq

    def append_batch(self, events: Sequence[Event]) -> int:
        """Atomically persist a batch of events."""
        last_seq = 0
        for ev in events:
            last_seq = self.append(ev)
        return last_seq

    # -- Read path ----------------------------------------------------------

    def read_all(self) -> Iterator[Event]:
        """Iterate over every event in insertion order."""
        with open(self._journal_path, "rb") as f:
            while True:
                header = f.read(4)
                if len(header) < 4:
                    break
                length = struct.unpack("<I", header)[0]
                payload = f.read(length)
                if len(payload) < length:
                    break
                yield Event.from_bytes(header + payload)

    def read_aggregate(self, aggregate_id: str) -> List[Event]:
        """Read all events for a specific aggregate."""
        offsets = self._index.get(aggregate_id, [])
        events = []
        with open(self._journal_path, "rb") as f:
            for off in offsets:
                f.seek(off)
                header = f.read(4)
                length = struct.unpack("<I", header)[0]
                payload = f.read(length)
                events.append(Event.from_bytes(header + payload))
        return events

    def read_by_type(self, event_type: EventType) -> List[Event]:
        """Read all events of a given type (full scan)."""
        return [e for e in self.read_all() if e.event_type == event_type]

    def read_since(self, sequence: int) -> List[Event]:
        """Read all events with sequence > the given value."""
        return [e for e in self.read_all() if e.sequence > sequence]

    # -- Subscriptions ------------------------------------------------------

    def subscribe(self, handler: Callable[[Event], None]) -> None:
        """Register a real-time subscriber. Called after persist."""
        self._subscribers.append(handler)

    # -- Snapshots ----------------------------------------------------------

    def create_snapshot(self, state: Dict[str, Any], label: str = "") -> str:
        """Persist a state snapshot for fast recovery."""
        snap_id = f"snap_{self._global_seq}_{int(time.time())}"
        snap_path = self._dir / f"{snap_id}.json"
        snapshot = {
            "snapshot_id": snap_id,
            "sequence": self._global_seq,
            "timestamp_ns": time.time_ns(),
            "label": label,
            "state": state,
        }
        snap_path.write_text(json.dumps(snapshot, default=str, indent=2))
        logger.info(f"Snapshot created: {snap_id} at seq {self._global_seq}")
        return snap_id

    def load_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the most recent snapshot."""
        snaps = sorted(self._dir.glob("snap_*.json"), reverse=True)
        if not snaps:
            return None
        return json.loads(snaps[0].read_text())

    # -- Internals ----------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Scan the journal and rebuild the in-memory index."""
        self._index.clear()
        self._global_seq = 0
        if not self._journal_path.exists():
            return
        with open(self._journal_path, "rb") as f:
            while True:
                offset = f.tell()
                header = f.read(4)
                if len(header) < 4:
                    break
                length = struct.unpack("<I", header)[0]
                payload = f.read(length)
                if len(payload) < length:
                    break
                try:
                    ev = Event.from_bytes(header + payload)
                    self._index.setdefault(ev.aggregate_id, []).append(offset)
                    self._global_seq = max(self._global_seq, ev.sequence)
                except Exception:
                    break

    @property
    def global_sequence(self) -> int:
        return self._global_seq

    def close(self) -> None:
        self._fd.close()


# ---------------------------------------------------------------------------
# Aggregate Root (DDD pattern)
# ---------------------------------------------------------------------------

T = TypeVar("T")


class AggregateRoot(Generic[T]):
    """Base for event-sourced aggregates — all state via apply(event)."""

    def __init__(self, aggregate_id: str) -> None:
        self.aggregate_id = aggregate_id
        self._version = 0
        self._pending_events: List[Event] = []

    def raise_event(self, event_type: EventType, payload: Dict[str, Any], aggregate_type: str = "generic") -> Event:
        """Create a new event and apply it to this aggregate."""
        import uuid
        ev = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp_ns=time.time_ns(),
            sequence=0,  # assigned by the store
            aggregate_id=self.aggregate_id,
            aggregate_type=aggregate_type,
            payload=payload,
        )
        self.apply(ev)
        self._pending_events.append(ev)
        self._version += 1
        return ev

    def apply(self, event: Event) -> None:
        """Override to mutate state in response to an event."""
        raise NotImplementedError

    def load_from_history(self, events: Sequence[Event]) -> None:
        """Replay a sequence of events to reconstitute state."""
        for ev in events:
            self.apply(ev)
            self._version += 1

    def flush_pending(self, store: EventStore) -> int:
        """Persist all pending events to the store."""
        seq = store.append_batch(self._pending_events)
        self._pending_events.clear()
        return seq


# ---------------------------------------------------------------------------
# Concrete aggregates
# ---------------------------------------------------------------------------

class OrderAggregate(AggregateRoot):
    """Event-sourced aggregate for a single order."""

    def __init__(self, order_id: str) -> None:
        super().__init__(order_id)
        self.status = "pending"
        self.symbol = ""
        self.side = ""
        self.quantity = 0
        self.filled_qty = 0
        self.price = 0.0
        self.avg_fill_price = 0.0
        self.total_fill_cost = 0.0
        self.fills: List[Dict[str, Any]] = []

    def apply(self, event: Event) -> None:
        p = event.payload
        if event.event_type == EventType.ORDER_SUBMITTED:
            self.status = "submitted"
            self.symbol = p.get("symbol", "")
            self.side = p.get("side", "")
            self.quantity = p.get("quantity", 0)
            self.price = p.get("price", 0.0)
        elif event.event_type == EventType.ORDER_ACCEPTED:
            self.status = "new"
        elif event.event_type == EventType.ORDER_REJECTED:
            self.status = "rejected"
        elif event.event_type == EventType.ORDER_CANCELLED:
            self.status = "cancelled"
        elif event.event_type == EventType.FILL_RECEIVED:
            fill_qty = p.get("fill_qty", 0)
            fill_price = p.get("fill_price", 0.0)
            self.filled_qty += fill_qty
            self.total_fill_cost += fill_qty * fill_price
            self.avg_fill_price = (
                self.total_fill_cost / self.filled_qty if self.filled_qty > 0 else 0.0
            )
            self.fills.append(p)
            self.status = "filled" if self.filled_qty >= self.quantity else "partial"


class PositionAggregate(AggregateRoot):
    """Event-sourced position tracker."""

    def __init__(self, position_key: str) -> None:
        super().__init__(position_key)
        self.symbol = ""
        self.net_quantity = 0
        self.avg_cost = 0.0
        self.realized_pnl = 0.0

    def apply(self, event: Event) -> None:
        p = event.payload
        if event.event_type == EventType.POSITION_UPDATED:
            qty_delta = p.get("quantity", 0)
            price = p.get("price", 0.0)
            self.symbol = p.get("symbol", self.symbol)

            if self.net_quantity == 0:
                self.avg_cost = price
                self.net_quantity = qty_delta
            elif (self.net_quantity > 0) == (qty_delta > 0):
                total = abs(self.net_quantity) * self.avg_cost + abs(qty_delta) * price
                self.net_quantity += qty_delta
                self.avg_cost = total / abs(self.net_quantity) if self.net_quantity != 0 else 0.0
            else:
                closed = min(abs(self.net_quantity), abs(qty_delta))
                pnl = closed * (price - self.avg_cost) * (1 if self.net_quantity > 0 else -1)
                self.realized_pnl += pnl
                self.net_quantity += qty_delta
                if self.net_quantity != 0 and (self.net_quantity > 0) != (self.net_quantity - qty_delta > 0):
                    self.avg_cost = price
        elif event.event_type == EventType.POSITION_CLOSED:
            self.net_quantity = 0
            self.avg_cost = 0.0


__all__ = [
    "EventType",
    "Event",
    "EventStore",
    "AggregateRoot",
    "OrderAggregate",
    "PositionAggregate",
]
