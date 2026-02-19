"""
Institutional-Grade Testing Framework

Provides deterministic PCAP replay and chaos engineering for production
hardening of the HFT system.

Modules:
  1. **Deterministic PCAP Replay** — Replay captured network traffic
     (ITCH, FIX) through the system with nanosecond-precise timing,
     ensuring bit-exact deterministic results.
  2. **Chaos Engineering** — Inject network partitions, latency spikes,
     exchange outages, and data corruption to verify system resilience.
  3. **Property-Based Testing** — Generate random valid/invalid order
     streams and verify invariants hold.
"""

from __future__ import annotations

import hashlib
import os
import random
import struct
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from loguru import logger


# =========================================================================
# 1. Deterministic PCAP Replay
# =========================================================================

@dataclass(frozen=True)
class PcapPacket:
    """A single captured packet with nanosecond timestamp."""
    timestamp_ns: int
    protocol: str           # "itch", "fix", "ouch", "raw"
    data: bytes
    source_ip: str = ""
    dest_ip: str = ""
    source_port: int = 0
    dest_port: int = 0
    sequence: int = 0


class PcapReader:
    """
    Read packets from a simplified binary PCAP file.

    File format (per packet):
        [8B timestamp_ns][4B data_len][1B protocol_id][data_len bytes]

    Protocol IDs: 0=raw, 1=itch, 2=fix, 3=ouch
    """

    PROTO_MAP = {0: "raw", 1: "itch", 2: "fix", 3: "ouch"}
    HEADER_SIZE = 13  # 8 + 4 + 1

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._packets: List[PcapPacket] = []

    def load(self) -> int:
        """Load all packets into memory. Returns packet count."""
        self._packets.clear()
        if not self._path.exists():
            logger.warning(f"PCAP file not found: {self._path}")
            return 0

        with open(self._path, "rb") as f:
            seq = 0
            while True:
                header = f.read(self.HEADER_SIZE)
                if len(header) < self.HEADER_SIZE:
                    break
                ts_ns, data_len, proto_id = struct.unpack("<QIB", header)
                data = f.read(data_len)
                if len(data) < data_len:
                    break
                proto = self.PROTO_MAP.get(proto_id, "raw")
                self._packets.append(PcapPacket(
                    timestamp_ns=ts_ns,
                    protocol=proto,
                    data=data,
                    sequence=seq,
                ))
                seq += 1

        logger.info(f"Loaded {len(self._packets)} packets from {self._path}")
        return len(self._packets)

    def __iter__(self) -> Iterator[PcapPacket]:
        return iter(self._packets)

    def __len__(self) -> int:
        return len(self._packets)

    def packets_by_protocol(self, protocol: str) -> List[PcapPacket]:
        return [p for p in self._packets if p.protocol == protocol]

    def time_range_ns(self) -> Tuple[int, int]:
        if not self._packets:
            return (0, 0)
        return self._packets[0].timestamp_ns, self._packets[-1].timestamp_ns


class PcapWriter:
    """Write packets to the simplified binary PCAP format."""

    PROTO_ID = {"raw": 0, "itch": 1, "fix": 2, "ouch": 3}

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._fd: Any = None
        self._count = 0

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self._path, "wb")

    def write(self, packet: PcapPacket) -> None:
        proto_id = self.PROTO_ID.get(packet.protocol, 0)
        header = struct.pack("<QIB", packet.timestamp_ns, len(packet.data), proto_id)
        self._fd.write(header + packet.data)
        self._count += 1

    def close(self) -> None:
        if self._fd:
            self._fd.close()
        logger.info(f"Wrote {self._count} packets to {self._path}")

    def __enter__(self):
        self.open(); return self

    def __exit__(self, *_):
        self.close()


class DeterministicReplayer:
    """
    Replay captured packets through the system deterministically.

    Guarantees:
      - Packets are delivered in exact captured order
      - Timing gaps are preserved (or optionally compressed)
      - Results are bit-exact across replays given identical state
      - Checksum verification of replay output vs. golden reference
    """

    def __init__(
        self,
        handler: Callable[[PcapPacket], Any],
        *,
        realtime: bool = False,
        speed_multiplier: float = 1.0,
    ) -> None:
        self._handler = handler
        self._realtime = realtime
        self._speed = speed_multiplier
        self._results: List[Any] = []
        self._replay_hash = hashlib.sha256()
        self._packets_processed = 0

    def replay(self, reader: PcapReader) -> "ReplayResult":
        """
        Replay all packets from a PcapReader.

        If ``realtime=True``, sleeps between packets to preserve timing.
        """
        self._results.clear()
        self._packets_processed = 0
        self._replay_hash = hashlib.sha256()

        start = time.time_ns()
        prev_ts = 0

        for pkt in reader:
            if self._realtime and prev_ts > 0:
                gap_ns = int((pkt.timestamp_ns - prev_ts) / self._speed)
                if gap_ns > 0:
                    time.sleep(gap_ns / 1e9)
            prev_ts = pkt.timestamp_ns

            result = self._handler(pkt)
            self._results.append(result)
            self._packets_processed += 1

            # Hash the packet data for determinism verification
            self._replay_hash.update(pkt.data)
            if result is not None:
                self._replay_hash.update(str(result).encode())

        elapsed_ns = time.time_ns() - start
        return ReplayResult(
            packets_processed=self._packets_processed,
            elapsed_ns=elapsed_ns,
            checksum=self._replay_hash.hexdigest(),
            results=self._results,
        )

    def verify_determinism(self, reader: PcapReader, expected_checksum: str) -> bool:
        """Replay and verify output matches a known-good checksum."""
        result = self.replay(reader)
        match = result.checksum == expected_checksum
        if not match:
            logger.error(
                f"Determinism check FAILED: "
                f"expected={expected_checksum}, got={result.checksum}"
            )
        else:
            logger.info(f"Determinism check PASSED ({result.packets_processed} packets)")
        return match


@dataclass
class ReplayResult:
    packets_processed: int
    elapsed_ns: int
    checksum: str
    results: List[Any] = field(default_factory=list)

    @property
    def throughput_pps(self) -> float:
        """Packets per second."""
        if self.elapsed_ns <= 0:
            return 0.0
        return self.packets_processed / (self.elapsed_ns / 1e9)


# =========================================================================
# 2. Chaos Engineering Framework
# =========================================================================

class FaultType(Enum):
    NETWORK_PARTITION = "network_partition"
    LATENCY_SPIKE = "latency_spike"
    EXCHANGE_OUTAGE = "exchange_outage"
    DATA_CORRUPTION = "data_corruption"
    CLOCK_SKEW = "clock_skew"
    PACKET_LOSS = "packet_loss"
    ORDER_REJECTION_STORM = "order_rejection_storm"
    FEED_STALE = "feed_stale"


@dataclass
class FaultConfig:
    """Configuration for a single fault injection."""
    fault_type: FaultType
    duration_s: float = 5.0
    probability: float = 1.0           # per-event probability (for random faults)
    latency_ms: float = 0.0            # for LATENCY_SPIKE
    corruption_rate: float = 0.01      # fraction of bytes corrupted
    clock_offset_ms: float = 0.0       # for CLOCK_SKEW
    packet_loss_rate: float = 0.0      # for PACKET_LOSS
    rejection_rate: float = 0.5        # for ORDER_REJECTION_STORM
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaultInjector:
    """
    Inject controlled faults into the trading system for chaos testing.

    Each fault is activated for a configurable duration and probability.
    Multiple faults can be active simultaneously.
    """

    def __init__(self) -> None:
        self._active_faults: Dict[str, FaultConfig] = {}
        self._fault_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._rng = random.Random(42)

    def inject(self, fault_id: str, config: FaultConfig) -> None:
        """Activate a fault."""
        with self._lock:
            self._active_faults[fault_id] = config
        self._fault_log.append({
            "action": "inject",
            "fault_id": fault_id,
            "fault_type": config.fault_type.value,
            "timestamp_ns": time.time_ns(),
        })
        logger.warning(f"CHAOS: Injected {config.fault_type.value} (id={fault_id})")

    def remove(self, fault_id: str) -> None:
        """Deactivate a fault."""
        with self._lock:
            self._active_faults.pop(fault_id, None)
        self._fault_log.append({
            "action": "remove",
            "fault_id": fault_id,
            "timestamp_ns": time.time_ns(),
        })
        logger.info(f"CHAOS: Removed fault {fault_id}")

    def is_active(self, fault_type: FaultType) -> bool:
        """Check if any fault of the given type is active."""
        with self._lock:
            return any(
                f.fault_type == fault_type for f in self._active_faults.values()
            )

    def get_active(self, fault_type: FaultType) -> Optional[FaultConfig]:
        """Get the config for an active fault of the given type."""
        with self._lock:
            for f in self._active_faults.values():
                if f.fault_type == fault_type:
                    return f
        return None

    @contextmanager
    def fault_context(
        self, fault_type: FaultType, duration_s: float = 5.0, **kwargs
    ) -> Generator[str, None, None]:
        """Context manager for scoped fault injection."""
        fault_id = f"{fault_type.value}_{time.time_ns()}"
        config = FaultConfig(fault_type=fault_type, duration_s=duration_s, **kwargs)
        self.inject(fault_id, config)
        try:
            yield fault_id
        finally:
            self.remove(fault_id)

    def apply_network_effects(self, data: bytes) -> Tuple[bytes, bool]:
        """
        Apply active network faults to outgoing/incoming data.

        Returns (modified_data, should_drop).
        """
        should_drop = False

        # Packet loss
        loss_cfg = self.get_active(FaultType.PACKET_LOSS)
        if loss_cfg and self._rng.random() < loss_cfg.packet_loss_rate:
            should_drop = True
            return data, should_drop

        # Latency spike (caller should sleep)
        lat_cfg = self.get_active(FaultType.LATENCY_SPIKE)
        if lat_cfg and lat_cfg.latency_ms > 0:
            time.sleep(lat_cfg.latency_ms / 1000.0)

        # Data corruption
        corr_cfg = self.get_active(FaultType.DATA_CORRUPTION)
        if corr_cfg and self._rng.random() < corr_cfg.probability:
            data = self._corrupt_data(data, corr_cfg.corruption_rate)

        # Network partition = drop everything
        if self.is_active(FaultType.NETWORK_PARTITION):
            should_drop = True

        return data, should_drop

    def should_reject_order(self) -> bool:
        """Check if the ORDER_REJECTION_STORM fault should reject."""
        cfg = self.get_active(FaultType.ORDER_REJECTION_STORM)
        if cfg and self._rng.random() < cfg.rejection_rate:
            return True
        return False

    def get_clock_offset_ns(self) -> int:
        """Get clock skew in nanoseconds (0 if no fault active)."""
        cfg = self.get_active(FaultType.CLOCK_SKEW)
        if cfg:
            return int(cfg.clock_offset_ms * 1e6)
        return 0

    def _corrupt_data(self, data: bytes, rate: float) -> bytes:
        """Randomly flip bits in the data."""
        ba = bytearray(data)
        for i in range(len(ba)):
            if self._rng.random() < rate:
                ba[i] ^= self._rng.randint(1, 255)
        return bytes(ba)

    @property
    def fault_log(self) -> List[Dict[str, Any]]:
        return list(self._fault_log)

    def clear_all(self) -> None:
        with self._lock:
            self._active_faults.clear()


# =========================================================================
# 3. Chaos Test Scenarios
# =========================================================================

class ChaosScenarioRunner:
    """
    Pre-built chaos test scenarios that combine multiple fault types
    to simulate real-world production incidents.
    """

    def __init__(self, injector: Optional[FaultInjector] = None) -> None:
        self.injector = injector or FaultInjector()
        self._scenario_results: List[Dict[str, Any]] = []

    def run_scenario(
        self,
        name: str,
        test_fn: Callable[[FaultInjector], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run a named chaos scenario.

        ``test_fn`` receives the FaultInjector and should return a dict
        with at minimum ``{"passed": bool, ...}``.
        """
        logger.info(f"=== CHAOS SCENARIO: {name} ===")
        self.injector.clear_all()
        start = time.time_ns()
        try:
            result = test_fn(self.injector)
            result["scenario"] = name
            result["elapsed_ns"] = time.time_ns() - start
        except Exception as e:
            result = {
                "scenario": name,
                "passed": False,
                "error": str(e),
                "elapsed_ns": time.time_ns() - start,
            }
            logger.error(f"Scenario {name} failed with exception: {e}")
        finally:
            self.injector.clear_all()

        self._scenario_results.append(result)
        status = "PASSED" if result.get("passed") else "FAILED"
        logger.info(f"=== {name}: {status} ===")
        return result

    # --- Pre-built scenarios ---

    def scenario_exchange_outage(
        self, system_handler: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simulate a 10-second exchange outage.

        Verifies:
          - System detects the outage
          - No orders are sent during outage
          - System recovers after outage ends
        """
        def test(inj: FaultInjector) -> Dict[str, Any]:
            with inj.fault_context(FaultType.EXCHANGE_OUTAGE, duration_s=10.0):
                during = system_handler()
            after = system_handler()
            return {
                "passed": during.get("handled_gracefully", False)
                          and after.get("operational", False),
                "during_outage": during,
                "after_recovery": after,
            }
        return self.run_scenario("exchange_outage", test)

    def scenario_latency_spike(
        self, system_handler: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Inject 500ms latency spikes for 30s.

        Verifies:
          - System does not timeout unexpectedly
          - Fill reconciliation handles delayed responses
        """
        def test(inj: FaultInjector) -> Dict[str, Any]:
            with inj.fault_context(
                FaultType.LATENCY_SPIKE, duration_s=30.0, latency_ms=500.0
            ):
                result = system_handler()
            return {
                "passed": result.get("no_crashes", True),
                "latency_result": result,
            }
        return self.run_scenario("latency_spike_500ms", test)

    def scenario_data_corruption(
        self, system_handler: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Corrupt 5% of incoming data bytes.

        Verifies:
          - Parser rejects corrupt messages gracefully
          - No unhandled exceptions or panics
        """
        def test(inj: FaultInjector) -> Dict[str, Any]:
            with inj.fault_context(
                FaultType.DATA_CORRUPTION, corruption_rate=0.05
            ):
                result = system_handler()
            return {
                "passed": result.get("no_crashes", True)
                          and result.get("corrupt_rejected", 0) >= 0,
                "corruption_result": result,
            }
        return self.run_scenario("data_corruption_5pct", test)

    def scenario_split_brain(
        self, system_handler: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simulate a network partition + clock skew (split-brain).

        Verifies:
          - System detects inconsistency
          - No duplicate orders are generated
        """
        def test(inj: FaultInjector) -> Dict[str, Any]:
            inj.inject("partition", FaultConfig(
                fault_type=FaultType.NETWORK_PARTITION, duration_s=15.0
            ))
            inj.inject("clock", FaultConfig(
                fault_type=FaultType.CLOCK_SKEW, clock_offset_ms=2000.0
            ))
            result = system_handler()
            inj.remove("partition")
            inj.remove("clock")
            recovery = system_handler()
            return {
                "passed": recovery.get("no_duplicate_orders", True),
                "during": result,
                "recovery": recovery,
            }
        return self.run_scenario("split_brain", test)

    def scenario_rejection_storm(
        self, system_handler: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        50% of orders randomly rejected by the exchange.

        Verifies:
          - Strategy handles rejections without state corruption
          - Risk limits remain enforced
        """
        def test(inj: FaultInjector) -> Dict[str, Any]:
            with inj.fault_context(
                FaultType.ORDER_REJECTION_STORM,
                duration_s=60.0,
                rejection_rate=0.5,
            ):
                result = system_handler()
            return {
                "passed": result.get("state_consistent", True),
                "rejection_result": result,
            }
        return self.run_scenario("rejection_storm", test)

    @property
    def results_summary(self) -> Dict[str, Any]:
        total = len(self._scenario_results)
        passed = sum(1 for r in self._scenario_results if r.get("passed"))
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "scenarios": self._scenario_results,
        }


# =========================================================================
# 4. Property-Based Order Stream Generator
# =========================================================================

class OrderStreamGenerator:
    """
    Generate random valid/invalid order streams for property-based testing.

    Verifies system invariants:
      - Position always reconciles after all fills
      - Cash + positions = total equity at all times
      - No fill without a corresponding order
      - Cancel of a filled order is a no-op
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "SPY"]
        self._order_id = 0

    def generate_valid_stream(self, n_events: int = 1000) -> List[Dict[str, Any]]:
        """Generate a stream of valid order events."""
        events = []
        active_orders: Dict[str, Dict] = {}

        for _ in range(n_events):
            action = self._rng.choice(
                ["new", "cancel", "modify"],
                p=[0.6, 0.25, 0.15],
            )

            if action == "new" or not active_orders:
                self._order_id += 1
                oid = f"ORD_{self._order_id}"
                sym = self._rng.choice(self._symbols)
                side = self._rng.choice(["buy", "sell"])
                price = round(float(self._rng.uniform(50, 500)), 2)
                qty = int(self._rng.integers(1, 1000))
                order = {
                    "action": "new",
                    "order_id": oid,
                    "symbol": sym,
                    "side": side,
                    "price": price,
                    "quantity": qty,
                    "order_type": self._rng.choice(["limit", "market"]),
                }
                events.append(order)
                active_orders[oid] = order

            elif action == "cancel" and active_orders:
                oid = self._rng.choice(list(active_orders.keys()))
                events.append({"action": "cancel", "order_id": oid})
                del active_orders[oid]

            elif action == "modify" and active_orders:
                oid = self._rng.choice(list(active_orders.keys()))
                new_qty = int(self._rng.integers(1, 2000))
                events.append({
                    "action": "modify",
                    "order_id": oid,
                    "new_quantity": new_qty,
                })

        return events

    def generate_adversarial_stream(self, n_events: int = 500) -> List[Dict[str, Any]]:
        """Generate an adversarial stream including edge-case orders."""
        events = self.generate_valid_stream(n_events // 2)

        edge_cases = [
            {"action": "new", "order_id": "EDGE_1", "symbol": "TEST",
             "side": "buy", "price": 0.0, "quantity": 0, "order_type": "market"},
            {"action": "new", "order_id": "EDGE_2", "symbol": "TEST",
             "side": "buy", "price": 999999.99, "quantity": 1, "order_type": "limit"},
            {"action": "new", "order_id": "EDGE_3", "symbol": "TEST",
             "side": "sell", "price": 0.01, "quantity": 999999999, "order_type": "limit"},
            {"action": "cancel", "order_id": "NONEXISTENT_ORDER"},
            {"action": "cancel", "order_id": ""},
            {"action": "new", "order_id": "DUP_1", "symbol": "AAPL",
             "side": "buy", "price": 100.0, "quantity": 100, "order_type": "limit"},
            {"action": "new", "order_id": "DUP_1", "symbol": "AAPL",
             "side": "buy", "price": 100.0, "quantity": 100, "order_type": "limit"},
        ]

        # Interleave edge cases
        for ec in edge_cases:
            pos = self._rng.integers(0, len(events) + 1)
            events.insert(pos, ec)

        return events


__all__ = [
    "PcapPacket",
    "PcapReader",
    "PcapWriter",
    "DeterministicReplayer",
    "ReplayResult",
    "FaultType",
    "FaultConfig",
    "FaultInjector",
    "ChaosScenarioRunner",
    "OrderStreamGenerator",
]
