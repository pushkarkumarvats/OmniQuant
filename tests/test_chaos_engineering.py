"""
Institutional-Grade Test Suite

End-to-end tests exercising the PCAP replay engine, chaos engineering
framework, and property-based order-stream invariants.
"""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
from unittest.mock import MagicMock

from tests.test_infrastructure import (
    ChaosScenarioRunner,
    DeterministicReplayer,
    FaultConfig,
    FaultInjector,
    FaultType,
    OrderStreamGenerator,
    PcapPacket,
    PcapReader,
    PcapWriter,
    ReplayResult,
)


# =========================================================================
# PCAP Replay Tests
# =========================================================================

class TestPcapReadWrite(unittest.TestCase):
    """Round-trip serialisation/deserialisation of PCAP files."""

    def _write_packets(
        self, path: str, packets: list[PcapPacket]
    ) -> None:
        with PcapWriter(path) as w:
            for pkt in packets:
                w.write(pkt)

    def test_roundtrip(self):
        packets = [
            PcapPacket(timestamp_ns=1_000_000, protocol="itch", data=b"\x41AAPL"),
            PcapPacket(timestamp_ns=2_000_000, protocol="fix", data=b"8=FIX.4.4"),
            PcapPacket(timestamp_ns=3_000_000, protocol="ouch", data=b"\x01\x02"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            path = f.name

        try:
            self._write_packets(path, packets)
            reader = PcapReader(path)
            n = reader.load()
            self.assertEqual(n, 3)

            loaded = list(reader)
            for orig, loaded_pkt in zip(packets, loaded):
                self.assertEqual(orig.timestamp_ns, loaded_pkt.timestamp_ns)
                self.assertEqual(orig.protocol, loaded_pkt.protocol)
                self.assertEqual(orig.data, loaded_pkt.data)
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            path = f.name
        try:
            reader = PcapReader(path)
            n = reader.load()
            self.assertEqual(n, 0)
        finally:
            os.unlink(path)

    def test_time_range(self):
        packets = [
            PcapPacket(timestamp_ns=100, protocol="raw", data=b"\x00"),
            PcapPacket(timestamp_ns=999, protocol="raw", data=b"\xFF"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            path = f.name
        try:
            self._write_packets(path, packets)
            reader = PcapReader(path)
            reader.load()
            self.assertEqual(reader.time_range_ns(), (100, 999))
        finally:
            os.unlink(path)

    def test_protocol_filter(self):
        packets = [
            PcapPacket(timestamp_ns=1, protocol="itch", data=b"a"),
            PcapPacket(timestamp_ns=2, protocol="fix", data=b"b"),
            PcapPacket(timestamp_ns=3, protocol="itch", data=b"c"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            path = f.name
        try:
            self._write_packets(path, packets)
            reader = PcapReader(path)
            reader.load()
            itch = reader.packets_by_protocol("itch")
            self.assertEqual(len(itch), 2)
        finally:
            os.unlink(path)


class TestDeterministicReplay(unittest.TestCase):
    """Ensure replays are bit-exact across runs."""

    def _make_reader(self, n: int = 50) -> PcapReader:
        path = tempfile.mktemp(suffix=".pcap")
        with PcapWriter(path) as w:
            for i in range(n):
                w.write(PcapPacket(
                    timestamp_ns=i * 1000,
                    protocol="itch",
                    data=struct.pack("<I", i),
                ))
        reader = PcapReader(path)
        reader.load()
        self._tmp_path = path
        return reader

    def tearDown(self):
        if hasattr(self, "_tmp_path") and os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)

    def test_deterministic_checksum(self):
        reader = self._make_reader(100)
        handler = lambda pkt: int.from_bytes(pkt.data[:4], "little")

        r1 = DeterministicReplayer(handler).replay(reader)
        r2 = DeterministicReplayer(handler).replay(reader)

        self.assertEqual(r1.checksum, r2.checksum)
        self.assertEqual(r1.packets_processed, 100)

    def test_verify_determinism_pass(self):
        reader = self._make_reader(20)
        handler = lambda pkt: pkt.data.hex()

        r1 = DeterministicReplayer(handler).replay(reader)

        replayer2 = DeterministicReplayer(handler)
        self.assertTrue(replayer2.verify_determinism(reader, r1.checksum))

    def test_verify_determinism_fail(self):
        reader = self._make_reader(20)
        handler = lambda pkt: pkt.data.hex()

        replayer = DeterministicReplayer(handler)
        self.assertFalse(replayer.verify_determinism(reader, "bad_checksum"))

    def test_throughput_metric(self):
        reader = self._make_reader(1000)
        handler = lambda pkt: None
        result = DeterministicReplayer(handler).replay(reader)
        self.assertGreater(result.throughput_pps, 0)


# =========================================================================
# Chaos Engineering Tests
# =========================================================================

class TestFaultInjector(unittest.TestCase):
    """Unit tests for the FaultInjector."""

    def setUp(self):
        self.inj = FaultInjector()

    def test_inject_and_remove(self):
        self.inj.inject("f1", FaultConfig(fault_type=FaultType.NETWORK_PARTITION))
        self.assertTrue(self.inj.is_active(FaultType.NETWORK_PARTITION))

        self.inj.remove("f1")
        self.assertFalse(self.inj.is_active(FaultType.NETWORK_PARTITION))

    def test_network_partition_drops(self):
        self.inj.inject("p", FaultConfig(fault_type=FaultType.NETWORK_PARTITION))
        _, dropped = self.inj.apply_network_effects(b"hello")
        self.assertTrue(dropped)

    def test_data_corruption(self):
        self.inj.inject("c", FaultConfig(
            fault_type=FaultType.DATA_CORRUPTION,
            corruption_rate=1.0,
            probability=1.0,
        ))
        original = bytes(range(256))
        corrupted, dropped = self.inj.apply_network_effects(original)
        self.assertFalse(dropped)
        self.assertNotEqual(corrupted, original)

    def test_clock_skew(self):
        self.inj.inject("clk", FaultConfig(
            fault_type=FaultType.CLOCK_SKEW, clock_offset_ms=100.0
        ))
        offset = self.inj.get_clock_offset_ns()
        self.assertEqual(offset, 100_000_000)

    def test_rejection_storm(self):
        self.inj.inject("rej", FaultConfig(
            fault_type=FaultType.ORDER_REJECTION_STORM,
            rejection_rate=1.0,
        ))
        self.assertTrue(self.inj.should_reject_order())

    def test_fault_context_manager(self):
        with self.inj.fault_context(FaultType.LATENCY_SPIKE, latency_ms=10.0) as fid:
            self.assertTrue(self.inj.is_active(FaultType.LATENCY_SPIKE))
        self.assertFalse(self.inj.is_active(FaultType.LATENCY_SPIKE))

    def test_fault_log(self):
        self.inj.inject("x", FaultConfig(fault_type=FaultType.FEED_STALE))
        self.inj.remove("x")
        log = self.inj.fault_log
        self.assertEqual(len(log), 2)
        self.assertEqual(log[0]["action"], "inject")
        self.assertEqual(log[1]["action"], "remove")

    def test_clear_all(self):
        self.inj.inject("a", FaultConfig(fault_type=FaultType.NETWORK_PARTITION))
        self.inj.inject("b", FaultConfig(fault_type=FaultType.LATENCY_SPIKE))
        self.inj.clear_all()
        self.assertFalse(self.inj.is_active(FaultType.NETWORK_PARTITION))
        self.assertFalse(self.inj.is_active(FaultType.LATENCY_SPIKE))

    def test_packet_loss(self):
        self.inj.inject("loss", FaultConfig(
            fault_type=FaultType.PACKET_LOSS, packet_loss_rate=1.0
        ))
        _, dropped = self.inj.apply_network_effects(b"data")
        self.assertTrue(dropped)

    def test_no_faults_passthrough(self):
        data, dropped = self.inj.apply_network_effects(b"clean")
        self.assertFalse(dropped)
        self.assertEqual(data, b"clean")


class TestChaosScenarioRunner(unittest.TestCase):
    """Test the pre-built chaos scenarios."""

    def setUp(self):
        self.runner = ChaosScenarioRunner()

    def test_exchange_outage_pass(self):
        handler = MagicMock(side_effect=[
            {"handled_gracefully": True},
            {"operational": True},
        ])
        result = self.runner.scenario_exchange_outage(handler)
        self.assertTrue(result["passed"])

    def test_exchange_outage_fail(self):
        handler = MagicMock(side_effect=[
            {"handled_gracefully": False},
            {"operational": True},
        ])
        result = self.runner.scenario_exchange_outage(handler)
        self.assertFalse(result["passed"])

    def test_rejection_storm(self):
        handler = MagicMock(return_value={"state_consistent": True})
        result = self.runner.scenario_rejection_storm(handler)
        self.assertTrue(result["passed"])

    def test_results_summary(self):
        handler_ok = MagicMock(return_value={"state_consistent": True})
        self.runner.scenario_rejection_storm(handler_ok)

        handler_fail = MagicMock(side_effect=[
            {"handled_gracefully": False},
            {"operational": False},
        ])
        self.runner.scenario_exchange_outage(handler_fail)

        summary = self.runner.results_summary
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_scenario_handles_exception(self):
        def boom(inj):
            raise RuntimeError("unexpected crash")
        result = self.runner.run_scenario("crash_test", boom)
        self.assertFalse(result["passed"])
        self.assertIn("unexpected crash", result["error"])


# =========================================================================
# Property-Based Order Stream Tests
# =========================================================================

class TestOrderStreamGenerator(unittest.TestCase):
    """Verify invariants of generated order streams."""

    def test_valid_stream_structure(self):
        gen = OrderStreamGenerator(seed=123)
        stream = gen.generate_valid_stream(500)
        self.assertGreaterEqual(len(stream), 450)

        for event in stream:
            self.assertIn("action", event)
            if event["action"] == "new":
                self.assertIn("order_id", event)
                self.assertIn("symbol", event)
                self.assertIn("price", event)
                self.assertIn("quantity", event)

    def test_valid_stream_no_cancel_without_order(self):
        gen = OrderStreamGenerator(seed=42)
        stream = gen.generate_valid_stream(1000)
        active = set()
        for event in stream:
            if event["action"] == "new":
                active.add(event["order_id"])
            elif event["action"] == "cancel":
                self.assertIn(
                    event["order_id"], active,
                    "Cancel for non-existent order in valid stream"
                )
                active.discard(event["order_id"])

    def test_adversarial_stream_has_edge_cases(self):
        gen = OrderStreamGenerator(seed=99)
        stream = gen.generate_adversarial_stream(200)

        # Should contain zero-price, zero-quantity, and nonexistent cancel
        has_zero_price = any(
            e.get("price") == 0.0 for e in stream
        )
        has_nonexistent_cancel = any(
            e.get("action") == "cancel"
            and e.get("order_id") == "NONEXISTENT_ORDER"
            for e in stream
        )
        has_duplicate_id = False
        seen = set()
        for e in stream:
            if e.get("action") == "new":
                oid = e["order_id"]
                if oid in seen:
                    has_duplicate_id = True
                seen.add(oid)

        self.assertTrue(has_zero_price, "Missing zero-price edge case")
        self.assertTrue(has_nonexistent_cancel, "Missing nonexistent cancel")
        self.assertTrue(has_duplicate_id, "Missing duplicate order ID")

    def test_deterministic_generation(self):
        s1 = OrderStreamGenerator(seed=7).generate_valid_stream(100)
        s2 = OrderStreamGenerator(seed=7).generate_valid_stream(100)
        self.assertEqual(s1, s2)


# =========================================================================
# Integration: Replay through Chaos
# =========================================================================

class TestReplayWithChaos(unittest.TestCase):
    """Combine PCAP replay with fault injection for integration testing."""

    def test_replay_with_packet_loss(self):
        """Verify system handles packet loss during replay."""
        inj = FaultInjector()
        received: list[bytes] = []

        def handler(pkt: PcapPacket):
            data, dropped = inj.apply_network_effects(pkt.data)
            if not dropped:
                received.append(data)
            return not dropped

        # Write a small PCAP
        path = tempfile.mktemp(suffix=".pcap")
        with PcapWriter(path) as w:
            for i in range(100):
                w.write(PcapPacket(
                    timestamp_ns=i * 1000,
                    protocol="raw",
                    data=struct.pack("<I", i),
                ))

        reader = PcapReader(path)
        reader.load()

        # 30% packet loss
        inj.inject("loss", FaultConfig(
            fault_type=FaultType.PACKET_LOSS, packet_loss_rate=0.3
        ))

        replayer = DeterministicReplayer(handler)
        result = replayer.replay(reader)

        self.assertEqual(result.packets_processed, 100)
        # With 30% loss, expect roughly 70 received (±20)
        self.assertGreater(len(received), 40)
        self.assertLess(len(received), 100)

        os.unlink(path)

    def test_replay_with_corruption(self):
        """Verify corrupted packets are detectable."""
        inj = FaultInjector()
        clean_count = 0
        corrupt_count = 0

        original_data = b"\xDE\xAD\xBE\xEF" * 10

        def handler(pkt: PcapPacket):
            nonlocal clean_count, corrupt_count
            data, _ = inj.apply_network_effects(pkt.data)
            if data == original_data:
                clean_count += 1
            else:
                corrupt_count += 1
            return data

        path = tempfile.mktemp(suffix=".pcap")
        with PcapWriter(path) as w:
            for i in range(200):
                w.write(PcapPacket(
                    timestamp_ns=i, protocol="raw", data=original_data
                ))

        reader = PcapReader(path)
        reader.load()

        inj.inject("corrupt", FaultConfig(
            fault_type=FaultType.DATA_CORRUPTION,
            corruption_rate=0.5,
            probability=1.0,
        ))

        DeterministicReplayer(handler).replay(reader)

        # Most packets should be corrupted at 50% byte-flip rate
        self.assertGreater(corrupt_count, 100)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
