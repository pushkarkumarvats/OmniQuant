"""
Native OMS Bridge - Python FFI bindings to the C++20/Rust Order Management System.

This bridge loads the compiled native OMS shared library and exposes a Pythonic API.
The native OMS guarantees:
  - Deterministic sub-microsecond order state transitions
  - Lock-free concurrent order book management
  - Zero-copy memory-mapped order journals
  - Kernel-bypass network I/O (DPDK/io_uring)

Build the native library:
  C++: cmake --build build/ --target oms_core
  Rust: cargo build --release -p oms-core

The shared library is expected at:
  Linux:   lib/liboms_core.so
  macOS:   lib/liboms_core.dylib
  Windows: lib/oms_core.dll
"""

from __future__ import annotations

import ctypes
import enum
import os
import platform
import struct
import time
from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_double,
    c_int,
    c_int64,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Order / Fill data structures (mirroring the native C++ structs)
# ---------------------------------------------------------------------------


class NativeOrderStatus(enum.IntEnum):
    PENDING_NEW = 0
    NEW = 1
    PARTIALLY_FILLED = 2
    FILLED = 3
    PENDING_CANCEL = 4
    CANCELLED = 5
    REJECTED = 6
    EXPIRED = 7


class NativeOrderSide(enum.IntEnum):
    BUY = 0
    SELL = 1


class NativeOrderType(enum.IntEnum):
    MARKET = 0
    LIMIT = 1
    STOP = 2
    STOP_LIMIT = 3
    IOC = 4       # Immediate-or-Cancel
    FOK = 5       # Fill-or-Kill
    GTC = 6       # Good-til-Cancel
    ICEBERG = 7
    PEG = 8
    TRAILING_STOP = 9


class NativeTimeInForce(enum.IntEnum):
    DAY = 0
    GTC = 1
    IOC = 2
    FOK = 3
    GTD = 4       # Good-til-Date
    OPG = 5       # At-the-Opening
    CLS = 6       # At-the-Close


@dataclass(frozen=True)
class NativeOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: NativeOrderSide
    order_type: NativeOrderType
    quantity: int
    price: float
    stop_price: float = 0.0
    time_in_force: NativeTimeInForce = NativeTimeInForce.DAY
    display_qty: int = 0          # For iceberg orders
    min_qty: int = 0
    account: str = ""
    strategy_id: str = ""
    parent_order_id: str = ""     # For child orders
    timestamp_ns: int = 0         # Nanosecond precision timestamp
    status: NativeOrderStatus = NativeOrderStatus.PENDING_NEW

    def as_ctypes_struct(self) -> "_COrder":
        s = _COrder()
        s.order_id = self.order_id.encode("utf-8")[:64]
        s.client_order_id = self.client_order_id.encode("utf-8")[:64]
        s.symbol = self.symbol.encode("utf-8")[:16]
        s.side = self.side.value
        s.order_type = self.order_type.value
        s.quantity = self.quantity
        s.price = self.price
        s.stop_price = self.stop_price
        s.time_in_force = self.time_in_force.value
        s.display_qty = self.display_qty
        s.min_qty = self.min_qty
        s.timestamp_ns = self.timestamp_ns or int(time.time_ns())
        s.status = self.status.value
        return s


@dataclass(frozen=True)
class NativeFill:
    fill_id: str
    order_id: str
    symbol: str
    side: NativeOrderSide
    fill_qty: int
    fill_price: float
    commission: float
    liquidity_flag: str  # "A" = add, "R" = remove
    exchange: str
    timestamp_ns: int
    leaves_qty: int      # Remaining quantity
    cum_qty: int         # Cumulative filled quantity
    avg_price: float     # Volume-weighted average fill price


@dataclass
class OMSConfig:
    # Network
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 9100
    use_kernel_bypass: bool = False  # DPDK / io_uring
    
    # Order management
    max_open_orders: int = 100_000
    max_orders_per_second: int = 50_000
    order_timeout_ms: int = 5_000
    
    # Risk pre-checks (native-side)
    max_order_value: float = 10_000_000.0
    max_position_notional: float = 50_000_000.0
    max_order_qty: int = 1_000_000
    fat_finger_threshold_pct: float = 5.0  # Reject if price > 5% from mid
    
    # Journaling
    journal_path: str = "/var/log/oms/journal"
    journal_sync: bool = True  # fsync after every write
    
    # Latency monitoring
    latency_histogram_buckets_us: List[int] = field(
        default_factory=lambda: [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]
    )
    
    # Shared memory ring buffer for IPC
    shm_ring_buffer_size: int = 64 * 1024 * 1024  # 64 MB
    shm_name: str = "/oms_ring_buffer"


# ---------------------------------------------------------------------------
# C-compatible struct layouts for FFI
# ---------------------------------------------------------------------------


class _COrder(Structure):
    _fields_ = [
        ("order_id", ctypes.c_char * 64),
        ("client_order_id", ctypes.c_char * 64),
        ("symbol", ctypes.c_char * 16),
        ("side", c_int),
        ("order_type", c_int),
        ("quantity", c_int64),
        ("price", c_double),
        ("stop_price", c_double),
        ("time_in_force", c_int),
        ("display_qty", c_int64),
        ("min_qty", c_int64),
        ("timestamp_ns", c_uint64),
        ("status", c_int),
    ]


class _CFill(Structure):
    _fields_ = [
        ("fill_id", ctypes.c_char * 64),
        ("order_id", ctypes.c_char * 64),
        ("symbol", ctypes.c_char * 16),
        ("side", c_int),
        ("fill_qty", c_int64),
        ("fill_price", c_double),
        ("commission", c_double),
        ("liquidity_flag", ctypes.c_char * 2),
        ("exchange", ctypes.c_char * 16),
        ("timestamp_ns", c_uint64),
        ("leaves_qty", c_int64),
        ("cum_qty", c_int64),
        ("avg_price", c_double),
    ]


class _CBookLevel(Structure):
    _fields_ = [
        ("price", c_double),
        ("quantity", c_int64),
        ("order_count", c_int),
        ("_padding", c_int),
    ]


class _CITCHMessage(Structure):
    _fields_ = [
        ("msg_type", ctypes.c_uint8),
        ("timestamp_ns", c_uint64),
        ("order_ref", c_uint64),
        ("side", ctypes.c_uint8),
        ("shares", c_uint32),
        ("symbol", ctypes.c_char * 8),
        ("price", c_uint32),
        ("match_number", c_uint64),
    ]


class _CFPGARiskResult(Structure):
    _fields_ = [
        ("passed", c_int),
        ("latency_ns", c_uint64),
        ("error_code", c_int),
        ("max_position_ok", c_int),
        ("fat_finger_ok", c_int),
        ("rate_limit_ok", c_int),
    ]


# ---------------------------------------------------------------------------
# Native OMS Bridge
# ---------------------------------------------------------------------------


class NativeOMSBridge:
    """FFI bridge to the native C++/Rust OMS, with a pure-Python fallback."""

    def __init__(self, config: Optional[OMSConfig] = None) -> None:
        self.config = config or OMSConfig()
        self._lib: Optional[ctypes.CDLL] = None
        self._connected = False
        self._order_counter = 0
        self._pending_orders: Dict[str, NativeOrder] = {}
        self._fill_callbacks: List[Callable[[NativeFill], None]] = []
        self._fallback: Optional[Any] = None
        
        # Latency tracking
        self._submit_latencies_ns: List[int] = []
        
        # Try to load native library
        self._load_native_library()

    def _load_native_library(self) -> None:
        system = platform.system()
        lib_name = {
            "Linux": "liboms_core.so",
            "Darwin": "liboms_core.dylib",
            "Windows": "oms_core.dll",
        }.get(system, "liboms_core.so")

        search_paths = [
            Path(__file__).parent / "lib" / lib_name,
            Path("lib") / lib_name,
            Path("/usr/local/lib") / lib_name,
            Path(os.environ.get("OMS_LIB_PATH", "")) / lib_name,
        ]

        for path in search_paths:
            if path.exists():
                try:
                    self._lib = ctypes.CDLL(str(path))
                    self._setup_function_signatures()
                    logger.info(f"Loaded native OMS library from {path}")
                    return
                except OSError as e:
                    logger.warning(f"Failed to load {path}: {e}")

        logger.warning(
            "Native OMS library not found. Using Python reference implementation. "
            "Build the native library for production use."
        )
        from .reference_oms import ReferenceOMS
        self._fallback = ReferenceOMS(self.config)

    def _setup_function_signatures(self) -> None:
        if self._lib is None:
            return

        # oms_init(config_json: *const c_char) -> c_int
        self._lib.oms_init.argtypes = [c_char_p]
        self._lib.oms_init.restype = c_int

        # oms_submit_order(order: *const COrder) -> c_int
        self._lib.oms_submit_order.argtypes = [POINTER(_COrder)]
        self._lib.oms_submit_order.restype = c_int

        # oms_cancel_order(order_id: *const c_char) -> c_int
        self._lib.oms_cancel_order.argtypes = [c_char_p]
        self._lib.oms_cancel_order.restype = c_int

        # oms_get_order_status(order_id: *const c_char, status: *mut c_int) -> c_int
        self._lib.oms_get_order_status.argtypes = [c_char_p, POINTER(c_int)]
        self._lib.oms_get_order_status.restype = c_int

        # oms_poll_fills(fills: *mut CFill, max_count: c_int) -> c_int
        self._lib.oms_poll_fills.argtypes = [POINTER(_CFill), c_int]
        self._lib.oms_poll_fills.restype = c_int

        # oms_get_latency_stats(min_ns: *mut u64, max_ns: *mut u64, avg_ns: *mut u64) -> c_int
        self._lib.oms_get_latency_stats.argtypes = [
            POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64)
        ]
        self._lib.oms_get_latency_stats.restype = c_int

        # oms_shutdown() -> c_int
        self._lib.oms_shutdown.argtypes = []
        self._lib.oms_shutdown.restype = c_int

        # --- v2.0 LOB exports ---

        # oms_feed_market_data(symbol, bid, ask, last, volume) -> c_int
        self._lib.oms_feed_market_data.argtypes = [c_char_p, c_double, c_double, c_double, c_int64]
        self._lib.oms_feed_market_data.restype = c_int

        # oms_get_book_snapshot(symbol, bids, asks, max_levels) -> c_int
        self._lib.oms_get_book_snapshot.argtypes = [
            c_char_p, POINTER(_CBookLevel), POINTER(_CBookLevel), c_int
        ]
        self._lib.oms_get_book_snapshot.restype = c_int

        # oms_parse_itch_message(data, len, result) -> c_int
        self._lib.oms_parse_itch_message.argtypes = [
            POINTER(ctypes.c_uint8), c_int, POINTER(_CITCHMessage)
        ]
        self._lib.oms_parse_itch_message.restype = c_int

        # oms_replay_journal(path) -> c_int
        self._lib.oms_replay_journal.argtypes = [c_char_p]
        self._lib.oms_replay_journal.restype = c_int

        # oms_fpga_submit_risk_check(order, result) -> c_int
        self._lib.oms_fpga_submit_risk_check.argtypes = [POINTER(_COrder), POINTER(_CFPGARiskResult)]
        self._lib.oms_fpga_submit_risk_check.restype = c_int

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if self._fallback:
            return self._fallback.connect()

        if self._lib is None:
            return False

        import json
        config_json = json.dumps({
            "gateway_host": self.config.gateway_host,
            "gateway_port": self.config.gateway_port,
            "max_open_orders": self.config.max_open_orders,
            "max_orders_per_second": self.config.max_orders_per_second,
            "journal_path": self.config.journal_path,
            "shm_ring_buffer_size": self.config.shm_ring_buffer_size,
            "shm_name": self.config.shm_name,
        }).encode("utf-8")

        result = self._lib.oms_init(config_json)
        self._connected = result == 0

        if self._connected:
            logger.info("Native OMS connected successfully")
        else:
            logger.error(f"Native OMS connection failed with code {result}")

        return self._connected

    def disconnect(self) -> None:
        if self._fallback:
            self._fallback.disconnect()
            return

        if self._lib and self._connected:
            self._lib.oms_shutdown()
            self._connected = False
            logger.info("Native OMS disconnected")

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def submit_order(self, order: NativeOrder) -> Tuple[bool, str]:
        """Synchronous order submission via ctypes FFI; holds the GIL."""
        if self._fallback:
            return self._fallback.submit_order(order)

        if not self._connected:
            return False, "OMS not connected"

        start_ns = time.time_ns()
        
        c_order = order.as_ctypes_struct()
        result = self._lib.oms_submit_order(ctypes.byref(c_order))
        
        latency_ns = time.time_ns() - start_ns
        self._submit_latencies_ns.append(latency_ns)
        
        if result == 0:
            self._pending_orders[order.order_id] = order
            logger.debug(
                f"Order submitted: {order.order_id} {order.side.name} "
                f"{order.quantity}x{order.symbol}@{order.price} "
                f"[latency={latency_ns/1000:.1f}µs]"
            )
            return True, f"Submitted in {latency_ns/1000:.1f}µs"
        else:
            return False, f"Native OMS rejected with code {result}"

    async def submit_order_async(self, order: NativeOrder) -> Tuple[bool, str]:
        """Offloads submit_order to a thread-pool so the event loop isn't blocked."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.submit_order, order)

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        if self._fallback:
            return self._fallback.cancel_order(order_id)

        if not self._connected:
            return False, "OMS not connected"

        result = self._lib.oms_cancel_order(order_id.encode("utf-8"))
        if result == 0:
            self._pending_orders.pop(order_id, None)
            return True, "Cancel submitted"
        else:
            return False, f"Cancel failed with code {result}"

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders, optionally filtered by symbol."""
        if self._fallback:
            return self._fallback.cancel_all(symbol)

        cancelled = 0
        for oid, order in list(self._pending_orders.items()):
            if symbol is None or order.symbol == symbol:
                success, _ = self.cancel_order(oid)
                if success:
                    cancelled += 1
        return cancelled

    def get_order_status(self, order_id: str) -> Optional[NativeOrderStatus]:
        if self._fallback:
            return self._fallback.get_order_status(order_id)

        if not self._connected:
            return None

        status = c_int(0)
        result = self._lib.oms_get_order_status(
            order_id.encode("utf-8"), ctypes.byref(status)
        )
        if result == 0:
            return NativeOrderStatus(status.value)
        return None

    # ------------------------------------------------------------------
    # Fill notifications
    # ------------------------------------------------------------------

    def register_fill_callback(self, callback: Callable[[NativeFill], None]) -> None:
        self._fill_callbacks.append(callback)

    def poll_fills(self, max_count: int = 1000) -> List[NativeFill]:
        """Non-blocking poll for fills; prefer poll_fills_async from event loops."""
        if self._fallback:
            return self._fallback.poll_fills(max_count)

        if not self._connected:
            return []

        fills_array = (_CFill * max_count)()
        count = self._lib.oms_poll_fills(fills_array, max_count)

        fills = []
        for i in range(count):
            cf = fills_array[i]
            fill = NativeFill(
                fill_id=cf.fill_id.decode("utf-8").rstrip("\x00"),
                order_id=cf.order_id.decode("utf-8").rstrip("\x00"),
                symbol=cf.symbol.decode("utf-8").rstrip("\x00"),
                side=NativeOrderSide(cf.side),
                fill_qty=cf.fill_qty,
                fill_price=cf.fill_price,
                commission=cf.commission,
                liquidity_flag=cf.liquidity_flag.decode("utf-8").rstrip("\x00"),
                exchange=cf.exchange.decode("utf-8").rstrip("\x00"),
                timestamp_ns=cf.timestamp_ns,
                leaves_qty=cf.leaves_qty,
                cum_qty=cf.cum_qty,
                avg_price=cf.avg_price,
            )
            fills.append(fill)
            
            # Invoke callbacks
            for cb in self._fill_callbacks:
                try:
                    cb(fill)
                except Exception as e:
                    logger.error(f"Fill callback error: {e}")

        return fills

    async def poll_fills_async(self, max_count: int = 1000) -> List[NativeFill]:
        """Non-blocking async wrapper around :meth:`poll_fills`."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.poll_fills, max_count)

    # ------------------------------------------------------------------
    # Latency / telemetry
    # ------------------------------------------------------------------

    def get_latency_stats(self) -> Dict[str, float]:
        if self._fallback:
            return self._fallback.get_latency_stats()

        if not self._submit_latencies_ns:
            return {"min_us": 0, "max_us": 0, "avg_us": 0, "p50_us": 0, "p99_us": 0}

        import numpy as np
        arr = np.array(self._submit_latencies_ns) / 1000.0  # Convert to µs
        return {
            "min_us": float(np.min(arr)),
            "max_us": float(np.max(arr)),
            "avg_us": float(np.mean(arr)),
            "p50_us": float(np.percentile(arr, 50)),
            "p99_us": float(np.percentile(arr, 99)),
            "count": len(arr),
        }

    def get_open_orders(self) -> Dict[str, NativeOrder]:
        if self._fallback:
            return self._fallback.get_open_orders()
        return dict(self._pending_orders)

    # ------------------------------------------------------------------
    # v2.0 LOB / Market Data / ITCH / FPGA
    # ------------------------------------------------------------------

    def feed_market_data(
        self, symbol: str, bid: float, ask: float, last: float, volume: int = 0
    ) -> bool:
        """Feed market data into the native LOB (triggers stop orders, pegs)."""
        if self._fallback:
            return False
        if not self._connected or self._lib is None:
            return False
        result = self._lib.oms_feed_market_data(
            symbol.encode("utf-8"), bid, ask, last, volume
        )
        return result == 0

    def get_book_snapshot(
        self, symbol: str, max_levels: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get L2 order book snapshot from the native LOB."""
        if self._fallback or not self._connected or self._lib is None:
            return {"bids": [], "asks": []}
        bids = (_CBookLevel * max_levels)()
        asks = (_CBookLevel * max_levels)()
        cnt = self._lib.oms_get_book_snapshot(
            symbol.encode("utf-8"), bids, asks, max_levels
        )
        result: Dict[str, List[Dict[str, Any]]] = {"bids": [], "asks": []}
        for i in range(cnt):
            if bids[i].quantity > 0:
                result["bids"].append({
                    "price": bids[i].price,
                    "quantity": bids[i].quantity,
                    "order_count": bids[i].order_count,
                })
            if asks[i].quantity > 0:
                result["asks"].append({
                    "price": asks[i].price,
                    "quantity": asks[i].quantity,
                    "order_count": asks[i].order_count,
                })
        return result

    def parse_itch_message(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse an ITCH 5.0 binary message via the native parser."""
        if self._fallback or not self._connected or self._lib is None:
            return None
        buf = (ctypes.c_uint8 * len(data))(*data)
        result = _CITCHMessage()
        rc = self._lib.oms_parse_itch_message(buf, len(data), ctypes.byref(result))
        if rc != 0:
            return None
        return {
            "msg_type": chr(result.msg_type),
            "timestamp_ns": result.timestamp_ns,
            "order_ref": result.order_ref,
            "side": chr(result.side) if result.side else "",
            "shares": result.shares,
            "symbol": result.symbol.decode("utf-8").rstrip("\x00"),
            "price": result.price,
            "match_number": result.match_number,
        }

    def replay_journal(self, path: str = "") -> int:
        """Replay the event journal returning the number of events."""
        if self._fallback or not self._connected or self._lib is None:
            return -1
        return self._lib.oms_replay_journal(path.encode("utf-8") if path else None)

    def fpga_risk_check(self, order: NativeOrder) -> Optional[Dict[str, Any]]:
        """Submit an order for hardware FPGA risk pre-check."""
        if self._fallback or not self._connected or self._lib is None:
            return None
        c_order = order.as_ctypes_struct()
        result = _CFPGARiskResult()
        rc = self._lib.oms_fpga_submit_risk_check(
            ctypes.byref(c_order), ctypes.byref(result)
        )
        if rc != 0:
            return None
        return {
            "passed": result.passed == 1,
            "latency_ns": result.latency_ns,
            "error_code": result.error_code,
            "max_position_ok": result.max_position_ok == 1,
            "fat_finger_ok": result.fat_finger_ok == 1,
            "rate_limit_ok": result.rate_limit_ok == 1,
        }

    @property
    def is_connected(self) -> bool:
        if self._fallback:
            return self._fallback.is_connected
        return self._connected
