"""
Binary Market Data Feed Handlers

Production-grade handlers for direct exchange feeds:
  - Nasdaq ITCH 5.0 (TotalView)
  - Nasdaq OUCH 4.2 (Order entry)

Architecture:
    Exchange Multicast -> Feed Handler -> Ring Buffer -> Strategy

The handlers parse binary wire protocols at nanosecond precision and publish
normalized market data events to the messaging backbone.
"""

from __future__ import annotations

import asyncio
import mmap
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# ITCH 5.0 Message Types (Nasdaq TotalView)
# ---------------------------------------------------------------------------

class ITCHMessageType(IntEnum):
    SYSTEM_EVENT = ord("S")
    STOCK_DIRECTORY = ord("R")
    STOCK_TRADING_ACTION = ord("H")
    REG_SHO_RESTRICTION = ord("Y")
    MARKET_PARTICIPANT_POSITION = ord("L")
    MWCB_DECLINE_LEVEL = ord("V")
    MWCB_STATUS = ord("W")
    IPO_QUOTING_PERIOD = ord("K")
    LULD_AUCTION_COLLAR = ord("J")
    ADD_ORDER = ord("A")
    ADD_ORDER_MPID = ord("F")
    ORDER_EXECUTED = ord("E")
    ORDER_EXECUTED_WITH_PRICE = ord("C")
    ORDER_CANCEL = ord("X")
    ORDER_DELETE = ord("D")
    ORDER_REPLACE = ord("U")
    TRADE = ord("P")
    CROSS_TRADE = ord("Q")
    BROKEN_TRADE = ord("B")
    NOII = ord("I")  # Net Order Imbalance Indicator


# ---------------------------------------------------------------------------
# Parsed message structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ITCHAddOrder:
    timestamp_ns: int
    order_ref: int
    side: str        # "B" or "S"
    shares: int
    symbol: str
    price: float     # Price in dollars (converted from fixed-point)
    mpid: str = ""   # Market participant ID (type F only)


@dataclass(frozen=True, slots=True)
class ITCHOrderExecuted:
    timestamp_ns: int
    order_ref: int
    shares: int
    match_number: int
    price: float = 0.0  # Only for type C (executed with price)


@dataclass(frozen=True, slots=True)
class ITCHOrderCancel:
    timestamp_ns: int
    order_ref: int
    cancelled_shares: int


@dataclass(frozen=True, slots=True)
class ITCHOrderDelete:
    timestamp_ns: int
    order_ref: int


@dataclass(frozen=True, slots=True)
class ITCHOrderReplace:
    timestamp_ns: int
    original_order_ref: int
    new_order_ref: int
    shares: int
    price: float


@dataclass(frozen=True, slots=True)
class ITCHTrade:
    timestamp_ns: int
    order_ref: int
    side: str
    shares: int
    symbol: str
    price: float
    match_number: int


@dataclass(frozen=True, slots=True)
class ITCHSystemEvent:
    timestamp_ns: int
    event_code: str  # "O"=Start, "S"=Start, "Q"=Market Hours Start, etc.


# ---------------------------------------------------------------------------
# ITCH 5.0 Parser
# ---------------------------------------------------------------------------

class ITCHParser:
    """High-performance Nasdaq ITCH 5.0 binary protocol parser."""

    # Message lengths by type (excluding the 2-byte length prefix in TCP)
    MSG_LENGTHS = {
        ITCHMessageType.SYSTEM_EVENT: 12,
        ITCHMessageType.STOCK_DIRECTORY: 39,
        ITCHMessageType.STOCK_TRADING_ACTION: 25,
        ITCHMessageType.REG_SHO_RESTRICTION: 20,
        ITCHMessageType.MARKET_PARTICIPANT_POSITION: 26,
        ITCHMessageType.MWCB_DECLINE_LEVEL: 35,
        ITCHMessageType.MWCB_STATUS: 12,
        ITCHMessageType.IPO_QUOTING_PERIOD: 28,
        ITCHMessageType.LULD_AUCTION_COLLAR: 35,
        ITCHMessageType.ADD_ORDER: 36,
        ITCHMessageType.ADD_ORDER_MPID: 40,
        ITCHMessageType.ORDER_EXECUTED: 31,
        ITCHMessageType.ORDER_EXECUTED_WITH_PRICE: 36,
        ITCHMessageType.ORDER_CANCEL: 23,
        ITCHMessageType.ORDER_DELETE: 19,
        ITCHMessageType.ORDER_REPLACE: 35,
        ITCHMessageType.TRADE: 44,
        ITCHMessageType.CROSS_TRADE: 40,
        ITCHMessageType.BROKEN_TRADE: 19,
        ITCHMessageType.NOII: 50,
    }

    def __init__(self) -> None:
        self._message_count = 0
        self._error_count = 0
        self._stats: Dict[int, int] = defaultdict(int)
        
        # Callbacks by message type
        self._callbacks: Dict[int, List[Callable]] = defaultdict(list)

    def register_callback(self, msg_type: ITCHMessageType, callback: Callable) -> None:
        self._callbacks[msg_type.value].append(callback)

    def parse_message(self, data: bytes, offset: int = 0) -> Optional[Any]:
        """Parse a single ITCH message from a byte buffer starting at the given offset."""
        if offset >= len(data):
            return None
        
        msg_type = data[offset]
        self._message_count += 1
        self._stats[msg_type] += 1

        try:
            if msg_type == ITCHMessageType.ADD_ORDER:
                return self._parse_add_order(data, offset)
            elif msg_type == ITCHMessageType.ADD_ORDER_MPID:
                return self._parse_add_order_mpid(data, offset)
            elif msg_type == ITCHMessageType.ORDER_EXECUTED:
                return self._parse_order_executed(data, offset)
            elif msg_type == ITCHMessageType.ORDER_EXECUTED_WITH_PRICE:
                return self._parse_order_executed_price(data, offset)
            elif msg_type == ITCHMessageType.ORDER_CANCEL:
                return self._parse_order_cancel(data, offset)
            elif msg_type == ITCHMessageType.ORDER_DELETE:
                return self._parse_order_delete(data, offset)
            elif msg_type == ITCHMessageType.ORDER_REPLACE:
                return self._parse_order_replace(data, offset)
            elif msg_type == ITCHMessageType.TRADE:
                return self._parse_trade(data, offset)
            elif msg_type == ITCHMessageType.SYSTEM_EVENT:
                return self._parse_system_event(data, offset)
            else:
                return None
        except Exception as e:
            self._error_count += 1
            if self._error_count <= 10:
                logger.error(f"ITCH parse error at offset {offset}, type {msg_type}: {e}")
            return None

    def parse_stream(self, data: bytes) -> List[Any]:
        """Parse a length-prefixed stream of ITCH messages."""
        messages = []
        offset = 0
        
        while offset + 2 < len(data):
            # 2-byte length prefix
            msg_len = struct.unpack_from(">H", data, offset)[0]
            offset += 2
            
            if offset + msg_len > len(data):
                break
            
            msg = self.parse_message(data, offset)
            if msg is not None:
                messages.append(msg)
                
                # Invoke callbacks
                msg_type = data[offset]
                for cb in self._callbacks.get(msg_type, []):
                    cb(msg)
            
            offset += msg_len
        
        return messages

    # ------------------------------------------------------------------
    # Individual message parsers
    # ------------------------------------------------------------------

    def _parse_timestamp(self, data: bytes, offset: int) -> int:
        # Bytes 1-6 after message type, big-endian
        hi = struct.unpack_from(">H", data, offset + 1)[0]
        lo = struct.unpack_from(">I", data, offset + 3)[0]
        return (hi << 32) | lo

    def _parse_add_order(self, data: bytes, offset: int) -> ITCHAddOrder:
        ts = self._parse_timestamp(data, offset)
        order_ref = struct.unpack_from(">Q", data, offset + 7)[0]  # spec uses 6-byte ref; upper 2 bytes zeroed
        side = chr(data[offset + 15])
        shares = struct.unpack_from(">I", data, offset + 16)[0]
        symbol = data[offset + 20:offset + 28].decode("ascii").strip()
        price_raw = struct.unpack_from(">I", data, offset + 28)[0]
        price = price_raw / 10000.0
        
        return ITCHAddOrder(
            timestamp_ns=ts, order_ref=order_ref, side=side,
            shares=shares, symbol=symbol, price=price
        )

    def _parse_add_order_mpid(self, data: bytes, offset: int) -> ITCHAddOrder:
        ts = self._parse_timestamp(data, offset)
        order_ref = struct.unpack_from(">Q", data, offset + 7)[0]
        side = chr(data[offset + 15])
        shares = struct.unpack_from(">I", data, offset + 16)[0]
        symbol = data[offset + 20:offset + 28].decode("ascii").strip()
        price_raw = struct.unpack_from(">I", data, offset + 28)[0]
        price = price_raw / 10000.0
        mpid = data[offset + 32:offset + 36].decode("ascii").strip()
        
        return ITCHAddOrder(
            timestamp_ns=ts, order_ref=order_ref, side=side,
            shares=shares, symbol=symbol, price=price, mpid=mpid
        )

    def _parse_order_executed(self, data: bytes, offset: int) -> ITCHOrderExecuted:
        ts = self._parse_timestamp(data, offset)
        order_ref = struct.unpack_from(">Q", data, offset + 7)[0]
        shares = struct.unpack_from(">I", data, offset + 15)[0]
        match_num = struct.unpack_from(">Q", data, offset + 19)[0]
        
        return ITCHOrderExecuted(
            timestamp_ns=ts, order_ref=order_ref,
            shares=shares, match_number=match_num
        )

    def _parse_order_executed_price(self, data: bytes, offset: int) -> ITCHOrderExecuted:
        ts = self._parse_timestamp(data, offset)
        order_ref = struct.unpack_from(">Q", data, offset + 7)[0]
        shares = struct.unpack_from(">I", data, offset + 15)[0]
        match_num = struct.unpack_from(">Q", data, offset + 19)[0]
        price_raw = struct.unpack_from(">I", data, offset + 28)[0]
        price = price_raw / 10000.0
        
        return ITCHOrderExecuted(
            timestamp_ns=ts, order_ref=order_ref,
            shares=shares, match_number=match_num, price=price
        )

    def _parse_order_cancel(self, data: bytes, offset: int) -> ITCHOrderCancel:
        ts = self._parse_timestamp(data, offset)
        order_ref = struct.unpack_from(">Q", data, offset + 7)[0]
        cancelled = struct.unpack_from(">I", data, offset + 15)[0]
        
        return ITCHOrderCancel(
            timestamp_ns=ts, order_ref=order_ref, cancelled_shares=cancelled
        )

    def _parse_order_delete(self, data: bytes, offset: int) -> ITCHOrderDelete:
        ts = self._parse_timestamp(data, offset)
        order_ref = struct.unpack_from(">Q", data, offset + 7)[0]
        
        return ITCHOrderDelete(timestamp_ns=ts, order_ref=order_ref)

    def _parse_order_replace(self, data: bytes, offset: int) -> ITCHOrderReplace:
        ts = self._parse_timestamp(data, offset)
        orig_ref = struct.unpack_from(">Q", data, offset + 7)[0]
        new_ref = struct.unpack_from(">Q", data, offset + 15)[0]
        shares = struct.unpack_from(">I", data, offset + 23)[0]
        price_raw = struct.unpack_from(">I", data, offset + 27)[0]
        price = price_raw / 10000.0
        
        return ITCHOrderReplace(
            timestamp_ns=ts, original_order_ref=orig_ref,
            new_order_ref=new_ref, shares=shares, price=price
        )

    def _parse_trade(self, data: bytes, offset: int) -> ITCHTrade:
        ts = self._parse_timestamp(data, offset)
        order_ref = struct.unpack_from(">Q", data, offset + 7)[0]
        side = chr(data[offset + 15])
        shares = struct.unpack_from(">I", data, offset + 16)[0]
        symbol = data[offset + 20:offset + 28].decode("ascii").strip()
        price_raw = struct.unpack_from(">I", data, offset + 28)[0]
        price = price_raw / 10000.0
        match_num = struct.unpack_from(">Q", data, offset + 32)[0]
        
        return ITCHTrade(
            timestamp_ns=ts, order_ref=order_ref, side=side,
            shares=shares, symbol=symbol, price=price, match_number=match_num
        )

    def _parse_system_event(self, data: bytes, offset: int) -> ITCHSystemEvent:
        ts = self._parse_timestamp(data, offset)
        event_code = chr(data[offset + 7])
        
        return ITCHSystemEvent(timestamp_ns=ts, event_code=event_code)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": self._message_count,
            "error_count": self._error_count,
            "messages_by_type": {
                chr(k) if k < 128 else str(k): v for k, v in self._stats.items()
            },
        }


# ---------------------------------------------------------------------------
# OUCH 4.2 Protocol (Order Entry)
# ---------------------------------------------------------------------------

class OUCHMessageType(IntEnum):
    # Inbound (to exchange)
    ENTER_ORDER = ord("O")
    REPLACE_ORDER = ord("U")
    CANCEL_ORDER = ord("X")
    # Outbound (from exchange)
    SYSTEM_EVENT = ord("S")
    ACCEPTED = ord("A")
    REPLACED = ord("U")
    CANCELED = ord("C")
    EXECUTED = ord("E")
    BROKEN_TRADE = ord("B")
    REJECTED = ord("J")


@dataclass
class OUCHOrder:
    order_token: str      # 14-character token
    side: str             # "B" or "S"
    shares: int
    symbol: str           # 8 characters, right-padded
    price: float          # In fixed-point (price * 10000)
    time_in_force: int    # 0=Day, 99999=IOC, etc.
    firm: str = ""        # 4-character firm ID
    display: str = "Y"    # "Y"=Displayed, "N"=Non-displayed
    intermarket_sweep: str = "N"
    minimum_quantity: int = 0


class OUCHProtocol:
    """Nasdaq OUCH 4.2 order entry protocol handler."""

    def __init__(self) -> None:
        self._order_token_counter = 0
        self._callbacks: Dict[int, List[Callable]] = defaultdict(list)
    
    def build_enter_order(self, order: OUCHOrder) -> bytes:
        """Build an OUCH Enter Order message."""
        price_fixed = int(order.price * 10000)
        
        msg = struct.pack(
            ">c14s c I 8s I I 4s c c I",
            b"O",
            order.order_token.encode("ascii").ljust(14),
            order.side.encode("ascii"),
            order.shares,
            order.symbol.encode("ascii").ljust(8),
            price_fixed,
            order.time_in_force,
            order.firm.encode("ascii").ljust(4),
            order.display.encode("ascii"),
            order.intermarket_sweep.encode("ascii"),
            order.minimum_quantity,
        )
        return msg

    def build_cancel_order(self, order_token: str, shares: int = 0) -> bytes:
        """Build an OUCH Cancel Order message."""
        msg = struct.pack(
            ">c14sI",
            b"X",
            order_token.encode("ascii").ljust(14),
            shares,
        )
        return msg

    def build_replace_order(
        self, existing_token: str, replacement_token: str,
        shares: int, price: float
    ) -> bytes:
        """Build an OUCH Replace Order message."""
        price_fixed = int(price * 10000)
        
        msg = struct.pack(
            ">c14s14sII",
            b"U",
            existing_token.encode("ascii").ljust(14),
            replacement_token.encode("ascii").ljust(14),
            shares,
            price_fixed,
        )
        return msg

    def parse_response(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse an OUCH response message."""
        if not data:
            return None
        
        msg_type = chr(data[0])
        
        if msg_type == "A":  # Accepted
            return {
                "type": "ACCEPTED",
                "order_token": data[1:15].decode("ascii").strip(),
                "side": chr(data[15]),
                "shares": struct.unpack_from(">I", data, 16)[0],
                "symbol": data[20:28].decode("ascii").strip(),
                "timestamp_ns": time.time_ns(),
            }
        elif msg_type == "E":  # Executed
            return {
                "type": "EXECUTED",
                "order_token": data[1:15].decode("ascii").strip(),
                "executed_shares": struct.unpack_from(">I", data, 15)[0],
                "executed_price": struct.unpack_from(">I", data, 19)[0] / 10000.0,
                "match_number": struct.unpack_from(">Q", data, 23)[0],
                "timestamp_ns": time.time_ns(),
            }
        elif msg_type == "C":  # Canceled
            return {
                "type": "CANCELED",
                "order_token": data[1:15].decode("ascii").strip(),
                "decrement_shares": struct.unpack_from(">I", data, 15)[0],
                "reason": chr(data[19]),
                "timestamp_ns": time.time_ns(),
            }
        elif msg_type == "J":  # Rejected
            return {
                "type": "REJECTED",
                "order_token": data[1:15].decode("ascii").strip(),
                "reason": chr(data[15]),
                "timestamp_ns": time.time_ns(),
            }
        
        return {"type": "UNKNOWN", "raw": data.hex()}

    def generate_token(self) -> str:
        """Generate a unique 14-character order token."""
        self._order_token_counter += 1
        return f"T{self._order_token_counter:013d}"


class FeedHandlerManager:
    """Manages multiple feed handlers and normalizes data across venues."""

    def __init__(self) -> None:
        self._itch_parser = ITCHParser()
        self._ouch_protocol = OUCHProtocol()
        self._handlers: Dict[str, Any] = {}
        self._running = False
        self._market_data_callbacks: List[Callable] = []
        self._trade_callbacks: List[Callable] = []

    def register_feed(self, venue: str, host: str, port: int, feed_type: str = "itch") -> None:
        self._handlers[venue] = {
            "host": host,
            "port": port,
            "feed_type": feed_type,
            "connected": False,
            "messages_received": 0,
        }
        logger.info(f"Registered feed handler: {venue} ({feed_type}) @ {host}:{port}")

    def on_market_data(self, callback: Callable) -> None:
        self._market_data_callbacks.append(callback)

    def on_trade(self, callback: Callable) -> None:
        self._trade_callbacks.append(callback)

    def get_itch_parser(self) -> ITCHParser:
        return self._itch_parser

    def get_ouch_protocol(self) -> OUCHProtocol:
        return self._ouch_protocol

    def get_stats(self) -> Dict[str, Any]:
        return {
            "handlers": self._handlers,
            "itch_stats": self._itch_parser.get_stats(),
        }


__all__ = [
    "ITCHParser",
    "ITCHMessageType",
    "ITCHAddOrder",
    "ITCHOrderExecuted",
    "ITCHOrderCancel",
    "ITCHOrderDelete",
    "ITCHOrderReplace",
    "ITCHTrade",
    "ITCHSystemEvent",
    "OUCHProtocol",
    "OUCHOrder",
    "OUCHMessageType",
    "FeedHandlerManager",
]
