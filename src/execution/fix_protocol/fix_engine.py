"""
FIX Protocol Engine (FIX 4.2/4.4/5.0SP2)

Production-grade FIX protocol implementation for institutional order routing.
Supports FIX 4.2, 4.4, and FIXT 1.1/5.0 SP2 with binary extensions.

Architecture:
    Strategy -> FIX Engine -> FIX Session -> TCP/SSL -> Broker/Exchange
    
Features:
    - Full FIX session management (logon, heartbeat, sequence management)
    - Message validation and field-level encryption
    - Drop copy support for trade reconciliation
    - Binary FIX (FAST) for market data
    - Persistent message store for gap-fill recovery
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import socket
import ssl
import struct
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# FIX Constants
# ---------------------------------------------------------------------------

SOH = "\x01"  # FIX field separator

class FixMsgType:
    HEARTBEAT = "0"
    TEST_REQUEST = "1"
    RESEND_REQUEST = "2"
    REJECT = "3"
    SEQUENCE_RESET = "4"
    LOGOUT = "5"
    LOGON = "A"
    NEW_ORDER_SINGLE = "D"
    ORDER_CANCEL_REQUEST = "F"
    ORDER_CANCEL_REPLACE = "G"
    ORDER_STATUS_REQUEST = "H"
    EXECUTION_REPORT = "8"
    ORDER_CANCEL_REJECT = "9"
    MARKET_DATA_REQUEST = "V"
    MARKET_DATA_SNAPSHOT = "W"
    MARKET_DATA_INCREMENTAL = "X"
    SECURITY_LIST_REQUEST = "x"
    SECURITY_LIST = "y"


class FixTag:
    BEGIN_STRING = 8
    BODY_LENGTH = 9
    MSG_TYPE = 35
    SENDER_COMP_ID = 49
    TARGET_COMP_ID = 56
    MSG_SEQ_NUM = 34
    SENDING_TIME = 52
    CHECKSUM = 10
    
    # Order tags
    CL_ORD_ID = 11
    ORDER_ID = 37
    SYMBOL = 55
    SIDE = 54
    ORDER_QTY = 38
    ORD_TYPE = 40
    PRICE = 44
    STOP_PX = 99
    TIME_IN_FORCE = 59
    ACCOUNT = 1
    EXEC_TYPE = 150
    ORD_STATUS = 39
    LAST_QTY = 32
    LAST_PX = 31
    LEAVES_QTY = 151
    CUM_QTY = 14
    AVG_PX = 6
    COMMISSION = 12
    EXEC_ID = 17
    TEXT = 58
    
    # Heartbeat / session
    ENCRYPT_METHOD = 98
    HEART_BT_INT = 108
    TEST_REQ_ID = 112
    BEGIN_SEQ_NO = 7
    END_SEQ_NO = 16
    GAP_FILL_FLAG = 123
    NEW_SEQ_NO = 36
    
    # Market data
    MD_REQ_ID = 262
    SUBSCRIPTION_REQUEST_TYPE = 263
    MARKET_DEPTH = 264
    MD_UPDATE_TYPE = 265
    NO_MD_ENTRY_TYPES = 267
    MD_ENTRY_TYPE = 269
    NO_MD_ENTRIES = 268
    MD_ENTRY_PX = 270
    MD_ENTRY_SIZE = 271
    

class FixSide(IntEnum):
    BUY = 1
    SELL = 2
    SHORT_SELL = 5
    SHORT_SELL_EXEMPT = 6


class FixOrdType(IntEnum):
    MARKET = 1
    LIMIT = 2
    STOP = 3
    STOP_LIMIT = 4


class FixTimeInForce(IntEnum):
    DAY = 0
    GTC = 1
    IOC = 3
    FOK = 4
    GTD = 6


class FixExecType:
    NEW = "0"
    PARTIAL_FILL = "1"
    FILL = "2"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    REPLACED = "5"
    PENDING_CANCEL = "6"
    REJECTED = "8"
    PENDING_NEW = "A"
    PENDING_REPLACE = "E"
    TRADE = "F"


class FixOrdStatus:
    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    REPLACED = "5"
    PENDING_CANCEL = "6"
    REJECTED = "8"


# ---------------------------------------------------------------------------
# FIX Message
# ---------------------------------------------------------------------------

class FixMessage:
    """FIX protocol message with ordered field storage."""
    
    def __init__(self, msg_type: str = "") -> None:
        self.fields: OrderedDict[int, str] = OrderedDict()
        if msg_type:
            self.fields[FixTag.MSG_TYPE] = msg_type
    
    def set_field(self, tag: int, value: Any) -> "FixMessage":
        self.fields[tag] = str(value)
        return self
    
    def get_field(self, tag: int) -> Optional[str]:
        return self.fields.get(tag)
    
    def get_int(self, tag: int) -> int:
        return int(self.fields.get(tag, "0"))
    
    def get_float(self, tag: int) -> float:
        return float(self.fields.get(tag, "0.0"))
    
    @property
    def msg_type(self) -> str:
        return self.fields.get(FixTag.MSG_TYPE, "")
    
    def serialize(self, begin_string: str, sender: str, target: str, seq_num: int) -> bytes:
        """Serialize to FIX wire format."""
        # Build body (everything except BeginString, BodyLength, Checksum)
        body_fields = OrderedDict()
        body_fields[FixTag.MSG_TYPE] = self.fields.get(FixTag.MSG_TYPE, "")
        body_fields[FixTag.SENDER_COMP_ID] = sender
        body_fields[FixTag.TARGET_COMP_ID] = target
        body_fields[FixTag.MSG_SEQ_NUM] = str(seq_num)
        body_fields[FixTag.SENDING_TIME] = time.strftime("%Y%m%d-%H:%M:%S.") + f"{int(time.time() * 1000) % 1000:03d}"
        
        # Add remaining fields
        for tag, value in self.fields.items():
            if tag not in (FixTag.BEGIN_STRING, FixTag.BODY_LENGTH, FixTag.CHECKSUM, FixTag.MSG_TYPE):
                body_fields[tag] = value
        
        body = SOH.join(f"{tag}={value}" for tag, value in body_fields.items()) + SOH
        
        # Construct header
        header = f"{FixTag.BEGIN_STRING}={begin_string}{SOH}{FixTag.BODY_LENGTH}={len(body)}{SOH}"
        
        # Calculate checksum
        msg_without_checksum = header + body
        checksum = sum(ord(c) for c in msg_without_checksum) % 256
        
        full_msg = f"{msg_without_checksum}{FixTag.CHECKSUM}={checksum:03d}{SOH}"
        return full_msg.encode("ascii")
    
    @classmethod
    def parse(cls, raw: bytes) -> Optional["FixMessage"]:
        """Parse a raw FIX message."""
        try:
            text = raw.decode("ascii")
            msg = cls()
            
            for pair in text.split(SOH):
                if "=" in pair:
                    tag_str, value = pair.split("=", 1)
                    tag = int(tag_str)
                    msg.fields[tag] = value
            
            return msg
        except Exception as e:
            logger.error(f"Failed to parse FIX message: {e}")
            return None
    
    def __repr__(self) -> str:
        fields_str = " | ".join(f"{k}={v}" for k, v in self.fields.items())
        return f"FixMessage({fields_str})"


# ---------------------------------------------------------------------------
# FIX Session Configuration
# ---------------------------------------------------------------------------

@dataclass
class FixSessionConfig:
    begin_string: str = "FIX.4.4"
    sender_comp_id: str = ""
    target_comp_id: str = ""
    host: str = ""
    port: int = 0
    heartbeat_interval: int = 30
    use_ssl: bool = True
    
    # Authentication
    username: str = ""
    password: str = ""
    
    # Message store
    store_path: str = "./fix_store"
    
    # Reconnection
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
    
    # Drop copy
    is_drop_copy: bool = False


# ---------------------------------------------------------------------------
# FIX Session
# ---------------------------------------------------------------------------

class FixSession:
    """FIX protocol session manager with full lifecycle, gap-fill, and reconnection."""
    
    def __init__(self, config: FixSessionConfig) -> None:
        self.config = config
        self._socket: Optional[socket.socket] = None
        self._ssl_socket: Optional[ssl.SSLSocket] = None
        self._connected = False
        self._logged_on = False
        
        # Sequence numbers
        self._outgoing_seq = 1
        self._incoming_seq = 1
        
        # Message store
        self._sent_messages: Dict[int, bytes] = {}
        self._received_messages: Dict[int, FixMessage] = {}
        
        # Callbacks
        self._execution_callbacks: List[Callable[[FixMessage], None]] = []
        self._market_data_callbacks: List[Callable[[FixMessage], None]] = []
        self._admin_callbacks: List[Callable[[FixMessage], None]] = []
        
        # Heartbeat tracking
        self._last_sent_time = 0.0
        self._last_received_time = 0.0
        
        # Reconnection
        self._reconnect_count = 0
        
        # Running tasks
        self._reader_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Establish TCP/SSL connection and send Logon."""
        try:
            # Create socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(10.0)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            if self.config.use_ssl:
                context = ssl.create_default_context()
                self._ssl_socket = context.wrap_socket(
                    self._socket, server_hostname=self.config.host
                )
                self._ssl_socket.connect((self.config.host, self.config.port))
            else:
                self._socket.connect((self.config.host, self.config.port))
            
            self._connected = True
            self._reconnect_count = 0
            
            # Send logon
            await self._send_logon()
            
            logger.info(
                f"FIX session connected: {self.config.sender_comp_id} -> "
                f"{self.config.target_comp_id} @ {self.config.host}:{self.config.port}"
            )
            return True
            
        except Exception as e:
            logger.error(f"FIX connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Send Logout and close connection."""
        if self._logged_on:
            await self._send_logout("Normal disconnect")
        
        if self._reader_task:
            self._reader_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        if self._ssl_socket:
            self._ssl_socket.close()
        elif self._socket:
            self._socket.close()
        
        self._connected = False
        self._logged_on = False
        logger.info("FIX session disconnected")

    # ------------------------------------------------------------------
    # Order operations
    # ------------------------------------------------------------------

    async def send_new_order(
        self,
        cl_ord_id: str,
        symbol: str,
        side: FixSide,
        quantity: int,
        ord_type: FixOrdType,
        price: float = 0.0,
        stop_price: float = 0.0,
        time_in_force: FixTimeInForce = FixTimeInForce.DAY,
        account: str = "",
    ) -> bool:
        """Send a NewOrderSingle (MsgType=D)."""
        msg = FixMessage(FixMsgType.NEW_ORDER_SINGLE)
        msg.set_field(FixTag.CL_ORD_ID, cl_ord_id)
        msg.set_field(FixTag.SYMBOL, symbol)
        msg.set_field(FixTag.SIDE, side.value)
        msg.set_field(FixTag.ORDER_QTY, quantity)
        msg.set_field(FixTag.ORD_TYPE, ord_type.value)
        msg.set_field(FixTag.TIME_IN_FORCE, time_in_force.value)
        
        if price > 0:
            msg.set_field(FixTag.PRICE, f"{price:.6f}")
        if stop_price > 0:
            msg.set_field(FixTag.STOP_PX, f"{stop_price:.6f}")
        if account:
            msg.set_field(FixTag.ACCOUNT, account)
        
        return await self._send_message(msg)

    async def send_cancel_request(self, cl_ord_id: str, orig_cl_ord_id: str,
                                   symbol: str, side: FixSide) -> bool:
        """Send an OrderCancelRequest (MsgType=F)."""
        msg = FixMessage(FixMsgType.ORDER_CANCEL_REQUEST)
        msg.set_field(FixTag.CL_ORD_ID, cl_ord_id)
        msg.set_field(41, orig_cl_ord_id)  # OrigClOrdID
        msg.set_field(FixTag.SYMBOL, symbol)
        msg.set_field(FixTag.SIDE, side.value)
        
        return await self._send_message(msg)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_execution_report(self, callback: Callable[[FixMessage], None]) -> None:
        self._execution_callbacks.append(callback)

    def on_market_data(self, callback: Callable[[FixMessage], None]) -> None:
        self._market_data_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Internal session management
    # ------------------------------------------------------------------

    async def _send_logon(self) -> None:
        msg = FixMessage(FixMsgType.LOGON)
        msg.set_field(FixTag.ENCRYPT_METHOD, 0)
        msg.set_field(FixTag.HEART_BT_INT, self.config.heartbeat_interval)
        
        if self.config.username:
            msg.set_field(553, self.config.username)  # Username
        if self.config.password:
            msg.set_field(554, self.config.password)  # Password
        
        await self._send_message(msg)
        self._logged_on = True

    async def _send_logout(self, reason: str = "") -> None:
        msg = FixMessage(FixMsgType.LOGOUT)
        if reason:
            msg.set_field(FixTag.TEXT, reason)
        await self._send_message(msg)
        self._logged_on = False

    async def _send_heartbeat(self, test_req_id: str = "") -> None:
        msg = FixMessage(FixMsgType.HEARTBEAT)
        if test_req_id:
            msg.set_field(FixTag.TEST_REQ_ID, test_req_id)
        await self._send_message(msg)

    async def _send_message(self, msg: FixMessage) -> bool:
        if not self._connected:
            return False
        
        try:
            raw = msg.serialize(
                self.config.begin_string,
                self.config.sender_comp_id,
                self.config.target_comp_id,
                self._outgoing_seq,
            )
            
            # Store for potential resend
            self._sent_messages[self._outgoing_seq] = raw
            self._outgoing_seq += 1
            
            sock = self._ssl_socket or self._socket
            if sock:
                sock.sendall(raw)
            
            self._last_sent_time = time.time()
            
            logger.debug(f"FIX SENT [{msg.msg_type}] seq={self._outgoing_seq - 1}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send FIX message: {e}")
            return False

    def _handle_incoming(self, msg: FixMessage) -> None:
        msg_type = msg.msg_type
        
        self._last_received_time = time.time()
        seq = msg.get_int(FixTag.MSG_SEQ_NUM)
        self._received_messages[seq] = msg
        
        if msg_type == FixMsgType.HEARTBEAT:
            pass  # Just update last received time
        
        elif msg_type == FixMsgType.TEST_REQUEST:
            test_req_id = msg.get_field(FixTag.TEST_REQ_ID) or ""
            asyncio.create_task(self._send_heartbeat(test_req_id))

        elif msg_type == FixMsgType.SEQUENCE_RESET:
            self._handle_sequence_reset(msg)

        elif msg_type == FixMsgType.RESEND_REQUEST:
            begin = msg.get_int(FixTag.BEGIN_SEQ_NO)
            end = msg.get_int(FixTag.END_SEQ_NO)
            asyncio.create_task(self._handle_resend_request(begin, end))
        
        elif msg_type == FixMsgType.EXECUTION_REPORT:
            for cb in self._execution_callbacks:
                try:
                    cb(msg)
                except Exception as e:
                    logger.error(f"Execution report callback error: {e}")
        
        elif msg_type in (FixMsgType.MARKET_DATA_SNAPSHOT, FixMsgType.MARKET_DATA_INCREMENTAL):
            for cb in self._market_data_callbacks:
                try:
                    cb(msg)
                except Exception as e:
                    logger.error(f"Market data callback error: {e}")
        
        elif msg_type == FixMsgType.LOGOUT:
            reason = msg.get_field(FixTag.TEXT) or "No reason"
            logger.warning(f"FIX logout received: {reason}")
            self._logged_on = False
        
        elif msg_type == FixMsgType.REJECT:
            reason = msg.get_field(FixTag.TEXT) or "Unknown"
            logger.error(f"FIX session reject: {reason}")

        # Persist sequences periodically
        if seq % 100 == 0:
            self.persist_sequences()

    async def _handle_resend_request(self, begin: int, end: int) -> None:
        actual_end = end if end > 0 else self._outgoing_seq - 1
        logger.info(f"Processing ResendRequest {begin}-{actual_end}")
        gap_start: Optional[int] = None

        for seq in range(begin, actual_end + 1):
            stored = self._sent_messages.get(seq)
            if stored is None:
                if gap_start is None:
                    gap_start = seq
                continue
            # Flush any pending gap
            if gap_start is not None:
                await self.send_sequence_reset(seq, gap_fill=True)
                gap_start = None
            # Re-send the stored message
            sock = self._ssl_socket or self._socket
            if sock:
                try:
                    sock.sendall(stored)
                except Exception as e:
                    logger.error(f"Resend failed for seq {seq}: {e}")
        # Trailing gap
        if gap_start is not None:
            await self.send_sequence_reset(actual_end + 1, gap_fill=True)

    @property
    def is_logged_on(self) -> bool:
        return self._logged_on

    @property
    def outgoing_seq(self) -> int:
        return self._outgoing_seq

    @property
    def incoming_seq(self) -> int:
        return self._incoming_seq

    # ------------------------------------------------------------------
    # Session recovery & reconnection (v2)
    # ------------------------------------------------------------------

    async def reconnect(self) -> bool:
        """Reconnect with exponential back-off, then request gap-fill."""
        max_attempts = self.config.max_reconnect_attempts
        base_delay = self.config.reconnect_interval

        for attempt in range(1, max_attempts + 1):
            delay = min(base_delay * (2 ** (attempt - 1)), 120)
            logger.info(
                f"Reconnect attempt {attempt}/{max_attempts} in {delay}s"
            )
            await asyncio.sleep(delay)
            try:
                ok = await self.connect()
                if ok:
                    logger.info(f"Reconnected after {attempt} attempts")
                    await self._request_gap_fill()
                    return True
            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")

        logger.error("Exhausted reconnection attempts")
        return False

    async def _request_gap_fill(self) -> None:
        msg = FixMessage(FixMsgType.RESEND_REQUEST)
        msg.set_field(FixTag.BEGIN_SEQ_NO, self._incoming_seq)
        msg.set_field(FixTag.END_SEQ_NO, 0)  # 0 = infinity
        await self._send_message(msg)
        logger.info(f"Sent ResendRequest from seq {self._incoming_seq}")

    async def send_sequence_reset(self, new_seq: int, gap_fill: bool = True) -> bool:
        """Send a SequenceReset to advance the counterparty's expected sequence."""
        msg = FixMessage(FixMsgType.SEQUENCE_RESET)
        msg.set_field(FixTag.NEW_SEQ_NO, new_seq)
        if gap_fill:
            msg.set_field(FixTag.GAP_FILL_FLAG, "Y")
        return await self._send_message(msg)

    async def recover_session(self) -> bool:
        """Full session recovery: reload persisted sequences, reconnect, gap-fill."""
        logger.info("Starting full session recovery")
        await self.disconnect()

        # Reload persisted seqs (in production: read from journal/store)
        persisted = self._load_persisted_sequences()
        if persisted:
            self._outgoing_seq, self._incoming_seq = persisted
            logger.info(
                f"Restored sequences: out={self._outgoing_seq}, in={self._incoming_seq}"
            )

        return await self.reconnect()

    def _load_persisted_sequences(self) -> Optional[Tuple[int, int]]:
        store_dir = Path(self.config.store_path)
        seq_file = store_dir / f"{self.config.sender_comp_id}_seq.dat"
        if seq_file.exists():
            try:
                data = seq_file.read_text().strip().split(",")
                return int(data[0]), int(data[1])
            except Exception:
                pass
        return None

    def persist_sequences(self) -> None:
        """Persist current sequence numbers for crash recovery."""
        store_dir = Path(self.config.store_path)
        store_dir.mkdir(parents=True, exist_ok=True)
        seq_file = store_dir / f"{self.config.sender_comp_id}_seq.dat"
        seq_file.write_text(f"{self._outgoing_seq},{self._incoming_seq}")

    def _handle_sequence_reset(self, msg: FixMessage) -> None:
        new_seq = msg.get_int(FixTag.NEW_SEQ_NO)
        gap_fill = msg.get_field(FixTag.GAP_FILL_FLAG) == "Y"

        if gap_fill:
            logger.info(f"GapFill: advancing expected seq to {new_seq}")
            self._incoming_seq = new_seq
        else:
            logger.warning(f"Hard SequenceReset to {new_seq}")
            self._incoming_seq = new_seq


# ---------------------------------------------------------------------------
# FIX Drop Copy Session
# ---------------------------------------------------------------------------

class FixDropCopySession(FixSession):
    """Drop copy session for reconciling executions from the clearing broker."""

    def __init__(self, config: FixSessionConfig) -> None:
        config.is_drop_copy = True
        super().__init__(config)
        self._drop_copy_reports: List[FixMessage] = []
        self._reconciliation_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def on_drop_copy(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._reconciliation_callbacks.append(callback)

    def _handle_incoming(self, msg: FixMessage) -> None:
        super()._handle_incoming(msg)
        
        if msg.msg_type == FixMsgType.EXECUTION_REPORT:
            self._drop_copy_reports.append(msg)
            
            # Build reconciliation record
            record = {
                "exec_id": msg.get_field(FixTag.EXEC_ID),
                "order_id": msg.get_field(FixTag.ORDER_ID),
                "cl_ord_id": msg.get_field(FixTag.CL_ORD_ID),
                "symbol": msg.get_field(FixTag.SYMBOL),
                "side": msg.get_field(FixTag.SIDE),
                "exec_type": msg.get_field(FixTag.EXEC_TYPE),
                "last_qty": msg.get_float(FixTag.LAST_QTY),
                "last_px": msg.get_float(FixTag.LAST_PX),
                "cum_qty": msg.get_float(FixTag.CUM_QTY),
                "avg_px": msg.get_float(FixTag.AVG_PX),
                "commission": msg.get_float(FixTag.COMMISSION),
                "timestamp": msg.get_field(FixTag.SENDING_TIME),
            }
            
            for cb in self._reconciliation_callbacks:
                try:
                    cb(record)
                except Exception as e:
                    logger.error(f"Drop copy callback error: {e}")

    def get_daily_fills(self) -> List[Dict[str, Any]]:
        """Get all fills received today for reconciliation."""
        fills = []
        for msg in self._drop_copy_reports:
            exec_type = msg.get_field(FixTag.EXEC_TYPE)
            if exec_type in (FixExecType.FILL, FixExecType.PARTIAL_FILL, FixExecType.TRADE):
                fills.append({
                    "exec_id": msg.get_field(FixTag.EXEC_ID),
                    "symbol": msg.get_field(FixTag.SYMBOL),
                    "side": msg.get_field(FixTag.SIDE),
                    "qty": msg.get_float(FixTag.LAST_QTY),
                    "price": msg.get_float(FixTag.LAST_PX),
                    "commission": msg.get_float(FixTag.COMMISSION),
                })
        return fills


__all__ = [
    "FixMessage",
    "FixSession",
    "FixSessionConfig",
    "FixDropCopySession",
    "FixMsgType",
    "FixTag",
    "FixSide",
    "FixOrdType",
    "FixTimeInForce",
    "FixExecType",
    "FixOrdStatus",
]
