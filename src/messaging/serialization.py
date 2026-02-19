"""
Message Serialization Formats

Supports multiple serialization strategies optimized for different use cases:
  - FlatBuffers: Zero-copy deserialization, ideal for hot path
  - Protobuf: Schema evolution, cross-language compatibility
  - MsgPack: Compact binary, fast serialization
  - JSON: Human-readable, debugging
"""

from __future__ import annotations

import json
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from loguru import logger


class MessageSerializer(ABC):
    """Abstract serializer interface."""
    
    @abstractmethod
    def serialize(self, data: Dict[str, Any]) -> bytes:
        ...
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        ...
    
    @property
    @abstractmethod
    def content_type(self) -> str:
        ...


class JsonSerializer(MessageSerializer):
    """JSON serializer - human-readable, good for debugging."""
    
    def serialize(self, data: Dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode("utf-8"))
    
    @property
    def content_type(self) -> str:
        return "application/json"


class MsgPackSerializer(MessageSerializer):
    """Default wire serializer -- compact binary via msgpack, JSON fallback in dev."""

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = strict
        self._warned = False
        # Eagerly check availability so CI catches missing deps early.
        try:
            import msgpack  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            if strict:
                raise ImportError(
                    "msgpack is required for wire serialization. "
                    "Install it:  pip install msgpack>=1.0.7"
                )
    
    def serialize(self, data: Dict[str, Any]) -> bytes:
        if self._available:
            import msgpack
            return msgpack.packb(data, use_bin_type=True, default=str)
        # Permissive fallback - development only
        if not self._warned:
            logger.warning(
                "msgpack not installed - falling back to JSON. "
                "This MUST NOT happen in production (adds ~5x payload size)."
            )
            self._warned = True
        return json.dumps(data, default=str).encode("utf-8")
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        if self._available:
            import msgpack
            return msgpack.unpackb(data, raw=False)
        return json.loads(data.decode("utf-8"))
    
    @property
    def content_type(self) -> str:
        return "application/x-msgpack"


class CompactBinarySerializer(MessageSerializer):
    """Zero-copy-inspired binary format with offset-table layout for the hot path."""

    # Wire format:
    # [4 bytes: num_fields][per-field: 1B type + 1B key_len + key + 4B val_offset + 4B val_len][body]
    
    _TYPE_INT = 0
    _TYPE_FLOAT = 1
    _TYPE_STRING = 2
    _TYPE_BYTES = 3
    
    def serialize(self, data: Dict[str, Any]) -> bytes:
        fields = []
        body_parts = []
        body_offset = 0
        
        for key, value in data.items():
            key_bytes = key.encode("utf-8")
            
            if isinstance(value, int):
                val_bytes = struct.pack("<q", value)
                field_type = self._TYPE_INT
            elif isinstance(value, float):
                val_bytes = struct.pack("<d", value)
                field_type = self._TYPE_FLOAT
            elif isinstance(value, bytes):
                val_bytes = struct.pack("<I", len(value)) + value
                field_type = self._TYPE_BYTES
            else:
                val_str = str(value).encode("utf-8")
                val_bytes = struct.pack("<I", len(val_str)) + val_str
                field_type = self._TYPE_STRING
            
            # Field entry: [1 byte type][1 byte key_len][key][4 bytes val_offset][4 bytes val_len]
            field_entry = struct.pack("<BB", field_type, len(key_bytes)) + key_bytes
            field_entry += struct.pack("<II", body_offset, len(val_bytes))
            fields.append(field_entry)
            body_parts.append(val_bytes)
            body_offset += len(val_bytes)
        
        header = struct.pack("<I", len(fields))
        field_data = b"".join(fields)
        body_data = b"".join(body_parts)
        
        return header + field_data + body_data
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        result = {}
        offset = 0
        
        num_fields = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        
        fields_info = []
        for _ in range(num_fields):
            field_type, key_len = struct.unpack_from("<BB", data, offset)
            offset += 2
            key = data[offset:offset + key_len].decode("utf-8")
            offset += key_len
            val_offset, val_len = struct.unpack_from("<II", data, offset)
            offset += 8
            fields_info.append((key, field_type, val_offset, val_len))
        
        body_start = offset
        for key, field_type, val_offset, val_len in fields_info:
            abs_offset = body_start + val_offset
            
            if field_type == self._TYPE_INT:
                result[key] = struct.unpack_from("<q", data, abs_offset)[0]
            elif field_type == self._TYPE_FLOAT:
                result[key] = struct.unpack_from("<d", data, abs_offset)[0]
            elif field_type == self._TYPE_STRING:
                str_len = struct.unpack_from("<I", data, abs_offset)[0]
                result[key] = data[abs_offset + 4:abs_offset + 4 + str_len].decode("utf-8")
            elif field_type == self._TYPE_BYTES:
                bytes_len = struct.unpack_from("<I", data, abs_offset)[0]
                result[key] = data[abs_offset + 4:abs_offset + 4 + bytes_len]
        
        return result
    
    @property
    def content_type(self) -> str:
        return "application/x-compact-binary"


class ProtobufSerializer(MessageSerializer):
    """Protobuf shim -- delegates to MsgPack until .proto schemas exist."""

    def __init__(self) -> None:
        self._fallback = MsgPackSerializer()

    def serialize(self, data: Dict[str, Any]) -> bytes:
        return self._fallback.serialize(data)

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        return self._fallback.deserialize(data)

    @property
    def content_type(self) -> str:
        return "application/x-protobuf"
