"""
Native Execution Gateway (C++/Rust OMS)

This module provides Python bindings to the native C++20/Rust Order Management System.
The native OMS handles all network I/O and order state management with deterministic,
microsecond-level latency. Python only generates signals; the native layer handles execution.

Architecture:
    Python Signal Generator -> FFI Bridge -> Native OMS -> Exchange/Broker
    
The native gateway is compiled as a shared library (.so/.dll) and loaded via ctypes/cffi.
For development, a pure-Python reference implementation is provided that mirrors the native API.
"""

from .oms_bridge import (
    NativeOMSBridge,
    NativeOrder,
    NativeOrderStatus,
    NativeFill,
    OMSConfig,
)
from .reference_oms import ReferenceOMS

__all__ = [
    "NativeOMSBridge",
    "NativeOrder",
    "NativeOrderStatus",
    "NativeFill",
    "OMSConfig",
    "ReferenceOMS",
]
