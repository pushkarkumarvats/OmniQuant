"""
Execution algorithms, order management, and connectivity.

Includes legacy execution algorithms (TWAP, VWAP, POV, IS, Adaptive) and
Phase 2.0 subsystems: native OMS gateway, FIX protocol engine, binary feed
handlers.
"""

from .algorithms import (
    TWAP,
    VWAP,
    POV,
    ImplementationShortfall,
    AdaptiveExecution,
    ExecutionManager,
    Order,
    ChildOrder,
)

__all__ = [
    # Legacy execution algos
    "TWAP",
    "VWAP",
    "POV",
    "ImplementationShortfall",
    "AdaptiveExecution",
    "ExecutionManager",
    "Order",
    "ChildOrder",
]

# Phase 2.0 - lazy imports to avoid hard dependency on Rust lib / optional packages
try:
    from .native_gateway.oms_bridge import NativeOMSBridge, NativeOrder, NativeOrderSide, NativeOrderType
    __all__ += ["NativeOMSBridge", "NativeOrder", "NativeOrderSide", "NativeOrderType"]
except ImportError:
    pass

try:
    from .fix_protocol.fix_engine import FIXEngine, FIXSession, FIXDropCopySession, FIXConfig
    __all__ += ["FIXEngine", "FIXSession", "FIXDropCopySession", "FIXConfig"]
except ImportError:
    pass

try:
    from .feed_handlers.binary_feeds import ITCHFeedHandler, OUCHOrderEntry
    __all__ += ["ITCHFeedHandler", "OUCHOrderEntry"]
except ImportError:
    pass
