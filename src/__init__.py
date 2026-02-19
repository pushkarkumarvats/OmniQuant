"""
OmniQuant - Institutional Quantitative Research & Trading Framework

Architecture:
  Phase 1: Core Execution & Latency (native OMS, FIX, binary feeds, messaging)
  Phase 2: Institutional Data Platform (time-series DB, feature store, reconciliation)
  Phase 3: Distributed Research & Alpha (distributed backtest, GPU training, alt data)
  Phase 4: Risk, Ops & UI (pre-trade risk engine, drop copy, React dashboard)
"""

__version__ = "2.0.0"
__author__ = "HRT Research"
__license__ = "MIT"

from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/omniquant_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

__all__ = [
    "__version__",
    "__author__",
    "logger",
]

# ---------------------------------------------------------------------------
# Convenience re-exports for top-level access
# ---------------------------------------------------------------------------
from src.integration import TradingSystem, SystemConfig  # noqa: E402

__all__ += ["TradingSystem", "SystemConfig"]
