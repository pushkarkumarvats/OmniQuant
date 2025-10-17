"""
OmniQuant - Unified Quantitative Research & Trading Framework
"""

__version__ = "0.1.0"
__author__ = "Your Name"
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
