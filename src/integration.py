"""
System Integration - Wires all institutional subsystems together.

Usage:
    from src.integration import TradingSystem
    system = TradingSystem.create()
    await system.start()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    # Hot-path transport (MD → Signal → Risk → OMS): must be ultra-low-latency
    hot_transport_backend: str = "in_memory"  # aeron | in_memory (NEVER kafka)
    # Cold-path transport (feature store, drop copy, analytics): throughput-oriented
    cold_transport_backend: str = "in_memory"  # kafka | redpanda | in_memory
    kafka_bootstrap: str = "localhost:9092"
    # Time-series DB
    tsdb_backend: str = "duckdb"  # duckdb | clickhouse | arcticdb
    duckdb_path: str = "data/timeseries.duckdb"
    # Feature store
    feature_store_path: str = "data/feature_store.duckdb"
    online_feature_ttl: int = 3600
    # Risk
    max_daily_loss: float = 500_000
    max_position_notional: float = 10_000_000
    max_drawdown_pct: float = 0.05
    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    # OMS
    use_native_oms: bool = False  # requires compiled Rust lib


class TradingSystem:
    """
    Top-level container that wires all subsystems.

    Lifecycle:
        system = TradingSystem.create(config)
        await system.start()
        ...
        await system.stop()
    """

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        # Hot path: ultra-low-latency (Aeron/InMemory) for MD → Signal → Risk → OMS
        self.hot_transport = None
        # Cold path: high-throughput (Kafka/Redpanda) for feature store, drop copy, analytics
        self.cold_transport = None
        self.hot_channels = None
        self.cold_channels = None
        self.timeseries_db = None
        self.feature_engine = None
        self.online_store = None
        self.offline_store = None
        self.risk_engine = None
        self.reconciler = None
        self.oms = None
        self._running = False

    @classmethod
    def create(cls, config: Optional[SystemConfig] = None) -> "TradingSystem":
        """Factory method to create a fully wired trading system."""
        config = config or SystemConfig()
        system = cls(config)
        system._wire()
        return system

    def _wire(self) -> None:
        """Instantiate and wire all subsystems with split hot/cold paths."""
        cfg = self.config

        # ── 1. Dual messaging transports ──────────────────────────────
        from src.messaging.transport import TransportConfig, TransportBackend, create_transport

        _backend_map = {
            "in_memory": TransportBackend.IN_MEMORY,
            "aeron": TransportBackend.AERON,
            "kafka": TransportBackend.KAFKA,
            "redpanda": TransportBackend.REDPANDA,
        }

        # Hot path - sub-microsecond IPC for MD/Signal/Risk/Order
        hot_backend = _backend_map.get(cfg.hot_transport_backend, TransportBackend.IN_MEMORY)
        if hot_backend in (TransportBackend.KAFKA, TransportBackend.REDPANDA):
            logger.warning(
                "Kafka/Redpanda selected for hot path - overriding to InMemory. "
                "Kafka adds 1-5 ms latency, unsuitable for the order hot path."
            )
            hot_backend = TransportBackend.IN_MEMORY
        self.hot_transport = create_transport(
            TransportConfig(backend=hot_backend, linger_ms=0)
        )

        # Cold path - durable high-throughput for analytics, drop copy
        cold_backend = _backend_map.get(cfg.cold_transport_backend, TransportBackend.IN_MEMORY)
        self.cold_transport = create_transport(
            TransportConfig(
                backend=cold_backend,
                bootstrap_servers=cfg.kafka_bootstrap,
                compression="lz4",
            )
        )

        # ── 2. Typed channels (hot path for latency-sensitive) ────────
        from src.messaging.channels import (
            ChannelManager, MarketDataChannel, OrderChannel,
            FillChannel, SignalChannel, RiskChannel,
        )
        # Hot channels: market data, orders, fills, signals, risk
        self.hot_channels = ChannelManager(self.hot_transport)
        self.hot_channels.register(MarketDataChannel(self.hot_transport))
        self.hot_channels.register(OrderChannel(self.hot_transport))
        self.hot_channels.register(SignalChannel(self.hot_transport))
        self.hot_channels.register(RiskChannel(self.hot_transport))

        # Cold channels: fills go to BOTH hot (real-time) and cold (persistence)
        self.cold_channels = ChannelManager(self.cold_transport)
        self.cold_channels.register(FillChannel(self.cold_transport))

        # ── 3. Time-series database ───────────────────────────────────
        from src.data_platform.timeseries_db import TimeSeriesConfig, TimeSeriesBackend, create_timeseries_db
        tsdb_backend_map = {
            "duckdb": TimeSeriesBackend.DUCKDB,
            "clickhouse": TimeSeriesBackend.CLICKHOUSE,
            "arcticdb": TimeSeriesBackend.ARCTICDB,
        }
        tsdb_cfg = TimeSeriesConfig(
            backend=tsdb_backend_map.get(cfg.tsdb_backend, TimeSeriesBackend.DUCKDB),
            duckdb_path=cfg.duckdb_path,
        )
        self.timeseries_db = create_timeseries_db(tsdb_cfg)

        # ── 4. Feature stores ─────────────────────────────────────────
        from src.data_platform.feature_store import (
            FeatureRegistry, StreamingFeatureEngine,
            OfflineFeatureStore, OnlineFeatureStore,
        )
        registry = FeatureRegistry()
        self.feature_engine = StreamingFeatureEngine(registry)
        self.online_store = OnlineFeatureStore(ttl_seconds=cfg.online_feature_ttl)
        self.offline_store = OfflineFeatureStore(db_path=cfg.feature_store_path)

        # ── 5. Risk engine ────────────────────────────────────────────
        from src.risk_ops.risk_engine import PreTradeRiskEngine, RiskLimits
        risk_limits = RiskLimits(
            max_daily_loss=cfg.max_daily_loss,
            max_position_notional=cfg.max_position_notional,
            max_drawdown_pct=cfg.max_drawdown_pct,
        )
        self.risk_engine = PreTradeRiskEngine(limits=risk_limits)

        # ── 6. Drop copy reconciler ───────────────────────────────────
        from src.risk_ops.drop_copy import DropCopyReconciler
        self.reconciler = DropCopyReconciler()

        # ── 7. OMS Bridge ─────────────────────────────────────────────
        from src.execution.native_gateway.oms_bridge import NativeOMSBridge, OMSConfig
        oms_config = OMSConfig()
        self.oms = NativeOMSBridge(oms_config)

        logger.info(
            f"TradingSystem wired - hot={cfg.hot_transport_backend}, "
            f"cold={cfg.cold_transport_backend}"
        )

    async def start(self) -> None:
        """Start all subsystems."""
        logger.info("Starting TradingSystem...")

        # Start both transport paths
        await self.hot_channels.start_all()
        await self.cold_channels.start_all()

        # Connect DB
        await self.timeseries_db.connect()
        await self.timeseries_db.create_tables()

        # Connect offline feature store
        await self.offline_store.connect()

        # Wire risk alerts → hot risk channel
        risk_channel = self.hot_channels.get("risk")
        if risk_channel and self.risk_engine:
            def _on_risk_alert(result):
                asyncio.ensure_future(risk_channel.publish({
                    "event": "risk_check",
                    "check_type": result.check_type.value,
                    "action": result.action.value,
                    "message": result.message,
                    "level": result.level.value,
                }))
            self.risk_engine.on_alert(_on_risk_alert)

        self._running = True
        logger.info("TradingSystem started")

    async def stop(self) -> None:
        """Stop all subsystems."""
        self._running = False
        await self.hot_channels.stop_all()
        await self.cold_channels.stop_all()
        await self.timeseries_db.disconnect()
        await self.offline_store.disconnect()
        logger.info("TradingSystem stopped")

    # ----- convenience helpers for backward compat -----

    @property
    def channels(self):
        """Return the hot-path channel manager (backward compat)."""
        return self.hot_channels

    @property
    def transport(self):
        """Return the hot-path transport (backward compat)."""
        return self.hot_transport

    def validate_order(self, order: Dict[str, Any], entity_id: str = "default"):
        """Pre-trade risk check shortcut."""
        return self.risk_engine.validate_order(order, entity_id)

    async def submit_order(self, order: Dict[str, Any], entity_id: str = "default"):
        """Full order submission: risk check → OMS (GIL-released) → channel publish."""
        from src.risk_ops.risk_engine import RiskAction

        action, results = self.validate_order(order, entity_id)
        if action in (RiskAction.REJECT, RiskAction.KILL):
            logger.warning(f"Order rejected by risk engine: {results[-1].message}")
            return {"status": "rejected", "reason": results[-1].message}

        # Submit to OMS via async bridge (offloads GIL-holding FFI to executor)
        import uuid
        from src.execution.native_gateway.oms_bridge import NativeOrder, NativeOrderSide, NativeOrderType

        order_id = order.get("order_id", uuid.uuid4().hex[:16])
        native_order = NativeOrder(
            order_id=order_id,
            client_order_id=order.get("client_order_id", order_id),
            symbol=order.get("symbol", ""),
            side=NativeOrderSide.BUY if order.get("side") == "buy" else NativeOrderSide.SELL,
            order_type=NativeOrderType.LIMIT,
            quantity=order.get("quantity", 0),
            price=order.get("price", 0),
        )
        result = await self.oms.submit_order_async(native_order)

        # Publish to hot-path order channel (no Kafka in the loop)
        order_channel = self.hot_channels.get("orders")
        if order_channel:
            await order_channel.publish({
                "event": "new_order",
                "order_id": native_order.order_id,
                "symbol": native_order.symbol,
                "side": order.get("side"),
                "quantity": native_order.quantity,
                "price": native_order.price,
                "oms_result": str(result),
            })

        return {"status": "submitted", "oms_result": str(result)}


def create_dashboard_server(system: TradingSystem):
    """Create the FastAPI dashboard wired to the trading system."""
    from src.dashboard.dashboard_api import create_dashboard_app
    return create_dashboard_app(
        risk_engine=system.risk_engine,
        reconciler=system.reconciler,
        feature_store=system.online_store,
        timeseries_db=system.timeseries_db,
    )
