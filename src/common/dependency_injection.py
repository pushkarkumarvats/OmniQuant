"""
Dependency Injection Framework
Uses the industry-standard ``dependency-injector`` library for a professional,
extensible IoC container with provider graph validation and wiring support.
"""

from __future__ import annotations

from typing import Any

from dependency_injector import containers, providers
from loguru import logger


class ServiceNotFoundError(Exception):
    pass


# ---------------------------------------------------------------------------
# Declarative DI Container
# ---------------------------------------------------------------------------


class ApplicationContainer(containers.DeclarativeContainer):
    """Central IoC container for all service providers."""

    wiring_config = containers.WiringConfiguration(
        modules=[
            "src.api.main",
        ],
    )

    # -- Configuration -------------------------------------------------------
    config = providers.Configuration()

    # -- Data Layer ----------------------------------------------------------
    data_ingestion = providers.Singleton(
        "src.data_pipeline.ingestion.DataIngestion",
    )

    # -- Feature Engineering -------------------------------------------------
    technical_features = providers.Singleton(
        "src.feature_engineering.technical_features.TechnicalFeatures",
    )
    microstructure_features = providers.Singleton(
        "src.feature_engineering.microstructure_features.MicrostructureFeatures",
    )

    # -- Portfolio & Risk ----------------------------------------------------
    portfolio_optimizer = providers.Singleton(
        "src.portfolio.optimizer.PortfolioOptimizer",
    )
    risk_manager = providers.Singleton(
        "src.portfolio.risk_manager.RiskManager",
    )


# ---------------------------------------------------------------------------
# Legacy-compatible DIContainer wrapper
# ---------------------------------------------------------------------------


class DIContainer:
    """Legacy-compatible wrapper around ApplicationContainer."""

    def __init__(self) -> None:
        self._container = ApplicationContainer()
        self._overrides: dict[type, Any] = {}
        logger.debug("DI Container initialized (dependency-injector backend)")

    def register_singleton(self, service_type: type, implementation: type | None = None) -> None:
        impl = implementation or service_type
        self._overrides[service_type] = providers.Singleton(impl)
        logger.debug(f"Registered singleton: {service_type.__name__}")

    def register_transient(self, service_type: type, implementation: type | None = None) -> None:
        impl = implementation or service_type
        self._overrides[service_type] = providers.Factory(impl)
        logger.debug(f"Registered transient: {service_type.__name__}")

    def register_instance(self, service_type: type, instance: Any) -> None:
        self._overrides[service_type] = providers.Object(instance)
        logger.debug(f"Registered instance: {service_type.__name__}")

    def register_factory(self, service_type: type, factory: Any, singleton: bool = False) -> None:
        if singleton:
            self._overrides[service_type] = providers.Singleton(factory)
        else:
            self._overrides[service_type] = providers.Factory(factory)
        logger.debug(f"Registered factory: {service_type.__name__}")

    def resolve(self, service_type: type) -> Any:
        """Resolve a service, checking overrides first, then the container."""
        if service_type in self._overrides:
            provider = self._overrides[service_type]
            return provider()

        # Try well-known container providers by matching type
        for attr in dir(self._container):
            provider = getattr(self._container, attr)
            if isinstance(provider, providers.Provider):
                try:
                    instance = provider()
                    if isinstance(instance, service_type):
                        return instance
                except Exception:
                    continue

        raise ServiceNotFoundError(f"Service {service_type.__name__} not registered")

    def clear(self) -> None:
        self._overrides.clear()
        logger.debug("DI Container cleared")


# ---------------------------------------------------------------------------
# Global container instance
# ---------------------------------------------------------------------------

_container = DIContainer()


def get_container() -> DIContainer:
    return _container


def configure_services() -> None:
    """Configure default services (legacy + Phase 2.0 subsystems)."""
    from src.data_pipeline.ingestion import DataIngestion
    from src.feature_engineering.technical_features import TechnicalFeatures
    from src.feature_engineering.microstructure_features import MicrostructureFeatures
    from src.portfolio.optimizer import PortfolioOptimizer
    from src.portfolio.risk_manager import RiskManager

    container = get_container()
    container.register_singleton(DataIngestion)
    container.register_singleton(TechnicalFeatures)
    container.register_singleton(MicrostructureFeatures)
    container.register_singleton(PortfolioOptimizer)
    container.register_singleton(RiskManager)

    # --- Phase 2.0 subsystems ---
    try:
        from src.integration import TradingSystem, SystemConfig
        system = TradingSystem.create(SystemConfig())

        # Register individual subsystem instances for direct resolution
        from src.messaging.transport import MessageTransport
        if system.hot_transport:
            container.register_instance(MessageTransport, system.hot_transport)

        from src.messaging.channels import ChannelManager
        if system.hot_channels:
            container.register_instance(ChannelManager, system.hot_channels)

        from src.risk_ops.risk_engine import PreTradeRiskEngine
        if system.risk_engine:
            container.register_instance(PreTradeRiskEngine, system.risk_engine)

        from src.risk_ops.drop_copy import DropCopyReconciler
        if system.reconciler:
            container.register_instance(DropCopyReconciler, system.reconciler)

        from src.data_platform.feature_store import (
            StreamingFeatureEngine, OnlineFeatureStore, OfflineFeatureStore,
        )
        if system.feature_engine:
            container.register_instance(StreamingFeatureEngine, system.feature_engine)
        if system.online_store:
            container.register_instance(OnlineFeatureStore, system.online_store)
        if system.offline_store:
            container.register_instance(OfflineFeatureStore, system.offline_store)

        # Register the top-level system itself
        container.register_instance(TradingSystem, system)

        logger.info("Phase 2.0 services configured")
    except Exception as e:
        logger.warning(f"Phase 2.0 subsystems not available: {e}")
    
    logger.info("Services configured")


if __name__ == "__main__":
    # Example usage
    configure_services()
    
    # Resolve service - import here to match configure_services() scope
    from src.data_pipeline.ingestion import DataIngestion  # noqa: E402
    container = get_container()
    ingestion = container.resolve(DataIngestion)
    print(f"Resolved: {ingestion}")
