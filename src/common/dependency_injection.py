"""
Dependency Injection Framework
Implements IoC container for better modularity and testability
"""

from typing import Dict, Any, Type, Optional, Callable
from dataclasses import dataclass
import inspect
from loguru import logger


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered"""
    pass


@dataclass
class ServiceDescriptor:
    """Describes a service registration"""
    service_type: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    singleton: bool = False


class DIContainer:
    """
    Dependency Injection Container
    Supports singleton and transient lifetimes
    """
    
    def __init__(self):
        """Initialize DI container"""
        self._services: Dict[Type, ServiceDescriptor] = {}
        logger.debug("DI Container initialized")
    
    def register_singleton(self, service_type: Type, implementation: Optional[Type] = None):
        """
        Register a singleton service
        
        Args:
            service_type: Interface or base class
            implementation: Concrete implementation (defaults to service_type)
        """
        impl = implementation or service_type
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=impl,
            singleton=True
        )
        logger.debug(f"Registered singleton: {service_type.__name__} -> {impl.__name__}")
    
    def register_transient(self, service_type: Type, implementation: Optional[Type] = None):
        """
        Register a transient service (new instance each time)
        
        Args:
            service_type: Interface or base class
            implementation: Concrete implementation
        """
        impl = implementation or service_type
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=impl,
            singleton=False
        )
        logger.debug(f"Registered transient: {service_type.__name__} -> {impl.__name__}")
    
    def register_instance(self, service_type: Type, instance: Any):
        """
        Register an existing instance
        
        Args:
            service_type: Service type
            instance: Pre-created instance
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            singleton=True
        )
        logger.debug(f"Registered instance: {service_type.__name__}")
    
    def register_factory(self, service_type: Type, factory: Callable, singleton: bool = False):
        """
        Register a factory function
        
        Args:
            service_type: Service type
            factory: Factory function
            singleton: Whether to cache the instance
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            singleton=singleton
        )
        logger.debug(f"Registered factory: {service_type.__name__}")
    
    def resolve(self, service_type: Type) -> Any:
        """
        Resolve a service instance
        
        Args:
            service_type: Type to resolve
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotFoundError: If service not registered
        """
        if service_type not in self._services:
            raise ServiceNotFoundError(f"Service {service_type.__name__} not registered")
        
        descriptor = self._services[service_type]
        
        # Return existing instance for singletons
        if descriptor.singleton and descriptor.instance is not None:
            return descriptor.instance
        
        # Create instance
        if descriptor.factory:
            instance = descriptor.factory(self)
        elif descriptor.implementation:
            instance = self._create_instance(descriptor.implementation)
        elif descriptor.instance:
            return descriptor.instance
        else:
            raise ServiceNotFoundError(f"No implementation for {service_type.__name__}")
        
        # Cache for singletons
        if descriptor.singleton:
            descriptor.instance = instance
        
        return instance
    
    def _create_instance(self, cls: Type) -> Any:
        """
        Create instance with automatic dependency injection
        
        Args:
            cls: Class to instantiate
            
        Returns:
            Instance with dependencies injected
        """
        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        
        # Resolve dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Try to resolve parameter type
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except ServiceNotFoundError:
                    # Use default if available
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
        
        return cls(**kwargs)
    
    def clear(self):
        """Clear all registrations"""
        self._services.clear()
        logger.debug("DI Container cleared")


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get global DI container"""
    return _container


def configure_services():
    """Configure default services"""
    from src.data_pipeline.ingestion import DataIngestion
    from src.feature_engineering.technical_features import TechnicalFeatures
    from src.feature_engineering.microstructure_features import MicrostructureFeatures
    from src.portfolio.optimizer import PortfolioOptimizer
    from src.portfolio.risk_manager import RiskManager
    
    container = get_container()
    
    # Register services
    container.register_singleton(DataIngestion)
    container.register_singleton(TechnicalFeatures)
    container.register_singleton(MicrostructureFeatures)
    container.register_singleton(PortfolioOptimizer)
    container.register_singleton(RiskManager)
    
    logger.info("Services configured")


if __name__ == "__main__":
    # Example usage
    configure_services()
    
    # Resolve service
    container = get_container()
    ingestion = container.resolve(DataIngestion)
    print(f"Resolved: {ingestion}")
