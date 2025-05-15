"""
Common Adapter Factory for Analysis Engine Service.

This module provides a factory for creating adapters to interact with other services
using the common interfaces defined in common-lib. This helps to break circular dependencies
between services.
"""

import logging
from typing import Dict, Any, Type, TypeVar, Optional, cast

from common_lib.interfaces.trading_gateway import ITradingGateway
from common_lib.interfaces.ml_integration import IMLModelRegistry, IMLJobTracker, IMLModelDeployment, IMLMetricsProvider
from common_lib.interfaces.ml_workbench import IExperimentManager, IModelEvaluator, IDatasetManager
from common_lib.interfaces.risk_management import IRiskManager
from common_lib.interfaces.feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer

from common_lib.adapters.trading_gateway_adapter import TradingGatewayAdapter
from common_lib.adapters.ml_workbench_adapter import MLWorkbenchAdapter
from common_lib.adapters.risk_management_adapter import RiskManagementAdapter
from common_lib.adapters.feature_store_adapter import FeatureStoreAdapter

from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.config.service_config import ServiceConfig


logger = logging.getLogger(__name__)

T = TypeVar('T')


class CommonAdapterFactory:
    """Factory for creating service adapters using common interfaces."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter factory.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.service_config = ServiceConfig()
        self.adapters: Dict[Type, Any] = {}
        self.logger = logger
    
    def get_adapter(self, interface_type: Type[T]) -> T:
        """
        Get an adapter for the specified interface type.
        
        Args:
            interface_type: The interface type to get an adapter for
            
        Returns:
            An adapter implementing the specified interface
        """
        # Check if we already have an adapter for this interface
        if interface_type in self.adapters:
            return cast(T, self.adapters[interface_type])
        
        # Create a new adapter based on the interface type
        adapter = self._create_adapter(interface_type)
        self.adapters[interface_type] = adapter
        return adapter
    
    def _create_adapter(self, interface_type: Type[T]) -> T:
        """
        Create an adapter for the specified interface type.
        
        Args:
            interface_type: The interface type to create an adapter for
            
        Returns:
            An adapter implementing the specified interface
        """
        # Trading Gateway interfaces
        if issubclass(interface_type, ITradingGateway):
            return cast(T, self._create_trading_gateway_adapter())
        
        # ML Integration interfaces
        if issubclass(interface_type, (IMLModelRegistry, IMLJobTracker, IMLModelDeployment, IMLMetricsProvider)):
            return cast(T, self._create_ml_integration_adapter())
        
        # ML Workbench interfaces
        if issubclass(interface_type, (IExperimentManager, IModelEvaluator, IDatasetManager)):
            return cast(T, self._create_ml_workbench_adapter())
        
        # Risk Management interfaces
        if issubclass(interface_type, IRiskManager):
            return cast(T, self._create_risk_management_adapter())
        
        # Feature Store interfaces
        if issubclass(interface_type, (IFeatureProvider, IFeatureStore, IFeatureGenerator)):
            return cast(T, self._create_feature_store_adapter())
        
        # Analysis Engine interfaces (for self-reference or testing)
        if issubclass(interface_type, (IAnalysisProvider, IIndicatorProvider, IPatternRecognizer)):
            return cast(T, self._create_analysis_engine_adapter())
        
        raise ValueError(f"No adapter available for interface type: {interface_type.__name__}")
    
    def _create_trading_gateway_adapter(self) -> TradingGatewayAdapter:
        """
        Create an adapter for the Trading Gateway Service.
        
        Returns:
            A Trading Gateway adapter
        """
        config = self.service_config.get_service_config("trading-gateway-service")
        client_config = ServiceClientConfig(
            base_url=config.get("base_url", "http://trading-gateway-service:8000/api/v1"),
            timeout=config.get("timeout", 30),
            retry_count=config.get("retry_count", 3),
            retry_backoff=config.get("retry_backoff", 1.0),
            circuit_breaker_threshold=config.get("circuit_breaker_threshold", 5),
            circuit_breaker_recovery_time=config.get("circuit_breaker_recovery_time", 30)
        )
        return TradingGatewayAdapter(client_config)
    
    def _create_ml_integration_adapter(self) -> Any:  # Using Any to avoid import errors
        """
        Create an adapter for the ML Integration Service.
        
        Returns:
            An ML Integration adapter
        """
        config = self.service_config.get_service_config("ml-integration-service")
        client_config = ServiceClientConfig(
            base_url=config.get("base_url", "http://ml-integration-service:8000/api/v1"),
            timeout=config.get("timeout", 30),
            retry_count=config.get("retry_count", 3),
            retry_backoff=config.get("retry_backoff", 1.0),
            circuit_breaker_threshold=config.get("circuit_breaker_threshold", 5),
            circuit_breaker_recovery_time=config.get("circuit_breaker_recovery_time", 30)
        )
        
        # Import here to avoid circular imports
        from common_lib.adapters.ml_integration_adapter import MLIntegrationAdapter
        return MLIntegrationAdapter(client_config)
    
    def _create_ml_workbench_adapter(self) -> MLWorkbenchAdapter:
        """
        Create an adapter for the ML Workbench Service.
        
        Returns:
            An ML Workbench adapter
        """
        config = self.service_config.get_service_config("ml-workbench-service")
        client_config = ServiceClientConfig(
            base_url=config.get("base_url", "http://ml-workbench-service:8000/api/v1"),
            timeout=config.get("timeout", 30),
            retry_count=config.get("retry_count", 3),
            retry_backoff=config.get("retry_backoff", 1.0),
            circuit_breaker_threshold=config.get("circuit_breaker_threshold", 5),
            circuit_breaker_recovery_time=config.get("circuit_breaker_recovery_time", 30)
        )
        return MLWorkbenchAdapter(client_config)
    
    def _create_risk_management_adapter(self) -> RiskManagementAdapter:
        """
        Create an adapter for the Risk Management Service.
        
        Returns:
            A Risk Management adapter
        """
        config = self.service_config.get_service_config("risk-management-service")
        client_config = ServiceClientConfig(
            base_url=config.get("base_url", "http://risk-management-service:8000/api/v1"),
            timeout=config.get("timeout", 30),
            retry_count=config.get("retry_count", 3),
            retry_backoff=config.get("retry_backoff", 1.0),
            circuit_breaker_threshold=config.get("circuit_breaker_threshold", 5),
            circuit_breaker_recovery_time=config.get("circuit_breaker_recovery_time", 30)
        )
        return RiskManagementAdapter(client_config)
    
    def _create_feature_store_adapter(self) -> FeatureStoreAdapter:
        """
        Create an adapter for the Feature Store Service.
        
        Returns:
            A Feature Store adapter
        """
        config = self.service_config.get_service_config("feature-store-service")
        client_config = ServiceClientConfig(
            base_url=config.get("base_url", "http://feature-store-service:8000/api/v1"),
            timeout=config.get("timeout", 30),
            retry_count=config.get("retry_count", 3),
            retry_backoff=config.get("retry_backoff", 1.0),
            circuit_breaker_threshold=config.get("circuit_breaker_threshold", 5),
            circuit_breaker_recovery_time=config.get("circuit_breaker_recovery_time", 30)
        )
        return FeatureStoreAdapter(client_config)
    
    def _create_analysis_engine_adapter(self) -> Any:  # Using Any to avoid import errors
        """
        Create an adapter for the Analysis Engine Service (self-reference).
        
        Returns:
            An Analysis Engine adapter
        """
        # This is used for testing or self-reference
        # Import here to avoid circular imports
        from analysis_engine.adapters.analysis_adapter import AnalysisAdapter
        return AnalysisAdapter()


# Singleton instance of the adapter factory
common_adapter_factory = CommonAdapterFactory()


def get_common_adapter_factory() -> CommonAdapterFactory:
    """
    Get the singleton instance of the common adapter factory.
    
    Returns:
        The common adapter factory instance
    """
    return common_adapter_factory
