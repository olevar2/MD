"""
Adapter Factory

This module provides a factory for creating adapters for external services.
"""
from typing import Dict, Any, Type, TypeVar, cast
from common_lib.interfaces.data_interfaces import IFeatureProvider, IDataPipeline
from common_lib.interfaces.ml_interfaces import IModelProvider, IModelRegistry
from common_lib.interfaces.trading_interfaces import IRiskManager, ITradingGateway
from analysis_engine.adapters.data_adapters import FeatureStoreAdapter, DataPipelineAdapter
from analysis_engine.adapters.ml_adapters import MLWorkbenchAdapter, ModelRegistryAdapter
from analysis_engine.adapters.trading_adapters import RiskManagementAdapter, TradingGatewayAdapter
T = TypeVar('T')


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AdapterFactory:
    """Factory for creating adapters for external services."""
    _instance = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(AdapterFactory, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the factory."""
        self._adapters = {}

    @with_resilience('get_adapter')
    def get_adapter(self, interface_type: Type[T]) ->T:
        """
        Get an adapter for the specified interface type.
        
        Args:
            interface_type: Type of the interface
            
        Returns:
            Adapter instance
        """
        if interface_type not in self._adapters:
            self._adapters[interface_type] = self._create_adapter(
                interface_type)
        return cast(T, self._adapters[interface_type])

    def _create_adapter(self, interface_type: Type[T]) ->T:
        """
        Create an adapter for the specified interface type.
        
        Args:
            interface_type: Type of the interface
            
        Returns:
            Adapter instance
        """
        if interface_type == IFeatureProvider:
            return FeatureStoreAdapter()
        elif interface_type == IDataPipeline:
            return DataPipelineAdapter()
        elif interface_type == IModelProvider:
            return MLWorkbenchAdapter()
        elif interface_type == IModelRegistry:
            return ModelRegistryAdapter()
        elif interface_type == IRiskManager:
            return RiskManagementAdapter()
        elif interface_type == ITradingGateway:
            return TradingGatewayAdapter()
        else:
            raise ValueError(
                f'No adapter available for interface type {interface_type}')


adapter_factory = AdapterFactory()
