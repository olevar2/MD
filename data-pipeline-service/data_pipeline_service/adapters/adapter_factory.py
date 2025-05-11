"""
Adapter Factory Module

This module provides a factory for creating adapter instances for various services.
"""

import logging
from typing import Optional, Dict, Any, Type, TypeVar, cast

from common_lib.interfaces.market_data import IMarketDataProvider, IMarketDataCache
from common_lib.interfaces.feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer
from common_lib.interfaces.trading import ITradingProvider, IOrderBookProvider
from common_lib.interfaces.risk_management import IRiskManager

from data_pipeline_service.adapters.market_data_adapter import MarketDataProviderAdapter, MarketDataCacheAdapter
from data_pipeline_service.adapters.analysis_engine_adapter import AnalysisEngineAdapter
from data_pipeline_service.services.market_data_service import MarketDataService
from data_pipeline_service.caching.market_data_cache import MarketDataCache

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for interface types
T = TypeVar('T')


class AdapterFactory:
    """
    Factory for creating adapter instances for various services.

    This factory provides methods for creating adapter instances that implement
    the standardized interfaces defined in common-lib, enabling better service
    integration and reducing circular dependencies.
    """

    _instance: Optional['AdapterFactory'] = None
    _adapters: Dict[Type, Any] = {}

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(AdapterFactory, cls).__new__(cls)
            cls._instance._adapters = {}
        return cls._instance

    def get_market_data_provider(self) -> IMarketDataProvider:
        """
        Get an instance of IMarketDataProvider.

        Returns:
            An instance of IMarketDataProvider
        """
        if IMarketDataProvider not in self._adapters:
            self._adapters[IMarketDataProvider] = MarketDataProviderAdapter()
        return cast(IMarketDataProvider, self._adapters[IMarketDataProvider])

    def get_market_data_cache(self) -> IMarketDataCache:
        """
        Get an instance of IMarketDataCache.

        Returns:
            An instance of IMarketDataCache
        """
        if IMarketDataCache not in self._adapters:
            self._adapters[IMarketDataCache] = MarketDataCacheAdapter()
        return cast(IMarketDataCache, self._adapters[IMarketDataCache])

    def get_analysis_provider(self) -> IAnalysisProvider:
        """
        Get an instance of IAnalysisProvider.

        Returns:
            An instance of IAnalysisProvider
        """
        if IAnalysisProvider not in self._adapters:
            self._adapters[IAnalysisProvider] = AnalysisEngineAdapter()
        return cast(IAnalysisProvider, self._adapters[IAnalysisProvider])

    def get_indicator_provider(self) -> IIndicatorProvider:
        """
        Get an instance of IIndicatorProvider.

        Returns:
            An instance of IIndicatorProvider
        """
        if IIndicatorProvider not in self._adapters:
            # Use the same adapter instance for all analysis engine interfaces
            self._adapters[IIndicatorProvider] = self.get_analysis_provider()
        return cast(IIndicatorProvider, self._adapters[IIndicatorProvider])

    def get_pattern_recognizer(self) -> IPatternRecognizer:
        """
        Get an instance of IPatternRecognizer.

        Returns:
            An instance of IPatternRecognizer
        """
        if IPatternRecognizer not in self._adapters:
            # Use the same adapter instance for all analysis engine interfaces
            self._adapters[IPatternRecognizer] = self.get_analysis_provider()
        return cast(IPatternRecognizer, self._adapters[IPatternRecognizer])

    def get_adapter(self, interface_type: Type[T]) -> T:
        """
        Get an adapter instance for the specified interface type.

        Args:
            interface_type: The interface type to get an adapter for

        Returns:
            An instance of the specified interface type

        Raises:
            ValueError: If no adapter is available for the specified interface type
        """
        if interface_type == IMarketDataProvider:
            return cast(T, self.get_market_data_provider())
        elif interface_type == IMarketDataCache:
            return cast(T, self.get_market_data_cache())
        elif interface_type == IAnalysisProvider:
            return cast(T, self.get_analysis_provider())
        elif interface_type == IIndicatorProvider:
            return cast(T, self.get_indicator_provider())
        elif interface_type == IPatternRecognizer:
            return cast(T, self.get_pattern_recognizer())
        # Add more interface types as they are implemented
        else:
            raise ValueError(f"No adapter available for interface type: {interface_type.__name__}")

    def register_adapter(self, interface_type: Type, adapter_instance: Any) -> None:
        """
        Register an adapter instance for the specified interface type.

        Args:
            interface_type: The interface type to register an adapter for
            adapter_instance: The adapter instance to register
        """
        self._adapters[interface_type] = adapter_instance
        logger.info(f"Registered adapter for interface type: {interface_type.__name__}")

    def clear_adapters(self) -> None:
        """Clear all registered adapters."""
        self._adapters.clear()
        logger.info("Cleared all registered adapters")


# Create a singleton instance
adapter_factory = AdapterFactory()
