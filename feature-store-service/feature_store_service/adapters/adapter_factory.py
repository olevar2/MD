"""
Adapter Factory Module

This module provides a factory for creating adapter instances for various services.
"""

import logging
from typing import Optional, Dict, Any, Type, TypeVar, cast

from common_lib.interfaces.feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer
from common_lib.interfaces.market_data import IMarketDataProvider, IMarketDataCache

from feature_store_service.adapters.service_adapters import (
    FeatureProviderAdapter,
    FeatureStoreAdapter,
    FeatureGeneratorAdapter
)
from feature_store_service.adapters.analysis_engine_adapter import AnalysisEngineAdapter
from feature_store_service.core.logging import get_logger

# Type variable for generic adapter types
T = TypeVar('T')

# Configure logging
logger = get_logger("feature-store-service.adapter-factory")


class AdapterFactory:
    """
    Factory for creating adapter instances for various services.

    This factory provides methods to create and retrieve adapter instances
    for different service interfaces, helping to break circular dependencies
    between services.
    """

    _instance = None

    def __new__(cls):
        """
        Create a new instance of the AdapterFactory or return the existing instance.

        Returns:
            AdapterFactory instance
        """
        if cls._instance is None:
            cls._instance = super(AdapterFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the AdapterFactory.
        """
        if self._initialized:
            return

        # Initialize adapter instances
        self._feature_provider = None
        self._feature_store = None
        self._feature_generator = None
        self._analysis_provider = None
        self._indicator_provider = None
        self._pattern_recognizer = None

        # Mark as initialized
        self._initialized = True
        logger.info("AdapterFactory initialized")

    def initialize(self):
        """
        Initialize adapter instances.

        This method creates adapter instances for all supported interfaces.
        """
        # Create adapter instances
        self._feature_provider = FeatureProviderAdapter(logger=logger)
        self._feature_store = FeatureStoreAdapter(logger=logger)
        self._feature_generator = FeatureGeneratorAdapter(logger=logger)

        # Create analysis engine adapter instance
        self._analysis_provider = AnalysisEngineAdapter(logger=logger)
        # Use the same adapter instance for all analysis engine interfaces
        self._indicator_provider = self._analysis_provider
        self._pattern_recognizer = self._analysis_provider

        logger.info("AdapterFactory adapters initialized")

    def cleanup(self):
        """
        Clean up adapter instances.

        This method cleans up any resources used by the adapter instances.
        """
        # Clean up adapter instances
        self._feature_provider = None
        self._feature_store = None
        self._feature_generator = None
        self._analysis_provider = None
        self._indicator_provider = None
        self._pattern_recognizer = None

        logger.info("AdapterFactory adapters cleaned up")

    def get_feature_provider(self) -> FeatureProviderAdapter:
        """
        Get the Feature Provider adapter.

        Returns:
            Feature Provider adapter
        """
        if self._feature_provider is None:
            self._feature_provider = FeatureProviderAdapter(logger=logger)
        return self._feature_provider

    def get_feature_store(self) -> FeatureStoreAdapter:
        """
        Get the Feature Store adapter.

        Returns:
            Feature Store adapter
        """
        if self._feature_store is None:
            self._feature_store = FeatureStoreAdapter(logger=logger)
        return self._feature_store

    def get_feature_generator(self) -> FeatureGeneratorAdapter:
        """
        Get the Feature Generator adapter.

        Returns:
            Feature Generator adapter
        """
        if self._feature_generator is None:
            self._feature_generator = FeatureGeneratorAdapter(logger=logger)
        return self._feature_generator

    def get_analysis_provider(self) -> AnalysisEngineAdapter:
        """
        Get the Analysis Provider adapter.

        Returns:
            Analysis Provider adapter
        """
        if self._analysis_provider is None:
            self._analysis_provider = AnalysisEngineAdapter(logger=logger)
            # Use the same adapter instance for all analysis engine interfaces
            self._indicator_provider = self._analysis_provider
            self._pattern_recognizer = self._analysis_provider
        return self._analysis_provider

    def get_indicator_provider(self) -> AnalysisEngineAdapter:
        """
        Get the Indicator Provider adapter.

        Returns:
            Indicator Provider adapter
        """
        if self._indicator_provider is None:
            self._analysis_provider = AnalysisEngineAdapter(logger=logger)
            # Use the same adapter instance for all analysis engine interfaces
            self._indicator_provider = self._analysis_provider
            self._pattern_recognizer = self._analysis_provider
        return self._indicator_provider

    def get_pattern_recognizer(self) -> AnalysisEngineAdapter:
        """
        Get the Pattern Recognizer adapter.

        Returns:
            Pattern Recognizer adapter
        """
        if self._pattern_recognizer is None:
            self._analysis_provider = AnalysisEngineAdapter(logger=logger)
            # Use the same adapter instance for all analysis engine interfaces
            self._indicator_provider = self._analysis_provider
            self._pattern_recognizer = self._analysis_provider
        return self._pattern_recognizer

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
        if interface_type == IFeatureProvider:
            return cast(T, self.get_feature_provider())
        elif interface_type == IFeatureStore:
            return cast(T, self.get_feature_store())
        elif interface_type == IFeatureGenerator:
            return cast(T, self.get_feature_generator())
        elif interface_type == IAnalysisProvider:
            return cast(T, self.get_analysis_provider())
        elif interface_type == IIndicatorProvider:
            return cast(T, self.get_indicator_provider())
        elif interface_type == IPatternRecognizer:
            return cast(T, self.get_pattern_recognizer())
        # Add more interface types as they are implemented
        else:
            raise ValueError(f"No adapter available for interface type: {interface_type.__name__}")


# Singleton instance
adapter_factory = AdapterFactory()
