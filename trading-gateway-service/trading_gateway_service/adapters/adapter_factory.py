"""
Adapter Factory Module

This module provides a factory for creating adapter instances for various services.
"""

import logging
from typing import Optional, Dict, Any, Type, TypeVar, cast

from common_lib.interfaces.trading import ITradingProvider, IOrderBookProvider
from common_lib.interfaces.risk_management import IRiskManager
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer

from trading_gateway_service.adapters.trading_provider_adapter import TradingProviderAdapter
from trading_gateway_service.adapters.order_book_adapter import OrderBookProviderAdapter
from trading_gateway_service.adapters.risk_management_adapter import TradingRiskManagementAdapter
from trading_gateway_service.adapters.analysis_engine_adapter import AnalysisEngineAdapter
from trading_gateway_service.core.logging import get_logger

# Type variable for generic adapter types
T = TypeVar('T')

# Configure logging
logger = get_logger("trading-gateway-service.adapter-factory")


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
        self._trading_provider = None
        self._order_book_provider = None
        self._risk_manager = None
        self._analysis_provider = None

        # Mark as initialized
        self._initialized = True
        logger.info("AdapterFactory initialized")

    def initialize(self):
        """
        Initialize adapter instances.

        This method creates adapter instances for all supported interfaces.
        """
        # Create adapter instances
        self._trading_provider = TradingProviderAdapter(logger=logger)
        self._order_book_provider = OrderBookProviderAdapter(logger=logger)
        self._risk_manager = TradingRiskManagementAdapter()
        self._analysis_provider = AnalysisEngineAdapter(logger=logger)

        logger.info("AdapterFactory adapters initialized")

    def cleanup(self):
        """
        Clean up adapter instances.

        This method cleans up any resources used by the adapter instances.
        """
        # Clean up adapter instances
        self._trading_provider = None
        self._order_book_provider = None
        self._risk_manager = None
        self._analysis_provider = None

        logger.info("AdapterFactory adapters cleaned up")

    def get_trading_provider(self) -> TradingProviderAdapter:
        """
        Get the Trading Provider adapter.

        Returns:
            Trading Provider adapter
        """
        if self._trading_provider is None:
            self._trading_provider = TradingProviderAdapter(logger=logger)
        return self._trading_provider

    def get_order_book_provider(self) -> OrderBookProviderAdapter:
        """
        Get the Order Book Provider adapter.

        Returns:
            Order Book Provider adapter
        """
        if self._order_book_provider is None:
            self._order_book_provider = OrderBookProviderAdapter(logger=logger)
        return self._order_book_provider

    def get_risk_manager(self) -> TradingRiskManagementAdapter:
        """
        Get the Risk Manager adapter.

        Returns:
            Risk Manager adapter
        """
        if self._risk_manager is None:
            self._risk_manager = TradingRiskManagementAdapter()
        return self._risk_manager

    def get_analysis_provider(self) -> AnalysisEngineAdapter:
        """
        Get the Analysis Provider adapter.

        Returns:
            Analysis Provider adapter
        """
        if self._analysis_provider is None:
            self._analysis_provider = AnalysisEngineAdapter(logger=logger)
        return self._analysis_provider

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
        if interface_type == ITradingProvider:
            return cast(T, self.get_trading_provider())
        elif interface_type == IOrderBookProvider:
            return cast(T, self.get_order_book_provider())
        elif interface_type == IRiskManager:
            return cast(T, self.get_risk_manager())
        elif interface_type == IAnalysisProvider:
            return cast(T, self.get_analysis_provider())
        elif interface_type == IIndicatorProvider:
            return cast(T, self.get_analysis_provider())
        elif interface_type == IPatternRecognizer:
            return cast(T, self.get_analysis_provider())
        # Add more interface types as they are implemented
        else:
            raise ValueError(f"No adapter available for interface type: {interface_type.__name__}")


# Singleton instance
adapter_factory = AdapterFactory()
