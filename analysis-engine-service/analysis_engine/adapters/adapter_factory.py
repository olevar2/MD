"""
Adapter Factory Module

This module provides a factory for creating adapter instances for various services.
"""

import logging
from typing import Optional, Dict, Any, Type, TypeVar, cast

from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer

from analysis_engine.adapters.analysis_adapter import AnalysisProviderAdapter
from analysis_engine.adapters.indicator_adapter import IndicatorProviderAdapter
from analysis_engine.adapters.pattern_adapter import PatternRecognizerAdapter
from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.services.indicator_service import IndicatorService
from analysis_engine.services.pattern_service import PatternService

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
    
    def get_analysis_provider(self) -> IAnalysisProvider:
        """
        Get an instance of IAnalysisProvider.
        
        Returns:
            An instance of IAnalysisProvider
        """
        if IAnalysisProvider not in self._adapters:
            self._adapters[IAnalysisProvider] = AnalysisProviderAdapter()
        return cast(IAnalysisProvider, self._adapters[IAnalysisProvider])
    
    def get_indicator_provider(self) -> IIndicatorProvider:
        """
        Get an instance of IIndicatorProvider.
        
        Returns:
            An instance of IIndicatorProvider
        """
        if IIndicatorProvider not in self._adapters:
            self._adapters[IIndicatorProvider] = IndicatorProviderAdapter()
        return cast(IIndicatorProvider, self._adapters[IIndicatorProvider])
    
    def get_pattern_recognizer(self) -> IPatternRecognizer:
        """
        Get an instance of IPatternRecognizer.
        
        Returns:
            An instance of IPatternRecognizer
        """
        if IPatternRecognizer not in self._adapters:
            self._adapters[IPatternRecognizer] = PatternRecognizerAdapter()
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
        if interface_type == IAnalysisProvider:
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
