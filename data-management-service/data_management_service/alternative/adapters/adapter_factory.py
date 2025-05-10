"""
Alternative Data Adapter Factory.

This module provides a factory for creating alternative data adapters.
"""
import logging
from typing import Any, Dict, List, Optional, Type, Union

from data_management_service.alternative.adapters.base_adapter import BaseAlternativeDataAdapter
from data_management_service.alternative.adapters.news_adapter import NewsDataAdapter, MockNewsDataAdapter
from data_management_service.alternative.adapters.economic_adapter import EconomicDataAdapter, MockEconomicDataAdapter
from data_management_service.alternative.adapters.sentiment_adapter import SentimentDataAdapter, MockSentimentDataAdapter
from data_management_service.alternative.adapters.social_media_adapter import SocialMediaDataAdapter, MockSocialMediaDataAdapter
from data_management_service.alternative.models import AlternativeDataType

logger = logging.getLogger(__name__)


class AdapterFactory:
    """Factory for creating alternative data adapters."""

    def __init__(self):
        """Initialize the adapter factory."""
        self.adapter_registry = {}
        self.mock_adapter_registry = {}
        self._register_adapters()

    def _register_adapters(self) -> None:
        """Register all available adapters."""
        # Register real adapters
        self.adapter_registry[AlternativeDataType.NEWS] = NewsDataAdapter
        self.adapter_registry[AlternativeDataType.ECONOMIC] = EconomicDataAdapter
        self.adapter_registry[AlternativeDataType.SENTIMENT] = SentimentDataAdapter
        self.adapter_registry[AlternativeDataType.SOCIAL_MEDIA] = SocialMediaDataAdapter
        
        # Register mock adapters
        self.mock_adapter_registry[AlternativeDataType.NEWS] = MockNewsDataAdapter
        self.mock_adapter_registry[AlternativeDataType.ECONOMIC] = MockEconomicDataAdapter
        self.mock_adapter_registry[AlternativeDataType.SENTIMENT] = MockSentimentDataAdapter
        self.mock_adapter_registry[AlternativeDataType.SOCIAL_MEDIA] = MockSocialMediaDataAdapter

    def create_adapter(
        self,
        data_type: str,
        config: Dict[str, Any],
        use_mock: bool = False
    ) -> BaseAlternativeDataAdapter:
        """
        Create an adapter for the specified data type.

        Args:
            data_type: Type of alternative data
            config: Configuration for the adapter
            use_mock: Whether to use a mock adapter

        Returns:
            Adapter instance
        """
        # Convert string to enum if needed
        if isinstance(data_type, str):
            try:
                data_type = AlternativeDataType(data_type)
            except ValueError:
                raise ValueError(f"Unknown data type: {data_type}")
        
        # Get adapter class
        if use_mock:
            adapter_class = self.mock_adapter_registry.get(data_type)
            if not adapter_class:
                raise ValueError(f"No mock adapter available for data type: {data_type}")
        else:
            adapter_class = self.adapter_registry.get(data_type)
            if not adapter_class:
                raise ValueError(f"No adapter available for data type: {data_type}")
        
        # Create adapter instance
        try:
            adapter = adapter_class(config)
            logger.info(f"Created {'mock ' if use_mock else ''}adapter for data type: {data_type}")
            return adapter
        except Exception as e:
            logger.error(f"Error creating adapter for data type {data_type}: {str(e)}")
            raise


class MultiSourceAdapterFactory:
    """Factory for creating multi-source alternative data adapters."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-source adapter factory.

        Args:
            config: Configuration for the factory
        """
        self.config = config
        self.adapter_factory = AdapterFactory()
        self.adapters = {}
        self._initialize_adapters()

    def _initialize_adapters(self) -> None:
        """Initialize adapters based on configuration."""
        adapter_configs = self.config.get("adapters", {})
        
        for data_type, configs in adapter_configs.items():
            if not isinstance(configs, list):
                configs = [configs]
            
            for config in configs:
                use_mock = config.get("use_mock", False)
                adapter_id = config.get("id", f"{data_type}_{len(self.adapters)}")
                
                try:
                    adapter = self.adapter_factory.create_adapter(data_type, config, use_mock)
                    self.adapters[adapter_id] = adapter
                    logger.info(f"Initialized adapter {adapter_id} for data type {data_type}")
                except Exception as e:
                    logger.error(f"Error initializing adapter {adapter_id} for data type {data_type}: {str(e)}")

    def get_adapter(self, adapter_id: str) -> BaseAlternativeDataAdapter:
        """
        Get an adapter by ID.

        Args:
            adapter_id: Adapter ID

        Returns:
            Adapter instance
        """
        if adapter_id not in self.adapters:
            raise ValueError(f"Unknown adapter ID: {adapter_id}")
        
        return self.adapters[adapter_id]

    def get_adapters_by_type(self, data_type: str) -> List[BaseAlternativeDataAdapter]:
        """
        Get all adapters for a specific data type.

        Args:
            data_type: Type of alternative data

        Returns:
            List of adapter instances
        """
        # Convert string to enum if needed
        if isinstance(data_type, str):
            try:
                data_type = AlternativeDataType(data_type)
            except ValueError:
                raise ValueError(f"Unknown data type: {data_type}")
        
        return [adapter for adapter in self.adapters.values() 
                if data_type in adapter.get_available_data_types()]

    def get_all_adapters(self) -> Dict[str, BaseAlternativeDataAdapter]:
        """
        Get all adapters.

        Returns:
            Dictionary of adapter instances
        """
        return self.adapters
