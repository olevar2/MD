"""
Adapter Factory.

This module provides a factory class for creating adapter instances,
making it easier to use the adapters in other services.
"""

import logging
from typing import Optional, Dict, Any

from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.adapters.market_data_adapter import (
    MarketDataProviderAdapter,
    MarketDataCacheAdapter
)
from common_lib.adapters.feature_store_adapter import (
    FeatureProviderAdapter,
    FeatureStoreAdapter,
    FeatureGeneratorAdapter
)
from common_lib.adapters.analysis_engine_adapter import (
    AnalysisProviderAdapter,
    IndicatorProviderAdapter,
    PatternRecognizerAdapter
)
from common_lib.adapters.trading_adapter import (
    TradingProviderAdapter,
    OrderBookProviderAdapter,
    RiskManagerAdapter
)
from common_lib.adapters.ml_integration_adapter import (
    MLModelRegistryAdapter,
    MLJobTrackerAdapter,
    MLModelDeploymentAdapter,
    MLMetricsProviderAdapter
)
from common_lib.adapters.risk_management_adapter import (
    RiskManagementAdapter
)


class AdapterFactory:
    """
    Factory class for creating adapter instances.

    This class provides methods for creating adapter instances for each service,
    making it easier to use the adapters in other services.
    """

    def __init__(
        self,
        config_provider: Optional[Dict[str, ServiceClientConfig]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the adapter factory.

        Args:
            config_provider: Dictionary mapping service names to service client configurations
            logger: Logger to use (if None, creates a new logger)
        """
        self.config_provider = config_provider or {}
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Default service names
        self.service_names = {
            "market_data": "market-data-service",
            "feature_store": "feature-store-service",
            "analysis_engine": "analysis-engine-service",
            "trading": "trading-service",
            "ml_integration": "ml-integration-service",
            "risk_management": "risk-management-service"
        }

    def get_config(self, service_name: str) -> ServiceClientConfig:
        """
        Get the service client configuration for a service.

        Args:
            service_name: Name of the service

        Returns:
            Service client configuration
        """
        if service_name in self.config_provider:
            return self.config_provider[service_name]
        else:
            # Create a default configuration
            return ServiceClientConfig(
                base_url=f"http://{service_name}:8000",
                timeout=30,
                retry_attempts=3,
                retry_backoff=1.5
            )

    # Market Data Service Adapters

    def create_market_data_provider(self) -> MarketDataProviderAdapter:
        """
        Create a Market Data Provider adapter.

        Returns:
            Market Data Provider adapter
        """
        config = self.get_config(self.service_names["market_data"])
        return MarketDataProviderAdapter(config, self.logger)

    def create_market_data_cache(self) -> MarketDataCacheAdapter:
        """
        Create a Market Data Cache adapter.

        Returns:
            Market Data Cache adapter
        """
        config = self.get_config(self.service_names["market_data"])
        return MarketDataCacheAdapter(config, self.logger)

    # Feature Store Service Adapters

    def create_feature_provider(self) -> FeatureProviderAdapter:
        """
        Create a Feature Provider adapter.

        Returns:
            Feature Provider adapter
        """
        config = self.get_config(self.service_names["feature_store"])
        return FeatureProviderAdapter(config, self.logger)

    def create_feature_store(self) -> FeatureStoreAdapter:
        """
        Create a Feature Store adapter.

        Returns:
            Feature Store adapter
        """
        config = self.get_config(self.service_names["feature_store"])
        return FeatureStoreAdapter(config, self.logger)

    def create_feature_generator(self) -> FeatureGeneratorAdapter:
        """
        Create a Feature Generator adapter.

        Returns:
            Feature Generator adapter
        """
        config = self.get_config(self.service_names["feature_store"])
        return FeatureGeneratorAdapter(config, self.logger)

    # Analysis Engine Service Adapters

    def create_analysis_provider(self) -> AnalysisProviderAdapter:
        """
        Create an Analysis Provider adapter.

        Returns:
            Analysis Provider adapter
        """
        config = self.get_config(self.service_names["analysis_engine"])
        return AnalysisProviderAdapter(config, self.logger)

    def create_indicator_provider(self) -> IndicatorProviderAdapter:
        """
        Create an Indicator Provider adapter.

        Returns:
            Indicator Provider adapter
        """
        config = self.get_config(self.service_names["analysis_engine"])
        return IndicatorProviderAdapter(config, self.logger)

    def create_pattern_recognizer(self) -> PatternRecognizerAdapter:
        """
        Create a Pattern Recognizer adapter.

        Returns:
            Pattern Recognizer adapter
        """
        config = self.get_config(self.service_names["analysis_engine"])
        return PatternRecognizerAdapter(config, self.logger)

    # Trading Service Adapters

    def create_trading_provider(self) -> TradingProviderAdapter:
        """
        Create a Trading Provider adapter.

        Returns:
            Trading Provider adapter
        """
        config = self.get_config(self.service_names["trading"])
        return TradingProviderAdapter(config, self.logger)

    def create_order_book_provider(self) -> OrderBookProviderAdapter:
        """
        Create an Order Book Provider adapter.

        Returns:
            Order Book Provider adapter
        """
        config = self.get_config(self.service_names["trading"])
        return OrderBookProviderAdapter(config, self.logger)

    def create_risk_manager(self) -> RiskManagerAdapter:
        """
        Create a Risk Manager adapter.

        Returns:
            Risk Manager adapter
        """
        config = self.get_config(self.service_names["trading"])
        return RiskManagerAdapter(config, self.logger)

    # ML Integration Service Adapters

    def create_ml_model_registry(self) -> MLModelRegistryAdapter:
        """
        Create an ML Model Registry adapter.

        Returns:
            ML Model Registry adapter
        """
        config = self.get_config(self.service_names["ml_integration"])
        return MLModelRegistryAdapter(config=config)

    def create_ml_job_tracker(self) -> MLJobTrackerAdapter:
        """
        Create an ML Job Tracker adapter.

        Returns:
            ML Job Tracker adapter
        """
        config = self.get_config(self.service_names["ml_integration"])
        return MLJobTrackerAdapter(config=config)

    def create_ml_model_deployment(self) -> MLModelDeploymentAdapter:
        """
        Create an ML Model Deployment adapter.

        Returns:
            ML Model Deployment adapter
        """
        config = self.get_config(self.service_names["ml_integration"])
        return MLModelDeploymentAdapter(config=config)

    def create_ml_metrics_provider(self) -> MLMetricsProviderAdapter:
        """
        Create an ML Metrics Provider adapter.

        Returns:
            ML Metrics Provider adapter
        """
        config = self.get_config(self.service_names["ml_integration"])
        return MLMetricsProviderAdapter(config=config)

    # Risk Management Service Adapters

    def create_risk_management(self) -> RiskManagementAdapter:
        """
        Create a Risk Management adapter.

        Returns:
            Risk Management adapter
        """
        config = self.get_config(self.service_names["risk_management"])
        return RiskManagementAdapter(config=config)