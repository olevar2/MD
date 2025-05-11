"""
Adapters package for the forex trading platform.

This package provides adapter implementations for the service interfaces,
allowing services to interact with each other without direct dependencies,
breaking circular dependencies.
"""

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

from common_lib.adapters.factory import AdapterFactory

__all__ = [
    # Market Data Adapters
    'MarketDataProviderAdapter',
    'MarketDataCacheAdapter',

    # Feature Store Adapters
    'FeatureProviderAdapter',
    'FeatureStoreAdapter',
    'FeatureGeneratorAdapter',

    # Analysis Engine Adapters
    'AnalysisProviderAdapter',
    'IndicatorProviderAdapter',
    'PatternRecognizerAdapter',

    # Trading Adapters
    'TradingProviderAdapter',
    'OrderBookProviderAdapter',
    'RiskManagerAdapter',

    # ML Integration Adapters
    'MLModelRegistryAdapter',
    'MLJobTrackerAdapter',
    'MLModelDeploymentAdapter',
    'MLMetricsProviderAdapter',

    # Risk Management Adapters
    'RiskManagementAdapter',

    # Factory
    'AdapterFactory'
]