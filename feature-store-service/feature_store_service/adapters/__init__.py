"""
Adapters package for feature store service.

This package contains adapter implementations for interfaces
to break circular dependencies between services.
"""

from .ml_integration_adapter import MLFeatureConsumerAdapter
from .advanced_indicator_adapter import (
    AdvancedIndicatorAdapter,
    FibonacciAnalyzerAdapter,
    load_advanced_indicators
)
from .fibonacci_adapter import (
    FibonacciBaseAdapter,
    FibonacciRetracementAdapter,
    FibonacciExtensionAdapter,
    FibonacciFanAdapter,
    FibonacciTimeZonesAdapter,
    FibonacciCirclesAdapter,
    FibonacciClustersAdapter,
    FibonacciUtilsAdapter
)
from .service_adapters import (
    FeatureProviderAdapter,
    FeatureStoreAdapter,
    FeatureGeneratorAdapter
)
from .analysis_engine_adapter import AnalysisEngineAdapter

__all__ = [
    'MLFeatureConsumerAdapter',
    'AdvancedIndicatorAdapter',
    'FibonacciAnalyzerAdapter',
    'load_advanced_indicators',
    # Fibonacci adapters
    'FibonacciBaseAdapter',
    'FibonacciRetracementAdapter',
    'FibonacciExtensionAdapter',
    'FibonacciFanAdapter',
    'FibonacciTimeZonesAdapter',
    'FibonacciCirclesAdapter',
    'FibonacciClustersAdapter',
    'FibonacciUtilsAdapter',
    # Service adapters
    'FeatureProviderAdapter',
    'FeatureStoreAdapter',
    'FeatureGeneratorAdapter',
    'AnalysisEngineAdapter'
]