
"""Interface definitions for service integration"""

# Market Data interfaces
from .market_data import IMarketDataProvider, IMarketDataCache

# Feature Store interfaces
from .feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator

# Analysis Engine interfaces
from .analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer

# Trading interfaces
from .trading import ITradingProvider, IOrderBookProvider, OrderType, OrderSide, OrderStatus

# ML Integration interfaces
from .ml_integration import (
    IMLModelRegistry, IMLJobTracker, IMLModelDeployment, IMLMetricsProvider,
    ModelStatus, ModelType, ModelFramework
)

# Risk Management interfaces
from .risk_management import (
    IRiskManager, RiskLimitType, RiskCheckResult
)

__all__ = [
    # Market Data interfaces
    'IMarketDataProvider',
    'IMarketDataCache',

    # Feature Store interfaces
    'IFeatureProvider',
    'IFeatureStore',
    'IFeatureGenerator',

    # Analysis Engine interfaces
    'IAnalysisProvider',
    'IIndicatorProvider',
    'IPatternRecognizer',

    # Trading interfaces
    'ITradingProvider',
    'IOrderBookProvider',
    'OrderType',
    'OrderSide',
    'OrderStatus',

    # ML Integration interfaces
    'IMLModelRegistry',
    'IMLJobTracker',
    'IMLModelDeployment',
    'IMLMetricsProvider',
    'ModelStatus',
    'ModelType',
    'ModelFramework',

    # Risk Management interfaces
    'IRiskManager',
    'RiskLimitType',
    'RiskCheckResult'
]
