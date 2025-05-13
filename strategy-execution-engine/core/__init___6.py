"""
Clients Package for Strategy Execution Engine

This package contains clients for interacting with other services in the Forex Trading Platform.
"""

from adapters.analysis_engine_client import AnalysisEngineClient
from adapters.feature_store_client import FeatureStoreClient
from adapters.trading_gateway_client import TradingGatewayClient

__all__ = [
    "AnalysisEngineClient",
    "FeatureStoreClient",
    "TradingGatewayClient"
]
