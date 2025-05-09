"""
Clients Package for Strategy Execution Engine

This package contains clients for interacting with other services in the Forex Trading Platform.
"""

from strategy_execution_engine.clients.analysis_engine_client import AnalysisEngineClient
from strategy_execution_engine.clients.feature_store_client import FeatureStoreClient
from strategy_execution_engine.clients.trading_gateway_client import TradingGatewayClient

__all__ = [
    "AnalysisEngineClient",
    "FeatureStoreClient",
    "TradingGatewayClient"
]
