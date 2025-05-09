"""
Standardized API Clients Package

This package contains clients for interacting with standardized API endpoints.
All clients follow the platform's standardized client design patterns,
including resilience patterns like retry with backoff and circuit breaking.
"""

from analysis_engine.clients.standardized.adaptive_layer_client import AdaptiveLayerClient
from analysis_engine.clients.standardized.market_regime_client import MarketRegimeClient
from analysis_engine.clients.standardized.signal_quality_client import SignalQualityClient
from analysis_engine.clients.standardized.nlp_analysis_client import NLPAnalysisClient
from analysis_engine.clients.standardized.correlation_analysis_client import CorrelationAnalysisClient
from analysis_engine.clients.standardized.manipulation_detection_client import ManipulationDetectionClient
from analysis_engine.clients.standardized.effectiveness_client import EffectivenessClient
from analysis_engine.clients.standardized.feedback_client import FeedbackClient
from analysis_engine.clients.standardized.monitoring_client import MonitoringClient
from analysis_engine.clients.standardized.causal_client import CausalClient
from analysis_engine.clients.standardized.backtesting_client import BacktestingClient
from analysis_engine.clients.standardized.client_factory import StandardizedClientFactory, get_client_factory

__all__ = [
    "AdaptiveLayerClient",
    "MarketRegimeClient",
    "SignalQualityClient",
    "NLPAnalysisClient",
    "CorrelationAnalysisClient",
    "ManipulationDetectionClient",
    "EffectivenessClient",
    "FeedbackClient",
    "MonitoringClient",
    "CausalClient",
    "BacktestingClient",
    "StandardizedClientFactory",
    "get_client_factory"
]
