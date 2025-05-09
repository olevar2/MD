"""
Standardized Client Factory

This module provides a factory for creating standardized API clients.
"""

from typing import Optional

from analysis_engine.core.config import get_settings
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
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class StandardizedClientFactory:
    """
    Factory for creating standardized API clients.

    This factory provides methods for creating clients for interacting with
    standardized API endpoints. It ensures that clients are configured with
    the correct base URLs and timeouts.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize the standardized client factory.

        Args:
            base_url: Base URL for API endpoints. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout

        logger.info(f"Initialized standardized client factory with base URL: {self.base_url}")

    def get_adaptive_layer_client(self) -> AdaptiveLayerClient:
        """
        Get a client for interacting with the standardized Adaptive Layer API.

        Returns:
            AdaptiveLayerClient instance
        """
        return AdaptiveLayerClient(base_url=self.base_url, timeout=self.timeout)

    def get_market_regime_client(self) -> MarketRegimeClient:
        """
        Get a client for interacting with the standardized Market Regime API.

        Returns:
            MarketRegimeClient instance
        """
        return MarketRegimeClient(base_url=self.base_url, timeout=self.timeout)

    def get_signal_quality_client(self) -> SignalQualityClient:
        """
        Get a client for interacting with the standardized Signal Quality API.

        Returns:
            SignalQualityClient instance
        """
        return SignalQualityClient(base_url=self.base_url, timeout=self.timeout)

    def get_nlp_analysis_client(self) -> NLPAnalysisClient:
        """
        Get a client for interacting with the standardized NLP Analysis API.

        Returns:
            NLPAnalysisClient instance
        """
        return NLPAnalysisClient(base_url=self.base_url, timeout=self.timeout)

    def get_correlation_analysis_client(self) -> CorrelationAnalysisClient:
        """
        Get a client for interacting with the standardized Correlation Analysis API.

        Returns:
            CorrelationAnalysisClient instance
        """
        return CorrelationAnalysisClient(base_url=self.base_url, timeout=self.timeout)

    def get_manipulation_detection_client(self) -> ManipulationDetectionClient:
        """
        Get a client for interacting with the standardized Manipulation Detection API.

        Returns:
            ManipulationDetectionClient instance
        """
        return ManipulationDetectionClient(base_url=self.base_url, timeout=self.timeout)

    def get_effectiveness_client(self) -> EffectivenessClient:
        """
        Get a client for interacting with the standardized Tool Effectiveness API.

        Returns:
            EffectivenessClient instance
        """
        return EffectivenessClient(base_url=self.base_url, timeout=self.timeout)

    def get_feedback_client(self) -> FeedbackClient:
        """
        Get a client for interacting with the standardized Feedback API.

        Returns:
            FeedbackClient instance
        """
        return FeedbackClient(base_url=self.base_url, timeout=self.timeout)

    def get_monitoring_client(self) -> MonitoringClient:
        """
        Get a client for interacting with the standardized Monitoring API.

        Returns:
            MonitoringClient instance
        """
        return MonitoringClient(base_url=self.base_url, timeout=self.timeout)

    def get_causal_client(self) -> CausalClient:
        """
        Get a client for interacting with the standardized Causal Analysis API.

        Returns:
            CausalClient instance
        """
        return CausalClient(base_url=self.base_url, timeout=self.timeout)

    def get_backtesting_client(self) -> BacktestingClient:
        """
        Get a client for interacting with the standardized Backtesting API.

        Returns:
            BacktestingClient instance
        """
        return BacktestingClient(base_url=self.base_url, timeout=self.timeout)

    # Add methods for other standardized clients as they are implemented


# Singleton instance for easy access
_factory_instance = None

def get_client_factory() -> StandardizedClientFactory:
    """
    Get the standardized client factory instance.

    Returns:
        StandardizedClientFactory instance
    """
    global _factory_instance

    if _factory_instance is None:
        _factory_instance = StandardizedClientFactory()

    return _factory_instance
