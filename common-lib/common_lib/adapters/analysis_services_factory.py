"""
Factory for creating analysis service adapters.

This module provides a factory for creating adapters for the various analysis services,
including causal analysis, backtesting, market analysis, and analysis coordination.
"""
from typing import Dict, Any, Optional
import logging
from common_lib.interfaces.causal_analysis_service_interface import ICausalAnalysisService
from common_lib.interfaces.backtesting_service_interface import IBacktestingService
from common_lib.interfaces.market_analysis_service_interface import IMarketAnalysisService
from common_lib.interfaces.analysis_coordinator_service_interface import IAnalysisCoordinatorService
from common_lib.adapters.causal_analysis.causal_analysis_adapter import CausalAnalysisAdapter
from common_lib.adapters.backtesting.backtesting_adapter import BacktestingAdapter
from common_lib.adapters.market_analysis.market_analysis_adapter import MarketAnalysisAdapter
from common_lib.adapters.analysis_coordinator.analysis_coordinator_adapter import AnalysisCoordinatorAdapter

logger = logging.getLogger(__name__)


class AnalysisServicesFactory:
    """Factory for creating analysis service adapters."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AnalysisServicesFactory.

        Args:
            config: Configuration dictionary containing service URLs and timeouts
        """
        self.config = config
        logger.info("Initialized AnalysisServicesFactory")

    def create_causal_analysis_adapter(self) -> ICausalAnalysisService:
        """
        Create a CausalAnalysisAdapter.

        Returns:
            An instance of CausalAnalysisAdapter implementing ICausalAnalysisService
        """
        service_config = self.config.get("causal_analysis_service", {})
        base_url = service_config.get("base_url", "http://causal-analysis-service:8000")
        timeout = service_config.get("timeout", 30.0)
        
        logger.info(f"Creating CausalAnalysisAdapter with base URL: {base_url}")
        return CausalAnalysisAdapter(base_url=base_url, timeout=timeout)

    def create_backtesting_adapter(self) -> IBacktestingService:
        """
        Create a BacktestingAdapter.

        Returns:
            An instance of BacktestingAdapter implementing IBacktestingService
        """
        service_config = self.config.get("backtesting_service", {})
        base_url = service_config.get("base_url", "http://backtesting-service:8000")
        timeout = service_config.get("timeout", 60.0)
        
        logger.info(f"Creating BacktestingAdapter with base URL: {base_url}")
        return BacktestingAdapter(base_url=base_url, timeout=timeout)

    def create_market_analysis_adapter(self) -> IMarketAnalysisService:
        """
        Create a MarketAnalysisAdapter.

        Returns:
            An instance of MarketAnalysisAdapter implementing IMarketAnalysisService
        """
        service_config = self.config.get("market_analysis_service", {})
        base_url = service_config.get("base_url", "http://market-analysis-service:8000")
        timeout = service_config.get("timeout", 30.0)
        
        logger.info(f"Creating MarketAnalysisAdapter with base URL: {base_url}")
        return MarketAnalysisAdapter(base_url=base_url, timeout=timeout)

    def create_analysis_coordinator_adapter(self) -> IAnalysisCoordinatorService:
        """
        Create an AnalysisCoordinatorAdapter.

        Returns:
            An instance of AnalysisCoordinatorAdapter implementing IAnalysisCoordinatorService
        """
        service_config = self.config.get("analysis_coordinator_service", {})
        base_url = service_config.get("base_url", "http://analysis-coordinator-service:8000")
        timeout = service_config.get("timeout", 60.0)
        
        logger.info(f"Creating AnalysisCoordinatorAdapter with base URL: {base_url}")
        return AnalysisCoordinatorAdapter(base_url=base_url, timeout=timeout)