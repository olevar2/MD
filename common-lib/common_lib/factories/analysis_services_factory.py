from typing import Dict, Any, Optional

from common_lib.interfaces.causal_analysis.causal_analysis import ICausalAnalysisService
from common_lib.interfaces.backtesting.backtesting import IBacktestingService
from common_lib.interfaces.market_analysis.market_analysis import IMarketAnalysisService
from common_lib.interfaces.analysis_coordinator.analysis_coordinator import IAnalysisCoordinatorService

from common_lib.adapters.causal_analysis.causal_analysis_adapter import CausalAnalysisAdapter
from common_lib.adapters.backtesting.backtesting_adapter import BacktestingAdapter
from common_lib.adapters.market_analysis.market_analysis_adapter import MarketAnalysisAdapter
from common_lib.adapters.analysis_coordinator.analysis_coordinator_adapter import AnalysisCoordinatorAdapter

class AnalysisServicesFactory:
    """
    Factory for creating adapters for analysis services.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the factory with configuration.
        
        Args:
            config: Configuration dictionary with service URLs
        """
        self.config = config
        
    def create_causal_analysis_service(self) -> ICausalAnalysisService:
        """
        Create a causal analysis service adapter.
        
        Returns:
            Causal analysis service adapter
        """
        base_url = self.config.get("causal_analysis_service_url", "http://causal-analysis-service:8000")
        return CausalAnalysisAdapter(base_url=base_url)
        
    def create_backtesting_service(self) -> IBacktestingService:
        """
        Create a backtesting service adapter.
        
        Returns:
            Backtesting service adapter
        """
        base_url = self.config.get("backtesting_service_url", "http://backtesting-service:8000")
        return BacktestingAdapter(base_url=base_url)
        
    def create_market_analysis_service(self) -> IMarketAnalysisService:
        """
        Create a market analysis service adapter.
        
        Returns:
            Market analysis service adapter
        """
        base_url = self.config.get("market_analysis_service_url", "http://market-analysis-service:8000")
        return MarketAnalysisAdapter(base_url=base_url)
        
    def create_analysis_coordinator_service(self) -> IAnalysisCoordinatorService:
        """
        Create an analysis coordinator service adapter.
        
        Returns:
            Analysis coordinator service adapter
        """
        base_url = self.config.get("analysis_coordinator_service_url", "http://analysis-coordinator-service:8000")
        return AnalysisCoordinatorAdapter(base_url=base_url)