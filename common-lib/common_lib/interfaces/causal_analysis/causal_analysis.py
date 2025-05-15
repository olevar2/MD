from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class ICausalAnalysisService(ABC):
    """
    Interface for causal analysis service.
    """
    
    @abstractmethod
    async def generate_causal_graph(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        variables: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a causal graph from market data.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            variables: Variables to include in the causal graph
            parameters: Additional parameters for the analysis
            
        Returns:
            Causal graph data
        """
        pass
        
    @abstractmethod
    async def calculate_intervention_effect(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        intervention: Dict[str, Any] = None,
        target: str = "price",
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate the effect of an intervention on the causal graph.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            intervention: Intervention to apply
            target: Target variable to measure effect on
            parameters: Additional parameters for the analysis
            
        Returns:
            Intervention effect data
        """
        pass
        
    @abstractmethod
    async def generate_counterfactual_scenario(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        scenario: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a counterfactual scenario based on the causal graph.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            scenario: Counterfactual scenario to generate
            parameters: Additional parameters for the analysis
            
        Returns:
            Counterfactual scenario data
        """
        pass