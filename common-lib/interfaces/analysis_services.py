"""
Interface definitions for the decomposed analysis services.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

class IChatService(ABC):
    """Interface for chat service operations."""
    
    @abstractmethod
    async def process_message(self, user_id: str, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chat message."""
        pass
    
    @abstractmethod
    async def get_history(self, user_id: str, limit: int, before: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get chat history."""
        pass

class IAnalysisCore(ABC):
    """Interface for core analysis operations."""
    
    @abstractmethod
    async def analyze_market_data(self, data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Perform core market data analysis."""
        pass
    
    @abstractmethod
    async def get_analysis_results(self, analysis_id: str) -> Dict[str, Any]:
        """Retrieve analysis results."""
        pass

class IMLAnalysis(ABC):
    """Interface for ML-based analysis operations."""
    
    @abstractmethod
    async def execute_model(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an ML model."""
        pass
    
    @abstractmethod
    async def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics."""
        pass

class IMarketRegime(ABC):
    """Interface for market regime analysis operations."""
    
    @abstractmethod
    async def detect_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current market regime."""
        pass
    
    @abstractmethod
    async def get_regime_history(self, market_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get market regime history."""
        pass

class ISignalGeneration(ABC):
    """Interface for trading signal generation operations."""
    
    @abstractmethod
    async def generate_signals(self, analysis_results: Dict[str, Any], market_regime: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on analysis results and market regime."""
        pass
    
    @abstractmethod
    async def validate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a trading signal."""
        pass
    
    @abstractmethod
    async def get_signal_history(self, market_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get trading signal history."""
        pass

class IAnalysisEventBus(ABC):
    """Interface for analysis-related event handling."""
    
    @abstractmethod
    async def publish_analysis_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish an analysis-related event."""
        pass
    
    @abstractmethod
    async def subscribe_to_events(self, event_types: List[str], callback: callable) -> None:
        """Subscribe to analysis-related events."""
        pass