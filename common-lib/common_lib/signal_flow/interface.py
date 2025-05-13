"""
Core interfaces for signal flow management.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from common_lib.signal_flow.model import (
    SignalFlow,
    SignalFlowState,
    SignalValidationResult,
    SignalAggregationResult
)

class ISignalFlowManager(ABC):
    """Interface for managing signal flow between services"""
    
    @abstractmethod
    async def publish_signal(self, signal: SignalFlow) -> bool:
        """Publish a new signal to the flow"""
        pass
        
    @abstractmethod
    async def get_signal(self, signal_id: str) -> Optional[SignalFlow]:
        """Get a signal by ID"""
        pass
        
    @abstractmethod
    async def update_signal_state(self, signal_id: str, state: SignalFlowState, metadata: Dict[str, Any]) -> bool:
        """Update the state of a signal"""
        pass
        
    @abstractmethod
    async def get_active_signals(self, symbol: str) -> List[SignalFlow]:
        """Get all active signals for a symbol"""
        pass

class ISignalValidator(ABC):
    """Interface for signal validation"""
    
    @abstractmethod
    async def validate_signal(self, signal: SignalFlow) -> SignalValidationResult:
        """Validate a signal before processing"""
        pass

class ISignalAggregator(ABC):
    """Interface for signal aggregation"""
    
    @abstractmethod
    async def aggregate_signals(
        self,
        signals: List[SignalFlow],
        symbol: str,
        timeframe: str
    ) -> SignalAggregationResult:
        """Aggregate multiple signals into a single trading decision"""
        pass

class ISignalExecutor(ABC):
    """Interface for signal execution"""
    
    @abstractmethod
    async def execute_signal(self, signal: SignalFlow) -> bool:
        """Execute a trading signal"""
        pass
        
    @abstractmethod
    async def cancel_signal(self, signal_id: str) -> bool:
        """Cancel a pending signal"""
        pass

class ISignalMonitor(ABC):
    """Interface for signal monitoring and tracking"""
    
    @abstractmethod
    async def track_signal(self, signal: SignalFlow) -> None:
        """Start tracking a signal"""
        pass
        
    @abstractmethod
    async def update_signal_metrics(self, signal_id: str, metrics: Dict[str, Any]) -> None:
        """Update metrics for a tracked signal"""
        pass
        
    @abstractmethod
    async def get_signal_metrics(self, signal_id: str) -> Dict[str, Any]:
        """Get metrics for a tracked signal"""
        pass
