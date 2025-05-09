"""
Tool Effectiveness Interfaces Module

This module provides interfaces for tool effectiveness functionality used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass


class MarketRegimeEnum(str, Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class TimeFrameEnum(str, Enum):
    """Trading timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


@dataclass
class SignalEvent:
    """Trading signal event data"""
    tool_id: str
    symbol: str
    timestamp: datetime
    direction: str  # "buy", "sell", "neutral"
    timeframe: str
    confidence: float
    market_regime: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SignalOutcome:
    """Outcome of a trading signal"""
    signal_id: str
    result: str  # "win", "loss", "neutral"
    pips: float
    profit_loss: float
    exit_timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class IToolEffectivenessMetric(ABC):
    """Interface for tool effectiveness metrics"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the metric name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get the metric description."""
        pass
    
    @abstractmethod
    def calculate(self, signals: List[SignalEvent], outcomes: List[SignalOutcome]) -> float:
        """
        Calculate the metric value.
        
        Args:
            signals: List of signal events
            outcomes: List of signal outcomes
            
        Returns:
            Calculated metric value
        """
        pass


class IToolEffectivenessTracker(ABC):
    """Interface for tool effectiveness tracking"""
    
    @abstractmethod
    def record_signal(self, signal: SignalEvent) -> str:
        """
        Record a new trading signal.
        
        Args:
            signal: Signal event to record
            
        Returns:
            Signal ID
        """
        pass
    
    @abstractmethod
    def record_outcome(self, outcome: SignalOutcome) -> bool:
        """
        Record the outcome of a signal.
        
        Args:
            outcome: Signal outcome to record
            
        Returns:
            Success flag
        """
        pass
    
    @abstractmethod
    def get_tool_effectiveness(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness metrics for a tool.
        
        Args:
            tool_id: Tool identifier
            market_regime: Optional market regime filter
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary containing effectiveness metrics
        """
        pass
    
    @abstractmethod
    def get_best_tools(
        self,
        market_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        min_signals: int = 10,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get best performing tools for a market regime.
        
        Args:
            market_regime: Market regime to analyze
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            min_signals: Minimum number of signals required
            top_n: Optional limit on number of tools to return
            
        Returns:
            List of tool performance metrics, sorted by effectiveness
        """
        pass
    
    @abstractmethod
    def get_tool_history(
        self,
        tool_id: str,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get signal history for a tool.
        
        Args:
            tool_id: Tool identifier
            limit: Optional limit on number of signals to return
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of signal results
        """
        pass
    
    @abstractmethod
    def get_performance_summary(
        self,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get overall performance summary for all tools.
        
        Args:
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Performance summary dictionary
        """
        pass


class IToolEffectivenessRepository(ABC):
    """Interface for tool effectiveness data repository"""
    
    @abstractmethod
    async def save_signal(self, signal: SignalEvent) -> str:
        """
        Save a signal event to the repository.
        
        Args:
            signal: Signal event to save
            
        Returns:
            Signal ID
        """
        pass
    
    @abstractmethod
    async def save_outcome(self, outcome: SignalOutcome) -> bool:
        """
        Save a signal outcome to the repository.
        
        Args:
            outcome: Signal outcome to save
            
        Returns:
            Success flag
        """
        pass
    
    @abstractmethod
    async def get_signals(
        self,
        tool_id: Optional[str] = None,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[SignalEvent]:
        """
        Get signals matching the specified criteria.
        
        Args:
            tool_id: Optional tool ID filter
            market_regime: Optional market regime filter
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit on number of signals to return
            
        Returns:
            List of signal events
        """
        pass
    
    @abstractmethod
    async def get_outcomes(
        self,
        signal_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[SignalOutcome]:
        """
        Get outcomes matching the specified criteria.
        
        Args:
            signal_ids: Optional list of signal IDs to filter by
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit on number of outcomes to return
            
        Returns:
            List of signal outcomes
        """
        pass
    
    @abstractmethod
    async def get_tool_metrics(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness metrics for a tool.
        
        Args:
            tool_id: Tool identifier
            market_regime: Optional market regime filter
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary containing effectiveness metrics
        """
        pass
    
    @abstractmethod
    async def get_tool_metrics_summary_async(
        self,
        tool_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness metrics summary for a tool.
        
        Args:
            tool_id: Tool identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary containing effectiveness metrics summary
        """
        pass
    
    @abstractmethod
    async def count_tool_outcomes_async(
        self,
        tool_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Count outcomes for a tool.
        
        Args:
            tool_id: Tool identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Count of outcomes
        """
        pass
    
    @abstractmethod
    async def get_previous_regime_async(
        self,
        current_regime: str,
        before_date: datetime,
        max_lookback_hours: int = 48
    ) -> Optional[Dict[str, Any]]:
        """
        Get the previous market regime before the specified date.
        
        Args:
            current_regime: Current market regime
            before_date: Date to look before
            max_lookback_hours: Maximum hours to look back
            
        Returns:
            Previous regime information or None if not found
        """
        pass
