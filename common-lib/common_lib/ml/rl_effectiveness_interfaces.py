"""
Reinforcement Learning Effectiveness Interfaces Module

This module provides interfaces for RL model effectiveness analysis,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime


class EffectivenessMetricType(str, Enum):
    """Types of effectiveness metrics for RL models."""
    REWARD = "reward"
    PNL = "pnl"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    AVERAGE_TRADE = "average_trade"
    TRADE_FREQUENCY = "trade_frequency"
    VAR = "value_at_risk"
    REGIME_ADAPTABILITY = "regime_adaptability"
    NEWS_SENSITIVITY = "news_sensitivity"
    CUSTOM = "custom"


class MarketRegimeType(str, Enum):
    """Types of market regimes."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING_NARROW = "ranging_narrow"
    RANGING_WIDE = "ranging_wide"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class IRLModelEffectivenessAnalyzer(ABC):
    """Interface for RL model effectiveness analyzers."""
    
    @abstractmethod
    def analyze_performance(
        self,
        model_id: str,
        performance_data: Dict[str, Any],
        market_regime: Optional[MarketRegimeType] = None
    ) -> Dict[str, Any]:
        """
        Analyze the performance of an RL model.
        
        Args:
            model_id: ID of the model to analyze
            performance_data: Performance data for the model
            market_regime: Optional market regime to analyze for
            
        Returns:
            Dictionary with analysis results
        """
        pass
    
    @abstractmethod
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
        metrics: Optional[List[EffectivenessMetricType]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple RL models.
        
        Args:
            model_results: Dictionary mapping model IDs to their results
            metrics: Optional list of metrics to compare
            
        Returns:
            Dictionary with comparison results
        """
        pass
    
    @abstractmethod
    def get_regime_specific_performance(
        self,
        model_id: str,
        regime: MarketRegimeType
    ) -> Dict[str, Any]:
        """
        Get regime-specific performance for an RL model.
        
        Args:
            model_id: ID of the model to analyze
            regime: Market regime to get performance for
            
        Returns:
            Dictionary with regime-specific performance metrics
        """
        pass
    
    @abstractmethod
    def calculate_effectiveness_score(
        self,
        model_id: str,
        metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate an overall effectiveness score for an RL model.
        
        Args:
            model_id: ID of the model to score
            metrics: Dictionary of metric values
            weights: Optional dictionary of metric weights
            
        Returns:
            Effectiveness score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_historical_effectiveness(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "day"
    ) -> Dict[str, Any]:
        """
        Get historical effectiveness for an RL model.
        
        Args:
            model_id: ID of the model to analyze
            start_date: Optional start date
            end_date: Optional end date
            interval: Interval for the data points (day, week, month)
            
        Returns:
            Dictionary with historical effectiveness data
        """
        pass


class IRLToolEffectivenessIntegration(ABC):
    """Interface for RL tool effectiveness integration."""
    
    @abstractmethod
    def register_model(
        self,
        model_id: str,
        model_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register an RL model for effectiveness tracking.
        
        Args:
            model_id: ID of the model to register
            model_type: Type of the model
            description: Description of the model
            metadata: Optional metadata about the model
            
        Returns:
            True if registration was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def record_model_decision(
        self,
        model_id: str,
        decision_data: Dict[str, Any],
        market_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Record a decision made by an RL model.
        
        Args:
            model_id: ID of the model that made the decision
            decision_data: Data about the decision
            market_data: Market data at the time of the decision
            timestamp: Optional timestamp for the decision
            
        Returns:
            ID of the recorded decision
        """
        pass
    
    @abstractmethod
    def record_decision_outcome(
        self,
        decision_id: str,
        outcome_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Record the outcome of a decision made by an RL model.
        
        Args:
            decision_id: ID of the decision
            outcome_data: Data about the outcome
            timestamp: Optional timestamp for the outcome
            
        Returns:
            True if recording was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_effectiveness(
        self,
        model_id: str,
        market_regime: Optional[MarketRegimeType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness metrics for an RL model.
        
        Args:
            model_id: ID of the model to get metrics for
            market_regime: Optional market regime to filter by
            start_date: Optional start date to filter by
            end_date: Optional end date to filter by
            
        Returns:
            Dictionary with effectiveness metrics
        """
        pass
