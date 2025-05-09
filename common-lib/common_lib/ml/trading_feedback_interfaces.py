"""
Trading Feedback Interfaces Module

This module provides interfaces for trading feedback collection and model training feedback,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime


class FeedbackCategory(str, Enum):
    """Categories of trading feedback."""
    SIGNAL_QUALITY = "signal_quality"
    ENTRY_TIMING = "entry_timing"
    EXIT_TIMING = "exit_timing"
    POSITION_SIZING = "position_sizing"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MARKET_REGIME = "market_regime"
    NEWS_IMPACT = "news_impact"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    SLIPPAGE = "slippage"
    EXECUTION_QUALITY = "execution_quality"
    OVERALL_PERFORMANCE = "overall_performance"
    OTHER = "other"


class FeedbackSource(str, Enum):
    """Sources of trading feedback."""
    TRADING_SYSTEM = "trading_system"
    ANALYSIS_ENGINE = "analysis_engine"
    RISK_MANAGEMENT = "risk_management"
    ML_MODEL = "ml_model"
    MANUAL = "manual"
    BACKTEST = "backtest"
    SIMULATION = "simulation"
    LIVE_TRADING = "live_trading"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    OTHER = "other"


class ITradeFeedback(ABC):
    """Interface for trade feedback."""
    
    @property
    @abstractmethod
    def feedback_id(self) -> str:
        """Get the unique identifier for this feedback."""
        pass
    
    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Get the timestamp when the feedback was generated."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> FeedbackCategory:
        """Get the category of the feedback."""
        pass
    
    @property
    @abstractmethod
    def source(self) -> FeedbackSource:
        """Get the source of the feedback."""
        pass
    
    @property
    @abstractmethod
    def instrument(self) -> str:
        """Get the trading instrument the feedback applies to."""
        pass
    
    @property
    @abstractmethod
    def timeframe(self) -> str:
        """Get the timeframe the feedback applies to."""
        pass
    
    @property
    @abstractmethod
    def strategy_id(self) -> Optional[str]:
        """Get the strategy ID the feedback applies to, if any."""
        pass
    
    @property
    @abstractmethod
    def trade_id(self) -> Optional[str]:
        """Get the trade ID the feedback applies to, if any."""
        pass
    
    @property
    @abstractmethod
    def score(self) -> float:
        """Get the feedback score (-1.0 to 1.0)."""
        pass
    
    @property
    @abstractmethod
    def market_regime(self) -> Optional[str]:
        """Get the market regime during the feedback, if known."""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get additional metadata about the feedback."""
        pass


class ITradingFeedbackCollector(ABC):
    """Interface for trading feedback collectors."""
    
    @abstractmethod
    async def collect_feedback(self, feedback: ITradeFeedback) -> bool:
        """
        Collect trading feedback.
        
        Args:
            feedback: The feedback to collect
            
        Returns:
            True if the feedback was successfully collected, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_feedback(
        self,
        strategy_id: Optional[str] = None,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        category: Optional[FeedbackCategory] = None,
        source: Optional[FeedbackSource] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ITradeFeedback]:
        """
        Get collected feedback.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            instrument: Optional instrument to filter by
            timeframe: Optional timeframe to filter by
            category: Optional category to filter by
            source: Optional source to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            limit: Optional limit on the number of results
            
        Returns:
            List of feedback matching the filters
        """
        pass
    
    @abstractmethod
    async def get_feedback_summary(
        self,
        strategy_id: Optional[str] = None,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        category: Optional[FeedbackCategory] = None,
        source: Optional[FeedbackSource] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of collected feedback.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            instrument: Optional instrument to filter by
            timeframe: Optional timeframe to filter by
            category: Optional category to filter by
            source: Optional source to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            Dictionary with feedback summary statistics
        """
        pass


class IModelTrainingFeedback(ABC):
    """Interface for model training feedback."""
    
    @abstractmethod
    async def process_trading_feedback(
        self,
        feedback: List[ITradeFeedback]
    ) -> Dict[str, Any]:
        """
        Process trading feedback for model training.
        
        Args:
            feedback: List of trading feedback
            
        Returns:
            Dictionary with processing results
        """
        pass
    
    @abstractmethod
    async def prepare_training_data(
        self,
        model_id: str,
        feedback: List[ITradeFeedback]
    ) -> Dict[str, Any]:
        """
        Prepare training data for a model.
        
        Args:
            model_id: ID of the model to prepare data for
            feedback: List of trading feedback
            
        Returns:
            Dictionary with prepared training data
        """
        pass
    
    @abstractmethod
    async def trigger_model_retraining(
        self,
        model_id: str,
        training_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Trigger retraining of a model.
        
        Args:
            model_id: ID of the model to retrain
            training_data: Training data for the model
            
        Returns:
            Dictionary with retraining results
        """
        pass
    
    @abstractmethod
    async def track_model_performance(
        self,
        model_id: str,
        performance_metrics: Dict[str, Any]
    ) -> bool:
        """
        Track model performance.
        
        Args:
            model_id: ID of the model to track
            performance_metrics: Performance metrics for the model
            
        Returns:
            True if tracking was successful, False otherwise
        """
        pass
