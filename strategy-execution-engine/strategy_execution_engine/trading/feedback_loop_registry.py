"""
Service registry for the feedback loop system.

This module provides service registry and dependency injection configuration
for the bidirectional feedback loop components.
"""

from typing import Dict, Any, Optional
import os

from strategy_execution_engine.trading.feedback_collector import FeedbackCollector
from strategy_execution_engine.trading.feedback_router import FeedbackRouter
from strategy_execution_engine.adaptive_layer.strategy_mutator_factory import StrategyMutatorFactory
from analysis_engine.adaptive_layer.timeframe_feedback_service import TimeframeFeedbackService
from analysis_engine.adaptive_layer.statistical_validator import StatisticalValidator
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackLoopRegistry:
    """Registry for feedback loop services and components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feedback loop registry with optional configuration.
        
        Args:
            config: Configuration for the feedback loop components
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize components
        self._statistical_validator = None
        self._strategy_mutator_factory = None
        self._feedback_collector = None
        self._timeframe_feedback_service = None
        self._feedback_router = None
        
        # Initialize registry
        self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Initialize and wire up all feedback loop components."""
        # Create components in dependency order
        self._create_statistical_validator()
        self._create_strategy_mutator_factory()
        self._create_feedback_collector()
        self._create_timeframe_feedback_service()
        self._create_feedback_router()
        
        self.logger.info("Feedback loop registry initialized with all components")
    
    def _create_statistical_validator(self) -> None:
        """Create the statistical validator component."""
        validator_config = self.config.get("statistical_validator", {})
        
        self._statistical_validator = StatisticalValidator(
            min_samples=validator_config.get("min_samples", 20),
            significance_level=validator_config.get("significance_level", 0.05),
            correlation_thresholds=validator_config.get("correlation_thresholds")
        )
        
        self.logger.debug("Statistical validator created")
    
    def _create_strategy_mutator_factory(self) -> None:
        """Create the strategy mutator factory component."""
        factory_config = self.config.get("strategy_mutator_factory", {})
        
        self._strategy_mutator_factory = StrategyMutatorFactory(
            config_path=factory_config.get("config_path"),
            statistical_validator=self._statistical_validator
        )
        
        self.logger.debug("Strategy mutator factory created")
    
    def _create_feedback_collector(self) -> None:
        """Create the feedback collector component."""
        collector_config = self.config.get("feedback_collector", {})
        
        self._feedback_collector = FeedbackCollector(
            config=collector_config
        )
        
        self.logger.debug("Feedback collector created")
    
    def _create_timeframe_feedback_service(self) -> None:
        """Create the timeframe feedback service component."""
        # This assumes TimeframeFeedbackService is already implemented in your codebase
        timeframe_config = self.config.get("timeframe_feedback_service", {})
        
        # Initialize with appropriate configuration from your existing system
        self._timeframe_feedback_service = TimeframeFeedbackService(
            config=timeframe_config
        )
        
        self.logger.debug("Timeframe feedback service created")
    
    def _create_feedback_router(self) -> None:
        """Create and wire up the feedback router component."""
        # Create the feedback router with all required dependencies
        self._feedback_router = FeedbackRouter(
            feedback_collector=self._feedback_collector,
            timeframe_feedback_service=self._timeframe_feedback_service,
            statistical_validator=self._statistical_validator,
            strategy_mutator=None  # This is handled differently since it's strategy-specific
        )
        
        self.logger.debug("Feedback router created")
    
    def get_statistical_validator(self) -> StatisticalValidator:
        """Get the statistical validator instance."""
        return self._statistical_validator
    
    def get_strategy_mutator_factory(self) -> StrategyMutatorFactory:
        """Get the strategy mutator factory instance."""
        return self._strategy_mutator_factory
    
    def get_feedback_collector(self) -> FeedbackCollector:
        """Get the feedback collector instance."""
        return self._feedback_collector
    
    def get_timeframe_feedback_service(self) -> TimeframeFeedbackService:
        """Get the timeframe feedback service instance."""
        return self._timeframe_feedback_service
    
    def get_feedback_router(self) -> FeedbackRouter:
        """Get the feedback router instance."""
        return self._feedback_router
    
    def create_strategy_specific_router(self, 
                                      strategy_id: str, 
                                      strategy_config: Dict[str, Any]) -> FeedbackRouter:
        """
        Create a feedback router with a strategy-specific mutator.
        
        Args:
            strategy_id: The ID of the strategy
            strategy_config: The strategy configuration
            
        Returns:
            A feedback router with the strategy-specific mutator
        """
        # Create a strategy-specific mutator
        strategy_mutator = self._strategy_mutator_factory.create_mutator(
            strategy_id=strategy_id,
            strategy_config=strategy_config
        )
        
        # Create a new feedback router with this mutator
        router = FeedbackRouter(
            feedback_collector=self._feedback_collector,
            timeframe_feedback_service=self._timeframe_feedback_service,
            statistical_validator=self._statistical_validator,
            strategy_mutator=strategy_mutator
        )
        
        self.logger.info(f"Created strategy-specific feedback router for strategy {strategy_id}")
        return router
