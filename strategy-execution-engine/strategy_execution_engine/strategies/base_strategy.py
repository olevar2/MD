"""
Base Strategy Module for Forex Trading Platform

This module defines the BaseStrategy abstract class that all trading
strategies must inherit from. It provides the common interface and
functionality that all strategies should implement.
"""
import abc
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import traceback

from ..error import (
    with_error_handling,
    async_with_error_handling,
    StrategyExecutionError,
    SignalGenerationError,
    OrderGenerationError
)

logger = logging.getLogger(__name__)

class BaseStrategy(abc.ABC):
    """
    Abstract base class for all trading strategies

    This class defines the interface that all trading strategies must implement.
    It provides common functionality and enforces a consistent API across
    different strategy implementations.
    """

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize a strategy instance

        Args:
            name: Name of the strategy instance
            parameters: Dictionary of strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.metadata: Dict[str, Any] = {}
        self.last_run_time: Optional[datetime] = None
        self.is_active = False
        self.performance_metrics: Dict[str, Any] = {}

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set metadata for the strategy

        Args:
            metadata: Dictionary containing strategy metadata
        """
        self.metadata = metadata

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get strategy metadata

        Returns:
            Dictionary containing strategy metadata
        """
        return self.metadata

    def validate_parameters(self) -> Tuple[bool, Optional[str]]:
        """
        Validate strategy parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Base implementation performs no validation
        return True, None

    @abc.abstractmethod
    @with_error_handling(error_class=StrategyExecutionError)
    def initialize(self) -> bool:
        """
        Initialize the strategy

        This method is called once before the strategy starts execution.
        It should perform any setup or resource allocation needed by the strategy.

        Returns:
            True if initialization was successful, False otherwise
        """
        pass

    @abc.abstractmethod
    @with_error_handling(error_class=SignalGenerationError)
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data and generate trading signals

        Args:
            data: Dictionary containing market data

        Returns:
            Dictionary containing generated signals and analysis results
        """
        pass

    @abc.abstractmethod
    @with_error_handling(error_class=OrderGenerationError)
    def generate_orders(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading orders based on signals

        Args:
            signals: Dictionary containing trading signals

        Returns:
            List of order dictionaries
        """
        pass

    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update strategy performance metrics

        Args:
            metrics: Dictionary containing performance metrics
        """
        self.performance_metrics.update(metrics)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics

        Returns:
            Dictionary containing performance metrics
        """
        return self.performance_metrics

    def activate(self) -> None:
        """
        Activate the strategy
        """
        self.is_active = True
        logger.info(f"Strategy {self.name} activated")

    def deactivate(self) -> None:
        """
        Deactivate the strategy
        """
        self.is_active = False
        logger.info(f"Strategy {self.name} deactivated")

    def is_activated(self) -> bool:
        """
        Check if strategy is active

        Returns:
            True if strategy is active, False otherwise
        """
        return self.is_active

    def on_error(self, error: Exception) -> None:
        """
        Handle strategy execution error

        Args:
            error: The exception that occurred
        """
        if isinstance(error, StrategyExecutionError):
            # Already a StrategyExecutionError, just log it
            logger.error(
                f"Error in strategy {self.name}: {error.message}",
                extra={
                    "error_code": error.error_code,
                    "details": error.details,
                    "strategy_name": self.name
                }
            )
        else:
            # Convert to StrategyExecutionError and log
            error_details = {
                "strategy_name": self.name,
                "traceback": traceback.format_exc()
            }
            logger.error(
                f"Error in strategy {self.name}: {str(error)}",
                extra=error_details
            )

    @with_error_handling(error_class=StrategyExecutionError, reraise=False)
    def teardown(self) -> None:
        """
        Clean up resources used by the strategy

        This method is called when the strategy is being shut down.
        It should release any resources allocated by the strategy.
        """
        try:
            # Perform cleanup tasks
            # This is a base implementation that can be overridden by subclasses

            logger.info(f"Strategy {self.name} teardown complete")
        except Exception as e:
            # This will be caught by the decorator and logged, but not re-raised
            error_details = {
                "strategy_name": self.name,
                "traceback": traceback.format_exc()
            }
            logger.error(
                f"Error during strategy teardown {self.name}: {str(e)}",
                extra=error_details
            )
