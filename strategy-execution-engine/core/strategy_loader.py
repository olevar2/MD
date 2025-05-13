"""
Strategy Loader Module for Forex Trading Platform

This module handles loading, registering, and managing trading strategies.
It provides functionality to dynamically load strategy implementations and
maintain a registry of available strategies.
"""
import importlib
import inspect
import json
import os
import logging
import traceback
from typing import Dict, List, Optional, Type, Any, Union, Tuple

from .base_strategy import BaseStrategy
from .advanced_ta_strategy import AdvancedTAStrategy
from ..error import (
    with_error_handling,
    StrategyLoadError,
    StrategyConfigurationError
)

logger = logging.getLogger(__name__)

class StrategyLoader:
    """
    Strategy loader for managing and loading trading strategies

    This class handles strategy registration, discovery, and instantiation.
    It maintains a registry of available strategies and provides methods to
    load strategies from various sources.
    """

    def __init__(self, strategies_directory: Optional[str] = None):
        """
        Initialize the strategy loader

        Args:
            strategies_directory: Directory containing strategy implementations
        """
        self.strategies_directory = strategies_directory
        self.strategy_registry: Dict[str, Type[BaseStrategy]] = {}
        self._discover_strategies()

    @with_error_handling(error_class=StrategyLoadError)
    def _discover_strategies(self) -> None:
        """
        Discover and register available strategies from the strategies directory
        """
        if not self.strategies_directory or not os.path.exists(self.strategies_directory):
            logger.warning(f"Strategies directory not found: {self.strategies_directory}")
            return

        logger.info(f"Discovering strategies in: {self.strategies_directory}")

        # Look for Python files that might contain strategy implementations
        for filename in os.listdir(self.strategies_directory):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]  # Remove .py extension
                module_path = f"strategy_execution_engine.strategies.{module_name}"

                self._load_strategy_module(module_name, module_path)

    @with_error_handling(error_class=StrategyLoadError)
    def _load_strategy_module(self, module_name: str, module_path: str) -> None:
        """
        Load a strategy module and register its strategy classes

        Args:
            module_name: Name of the module
            module_path: Import path for the module
        """
        try:
            module = importlib.import_module(module_path)

            # Find all classes in the module that are subclasses of BaseStrategy
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and (
                    (issubclass(obj, BaseStrategy) and obj is not BaseStrategy) or
                    (issubclass(obj, AdvancedTAStrategy) and obj is not AdvancedTAStrategy)
                ):

                    strategy_id = f"{module_name}.{name}"
                    self.strategy_registry[strategy_id] = obj
                    logger.info(f"Registered strategy: {strategy_id}")

        except Exception as e:
            # This will be caught by the decorator and wrapped in a StrategyLoadError
            error_details = {
                "module_name": module_name,
                "module_path": module_path,
                "traceback": traceback.format_exc()
            }
            raise StrategyLoadError(
                message=f"Error loading strategy module {module_name}: {str(e)}",
                details=error_details
            ) from e

    @with_error_handling(error_class=StrategyLoadError)
    def register_strategy(self, strategy_id: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy manually

        Args:
            strategy_id: Unique identifier for the strategy
            strategy_class: Strategy class implementing BaseStrategy interface
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError("Strategy class must be a subclass of BaseStrategy")

        self.strategy_registry[strategy_id] = strategy_class
        logger.info(f"Manually registered strategy: {strategy_id}")

    def get_strategy_class(self, strategy_id: str) -> Optional[Type[BaseStrategy]]:
        """
        Get a strategy class by its ID

        Args:
            strategy_id: Unique identifier for the strategy

        Returns:
            Strategy class or None if not found
        """
        return self.strategy_registry.get(strategy_id)

    @with_error_handling(error_class=StrategyConfigurationError)
    def create_strategy_instance(self, strategy_id: str, parameters: Dict[str, Any] = None) -> Optional[BaseStrategy]:
        """
        Create an instance of a strategy with the specified parameters

        Args:
            strategy_id: Unique identifier for the strategy
            parameters: Dictionary of strategy parameters

        Returns:
            Strategy instance or None if strategy not found
        """
        strategy_class = self.get_strategy_class(strategy_id)
        if not strategy_class:
            raise StrategyLoadError(
                message=f"Strategy not found: {strategy_id}",
                strategy_name=strategy_id
            )

        try:
            strategy = strategy_class(name=strategy_id, parameters=parameters or {})
            is_valid, error = strategy.validate_parameters()
            if not is_valid:
                raise StrategyConfigurationError(
                    message=f"Invalid parameters for strategy {strategy_id}: {error}",
                    strategy_name=strategy_id,
                    details={"parameters": parameters, "error": error}
                )

            return strategy

        except Exception as e:
            if isinstance(e, StrategyConfigurationError):
                raise

            error_details = {
                "strategy_id": strategy_id,
                "parameters": parameters,
                "traceback": traceback.format_exc()
            }
            raise StrategyConfigurationError(
                message=f"Error creating strategy instance {strategy_id}: {str(e)}",
                strategy_name=strategy_id,
                details=error_details
            ) from e

    @with_error_handling(error_class=StrategyConfigurationError)
    def load_strategy_from_config(self, config_path: str) -> Optional[BaseStrategy]:
        """
        Load a strategy from a configuration file

        Args:
            config_path: Path to the strategy configuration file (JSON)

        Returns:
            Strategy instance or None if loading failed
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            strategy_id = config_manager.get('strategy_id')
            parameters = config_manager.get('parameters', {})
            metadata = config_manager.get('metadata', {})

            if not strategy_id:
                raise StrategyConfigurationError(
                    message=f"Missing strategy_id in config file: {config_path}",
                    details={"config_path": config_path, "config": config}
                )

            strategy = self.create_strategy_instance(strategy_id, parameters)
            if strategy and metadata:
                strategy.set_metadata(metadata)

            return strategy

        except Exception as e:
            if isinstance(e, StrategyConfigurationError):
                raise

            error_details = {
                "config_path": config_path,
                "traceback": traceback.format_exc()
            }
            raise StrategyConfigurationError(
                message=f"Error loading strategy from config {config_path}: {str(e)}",
                details=error_details
            ) from e

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategy IDs

        Returns:
            List of strategy IDs
        """
        return list(self.strategy_registry.keys())
