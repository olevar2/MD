"""
Factory for creating StrategyMutator instances.

This module provides factory functions to create and configure StrategyMutator instances
that can be used in the bidirectional feedback loop system.
"""

from typing import Dict, Any, Optional, List
import os
import json

from strategy_execution_engine.adaptive_layer.strategy_mutator import StrategyMutator
from analysis_engine.adaptive_layer.statistical_validator import StatisticalValidator
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyMutatorFactory:
    """Factory class for creating StrategyMutator instances."""
    
    def __init__(self,
                config_path: str = None,
                statistical_validator: Optional[StatisticalValidator] = None):
        """
        Initialize the StrategyMutator factory.
        
        Args:
            config_path: Path to configuration files for strategy mutation
            statistical_validator: The StatisticalValidator instance to use
        """
        self.config_path = config_path or os.path.join(
            "strategy-execution-engine", "config", "strategy_mutation"
        )
        self.statistical_validator = statistical_validator or StatisticalValidator()
        self.logger = logger
        
        # Cache created mutators
        self._mutators = {}
    
    def create_mutator(self, 
                      strategy_id: str, 
                      strategy_config: Dict[str, Any]) -> StrategyMutator:
        """
        Create a StrategyMutator for a specific strategy.
        
        Args:
            strategy_id: The ID of the strategy
            strategy_config: The strategy configuration
            
        Returns:
            StrategyMutator instance
        """
        # If we already have a cached mutator for this strategy, return it
        if strategy_id in self._mutators:
            return self._mutators[strategy_id]
            
        # Load strategy-specific mutation configuration if available
        mutation_config = self._load_mutation_config(strategy_id)
        
        # Merge with strategy_config if mutation_config was found
        if mutation_config:
            if "mutation_config" not in strategy_config:
                strategy_config["mutation_config"] = {}
                
            strategy_config["mutation_config"].update(mutation_config)
        
        # Create the mutator
        mutator = StrategyMutator(
            strategy_config=strategy_config,
            validator=self.statistical_validator
        )
        
        # Cache and return
        self._mutators[strategy_id] = mutator
        return mutator
    
    def get_mutator(self, strategy_id: str) -> Optional[StrategyMutator]:
        """
        Get an existing StrategyMutator for a specific strategy.
        
        Args:
            strategy_id: The ID of the strategy
            
        Returns:
            StrategyMutator instance if exists, None otherwise
        """
        return self._mutators.get(strategy_id)
    
    def _load_mutation_config(self, strategy_id: str) -> Dict[str, Any]:
        """
        Load mutation configuration for a specific strategy.
        
        Args:
            strategy_id: The ID of the strategy
            
        Returns:
            Dictionary with mutation configuration
        """
        config = {}
        
        # Try to load strategy-specific config
        config_file = os.path.join(self.config_path, f"{strategy_id}.json")
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded mutation config for strategy {strategy_id}")
        except Exception as e:
            self.logger.warning(f"Error loading mutation config for strategy {strategy_id}: {e}")
            
        # If no strategy-specific config, try to load default config
        if not config:
            default_config_file = os.path.join(self.config_path, "default.json")
            try:
                if os.path.exists(default_config_file):
                    with open(default_config_file, 'r') as f:
                        config = json.load(f)
                    self.logger.info(f"Loaded default mutation config for strategy {strategy_id}")
            except Exception as e:
                self.logger.warning(f"Error loading default mutation config: {e}")
                
        return config
