"""
Parameter Adjustment Service

This module implements the parameter adjustment service that utilizes the AdaptationEngine
to manage the automated parameter adjustments for trading strategies.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService
from analysis_engine.tools.market_regime_identifier import MarketRegimeIdentifier
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

class ParameterAdjustmentService:
    """
    Service for managing automated parameter adjustments based on market conditions and
    effectiveness metrics. This service acts as a facade for the AdaptationEngine and
    provides additional functionality for parameter management and persistence.
    """
    
    def __init__(
        self,
        adaptation_engine: AdaptationEngine,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the ParameterAdjustmentService.
        
        Args:
            adaptation_engine: The adaptation engine for parameter adjustments
            config: Configuration parameters for the service
        """
        self.adaptation_engine = adaptation_engine
        self.config = config or {}
        self.parameter_history = {}  # Store parameter history by strategy
        
        # Thresholds for significant parameter changes
        self.change_thresholds = self.config.get('change_thresholds', {
            'default': 0.15,  # 15% change is significant by default
            'position_size_factor': 0.10,  # More sensitive for position sizing
            'stop_loss_pips': 0.10,
            'take_profit_pips': 0.10,
        })
        
        logger.info("ParameterAdjustmentService initialized")

    def adjust_parameters(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        current_parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on market conditions and tool effectiveness.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument (e.g., 'EUR_USD')
            timeframe: The timeframe for analysis (e.g., '1H', '4H', 'D')
            current_parameters: Current strategy parameters
            context: Additional context information
            
        Returns:
            Dict[str, Any]: Adjusted parameters
        """
        context = context or {}
        
        # Use the adaptation engine to adapt parameters
        adjusted_parameters = self.adaptation_engine.adapt_parameters(
            instrument=instrument,
            timeframe=timeframe,
            current_parameters=current_parameters,
            context=context
        )
        
        # Store the parameter update in history
        self._store_parameter_update(
            strategy_id=strategy_id,
            instrument=instrument,
            timeframe=timeframe,
            original_params=current_parameters,
            adjusted_params=adjusted_parameters
        )
        
        # Log significant parameter changes
        self._log_significant_changes(
            strategy_id=strategy_id,
            original_params=current_parameters,
            adjusted_params=adjusted_parameters
        )
        
        return adjusted_parameters
    
    def _store_parameter_update(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        original_params: Dict[str, Any],
        adjusted_params: Dict[str, Any]
    ) -> None:
        """
        Store a parameter update in the history.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            original_params: Original strategy parameters
            adjusted_params: Adjusted strategy parameters
        """
        key = f"{strategy_id}:{instrument}:{timeframe}"
        
        if key not in self.parameter_history:
            self.parameter_history[key] = []
            
        update_record = {
            'timestamp': datetime.utcnow(),
            'original_params': original_params,
            'adjusted_params': adjusted_params,
        }
        
        self.parameter_history[key].append(update_record)
        
        # Limit history size to avoid memory issues
        max_history = self.config.get('max_parameter_history', 100)
        if len(self.parameter_history[key]) > max_history:
            self.parameter_history[key].pop(0)  # Remove oldest entry
    
    def _log_significant_changes(
        self,
        strategy_id: str,
        original_params: Dict[str, Any],
        adjusted_params: Dict[str, Any]
    ) -> None:
        """
        Log significant parameter changes for monitoring.
        
        Args:
            strategy_id: Identifier for the strategy
            original_params: Original strategy parameters
            adjusted_params: Adjusted strategy parameters
        """
        significant_changes = {}
        
        for param_name, new_value in adjusted_params.items():
            if param_name not in original_params:
                continue
                
            old_value = original_params[param_name]
            
            # Skip non-numeric parameters
            if not isinstance(new_value, (int, float)) or not isinstance(old_value, (int, float)):
                continue
                
            if old_value == 0:
                # Avoid division by zero
                if new_value != 0:
                    significant_changes[param_name] = f"{old_value} → {new_value} (new)"
            else:
                # Calculate percentage change
                pct_change = abs(new_value - old_value) / abs(old_value)
                
                # Get the threshold for this parameter
                threshold = self.change_thresholds.get(param_name, self.change_thresholds['default'])
                
                # If change exceeds threshold, consider it significant
                if pct_change > threshold:
                    significant_changes[param_name] = f"{old_value:.4f} → {new_value:.4f} ({pct_change:.1%} change)"
        
        if significant_changes:
            logger.info(
                "Significant parameter changes for strategy %s: %s", 
                strategy_id, 
                significant_changes
            )
    
    def get_parameter_history(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get parameter adjustment history for a specific strategy.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            limit: Maximum number of history entries to return
            
        Returns:
            List[Dict[str, Any]]: Parameter adjustment history
        """
        key = f"{strategy_id}:{instrument}:{timeframe}"
        
        if key not in self.parameter_history:
            return []
            
        # Return the most recent entries first, limited by 'limit'
        return list(reversed(self.parameter_history[key][-limit:]))
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of adaptation decisions from the adaptation engine.
        
        Returns:
            List[Dict[str, Any]]: Adaptation history
        """
        return self.adaptation_engine.get_adaptation_history()
