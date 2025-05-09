"""
Dynamic Risk Adapter Module

This module provides adapter implementations for dynamic risk tuning interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import copy
import random

from common_lib.risk.interfaces import (
    IRiskParameters, RiskRegimeType, RiskParameterType, IDynamicRiskTuner
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class DynamicRiskTunerAdapter(IDynamicRiskTuner):
    """
    Adapter for dynamic risk tuners that implements the common interface.
    
    This adapter can either wrap an actual tuner instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, tuner_instance=None):
        """
        Initialize the adapter.
        
        Args:
            tuner_instance: Optional actual tuner instance to wrap
        """
        self.tuner = tuner_instance
        self.parameter_history = []
        self.current_parameters = {
            "position_size_value": 1.0,
            "position_size_method": "fixed",
            "max_position_size": 10.0,
            "stop_loss_atr_multiplier": 2.0,
            "take_profit_atr_multiplier": 3.0,
            "max_risk_per_trade_pct": 0.02,
            "max_total_risk_pct": 0.10,
            "volatility_scaling_factor": 1.0,
            "drawdown_reduction_factor": 1.0,
            "correlation_limit": 0.7,
            "max_correlated_exposure": 0.2
        }
        self.current_regime = RiskRegimeType.MODERATE_RISK
    
    def update_parameters(
        self,
        market_data: Optional[Any] = None,
        current_time: Optional[datetime] = None,
        rl_recommendations: Optional[Dict[str, float]] = None,
        force_update: bool = False
    ) -> IRiskParameters:
        """
        Update risk parameters based on regime, RL insights, and time.
        
        Args:
            market_data: Optional market data for regime detection
            current_time: Current time (defaults to now)
            rl_recommendations: Optional parameter recommendations from RL
            force_update: Whether to force update regardless of time
            
        Returns:
            Updated risk parameters
        """
        if self.tuner:
            try:
                # Try to use the wrapped tuner if available
                return self.tuner.update_parameters(
                    market_data=market_data,
                    current_time=current_time,
                    rl_recommendations=rl_recommendations,
                    force_update=force_update
                )
            except Exception as e:
                logger.warning(f"Error updating risk parameters: {str(e)}")
        
        # Fallback to simple update if no tuner available
        now = current_time or datetime.now()
        
        # Detect regime from market data if available
        if market_data:
            self.current_regime = self._detect_regime_from_data(market_data)
        
        # Apply RL recommendations if available
        if rl_recommendations:
            for param, value in rl_recommendations.items():
                if param in self.current_parameters:
                    self.current_parameters[param] = value
        
        # Apply regime-based adjustments
        adjusted_params = self._adjust_for_regime(self.current_parameters, self.current_regime)
        
        # Record the update
        self.parameter_history.append((now, copy.deepcopy(adjusted_params)))
        
        # Create a parameters object
        return SimpleRiskParameters(adjusted_params)
    
    def get_current_parameters(self) -> IRiskParameters:
        """
        Get current risk parameters.
        
        Returns:
            Current risk parameters
        """
        if self.tuner:
            try:
                # Try to use the wrapped tuner if available
                return self.tuner.get_current_parameters()
            except Exception as e:
                logger.warning(f"Error getting current risk parameters: {str(e)}")
        
        # Return current parameters
        return SimpleRiskParameters(self.current_parameters)
    
    def get_parameter_history(self) -> List[Tuple[datetime, Dict[str, Any]]]:
        """
        Get history of parameter changes.
        
        Returns:
            List of (timestamp, parameters) tuples
        """
        if self.tuner:
            try:
                # Try to use the wrapped tuner if available
                return self.tuner.get_parameter_history()
            except Exception as e:
                logger.warning(f"Error getting parameter history: {str(e)}")
        
        return self.parameter_history
    
    def _detect_regime_from_data(self, market_data: Any) -> RiskRegimeType:
        """Detect risk regime from market data."""
        # Simple detection logic based on volatility
        try:
            # Try to extract volatility from market data
            volatility = None
            
            if hasattr(market_data, 'volatility'):
                volatility = market_data.volatility
            elif isinstance(market_data, dict) and 'volatility' in market_data:
                volatility = market_data['volatility']
            
            if volatility is not None:
                if volatility > 0.3:
                    return RiskRegimeType.EXTREME_RISK
                elif volatility > 0.2:
                    return RiskRegimeType.HIGH_RISK
                elif volatility > 0.1:
                    return RiskRegimeType.MODERATE_RISK
                else:
                    return RiskRegimeType.LOW_RISK
        except Exception as e:
            logger.warning(f"Error detecting regime from market data: {str(e)}")
        
        # Default to current regime if detection fails
        return self.current_regime
    
    def _adjust_for_regime(self, params: Dict[str, Any], regime: RiskRegimeType) -> Dict[str, Any]:
        """Adjust parameters for the given risk regime."""
        adjusted = copy.deepcopy(params)
        
        if regime == RiskRegimeType.LOW_RISK:
            # Conservative parameters
            adjusted['position_size_value'] *= 0.5
            adjusted['max_position_size'] *= 0.7
            adjusted['stop_loss_atr_multiplier'] *= 1.5
            adjusted['max_risk_per_trade_pct'] *= 0.7
            adjusted['volatility_scaling_factor'] *= 0.8
        
        elif regime == RiskRegimeType.MODERATE_RISK:
            # Default parameters, no changes needed
            pass
        
        elif regime == RiskRegimeType.HIGH_RISK:
            # More conservative parameters
            adjusted['position_size_value'] *= 0.3
            adjusted['max_position_size'] *= 0.5
            adjusted['stop_loss_atr_multiplier'] *= 1.8
            adjusted['max_risk_per_trade_pct'] *= 0.5
            adjusted['max_total_risk_pct'] *= 0.7
            adjusted['volatility_scaling_factor'] *= 0.6
        
        elif regime == RiskRegimeType.EXTREME_RISK:
            # Very conservative parameters
            adjusted['position_size_value'] *= 0.2
            adjusted['max_position_size'] *= 0.4
            adjusted['stop_loss_atr_multiplier'] *= 2.0
            adjusted['max_risk_per_trade_pct'] *= 0.4
            adjusted['max_total_risk_pct'] *= 0.5
            adjusted['volatility_scaling_factor'] *= 0.5
            adjusted['drawdown_reduction_factor'] *= 1.3
        
        elif regime == RiskRegimeType.CRISIS:
            # Ultra-conservative parameters
            adjusted['position_size_value'] *= 0.1
            adjusted['max_position_size'] *= 0.2
            adjusted['stop_loss_atr_multiplier'] *= 2.5
            adjusted['max_risk_per_trade_pct'] *= 0.2
            adjusted['max_total_risk_pct'] *= 0.3
            adjusted['volatility_scaling_factor'] *= 0.3
            adjusted['drawdown_reduction_factor'] *= 2.0
        
        return adjusted


class SimpleRiskParameters(IRiskParameters):
    """Simple implementation of risk parameters interface."""
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize with parameters.
        
        Args:
            parameters: Dictionary of parameter values
        """
        self.parameters = copy.deepcopy(parameters)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to dictionary.
        
        Returns:
            Dictionary representation of parameters
        """
        return copy.deepcopy(self.parameters)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleRiskParameters':
        """
        Create parameters from dictionary.
        
        Args:
            data: Dictionary representation of parameters
            
        Returns:
            Risk parameters instance
        """
        return cls(data)
    
    def adjust_for_regime(self, regime_type: RiskRegimeType) -> 'SimpleRiskParameters':
        """
        Create a copy of risk parameters adjusted for the given risk regime.
        
        Args:
            regime_type: Risk regime type
            
        Returns:
            Adjusted risk parameters
        """
        params = self.to_dict()
        
        if regime_type == RiskRegimeType.LOW_RISK:
            # Conservative parameters
            params['position_size_value'] *= 0.5
            params['max_position_size'] *= 0.7
            params['stop_loss_atr_multiplier'] *= 1.5
            params['max_risk_per_trade_pct'] *= 0.7
            params['volatility_scaling_factor'] *= 0.8
        
        elif regime_type == RiskRegimeType.MODERATE_RISK:
            # Default parameters, no changes needed
            pass
        
        elif regime_type == RiskRegimeType.HIGH_RISK:
            # More conservative parameters
            params['position_size_value'] *= 0.3
            params['max_position_size'] *= 0.5
            params['stop_loss_atr_multiplier'] *= 1.8
            params['max_risk_per_trade_pct'] *= 0.5
            params['max_total_risk_pct'] *= 0.7
            params['volatility_scaling_factor'] *= 0.6
        
        elif regime_type == RiskRegimeType.EXTREME_RISK:
            # Very conservative parameters
            params['position_size_value'] *= 0.2
            params['max_position_size'] *= 0.4
            params['stop_loss_atr_multiplier'] *= 2.0
            params['max_risk_per_trade_pct'] *= 0.4
            params['max_total_risk_pct'] *= 0.5
            params['volatility_scaling_factor'] *= 0.5
            params['drawdown_reduction_factor'] *= 1.3
        
        elif regime_type == RiskRegimeType.CRISIS:
            # Ultra-conservative parameters
            params['position_size_value'] *= 0.1
            params['max_position_size'] *= 0.2
            params['stop_loss_atr_multiplier'] *= 2.5
            params['max_risk_per_trade_pct'] *= 0.2
            params['max_total_risk_pct'] *= 0.3
            params['volatility_scaling_factor'] *= 0.3
            params['drawdown_reduction_factor'] *= 2.0
        
        return SimpleRiskParameters(params)
