"""
Risk Optimizer Adapter Module

This module provides adapter implementations for risk parameter optimization interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import asyncio
import copy
from common_lib.risk.interfaces import IRiskParameters, RiskRegimeType, RiskParameterType, IRiskParameterOptimizer
from common_lib.reinforcement.interfaces import IRLModel, IRLOptimizer
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RiskParametersAdapter(IRiskParameters):
    """
    Adapter for risk parameters that implements the common interface.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            parameters: Optional dictionary of parameter values
        """
        self.parameters = parameters or {'position_size_value': 1.0,
            'position_size_method': 'fixed', 'max_position_size': 10.0,
            'stop_loss_atr_multiplier': 2.0, 'take_profit_atr_multiplier': 
            3.0, 'max_risk_per_trade_pct': 0.02, 'max_total_risk_pct': 0.1,
            'volatility_scaling_factor': 1.0, 'drawdown_reduction_factor': 
            1.0, 'correlation_limit': 0.7, 'max_correlated_exposure': 0.2}

    def to_dict(self) ->Dict[str, Any]:
        """
        Convert parameters to dictionary.
        
        Returns:
            Dictionary representation of parameters
        """
        return copy.deepcopy(self.parameters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'RiskParametersAdapter':
        """
        Create parameters from dictionary.
        
        Args:
            data: Dictionary representation of parameters
            
        Returns:
            Risk parameters instance
        """
        instance = cls()
        instance.parameters = copy.deepcopy(data)
        return instance

    def adjust_for_regime(self, regime_type: RiskRegimeType
        ) ->'RiskParametersAdapter':
        """
        Create a copy of risk parameters adjusted for the given risk regime.
        
        Args:
            regime_type: Risk regime type
            
        Returns:
            Adjusted risk parameters
        """
        params = self.to_dict()
        if regime_type == RiskRegimeType.LOW_RISK:
            params['position_size_value'] *= 0.5
            params['max_position_size'] *= 0.7
            params['stop_loss_atr_multiplier'] *= 1.5
            params['max_risk_per_trade_pct'] *= 0.7
            params['volatility_scaling_factor'] *= 0.8
        elif regime_type == RiskRegimeType.MODERATE_RISK:
            pass
        elif regime_type == RiskRegimeType.HIGH_RISK:
            params['position_size_value'] *= 0.3
            params['max_position_size'] *= 0.5
            params['stop_loss_atr_multiplier'] *= 1.8
            params['max_risk_per_trade_pct'] *= 0.5
            params['max_total_risk_pct'] *= 0.7
            params['volatility_scaling_factor'] *= 0.6
        elif regime_type == RiskRegimeType.EXTREME_RISK:
            params['position_size_value'] *= 0.2
            params['max_position_size'] *= 0.4
            params['stop_loss_atr_multiplier'] *= 2.0
            params['max_risk_per_trade_pct'] *= 0.4
            params['max_total_risk_pct'] *= 0.5
            params['volatility_scaling_factor'] *= 0.5
            params['drawdown_reduction_factor'] *= 1.3
        elif regime_type == RiskRegimeType.CRISIS:
            params['position_size_value'] *= 0.1
            params['max_position_size'] *= 0.2
            params['stop_loss_atr_multiplier'] *= 2.5
            params['max_risk_per_trade_pct'] *= 0.2
            params['max_total_risk_pct'] *= 0.3
            params['volatility_scaling_factor'] *= 0.3
            params['drawdown_reduction_factor'] *= 2.0
        return RiskParametersAdapter.from_dict(params)


class RiskParameterOptimizerAdapter(IRiskParameterOptimizer):
    """
    Adapter for risk parameter optimizer that implements the common interface.
    
    This adapter can either wrap an actual optimizer instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, optimizer_instance=None):
        """
        Initialize the adapter.
        
        Args:
            optimizer_instance: Optional actual optimizer instance to wrap
        """
        self.optimizer = optimizer_instance
        self.adjustment_history = []
        self.last_confidence = 0.5
        self.base_parameters = RiskParametersAdapter()

    @async_with_exception_handling
    async def optimize_risk_parameters(self, symbol: str,
        current_market_data: Any, market_regime: Any,
        current_portfolio_state: Dict[str, Any], prediction_confidence:
        Optional[float]=None) ->Tuple[RiskParametersAdapter, Dict[str, Any]]:
        """
        Generate optimized risk parameters using RL insights.
        
        Args:
            symbol: Trading symbol
            current_market_data: Recent market data for context
            market_regime: Detected market regime
            current_portfolio_state: Current positions and exposure
            prediction_confidence: Optional confidence score from prediction model
            
        Returns:
            Tuple of (optimized risk parameters, metadata about adjustments)
        """
        if self.optimizer:
            try:
                params, metadata = (await self.optimizer.
                    optimize_risk_parameters(symbol=symbol,
                    current_market_data=current_market_data, market_regime=
                    market_regime, current_portfolio_state=
                    current_portfolio_state, prediction_confidence=
                    prediction_confidence))
                return params, metadata
            except Exception as e:
                logger.warning(f'Error optimizing risk parameters: {str(e)}')
        confidence = prediction_confidence or 0.5
        self.last_confidence = confidence
        regime_type = self._map_market_regime(market_regime)
        params = self.base_parameters.adjust_for_regime(regime_type)
        self._record_adjustment(symbol=symbol, regime=regime_type,
            confidence=confidence, params=params.to_dict())
        return params, {'adjusted': True, 'source': 'adapter_fallback',
            'confidence': confidence, 'market_regime': str(regime_type)}

    @async_with_exception_handling
    async def update_rl_models(self) ->bool:
        """
        Update the RL models with the latest training data.
        
        Returns:
            Success status of the update
        """
        if self.optimizer:
            try:
                return await self.optimizer.update_rl_models()
            except Exception as e:
                logger.warning(f'Error updating RL models: {str(e)}')
        return True

    @with_exception_handling
    def get_adjustment_history(self) ->List[Dict[str, Any]]:
        """
        Get the history of parameter adjustments.
        
        Returns:
            List of adjustment records
        """
        if self.optimizer:
            try:
                return self.optimizer.get_adjustment_history()
            except Exception as e:
                logger.warning(f'Error getting adjustment history: {str(e)}')
        return self.adjustment_history

    def _map_market_regime(self, market_regime: Any) ->RiskRegimeType:
        """Map market regime to risk regime type."""
        regime_str = str(market_regime).upper()
        if 'VOLATILE' in regime_str or 'HIGH_VOLATILITY' in regime_str:
            return RiskRegimeType.HIGH_RISK
        elif 'TRENDING' in regime_str:
            return RiskRegimeType.LOW_RISK
        elif 'RANGING' in regime_str:
            return RiskRegimeType.MODERATE_RISK
        elif 'BREAKOUT' in regime_str:
            return RiskRegimeType.HIGH_RISK
        elif 'REVERSAL' in regime_str:
            return RiskRegimeType.HIGH_RISK
        elif 'CRISIS' in regime_str:
            return RiskRegimeType.CRISIS
        return RiskRegimeType.MODERATE_RISK

    def _record_adjustment(self, symbol: str, regime: RiskRegimeType,
        confidence: float, params: Dict[str, Any]) ->None:
        """Record a parameter adjustment."""
        self.adjustment_history.append({'timestamp': datetime.now().
            isoformat(), 'symbol': symbol, 'regime': str(regime),
            'confidence': confidence, 'parameters': params})
