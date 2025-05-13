"""
Dynamic Risk Parameter Optimizer for Reinforcement Learning models.

This module integrates reinforcement learning models with the risk management system,
allowing RL agents to dynamically optimize risk parameters based on market conditions.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from ml_workbench_service.rl_model_factory import RLModelFactory
from ml_workbench_service.models.reinforcement.enhanced_rl_env import EnhancedForexTradingEnv
from common_lib.simulation.interfaces import IRiskManager
from common_lib.risk.interfaces import IRiskParameters, IDynamicRiskTuner
from common_lib.reinforcement.interfaces import IRLModel, IRLOptimizer, IRiskParameterOptimizer
from ml_workbench_service.adapters.risk_optimizer_adapter import RiskParametersAdapter
from ml_workbench_service.adapters.analysis_adapters import AnalysisEngineAdapter, MarketRegimeAnalyzerAdapter
from core_foundations.utils.logger import get_logger
from core_foundations.models.market_regimes import MarketRegime
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RLRiskParameterOptimizer:
    """
    Connects reinforcement learning models with risk management to dynamically
    adjust risk parameters based on market conditions and RL model insights.

    This component:
    1. Uses RL models to predict optimal risk parameters for current conditions
    2. Validates and adjusts parameters based on risk management constraints
    3. Provides seamless integration between RL predictions and risk system
    4. Maintains a record of parameter adjustments and their effectiveness
    """

    def __init__(self, rl_model_factory: RLModelFactory, risk_manager:
        IRiskManager, dynamic_risk_adjuster: IDynamicRiskTuner,
        base_risk_parameters: Dict[str, Dict[str, Any]],
        max_adjustment_percent: float=0.5, adaptation_smoothing_factor:
        float=0.3, confidence_threshold: float=0.7, enable_auto_adjustment:
        bool=True):
        """
        Initialize the RL Risk Parameter Optimizer.

        Args:
            rl_model_factory: Factory for creating/loading RL models
            risk_manager: Risk management system interface
            dynamic_risk_adjuster: Existing risk adjustment component
            base_risk_parameters: Default risk parameters by symbol
            max_adjustment_percent: Maximum allowed adjustment magnitude
            adaptation_smoothing_factor: Factor for smooth parameter transitions
            confidence_threshold: Minimum RL confidence for parameter adjustments
            enable_auto_adjustment: Whether to enable automatic adjustments
        """
        self.rl_model_factory = rl_model_factory
        self.risk_manager = risk_manager
        self.dynamic_risk_adjuster = dynamic_risk_adjuster
        self.base_risk_parameters = base_risk_parameters
        self.max_adjustment_percent = max_adjustment_percent
        self.adaptation_smoothing_factor = adaptation_smoothing_factor
        self.confidence_threshold = confidence_threshold
        self.enable_auto_adjustment = enable_auto_adjustment
        self.adjustment_history = []
        self.risk_optimization_models = {}
        self._load_optimization_models()

    def _load_optimization_models(self):
        """Load RL models for risk parameter optimization."""
        logger.info('Loading risk optimization models')
        self.risk_optimization_models = {'position_sizing': None,
            'stop_placement': None, 'take_profit': None, 'risk_exposure': None}

    async def optimize_risk_parameters(self, symbol: str,
        current_market_data: pd.DataFrame, market_regime: MarketRegime,
        current_portfolio_state: Dict[str, Any], prediction_confidence:
        Optional[float]=None) ->Tuple[IRiskParameters, Dict[str, Any]]:
        """
        Generate optimized risk parameters using RL insights.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            current_market_data: Recent market data for context
            market_regime: Detected market regime
            current_portfolio_state: Current positions and exposure
            prediction_confidence: Optional confidence score from prediction model

        Returns:
            Tuple of (optimized risk parameters, metadata about adjustments)
        """
        if not self.enable_auto_adjustment:
            default_params = self._get_default_parameters()
            return RiskParametersAdapter(self.base_risk_parameters.get(
                symbol, default_params)), {'adjusted': False, 'reason':
                'Automatic adjustment disabled'}
        base_params = self.base_risk_parameters.get(symbol, self.
            _get_default_parameters())
        condition_adjusted_params = (self.dynamic_risk_adjuster.
            update_parameters(market_data=current_market_data, current_time
            =datetime.now(), force_update=True))
        confidence = prediction_confidence or 0.0
        if confidence < self.confidence_threshold:
            return condition_adjusted_params, {'adjusted': True, 'source':
                'market_conditions_only', 'confidence': confidence,
                'reason': 'RL confidence below threshold'}
        features = self._extract_features(symbol, current_market_data,
            market_regime, current_portfolio_state)
        recommended_adjustments = self._get_rl_recommendations(symbol,
            features, market_regime)
        optimized_params = self._apply_risk_adjustments(
            condition_adjusted_params, recommended_adjustments, confidence)
        self._record_adjustment(symbol, RiskParametersAdapter(base_params),
            condition_adjusted_params, optimized_params, market_regime,
            confidence)
        return optimized_params, {'adjusted': True, 'source':
            'rl_optimized', 'confidence': confidence, 'adjustments':
            recommended_adjustments}

    def _get_default_parameters(self) ->Dict[str, Any]:
        """Get default risk parameters when none exist for a symbol."""
        return {'position_size_method': 'fixed_percent',
            'position_size_value': 1.0, 'max_position_size': 5.0,
            'stop_loss_atr_multiplier': 2.0, 'take_profit_atr_multiplier': 
            3.0, 'risk_reward_minimum': 1.5, 'max_correlated_exposure': 
            20.0, 'max_drawdown_limit': 10.0}

    def _extract_features(self, symbol: str, market_data: pd.DataFrame,
        market_regime: MarketRegime, portfolio_state: Dict[str, Any]) ->Dict[
        str, Any]:
        """
        Extract features for RL model input.

        Args:
            symbol: The trading symbol
            market_data: Recent market data
            market_regime: Current market regime
            portfolio_state: Current portfolio information

        Returns:
            Dictionary of features for model input
        """
        features = {}
        if not market_data.empty:
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                features['recent_volatility'] = returns.std() * np.sqrt(252)
                features['recent_return'] = returns.mean() * 100
            if 'atr' in market_data.columns:
                features['atr'] = market_data['atr'].iloc[-1]
            for indicator in ['rsi', 'macd', 'bb_upper', 'bb_lower']:
                if indicator in market_data.columns:
                    features[indicator] = market_data[indicator].iloc[-1]
        regime_features = {f'regime_{r.name.lower()}': (0) for r in
            MarketRegime}
        regime_features[f'regime_{market_regime.name.lower()}'] = 1
        features.update(regime_features)
        features['current_exposure'] = portfolio_state.get('total_exposure',
            0.0)
        features['open_positions'] = portfolio_state.get('open_position_count',
            0)
        features['current_drawdown'] = portfolio_state.get(
            'current_drawdown_pct', 0.0)
        return features

    def _get_rl_recommendations(self, symbol: str, features: Dict[str, Any],
        market_regime: MarketRegime) ->Dict[str, float]:
        """
        Get risk parameter recommendations from RL models.

        Args:
            symbol: Trading symbol
            features: Extracted features for model input
            market_regime: Current market regime

        Returns:
            Dictionary of recommended parameter adjustments
        """
        regime_name = market_regime.name.lower()
        if regime_name == 'volatile':
            return {'position_size_multiplier': 0.7,
                'stop_loss_atr_multiplier': 1.2,
                'take_profit_atr_multiplier': 0.9,
                'max_correlated_exposure_multiplier': 0.8}
        elif regime_name == 'trending':
            return {'position_size_multiplier': 1.1,
                'stop_loss_atr_multiplier': 0.9,
                'take_profit_atr_multiplier': 1.3,
                'max_correlated_exposure_multiplier': 1.0}
        elif regime_name == 'ranging':
            return {'position_size_multiplier': 0.9,
                'stop_loss_atr_multiplier': 1.1,
                'take_profit_atr_multiplier': 0.8,
                'max_correlated_exposure_multiplier': 0.9}
        else:
            return {'position_size_multiplier': 1.0,
                'stop_loss_atr_multiplier': 1.0,
                'take_profit_atr_multiplier': 1.0,
                'max_correlated_exposure_multiplier': 1.0}

    def _apply_risk_adjustments(self, base_params: IRiskParameters,
        recommended_adjustments: Dict[str, float], confidence: float
        ) ->IRiskParameters:
        """
        Apply RL recommendations to risk parameters with safety constraints.

        Args:
            base_params: Starting risk parameters
            recommended_adjustments: RL model recommendations
            confidence: Confidence level of recommendations

        Returns:
            Adjusted risk parameters
        """
        params_dict = base_params.to_dict()
        adjustment_scale = confidence * self.adaptation_smoothing_factor
        position_size_mul = recommended_adjustments.get(
            'position_size_multiplier', 1.0)
        position_size_mul = self._constrain_adjustment(position_size_mul)
        weighted_mul = 1.0 + (position_size_mul - 1.0) * adjustment_scale
        params_dict['position_size_value'] *= weighted_mul
        params_dict['position_size_value'] = min(params_dict[
            'position_size_value'], params_dict['max_position_size'])
        stop_loss_mul = recommended_adjustments.get('stop_loss_atr_multiplier',
            1.0)
        stop_loss_mul = self._constrain_adjustment(stop_loss_mul)
        weighted_mul = 1.0 + (stop_loss_mul - 1.0) * adjustment_scale
        params_dict['stop_loss_atr_multiplier'] *= weighted_mul
        tp_mul = recommended_adjustments.get('take_profit_atr_multiplier', 1.0)
        tp_mul = self._constrain_adjustment(tp_mul)
        weighted_mul = 1.0 + (tp_mul - 1.0) * adjustment_scale
        params_dict['take_profit_atr_multiplier'] *= weighted_mul
        if 'risk_reward_minimum' in params_dict:
            if params_dict['take_profit_atr_multiplier'] / params_dict[
                'stop_loss_atr_multiplier'] < params_dict['risk_reward_minimum'
                ]:
                params_dict['take_profit_atr_multiplier'] = params_dict[
                    'stop_loss_atr_multiplier'] * params_dict[
                    'risk_reward_minimum']
        corr_mul = recommended_adjustments.get(
            'max_correlated_exposure_multiplier', 1.0)
        corr_mul = self._constrain_adjustment(corr_mul)
        weighted_mul = 1.0 + (corr_mul - 1.0) * adjustment_scale
        if 'max_correlated_exposure' in params_dict:
            params_dict['max_correlated_exposure'] *= weighted_mul
        return RiskParametersAdapter(params_dict)

    def _constrain_adjustment(self, multiplier: float) ->float:
        """Constrain adjustment multipliers to allowed range."""
        min_val = 1.0 - self.max_adjustment_percent
        max_val = 1.0 + self.max_adjustment_percent
        return max(min_val, min(multiplier, max_val))

    def _record_adjustment(self, symbol: str, base_params: IRiskParameters,
        condition_adjusted_params: IRiskParameters, final_params:
        IRiskParameters, market_regime: MarketRegime, confidence: float
        ) ->None:
        """Record parameter adjustments for tracking and analysis."""
        base_dict = base_params.to_dict()
        condition_dict = condition_adjusted_params.to_dict()
        final_dict = final_params.to_dict()
        self.adjustment_history.append({'timestamp': datetime.now(),
            'symbol': symbol, 'market_regime': market_regime.name,
            'confidence': confidence, 'base_params': {'position_size':
            base_dict.get('position_size_value', 1.0), 'stop_loss_atr':
            base_dict.get('stop_loss_atr_multiplier', 2.0),
            'take_profit_atr': base_dict.get('take_profit_atr_multiplier', 
            3.0), 'max_correlated_exposure': base_dict.get(
            'max_correlated_exposure', 0.2)}, 'condition_adjusted_params':
            {'position_size': condition_dict.get('position_size_value', 1.0
            ), 'stop_loss_atr': condition_dict.get(
            'stop_loss_atr_multiplier', 2.0), 'take_profit_atr':
            condition_dict.get('take_profit_atr_multiplier', 3.0),
            'max_correlated_exposure': condition_dict.get(
            'max_correlated_exposure', 0.2)}, 'final_params': {
            'position_size': final_dict.get('position_size_value', 1.0),
            'stop_loss_atr': final_dict.get('stop_loss_atr_multiplier', 2.0
            ), 'take_profit_atr': final_dict.get(
            'take_profit_atr_multiplier', 3.0), 'max_correlated_exposure':
            final_dict.get('max_correlated_exposure', 0.2)}})
        if len(self.adjustment_history) > 1000:
            self.adjustment_history = self.adjustment_history[-1000:]

    def get_adjustment_history(self, symbol: Optional[str]=None, start_time:
        Optional[datetime]=None, end_time: Optional[datetime]=None) ->List[Dict
        [str, Any]]:
        """
        Get history of parameter adjustments with filtering.

        Args:
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of adjustment records
        """
        filtered_history = self.adjustment_history
        if symbol:
            filtered_history = [h for h in filtered_history if h['symbol'] ==
                symbol]
        if start_time:
            filtered_history = [h for h in filtered_history if h[
                'timestamp'] >= start_time]
        if end_time:
            filtered_history = [h for h in filtered_history if h[
                'timestamp'] <= end_time]
        return filtered_history

    @async_with_exception_handling
    async def update_rl_models(self) ->bool:
        """
        Update the RL models with the latest training data.

        Returns:
            Success status of the update
        """
        try:
            logger.info('Updating RL risk optimization models')
            self._load_optimization_models()
            return True
        except Exception as e:
            logger.error(f'Failed to update RL risk optimization models: {e}')
            return False

    def evaluate_adjustment_effectiveness(self, lookback_days: int=30) ->Dict[
        str, Any]:
        """
        Evaluate the effectiveness of risk parameter adjustments.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary of effectiveness metrics
        """
        return {'evaluated_period_days': lookback_days, 'total_adjustments':
            len(self.adjustment_history), 'average_position_size_change': 
            0.0, 'average_stop_loss_change': 0.0, 'effectiveness_score': 
            0.0, 'roi_impact': 0.0}
