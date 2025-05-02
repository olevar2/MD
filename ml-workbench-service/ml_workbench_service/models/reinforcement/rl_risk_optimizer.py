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
from risk_management_service.risk_manager import RiskManager
from risk_management_service.models.risk_parameters import RiskParameters, PositionSizeMethod
from risk_management_service.dynamic_risk_adjuster import DynamicRiskAdjuster
from core_foundations.utils.logger import get_logger
from core_foundations.models.market_regimes import MarketRegime

logger = get_logger(__name__)


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
    
    def __init__(
        self,
        rl_model_factory: RLModelFactory,
        risk_manager: RiskManager,
        dynamic_risk_adjuster: DynamicRiskAdjuster,
        base_risk_parameters: Dict[str, RiskParameters],
        max_adjustment_percent: float = 0.5,  # Maximum adjustment allowed (50%)
        adaptation_smoothing_factor: float = 0.3,  # Smooth transitions
        confidence_threshold: float = 0.7,  # Minimum confidence for adjustments
        enable_auto_adjustment: bool = True
    ):
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
        
        # Track adjustment history and effectiveness
        self.adjustment_history = []
        
        # Load optimization models
        self.risk_optimization_models = {}
        self._load_optimization_models()
        
    def _load_optimization_models(self):
        """Load RL models for risk parameter optimization."""
        # This would normally load pre-trained models from storage
        # For now, we'll create placeholder entries
        logger.info("Loading risk optimization models")
        
        # In production, these would be loaded from saved model files
        self.risk_optimization_models = {
            "position_sizing": None,
            "stop_placement": None,
            "take_profit": None,
            "risk_exposure": None
        }
    
    async def optimize_risk_parameters(
        self,
        symbol: str,
        current_market_data: pd.DataFrame,
        market_regime: MarketRegime,
        current_portfolio_state: Dict[str, Any],
        prediction_confidence: Optional[float] = None
    ) -> Tuple[RiskParameters, Dict[str, Any]]:
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
            return self.base_risk_parameters.get(symbol, self._get_default_parameters()), {
                "adjusted": False,
                "reason": "Automatic adjustment disabled"
            }
            
        # Get base parameters for this symbol
        base_params = self.base_risk_parameters.get(
            symbol, self._get_default_parameters())
        
        # Get adjustments from dynamic risk adjuster (based on market conditions)
        condition_adjusted_params = await self.dynamic_risk_adjuster.adjust_parameters(
            symbol, base_params, market_regime)
            
        # If we don't have model predictions or confidence is too low,
        # just return the condition-based adjustments
        confidence = prediction_confidence or 0.0
        if confidence < self.confidence_threshold:
            return condition_adjusted_params, {
                "adjusted": True,
                "source": "market_conditions_only",
                "confidence": confidence,
                "reason": "RL confidence below threshold"
            }
            
        # Get features for RL-based optimization
        features = self._extract_features(
            symbol, current_market_data, market_regime, current_portfolio_state)
            
        # Get RL model recommendations for risk parameters
        recommended_adjustments = self._get_rl_recommendations(
            symbol, features, market_regime)
            
        # Apply RL recommendations with safety constraints
        optimized_params = self._apply_risk_adjustments(
            condition_adjusted_params, recommended_adjustments, confidence)
            
        # Record the adjustment for tracking
        self._record_adjustment(
            symbol, base_params, condition_adjusted_params, 
            optimized_params, market_regime, confidence)
            
        return optimized_params, {
            "adjusted": True,
            "source": "rl_optimized",
            "confidence": confidence,
            "adjustments": recommended_adjustments
        }
        
    def _get_default_parameters(self) -> RiskParameters:
        """Get default risk parameters when none exist for a symbol."""
        return RiskParameters(
            position_size_method=PositionSizeMethod.FIXED_PERCENT,
            position_size_value=1.0,  # 1% of account
            max_position_size=5.0,    # 5% of account maximum
            stop_loss_atr_multiplier=2.0,
            take_profit_atr_multiplier=3.0,
            risk_reward_minimum=1.5,
            max_correlated_exposure=20.0,
            max_drawdown_limit=10.0
        )
    
    def _extract_features(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        market_regime: MarketRegime,
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        # Extract relevant features from market data and portfolio state
        # This would be customized based on what the RL models were trained on
        features = {}
        
        # Market volatility features
        if not market_data.empty:
            # Calculate recent volatility
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                features['recent_volatility'] = returns.std() * np.sqrt(252)  # Annualized
                features['recent_return'] = returns.mean() * 100  # Mean return as percentage
                
            # ATR if available
            if 'atr' in market_data.columns:
                features['atr'] = market_data['atr'].iloc[-1]
                
            # Technical indicators if available
            for indicator in ['rsi', 'macd', 'bb_upper', 'bb_lower']:
                if indicator in market_data.columns:
                    features[indicator] = market_data[indicator].iloc[-1]
        
        # Market regime features (one-hot encoded)
        regime_features = {f"regime_{r.name.lower()}": 0 for r in MarketRegime}
        regime_features[f"regime_{market_regime.name.lower()}"] = 1
        features.update(regime_features)
        
        # Portfolio state features
        features['current_exposure'] = portfolio_state.get('total_exposure', 0.0)
        features['open_positions'] = portfolio_state.get('open_position_count', 0)
        features['current_drawdown'] = portfolio_state.get('current_drawdown_pct', 0.0)
        
        return features
    
    def _get_rl_recommendations(
        self,
        symbol: str,
        features: Dict[str, Any],
        market_regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Get risk parameter recommendations from RL models.
        
        Args:
            symbol: Trading symbol
            features: Extracted features for model input
            market_regime: Current market regime
            
        Returns:
            Dictionary of recommended parameter adjustments
        """
        # This would use the loaded RL models to generate recommendations
        # For now, we'll generate some reasonable adjustments based on regime
        
        # In a real implementation, this would run RL model inference
        # adjustments = self.risk_optimization_models["position_sizing"].predict(features)
        
        # Simplified logic by market regime 
        regime_name = market_regime.name.lower()
        if regime_name == 'volatile':
            return {
                'position_size_multiplier': 0.7,  # Reduce position size
                'stop_loss_atr_multiplier': 1.2,  # Wider stops
                'take_profit_atr_multiplier': 0.9,  # Tighter take profits
                'max_correlated_exposure_multiplier': 0.8  # Reduce correlation exposure
            }
        elif regime_name == 'trending':
            return {
                'position_size_multiplier': 1.1,  # Slightly larger positions
                'stop_loss_atr_multiplier': 0.9,  # Tighter stops
                'take_profit_atr_multiplier': 1.3,  # Wider take profits
                'max_correlated_exposure_multiplier': 1.0  # Normal correlation exposure
            }
        elif regime_name == 'ranging':
            return {
                'position_size_multiplier': 0.9,  # Slightly reduced positions
                'stop_loss_atr_multiplier': 1.1,  # Slightly wider stops
                'take_profit_atr_multiplier': 0.8,  # Tighter take profits
                'max_correlated_exposure_multiplier': 0.9  # Slightly reduced correlation
            }
        else:  # normal or unknown
            return {
                'position_size_multiplier': 1.0,  # No change
                'stop_loss_atr_multiplier': 1.0,  # No change
                'take_profit_atr_multiplier': 1.0,  # No change
                'max_correlated_exposure_multiplier': 1.0  # No change
            }
        
    def _apply_risk_adjustments(
        self,
        base_params: RiskParameters,
        recommended_adjustments: Dict[str, float],
        confidence: float
    ) -> RiskParameters:
        """
        Apply RL recommendations to risk parameters with safety constraints.
        
        Args:
            base_params: Starting risk parameters
            recommended_adjustments: RL model recommendations
            confidence: Confidence level of recommendations
            
        Returns:
            Adjusted risk parameters
        """
        # Create a copy to avoid modifying the original
        adjusted_params = self._copy_risk_parameters(base_params)
        
        # Scale adjustments by confidence and smoothing factor
        adjustment_scale = confidence * self.adaptation_smoothing_factor
        
        # Apply position size adjustment
        position_size_mul = recommended_adjustments.get('position_size_multiplier', 1.0)
        position_size_mul = self._constrain_adjustment(position_size_mul)
        weighted_mul = 1.0 + (position_size_mul - 1.0) * adjustment_scale
        adjusted_params.position_size_value *= weighted_mul
        
        # Ensure position size doesn't exceed max allowed
        adjusted_params.position_size_value = min(
            adjusted_params.position_size_value, 
            adjusted_params.max_position_size)
        
        # Apply stop loss adjustment
        stop_loss_mul = recommended_adjustments.get('stop_loss_atr_multiplier', 1.0)
        stop_loss_mul = self._constrain_adjustment(stop_loss_mul)
        weighted_mul = 1.0 + (stop_loss_mul - 1.0) * adjustment_scale
        adjusted_params.stop_loss_atr_multiplier *= weighted_mul
        
        # Apply take profit adjustment
        tp_mul = recommended_adjustments.get('take_profit_atr_multiplier', 1.0)
        tp_mul = self._constrain_adjustment(tp_mul)
        weighted_mul = 1.0 + (tp_mul - 1.0) * adjustment_scale
        adjusted_params.take_profit_atr_multiplier *= weighted_mul
        
        # Ensure risk-reward ratio is maintained
        if (adjusted_params.take_profit_atr_multiplier / 
                adjusted_params.stop_loss_atr_multiplier < 
                adjusted_params.risk_reward_minimum):
            # Adjust take profit to maintain minimum risk-reward ratio
            adjusted_params.take_profit_atr_multiplier = (
                adjusted_params.stop_loss_atr_multiplier * 
                adjusted_params.risk_reward_minimum
            )
        
        # Apply correlation exposure adjustment
        corr_mul = recommended_adjustments.get('max_correlated_exposure_multiplier', 1.0)
        corr_mul = self._constrain_adjustment(corr_mul)
        weighted_mul = 1.0 + (corr_mul - 1.0) * adjustment_scale
        adjusted_params.max_correlated_exposure *= weighted_mul
        
        return adjusted_params
        
    def _constrain_adjustment(self, multiplier: float) -> float:
        """Constrain adjustment multipliers to allowed range."""
        min_val = 1.0 - self.max_adjustment_percent
        max_val = 1.0 + self.max_adjustment_percent
        return max(min_val, min(multiplier, max_val))
    
    def _copy_risk_parameters(self, params: RiskParameters) -> RiskParameters:
        """Create a deep copy of risk parameters."""
        return RiskParameters(
            position_size_method=params.position_size_method,
            position_size_value=params.position_size_value,
            max_position_size=params.max_position_size,
            stop_loss_atr_multiplier=params.stop_loss_atr_multiplier,
            take_profit_atr_multiplier=params.take_profit_atr_multiplier,
            risk_reward_minimum=params.risk_reward_minimum,
            max_correlated_exposure=params.max_correlated_exposure,
            max_drawdown_limit=params.max_drawdown_limit
        )
    
    def _record_adjustment(
        self,
        symbol: str,
        base_params: RiskParameters,
        condition_adjusted_params: RiskParameters,
        final_params: RiskParameters,
        market_regime: MarketRegime,
        confidence: float
    ) -> None:
        """Record parameter adjustments for tracking and analysis."""
        self.adjustment_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'market_regime': market_regime.name,
            'confidence': confidence,
            'base_params': {
                'position_size': base_params.position_size_value,
                'stop_loss_atr': base_params.stop_loss_atr_multiplier,
                'take_profit_atr': base_params.take_profit_atr_multiplier,
                'max_correlated_exposure': base_params.max_correlated_exposure
            },
            'condition_adjusted_params': {
                'position_size': condition_adjusted_params.position_size_value,
                'stop_loss_atr': condition_adjusted_params.stop_loss_atr_multiplier,
                'take_profit_atr': condition_adjusted_params.take_profit_atr_multiplier,
                'max_correlated_exposure': condition_adjusted_params.max_correlated_exposure
            },
            'final_params': {
                'position_size': final_params.position_size_value,
                'stop_loss_atr': final_params.stop_loss_atr_multiplier,
                'take_profit_atr': final_params.take_profit_atr_multiplier,
                'max_correlated_exposure': final_params.max_correlated_exposure
            }
        })
        
        # Keep history at a reasonable size
        if len(self.adjustment_history) > 1000:
            self.adjustment_history = self.adjustment_history[-1000:]
    
    def get_adjustment_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
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
            filtered_history = [h for h in filtered_history if h['symbol'] == symbol]
            
        if start_time:
            filtered_history = [h for h in filtered_history if h['timestamp'] >= start_time]
            
        if end_time:
            filtered_history = [h for h in filtered_history if h['timestamp'] <= end_time]
            
        return filtered_history
    
    async def update_rl_models(self) -> bool:
        """
        Update the RL models with the latest training data.
        
        Returns:
            Success status of the update
        """
        try:
            # This would reload models from the model registry
            # after they've been retrained with new data
            logger.info("Updating RL risk optimization models")
            self._load_optimization_models()
            return True
        except Exception as e:
            logger.error(f"Failed to update RL risk optimization models: {e}")
            return False
    
    def evaluate_adjustment_effectiveness(
        self, 
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of risk parameter adjustments.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary of effectiveness metrics
        """
        # This would analyze trade outcomes with adjusted parameters
        # compared to baseline parameters
        
        # In a real implementation, this would correlate adjustments
        # with actual trading results
        
        # For now, return a placeholder
        return {
            'evaluated_period_days': lookback_days,
            'total_adjustments': len(self.adjustment_history),
            'average_position_size_change': 0.0,
            'average_stop_loss_change': 0.0,
            'effectiveness_score': 0.0,
            'roi_impact': 0.0
        }
