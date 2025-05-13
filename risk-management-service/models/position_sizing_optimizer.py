"""
Position Sizing Optimizer Service Module.

Provides advanced functionality for optimizing position sizes based on market conditions,
strategy performance, and risk parameters as part of Phase 4 implementation.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

from core_foundations.utils.logger import get_logger
from core.dynamic_risk_adjuster import DynamicRiskAdjuster, MarketRegime
from core.risk_calculator import RiskCalculator
from repositories.risk_repository import RiskRepository

logger = get_logger("position-sizing-optimizer")


class PositionSizingModel:
    """Enumeration of different position sizing models"""
    FIXED = "FIXED"                 # Fixed percentage of account balance
    ATR_BASED = "ATR_BASED"         # Based on ATR (Average True Range)
    KELLY = "KELLY"                 # Kelly criterion
    ANTI_MARTINGALE = "ANTI_MARTINGALE"  # Progressive sizing based on winning streaks
    REGIME_ADAPTIVE = "REGIME_ADAPTIVE"   # Adapts based on market regime


class PositionSizingOptimizer:
    """
    Service for optimizing position sizes based on market conditions, strategy performance,
    and risk parameters.
    
    This service provides functionality to:
    - Calculate optimal position sizes based on multiple models
    - Adapt position sizing to current market conditions
    - Analyze historical strategy performance to optimize future sizing
    - Apply constraints based on risk limits and volatility
    - Provide recommendations for risk-adjusted position sizing
    - Integrate with machine learning for predictive position sizing
    """
    
    def __init__(self, risk_calculator: Optional[RiskCalculator] = None, 
                 risk_adjuster: Optional[DynamicRiskAdjuster] = None):
        """
        Initialize the position sizing optimizer.
        
        Args:
            risk_calculator: Risk calculator service instance
            risk_adjuster: Dynamic risk adjuster service instance
        """
        self.risk_calculator = risk_calculator or RiskCalculator()
        self.risk_adjuster = risk_adjuster or DynamicRiskAdjuster()
        self.performance_history = {}  # Store historical performance metrics by strategy
        
    def calculate_position_size(self,
                               strategy_id: str,
                               symbol: str,
                               account_balance: float,
                               risk_percentage: float,
                               entry_price: float,
                               stop_loss_price: float,
                               model_type: str = PositionSizingModel.REGIME_ADAPTIVE,
                               market_regime: Optional[str] = None,
                               additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size based on specified model and parameters.
        
        Args:
            strategy_id: Identifier of the strategy
            symbol: Trading symbol (e.g., "EUR/USD")
            account_balance: Current account balance
            risk_percentage: Base risk percentage (0-100)
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            model_type: Position sizing model to use
            market_regime: Current market regime (if None, will be determined)
            additional_params: Additional parameters for specific sizing models
            
        Returns:
            Dictionary with position size and calculation details
        """
        additional_params = additional_params or {}
        
        # Determine market regime if not provided
        if market_regime is None and model_type == PositionSizingModel.REGIME_ADAPTIVE:
            market_data = additional_params.get("market_data", {})
            volatility_data = additional_params.get("volatility_data", {})
            market_regime = self.risk_adjuster.analyze_market_regime(volatility_data, market_data)
        
        # Calculate base position size using risk percentage of account
        risk_amount = account_balance * (risk_percentage / 100)
        
        # Calculate risk per unit (e.g., pip value or point value)
        stop_loss_distance = abs(entry_price - stop_loss_price)
        
        if stop_loss_distance <= 0:
            logger.warning(f"Invalid stop loss distance for {symbol}: entry={entry_price}, stop={stop_loss_price}")
            return {
                "position_size": 0,
                "risk_amount": 0,
                "risk_percentage": 0,
                "model_used": model_type,
                "warning": "Invalid stop loss distance"
            }
        
        # Base position size (units/lots)
        base_position_size = risk_amount / stop_loss_distance
        
        # Apply model-specific adjustments
        adjusted_position_size = self._apply_model_adjustment(
            model_type=model_type,
            base_position_size=base_position_size,
            market_regime=market_regime,
            strategy_id=strategy_id,
            symbol=symbol,
            additional_params=additional_params
        )
        
        # Apply minimum/maximum constraints
        final_position_size = self._apply_position_constraints(
            adjusted_position_size, 
            symbol, 
            account_balance,
            additional_params
        )
        
        return {
            "position_size": final_position_size,
            "base_position_size": base_position_size,
            "risk_amount": risk_amount,
            "risk_percentage": risk_percentage,
            "model_used": model_type,
            "market_regime": market_regime,
            "stop_loss_distance": stop_loss_distance
        }
        
    def _apply_model_adjustment(self,
                              model_type: str,
                              base_position_size: float,
                              market_regime: Optional[str],
                              strategy_id: str,
                              symbol: str,
                              additional_params: Dict[str, Any]) -> float:
        """
        Apply model-specific adjustments to the base position size.
        
        Args:
            model_type: Position sizing model to use
            base_position_size: Base calculated position size
            market_regime: Current market regime
            strategy_id: Identifier of the strategy
            symbol: Trading symbol
            additional_params: Additional parameters for specific sizing models
            
        Returns:
            Adjusted position size
        """
        if model_type == PositionSizingModel.FIXED:
            # No adjustment for fixed percentage model
            return base_position_size
            
        elif model_type == PositionSizingModel.ATR_BASED:
            # Adjust position size based on ATR
            atr = additional_params.get("atr", 0)
            atr_multiplier = additional_params.get("atr_multiplier", 1.0)
            
            if atr > 0:
                avg_atr = additional_params.get("avg_historical_atr", atr)
                # Scale position size inversely with volatility
                volatility_ratio = atr / avg_atr if avg_atr > 0 else 1
                return base_position_size * (1 / max(volatility_ratio, 0.5))
            return base_position_size
            
        elif model_type == PositionSizingModel.KELLY:
            # Kelly criterion: f* = (bp - q) / b
            # where p = win rate, q = 1-p, b = win/loss ratio
            win_rate = additional_params.get("win_rate", 0.5)
            avg_win = additional_params.get("avg_win", 0)
            avg_loss = additional_params.get("avg_loss", 0)
            
            if avg_loss <= 0:
                return base_position_size
                
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1
            kelly_fraction = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
            
            # Cap Kelly at 0.5 (half-Kelly) for safety
            kelly_fraction = min(max(kelly_fraction, 0), 0.5)
            
            return base_position_size * kelly_fraction
            
        elif model_type == PositionSizingModel.ANTI_MARTINGALE:
            # Increase position size after wins, decrease after losses
            win_streak = additional_params.get("win_streak", 0)
            lose_streak = additional_params.get("lose_streak", 0)
            
            if win_streak > 0:
                # Increase by 10% per win in streak, cap at 50% increase
                increase_factor = min(1 + (win_streak * 0.1), 1.5)
                return base_position_size * increase_factor
            elif lose_streak > 0:
                # Decrease by 10% per loss in streak
                decrease_factor = max(1 - (lose_streak * 0.1), 0.5)
                return base_position_size * decrease_factor
            return base_position_size
            
        elif model_type == PositionSizingModel.REGIME_ADAPTIVE:
            # Adjust position size based on market regime
            if market_regime == MarketRegime.LOW_VOLATILITY:
                # Can take slightly larger positions in low volatility
                return base_position_size * 1.2
            elif market_regime == MarketRegime.NORMAL:
                return base_position_size
            elif market_regime == MarketRegime.HIGH_VOLATILITY:
                # Reduce position size in high volatility
                return base_position_size * 0.7
            elif market_regime == MarketRegime.CRISIS:
                # Significantly reduce position size in crisis
                return base_position_size * 0.4
            return base_position_size
            
        else:
            logger.warning(f"Unknown position sizing model: {model_type}")
            return base_position_size
    
    def _apply_position_constraints(self,
                                  position_size: float,
                                  symbol: str,
                                  account_balance: float,
                                  additional_params: Dict[str, Any]) -> float:
        """
        Apply minimum and maximum constraints to position size.
        
        Args:
            position_size: Calculated position size
            symbol: Trading symbol
            account_balance: Current account balance
            additional_params: Additional constraint parameters
            
        Returns:
            Constrained position size
        """
        # Get constraints from parameters or use defaults
        min_position = additional_params.get("min_position_size", 0.01)  # Minimum position size (e.g., 0.01 lots)
        max_position = additional_params.get("max_position_size", None)  # Maximum position size
        
        # Default max position based on account size if not specified
        if max_position is None:
            # Default to 5% of account balance as max position value
            max_account_percent = additional_params.get("max_account_percent", 5.0)
            unit_value = additional_params.get("unit_value", 100000)  # Default for standard lot in Forex
            max_position = (account_balance * (max_account_percent / 100)) / unit_value
        
        # Apply constraints
        constrained_size = max(min(position_size, max_position), min_position)
        
        return constrained_size
    
    def optimize_for_strategy(self,
                            strategy_id: str,
                            historical_trades: List[Dict[str, Any]],
                            risk_profile: str = "moderate") -> Dict[str, Any]:
        """
        Optimize position sizing parameters for a specific strategy based on historical performance.
        
        Args:
            strategy_id: Identifier of the strategy
            historical_trades: List of historical trades with results
            risk_profile: Risk profile ("conservative", "moderate", "aggressive")
            
        Returns:
            Dictionary with optimized parameters
        """
        if not historical_trades:
            logger.warning(f"No historical trades provided for strategy {strategy_id}")
            return {
                "recommended_model": PositionSizingModel.FIXED,
                "base_risk_percentage": 1.0,
                "optimization_confidence": "low"
            }
        
        # Calculate performance metrics
        wins = [t for t in historical_trades if t.get("outcome") == "win"]
        losses = [t for t in historical_trades if t.get("outcome") == "loss"]
        
        win_rate = len(wins) / len(historical_trades) if historical_trades else 0.5
        avg_win = np.mean([t.get("profit_pips", 0) for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.get("loss_pips", 0) for t in losses])) if losses else 0
        
        # Store performance metrics for future reference
        self.performance_history[strategy_id] = {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": len(historical_trades),
            "last_updated": datetime.now()
        }
        
        # Determine optimal position sizing model
        if win_rate > 0.55 and avg_win > avg_loss:
            recommended_model = PositionSizingModel.KELLY
        elif win_rate > 0.6:
            recommended_model = PositionSizingModel.ANTI_MARTINGALE
        else:
            recommended_model = PositionSizingModel.REGIME_ADAPTIVE
        
        # Determine base risk percentage based on risk profile and performance
        if risk_profile == "conservative":
            base_risk = 1.0
        elif risk_profile == "moderate":
            base_risk = 1.5
        else:  # aggressive
            base_risk = 2.0
            
        # Adjust risk based on win rate and reward:risk ratio
        reward_risk = avg_win / avg_loss if avg_loss > 0 else 1
        risk_adjustment = (win_rate - 0.5) * 2  # Scale from -1 to +1
        
        adjusted_risk = base_risk * (1 + (risk_adjustment * 0.5))
        
        # Ensure risk stays within reasonable bounds
        final_risk = max(min(adjusted_risk, 3.0), 0.5)
        
        return {
            "recommended_model": recommended_model,
            "base_risk_percentage": round(final_risk, 2),
            "expected_win_rate": round(win_rate, 2),
            "reward_risk_ratio": round(reward_risk, 2),
            "kelly_fraction": round((win_rate - ((1 - win_rate) / reward_risk)), 2) if reward_risk > 0 else 0,
            "optimization_confidence": "high" if len(historical_trades) > 50 else "medium" if len(historical_trades) > 20 else "low"
        }
    
    def get_volatility_adjusted_risk(self,
                                   symbol: str,
                                   base_risk_percentage: float,
                                   volatility_data: Dict[str, Any]) -> float:
        """
        Adjust risk percentage based on current market volatility.
        
        Args:
            symbol: Trading symbol
            base_risk_percentage: Base risk percentage to adjust
            volatility_data: Data about current and historical volatility
            
        Returns:
            Volatility-adjusted risk percentage
        """
        current_volatility = volatility_data.get("current_volatility", 0)
        avg_volatility = volatility_data.get("avg_volatility", 0)
        
        if avg_volatility <= 0:
            return base_risk_percentage
            
        # Calculate volatility ratio
        volatility_ratio = current_volatility / avg_volatility
        
        # Adjust risk inversely to volatility
        if volatility_ratio < 0.8:
            # Lower volatility, can increase risk slightly
            adjustment = 1.2
        elif volatility_ratio < 1.2:
            # Normal volatility, no change
            adjustment = 1.0
        elif volatility_ratio < 1.5:
            # Higher volatility, reduce risk
            adjustment = 0.8
        else:
            # Much higher volatility, reduce risk significantly
            adjustment = 0.6
            
        return base_risk_percentage * adjustment
    
    def analyze_correlated_exposure(self,
                                  active_positions: List[Dict[str, Any]],
                                  correlation_matrix: Dict[str, Dict[str, float]],
                                  new_position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlated exposure across active positions to adjust new position size.
        
        Args:
            active_positions: List of currently active positions
            correlation_matrix: Matrix of correlations between symbols
            new_position: Details of the new position to be opened
            
        Returns:
            Analysis results with recommended position size adjustment
        """
        if not active_positions or not new_position:
            return {"adjustment_factor": 1.0, "correlated_exposure": 0.0}
            
        new_symbol = new_position.get("symbol")
        new_direction = new_position.get("direction", "buy")  # "buy" or "sell"
        
        # Calculate existing exposure considering correlations
        total_correlated_exposure = 0
        
        for position in active_positions:
            pos_symbol = position.get("symbol")
            pos_direction = position.get("direction", "buy")
            pos_size = position.get("position_size", 0)
            
            # Skip if same symbol (will be handled by position limits)
            if pos_symbol == new_symbol:
                continue
                
            # Get correlation between position symbol and new symbol
            correlation = correlation_matrix.get(new_symbol, {}).get(pos_symbol, 0)
            
            # Account for direction (positive correlation but opposite directions offset)
            direction_factor = 1 if pos_direction == new_direction else -1
            
            # Add to total correlated exposure
            total_correlated_exposure += pos_size * correlation * direction_factor
        
        # Determine adjustment factor based on correlated exposure
        # If total_correlated_exposure is positive, reduce new position size
        # If negative, can potentially increase new position size
        
        if total_correlated_exposure > 0:
            # Scale down based on exposure (0.5 to 1.0)
            adjustment_factor = max(0.5, 1.0 - (total_correlated_exposure * 0.1))
        elif total_correlated_exposure < 0:
            # Potential to scale up but be conservative (1.0 to 1.2)
            adjustment_factor = min(1.2, 1.0 + abs(total_correlated_exposure) * 0.05)
        else:
            adjustment_factor = 1.0
            
        return {
            "adjustment_factor": adjustment_factor,
            "correlated_exposure": total_correlated_exposure,
            "adjusted_position_size": new_position.get("position_size", 0) * adjustment_factor
        }
""""""
