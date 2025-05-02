import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class RiskOptimizationStrategy(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"
    PERSONALIZED = "personalized"

class MarketRegime(str, Enum):
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"

@dataclass
class RiskParameters:
    """Standard risk parameters structure used across the system"""
    max_position_size: float  # Maximum position size as percentage of account
    stop_loss_pct: float  # Stop loss percentage from entry
    take_profit_pct: float  # Take profit percentage from entry
    max_drawdown_pct: float  # Maximum allowed drawdown percentage
    position_sizing_factor: float  # Position sizing multiplier (for Kelly etc.)
    risk_per_trade_pct: float  # Risk percentage per trade
    correlation_limit: float  # Maximum correlation between positions
    max_trades_per_day: int  # Maximum number of trades per day
    min_risk_reward_ratio: float  # Minimum risk-reward ratio for new trades
    max_leverage: float  # Maximum leverage allowed
    
    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Convert parameters to dictionary"""
        return {
            "max_position_size": self.max_position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "position_sizing_factor": self.position_sizing_factor,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "correlation_limit": self.correlation_limit,
            "max_trades_per_day": self.max_trades_per_day,
            "min_risk_reward_ratio": self.min_risk_reward_ratio,
            "max_leverage": self.max_leverage
        }


class RLRiskParameterOptimizer:
    """
    Optimizes risk parameters based on reinforcement learning model insights.
    This component adapts risk management settings dynamically based on the
    RL model's confidence and market conditions.
    """
    
    def __init__(self, 
                 base_parameters: Optional[RiskParameters] = None,
                 optimization_strategy: RiskOptimizationStrategy = RiskOptimizationStrategy.MODERATE,
                 adapt_to_market_regimes: bool = True,
                 adapt_to_model_confidence: bool = True,
                 confidence_threshold: float = 0.7,
                 max_adjustment_pct: float = 0.5,  # Maximum 50% adjustment
                 min_data_points: int = 20,  # Minimum data points before adaptation
                 ):
        """
        Initialize the RL Risk Parameter Optimizer
        
        Parameters:
        -----------
        base_parameters : RiskParameters, optional
            Base risk parameters to start from
        optimization_strategy : RiskOptimizationStrategy
            Overall strategy for risk optimization (conservative to aggressive)
        adapt_to_market_regimes : bool
            Whether to adapt parameters based on detected market regimes
        adapt_to_model_confidence : bool
            Whether to adapt parameters based on the model's confidence
        confidence_threshold : float
            Threshold above which the model is considered confident
        max_adjustment_pct : float
            Maximum percentage by which parameters can be adjusted
        min_data_points : int
            Minimum number of data points needed before adaptation
        """
        # Default base parameters if none provided
        self.base_parameters = base_parameters or RiskParameters(
            max_position_size=0.05,  # 5% of account
            stop_loss_pct=0.02,  # 2% stop loss
            take_profit_pct=0.04,  # 4% take profit (2:1 RR ratio)
            max_drawdown_pct=0.10,  # 10% max drawdown
            position_sizing_factor=0.5,  # 50% of Kelly criterion
            risk_per_trade_pct=0.01,  # 1% risk per trade
            correlation_limit=0.7,  # 70% max correlation
            max_trades_per_day=5,  # Max 5 trades per day
            min_risk_reward_ratio=2.0,  # Minimum 2:1 risk-reward
            max_leverage=10.0,  # Max 10x leverage
        )
        
        self.optimization_strategy = optimization_strategy
        self.adapt_to_market_regimes = adapt_to_market_regimes
        self.adapt_to_model_confidence = adapt_to_model_confidence
        self.confidence_threshold = confidence_threshold
        self.max_adjustment_pct = max_adjustment_pct
        self.min_data_points = min_data_points
        
        # Storage for historical data
        self.model_confidence_history = []
        self.market_regime_history = []
        self.parameter_history = []
        self.trade_outcomes = []
        
        # Initialize performance metrics
        self.performance_metrics = {
            "win_rate": 0.5,
            "avg_profit": 0.0,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.0
        }
        
        # Strategy-based parameter adjustments
        self._apply_strategy_adjustments()
        
        logger.info(f"Initialized RL Risk Parameter Optimizer with {optimization_strategy} strategy")
    
    def _apply_strategy_adjustments(self):
        """Apply initial adjustments based on the selected optimization strategy"""
        if self.optimization_strategy == RiskOptimizationStrategy.CONSERVATIVE:
            # Reduce risk for conservative strategy
            self.base_parameters.max_position_size *= 0.7
            self.base_parameters.risk_per_trade_pct *= 0.7
            self.base_parameters.position_sizing_factor *= 0.7
            self.base_parameters.max_leverage *= 0.5
            self.base_parameters.min_risk_reward_ratio *= 1.5  # Higher RR required
            
        elif self.optimization_strategy == RiskOptimizationStrategy.AGGRESSIVE:
            # Increase risk for aggressive strategy
            self.base_parameters.max_position_size *= 1.3
            self.base_parameters.risk_per_trade_pct *= 1.3
            self.base_parameters.position_sizing_factor *= 1.3
            self.base_parameters.max_leverage *= 1.5
            self.base_parameters.max_trades_per_day *= 1.5
            self.base_parameters.min_risk_reward_ratio *= 0.75  # Lower RR accepted
    
    def update_model_confidence(self, timestamp: datetime, confidence: float):
        """
        Update the model's confidence history
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp of the confidence measurement
        confidence : float
            Model's confidence score (0-1)
        """
        self.model_confidence_history.append({
            "timestamp": timestamp,
            "confidence": confidence
        })
        
        # Keep only recent history
        cutoff = timestamp - timedelta(days=30)
        self.model_confidence_history = [
            entry for entry in self.model_confidence_history
            if entry["timestamp"] >= cutoff
        ]
    
    def update_market_regime(self, timestamp: datetime, regime: MarketRegime):
        """
        Update the market regime history
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp of the regime detection
        regime : MarketRegime
            Detected market regime
        """
        self.market_regime_history.append({
            "timestamp": timestamp,
            "regime": regime
        })
        
        # Keep only recent history
        cutoff = timestamp - timedelta(days=30)
        self.market_regime_history = [
            entry for entry in self.market_regime_history
            if entry["timestamp"] >= cutoff
        ]
    
    def add_trade_outcome(self, 
                          timestamp: datetime,
                          profit_loss: float,
                          parameters_used: RiskParameters,
                          market_regime: Optional[MarketRegime] = None,
                          model_confidence: Optional[float] = None):
        """
        Add a trade outcome to the history
        
        Parameters:
        -----------
        timestamp : datetime
            Trade execution timestamp
        profit_loss : float
            Profit/loss from the trade
        parameters_used : RiskParameters
            Risk parameters used for this trade
        market_regime : MarketRegime, optional
            Market regime during the trade
        model_confidence : float, optional
            Model confidence when taking the trade
        """
        self.trade_outcomes.append({
            "timestamp": timestamp,
            "profit_loss": profit_loss,
            "parameters_used": parameters_used.to_dict(),
            "market_regime": market_regime,
            "model_confidence": model_confidence
        })
        
        # Keep only recent history
        cutoff = timestamp - timedelta(days=90)  # Keep 90 days of trade history
        self.trade_outcomes = [
            entry for entry in self.trade_outcomes
            if entry["timestamp"] >= cutoff
        ]
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update performance metrics based on trade outcomes"""
        if not self.trade_outcomes:
            return
        
        # Calculate last 30 days metrics
        cutoff = datetime.now() - timedelta(days=30)
        recent_trades = [
            trade for trade in self.trade_outcomes
            if trade["timestamp"] >= cutoff
        ]
        
        if not recent_trades:
            return
            
        # Win rate
        wins = sum(1 for trade in recent_trades if trade["profit_loss"] > 0)
        self.performance_metrics["win_rate"] = wins / len(recent_trades)
        
        # Average profit
        self.performance_metrics["avg_profit"] = np.mean(
            [trade["profit_loss"] for trade in recent_trades]
        )
        
        # Sharpe ratio (simplified)
        returns = [trade["profit_loss"] for trade in recent_trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1.0
        self.performance_metrics["sharpe_ratio"] = (
            mean_return / std_return if std_return > 0 else 0
        )
        
        # Max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        self.performance_metrics["max_drawdown"] = abs(min(drawdown, default=0))
    
    def get_current_market_regime(self) -> Optional[MarketRegime]:
        """Get the most recent market regime"""
        if not self.market_regime_history:
            return None
        
        self.market_regime_history.sort(key=lambda x: x["timestamp"], reverse=True)
        return self.market_regime_history[0]["regime"]
    
    def get_average_model_confidence(self, days: int = 1) -> float:
        """
        Get the average model confidence over the specified recent period
        
        Parameters:
        -----------
        days : int
            Number of days to look back
            
        Returns:
        --------
        float
            Average confidence score (0-1)
        """
        if not self.model_confidence_history:
            return 0.5  # Default to neutral confidence
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_confidence = [
            entry["confidence"] for entry in self.model_confidence_history
            if entry["timestamp"] >= cutoff
        ]
        
        if not recent_confidence:
            return 0.5
        
        return np.mean(recent_confidence)
    
    def optimize_parameters(self, 
                           current_time: Optional[datetime] = None,
                           current_regime: Optional[MarketRegime] = None,
                           current_confidence: Optional[float] = None) -> RiskParameters:
        """
        Optimize risk parameters based on current conditions and history
        
        Parameters:
        -----------
        current_time : datetime, optional
            Current timestamp (defaults to now)
        current_regime : MarketRegime, optional
            Current market regime (if provided overrides history)
        current_confidence : float, optional
            Current model confidence (if provided overrides history)
            
        Returns:
        --------
        RiskParameters
            Optimized risk parameters
        """
        current_time = current_time or datetime.now()
        
        # Make a copy of base parameters to optimize
        optimized = RiskParameters(**self.base_parameters.to_dict())
        
        # Apply market regime adjustments if enabled
        if self.adapt_to_market_regimes:
            regime = current_regime or self.get_current_market_regime()
            if regime:
                optimized = self._adjust_for_market_regime(optimized, regime)
        
        # Apply model confidence adjustments if enabled
        if self.adapt_to_model_confidence:
            confidence = (
                current_confidence 
                if current_confidence is not None 
                else self.get_average_model_confidence()
            )
            optimized = self._adjust_for_model_confidence(optimized, confidence)
        
        # Apply performance-based adjustments
        if len(self.trade_outcomes) >= self.min_data_points:
            optimized = self._adjust_for_performance(optimized)
            
        # Apply safety limits
        optimized = self._apply_safety_limits(optimized)
        
        # Record the optimized parameters
        self.parameter_history.append({
            "timestamp": current_time,
            "parameters": optimized.to_dict(),
            "regime": current_regime,
            "confidence": current_confidence
        })
        
        return optimized
    
    def _adjust_for_market_regime(self, 
                                 params: RiskParameters, 
                                 regime: MarketRegime) -> RiskParameters:
        """
        Adjust parameters based on the current market regime
        
        Parameters:
        -----------
        params : RiskParameters
            Current parameters to adjust
        regime : MarketRegime
            Current market regime
            
        Returns:
        --------
        RiskParameters
            Adjusted parameters for the market regime
        """
        # Create a copy to avoid modifying the original
        adjusted = RiskParameters(**params.to_dict())
        
        # Apply regime-specific adjustments
        if regime == MarketRegime.TRENDING_BULLISH:
            # Larger positions, wider stops in bullish trends
            adjusted.max_position_size *= 1.2
            adjusted.stop_loss_pct *= 1.1
            adjusted.take_profit_pct *= 1.2
            adjusted.position_sizing_factor *= 1.1
            
        elif regime == MarketRegime.TRENDING_BEARISH:
            # Similar adjustments for bearish trends
            adjusted.max_position_size *= 1.1
            adjusted.stop_loss_pct *= 1.1
            adjusted.position_sizing_factor *= 1.1
            
        elif regime == MarketRegime.RANGING:
            # Tighter stops in ranging markets
            adjusted.stop_loss_pct *= 0.8
            adjusted.take_profit_pct *= 0.8
            adjusted.max_position_size *= 0.9
            
        elif regime == MarketRegime.VOLATILE:
            # Smaller positions, wider stops in volatile markets
            adjusted.max_position_size *= 0.7
            adjusted.risk_per_trade_pct *= 0.8
            adjusted.stop_loss_pct *= 1.3
            adjusted.take_profit_pct *= 1.3
            adjusted.max_trades_per_day = int(adjusted.max_trades_per_day * 0.7)
            
        elif regime == MarketRegime.BREAKOUT:
            # Larger positions for breakouts
            adjusted.max_position_size *= 1.2
            adjusted.take_profit_pct *= 1.3
            
        elif regime == MarketRegime.CRISIS:
            # Dramatic risk reduction in crisis
            adjusted.max_position_size *= 0.5
            adjusted.risk_per_trade_pct *= 0.5
            adjusted.max_leverage *= 0.5
            adjusted.max_trades_per_day = int(adjusted.max_trades_per_day * 0.5)
            adjusted.stop_loss_pct *= 0.7
            
        return adjusted
    
    def _adjust_for_model_confidence(self, 
                                    params: RiskParameters, 
                                    confidence: float) -> RiskParameters:
        """
        Adjust parameters based on the model's confidence
        
        Parameters:
        -----------
        params : RiskParameters
            Current parameters to adjust
        confidence : float
            Model's confidence score (0-1)
            
        Returns:
        --------
        RiskParameters
            Adjusted parameters based on confidence
        """
        # Create a copy to avoid modifying the original
        adjusted = RiskParameters(**params.to_dict())
        
        # Skip adjustment if confidence is near neutral
        if 0.4 <= confidence <= 0.6:
            return adjusted
        
        # Calculate confidence factor (-1 to 1) where 0.5 is neutral
        confidence_factor = (confidence - 0.5) * 2
        
        # Apply confidence-based adjustments
        # Higher confidence = bigger position sizes and risk
        adjustment = max(min(confidence_factor, self.max_adjustment_pct), -self.max_adjustment_pct)
        
        # Apply adjustments when confidence is above threshold
        if abs(confidence - 0.5) > abs(self.confidence_threshold - 0.5):
            factor = 1.0 + adjustment
            adjusted.max_position_size *= factor
            adjusted.position_sizing_factor *= factor
            adjusted.risk_per_trade_pct *= factor
            
            # When confidence is very high, also adjust stop distances
            if confidence > 0.8:
                adjusted.stop_loss_pct *= 1.1
                adjusted.take_profit_pct *= 1.1
            
            # When confidence is very low, tighten stops
            if confidence < 0.3:
                adjusted.stop_loss_pct *= 0.9
                adjusted.take_profit_pct *= 0.9
        
        return adjusted
    
    def _adjust_for_performance(self, params: RiskParameters) -> RiskParameters:
        """
        Adjust parameters based on recent performance
        
        Parameters:
        -----------
        params : RiskParameters
            Current parameters to adjust
            
        Returns:
        --------
        RiskParameters
            Adjusted parameters based on performance
        """
        # Create a copy to avoid modifying the original
        adjusted = RiskParameters(**params.to_dict())
        
        # Performance-based adjustments
        win_rate = self.performance_metrics["win_rate"]
        sharpe_ratio = self.performance_metrics["sharpe_ratio"]
        max_drawdown = self.performance_metrics["max_drawdown"]
        
        # Win rate adjustments
        if win_rate > 0.6:  # Good performance
            adjusted.position_sizing_factor *= min(1.1, 1 + (win_rate - 0.6) * 0.5)
        elif win_rate < 0.4:  # Poor performance
            adjusted.position_sizing_factor *= max(0.9, 1 - (0.4 - win_rate) * 0.5)
        
        # Sharpe ratio adjustments
        if sharpe_ratio > 1.5:  # Good risk-adjusted returns
            adjusted.max_position_size *= min(1.1, 1 + (sharpe_ratio - 1.5) * 0.1)
        elif sharpe_ratio < 0.5:  # Poor risk-adjusted returns
            adjusted.max_position_size *= max(0.9, 1 - (0.5 - sharpe_ratio) * 0.1)
        
        # Drawdown adjustments
        if max_drawdown > adjusted.max_drawdown_pct:
            # Reduce risk if drawdown exceeds tolerance
            reduction = max(0.8, 1 - (max_drawdown - adjusted.max_drawdown_pct) * 2)
            adjusted.risk_per_trade_pct *= reduction
            adjusted.max_position_size *= reduction
            adjusted.max_leverage *= reduction
        
        return adjusted
    
    def _apply_safety_limits(self, params: RiskParameters) -> RiskParameters:
        """
        Apply safety limits to prevent extreme parameter values
        
        Parameters:
        -----------
        params : RiskParameters
            Parameters to apply safety limits to
            
        Returns:
        --------
        RiskParameters
            Parameters with safety limits applied
        """
        # Create a copy to avoid modifying the original
        safe = RiskParameters(**params.to_dict())
        
        # Apply absolute limits
        safe.max_position_size = min(safe.max_position_size, 0.2)  # Max 20% position
        safe.risk_per_trade_pct = min(safe.risk_per_trade_pct, 0.05)  # Max 5% risk per trade
        safe.max_leverage = min(safe.max_leverage, 30.0)  # Max 30x leverage
        
        # Apply lower bounds
        safe.max_position_size = max(safe.max_position_size, 0.01)  # Min 1% position
        safe.risk_per_trade_pct = max(safe.risk_per_trade_pct, 0.001)  # Min 0.1% risk
        safe.stop_loss_pct = max(safe.stop_loss_pct, 0.005)  # Min 0.5% stop
        safe.position_sizing_factor = max(safe.position_sizing_factor, 0.1)  # Min 10% of Kelly
        
        return safe
    
    def get_parameter_history(self, 
                             days: int = 30) -> pd.DataFrame:
        """
        Get history of parameter adjustments
        
        Parameters:
        -----------
        days : int
            Number of days of history to return
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with parameter history
        """
        if not self.parameter_history:
            return pd.DataFrame()
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_history = [
            entry for entry in self.parameter_history
            if entry["timestamp"] >= cutoff
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_history)
        
        # Expand parameters dictionary to columns
        param_df = pd.json_normalize(df['parameters'])
        df = pd.concat([df.drop('parameters', axis=1), param_df], axis=1)
        
        return df
    
    def analyze_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of parameter adjustments
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with effectiveness analysis
        """
        if len(self.trade_outcomes) < self.min_data_points:
            return {"status": "insufficient_data", 
                    "message": f"Need at least {self.min_data_points} trades for analysis"}
        
        # Convert trade outcomes to DataFrame
        trades_df = pd.DataFrame(self.trade_outcomes)
        
        # Analyze by market regime if available
        regime_analysis = {}
        if self.adapt_to_market_regimes and 'market_regime' in trades_df.columns:
            for regime, group in trades_df.groupby('market_regime'):
                if len(group) < 5:  # Skip regimes with too few trades
                    continue
                
                win_rate = (group['profit_loss'] > 0).mean()
                avg_profit = group['profit_loss'].mean()
                
                regime_analysis[regime] = {
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "trade_count": len(group)
                }
        
        # Analyze by confidence level if available
        confidence_analysis = {}
        if self.adapt_to_model_confidence and 'model_confidence' in trades_df.columns:
            # Create confidence bins
            trades_df['confidence_bin'] = pd.cut(
                trades_df['model_confidence'], 
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=['very_low', 'low', 'medium', 'high']
            )
            
            for conf_bin, group in trades_df.groupby('confidence_bin'):
                if len(group) < 5:  # Skip bins with too few trades
                    continue
                
                win_rate = (group['profit_loss'] > 0).mean()
                avg_profit = group['profit_loss'].mean()
                
                confidence_analysis[conf_bin] = {
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "trade_count": len(group)
                }
        
        # Analyze parameter effectiveness
        param_correlation = {}
        for param in RiskParameters.__dataclass_fields__:
            if param not in trades_df.columns:
                # Extract parameter from the parameters_used dictionary
                if 'parameters_used' in trades_df.columns:
                    trades_df[param] = trades_df['parameters_used'].apply(
                        lambda x: x.get(param) if isinstance(x, dict) else None
                    )
            
            if param in trades_df.columns:
                # Calculate correlation with profit
                correlation = trades_df[[param, 'profit_loss']].corr().iloc[0, 1]
                param_correlation[param] = correlation
        
        return {
            "overall_metrics": self.performance_metrics,
            "regime_analysis": regime_analysis,
            "confidence_analysis": confidence_analysis,
            "parameter_correlation": param_correlation,
            "trade_count": len(self.trade_outcomes)
        }


class DynamicRiskAdapter:
    """
    Integrates RL-based risk parameter optimization with the broader risk management system.
    This adapter serves as a bridge between the RL optimizer and existing risk management
    components, providing a unified interface for dynamic risk adaptation.
    """
    
    def __init__(self, 
                 default_parameters: Optional[RiskParameters] = None,
                 adaptation_speed: float = 0.5,  # How quickly to adapt (0-1)
                 strategy_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the Dynamic Risk Adapter
        
        Parameters:
        -----------
        default_parameters : RiskParameters, optional
            Default risk parameters
        adaptation_speed : float
            How quickly to adapt to new recommendations (0-1)
        strategy_weights : Dict[str, float], optional
            Weights for different optimization strategies
        """
        self.default_parameters = default_parameters or RiskParameters(
            max_position_size=0.05,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            max_drawdown_pct=0.10,
            position_sizing_factor=0.5,
            risk_per_trade_pct=0.01,
            correlation_limit=0.7,
            max_trades_per_day=5,
            min_risk_reward_ratio=2.0,
            max_leverage=10.0,
        )
        
        self.adaptation_speed = adaptation_speed
        self.strategy_weights = strategy_weights or {
            RiskOptimizationStrategy.CONSERVATIVE.value: 0.3,
            RiskOptimizationStrategy.MODERATE.value: 0.5,
            RiskOptimizationStrategy.AGGRESSIVE.value: 0.2
        }
        
        # Initialize optimizers for each strategy
        self.optimizers = {
            strategy: RLRiskParameterOptimizer(
                base_parameters=default_parameters,
                optimization_strategy=strategy
            )
            for strategy in self.strategy_weights.keys()
        }
        
        # Current parameters in use
        self.current_parameters = RiskParameters(**default_parameters.to_dict())
        
        # History of parameter changes
        self.parameter_history = []
        
        logger.info("Initialized DynamicRiskAdapter")
    
    def update_model_data(self, 
                         timestamp: datetime, 
                         confidence: float, 
                         market_regime: MarketRegime):
        """
        Update model data across all optimizers
        
        Parameters:
        -----------
        timestamp : datetime
            Current timestamp
        confidence : float
            Model confidence score
        market_regime : MarketRegime
            Current market regime
        """
        for optimizer in self.optimizers.values():
            optimizer.update_model_confidence(timestamp, confidence)
            optimizer.update_market_regime(timestamp, market_regime)
    
    def record_trade_outcome(self,
                            timestamp: datetime,
                            profit_loss: float,
                            market_regime: Optional[MarketRegime] = None,
                            model_confidence: Optional[float] = None):
        """
        Record a trade outcome across all optimizers
        
        Parameters:
        -----------
        timestamp : datetime
            Trade timestamp
        profit_loss : float
            Profit/loss from the trade
        market_regime : MarketRegime, optional
            Market regime during the trade
        model_confidence : float, optional
            Model confidence when taking the trade
        """
        for optimizer in self.optimizers.values():
            optimizer.add_trade_outcome(
                timestamp=timestamp,
                profit_loss=profit_loss,
                parameters_used=self.current_parameters,
                market_regime=market_regime,
                model_confidence=model_confidence
            )
    
    def get_adapted_parameters(self,
                              timestamp: Optional[datetime] = None,
                              current_regime: Optional[MarketRegime] = None,
                              current_confidence: Optional[float] = None) -> RiskParameters:
        """
        Get risk parameters adapted to current conditions
        
        Parameters:
        -----------
        timestamp : datetime, optional
            Current timestamp
        current_regime : MarketRegime, optional
            Current market regime
        current_confidence : float, optional
            Current model confidence
            
        Returns:
        --------
        RiskParameters
            Adapted risk parameters
        """
        timestamp = timestamp or datetime.now()
        
        # Get recommendations from each optimizer
        recommendations = {}
        for strategy, optimizer in self.optimizers.items():
            recommendations[strategy] = optimizer.optimize_parameters(
                current_time=timestamp,
                current_regime=current_regime,
                current_confidence=current_confidence
            )
        
        # Blend recommendations according to weights
        blended_params = self._blend_parameters(recommendations)
        
        # Apply adaptation speed to smooth transitions
        adapted_params = self._adapt_parameters(blended_params)
        
        # Record the parameter update
        self.parameter_history.append({
            "timestamp": timestamp,
            "parameters": adapted_params.to_dict(),
            "regime": current_regime,
            "confidence": current_confidence
        })
        
        return adapted_params
    
    def _blend_parameters(self, 
                         recommendations: Dict[str, RiskParameters]) -> RiskParameters:
        """
        Blend multiple parameter recommendations according to strategy weights
        
        Parameters:
        -----------
        recommendations : Dict[str, RiskParameters]
            Parameter recommendations from different strategies
            
        Returns:
        --------
        RiskParameters
            Blended risk parameters
        """
        # Start with default values
        blended = {}
        
        # For each parameter field, calculate weighted average
        for field in RiskParameters.__dataclass_fields__:
            # Initialize accumulator
            weighted_sum = 0.0
            weight_total = 0.0
            
            for strategy, params in recommendations.items():
                if strategy in self.strategy_weights:
                    weight = self.strategy_weights[strategy]
                    value = getattr(params, field)
                    weighted_sum += value * weight
                    weight_total += weight
            
            # Normalize if weights don't sum to 1
            if weight_total > 0:
                blended[field] = weighted_sum / weight_total
            else:
                # Fallback to default if no weights
                blended[field] = getattr(self.default_parameters, field)
        
        # Handle integer parameters
        if 'max_trades_per_day' in blended:
            blended['max_trades_per_day'] = int(round(blended['max_trades_per_day']))
        
        return RiskParameters(**blended)
    
    def _adapt_parameters(self, target_params: RiskParameters) -> RiskParameters:
        """
        Gradually adapt current parameters toward target parameters
        
        Parameters:
        -----------
        target_params : RiskParameters
            Target risk parameters to adapt toward
            
        Returns:
        --------
        RiskParameters
            Adapted risk parameters
        """
        # Create a new parameter set
        adapted = {}
        
        # For each parameter, move toward target based on adaptation speed
        for field in RiskParameters.__dataclass_fields__:
            current_value = getattr(self.current_parameters, field)
            target_value = getattr(target_params, field)
            
            # Linear interpolation
            adapted_value = current_value + (target_value - current_value) * self.adaptation_speed
            adapted[field] = adapted_value
        
        # Handle integer parameters
        if 'max_trades_per_day' in adapted:
            adapted['max_trades_per_day'] = int(round(adapted['max_trades_per_day']))
        
        # Update current parameters
        self.current_parameters = RiskParameters(**adapted)
        
        return self.current_parameters
    
    def get_parameter_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get history of parameter adaptations
        
        Parameters:
        -----------
        days : int
            Number of days of history to return
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with parameter history
        """
        if not self.parameter_history:
            return pd.DataFrame()
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_history = [
            entry for entry in self.parameter_history
            if entry["timestamp"] >= cutoff
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_history)
        
        # Expand parameters dictionary to columns
        param_df = pd.json_normalize(df['parameters'])
        df = pd.concat([df.drop('parameters', axis=1), param_df], axis=1)
        
        return df
    
    def analyze_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze effectiveness of the dynamic risk adaptation
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with effectiveness analysis
        """
        results = {}
        
        for strategy, optimizer in self.optimizers.items():
            results[strategy] = optimizer.analyze_effectiveness()
        
        # Add blended strategy analysis
        if self.parameter_history:
            df = self.get_parameter_history()
            
            if 'profit_loss' in df.columns and len(df) > 20:
                # Calculate performance of blended approach
                win_rate = (df['profit_loss'] > 0).mean()
                avg_profit = df['profit_loss'].mean()
                sharpe = df['profit_loss'].mean() / df['profit_loss'].std() if df['profit_loss'].std() > 0 else 0
                
                results["blended"] = {
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "sharpe_ratio": sharpe,
                    "trade_count": len(df)
                }
        
        return results
