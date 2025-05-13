"""
Dynamic Risk Adjuster Service Module.

Provides functionality for dynamically adjusting risk parameters based on market conditions,
account performance, and volatility.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from core.risk_limits import (
    RiskLimit, RiskLimitUpdate, LimitType
)
from core.risk_calculator import RiskCalculator
from repositories.risk_repository import RiskRepository
from repositories.limits_repository import LimitsRepository
from core.connection import DatabaseSession

logger = get_logger("dynamic-risk-adjuster")


class MarketRegime(str):
    """Market regime types for risk adjustment."""
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NORMAL = "NORMAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CRISIS = "CRISIS"


class DynamicRiskAdjuster:
    """
    Service for dynamically adjusting risk parameters based on market conditions,
    account performance, and volatility.
    
    This service provides functionality to:
    - Analyze current market conditions and volatility
    - Adjust risk limits based on account performance
    - Implement adaptive position sizing
    - Provide volatility-based exposure scaling
    - Identify weaknesses in trading strategies
    - Generate risk-awareness metrics for machine learning integration
    - Implement proactive risk monitoring and alerts
    """
    
    def __init__(self):
        """Initialize the dynamic risk adjuster service."""
        self.risk_calculator = RiskCalculator()
    
    def analyze_market_regime(self, 
                              volatility_data: Dict[str, List[float]], 
                              market_data: Dict[str, Any]) -> MarketRegime:
        """
        Analyze current market regime based on volatility and market data.
        
        Args:
            volatility_data: Dictionary mapping symbols to their historical volatility
            market_data: Additional market data indicators
            
        Returns:
            Current market regime classification
        """
        # Calculate average volatility across symbols
        avg_volatility = np.mean([np.mean(vol_list) for vol_list in volatility_data.values()])
        
        # Get recent volatility (last 5 days)
        recent_volatility = np.mean([np.mean(vol_list[-5:]) for vol_list in volatility_data.values() 
                                    if len(vol_list) >= 5])
        
        # Calculate volatility ratio (recent vs historical)
        volatility_ratio = recent_volatility / avg_volatility if avg_volatility > 0 else 1.0
        
        # Determine market regime based on volatility ratio
        if volatility_ratio < 0.7:
            return MarketRegime.LOW_VOLATILITY
        elif volatility_ratio < 1.3:
            return MarketRegime.NORMAL
        elif volatility_ratio < 2.0:
            return MarketRegime.HIGH_VOLATILITY
        else:
            return MarketRegime.CRISIS
    
    def get_regime_risk_multiplier(self, regime: MarketRegime) -> float:
        """
        Get risk multiplier based on market regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Risk multiplier (percentage to adjust risk limits)
        """
        # Define risk multipliers for each regime
        multipliers = {
            MarketRegime.LOW_VOLATILITY: 1.2,      # Increase risk in low volatility
            MarketRegime.NORMAL: 1.0,              # Normal risk
            MarketRegime.HIGH_VOLATILITY: 0.7,     # Reduce risk in high volatility
            MarketRegime.CRISIS: 0.4               # Significantly reduce risk in crisis
        }
        
        return multipliers.get(regime, 1.0)
    
    def adjust_account_risk_limits(self, 
                                  account_id: str, 
                                  performance_metrics: Dict[str, Any],
                                  market_regime: MarketRegime) -> Dict[str, Any]:
        """
        Adjust risk limits based on account performance and market regime.
        
        Args:
            account_id: Account ID
            performance_metrics: Dictionary with account performance metrics
            market_regime: Current market regime
            
        Returns:
            Dictionary with adjustment results
        """
        with DatabaseSession() as session:
            limits_repo = LimitsRepository(session)
            account_limits = limits_repo.get_limits_by_account(account_id)
            
            # Get risk multiplier based on market regime
            regime_multiplier = self.get_regime_risk_multiplier(market_regime)
            
            # Get performance multiplier based on account metrics
            performance_multiplier = self._calculate_performance_multiplier(performance_metrics)
            
            # Combined multiplier
            combined_multiplier = regime_multiplier * performance_multiplier
            
            # Apply adjustments to each limit
            results = {}
            for limit in account_limits:
                # Skip limits that shouldn't be dynamically adjusted
                if limit.limit_type not in [LimitType.MAX_SINGLE_EXPOSURE_PCT, 
                                           LimitType.MAX_TOTAL_EXPOSURE_PCT]:
                    continue
                
                # Calculate new limit value
                new_limit_value = limit.limit_value * combined_multiplier
                
                # Apply min/max constraints
                if limit.limit_type == LimitType.MAX_SINGLE_EXPOSURE_PCT:
                    new_limit_value = max(1.0, min(10.0, new_limit_value))
                elif limit.limit_type == LimitType.MAX_TOTAL_EXPOSURE_PCT:
                    new_limit_value = max(5.0, min(50.0, new_limit_value))
                
                # Update the limit
                update_data = RiskLimitUpdate(limit_value=new_limit_value)
                updated_limit = limits_repo.update_limit(limit.id, update_data)
                
                if updated_limit:
                    results[limit.limit_type] = {
                        "previous_value": limit.limit_value,
                        "new_value": updated_limit.limit_value,
                        "multiplier_applied": combined_multiplier
                    }
            
            logger.info(f"Adjusted risk limits for account {account_id} with regime {market_regime}")
            return {
                "account_id": account_id,
                "market_regime": market_regime,
                "regime_multiplier": regime_multiplier,
                "performance_multiplier": performance_multiplier,
                "combined_multiplier": combined_multiplier,
                "adjustments": results
            }
    
    def _calculate_performance_multiplier(self, performance_metrics: Dict[str, Any]) -> float:
        """
        Calculate performance-based risk multiplier.
        
        Args:
            performance_metrics: Dictionary with account performance metrics
            
        Returns:
            Performance-based risk multiplier
        """
        # Extract relevant metrics
        win_rate = performance_metrics.get("win_rate", 0.5)
        sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
        profit_factor = performance_metrics.get("profit_factor", 1.0)
        
        # Calculate base multiplier from win rate
        win_rate_factor = 0.8 + (win_rate * 0.4)  # 0.8-1.2 range
        
        # Adjust based on Sharpe ratio
        sharpe_factor = 1.0
        if sharpe_ratio > 2.0:
            sharpe_factor = 1.2
        elif sharpe_ratio > 1.0:
            sharpe_factor = 1.1
        elif sharpe_ratio < 0:
            sharpe_factor = 0.8
        
        # Adjust based on profit factor
        profit_factor_adj = min(1.2, max(0.8, profit_factor / 2))
        
        # Combined performance multiplier
        return win_rate_factor * sharpe_factor * profit_factor_adj
    
    def calculate_position_size_adjustment(self,
                                         account_id: str,
                                         base_position_size: float,
                                         symbol: str,
                                         market_regime: MarketRegime,
                                         volatility_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate adjusted position size based on market conditions and volatility.
        
        Args:
            account_id: Account ID
            base_position_size: Base position size
            symbol: Trading symbol
            market_regime: Current market regime
            volatility_data: Dictionary mapping symbols to their historical volatility
            
        Returns:
            Dictionary with adjustment results
        """
        # Get regime-based risk multiplier
        regime_multiplier = self.get_regime_risk_multiplier(market_regime)
        
        # Get symbol-specific volatility adjustment
        volatility_adjustment = self._calculate_volatility_adjustment(symbol, volatility_data)
        
        # Calculate adjusted position size
        adjusted_position_size = base_position_size * regime_multiplier * volatility_adjustment
        
        # Ensure position size isn't reduced below a minimum threshold (20% of base)
        adjusted_position_size = max(base_position_size * 0.2, adjusted_position_size)
        
        return {
            "account_id": account_id,
            "symbol": symbol,
            "base_position_size": base_position_size,
            "adjusted_position_size": adjusted_position_size,
            "regime_multiplier": regime_multiplier,
            "volatility_adjustment": volatility_adjustment
        }
    
    def _calculate_volatility_adjustment(self, symbol: str, volatility_data: Dict[str, List[float]]) -> float:
        """
        Calculate volatility-based adjustment for a symbol.
        
        Args:
            symbol: Trading symbol
            volatility_data: Dictionary mapping symbols to their historical volatility
            
        Returns:
            Volatility adjustment factor
        """
        if symbol not in volatility_data or not volatility_data[symbol]:
            return 1.0
        
        # Get historical and recent volatility for the symbol
        vol_history = volatility_data[symbol]
        historical_vol = np.mean(vol_history)
        
        # Get recent volatility (last 5 days or what's available)
        recent_periods = min(5, len(vol_history))
        recent_vol = np.mean(vol_history[-recent_periods:])
        
        # Calculate inverse ratio of recent to historical volatility
        # Higher recent volatility = lower position size
        if historical_vol > 0:
            vol_ratio = historical_vol / recent_vol
            # Bound the adjustment between 0.5 and 1.5
            return max(0.5, min(1.5, vol_ratio))
        
        return 1.0
    
    def get_risk_recommendations(self,
                               account_id: str,
                               market_data: Dict[str, Any],
                               account_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive risk recommendations based on all factors.
        
        Args:
            account_id: Account ID
            market_data: Market data including volatility
            account_performance: Account performance metrics
            
        Returns:
            Dictionary with risk recommendations
        """
        # Analyze market regime
        market_regime = self.analyze_market_regime(
            market_data.get("volatility_data", {}),
            market_data
        )
        
        # Get drawdown info
        drawdown_info = self.risk_calculator.calculate_drawdown_risk(
            account_performance.get("equity_curve", []),
            max_drawdown_limit_pct=account_performance.get("max_drawdown_limit", 20.0)
        )
        
        # Generate recommendations
        exposure_recommendation = "maintain"
        if market_regime == MarketRegime.CRISIS:
            exposure_recommendation = "significantly_reduce"
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            exposure_recommendation = "reduce"
        elif market_regime == MarketRegime.LOW_VOLATILITY and drawdown_info["drawdown_risk_level"] == "LOW":
            exposure_recommendation = "increase"
        
        risk_per_trade_adjustment = 1.0
        if drawdown_info["drawdown_risk_level"] == "HIGH":
            risk_per_trade_adjustment = 0.7
        elif drawdown_info["drawdown_risk_level"] == "MEDIUM":
            risk_per_trade_adjustment = 0.85
        elif drawdown_info["drawdown_risk_level"] == "LOW":
            risk_per_trade_adjustment = 1.0
        
        return {
            "account_id": account_id,
            "market_regime": market_regime,
            "exposure_recommendation": exposure_recommendation,
            "risk_per_trade_adjustment": risk_per_trade_adjustment,
            "available_risk_pct": drawdown_info["available_risk_pct"],
            "drawdown_info": drawdown_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_strategy_weaknesses(self, 
                                  strategy_id: str,
                                  historical_performance: Dict[str, Any],
                                  market_regimes_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze strategy performance across different market conditions to identify weaknesses.
        
        Args:
            strategy_id: Unique identifier for the strategy
            historical_performance: Dictionary containing performance metrics across time
            market_regimes_history: List of historical market regime classifications
            
        Returns:
            Dictionary with strategy weakness analysis and recommendations
        """
        # Group performance by market regime
        performance_by_regime = {}
        regime_exposure = {}
        total_trades = 0
        
        for period, regime_data in enumerate(market_regimes_history):
            regime = regime_data["regime"]
            if regime not in performance_by_regime:
                performance_by_regime[regime] = {
                    "returns": [],
                    "sharpe": [],
                    "win_rate": [],
                    "trades": 0
                }
                regime_exposure[regime] = 0
            
            # Extract performance data for this period
            period_performance = historical_performance.get(f"period_{period}", {})
            if period_performance:
                performance_by_regime[regime]["returns"].append(period_performance.get("return", 0))
                performance_by_regime[regime]["sharpe"].append(period_performance.get("sharpe", 0))
                performance_by_regime[regime]["win_rate"].append(period_performance.get("win_rate", 0))
                period_trades = period_performance.get("trades", 0)
                performance_by_regime[regime]["trades"] += period_trades
                total_trades += period_trades
                regime_exposure[regime] += 1
        
        # Calculate average performance metrics by regime
        regime_performance = {}
        for regime, data in performance_by_regime.items():
            if data["returns"]:
                regime_performance[regime] = {
                    "avg_return": np.mean(data["returns"]),
                    "avg_sharpe": np.mean(data["sharpe"]),
                    "avg_win_rate": np.mean(data["win_rate"]),
                    "total_trades": data["trades"],
                    "exposure_pct": (regime_exposure[regime] / len(market_regimes_history)) * 100 if market_regimes_history else 0
                }
        
        # Identify underperforming regimes
        underperforming_regimes = []
        for regime, metrics in regime_performance.items():
            # Define underperformance criteria
            is_underperforming = False
            
            if metrics["avg_return"] < 0:
                is_underperforming = True
                
            if regime == MarketRegime.CRISIS and metrics["avg_return"] < -0.5:
                is_underperforming = True
                
            if regime == MarketRegime.HIGH_VOLATILITY and metrics["avg_sharpe"] < 0.3:
                is_underperforming = True
            
            if is_underperforming:
                underperforming_regimes.append({
                    "regime": regime,
                    "metrics": metrics,
                    "severity": "HIGH" if metrics["avg_return"] < -1.0 else "MEDIUM"
                })
        
        # Generate recommendations based on weaknesses
        risk_recommendations = []
        strategy_adaptations = []
        
        if underperforming_regimes:
            for regime_data in underperforming_regimes:
                regime = regime_data["regime"]
                severity = regime_data["severity"]
                
                if regime == MarketRegime.CRISIS:
                    risk_recommendations.append({
                        "type": "EXPOSURE_REDUCTION",
                        "description": "Significantly reduce exposure during crisis periods",
                        "priority": "HIGH" if severity == "HIGH" else "MEDIUM"
                    })
                    
                    strategy_adaptations.append({
                        "adaptation": "Add crisis-specific risk controls",
                        "description": "Implement tighter stop-losses during crisis periods"
                    })
                
                elif regime == MarketRegime.HIGH_VOLATILITY:
                    risk_recommendations.append({
                        "type": "POSITION_SIZING",
                        "description": "Reduce position sizes during high volatility",
                        "priority": "MEDIUM"
                    })
                    
                    strategy_adaptations.append({
                        "adaptation": "Add volatility filters",
                        "description": "Only trade when volatility is within optimal ranges"
                    })
        
        return {
            "strategy_id": strategy_id,
            "performance_by_regime": regime_performance,
            "underperforming_regimes": underperforming_regimes,
            "risk_recommendations": risk_recommendations,
            "strategy_adaptations": strategy_adaptations,
            "timestamp": datetime.now().isoformat()
        }

    def generate_risk_metrics_for_ml(self, account_id: str, timeframe: str = "daily") -> Dict[str, Any]:
        """
        Generate risk metrics for machine learning model integration.
        
        Args:
            account_id: Unique identifier for the account
            timeframe: Timeframe for risk metrics (daily, weekly, monthly)
            
        Returns:
            Dictionary with risk metrics formatted for ML consumption
        """
        with DatabaseSession() as session:
            risk_repo = RiskRepository(session)
            
            # Get historical risk metrics
            historical_metrics = risk_repo.get_risk_metrics_history(
                account_id=account_id,
                days=30 if timeframe == "daily" else 90
            )
            
            # Calculate volatility of volatility (vol of vol)
            volatility_values = [m.get("volatility", 0) for m in historical_metrics]
            vol_of_vol = np.std(volatility_values) if volatility_values else 0
            
            # Calculate risk metric trends
            risk_trends = {}
            for metric in ["drawdown", "var", "sharpe", "volatility"]:
                values = [m.get(metric, 0) for m in historical_metrics]
                if len(values) > 1:
                    # Simple trend calculation (positive = increasing, negative = decreasing)
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    risk_trends[f"{metric}_trend"] = trend
            
            # Create ML-friendly feature set
            ml_features = {
                "account_id": account_id,
                "current_drawdown": historical_metrics[-1].get("drawdown", 0) if historical_metrics else 0,
                "max_drawdown_30d": max([m.get("drawdown", 0) for m in historical_metrics]) if historical_metrics else 0,
                "volatility": historical_metrics[-1].get("volatility", 0) if historical_metrics else 0,
                "volatility_of_volatility": vol_of_vol,
                "sharpe_ratio": historical_metrics[-1].get("sharpe", 0) if historical_metrics else 0,
                "value_at_risk": historical_metrics[-1].get("var", 0) if historical_metrics else 0,
                **risk_trends
            }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "features": ml_features,
                "feature_version": "1.0.0",
                "timeframe": timeframe
            }

    def process_ml_risk_feedback(self, 
                               ml_predictions: Dict[str, Any], 
                               actual_outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process feedback from machine learning models to improve risk assessments.
        
        Args:
            ml_predictions: Predictions made by ML models
            actual_outcomes: Actual observed outcomes
            
        Returns:
            Dictionary with feedback analysis results
        """
        # Calculate prediction accuracy for various risk metrics
        accuracy_metrics = {}
        for metric in ml_predictions.keys():
            if metric in actual_outcomes:
                predicted = ml_predictions[metric]
                actual = actual_outcomes[metric]
                
                # Calculate error and store
                if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                    error = abs(predicted - actual)
                    error_pct = (error / abs(actual)) * 100 if actual != 0 else 0
                    
                    accuracy_metrics[f"{metric}_error"] = error
                    accuracy_metrics[f"{metric}_error_pct"] = error_pct
        
        # Analyze systematic biases in predictions
        bias_analysis = {}
        for metric in ml_predictions.keys():
            if metric in actual_outcomes:
                predicted = ml_predictions[metric]
                actual = actual_outcomes[metric]
                
                if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                    # Positive bias means overestimation
                    bias = predicted - actual
                    bias_analysis[f"{metric}_bias"] = bias
        
        # Generate recommendations for model improvement
        improvement_recommendations = []
        
        # Check for high error metrics
        high_error_metrics = [metric for metric, value in accuracy_metrics.items() 
                             if "error_pct" in metric and value > 20]
        
        if high_error_metrics:
            improvement_recommendations.append({
                "target": "model_accuracy",
                "description": f"High error in metrics: {', '.join(high_error_metrics)}",
                "suggestion": "Retrain model with more recent data or adjust feature weights"
            })
        
        # Check for systematic bias
        biased_metrics = [metric.replace("_bias", "") for metric, value in bias_analysis.items() 
                         if abs(value) > 0.2]
        
        if biased_metrics:
            improvement_recommendations.append({
                "target": "bias_correction",
                "description": f"Systematic bias detected in: {', '.join(biased_metrics)}",
                "suggestion": "Apply bias correction factor or recalibrate model"
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "accuracy_metrics": accuracy_metrics,
            "bias_analysis": bias_analysis,
            "improvement_recommendations": improvement_recommendations,
        }
        
    def monitor_risk_thresholds(self, 
                              account_id: str,
                              current_risk_metrics: Dict[str, Any],
                              thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor risk metrics against thresholds for proactive alerts.
        
        Args:
            account_id: Unique identifier for the account
            current_risk_metrics: Dictionary with current risk metrics
            thresholds: Dictionary with risk thresholds
            
        Returns:
            Dictionary with monitoring results and alerts
        """
        alerts = []
        warning_levels = {}
        
        # Check each risk metric against its threshold
        for metric, value in current_risk_metrics.items():
            if metric in thresholds:
                threshold = thresholds[metric]
                
                # Skip non-numeric values
                if not isinstance(value, (int, float)) or not isinstance(threshold, (int, float)):
                    continue
                
                # Calculate how close we are to the threshold (as a percentage)
                threshold_proximity = (value / threshold) * 100 if threshold != 0 else 0
                
                # Determine warning level
                warning_level = "NONE"
                if threshold_proximity >= 100:
                    warning_level = "CRITICAL"
                    alerts.append({
                        "level": "CRITICAL",
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "message": f"{metric} exceeds critical threshold of {threshold}"
                    })
                elif threshold_proximity >= 90:
                    warning_level = "HIGH"
                    alerts.append({
                        "level": "HIGH",
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "message": f"{metric} approaching critical threshold ({threshold_proximity:.1f}%)"
                    })
                elif threshold_proximity >= 75:
                    warning_level = "MEDIUM"
                    alerts.append({
                        "level": "MEDIUM",
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "message": f"{metric} requires attention ({threshold_proximity:.1f}%)"
                    })
                
                warning_levels[metric] = warning_level
        
        # Check if any automatic actions should be triggered
        auto_actions = []
        for alert in alerts:
            if alert["level"] == "CRITICAL":
                # For critical alerts, suggest automatic risk reduction
                if alert["metric"] in ["drawdown", "exposure", "var"]:
                    auto_actions.append({
                        "action_type": "REDUCE_EXPOSURE",
                        "target": "global",
                        "reduction_pct": 25,
                        "reason": f"Critical threshold exceeded for {alert['metric']}"
                    })
        
        return {
            "account_id": account_id,
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts,
            "warning_levels": warning_levels,
            "automatic_actions": auto_actions,
            "has_critical_alerts": any(alert["level"] == "CRITICAL" for alert in alerts)
        }
    
    def trigger_automated_risk_control(self,
                                     account_id: str,
                                     alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger automated risk control actions based on alerts.
        
        Args:
            account_id: Unique identifier for the account
            alert_data: Alert data from monitor_risk_thresholds
            
        Returns:
            Dictionary with action results
        """
        if not alert_data.get("has_critical_alerts", False):
            return {
                "account_id": account_id,
                "timestamp": datetime.now().isoformat(),
                "actions_taken": [],
                "status": "NO_ACTION_REQUIRED"
            }
        
        actions_taken = []
        
        # Process automatic actions
        for action in alert_data.get("automatic_actions", []):
            if action["action_type"] == "REDUCE_EXPOSURE":
                # Simulate reducing exposure
                reduction_pct = action.get("reduction_pct", 10)
                
                with DatabaseSession() as session:
                    risk_repo = RiskRepository(session)
                    limits_repo = LimitsRepository(session)
                    
                    # Get current exposure
                    current_exposure = risk_repo.get_current_exposure(account_id)
                    
                    # Calculate new exposure
                    new_exposure = current_exposure * (1 - (reduction_pct / 100))
                    
                    # Apply exposure reduction
                    # This would typically involve:
                    # 1. Updating risk limits
                    # 2. Sending commands to execution engine to reduce positions
                    # 3. Logging the action for audit
                    
                    # For now, just update the risk limits
                    exposure_limit = limits_repo.get_limit_by_type(
                        account_id, 
                        LimitType.MAX_TOTAL_EXPOSURE_PCT
                    )
                    
                    if exposure_limit:
                        new_limit_value = exposure_limit.limit_value * (1 - (reduction_pct / 100))
                        update_data = RiskLimitUpdate(limit_value=new_limit_value)
                        limits_repo.update_limit(exposure_limit.id, update_data)
                        
                        actions_taken.append({
                            "action_type": "REDUCE_EXPOSURE",
                            "old_value": exposure_limit.limit_value,
                            "new_value": new_limit_value,
                            "reduction_pct": reduction_pct,
                            "status": "SUCCESS",
                            "timestamp": datetime.now().isoformat()
                        })
        
        return {
            "account_id": account_id,
            "timestamp": datetime.now().isoformat(),
            "actions_taken": actions_taken,
            "status": "SUCCESS" if actions_taken else "NO_ACTION_TAKEN"
        }
