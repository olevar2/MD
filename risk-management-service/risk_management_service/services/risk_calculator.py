"""
Risk Calculator Module.

Provides functionality for calculating various risk metrics and limits.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger

logger = get_logger("risk-calculator")


class RiskCalculator:
    """
    Calculator for various risk metrics and limits.
    
    This class provides methods to calculate risk metrics like:
    - Maximum position size based on account risk percentage
    - Value at Risk (VaR) for portfolios
    - Correlation-based risk assessment
    - Drawdown management
    """
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        risk_per_trade_pct: float,
        stop_loss_pips: float,
        pip_value: float,
        leverage: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate position size based on account risk percentage.
        
        Args:
            account_balance: Current account balance
            risk_per_trade_pct: Maximum risk per trade as percentage of account
            stop_loss_pips: Distance to stop loss in pips
            pip_value: Value of one pip in account currency
            leverage: Account leverage (default: 1.0 = no leverage)
            
        Returns:
            Dictionary with calculated position sizes and risk values
        """
        # Calculate maximum amount to risk
        max_risk_amount = account_balance * (risk_per_trade_pct / 100)
        
        # Calculate position size based on stop loss
        if stop_loss_pips <= 0 or pip_value <= 0:
            logger.warning("Invalid stop loss pips or pip value")
            position_size = 0
        else:
            # Position size = risk amount / (stop loss in pips * pip value)
            position_size = max_risk_amount / (stop_loss_pips * pip_value)
        
        # Apply leverage
        position_size_with_leverage = position_size * leverage
        
        # Calculate required margin
        required_margin = (position_size_with_leverage * account_balance) / leverage
        
        return {
            "position_size": position_size_with_leverage,
            "risk_amount": max_risk_amount,
            "required_margin": required_margin,
            "risk_reward_ratio": None  # Would need take profit to calculate this
        }
    
    @staticmethod
    def calculate_value_at_risk(
        portfolio_value: float,
        daily_returns: List[float],
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) for a portfolio.
        
        Args:
            portfolio_value: Current portfolio value
            daily_returns: Historical daily returns as percentages
            confidence_level: Confidence level for VaR (default: 0.95 = 95%)
            time_horizon_days: Time horizon in days (default: 1)
            
        Returns:
            Dictionary with VaR metrics
        """
        # Convert returns to numpy array
        returns = np.array(daily_returns)
        
        # Calculate VaR using historical method
        var_percentile = 1.0 - confidence_level
        var_pct = np.percentile(returns, var_percentile * 100)
        
        # Scale to time horizon
        var_pct_scaled = var_pct * np.sqrt(time_horizon_days)
        
        # Convert to monetary value
        var_value = portfolio_value * abs(var_pct_scaled)
        
        # Calculate Conditional VaR (CVaR) / Expected Shortfall
        cvar_returns = returns[returns <= var_pct]
        cvar_pct = np.mean(cvar_returns) if len(cvar_returns) > 0 else var_pct * 1.25
        
        # Scale CVaR to time horizon
        cvar_pct_scaled = cvar_pct * np.sqrt(time_horizon_days)
        
        # Convert to monetary value
        cvar_value = portfolio_value * abs(cvar_pct_scaled)
        
        return {
            "var_pct": var_pct_scaled,
            "var_value": var_value,
            "cvar_pct": cvar_pct_scaled,
            "cvar_value": cvar_value,
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon_days
        }
    
    @staticmethod
    def calculate_drawdown_risk(
        current_balance: float,
        historical_balances: List[float],
        max_drawdown_limit_pct: float = 20.0
    ) -> Dict[str, Any]:
        """
        Calculate drawdown risk metrics.
        
        Args:
            current_balance: Current account balance
            historical_balances: List of historical account balances
            max_drawdown_limit_pct: Maximum allowed drawdown percentage
            
        Returns:
            Dictionary with drawdown risk metrics
        """
        if not historical_balances:
            return {
                "current_drawdown_pct": 0.0,
                "max_historical_drawdown_pct": 0.0,
                "drawdown_risk_level": "LOW",
                "available_risk_pct": max_drawdown_limit_pct
            }
        
        # Calculate current drawdown
        peak_balance = max(historical_balances + [current_balance])
        current_drawdown = peak_balance - current_balance
        current_drawdown_pct = (current_drawdown / peak_balance) * 100 if peak_balance > 0 else 0
        
        # Calculate historical maximum drawdown
        balances = historical_balances.copy()
        max_drawdown_pct = 0
        
        for i in range(len(balances)):
            peak_to_i = max(balances[:i+1])
            for j in range(i, len(balances)):
                dd_pct = ((peak_to_i - balances[j]) / peak_to_i) * 100 if peak_to_i > 0 else 0
                max_drawdown_pct = max(max_drawdown_pct, dd_pct)
        
        # Calculate risk level based on proximity to max drawdown limit
        risk_ratio = current_drawdown_pct / max_drawdown_limit_pct if max_drawdown_limit_pct > 0 else 0
        
        if risk_ratio < 0.5:
            risk_level = "LOW"
        elif risk_ratio < 0.8:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Calculate available risk percentage
        available_risk_pct = max(0, max_drawdown_limit_pct - current_drawdown_pct)
        
        return {
            "current_drawdown_pct": current_drawdown_pct,
            "max_historical_drawdown_pct": max_drawdown_pct,
            "drawdown_risk_level": risk_level,
            "available_risk_pct": available_risk_pct
        }
    
    @staticmethod
    def calculate_correlation_risk(
        symbols_returns: Dict[str, List[float]],
        positions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate correlation risk for a portfolio of positions.
        
        Args:
            symbols_returns: Dictionary mapping symbols to their historical returns
            positions: Dictionary mapping symbols to their position sizes
            
        Returns:
            Dictionary with correlation risk metrics
        """
        if not symbols_returns or not positions:
            return {
                "average_correlation": 0.0,
                "correlation_risk_level": "LOW",
                "highest_correlations": []
            }
        
        # Filter returns to only include symbols in the positions
        position_symbols = set(positions.keys())
        filtered_returns = {
            symbol: returns for symbol, returns in symbols_returns.items()
            if symbol in position_symbols
        }
        
        # Need at least two symbols to calculate correlation
        if len(filtered_returns) < 2:
            return {
                "average_correlation": 0.0,
                "correlation_risk_level": "LOW",
                "highest_correlations": []
            }
        
        # Calculate correlation matrix
        symbols = list(filtered_returns.keys())
        returns_array = np.array([filtered_returns[s] for s in symbols])
        
        # Handle case where some arrays may have different lengths
        min_length = min(len(ret) for ret in returns_array)
        returns_array = np.array([ret[:min_length] for ret in returns_array])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_array)
        
        # Extract unique correlation values (excluding self-correlations)
        correlation_pairs = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr_value = corr_matrix[i, j]
                if not np.isnan(corr_value):
                    correlation_pairs.append({
                        "symbol1": symbols[i],
                        "symbol2": symbols[j],
                        "correlation": corr_value,
                        "weight": abs(positions[symbols[i]] * positions[symbols[j]])
                    })
        
        # Calculate weighted average correlation
        if correlation_pairs:
            total_weight = sum(pair["weight"] for pair in correlation_pairs)
            if total_weight > 0:
                weighted_avg_corr = sum(pair["correlation"] * pair["weight"] for pair in correlation_pairs) / total_weight
            else:
                weighted_avg_corr = np.mean([pair["correlation"] for pair in correlation_pairs])
        else:
            weighted_avg_corr = 0.0
        
        # Determine risk level based on absolute correlation
        abs_avg_corr = abs(weighted_avg_corr)
        if abs_avg_corr < 0.3:
            risk_level = "LOW"
        elif abs_avg_corr < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Sort correlations by absolute value, highest first
        sorted_pairs = sorted(correlation_pairs, key=lambda x: abs(x["correlation"]), reverse=True)
        highest_correlations = sorted_pairs[:5]  # Top 5 highest correlations
        
        return {
            "average_correlation": weighted_avg_corr,
            "correlation_risk_level": risk_level,
            "highest_correlations": highest_correlations
        }
    
    @staticmethod
    def calculate_max_trades(
        account_balance: float,
        risk_per_trade_pct: float,
        portfolio_heat_limit_pct: float = 20.0
    ) -> Dict[str, int]:
        """
        Calculate maximum number of simultaneous trades based on risk limits.
        
        Args:
            account_balance: Current account balance
            risk_per_trade_pct: Risk percentage per trade
            portfolio_heat_limit_pct: Maximum total portfolio risk percentage
            
        Returns:
            Dictionary with maximum trades calculation
        """
        if risk_per_trade_pct <= 0:
            return {
                "max_simultaneous_trades": 0,
                "current_portfolio_heat": 0
            }
        
        max_trades = int(portfolio_heat_limit_pct / risk_per_trade_pct)
        
        return {
            "max_simultaneous_trades": max_trades,
            "current_portfolio_heat": 0  # This would be calculated based on actual open positions
        }
    
    @staticmethod
    def calculate_stress_test(
        portfolio_value: float,
        asset_allocations: Dict[str, float],
        stress_scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate stress test results for different market scenarios.
        
        Args:
            portfolio_value: Current portfolio value
            asset_allocations: Dictionary mapping assets to their allocation percentages
            stress_scenarios: Dictionary with stress scenarios and their impact on each asset
            
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        for scenario_name, scenario_impacts in stress_scenarios.items():
            # Calculate total impact on portfolio
            total_impact = 0.0
            
            for asset, allocation in asset_allocations.items():
                if asset in scenario_impacts:
                    # Impact = allocation * percentage change in scenario
                    impact = allocation * scenario_impacts[asset]
                    total_impact += impact
            
            # Calculate monetary impact
            monetary_impact = portfolio_value * total_impact / 100
            
            results[scenario_name] = {
                "percentage_impact": total_impact,
                "monetary_impact": monetary_impact,
                "remaining_value": portfolio_value + monetary_impact
            }
        
        return {
            "portfolio_value": portfolio_value,
            "scenarios": results
        }
