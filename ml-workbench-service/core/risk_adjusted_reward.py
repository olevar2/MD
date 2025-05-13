"""
Risk-Adjusted Reward Components

This module provides reward components that incorporate risk metrics.
"""

import numpy as np
from typing import List, Dict, Any

from .base_reward import RewardComponent, CompositeReward


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio from a list of returns.
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to per-step rate
    # Assuming 252 trading days per year
    per_step_rfr = risk_free_rate / 252
    
    excess_returns = [r - per_step_rfr for r in returns]
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    if std_excess_return == 0:
        return 0.0
    
    # Annualize by multiplying by sqrt(252)
    return mean_excess_return / std_excess_return * np.sqrt(252)


def calculate_max_drawdown(returns: List[float]) -> float:
    """
    Calculate the maximum drawdown from a list of returns.
    
    Args:
        returns: List of returns
        
    Returns:
        Maximum drawdown (as a positive value)
    """
    if not returns:
        return 0.0
    
    # Convert returns to cumulative returns
    cumulative = np.cumsum(returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdowns
    drawdowns = running_max - cumulative
    
    # Return maximum drawdown
    return np.max(drawdowns) if len(drawdowns) > 0 else 0.0


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino ratio from a list of returns.
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to per-step rate
    per_step_rfr = risk_free_rate / 252
    
    excess_returns = [r - per_step_rfr for r in returns]
    mean_excess_return = np.mean(excess_returns)
    
    # Calculate downside deviation (standard deviation of negative returns only)
    negative_returns = [r for r in excess_returns if r < 0]
    if not negative_returns:
        return 0.0  # No negative returns
    
    downside_deviation = np.std(negative_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    # Annualize by multiplying by sqrt(252)
    return mean_excess_return / downside_deviation * np.sqrt(252)


class RiskAdjustedReward(CompositeReward):
    """
    A reward function that incorporates risk metrics.
    
    This reward function balances profit maximization with risk management,
    using metrics like Sharpe ratio, maximum drawdown, and volatility.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, curriculum_level: int = 0):
        """
        Initialize the risk-adjusted reward function.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            curriculum_level: Difficulty level (higher means more risk components)
        """
        super().__init__()
        
        # Add basic PnL component
        self.add_component(
            RewardComponent(
                name="pnl",
                weight=1.0,
                function=lambda env: env.current_pnl,
                description="Profit and loss for the current step"
            )
        )
        
        # Add risk-adjusted components
        self.add_component(
            RewardComponent(
                name="drawdown_penalty",
                weight=-0.5,
                function=lambda env: max(0, calculate_max_drawdown(env.episode_returns)),
                description="Penalty for maximum drawdown"
            )
        )
        
        self.add_component(
            RewardComponent(
                name="sharpe_bonus",
                weight=0.3,
                function=lambda env: calculate_sharpe_ratio(env.episode_returns, risk_free_rate)
                                   if len(env.episode_returns) > 1 else 0.0,
                description="Bonus for high Sharpe ratio"
            )
        )
        
        self.add_component(
            RewardComponent(
                name="trade_frequency_penalty",
                weight=-0.1,
                function=lambda env: 1.0 if env.total_trades > 0 and env.current_step > 0
                                   and (env.total_trades / env.current_step) > 0.4 else 0.0,
                description="Penalty for excessive trading"
            )
        )
        
        # Add additional risk components for higher curriculum levels
        if curriculum_level > 1:
            self.add_component(
                RewardComponent(
                    name="volatility_penalty",
                    weight=-0.2,
                    function=lambda env: np.std(env.episode_returns) if len(env.episode_returns) > 1 else 0.0,
                    description="Penalty for high return volatility"
                )
            )
            
            self.add_component(
                RewardComponent(
                    name="sortino_bonus",
                    weight=0.2,
                    function=lambda env: calculate_sortino_ratio(env.episode_returns, risk_free_rate)
                                       if len(env.episode_returns) > 1 else 0.0,
                    description="Bonus for high Sortino ratio"
                )
            )


def create_risk_adjusted_reward(risk_free_rate: float = 0.02, 
                              curriculum_level: int = 0) -> CompositeReward:
    """
    Create a risk-adjusted reward function.
    
    Args:
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        curriculum_level: Difficulty level (higher means more risk components)
        
    Returns:
        Composite reward function
    """
    reward = CompositeReward()
    
    # Add basic PnL component
    reward.add_component(
        RewardComponent(
            name="pnl",
            weight=1.0,
            function=lambda env: env.current_pnl,
            description="Profit and loss for the current step"
        )
    )
    
    # Add risk-adjusted components
    reward.add_component(
        RewardComponent(
            name="drawdown_penalty",
            weight=-0.5,
            function=lambda env: max(0, calculate_max_drawdown(env.episode_returns)),
            description="Penalty for maximum drawdown"
        )
    )
    
    reward.add_component(
        RewardComponent(
            name="sharpe_bonus",
            weight=0.3,
            function=lambda env: calculate_sharpe_ratio(env.episode_returns, risk_free_rate)
                               if len(env.episode_returns) > 1 else 0.0,
            description="Bonus for high Sharpe ratio"
        )
    )
    
    reward.add_component(
        RewardComponent(
            name="trade_frequency_penalty",
            weight=-0.1,
            function=lambda env: 1.0 if env.total_trades > 0 and env.current_step > 0
                               and (env.total_trades / env.current_step) > 0.4 else 0.0,
            description="Penalty for excessive trading"
        )
    )
    
    # Add additional risk components for higher curriculum levels
    if curriculum_level > 1:
        reward.add_component(
            RewardComponent(
                name="volatility_penalty",
                weight=-0.2,
                function=lambda env: np.std(env.episode_returns) if len(env.episode_returns) > 1 else 0.0,
                description="Penalty for high return volatility"
            )
        )
        
        reward.add_component(
            RewardComponent(
                name="sortino_bonus",
                weight=0.2,
                function=lambda env: calculate_sortino_ratio(env.episode_returns, risk_free_rate)
                                   if len(env.episode_returns) > 1 else 0.0,
                description="Bonus for high Sortino ratio"
            )
        )
    
    return reward