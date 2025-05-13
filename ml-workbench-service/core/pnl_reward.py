"""
PnL-based Reward Components

This module provides reward components based on profit and loss (PnL).
"""

import numpy as np
from typing import List, Dict, Any

from .base_reward import RewardComponent, CompositeReward


class PnLReward(CompositeReward):
    """
    A reward function based on profit and loss (PnL).
    
    This reward function focuses on maximizing trading profits,
    with optional components for transaction costs.
    """
    
    def __init__(self, include_transaction_costs: bool = True):
        """
        Initialize the PnL reward function.
        
        Args:
            include_transaction_costs: Whether to include transaction costs
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
        
        # Add transaction cost component if enabled
        if include_transaction_costs:
            self.add_component(
                RewardComponent(
                    name="transaction_cost",
                    weight=-1.0,
                    function=lambda env: env.last_transaction_cost if hasattr(env, 'last_transaction_cost') else 0.0,
                    description="Transaction cost for the current step"
                )
            )


def create_pnl_reward(trading_fee_weight: float = 1.0) -> CompositeReward:
    """
    Create a PnL-based reward function.
    
    Args:
        trading_fee_weight: Weight for the trading fee component
        
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
    
    # Add trading fee component
    reward.add_component(
        RewardComponent(
            name="trading_fee",
            weight=-trading_fee_weight,
            function=lambda env: env.last_transaction_cost if hasattr(env, 'last_transaction_cost') else 0.0,
            description="Trading fee for the current step"
        )
    )
    
    return reward