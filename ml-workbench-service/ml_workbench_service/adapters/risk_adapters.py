"""
Risk Management Adapters Module

This module provides adapter implementations for risk management interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
import asyncio

from common_lib.simulation.interfaces import IRiskManager
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManagerAdapter(IRiskManager):
    """
    Adapter for risk manager that implements the common interface.
    
    This adapter can either wrap an actual risk manager instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, risk_manager_instance=None):
        """
        Initialize the adapter.
        
        Args:
            risk_manager_instance: Optional actual risk manager instance to wrap
        """
        self.risk_manager = risk_manager_instance
        self._positions = {}
        self._risk_limits = {
            "max_position_size": 10000.0,
            "max_leverage": 20.0,
            "max_drawdown": 0.20,
            "risk_per_trade": 0.02
        }
        self.metrics_history = []
        self.peak_balance = 100000.0
    
    async def check_order(self, symbol: str, direction: str, size: float, 
                         current_price: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if an order meets risk criteria.
        
        Args:
            symbol: The trading symbol
            direction: Order direction (buy or sell)
            size: Order size
            current_price: Current market price
            
        Returns:
            Risk check result with approval status
        """
        if self.risk_manager:
            try:
                return await self.risk_manager.check_order(
                    symbol=symbol,
                    direction=direction,
                    size=size,
                    current_price=current_price
                )
            except Exception as e:
                logger.warning(f"Error checking order with risk manager: {str(e)}")
        
        # Simple risk check logic if no risk manager available
        approved = True
        reason = ""
        
        # Check position size limit
        if size > self._risk_limits["max_position_size"]:
            approved = False
            reason = f"Position size {size} exceeds maximum {self._risk_limits['max_position_size']}"
        
        # Check total exposure
        total_exposure = sum(pos["size"] for pos in self._positions.values())
        if total_exposure + size > self._risk_limits["max_position_size"] * 3:
            approved = False
            reason = f"Total exposure {total_exposure + size} exceeds maximum"
        
        return {
            "approved": approved,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_position(self, symbol: str, size: float, price: float, 
                    direction: str, leverage: float = 1.0) -> None:
        """
        Add a new position for risk tracking.
        
        Args:
            symbol: The trading symbol
            size: Position size
            price: Entry price
            direction: Position direction (long or short)
            leverage: Position leverage
        """
        if self.risk_manager:
            try:
                self.risk_manager.add_position(
                    symbol=symbol,
                    size=size,
                    price=price,
                    direction=direction,
                    leverage=leverage
                )
                return
            except Exception as e:
                logger.warning(f"Error adding position to risk manager: {str(e)}")
        
        # Add position to local tracking if no risk manager available
        position_id = f"pos_{len(self._positions) + 1}"
        self._positions[position_id] = {
            "symbol": symbol,
            "size": size,
            "price": price,
            "direction": direction,
            "leverage": leverage,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Added position {position_id}: {symbol} {direction} {size}")
    
    def check_risk_limits(self) -> List[Dict[str, Any]]:
        """
        Check if any risk limits are breached.
        
        Returns:
            List of breached risk limits
        """
        if self.risk_manager:
            try:
                return self.risk_manager.check_risk_limits()
            except Exception as e:
                logger.warning(f"Error checking risk limits with risk manager: {str(e)}")
        
        # Simple risk limit check if no risk manager available
        breached_limits = []
        
        # Check total exposure
        total_exposure = sum(pos["size"] for pos in self._positions.values())
        if total_exposure > self._risk_limits["max_position_size"] * 3:
            breached_limits.append({
                "limit_type": "total_exposure",
                "current_value": total_exposure,
                "limit_value": self._risk_limits["max_position_size"] * 3,
                "severity": "high"
            })
        
        # Check drawdown if metrics history exists
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            current_drawdown = latest_metrics.get("drawdown", 0.0)
            
            if current_drawdown > self._risk_limits["max_drawdown"]:
                breached_limits.append({
                    "limit_type": "drawdown",
                    "current_value": current_drawdown,
                    "limit_value": self._risk_limits["max_drawdown"],
                    "severity": "high"
                })
        
        return breached_limits
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Get current portfolio risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        if self.risk_manager:
            try:
                return self.risk_manager.get_portfolio_metrics()
            except Exception as e:
                logger.warning(f"Error getting portfolio metrics from risk manager: {str(e)}")
        
        # Generate simple metrics if no risk manager available
        total_exposure = sum(pos["size"] for pos in self._positions.values())
        
        # Create default metrics
        metrics = {
            "total_exposure": total_exposure,
            "position_count": len(self._positions),
            "max_position_size": max([pos["size"] for pos in self._positions.values()]) if self._positions else 0,
            "drawdown": 0.0,
            "risk_per_trade": self._risk_limits["risk_per_trade"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.metrics_history.append(metrics)
        
        return metrics
