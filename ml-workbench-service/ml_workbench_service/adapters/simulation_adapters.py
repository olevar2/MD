"""
Simulation Adapters Module

This module provides adapter implementations for simulation interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from common_lib.simulation.interfaces import (
    IMarketRegimeSimulator,
    IBrokerSimulator,
    MarketRegimeType
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegimeSimulatorAdapter(IMarketRegimeSimulator):
    """
    Adapter for market regime simulator that implements the common interface.
    
    This adapter can either wrap an actual simulator instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, simulator_instance=None):
        """
        Initialize the adapter.
        
        Args:
            simulator_instance: Optional actual simulator instance to wrap
        """
        self.simulator = simulator_instance
        self.default_regime = MarketRegimeType.RANGING_NARROW
        self._regime_cache = {}
        
    def get_current_regime(self, symbol: str) -> MarketRegimeType:
        """
        Get the current market regime for a symbol.
        
        Args:
            symbol: The trading symbol to check
            
        Returns:
            The current market regime type
        """
        if self.simulator:
            try:
                # Try to use the wrapped simulator if available
                return self.simulator.get_current_regime(symbol)
            except Exception as e:
                logger.warning(f"Error getting regime from simulator: {str(e)}")
        
        # Fallback to cached or default regime
        return self._regime_cache.get(symbol, self.default_regime)
    
    def get_all_regimes(self) -> Dict[str, MarketRegimeType]:
        """
        Get the current market regimes for all symbols.
        
        Returns:
            Dictionary mapping symbols to their current market regime
        """
        if self.simulator:
            try:
                return self.simulator.get_all_regimes()
            except Exception as e:
                logger.warning(f"Error getting all regimes from simulator: {str(e)}")
        
        return self._regime_cache
    
    def get_regime_probabilities(self, symbol: str) -> Dict[MarketRegimeType, float]:
        """
        Get the probability distribution across different regime types.
        
        Args:
            symbol: The trading symbol to check
            
        Returns:
            Dictionary mapping regime types to their probabilities
        """
        if self.simulator:
            try:
                return self.simulator.get_regime_probabilities(symbol)
            except Exception as e:
                logger.warning(f"Error getting regime probabilities from simulator: {str(e)}")
        
        # Return default probabilities with highest for current regime
        current_regime = self.get_current_regime(symbol)
        probabilities = {regime: 0.05 for regime in MarketRegimeType}
        probabilities[current_regime] = 0.65
        
        return probabilities
    
    def get_regime_history(self, symbol: str, lookback_periods: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of regime changes.
        
        Args:
            symbol: The trading symbol to check
            lookback_periods: Number of historical periods to return
            
        Returns:
            List of historical regime data
        """
        if self.simulator:
            try:
                return self.simulator.get_regime_history(symbol, lookback_periods)
            except Exception as e:
                logger.warning(f"Error getting regime history from simulator: {str(e)}")
        
        # Return empty history if no simulator available
        return []
    
    def set_regime(self, symbol: str, regime: MarketRegimeType) -> None:
        """
        Set the current regime for a symbol (for testing/simulation).
        
        Args:
            symbol: The trading symbol
            regime: The market regime to set
        """
        self._regime_cache[symbol] = regime
        logger.info(f"Set {symbol} regime to {regime}")


class BrokerSimulatorAdapter(IBrokerSimulator):
    """
    Adapter for broker simulator that implements the common interface.
    
    This adapter can either wrap an actual simulator instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, simulator_instance=None):
        """
        Initialize the adapter.
        
        Args:
            simulator_instance: Optional actual simulator instance to wrap
        """
        self.simulator = simulator_instance
        self._price_cache = {}
        self._positions = {}
        self._account = {
            "balance": 100000.0,
            "equity": 100000.0,
            "margin_used": 0.0,
            "margin_level": 100.0,
            "free_margin": 100000.0
        }
    
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Dictionary with bid and ask prices
        """
        if self.simulator:
            try:
                return self.simulator.get_current_price(symbol)
            except Exception as e:
                logger.warning(f"Error getting price from simulator: {str(e)}")
        
        # Return cached or default price
        if symbol not in self._price_cache:
            # Generate default prices for common forex pairs
            if symbol == "EUR/USD":
                self._price_cache[symbol] = {"bid": 1.1000, "ask": 1.1002}
            elif symbol == "GBP/USD":
                self._price_cache[symbol] = {"bid": 1.3000, "ask": 1.3003}
            elif symbol == "USD/JPY":
                self._price_cache[symbol] = {"bid": 110.00, "ask": 110.03}
            else:
                self._price_cache[symbol] = {"bid": 1.0000, "ask": 1.0001}
        
        return self._price_cache[symbol]
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get the current account summary.
        
        Returns:
            Dictionary with account information
        """
        if self.simulator:
            try:
                return self.simulator.get_account_summary()
            except Exception as e:
                logger.warning(f"Error getting account summary from simulator: {str(e)}")
        
        return self._account
    
    async def submit_order(self, symbol: str, order_type: str, direction: str, 
                          size: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Submit a new order.
        
        Args:
            symbol: The trading symbol
            order_type: Type of order (market, limit, etc.)
            direction: Order direction (buy or sell)
            size: Order size
            price: Order price (for limit orders)
            
        Returns:
            Order result information
        """
        if self.simulator:
            try:
                return await self.simulator.submit_order(
                    symbol=symbol,
                    order_type=order_type,
                    direction=direction,
                    size=size,
                    price=price
                )
            except Exception as e:
                logger.warning(f"Error submitting order to simulator: {str(e)}")
        
        # Simple simulation logic if no simulator available
        order_id = f"order_{len(self._positions) + 1}"
        current_price = self.get_current_price(symbol)
        execution_price = current_price["ask"] if direction == "buy" else current_price["bid"]
        
        # Add position
        position_id = f"pos_{len(self._positions) + 1}"
        self._positions[position_id] = {
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "entry_price": execution_price,
            "current_price": execution_price,
            "pnl": 0.0,
            "open_time": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "order_id": order_id,
            "position_id": position_id,
            "execution_price": execution_price,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Get all current positions.
        
        Returns:
            Dictionary of current positions
        """
        if self.simulator:
            try:
                return self.simulator.get_positions()
            except Exception as e:
                logger.warning(f"Error getting positions from simulator: {str(e)}")
        
        return self._positions
