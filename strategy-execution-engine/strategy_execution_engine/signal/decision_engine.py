"""
Decision Logic Engine for Forex Trading Platform

This module processes aggregated signals, applies risk management,
and generates order requests based on trading rules and current market conditions.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from ..strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class DecisionLogicEngine:
    """
    Decision Logic Engine for generating order decisions
    
    This engine processes signals from the SignalAggregator, applies
    strategy-specific logic, incorporates risk parameters, and
    generates final order requests.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the decision logic engine
        
        Args:
            config: Configuration dictionary for decision making
        """
        self.config = config or {}
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.risk_management_client = None
        self.circuit_breaker_status = {
            "global": {"active": False, "reason": None},
            "pairs": {}
        }
        
        # Load configuration
        self.max_active_trades = self.config.get("max_active_trades", 5)
        self.max_trades_per_pair = self.config.get("max_trades_per_pair", 2)
        self.default_risk_per_trade = self.config.get("risk_per_trade", 1.0)  # % of account
        self.min_signal_confidence = self.config.get("min_signal_confidence", 0.6)
        self.enable_news_avoidance = self.config.get("enable_news_avoidance", True)
    
    def set_risk_management_client(self, risk_client) -> None:
        """
        Set the risk management service client
        
        Args:
            risk_client: Client for the risk management service
        """
        self.risk_management_client = risk_client
        logger.info("Risk management client configured")
        
    def register_strategy(self, strategy_id: str, strategy: BaseStrategy) -> None:
        """
        Register an active strategy
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy: Strategy instance implementing BaseStrategy
        """
        self.active_strategies[strategy_id] = strategy
        logger.info(f"Registered active strategy: {strategy_id}")
    
    def unregister_strategy(self, strategy_id: str) -> None:
        """
        Unregister an active strategy
        
        Args:
            strategy_id: Unique identifier for the strategy to remove
        """
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            logger.info(f"Unregistered strategy: {strategy_id}")
    
    def update_circuit_breaker_status(self, status: Dict[str, Any]) -> None:
        """
        Update the circuit breaker status
        
        Args:
            status: Circuit breaker status information
        """
        self.circuit_breaker_status = status
        
        if status.get("global", {}).get("active", False):
            logger.warning(f"Global circuit breaker active: {status['global'].get('reason', 'Unknown')}")
            
    def process_signals(
        self,
        strategy_id: str,
        data: pd.DataFrame,
        account_info: Dict[str, Any],
        active_positions: List[Dict[str, Any]],
        market_regime: str = None,
        news_events: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process signals for a specific strategy and generate order decisions
        
        Args:
            strategy_id: ID of the strategy to process
            data: Market data with aggregated signals
            account_info: Account information including balance
            active_positions: Currently active positions
            market_regime: Current market regime identifier
            news_events: Recent and upcoming news events
            
        Returns:
            List of order decision objects
        """
        # Check if the strategy is registered
        if strategy_id not in self.active_strategies:
            logger.error(f"Strategy {strategy_id} not found in active strategies")
            return []
        
        strategy = self.active_strategies[strategy_id]
        
        # Check circuit breaker status
        if self.circuit_breaker_status.get("global", {}).get("active", False):
            logger.warning(f"Global circuit breaker active - no new orders for {strategy_id}")
            return []
        
        # Get the symbol from the data
        if 'symbol' in data.columns:
            symbol = data['symbol'].iloc[0]
        else:
            symbol = self.config.get("default_symbol", "UNKNOWN")
            
        # Check symbol-specific circuit breaker
        if symbol in self.circuit_breaker_status.get("pairs", {}) and self.circuit_breaker_status["pairs"][symbol].get("active", False):
            logger.warning(f"Circuit breaker active for {symbol} - no new orders for {strategy_id}")
            return []
            
        # Check if we've exceeded maximum trades
        current_trades_for_pair = sum(1 for pos in active_positions if pos.get('symbol') == symbol)
        if current_trades_for_pair >= self.max_trades_per_pair:
            logger.info(f"Maximum trades ({self.max_trades_per_pair}) reached for {symbol}")
            return []
            
        total_active_trades = len(active_positions)
        if total_active_trades >= self.max_active_trades:
            logger.info(f"Maximum total trades ({self.max_active_trades}) reached")
            return []
        
        # Apply strategy to generate signals if not already done
        if 'signal' not in data.columns:
            data = strategy.generate_signals(data)
        
        # Extract only the latest signal (assuming we want to trade on the latest bar)
        latest_data = data.iloc[-1]
        
        # If no signal from strategy, use aggregated signal if available
        signal = latest_data.get('signal', 0)
        if signal == 0 and 'final_signal' in latest_data:
            signal = latest_data['final_signal']
            
        # If still no signal, skip processing
        if signal == 0:
            return []
            
        # Check confidence threshold if available
        confidence = latest_data.get('signal_confidence', 1.0)
        if confidence < self.min_signal_confidence:
            logger.info(f"Signal confidence {confidence} below threshold {self.min_signal_confidence}")
            return []
            
        # Avoid trading around high-impact news if enabled
        if self.enable_news_avoidance and news_events:
            if self._check_news_proximity(data.index[-1], news_events):
                logger.info(f"Avoiding trade due to proximity to high-impact news")
                return []
        
        # Calculate position size
        account_balance = account_info.get("balance", 0)
        risk_per_trade = self.config.get("risk_per_trade", self.default_risk_per_trade)
        
        # Apply risk adjustment based on market regime if available
        if market_regime:
            regime_risk_adj = self.config.get("regime_risk_adjustments", {}).get(market_regime, 1.0)
            risk_per_trade *= regime_risk_adj
            logger.debug(f"Adjusted risk for {market_regime} regime: {risk_per_trade}%")
        
        # Calculate position size using strategy's method
        position_size = strategy.calculate_position_size(
            data.iloc[-2:],  # Include previous bar for context
            account_balance,
            risk_per_trade
        ).iloc[-1]
        
        if position_size <= 0:
            logger.warning(f"Invalid position size {position_size}")
            return []
            
        # Create order decision
        order_decision = {
            "timestamp": data.index[-1],
            "symbol": symbol,
            "strategy_id": strategy_id,
            "trade_direction": "buy" if signal > 0 else "sell",
            "position_size": position_size,
            "entry_price": latest_data['close'],
            "stop_loss": latest_data.get('sl_level', None),
            "take_profit": latest_data.get('tp_level', None),
            "signal_confidence": confidence,
            "market_regime": market_regime,
        }
        
        # If risk management client is available, validate the trade
        if self.risk_management_client:
            is_valid, risk_check_result = self._validate_with_risk_management(order_decision, account_info)
            if not is_valid:
                logger.warning(f"Risk check failed: {risk_check_result.get('reason', 'Unknown')}")
                return []
                
            # Update order with potentially adjusted parameters
            if "adjusted_position_size" in risk_check_result:
                order_decision["position_size"] = risk_check_result["adjusted_position_size"]
                
            if "adjusted_stop_loss" in risk_check_result:
                order_decision["stop_loss"] = risk_check_result["adjusted_stop_loss"]
        
        logger.info(f"Generated order decision for {symbol}: {order_decision['trade_direction']} {order_decision['position_size']}")
        return [order_decision]
    
    def _validate_with_risk_management(
        self,
        order_decision: Dict[str, Any],
        account_info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate an order decision with the risk management service
        
        Args:
            order_decision: Proposed order decision
            account_info: Account information
            
        Returns:
            Tuple of (is_valid, risk_check_result)
        """
        try:
            result = self.risk_management_client.check_risk(
                symbol=order_decision["symbol"],
                direction=order_decision["trade_direction"],
                size=order_decision["position_size"],
                entry_price=order_decision["entry_price"],
                stop_loss=order_decision["stop_loss"],
                account_balance=account_info["balance"]
            )
            return result.get("is_valid", False), result
            
        except Exception as e:
            logger.error(f"Error calling risk management service: {e}")
            return False, {"is_valid": False, "reason": "Risk service error"}
            
    def _check_news_proximity(
        self,
        current_time: pd.Timestamp,
        news_events: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if current time is close to high-impact news events
        
        Args:
            current_time: Current timestamp
            news_events: List of news events
            
        Returns:
            True if close to high-impact news, False otherwise
        """
        if not news_events:
            return False
            
        # Define window (default: 15 minutes before, 30 minutes after)
        before_window = self.config.get('news_window_before', 15)  # minutes
        after_window = self.config.get('news_window_after', 30)  # minutes
        
        for event in news_events:
            if event.get('impact', '').lower() == 'high':
                event_time = pd.Timestamp(event['timestamp'])
                
                time_diff = (event_time - current_time).total_seconds() / 60
                
                # Check if within window
                if -before_window <= time_diff <= after_window:
                    return True
                    
        return False
