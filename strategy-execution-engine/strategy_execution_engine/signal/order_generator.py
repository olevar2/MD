"""
Order Generator Module for Forex Trading Platform

This module transforms trading decisions into executable orders
with proper formatting for the trading gateway service.
"""
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class OrderGenerator:
    """
    Generates executable orders from trading decisions
    
    This class takes the high-level order decisions from the DecisionLogicEngine
    and transforms them into properly formatted orders for the trading gateway.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the order generator
        
        Args:
            config: Configuration dictionary for order generation
        """
        self.config = config or {}
        self.default_order_type = self.config.get("default_order_type", "MARKET")
        self.default_time_in_force = self.config.get("default_time_in_force", "GTC")
        self.enable_oco_orders = self.config.get("enable_oco_orders", True)
        
    def generate_orders(
        self,
        order_decisions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate executable orders from order decisions
        
        Args:
            order_decisions: List of order decisions from the decision engine
            
        Returns:
            List of executable orders
        """
        if not order_decisions:
            return []
            
        executable_orders = []
        
        for decision in order_decisions:
            # Basic validation
            if not self._validate_decision(decision):
                logger.warning(f"Invalid order decision: {decision}")
                continue
                
            # Create a new order ID
            order_id = str(uuid.uuid4())
            
            # Determine order type based on configuration
            order_type = self._determine_order_type(decision)
            
            # Format the basic order
            order = {
                "order_id": order_id,
                "timestamp": datetime.now().isoformat(),
                "symbol": decision["symbol"],
                "direction": decision["trade_direction"],
                "quantity": decision["position_size"],
                "order_type": order_type,
                "price": decision.get("entry_price"),
                "time_in_force": self.default_time_in_force,
                "strategy_id": decision.get("strategy_id", "unknown"),
                "metadata": {
                    "signal_confidence": decision.get("signal_confidence", 1.0),
                    "market_regime": decision.get("market_regime", "unknown"),
                    "order_decision_id": decision.get("id", str(uuid.uuid4()))
                }
            }
            
            # Add stop loss if available
            if decision.get("stop_loss") is not None:
                if self.enable_oco_orders:
                    # Use OCO (One-Cancels-Other) order if enabled and take profit is also set
                    if decision.get("take_profit") is not None:
                        order["stop_loss"] = decision["stop_loss"]
                        order["take_profit"] = decision["take_profit"]
                        order["is_oco"] = True
                    else:
                        # Only stop loss
                        order["stop_loss"] = decision["stop_loss"]
                else:
                    # Separate stop loss order
                    sl_order = self._create_stop_loss_order(order_id, decision)
                    executable_orders.append(sl_order)
            
            # Add take profit as separate order if OCO not used
            if decision.get("take_profit") is not None and not order.get("is_oco"):
                tp_order = self._create_take_profit_order(order_id, decision)
                executable_orders.append(tp_order)
            
            # Add main order
            executable_orders.append(order)
            
            logger.info(f"Generated order: {order_type} {decision['trade_direction']} {decision['position_size']} {decision['symbol']}")
            
        return executable_orders
    
    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """
        Validate an order decision
        
        Args:
            decision: Order decision to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["symbol", "trade_direction", "position_size"]
        
        # Check required fields
        for field in required_fields:
            if field not in decision:
                logger.warning(f"Missing required field in order decision: {field}")
                return False
        
        # Validate direction
        if decision["trade_direction"] not in ["buy", "sell"]:
            logger.warning(f"Invalid trade direction: {decision['trade_direction']}")
            return False
            
        # Validate position size
        if decision["position_size"] <= 0:
            logger.warning(f"Invalid position size: {decision['position_size']}")
            return False
            
        return True
    
    def _determine_order_type(self, decision: Dict[str, Any]) -> str:
        """
        Determine the appropriate order type based on the decision
        
        Args:
            decision: Order decision
            
        Returns:
            Order type string
        """
        # Default to market order
        if "limit_price" in decision:
            return "LIMIT"
        elif "entry_price" in decision and decision.get("is_stop_entry", False):
            return "STOP"
        else:
            return self.default_order_type
    
    def _create_stop_loss_order(self, parent_id: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a stop loss order for the given parent order
        
        Args:
            parent_id: ID of the parent order
            decision: Original order decision
            
        Returns:
            Stop loss order dictionary
        """
        sl_order = {
            "order_id": str(uuid.uuid4()),
            "parent_order_id": parent_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": decision["symbol"],
            "direction": "sell" if decision["trade_direction"] == "buy" else "buy",  # Opposite
            "quantity": decision["position_size"],
            "order_type": "STOP",
            "price": decision["stop_loss"],
            "time_in_force": "GTC",
            "is_stop_loss": True,
            "strategy_id": decision.get("strategy_id", "unknown"),
            "metadata": {
                "original_decision_id": decision.get("id", "unknown")
            }
        }
        return sl_order
    
    def _create_take_profit_order(self, parent_id: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a take profit order for the given parent order
        
        Args:
            parent_id: ID of the parent order
            decision: Original order decision
            
        Returns:
            Take profit order dictionary
        """
        tp_order = {
            "order_id": str(uuid.uuid4()),
            "parent_order_id": parent_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": decision["symbol"],
            "direction": "sell" if decision["trade_direction"] == "buy" else "buy",  # Opposite
            "quantity": decision["position_size"],
            "order_type": "LIMIT",
            "price": decision["take_profit"],
            "time_in_force": "GTC",
            "is_take_profit": True,
            "strategy_id": decision.get("strategy_id", "unknown"),
            "metadata": {
                "original_decision_id": decision.get("id", "unknown")
            }
        }
        return tp_order
