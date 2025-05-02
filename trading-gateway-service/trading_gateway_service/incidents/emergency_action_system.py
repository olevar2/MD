"""
Emergency Action System for predefined responses to trading incidents.

This component provides a framework for defining and executing emergency actions
in response to trading incidents, with safety controls and logging.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
import json

from core_foundations.utils.logger import get_logger


class EmergencyActionSystem:
    """
    System for executing predefined emergency actions in response to trading incidents.
    
    This component manages a registry of emergency actions that can be executed
    automatically or manually in response to trading incidents, providing a
    structured approach to emergency response.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EmergencyActionSystem.
        
        Args:
            config: Configuration parameters
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
        
        # Registry of available actions
        self._actions: Dict[str, Dict[str, Any]] = {}
        
        # Action execution history
        self._execution_history: List[Dict[str, Any]] = []
        
        # Maximum history size
        self._max_history_size = self.config.get("max_history_size", 1000)
        
        # Safety controls
        self.require_confirmation = self.config.get("require_confirmation", True)
        self.safety_cooldown = self.config.get("safety_cooldown", 60)  # seconds
        self._last_execution_times: Dict[str, float] = {}
        
        # External services integration
        self.trading_gateway_client = self.config.get("trading_gateway_client")
        self.position_manager = self.config.get("position_manager")
        self.circuit_breaker = self.config.get("circuit_breaker")
        self.notification_service = self.config.get("notification_service")
        
        # Register default actions
        self._register_default_actions()
        
        self.logger.info("EmergencyActionSystem initialized")

    def register_action(
        self,
        action_id: str,
        name: str,
        description: str,
        execution_function: Callable,
        safety_level: str = "high",
        parameters: Optional[Dict[str, Any]] = None,
        confirmation_required: Optional[bool] = None,
    ) -> bool:
        """
        Register a new emergency action.
        
        Args:
            action_id: Unique identifier for the action
            name: Human-readable name of the action
            description: Description of what the action does
            execution_function: Function that implements the action
            safety_level: Safety level of the action (low, medium, high, critical)
            parameters: Parameter schema for the action
            confirmation_required: Whether confirmation is required
            
        Returns:
            True if successfully registered, False if action_id already exists
        """
        if action_id in self._actions:
            self.logger.warning(f"Action ID {action_id} already exists")
            return False
        
        # Default confirmation based on safety level if not specified
        if confirmation_required is None:
            confirmation_required = safety_level in ["high", "critical"]
        
        # Create action definition
        action = {
            "id": action_id,
            "name": name,
            "description": description,
            "execution_function": execution_function,
            "safety_level": safety_level,
            "parameters": parameters or {},
            "confirmation_required": confirmation_required,
            "registered_at": datetime.now().isoformat(),
        }
        
        self._actions[action_id] = action
        self.logger.info(f"Registered emergency action: {name} (ID: {action_id})")
        return True

    async def execute_action(
        self,
        action_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        incident_id: Optional[str] = None,
        bypass_confirmation: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute an emergency action.
        
        Args:
            action_id: ID of the action to execute
            parameters: Parameters for the action
            incident_id: Optional ID of the incident that triggered the action
            bypass_confirmation: Whether to bypass confirmation requirements
            
        Returns:
            Result of the action execution
        """
        # Check if action exists
        if action_id not in self._actions:
            message = f"Action {action_id} not found"
            self.logger.warning(message)
            return {"success": False, "message": message}
        
        action = self._actions[action_id]
        parameters = parameters or {}
        
        # Check cooldown period
        if action_id in self._last_execution_times:
            elapsed = datetime.now().timestamp() - self._last_execution_times[action_id]
            if elapsed < self.safety_cooldown:
                message = f"Safety cooldown for action {action_id} not elapsed. Try again in {self.safety_cooldown - elapsed:.1f} seconds"
                self.logger.warning(message)
                return {"success": False, "message": message}
        
        # Check if confirmation is required
        if self.require_confirmation and action["confirmation_required"] and not bypass_confirmation:
            message = f"Confirmation required for action {action_id}"
            self.logger.info(message)
            return {"success": False, "message": message, "requires_confirmation": True}
        
        # Execute the action
        execution_id = str(uuid.uuid4())
        execution_start = datetime.now()
        
        # Create execution record
        execution_record = {
            "execution_id": execution_id,
            "action_id": action_id,
            "action_name": action["name"],
            "parameters": parameters,
            "incident_id": incident_id,
            "start_time": execution_start.isoformat(),
            "end_time": None,
            "success": None,
            "result": None,
            "error": None,
        }
        
        self.logger.info(f"Executing emergency action: {action['name']} (ID: {action_id})")
        
        try:
            # Get the execution function
            execution_function = action["execution_function"]
            
            # Execute the function
            result = await execution_function(parameters)
            
            # Update execution record
            execution_record["success"] = True
            execution_record["result"] = result
            execution_record["end_time"] = datetime.now().isoformat()
            
            # Update last execution time for cooldown
            self._last_execution_times[action_id] = datetime.now().timestamp()
            
            self.logger.info(f"Emergency action executed successfully: {action['name']} (ID: {action_id})")
            
        except Exception as e:
            # Update execution record with error
            execution_record["success"] = False
            execution_record["error"] = str(e)
            execution_record["end_time"] = datetime.now().isoformat()
            
            self.logger.error(f"Error executing emergency action {action_id}: {str(e)}")
        
        # Store execution record
        self._add_to_execution_history(execution_record)
        
        # Return result
        return {
            "success": execution_record["success"],
            "execution_id": execution_id,
            "message": "Action executed successfully" if execution_record["success"] else execution_record["error"],
            "result": execution_record["result"],
        }

    async def get_available_actions(self) -> List[Dict[str, Any]]:
        """
        Get list of available emergency actions.
        
        Returns:
            List of action definitions (without execution functions)
        """
        actions = []
        
        for action_id, action in self._actions.items():
            # Create a copy without the execution function
            action_info = {k: v for k, v in action.items() if k != "execution_function"}
            actions.append(action_info)
        
        return actions

    async def get_execution_history(
        self,
        action_id: Optional[str] = None,
        incident_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get history of action executions.
        
        Args:
            action_id: Filter by action ID
            incident_id: Filter by incident ID
            limit: Maximum number of records to return
            
        Returns:
            List of execution records
        """
        filtered_history = []
        
        for record in self._execution_history:
            # Apply filters
            if action_id and record["action_id"] != action_id:
                continue
            if incident_id and record["incident_id"] != incident_id:
                continue
            
            filtered_history.append(record)
        
        # Return most recent entries first
        return sorted(filtered_history, key=lambda x: x["start_time"], reverse=True)[:limit]

    def _add_to_execution_history(self, record: Dict[str, Any]) -> None:
        """
        Add a record to the execution history.
        
        Args:
            record: Execution record to add
        """
        self._execution_history.append(record)
        
        # Trim history if too large
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = self._execution_history[-self._max_history_size:]

    def _register_default_actions(self) -> None:
        """Register the default set of emergency actions."""
        # Pause all trading
        self.register_action(
            action_id="pause_all_trading",
            name="Pause All Trading",
            description="Temporarily halt all trading activities",
            execution_function=self._pause_all_trading,
            safety_level="high",
            parameters={},
            confirmation_required=True
        )
        
        # Enable circuit breaker
        self.register_action(
            action_id="enable_circuit_breaker",
            name="Enable Circuit Breaker",
            description="Activate circuit breaker at specified level",
            execution_function=self._enable_circuit_breaker,
            safety_level="high",
            parameters={"level": "system"},
            confirmation_required=True
        )
        
        # Close high-risk positions
        self.register_action(
            action_id="close_risky_positions",
            name="Close High-Risk Positions",
            description="Close positions that exceed risk thresholds",
            execution_function=self._close_risky_positions,
            safety_level="high",
            parameters={"risk_level": "high"},
            confirmation_required=True
        )
        
        # Disable new positions
        self.register_action(
            action_id="disable_new_positions",
            name="Disable New Position Opening",
            description="Prevent opening of new positions while allowing closing",
            execution_function=self._disable_new_positions,
            safety_level="medium",
            parameters={},
            confirmation_required=True
        )
        
        # Attempt reconnection
        self.register_action(
            action_id="attempt_reconnect",
            name="Attempt Reconnection",
            description="Try to re-establish connection to trading services",
            execution_function=self._attempt_reconnect,
            safety_level="low",
            parameters={"max_attempts": 3},
            confirmation_required=False
        )
        
        # Switch to backup connection
        self.register_action(
            action_id="switch_to_backup",
            name="Switch to Backup Connection",
            description="Switch to backup trading connection",
            execution_function=self._switch_to_backup,
            safety_level="medium",
            parameters={},
            confirmation_required=True
        )
        
        # Throttle order submission
        self.register_action(
            action_id="throttle_order_submission",
            name="Throttle Order Submission",
            description="Reduce the rate of order submissions",
            execution_function=self._throttle_order_submission,
            safety_level="low",
            parameters={"rate": "medium"},
            confirmation_required=False
        )
        
        # Pause trading on affected instruments
        self.register_action(
            action_id="pause_affected_instruments",
            name="Pause Trading on Affected Instruments",
            description="Halt trading on specific instruments",
            execution_function=self._pause_affected_instruments,
            safety_level="medium",
            parameters={},
            confirmation_required=True
        )
        
        # Switch data source
        self.register_action(
            action_id="switch_data_source",
            name="Switch to Alternate Data Source",
            description="Change the source of market data",
            execution_function=self._switch_data_source,
            safety_level="medium",
            parameters={},
            confirmation_required=True
        )
        
        # Reconcile positions
        self.register_action(
            action_id="reconcile_positions",
            name="Reconcile Positions",
            description="Verify and reconcile positions with broker",
            execution_function=self._reconcile_positions,
            safety_level="low",
            parameters={},
            confirmation_required=False
        )
        
        # Adjust order parameters
        self.register_action(
            action_id="adjust_order_parameters",
            name="Adjust Order Parameters",
            description="Modify default order parameters",
            execution_function=self._adjust_order_parameters,
            safety_level="medium",
            parameters={"slippage": "normal"},
            confirmation_required=True
        )
        
        # Activate backup systems
        self.register_action(
            action_id="activate_backup_systems",
            name="Activate Backup Systems",
            description="Switch to backup trading systems",
            execution_function=self._activate_backup_systems,
            safety_level="high",
            parameters={},
            confirmation_required=True
        )

    # Implementation of action execution functions
    async def _pause_all_trading(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pause all trading activities.
        
        Args:
            parameters: Action parameters
            
        Returns:
            Result of action execution
        """
        if not self.trading_gateway_client:
            raise ValueError("Trading gateway client not available")
            
        try:
            # Call the trading gateway to pause trading
            if hasattr(self.trading_gateway_client, "pause_all_trading"):
                result = await self.trading_gateway_client.pause_all_trading()
                return {"paused": True, "details": result}
            else:
                raise ValueError("Trading gateway client does not support pause_all_trading")
                
        except Exception as e:
            self.logger.error(f"Error pausing all trading: {str(e)}")
            raise

    async def _enable_circuit_breaker(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enable circuit breaker at specified level.
        
        Args:
            parameters: Action parameters including 'level'
            
        Returns:
            Result of action execution
        """
        if not self.circuit_breaker:
            raise ValueError("Circuit breaker not available")
            
        level = parameters.get("level", "system")
        
        try:
            # Call the circuit breaker to activate
            if hasattr(self.circuit_breaker, "activate"):
                result = await self.circuit_breaker.activate(level=level)
                return {"activated": True, "level": level, "details": result}
            else:
                raise ValueError("Circuit breaker does not support activate method")
                
        except Exception as e:
            self.logger.error(f"Error enabling circuit breaker: {str(e)}")
            raise

    async def _close_risky_positions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Close positions that exceed risk thresholds.
        
        Args:
            parameters: Action parameters including 'risk_level'
            
        Returns:
            Result of action execution
        """
        if not self.position_manager:
            raise ValueError("Position manager not available")
            
        risk_level = parameters.get("risk_level", "high")
        
        try:
            # Call the position manager to close risky positions
            if hasattr(self.position_manager, "close_risky_positions"):
                result = await self.position_manager.close_risky_positions(risk_level=risk_level)
                return {
                    "positions_closed": result.get("positions_closed", 0),
                    "details": result
                }
            else:
                raise ValueError("Position manager does not support close_risky_positions method")
                
        except Exception as e:
            self.logger.error(f"Error closing risky positions: {str(e)}")
            raise

    async def _disable_new_positions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prevent opening of new positions while allowing closing.
        
        Args:
            parameters: Action parameters
            
        Returns:
            Result of action execution
        """
        if not self.trading_gateway_client:
            raise ValueError("Trading gateway client not available")
            
        try:
            # Call the trading gateway to disable new positions
            if hasattr(self.trading_gateway_client, "set_trading_mode"):
                result = await self.trading_gateway_client.set_trading_mode("close_only")
                return {"mode_set": "close_only", "details": result}
            else:
                raise ValueError("Trading gateway client does not support set_trading_mode method")
                
        except Exception as e:
            self.logger.error(f"Error disabling new positions: {str(e)}")
            raise

    async def _attempt_reconnect(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Try to re-establish connection to trading services.
        
        Args:
            parameters: Action parameters including 'max_attempts'
            
        Returns:
            Result of action execution
        """
        if not self.trading_gateway_client:
            raise ValueError("Trading gateway client not available")
            
        max_attempts = int(parameters.get("max_attempts", 3))
        
        try:
            # Call the trading gateway to reconnect
            if hasattr(self.trading_gateway_client, "reconnect"):
                attempts = 0
                success = False
                
                while attempts < max_attempts and not success:
                    attempts += 1
                    self.logger.info(f"Reconnection attempt {attempts}/{max_attempts}")
                    
                    try:
                        result = await self.trading_gateway_client.reconnect()
                        success = result.get("success", False)
                        
                        if success:
                            return {
                                "reconnected": True,
                                "attempts": attempts,
                                "details": result
                            }
                        
                        # Wait before next attempt
                        await asyncio.sleep(2 ** attempts)  # Exponential backoff
                        
                    except Exception as e:
                        self.logger.warning(f"Reconnection attempt {attempts} failed: {str(e)}")
                
                if not success:
                    return {
                        "reconnected": False,
                        "attempts": attempts,
                        "reason": "Max attempts reached"
                    }
            else:
                raise ValueError("Trading gateway client does not support reconnect method")
                
        except Exception as e:
            self.logger.error(f"Error attempting reconnection: {str(e)}")
            raise

    async def _switch_to_backup(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Switch to backup trading connection.
        
        Args:
            parameters: Action parameters
            
        Returns:
            Result of action execution
        """
        if not self.trading_gateway_client:
            raise ValueError("Trading gateway client not available")
            
        try:
            # Call the trading gateway to switch to backup connection
            if hasattr(self.trading_gateway_client, "switch_to_backup_connection"):
                result = await self.trading_gateway_client.switch_to_backup_connection()
                return {"switched": True, "details": result}
            else:
                raise ValueError("Trading gateway client does not support switch_to_backup_connection method")
                
        except Exception as e:
            self.logger.error(f"Error switching to backup connection: {str(e)}")
            raise

    async def _throttle_order_submission(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduce the rate of order submissions.
        
        Args:
            parameters: Action parameters including 'rate'
            
        Returns:
            Result of action execution
        """
        if not self.trading_gateway_client:
            raise ValueError("Trading gateway client not available")
            
        rate = parameters.get("rate", "medium")
        
        # Map rate to numeric value (orders per minute)
        rate_map = {
            "low": 1,
            "medium": 5,
            "high": 20,
            "normal": 60
        }
        
        numeric_rate = rate_map.get(rate, 5)
        
        try:
            # Call the trading gateway to set order rate limit
            if hasattr(self.trading_gateway_client, "set_order_rate_limit"):
                result = await self.trading_gateway_client.set_order_rate_limit(orders_per_minute=numeric_rate)
                return {"rate_limited": True, "rate": rate, "orders_per_minute": numeric_rate, "details": result}
            else:
                raise ValueError("Trading gateway client does not support set_order_rate_limit method")
                
        except Exception as e:
            self.logger.error(f"Error throttling order submission: {str(e)}")
            raise

    async def _pause_affected_instruments(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Halt trading on specific instruments.
        
        Args:
            parameters: Action parameters
            
        Returns:
            Result of action execution
        """
        if not self.trading_gateway_client:
            raise ValueError("Trading gateway client not available")
            
        # The actual instruments would typically be determined from the incident context
        # or specified in parameters, but for simplicity we'll mock this
        instruments = parameters.get("instruments", [])
        
        try:
            # Call the trading gateway to pause specific instruments
            if hasattr(self.trading_gateway_client, "pause_instruments"):
                result = await self.trading_gateway_client.pause_instruments(instruments=instruments)
                return {"paused_instruments": instruments, "details": result}
            else:
                raise ValueError("Trading gateway client does not support pause_instruments method")
                
        except Exception as e:
            self.logger.error(f"Error pausing affected instruments: {str(e)}")
            raise

    async def _switch_data_source(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Change the source of market data.
        
        Args:
            parameters: Action parameters
            
        Returns:
            Result of action execution
        """
        # This would typically involve interfacing with the data pipeline service
        # For now, we'll mock the implementation
        target_source = parameters.get("source", "backup")
        
        try:
            # Simulated success
            return {"switched_to": target_source, "status": "success"}
                
        except Exception as e:
            self.logger.error(f"Error switching data source: {str(e)}")
            raise

    async def _reconcile_positions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify and reconcile positions with broker.
        
        Args:
            parameters: Action parameters
            
        Returns:
            Result of action execution
        """
        if not self.position_manager:
            raise ValueError("Position manager not available")
            
        try:
            # Call the position manager to reconcile positions
            if hasattr(self.position_manager, "reconcile_positions"):
                result = await self.position_manager.reconcile_positions()
                return {
                    "reconciled": True,
                    "discrepancies_found": result.get("discrepancies", 0),
                    "details": result
                }
            else:
                raise ValueError("Position manager does not support reconcile_positions method")
                
        except Exception as e:
            self.logger.error(f"Error reconciling positions: {str(e)}")
            raise

    async def _adjust_order_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify default order parameters.
        
        Args:
            parameters: Action parameters including 'slippage'
            
        Returns:
            Result of action execution
        """
        if not self.trading_gateway_client:
            raise ValueError("Trading gateway client not available")
            
        # Extract parameters
        params_to_adjust = {}
        if "slippage" in parameters:
            slippage_map = {
                "low": 1,
                "normal": 3,
                "higher": 5,
                "high": 10
            }
            params_to_adjust["slippage_pips"] = slippage_map.get(parameters["slippage"], 3)
        
        try:
            # Call the trading gateway to adjust parameters
            if hasattr(self.trading_gateway_client, "set_default_order_parameters"):
                result = await self.trading_gateway_client.set_default_order_parameters(params_to_adjust)
                return {"parameters_adjusted": list(params_to_adjust.keys()), "details": result}
            else:
                raise ValueError("Trading gateway client does not support set_default_order_parameters method")
                
        except Exception as e:
            self.logger.error(f"Error adjusting order parameters: {str(e)}")
            raise

    async def _activate_backup_systems(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Switch to backup trading systems.
        
        Args:
            parameters: Action parameters
            
        Returns:
            Result of action execution
        """
        # This would typically involve interfacing with multiple services
        # For now, we'll mock the implementation
        try:
            # Simulate backup system activation
            await asyncio.sleep(2)  # Simulated activation time
            
            return {
                "activated": True,
                "systems": ["trading", "data", "execution"],
                "status": "backup_active"
            }
                
        except Exception as e:
            self.logger.error(f"Error activating backup systems: {str(e)}")
            raise
