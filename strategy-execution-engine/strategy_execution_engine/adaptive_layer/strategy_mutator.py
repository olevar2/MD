"""
Strategy mutation framework for adjusting parameters based on trading feedback.

This module provides functionality to adaptively mutate strategy parameters
based on incoming feedback, performance analysis, and statistical validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import uuid
import logging
import json
import copy

from analysis_engine.adaptive_layer.statistical_validator import StatisticalValidator
from analysis_engine.services.tool_effectiveness import MarketRegime
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyMutator:
    """
    Adaptively mutates strategy parameters based on performance feedback.
    This class implements the feedback-driven strategy mutation framework.
    """
    
    def __init__(self,
                strategy_config: Dict[str, Any],
                validator: Optional[StatisticalValidator] = None,
                min_mutation_interval: int = 10, # minimum number of trades before allowing mutation
                max_mutation_pct: float = 0.2,   # maximum parameter change per mutation
                enable_reversion: bool = True):  # allow reverting bad mutations
        """
        Initialize the strategy mutator.
        
        Args:
            strategy_config: Initial strategy configuration with parameters
            validator: Statistical validator for parameter changes
            min_mutation_interval: Minimum number of trades before re-mutation
            max_mutation_pct: Maximum percentage change for numerical parameters
            enable_reversion: Whether to enable automatic reversion of harmful mutations
        """
        self.strategy_config = copy.deepcopy(strategy_config)
        self.validator = validator or StatisticalValidator()
        self.min_mutation_interval = min_mutation_interval
        self.max_mutation_pct = max_mutation_pct
        self.enable_reversion = enable_reversion
        
        self.mutation_history = []
        self.trade_counter = 0
        self.mutation_counter = 0
        self.parameter_performance = {}
        
        # Initialize mutation configuration
        self._initialize_mutation_config()
        
        self.logger = logger
    
    def _initialize_mutation_config(self) -> None:
        """Initialize the mutation configuration with default settings."""
        # Extract parameter mutation settings if present
        if "mutation_config" in self.strategy_config:
            mutation_config = self.strategy_config["mutation_config"]
        else:
            # Create default mutation configuration
            mutation_config = {
                "mutable_parameters": {},
                "mutation_rules": {},
                "regime_specific_bounds": {}
            }
            
            # Automatically add numerical parameters as mutable
            for param_name, param_value in self.strategy_config.get("parameters", {}).items():
                if isinstance(param_value, (int, float)) and not param_name.startswith("_"):
                    # Define reasonable bounds based on parameter value
                    if isinstance(param_value, int):
                        lower_bound = max(1, int(param_value * 0.5))
                        upper_bound = int(param_value * 2)
                    else:  # float
                        lower_bound = max(0.001, param_value * 0.5)
                        upper_bound = param_value * 2
                        
                    mutation_config["mutable_parameters"][param_name] = {
                        "type": "int" if isinstance(param_value, int) else "float",
                        "bounds": [lower_bound, upper_bound],
                        "step": 1 if isinstance(param_value, int) else 0.01
                    }
        
        self.mutation_config = mutation_config
    
    def register_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Register a new trade result for adaptation feedback.
        
        Args:
            trade_result: Dictionary with trade result data
        """
        self.trade_counter += 1
        
        # Store relevant performance metrics for current parameter set
        current_param_set = json.dumps(self._get_current_parameter_values())
        
        if current_param_set not in self.parameter_performance:
            self.parameter_performance[current_param_set] = {
                "trades": [],
                "metrics": {}
            }
            
        # Add trade to the current parameter performance record
        self.parameter_performance[current_param_set]["trades"].append(trade_result)
        
        # Update aggregated metrics
        trades = self.parameter_performance[current_param_set]["trades"]
        
        # Calculate updated metrics
        win_trades = [t for t in trades if t.get("profit", 0) > 0]
        loss_trades = [t for t in trades if t.get("profit", 0) <= 0]
        
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        total_count = win_count + loss_count
        
        if total_count > 0:
            win_rate = win_count / total_count
            avg_profit = sum(t.get("profit", 0) for t in win_trades) / max(1, win_count)
            avg_loss = sum(t.get("profit", 0) for t in loss_trades) / max(1, loss_count)
            profit_factor = abs(avg_profit * win_count) / abs(avg_loss * loss_count) if loss_count > 0 and avg_loss != 0 else float('inf')
            
            # Update metrics in the performance record
            self.parameter_performance[current_param_set]["metrics"] = {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "win_count": win_count,
                "loss_count": loss_count,
                "total_count": total_count
            }
        
        # Check if we should attempt mutation
        if self.trade_counter % self.min_mutation_interval == 0:
            self._consider_mutation()
    
    def register_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process structured feedback for parameter adaptation.
        
        Args:
            feedback: Dictionary with structured feedback data
            
        Returns:
            Dictionary with processing results
        """
        feedback_type = feedback.get("type", "generic")
        params_affected = feedback.get("affected_parameters", [])
        suggested_changes = feedback.get("suggested_changes", {})
        
        if not params_affected and not suggested_changes:
            return {"status": "ignored", "reason": "No affected parameters or suggested changes"}
            
        # Record the feedback
        feedback_record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "feedback_type": feedback_type,
            "affected_parameters": params_affected,
            "suggested_changes": suggested_changes,
            "applied_changes": {}
        }
        
        # Apply suggested changes if they are within bounds
        applied_changes = {}
        for param_name, change in suggested_changes.items():
            if param_name in self.mutation_config.get("mutable_parameters", {}):
                param_config = self.mutation_config["mutable_parameters"][param_name]
                current_value = self._get_parameter_value(param_name)
                
                if isinstance(change, dict) and "value" in change:
                    # Direct value assignment
                    new_value = self._validate_parameter_value(param_name, change["value"])
                    if new_value is not None:
                        applied_changes[param_name] = {
                            "from": current_value,
                            "to": new_value,
                            "pct_change": ((new_value - current_value) / current_value) if current_value != 0 else float('inf')
                        }
                
                elif isinstance(change, dict) and "pct_change" in change:
                    # Percentage change
                    pct_change = max(min(change["pct_change"], self.max_mutation_pct), -self.max_mutation_pct)
                    new_value = current_value * (1 + pct_change)
                    new_value = self._validate_parameter_value(param_name, new_value)
                    
                    if new_value is not None:
                        applied_changes[param_name] = {
                            "from": current_value,
                            "to": new_value,
                            "pct_change": pct_change
                        }
        
        # Apply the changes to the strategy configuration
        for param_name, change_info in applied_changes.items():
            self._set_parameter_value(param_name, change_info["to"])
            
        # Record the mutation if any changes were applied
        if applied_changes:
            feedback_record["applied_changes"] = applied_changes
            self.mutation_history.append(feedback_record)
            self.mutation_counter += 1
            
            # Log the mutation
            self.logger.info(f"Applied parameter mutations based on feedback: {len(applied_changes)} parameters changed")
            
            return {
                "status": "success",
                "mutation_id": feedback_record["id"],
                "changes_applied": len(applied_changes),
                "details": applied_changes
            }
        else:
            return {
                "status": "no_changes",
                "reason": "No valid parameter changes could be applied"
            }
    
    def mutate_parameters(self, 
                         market_regime: Optional[Union[str, MarketRegime]] = None,
                         mutation_strength: float = 0.5) -> Dict[str, Any]:
        """
        Perform mutation of strategy parameters based on performance and current market regime.
        
        Args:
            market_regime: Current market regime for context-specific mutation
            mutation_strength: How aggressive the mutation should be (0.0 to 1.0)
            
        Returns:
            Dictionary with mutation results
        """
        # Validate mutation strength
        mutation_strength = max(0.0, min(1.0, mutation_strength))
        
        # Convert market regime to string if needed
        if market_regime is not None and not isinstance(market_regime, str):
            market_regime = market_regime.value
        
        # Create mutation record
        mutation_record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "market_regime": market_regime,
            "mutation_strength": mutation_strength,
            "changes": {}
        }
        
        # Get parameters eligible for mutation
        mutable_parameters = self.mutation_config.get("mutable_parameters", {})
        
        # Apply mutations
        param_changes = {}
        for param_name, param_config in mutable_parameters.items():
            # Skip if this parameter shouldn't be mutated for this market regime
            regime_specific_config = self._get_regime_specific_config(param_name, market_regime)
            if regime_specific_config.get("mutable", True) is False:
                continue
                
            current_value = self._get_parameter_value(param_name)
            
            # Get mutation bounds
            bounds = regime_specific_config.get("bounds", param_config["bounds"])
            
            # Calculate mutation step based on parameter type and mutation strength
            if param_config["type"] == "int":
                # Integer mutation
                step = max(1, int(param_config.get("step", 1) * mutation_strength))
                range_size = bounds[1] - bounds[0]
                max_step = max(1, int(range_size * self.max_mutation_pct * mutation_strength))
                
                # Random direction
                direction = 1 if np.random.random() > 0.5 else -1
                mutation_step = direction * np.random.randint(1, max_step + 1)
                
                new_value = current_value + mutation_step
                
            else:  # float
                # Float mutation
                step = param_config.get("step", 0.01) * mutation_strength
                range_size = bounds[1] - bounds[0]
                max_step = range_size * self.max_mutation_pct * mutation_strength
                
                # Random direction and magnitude
                mutation_step = (np.random.random() * 2 - 1) * max_step
                
                new_value = current_value + mutation_step
            
            # Ensure the new value is within bounds
            new_value = max(bounds[0], min(bounds[1], new_value))
            
            # Round to appropriate precision for floats
            if param_config["type"] == "float":
                precision = len(str(param_config.get("step", 0.01)).split(".")[-1])
                new_value = round(new_value, precision)
                
            # Only record changes if the value actually changed
            if new_value != current_value:
                param_changes[param_name] = {
                    "from": current_value,
                    "to": new_value,
                    "pct_change": ((new_value - current_value) / current_value) if current_value != 0 else float('inf')
                }
                
                # Apply the change
                self._set_parameter_value(param_name, new_value)
        
        # Record the mutation if any changes were made
        if param_changes:
            mutation_record["changes"] = param_changes
            self.mutation_history.append(mutation_record)
            self.mutation_counter += 1
            
            # Reset trade counter
            self.trade_counter = 0
            
            # Log the mutation
            self.logger.info(f"Applied parameter mutations: {len(param_changes)} parameters changed")
            
            return {
                "status": "success", 
                "mutation_id": mutation_record["id"],
                "changes": param_changes
            }
        else:
            return {
                "status": "no_changes",
                "reason": "No parameters were eligible for mutation or changed"
            }
    
    def revert_last_mutation(self) -> Dict[str, Any]:
        """
        Revert the most recent parameter mutation.
        
        Returns:
            Dictionary with reversion results
        """
        if not self.mutation_history:
            return {"status": "error", "reason": "No mutations to revert"}
            
        # Get the most recent mutation
        last_mutation = self.mutation_history[-1]
        
        # Revert all parameter changes
        reverted_params = {}
        for param_name, change_info in last_mutation.get("changes", {}).items():
            if "from" in change_info:
                original_value = change_info["from"]
                current_value = self._get_parameter_value(param_name)
                
                # Only revert if the current value matches what we mutated to
                if current_value == change_info.get("to"):
                    self._set_parameter_value(param_name, original_value)
                    reverted_params[param_name] = {
                        "from": current_value,
                        "to": original_value,
                        "reversion": True
                    }
        
        # Update mutation history
        reversion_record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "reverted_mutation_id": last_mutation["id"],
            "changes": reverted_params
        }
        
        self.mutation_history.append(reversion_record)
        
        # Log the reversion
        self.logger.info(f"Reverted mutation {last_mutation['id']}: {len(reverted_params)} parameters restored")
        
        return {
            "status": "success",
            "reversion_id": reversion_record["id"],
            "reverted_mutation_id": last_mutation["id"],
            "params_reverted": len(reverted_params),
            "details": reverted_params
        }
    
    def evaluate_mutation_effectiveness(self, 
                                      mutation_id: str, 
                                      min_trades: int = 10) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of a specific parameter mutation.
        
        Args:
            mutation_id: ID of the mutation to evaluate
            min_trades: Minimum number of trades required for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        # Find the mutation in history
        mutation = None
        for m in self.mutation_history:
            if m.get("id") == mutation_id:
                mutation = m
                break
                
        if not mutation:
            return {"status": "error", "reason": "Mutation ID not found"}
            
        # Get the parameter set before and after mutation
        params_before = {}
        params_after = {}
        
        for param_name, change_info in mutation.get("changes", {}).items():
            params_before[param_name] = change_info.get("from")
            params_after[param_name] = change_info.get("to")
            
        # Convert to JSON strings for lookup in parameter performance
        before_key = json.dumps({**self._get_current_parameter_values(), **params_before})
        after_key = json.dumps({**self._get_current_parameter_values(), **params_after})
        
        # Check if we have enough trade data for both parameter sets
        before_data = self.parameter_performance.get(before_key, {"trades": [], "metrics": {}})
        after_data = self.parameter_performance.get(after_key, {"trades": [], "metrics": {}})
        
        if len(before_data["trades"]) < min_trades or len(after_data["trades"]) < min_trades:
            return {
                "status": "insufficient_data",
                "trades_before": len(before_data["trades"]),
                "trades_after": len(after_data["trades"]),
                "min_trades_required": min_trades
            }
            
        # Perform statistical validation
        win_rate_validation = self.validator.validate_parameter_adjustment(
            before_performance=[{"win_rate": t.get("win_rate", 0)} for t in before_data["trades"]],
            after_performance=[{"win_rate": t.get("win_rate", 0)} for t in after_data["trades"]],
            metric_name="win_rate"
        )
        
        profit_factor_validation = self.validator.validate_parameter_adjustment(
            before_performance=[{"profit_factor": t.get("profit_factor", 1.0)} for t in before_data["trades"]],
            after_performance=[{"profit_factor": t.get("profit_factor", 1.0)} for t in after_data["trades"]],
            metric_name="profit_factor"
        )
        
        # Calculate overall effectiveness
        is_effective = (
            win_rate_validation.get("is_valid", False) or 
            profit_factor_validation.get("is_valid", False)
        )
        
        # Metrics comparison
        metrics_comparison = {}
        for metric, value in after_data["metrics"].items():
            before_value = before_data["metrics"].get(metric)
            if before_value is not None:
                change = value - before_value
                pct_change = (change / before_value) if before_value != 0 else float('inf')
                
                metrics_comparison[metric] = {
                    "before": before_value,
                    "after": value,
                    "change": change,
                    "pct_change": pct_change
                }
        
        return {
            "status": "evaluated",
            "mutation_id": mutation_id,
            "is_effective": is_effective,
            "metrics_comparison": metrics_comparison,
            "statistical_validation": {
                "win_rate": win_rate_validation,
                "profit_factor": profit_factor_validation
            },
            "sample_sizes": {
                "trades_before": len(before_data["trades"]),
                "trades_after": len(after_data["trades"])
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _consider_mutation(self) -> None:
        """Consider whether to perform a mutation based on current performance."""
        # This could involve more complex decision logic based on recent performance
        # For now, just perform a mutation with default settings
        self.mutate_parameters()
    
    def _get_parameter_value(self, param_name: str) -> Any:
        """Get the current value of a strategy parameter."""
        if "." in param_name:
            # Handle nested parameters
            parts = param_name.split(".")
            value = self.strategy_config
            for part in parts:
                if part in value:
                    value = value[part]
                else:
                    return None
            return value
        else:
            # Handle top-level parameters
            return self.strategy_config.get("parameters", {}).get(param_name)
    
    def _set_parameter_value(self, param_name: str, value: Any) -> None:
        """Set the value of a strategy parameter."""
        if "." in param_name:
            # Handle nested parameters
            parts = param_name.split(".")
            target = self.strategy_config
            
            # Navigate to the parent object
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
                
            # Set the value
            target[parts[-1]] = value
        else:
            # Handle top-level parameters
            if "parameters" not in self.strategy_config:
                self.strategy_config["parameters"] = {}
            self.strategy_config["parameters"][param_name] = value
    
    def _get_current_parameter_values(self) -> Dict[str, Any]:
        """Get all current parameter values."""
        return self.strategy_config.get("parameters", {})
    
    def _validate_parameter_value(self, param_name: str, value: Any) -> Any:
        """Validate that a parameter value is within bounds."""
        param_config = self.mutation_config.get("mutable_parameters", {}).get(param_name)
        
        if not param_config:
            return None
            
        # Check type
        if param_config["type"] == "int" and not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                return None
                
        elif param_config["type"] == "float" and not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return None
        
        # Check bounds
        bounds = param_config.get("bounds", [float('-inf'), float('inf')])
        if value < bounds[0] or value > bounds[1]:
            # Clamp to bounds
            value = max(bounds[0], min(bounds[1], value))
            
        return value
    
    def _get_regime_specific_config(self, param_name: str, market_regime: Optional[str]) -> Dict[str, Any]:
        """Get regime-specific configuration for a parameter."""
        # Default to the standard parameter config
        param_config = self.mutation_config.get("mutable_parameters", {}).get(param_name, {})
        
        # If no market regime specified, return default config
        if not market_regime:
            return param_config
            
        # Check for regime-specific config
        regime_configs = self.mutation_config.get("regime_specific_bounds", {})
        regime_param_config = regime_configs.get(market_regime, {}).get(param_name, {})
        
        # Merge with default config, with regime-specific taking precedence
        return {**param_config, **regime_param_config}
