"""
Advanced Optimization Components

This module implements sophisticated optimization algorithms for trading strategy
parameters, including regime-aware optimization, multi-objective optimization,
and online learning capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    regime_performance: Optional[Dict[str, Dict[str, float]]] = None
    convergence_info: Dict[str, Any] = None

class RegimeAwareOptimizer:
    """
    Optimizer that considers different market regimes when tuning parameters.
    """
    
    def __init__(
        self,
        regime_detector: Any,  # MarketRegimeDetector instance
        base_optimizer: Optional[Any] = None,
        regime_weights: Optional[Dict[str, float]] = None
    ):
        self.regime_detector = regime_detector
        self.base_optimizer = base_optimizer or GaussianProcessRegressor()
        self.regime_weights = regime_weights or {}
        self.regime_models = {}
        
    def optimize(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_func: Callable,
        market_data: pd.DataFrame,
        n_trials: int = 100,
        min_samples_per_regime: int = 20
    ) -> OptimizationResult:
        """
        Perform regime-aware parameter optimization.
        
        Args:
            parameter_space: Dictionary of parameter names and their bounds
            objective_func: Function to optimize
            market_data: Historical market data for regime detection
            n_trials: Number of optimization trials
            min_samples_per_regime: Minimum samples needed per regime
            
        Returns:
            OptimizationResult object with optimal parameters
        """
        # Detect regimes in market data
        regimes = self.regime_detector.detect_regimes(market_data)
        
        # Split data by regime
        regime_data = {}
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_data[regime] = market_data[regime_mask]
        
        # Optimize for each regime
        regime_params = {}
        for regime, data in regime_data.items():
            if len(data) >= min_samples_per_regime:
                # Create regime-specific objective function
                def regime_objective(params):
                    return objective_func(params, data)
                
                # Optimize for this regime
                result = self._optimize_single_regime(
                    parameter_space,
                    regime_objective,
                    n_trials
                )
                regime_params[regime] = result
        
        # Combine regime-specific results
        final_params = self._combine_regime_parameters(regime_params)
        
        return OptimizationResult(
            parameters=final_params,
            objectives={regime: res.objectives for regime, res in regime_params.items()},
            regime_performance={regime: self._evaluate_performance(params, objective_func, regime_data[regime])
                              for regime, params in regime_params.items()}
        )
    
    def _optimize_single_regime(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_func: Callable,
        n_trials: int
    ) -> OptimizationResult:
        """Optimize parameters for a single regime."""
        # Initialize parameter values for this trial
        best_params = None
        best_score = float('inf')
        
        for _ in range(n_trials):
            # Generate random starting point
            x0 = {param: np.random.uniform(low, high) 
                  for param, (low, high) in parameter_space.items()}
            
            # Optimize using scipy
            result = minimize(
                lambda x: objective_func({p: v for p, v in zip(parameter_space.keys(), x)}),
                x0=list(x0.values()),
                bounds=[parameter_space[p] for p in parameter_space.keys()],
                method='L-BFGS-B'
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = {p: v for p, v in zip(parameter_space.keys(), result.x)}
        
        return OptimizationResult(
            parameters=best_params,
            objectives={'objective': best_score}
        )
    
    def _combine_regime_parameters(
        self,
        regime_params: Dict[str, OptimizationResult]
    ) -> Dict[str, float]:
        """Combine parameters from different regimes using weights."""
        combined_params = {}
        
        # Use equal weights if not specified
        if not self.regime_weights:
            weight = 1.0 / len(regime_params)
            self.regime_weights = {regime: weight for regime in regime_params.keys()}
        
        # Combine parameters using weighted average
        for param_name in next(iter(regime_params.values())).parameters.keys():
            combined_params[param_name] = sum(
                res.parameters[param_name] * self.regime_weights[regime]
                for regime, res in regime_params.items()
            )
            
        return combined_params
    
    def _evaluate_performance(
        self,
        parameters: Dict[str, float],
        objective_func: Callable,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate parameter performance on given data."""
        score = objective_func(parameters, data)
        return {'objective': score}


class MultiObjectiveOptimizer:
    """
    Optimizer that handles multiple competing objectives (return, risk, robustness).
    """
    
    def __init__(
        self,
        objectives: Dict[str, Tuple[Callable, float]],  # (objective_func, weight)
        constraints: Optional[List[Callable]] = None
    ):
        self.objectives = objectives
        self.constraints = constraints or []
        
    def optimize(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        market_data: pd.DataFrame,
        n_trials: int = 100
    ) -> List[OptimizationResult]:
        """
        Perform multi-objective optimization using Pareto optimization.
        
        Args:
            parameter_space: Dictionary of parameter names and their bounds
            market_data: Historical market data
            n_trials: Number of optimization trials
            
        Returns:
            List of Pareto-optimal solutions
        """
        population = []
        
        # Generate initial population
        for _ in range(n_trials):
            params = {
                param: np.random.uniform(low, high)
                for param, (low, high) in parameter_space.items()
            }
            
            # Evaluate all objectives
            objective_values = {
                name: (func(params, market_data), weight)
                for name, (func, weight) in self.objectives.items()
            }
            
            # Check constraints
            feasible = all(constraint(params, market_data) <= 0 
                          for constraint in self.constraints)
            
            if feasible:
                population.append(OptimizationResult(
                    parameters=params,
                    objectives={name: score for name, (score, _) in objective_values.items()}
                ))
        
        # Find Pareto-optimal solutions
        pareto_front = self._find_pareto_front(population)
        
        return pareto_front
    
    def _find_pareto_front(
        self,
        population: List[OptimizationResult]
    ) -> List[OptimizationResult]:
        """Find Pareto-optimal solutions from the population."""
        pareto_front = []
        
        for solution in population:
            dominated = False
            
            for other in population:
                if self._dominates(other, solution):
                    dominated = True
                    break
            
            if not dominated:
                pareto_front.append(solution)
        
        return pareto_front
    
    def _dominates(
        self,
        solution1: OptimizationResult,
        solution2: OptimizationResult
    ) -> bool:
        """Check if solution1 dominates solution2."""
        better_in_any = False
        
        for obj_name, (_, weight) in self.objectives.items():
            score1 = solution1.objectives[obj_name] * weight
            score2 = solution2.objectives[obj_name] * weight
            
            if score1 > score2:  # Assuming higher is better after applying weights
                better_in_any = True
            elif score1 < score2:
                return False
        
        return better_in_any


class OnlineLearningOptimizer:
    """
    Optimizer that continuously adapts parameters based on new data.
    """
    
    def __init__(
        self,
        base_optimizer: Any,
        learning_rate: float = 0.1,
        window_size: int = 1000
    ):
        self.base_optimizer = base_optimizer
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.parameter_history = []
        
    def update(
        self,
        new_data: pd.DataFrame,
        current_params: Dict[str, float],
        objective_func: Callable
    ) -> Dict[str, float]:
        """
        Update parameters based on new data.
        
        Args:
            new_data: New market data
            current_params: Current parameter values
            objective_func: Objective function to optimize
            
        Returns:
            Updated parameters
        """
        # Add current parameters to history
        self.parameter_history.append(current_params)
        
        # Keep only recent history
        if len(self.parameter_history) > self.window_size:
            self.parameter_history.pop(0)
        
        # Calculate parameter update
        gradient = self._estimate_gradient(objective_func, current_params, new_data)
        
        # Update parameters with gradient and learning rate
        updated_params = {
            param: value + self.learning_rate * gradient.get(param, 0)
            for param, value in current_params.items()
        }
        
        return updated_params
    
    def _estimate_gradient(
        self,
        objective_func: Callable,
        current_params: Dict[str, float],
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Estimate gradient for each parameter."""
        gradient = {}
        epsilon = 1e-6
        
        base_score = objective_func(current_params, data)
        
        for param in current_params:
            # Create parameter variation
            test_params = current_params.copy()
            test_params[param] += epsilon
            
            # Calculate numerical gradient
            test_score = objective_func(test_params, data)
            gradient[param] = (test_score - base_score) / epsilon
        
        return gradient
