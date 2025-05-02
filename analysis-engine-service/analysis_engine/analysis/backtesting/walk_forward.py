"""
Walk-Forward Optimization Module.

This module implements walk-forward optimization techniques to prevent overfitting
and ensure proper out-of-sample testing for strategy validation.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import concurrent.futures
from dataclasses import dataclass, field

from analysis_engine.analysis.backtesting.core import (
    BacktestResult, BacktestConfiguration, BacktestEngine, DataSplit
)
from analysis_engine.utils.logger import get_logger

logger = get_logger(__name__)


class WalkForwardWindow(BaseModel):
    """Defines a single window for walk-forward optimization."""
    
    window_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimization_result: Optional[Dict[str, Any]] = None
    backtest_result: Optional[Dict[str, Any]] = None


class WalkForwardOptimizationConfig(BaseModel):
    """Configuration for walk-forward optimization process."""
    
    strategy_name: str
    parameter_space: Dict[str, List[Any]]
    start_date: datetime
    end_date: datetime
    instruments: List[str]
    train_size: int = 180  # days
    test_size: int = 60    # days
    step_size: int = 60    # days (how much to step forward)
    train_test_gap: int = 0  # days (gap between train and test)
    initial_capital: float = 100000.0
    data_timeframe: str = "1H"
    n_jobs: int = 1  # parallel jobs
    optimization_metric: str = "sharpe_ratio"
    optimization_direction: str = "maximize"
    min_window_bars: int = 100  # Minimum number of bars required for a window
    position_sizing: Optional[str] = "fixed"
    position_sizing_settings: Dict[str, Any] = Field(default_factory=dict)
    slippage_model: Optional[str] = "fixed"
    slippage_settings: Dict[str, Any] = Field(default_factory=dict)
    commission_model: Optional[str] = "fixed"
    commission_settings: Dict[str, Any] = Field(default_factory=dict)
    risk_management_settings: Dict[str, Any] = Field(default_factory=dict)
    data_settings: Dict[str, Any] = Field(default_factory=dict)
    
    def generate_windows(self) -> List[WalkForwardWindow]:
        """Generate walk-forward windows based on configuration."""
        windows = []
        current_train_start = self.start_date
        
        while True:
            # Calculate window boundaries
            train_end = current_train_start + timedelta(days=self.train_size)
            test_start = train_end + timedelta(days=self.train_test_gap)
            test_end = test_start + timedelta(days=self.test_size)
            
            # Stop if test end exceeds the overall end date
            if test_end > self.end_date:
                break
                
            # Create the window
            window = WalkForwardWindow(
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
            windows.append(window)
            
            # Move to the next window
            current_train_start += timedelta(days=self.step_size)
        
        return windows


class WalkForwardResult(BaseModel):
    """Results of a walk-forward optimization run."""
    
    optimization_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str
    start_date: datetime
    end_date: datetime
    windows: List[WalkForwardWindow]
    overall_performance: Dict[str, float] = Field(default_factory=dict)
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    best_parameters: Dict[str, Any] = Field(default_factory=dict)
    parameter_robustness: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WalkForwardOptimizer:
    """
    Performs walk-forward optimization for a trading strategy.
    
    Walk-forward optimization is a technique to reduce overfitting by:
    1. Dividing the data into multiple windows
    2. For each window, optimizing parameters on the training period
    3. Testing the best parameters on the out-of-sample test period
    4. Combining results to assess strategy robustness
    """
    
    def __init__(self, 
               config: WalkForwardOptimizationConfig,
               strategy_factory: Callable[[BacktestConfiguration], BacktestEngine],
               data_provider: Callable[[List[str], datetime, datetime, str], Dict[str, pd.DataFrame]]):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            config: Configuration for the walk-forward optimization
            strategy_factory: Function that creates a strategy instance
            data_provider: Function that provides market data
        """
        self.config = config
        self.strategy_factory = strategy_factory
        self.data_provider = data_provider
        self.windows = config.generate_windows()
        self.results = None
    
    def run_optimization(self) -> WalkForwardResult:
        """
        Run the walk-forward optimization process.
        
        Returns:
            WalkForwardResult: The results of the optimization
        """
        logger.info(f"Starting walk-forward optimization for {self.config.strategy_name}")
        logger.info(f"Generated {len(self.windows)} windows")
        
        # Process each window
        for i, window in enumerate(self.windows):
            logger.info(f"Processing window {i+1}/{len(self.windows)}: "
                      f"{window.train_start.date()} - {window.test_end.date()}")
            
            # Optimize on training data
            best_params, optimization_result = self._optimize_window(window)
            window.optimization_result = optimization_result
            
            # Test on out-of-sample data
            backtest_result = self._test_window(window, best_params)
            window.backtest_result = backtest_result
        
        # Combine results
        overall_performance = self._calculate_overall_performance()
        equity_curve = self._combine_equity_curves()
        parameter_robustness = self._analyze_parameter_robustness()
        
        # Create result object
        self.results = WalkForwardResult(
            strategy_name=self.config.strategy_name,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            windows=self.windows,
            overall_performance=overall_performance,
            equity_curve=equity_curve,
            best_parameters=self._get_most_robust_parameters(),
            parameter_robustness=parameter_robustness
        )
        
        return self.results
    
    def _optimize_window(self, window: WalkForwardWindow) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize strategy parameters for a single window's training period.
        
        Args:
            window: The window to optimize
            
        Returns:
            tuple: (best_parameters, optimization_details)
        """
        logger.info(f"Optimizing window: {window.train_start.date()} - {window.train_end.date()}")
        
        # Get training data
        train_data = self.data_provider(
            self.config.instruments, 
            window.train_start, 
            window.train_end,
            self.config.data_timeframe
        )
        
        # Check if we have enough data
        if not train_data or not all(len(df) >= self.config.min_window_bars for df in train_data.values()):
            logger.warning("Not enough data for training window")
            return {}, {"error": "Insufficient data"}
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        
        # Evaluate each parameter combination
        results = []
        
        # Use parallel processing if configured
        if self.config.n_jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                # Create a list of futures
                futures = [executor.submit(
                    self._evaluate_parameters, 
                    params, 
                    train_data, 
                    window
                ) for params in param_combinations]
                
                # Gather results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parameter evaluation: {e}")
        else:
            # Sequential processing
            for params in param_combinations:
                try:
                    result = self._evaluate_parameters(params, train_data, window)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
        
        # Find best result
        if not results:
            logger.error("No valid parameter combinations found")
            return {}, {"error": "No valid results"}
            
        # Sort by the optimization metric
        optimization_metric = self.config.optimization_metric
        if self.config.optimization_direction == "maximize":
            sorted_results = sorted(results, key=lambda x: x["metrics"].get(optimization_metric, float('-inf')), reverse=True)
        else:
            sorted_results = sorted(results, key=lambda x: x["metrics"].get(optimization_metric, float('inf')))
            
        best_result = sorted_results[0]
        best_params = best_result["parameters"]
        
        # Create optimization details
        optimization_details = {
            "evaluated_parameters": len(results),
            "best_parameters": best_params,
            "best_metric": {optimization_metric: best_result["metrics"].get(optimization_metric)},
            "parameter_statistics": self._calculate_parameter_statistics(results),
            "sorted_results": sorted_results[:10]  # Top 10 results
        }
        
        return best_params, optimization_details
        
    def _test_window(self, window: WalkForwardWindow, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test the optimized parameters on the window's out-of-sample test period.
        
        Args:
            window: The window to test
            parameters: Strategy parameters to test
            
        Returns:
            dict: Test results
        """
        logger.info(f"Testing window: {window.test_start.date()} - {window.test_end.date()}")
        
        # Get test data
        test_data = self.data_provider(
            self.config.instruments, 
            window.test_start, 
            window.test_end,
            self.config.data_timeframe
        )
        
        # Check if we have enough data
        if not test_data or not all(len(df) >= self.config.min_window_bars for df in test_data.values()):
            logger.warning("Not enough data for test window")
            return {"error": "Insufficient data"}
        
        # Create backtest configuration
        backtest_config = BacktestConfiguration(
            strategy_name=self.config.strategy_name,
            strategy_parameters=parameters,
            start_date=window.test_start,
            end_date=window.test_end,
            initial_capital=self.config.initial_capital,
            instruments=self.config.instruments,
            data_timeframe=self.config.data_timeframe,
            slippage_model=self.config.slippage_model,
            slippage_settings=self.config.slippage_settings,
            commission_model=self.config.commission_model,
            commission_settings=self.config.commission_settings,
            position_sizing=self.config.position_sizing,
            position_sizing_settings=self.config.position_sizing_settings,
            risk_management_settings=self.config.risk_management_settings,
            data_settings=self.config.data_settings
        )
        
        # Create and run strategy
        strategy = self.strategy_factory(backtest_config)
        result = strategy.run_backtest(test_data)
        
        # Extract results
        test_result = {
            "metrics": result.dict().get("metrics", {}),
            "final_portfolio_value": result.final_portfolio_value,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "trades_total": result.trades_total,
            "win_rate": result.win_rate
        }
        
        return test_result
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations to test."""
        param_space = self.config.parameter_space
        param_keys = list(param_space.keys())
        
        # Generate all combinations
        combinations = []
        
        # Helper function to generate combinations recursively
        def generate_combinations(index, current_params):
            if index == len(param_keys):
                combinations.append(current_params.copy())
                return
                
            key = param_keys[index]
            values = param_space[key]
            
            for value in values:
                current_params[key] = value
                generate_combinations(index + 1, current_params)
        
        generate_combinations(0, {})
        return combinations
    
    def _evaluate_parameters(
        self,
        parameters: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        window: WalkForwardWindow
    ) -> Dict[str, Any]:
        """
        Evaluate a single parameter set on training data.
        
        Args:
            parameters: Parameters to evaluate
            data: Training data
            window: The current window
            
        Returns:
            dict: Evaluation results
        """
        try:
            # Create backtest configuration
            backtest_config = BacktestConfiguration(
                strategy_name=self.config.strategy_name,
                strategy_parameters=parameters,
                start_date=window.train_start,
                end_date=window.train_end,
                initial_capital=self.config.initial_capital,
                instruments=self.config.instruments,
                data_timeframe=self.config.data_timeframe,
                slippage_model=self.config.slippage_model,
                slippage_settings=self.config.slippage_settings,
                commission_model=self.config.commission_model,
                commission_settings=self.config.commission_settings,
                position_sizing=self.config.position_sizing,
                position_sizing_settings=self.config.position_sizing_settings,
                risk_management_settings=self.config.risk_management_settings,
                data_settings=self.config.data_settings
            )
            
            # Create and run strategy
            strategy = self.strategy_factory(backtest_config)
            result = strategy.run_backtest(data)
            
            # Extract metrics
            evaluation = {
                "parameters": parameters,
                "metrics": result.dict().get("metrics", {}),
                "final_portfolio_value": result.final_portfolio_value,
                "total_return": result.total_return,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "trades_total": result.trades_total,
                "win_rate": result.win_rate
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return {
                "parameters": parameters,
                "metrics": {},
                "error": str(e)
            }
    
    def _calculate_parameter_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics about parameter performance across evaluations.
        
        Args:
            results: List of evaluation results
            
        Returns:
            dict: Parameter statistics
        """
        if not results:
            return {}
            
        # Extract all parameter names
        parameter_keys = results[0]["parameters"].keys()
        
        # Initialize statistics dictionary
        stats = {}
        
        # Get the optimization metric
        optimization_metric = self.config.optimization_metric
        
        # Process each parameter
        for param_key in parameter_keys:
            # Extract unique values for this parameter
            unique_values = set(result["parameters"].get(param_key) for result in results)
            
            value_stats = {}
            
            # Calculate statistics for each parameter value
            for value in unique_values:
                # Filter results with this parameter value
                filtered_results = [r for r in results if r["parameters"].get(param_key) == value]
                
                # Extract metric values
                metric_values = [r["metrics"].get(optimization_metric, 0) for r in filtered_results]
                
                if metric_values:
                    value_stats[str(value)] = {
                        "count": len(metric_values),
                        "mean": float(np.mean(metric_values)),
                        "median": float(np.median(metric_values)),
                        "std": float(np.std(metric_values)),
                        "min": float(np.min(metric_values)),
                        "max": float(np.max(metric_values))
                    }
            
            stats[param_key] = value_stats
        
        return stats
    
    def _calculate_overall_performance(self) -> Dict[str, float]:
        """
        Calculate overall performance across all test windows.
        
        Returns:
            dict: Overall performance metrics
        """
        # Collect metrics from all test windows
        test_results = [window.backtest_result for window in self.windows 
                      if window.backtest_result and "error" not in window.backtest_result]
        
        if not test_results:
            return {}
            
        # Calculate overall metrics
        overall = {}
        
        # Average metrics
        metric_keys = ["total_return", "max_drawdown", "sharpe_ratio", "win_rate"]
        for key in metric_keys:
            values = [result[key] for result in test_results if key in result]
            if values:
                overall[f"avg_{key}"] = sum(values) / len(values)
                overall[f"std_{key}"] = float(np.std(values))
                overall[f"min_{key}"] = min(values)
                overall[f"max_{key}"] = max(values)
        
        # Compounded return
        overall["compounded_return"] = np.prod([1 + result.get("total_return", 0) for result in test_results]) - 1
        
        # Number of profitable windows
        profitable_windows = sum(1 for result in test_results if result.get("total_return", 0) > 0)
        overall["profitable_windows"] = profitable_windows
        overall["profitable_window_percentage"] = profitable_windows / len(test_results) if test_results else 0
        
        # Consistency score - ratio of avg return to std dev of returns
        if "avg_total_return" in overall and "std_total_return" in overall and overall["std_total_return"] > 0:
            overall["consistency_score"] = overall["avg_total_return"] / overall["std_total_return"]
        else:
            overall["consistency_score"] = 0
        
        return overall
    
    def _combine_equity_curves(self) -> List[Dict[str, Any]]:
        """
        Combine equity curves from all test windows.
        
        Returns:
            list: Combined equity curve
        """
        # This implementation would combine the equity curves from each test window
        # For simplicity, we'll return a placeholder
        return []
    
    def _analyze_parameter_robustness(self) -> Dict[str, float]:
        """
        Analyze how robust parameters are across different windows.
        
        Returns:
            dict: Parameter robustness scores
        """
        if not self.windows or not all(window.optimization_result for window in self.windows):
            return {}
            
        # Extract best parameters from each window
        best_params_list = [
            window.optimization_result.get("best_parameters", {})
            for window in self.windows
            if (window.optimization_result and 
                "best_parameters" in window.optimization_result and 
                window.optimization_result["best_parameters"])
        ]
        
        if not best_params_list:
            return {}
        
        # Get all parameter names
        all_param_keys = set()
        for params in best_params_list:
            all_param_keys.update(params.keys())
            
        # Calculate robustness for each parameter
        robustness = {}
        
        for key in all_param_keys:
            # Get all values for this parameter
            values = [params.get(key) for params in best_params_list if key in params]
            
            if not values:
                continue
                
            # For numeric parameters
            if all(isinstance(v, (int, float)) for v in values):
                # Calculate variation coefficient (std/mean)
                mean_value = np.mean(values)
                std_value = np.std(values)
                if mean_value != 0:
                    variation = std_value / abs(mean_value)
                    # Convert to robustness score (0-1, higher is better)
                    robustness[key] = max(0, 1 - min(variation, 1))
                else:
                    robustness[key] = 0
            else:
                # For non-numeric parameters, calculate consistency
                value_counts = {}
                for v in values:
                    value_counts[v] = value_counts.get(v, 0) + 1
                
                # Most common value frequency
                most_common = max(value_counts.values()) if value_counts else 0
                consistency = most_common / len(values) if values else 0
                robustness[key] = consistency
        
        return robustness
    
    def _get_most_robust_parameters(self) -> Dict[str, Any]:
        """
        Get the most robust parameter set across all windows.
        
        Returns:
            dict: Most robust parameters
        """
        if not hasattr(self, 'results') or not self.results:
            return {}
            
        # Get parameter robustness
        parameter_robustness = self.results.parameter_robustness
        
        # Extract best parameters from each window
        best_params_list = [
            window.optimization_result.get("best_parameters", {})
            for window in self.windows
            if (window.optimization_result and 
                "best_parameters" in window.optimization_result and 
                window.optimization_result["best_parameters"])
        ]
        
        if not best_params_list:
            return {}
        
        # For each parameter, choose value with best performance or most common
        robust_params = {}
        
        for param, robustness in parameter_robustness.items():
            # Extract values for this parameter
            param_values = [params.get(param) for params in best_params_list if param in params]
            
            if not param_values:
                continue
                
            # For numeric parameters with high robustness, use average
            if all(isinstance(v, (int, float)) for v in param_values) and robustness > 0.7:
                mean_value = np.mean(param_values)
                
                # If parameter should be an integer
                if all(isinstance(v, int) for v in param_values):
                    robust_params[param] = int(round(mean_value))
                else:
                    robust_params[param] = mean_value
            else:
                # Otherwise use most common value
                value_counts = {}
                for v in param_values:
                    value_counts[v] = value_counts.get(v, 0) + 1
                
                # Find most common value
                most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                robust_params[param] = most_common
        
        return robust_params
