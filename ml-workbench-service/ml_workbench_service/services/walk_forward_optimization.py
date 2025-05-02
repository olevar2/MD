"""
Walk-Forward Optimization Module

This module implements walk-forward optimization (WFO) for trading strategies,
which helps prevent curve-fitting and provides more realistic performance estimates
by properly separating in-sample optimization from out-of-sample testing.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import uuid
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from pydantic import BaseModel, Field

from core_foundations.utils.logger import get_logger
from ml_workbench_service.services.auto_optimization_framework import OptimizationResult, OptimizationConfiguration

logger = get_logger(__name__)


class WalkForwardMethod(str, Enum):
    """Methods for walk-forward optimization."""
    ANCHORED = "anchored"  # Fixed start date, expanding window
    ROLLING = "rolling"    # Moving window of fixed size
    SLIDING = "sliding"    # Non-overlapping windows


class WindowConfig(BaseModel):
    """Configuration for a walk-forward window."""
    training_bars: int
    testing_bars: int
    step_bars: Optional[int] = None  # For sliding windows
    min_bars: Optional[int] = None   # Minimum required bars


class OptimizationMetric(BaseModel):
    """A metric used for optimization."""
    name: str
    direction: str  # "maximize" or "minimize"
    weight: float = 1.0


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward optimization."""
    method: WalkForwardMethod = WalkForwardMethod.ROLLING
    window_config: WindowConfig
    validation_percentage: Optional[float] = 0.3  # For validation set inside training
    metrics: List[OptimizationMetric]
    parameter_ranges: Dict[str, Any]
    optimization_config: OptimizationConfiguration
    min_trades_per_window: int = 10  # Minimum trades required for a valid window


class WalkForwardWindow(BaseModel):
    """A single window in walk-forward optimization."""
    window_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_idx: int
    train_end_idx: int
    test_end_idx: int
    training_data: Dict[str, Any] = Field(default_factory=dict)  # References to data
    testing_data: Dict[str, Any] = Field(default_factory=dict)   # References to data
    optimized_parameters: Dict[str, Any] = Field(default_factory=dict)
    training_metrics: Dict[str, float] = Field(default_factory=dict)
    testing_metrics: Dict[str, float] = Field(default_factory=dict)
    optimization_result: Optional[Dict[str, Any]] = None


class WalkForwardResult(BaseModel):
    """Results from walk-forward optimization."""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0
    config: WalkForwardConfig
    windows: List[WalkForwardWindow] = Field(default_factory=list)
    aggregated_metrics: Dict[str, float] = Field(default_factory=dict)
    robustness_score: float = 0.0
    optimization_stability: float = 0.0
    is_robust: bool = False
    best_parameters: Dict[str, Any] = Field(default_factory=dict)
    parameter_stability: Dict[str, float] = Field(default_factory=dict)
    plots: Optional[Dict[str, str]] = None  # Base64 encoded plots


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization engine that properly separates in-sample optimization 
    from out-of-sample testing to prevent curve fitting.
    """
    
    def __init__(
        self, 
        data_provider: Callable,
        strategy_evaluator: Callable,
        optimizer_factory: Callable
    ):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            data_provider: Function that provides market data
            strategy_evaluator: Function that evaluates a strategy on data
            optimizer_factory: Function that creates an optimizer for parameter tuning
        """
        self.data_provider = data_provider
        self.evaluate_strategy = strategy_evaluator
        self.create_optimizer = optimizer_factory
        
    def run_optimization(
        self,
        strategy_id: str,
        instruments: List[str],
        start_date: datetime,
        end_date: datetime,
        config: WalkForwardConfig,
        base_parameters: Dict[str, Any] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization for a strategy.
        
        Args:
            strategy_id: ID of the strategy to optimize
            instruments: List of instruments to trade
            start_date: Start date for the entire dataset
            end_date: End date for the entire dataset
            config: Walk-forward optimization configuration
            base_parameters: Base parameters for the strategy
            
        Returns:
            WalkForwardResult: Results of the walk-forward optimization
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting walk-forward optimization for strategy {strategy_id}")
        
        try:
            # Create result object
            result = WalkForwardResult(
                strategy_id=strategy_id,
                config=config
            )
            
            # Load data
            data = self._load_data(instruments, start_date, end_date)
            if not data:
                raise ValueError("Failed to load data for optimization")
                
            # Generate windows
            windows = self._generate_windows(data, config)
            if not windows:
                raise ValueError("Failed to generate valid windows for optimization")
                
            logger.info(f"Generated {len(windows)} windows for walk-forward optimization")
            
            # Optimize each window
            for i, window in enumerate(windows):
                logger.info(f"Processing window {i+1}/{len(windows)}")
                window = self._process_window(
                    window, 
                    strategy_id, 
                    data, 
                    config, 
                    base_parameters
                )
                result.windows.append(window)
                
            # Calculate aggregated metrics
            result.aggregated_metrics = self._calculate_aggregated_metrics(result.windows)
            
            # Calculate robustness score
            result.robustness_score = self._calculate_robustness_score(result.windows)
            
            # Calculate parameter stability
            result.parameter_stability = self._calculate_parameter_stability(result.windows)
            
            # Determine best parameters
            result.best_parameters = self._determine_best_parameters(result.windows)
            
            # Determine if result is robust
            result.is_robust = result.robustness_score >= 0.7  # Threshold for robustness
            
            # Create visualizations
            result.plots = self._create_visualizations(result)
            
            # Set completion time
            end_time = datetime.utcnow()
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            logger.info(f"Completed walk-forward optimization for strategy {strategy_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error during walk-forward optimization: {str(e)}")
            # Create partial result with error information
            end_time = datetime.utcnow()
            return WalkForwardResult(
                strategy_id=strategy_id,
                config=config,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                aggregated_metrics={"error": 1.0},
                robustness_score=0.0,
                is_robust=False
            )
            
    def _load_data(
        self, 
        instruments: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Load market data for optimization."""
        try:
            data = {}
            for instrument in instruments:
                instrument_data = self.data_provider(
                    instrument=instrument, 
                    start_date=start_date, 
                    end_date=end_date
                )
                
                if instrument_data is not None and not instrument_data.empty:
                    data[instrument] = instrument_data
                    
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {}
            
    def _generate_windows(
        self, 
        data: Dict[str, pd.DataFrame],
        config: WalkForwardConfig
    ) -> List[WalkForwardWindow]:
        """Generate walk-forward windows based on configuration."""
        windows = []
        
        # Use first instrument to determine data length
        if not data:
            return windows
            
        first_instrument = next(iter(data.values()))
        data_length = len(first_instrument)
        
        # Get window parameters
        train_size = config.window_config.training_bars
        test_size = config.window_config.testing_bars
        step_size = config.window_config.step_bars or test_size
        
        # Generate windows based on method
        if config.method == WalkForwardMethod.ANCHORED:
            # Anchored: Fixed start point, expanding training window
            start_idx = 0
            while start_idx + train_size + test_size <= data_length:
                train_end_idx = start_idx + train_size
                test_end_idx = train_end_idx + test_size
                
                window = WalkForwardWindow(
                    start_idx=start_idx,
                    train_end_idx=train_end_idx,
                    test_end_idx=test_end_idx
                )
                
                windows.append(window)
                
                # Increase training size rather than moving start
                train_size += step_size
                
        elif config.method == WalkForwardMethod.ROLLING:
            # Rolling: Moving window of fixed size
            start_idx = 0
            while start_idx + train_size + test_size <= data_length:
                train_end_idx = start_idx + train_size
                test_end_idx = train_end_idx + test_size
                
                window = WalkForwardWindow(
                    start_idx=start_idx,
                    train_end_idx=train_end_idx,
                    test_end_idx=test_end_idx
                )
                
                windows.append(window)
                
                # Move start index forward
                start_idx += step_size
                
        elif config.method == WalkForwardMethod.SLIDING:
            # Sliding: Non-overlapping windows
            start_idx = 0
            while start_idx + train_size + test_size <= data_length:
                train_end_idx = start_idx + train_size
                test_end_idx = train_end_idx + test_size
                
                window = WalkForwardWindow(
                    start_idx=start_idx,
                    train_end_idx=train_end_idx,
                    test_end_idx=test_end_idx
                )
                
                windows.append(window)
                
                # Move to next non-overlapping window
                start_idx = test_end_idx
                
        return windows
        
    def _process_window(
        self,
        window: WalkForwardWindow,
        strategy_id: str,
        data: Dict[str, pd.DataFrame],
        config: WalkForwardConfig,
        base_parameters: Dict[str, Any]
    ) -> WalkForwardWindow:
        """Process a single walk-forward window."""
        try:
            # Slice data for this window
            training_data = {}
            testing_data = {}
            
            for instrument, df in data.items():
                training_data[instrument] = df.iloc[window.start_idx:window.train_end_idx]
                testing_data[instrument] = df.iloc[window.train_end_idx:window.test_end_idx]
                
            window.training_data = {
                "start": training_data[next(iter(training_data))].index[0],
                "end": training_data[next(iter(training_data))].index[-1],
                "bars": window.train_end_idx - window.start_idx
            }
            
            window.testing_data = {
                "start": testing_data[next(iter(testing_data))].index[0],
                "end": testing_data[next(iter(testing_data))].index[-1],
                "bars": window.test_end_idx - window.train_end_idx
            }
                
            # Create evaluation function for the optimizer
            def evaluate_parameters(params):
                # Combine base parameters with optimization parameters
                full_params = {**(base_parameters or {}), **params}
                
                # Evaluate on training data
                metrics = self.evaluate_strategy(
                    strategy_id=strategy_id,
                    data=training_data,
                    parameters=full_params
                )
                
                # Extract relevant metrics for optimization
                result = {}
                for metric_config in config.metrics:
                    if metric_config.name in metrics:
                        result[metric_config.name] = metrics[metric_config.name]
                        
                return result
                
            # Optimize parameters using training data
            optimizer = self.create_optimizer(config.optimization_config, evaluate_parameters)
            optimization_result = optimizer.optimize()
            
            # Save optimized parameters
            window.optimized_parameters = optimization_result.best_parameters
            window.optimization_result = optimization_result.dict()
            
            # Combine with base parameters
            full_optimized_params = {**(base_parameters or {}), **window.optimized_parameters}
            
            # Evaluate strategy with optimized parameters on training data
            window.training_metrics = self.evaluate_strategy(
                strategy_id=strategy_id,
                data=training_data,
                parameters=full_optimized_params
            )
            
            # Evaluate strategy with optimized parameters on testing data
            window.testing_metrics = self.evaluate_strategy(
                strategy_id=strategy_id,
                data=testing_data,
                parameters=full_optimized_params
            )
            
            return window
            
        except Exception as e:
            logger.error(f"Error processing window: {str(e)}")
            # Return window with error information
            window.training_metrics = {"error": str(e)}
            window.testing_metrics = {"error": str(e)}
            return window
            
    def _calculate_aggregated_metrics(self, windows: List[WalkForwardWindow]) -> Dict[str, float]:
        """Calculate aggregated metrics across all windows."""
        # Initialize aggregated metrics
        aggregated = {}
        
        # Skip windows with errors
        valid_windows = [w for w in windows if "error" not in w.testing_metrics]
        if not valid_windows:
            return {"error": "No valid windows"}
            
        # Get a list of metrics available in all windows
        first_window = valid_windows[0]
        metric_keys = set(first_window.testing_metrics.keys())
        
        # Calculate aggregated values for each metric
        for metric in metric_keys:
            # Get metric values from all windows
            values = [w.testing_metrics.get(metric, 0.0) for w in valid_windows]
            
            # Calculate statistics
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_median"] = np.median(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_min"] = np.min(values)
            aggregated[f"{metric}_max"] = np.max(values)
            aggregated[f"{metric}_stability"] = 1.0 - (np.std(values) / (abs(np.mean(values)) + 1e-10))
            
        # Calculate key overall metrics
        if "net_profit" in metric_keys:
            total_profit = sum(w.testing_metrics.get("net_profit", 0.0) for w in valid_windows)
            aggregated["total_net_profit"] = total_profit
            
            # Count profitable windows
            profitable_windows = sum(1 for w in valid_windows if w.testing_metrics.get("net_profit", 0.0) > 0)
            aggregated["profitable_window_percentage"] = profitable_windows / len(valid_windows)
            
        if "sharpe_ratio" in metric_keys:
            sharpe_values = [w.testing_metrics.get("sharpe_ratio", 0.0) for w in valid_windows]
            aggregated["mean_sharpe_ratio"] = np.mean(sharpe_values)
            
        # Calculate overall robustness metrics
        training_vs_testing = []
        for window in valid_windows:
            for metric in metric_keys:
                if metric in window.training_metrics and metric in window.testing_metrics:
                    training_val = window.training_metrics[metric]
                    testing_val = window.testing_metrics[metric]
                    
                    # Avoid division by zero
                    if abs(training_val) > 1e-10:
                        ratio = testing_val / training_val
                        training_vs_testing.append(ratio)
        
        if training_vs_testing:
            aggregated["mean_out_of_sample_performance"] = np.mean(training_vs_testing)
            aggregated["median_out_of_sample_performance"] = np.median(training_vs_testing)
            
        return aggregated
            
    def _calculate_robustness_score(self, windows: List[WalkForwardWindow]) -> float:
        """
        Calculate a robustness score that represents how well the strategy 
        performs out-of-sample relative to in-sample.
        """
        # Skip windows with errors
        valid_windows = [w for w in windows if "error" not in w.testing_metrics]
        if not valid_windows:
            return 0.0
            
        # Initialize scores for different aspects of robustness
        consistency_score = 0.0
        performance_score = 0.0
        profitability_score = 0.0
        
        # Calculate consistency score (how consistently the strategy performs across windows)
        profitable_windows = 0
        profit_loss_ratios = []
        
        for window in valid_windows:
            # Count profitable windows
            if window.testing_metrics.get("net_profit", 0.0) > 0:
                profitable_windows += 1
                
            # Calculate profit/loss ratio for this window
            if "avg_profit" in window.testing_metrics and "avg_loss" in window.testing_metrics:
                avg_profit = window.testing_metrics["avg_profit"]
                avg_loss = abs(window.testing_metrics["avg_loss"])
                
                if avg_loss > 0:
                    profit_loss_ratio = avg_profit / avg_loss
                    profit_loss_ratios.append(profit_loss_ratio)
        
        # Consistency based on percentage of profitable windows
        if valid_windows:
            consistency_score = profitable_windows / len(valid_windows)
            
        # Performance score based on out-of-sample vs. in-sample performance
        performance_ratios = []
        
        for window in valid_windows:
            # Compare key metrics between training and testing
            for metric in ["net_profit", "sharpe_ratio", "profit_factor"]:
                if metric in window.training_metrics and metric in window.testing_metrics:
                    training_val = window.training_metrics[metric]
                    testing_val = window.testing_metrics[metric]
                    
                    # Avoid division by zero
                    if abs(training_val) > 1e-10:
                        ratio = testing_val / training_val
                        # Cap ratio at 2.0 (to avoid overweighting cases where testing vastly outperforms training)
                        ratio = min(2.0, max(0.0, ratio))
                        performance_ratios.append(ratio)
        
        if performance_ratios:
            performance_score = np.mean(performance_ratios)
            
        # Profitability score based on average profit/loss ratio
        if profit_loss_ratios:
            avg_profit_loss_ratio = np.mean(profit_loss_ratios)
            # Scale to 0-1 range (assume 2.0 is a good profit/loss ratio)
            profitability_score = min(1.0, avg_profit_loss_ratio / 2.0)
        
        # Combined robustness score (weighted average of the three components)
        weights = {
            "consistency": 0.3,
            "performance": 0.5,
            "profitability": 0.2
        }
        
        robustness_score = (
            weights["consistency"] * consistency_score +
            weights["performance"] * performance_score +
            weights["profitability"] * profitability_score
        )
        
        return robustness_score
            
    def _calculate_parameter_stability(self, windows: List[WalkForwardWindow]) -> Dict[str, float]:
        """
        Calculate parameter stability across windows.
        Returns a dictionary of parameters with their stability scores (0-1).
        """
        # Skip windows with errors or no optimized parameters
        valid_windows = [
            w for w in windows 
            if "error" not in w.testing_metrics and w.optimized_parameters
        ]
        
        if not valid_windows:
            return {}
            
        # Get all parameter names
        all_params = set()
        for window in valid_windows:
            all_params.update(window.optimized_parameters.keys())
            
        # Calculate stability for each parameter
        stability = {}
        
        for param in all_params:
            # Get values for this parameter across all windows
            values = []
            for window in valid_windows:
                if param in window.optimized_parameters:
                    value = window.optimized_parameters[param]
                    # Convert to number if possible
                    if isinstance(value, (int, float)):
                        values.append(value)
                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                        values.append(float(value))
            
            # Calculate stability if we have numerical values
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Calculate coefficient of variation (lower is more stable)
                if abs(mean_val) > 1e-10:
                    cv = std_val / abs(mean_val)
                    # Convert to stability score (0-1, higher is more stable)
                    stability_score = max(0.0, min(1.0, 1.0 - cv))
                    stability[param] = stability_score
                else:
                    stability[param] = 0.0
                    
        return stability
            
    def _determine_best_parameters(self, windows: List[WalkForwardWindow]) -> Dict[str, Any]:
        """Determine the best parameters based on results from all windows."""
        # Skip windows with errors
        valid_windows = [w for w in windows if "error" not in w.testing_metrics]
        if not valid_windows:
            return {}
            
        # Get all parameter names
        all_params = set()
        for window in valid_windows:
            all_params.update(window.optimized_parameters.keys())
            
        # For each parameter, take the median or mode value
        best_params = {}
        
        for param in all_params:
            values = []
            for window in valid_windows:
                if param in window.optimized_parameters:
                    value = window.optimized_parameters[param]
                    values.append(value)
            
            if values:
                # For numerical parameters, take median
                if all(isinstance(v, (int, float)) for v in values):
                    best_params[param] = float(np.median(values))
                    # Convert back to int if all values were integers
                    if all(isinstance(v, int) for v in values):
                        best_params[param] = int(best_params[param])
                else:
                    # For categorical parameters, take mode (most common value)
                    unique_values, counts = np.unique(values, return_counts=True)
                    best_params[param] = unique_values[np.argmax(counts)]
                    
        return best_params
            
    def _create_visualizations(self, result: WalkForwardResult) -> Dict[str, str]:
        """Create visualizations for walk-forward optimization results."""
        plots = {}
        
        try:
            # Skip if no valid windows
            valid_windows = [w for w in result.windows if "error" not in w.testing_metrics]
            if not valid_windows:
                return {}
                
            # 1. Training vs Testing Performance
            plt.figure(figsize=(12, 6))
            
            # Get window indices (1, 2, 3, ...)
            indices = list(range(1, len(valid_windows) + 1))
            
            # Plot training and testing performance for a key metric (e.g., net profit)
            metric_to_plot = self._find_key_metric(valid_windows)
            
            if metric_to_plot:
                training_values = [w.training_metrics.get(metric_to_plot, 0.0) for w in valid_windows]
                testing_values = [w.testing_metrics.get(metric_to_plot, 0.0) for w in valid_windows]
                
                plt.bar(
                    [i - 0.2 for i in indices], 
                    training_values,
                    width=0.4,
                    label='Training',
                    color='blue',
                    alpha=0.7
                )
                
                plt.bar(
                    [i + 0.2 for i in indices], 
                    testing_values,
                    width=0.4,
                    label='Testing',
                    color='green',
                    alpha=0.7
                )
                
                plt.xlabel('Window')
                plt.ylabel(f'{metric_to_plot.replace("_", " ").title()}')
                plt.title(f'Training vs Testing Performance: {metric_to_plot}')
                plt.xticks(indices)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot to base64 string
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plots['training_vs_testing'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
            # 2. Parameter Stability Chart
            plt.figure(figsize=(10, 6))
            
            params = list(result.parameter_stability.keys())
            stabilities = list(result.parameter_stability.values())
            
            if params and stabilities:
                # Sort by stability
                sorted_data = sorted(zip(params, stabilities), key=lambda x: x[1])
                params = [x[0] for x in sorted_data]
                stabilities = [x[1] for x in sorted_data]
                
                plt.barh(params, stabilities, color='teal', alpha=0.7)
                plt.xlabel('Stability Score (0-1)')
                plt.ylabel('Parameter')
                plt.title('Parameter Stability Across Windows')
                plt.xlim(0, 1)
                plt.grid(True, alpha=0.3)
                
                # Save plot to base64 string
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plots['parameter_stability'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
            # 3. Parameter Evolution Chart
            plt.figure(figsize=(12, 8))
            
            # Get top parameters by stability
            top_params = sorted(
                result.parameter_stability.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]  # Top 5 most stable parameters
            
            if top_params:
                for param_name, _ in top_params:
                    # Get values for this parameter across windows
                    param_values = []
                    for window in valid_windows:
                        if param_name in window.optimized_parameters:
                            value = window.optimized_parameters[param_name]
                            if isinstance(value, (int, float)):
                                param_values.append(value)
                            elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                param_values.append(float(value))
                            else:
                                # Skip non-numeric values
                                param_values.append(None)
                                
                    # Plot parameter evolution if we have numeric values
                    if param_values and not all(v is None for v in param_values):
                        plt.plot(
                            indices,
                            [v if v is not None else np.nan for v in param_values],
                            marker='o',
                            label=param_name
                        )
                        
                plt.xlabel('Window')
                plt.ylabel('Parameter Value')
                plt.title('Parameter Evolution Across Windows')
                plt.xticks(indices)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot to base64 string
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plots['parameter_evolution'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
            # 4. Robustness Score Components
            plt.figure(figsize=(8, 6))
            
            # Calculate robustness components
            consistency_score = 0
            performance_score = 0
            profitability_score = 0
            
            # Count profitable windows for consistency
            profitable_windows = sum(1 for w in valid_windows if w.testing_metrics.get("net_profit", 0.0) > 0)
            if valid_windows:
                consistency_score = profitable_windows / len(valid_windows)
                
            # Calculate performance ratio (out-of-sample vs. in-sample)
            performance_ratios = []
            for window in valid_windows:
                for metric in ["net_profit", "sharpe_ratio"]:
                    if metric in window.training_metrics and metric in window.testing_metrics:
                        training_val = window.training_metrics[metric]
                        testing_val = window.testing_metrics[metric]
                        if abs(training_val) > 1e-10:
                            ratio = min(2.0, max(0.0, testing_val / training_val))
                            performance_ratios.append(ratio)
                            
            if performance_ratios:
                performance_score = np.mean(performance_ratios)
                
            # Calculate profitability from average profit/loss ratio
            profit_loss_ratios = []
            for window in valid_windows:
                if "avg_profit" in window.testing_metrics and "avg_loss" in window.testing_metrics:
                    avg_profit = window.testing_metrics["avg_profit"]
                    avg_loss = abs(window.testing_metrics["avg_loss"])
                    if avg_loss > 0:
                        profit_loss_ratios.append(avg_profit / avg_loss)
                        
            if profit_loss_ratios:
                avg_profit_loss_ratio = np.mean(profit_loss_ratios)
                profitability_score = min(1.0, avg_profit_loss_ratio / 2.0)
                
            # Plot robustness components
            components = ['Consistency', 'Performance', 'Profitability', 'Overall']
            scores = [
                consistency_score, 
                performance_score, 
                profitability_score, 
                result.robustness_score
            ]
            
            plt.bar(components, scores, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
            plt.ylabel('Score (0-1)')
            plt.title('Robustness Score Components')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Add score values on top of bars
            for i, score in enumerate(scores):
                plt.text(
                    i, score + 0.02, 
                    f'{score:.2f}', 
                    ha='center'
                )
                
            # Save plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots['robustness_components'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
                
            return plots
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return {}
            
    def _find_key_metric(self, windows: List[WalkForwardWindow]) -> Optional[str]:
        """Find a key metric to plot based on what's available in the data."""
        # Priority order of metrics to look for
        priority_metrics = ["net_profit", "sharpe_ratio", "profit_factor", "win_rate", "total_return"]
        
        for metric in priority_metrics:
            if all(metric in w.training_metrics and metric in w.testing_metrics for w in windows):
                return metric
                
        # If none of the priority metrics are available, use the first common metric
        common_metrics = set(windows[0].testing_metrics.keys())
        for window in windows[1:]:
            common_metrics &= set(window.testing_metrics.keys())
            
        return next(iter(common_metrics)) if common_metrics else None
