"""
Backtesting-Optimization Integration Module

This module provides integration between the Backtesting System and Auto-Optimization Framework,
enabling seamless workflows for strategy optimization and performance analysis.
"""
import os
import json
import logging
import importlib
import uuid
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from core_foundations.utils.logger import get_logger


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def get_backtest_engine():
    """Dynamically import and return the BacktestEngine class"""
    from strategy_execution_engine.backtesting.backtest_engine import BacktestEngine
    return BacktestEngine


def get_auto_optimizer():
    """Return the ModelOptimizationServiceAdapter class"""
    from strategy_execution_engine.adapters.ml_workbench_adapter import ModelOptimizationServiceAdapter
    return ModelOptimizationServiceAdapter


logger = get_logger(__name__)


class BacktestOptimizationIntegrator:
    """
    Integration class for connecting Backtesting System with Auto-Optimization Framework

    This class enables seamless workflows between backtesting and optimization components,
    allowing for automated strategy parameter tuning and performance analysis.
    """

    def __init__(self, data: Optional[pd.DataFrame]=None, output_dir:
        Optional[str]=None, max_workers: int=4, optimization_id: Optional[
        str]=None):
        """
        Initialize the integrator

        Args:
            data: Historical price data as pandas DataFrame
            output_dir: Directory to save optimization results
            max_workers: Maximum number of parallel workers for optimization
            optimization_id: Optional ID for the optimization process
        """
        self.data = data
        self.optimization_id = (optimization_id or
            f"optim_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        self.max_workers = max_workers
        self.logger = logging.getLogger(
            f'backtest_optimization.{self.optimization_id}')
        self.output_dir = output_dir or os.path.join('output',
            'optimizations', self.optimization_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_optimization_adapter = get_auto_optimizer()
        self.backtest_engine_class = get_backtest_engine()
        self._optimizer = None
        self._backtest_engines = {}

    @property
    def optimizer(self):
        """Lazy initialization of the optimizer"""
        if self._optimizer is None:
            self._optimizer = self.model_optimization_adapter(config={
                'output_dir': self.output_dir, 'optimization_id': self.
                optimization_id})
        return self._optimizer

    @with_exception_handling
    def optimize_strategy(self, strategy_func: Callable, parameter_space:
        Dict[str, Any], optimization_algorithm: str='bayesian',
        objective_func: Optional[Callable]=None, target_metric: str=
        'sharpe_ratio', objective: str='maximize', constraints: Optional[
        Dict[str, Any]]=None, max_evaluations: int=50, base_parameters:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Optimize strategy parameters using the Auto-Optimization Framework

        Args:
            strategy_func: Trading strategy function to optimize
            parameter_space: Dictionary defining parameter search space
            optimization_algorithm: Optimization algorithm to use
            objective_func: Custom objective function, or None to use target_metric
            target_metric: Metric to optimize when using default objective
            objective: 'maximize' or 'minimize'
            constraints: Optional constraints for the optimization
            max_evaluations: Maximum number of evaluations to perform
            base_parameters: Fixed parameters to pass to the strategy

        Returns:
            Dictionary with optimization results
        """
        self.logger.info(
            f'Starting strategy optimization with {optimization_algorithm} algorithm'
            )
        self.logger.info(
            f'Parameter space: {json.dumps(parameter_space, default=str)}')
        base_params = base_parameters or {}

        @with_exception_handling
        def evaluate_parameters(params: Dict[str, Any]) ->Dict[str, float]:
            """Evaluate a set of parameters by running a backtest"""
            backtest_id = (
                f'optim_{self.optimization_id}_{uuid.uuid4().hex[:8]}')
            engine = self.backtest_engine_class(data=self.data, backtest_id
                =backtest_id, track_tool_effectiveness=False)
            self._backtest_engines[backtest_id] = engine
            run_params = {**base_params, **params}
            try:
                results = engine.run_strategy(strategy_func=strategy_func,
                    **run_params)
                if objective_func is not None:
                    score = objective_func(engine, results)
                    metrics = {'custom_objective': score}
                else:
                    if target_metric in engine.metrics:
                        score = engine.metrics[target_metric]
                    else:
                        self.logger.warning(
                            f'Target metric {target_metric} not found in results'
                            )
                        score = float('-inf'
                            ) if objective == 'maximize' else float('inf')
                    metrics = engine.metrics.copy()
                self.logger.info(
                    f'Evaluated parameters: {json.dumps(params, default=str)}, {target_metric}: {score}'
                    )
                return metrics
            except Exception as e:
                self.logger.error(f'Error evaluating parameters: {str(e)}')
                return {target_metric: float('-inf') if objective ==
                    'maximize' else float('inf')}
        optimization_results = self.optimizer.optimize(evaluate_func=
            evaluate_parameters, parameter_space=parameter_space, algorithm
            =optimization_algorithm, objective=objective, target_metric=
            target_metric, constraints=constraints, max_evaluations=
            max_evaluations)
        best_params = optimization_results.get('best_parameters', {})
        best_backtest_id = optimization_results.get('best_evaluation_id')
        if best_backtest_id and best_backtest_id in self._backtest_engines:
            best_engine = self._backtest_engines[best_backtest_id]
            report_data = best_engine.generate_performance_report()
            dashboard_path = best_engine.create_interactive_dashboard(
                report_data)
            pdf_path = best_engine.export_pdf_report(report_data)
            excel_path = best_engine.export_excel_report(report_data)
            optimization_results['report_paths'] = {'dashboard':
                dashboard_path, 'pdf_report': pdf_path, 'excel_report':
                excel_path}
            results_path = os.path.join(self.output_dir,
                'optimization_results.json')
            with open(results_path, 'w') as f:
                json.dump(optimization_results, f, default=str, indent=2)
            self.logger.info(
                f"Optimization completed. Best {target_metric}: {optimization_results.get('best_score')}"
                )
            self.logger.info(
                f'Best parameters: {json.dumps(best_params, default=str)}')
            self.logger.info(f'Full results saved to: {results_path}')
        return optimization_results

    def run_walk_forward_optimization(self, strategy_func: Callable,
        parameter_space: Dict[str, Any], train_periods: List[Tuple[datetime,
        datetime]], test_periods: List[Tuple[datetime, datetime]],
        optimization_algorithm: str='bayesian', target_metric: str=
        'sharpe_ratio', objective: str='maximize', base_parameters:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Run walk-forward optimization using the Backtesting System and Auto-Optimization Framework

        Args:
            strategy_func: Trading strategy function to optimize
            parameter_space: Dictionary defining parameter search space
            train_periods: List of (start_date, end_date) tuples for training
            test_periods: List of (start_date, end_date) tuples for testing
            optimization_algorithm: Optimization algorithm to use
            target_metric: Metric to optimize
            objective: 'maximize' or 'minimize'
            base_parameters: Fixed parameters to pass to the strategy

        Returns:
            Dictionary with walk-forward optimization results
        """
        if len(train_periods) != len(test_periods):
            raise ValueError(
                'Number of train periods must match number of test periods')
        if self.data is None:
            raise ValueError(
                'Data must be provided for walk-forward optimization')
        self.logger.info(
            f'Starting walk-forward optimization with {len(train_periods)} periods'
            )
        period_results = []
        aggregated_test_results = []
        base_params = base_parameters or {}
        for i, ((train_start, train_end), (test_start, test_end)) in enumerate(
            zip(train_periods, test_periods)):
            self.logger.info(
                f'Period {i + 1}/{len(train_periods)}: Training {train_start} to {train_end}, Testing {test_start} to {test_end}'
                )
            train_mask = (self.data.index >= train_start) & (self.data.
                index <= train_end)
            train_data = self.data[train_mask].copy()
            period_optimization_id = f'{self.optimization_id}_period_{i + 1}'
            period_output_dir = os.path.join(self.output_dir, f'period_{i + 1}'
                )
            os.makedirs(period_output_dir, exist_ok=True)
            period_integrator = BacktestOptimizationIntegrator(data=
                train_data, output_dir=period_output_dir, optimization_id=
                period_optimization_id)
            optimization_results = period_integrator.optimize_strategy(
                strategy_func=strategy_func, parameter_space=
                parameter_space, optimization_algorithm=
                optimization_algorithm, target_metric=target_metric,
                objective=objective, base_parameters=base_params,
                max_evaluations=30)
            best_params = optimization_results.get('best_parameters', {})
            test_mask = (self.data.index >= test_start) & (self.data.index <=
                test_end)
            test_data = self.data[test_mask].copy()
            test_backtest_id = f'{period_optimization_id}_test'
            test_engine = self.backtest_engine_class(data=test_data,
                backtest_id=test_backtest_id)
            test_params = {**base_params, **best_params}
            test_results = test_engine.run_strategy(strategy_func=
                strategy_func, **test_params)
            period_results.append({'period': i + 1, 'train_start':
                train_start, 'train_end': train_end, 'test_start':
                test_start, 'test_end': test_end, 'best_parameters':
                best_params, 'train_score': optimization_results.get(
                'best_score'), 'test_score': test_engine.metrics.get(
                target_metric), 'test_metrics': test_engine.metrics})
            aggregated_test_results.append(test_engine.metrics)
            test_report_data = test_engine.generate_performance_report()
            test_dashboard_path = test_engine.create_interactive_dashboard(
                test_report_data)
            self.logger.info(
                f"Period {i + 1} complete. Train score: {optimization_results.get('best_score')}, Test score: {test_engine.metrics.get(target_metric)}"
                )
        overall_metrics = self._aggregate_metrics(aggregated_test_results)
        walk_forward_results = {'optimization_id': self.optimization_id,
            'period_results': period_results, 'overall_metrics':
            overall_metrics, 'parameter_stability': self.
            _analyze_parameter_stability([p['best_parameters'] for p in
            period_results])}
        results_path = os.path.join(self.output_dir,
            'walk_forward_results.json')
        with open(results_path, 'w') as f:
            json.dump(walk_forward_results, f, default=str, indent=2)
        self.logger.info(
            f'Walk-forward optimization complete. Overall {target_metric}: {overall_metrics.get(target_metric)}'
            )
        self.logger.info(f'Full results saved to: {results_path}')
        return walk_forward_results

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) ->Dict[
        str, float]:
        """
        Aggregate metrics across multiple test periods

        Args:
            metrics_list: List of metric dictionaries from test periods

        Returns:
            Dictionary with aggregated metrics
        """
        if not metrics_list:
            return {}
        aggregated = {}
        common_metrics = set.intersection(*[set(m.keys()) for m in
            metrics_list])
        for metric in common_metrics:
            values = [m.get(metric) for m in metrics_list]
            if not all(isinstance(v, (int, float)) for v in values):
                continue
            aggregated[metric] = np.mean(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
            aggregated[f'{metric}_std'] = np.std(values)
        return aggregated

    def _analyze_parameter_stability(self, parameter_sets: List[Dict[str, Any]]
        ) ->Dict[str, Dict[str, float]]:
        """
        Analyze stability of optimized parameters across periods

        Args:
            parameter_sets: List of parameter dictionaries from different periods

        Returns:
            Dictionary with parameter stability metrics
        """
        if not parameter_sets:
            return {}
        common_params = set.intersection(*[set(p.keys()) for p in
            parameter_sets])
        stability = {}
        for param in common_params:
            values = []
            for params in parameter_sets:
                value = params.get(param)
                if not isinstance(value, (int, float)):
                    continue
                values.append(value)
            if values:
                mean = np.mean(values)
                std = np.std(values)
                cv = std / abs(mean) if mean != 0 else float('inf')
                stability[param] = {'mean': mean, 'std': std, 'min': np.min
                    (values), 'max': np.max(values), 'stability_score': 1.0 -
                    min(cv, 1.0)}
        return stability
