"""
Monte Carlo Simulation Module

This module implements Monte Carlo simulation techniques for robustness testing
of trading strategies by generating multiple variations of price data and/or
trade execution sequences to evaluate the distribution of possible outcomes.
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
from scipy import stats
from pydantic import BaseModel, Field
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class SimulationType(str, Enum):
    """Types of Monte Carlo simulations."""
    PRICE_SERIES = 'price_series'
    TRADE_OUTCOMES = 'trade_outcomes'
    PARAMETER_VARIATION = 'parameter_variation'
    EXECUTION_QUALITY = 'execution_quality'
    SLIPPAGE_IMPACT = 'slippage_impact'


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""
    simulation_type: SimulationType
    num_simulations: int = 1000
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    volatility_scale_range: Optional[Tuple[float, float]] = None
    drift_scale_range: Optional[Tuple[float, float]] = None
    bootstrap_method: Optional[str] = None
    block_size: Optional[int] = None
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    parameter_distribution: Optional[str] = None
    fill_rate_range: Optional[Tuple[float, float]] = None
    execution_delay_ms_range: Optional[Tuple[int, int]] = None
    slippage_model: Optional[str] = None
    slippage_range: Optional[Tuple[float, float]] = None


class SimulationResult(BaseModel):
    """Results from a single Monte Carlo simulation run."""
    simulation_id: int
    metrics: Dict[str, float]
    parameters: Optional[Dict[str, Any]] = None
    trades: Optional[List[Dict[str, Any]]] = None
    equity_curve: Optional[List[float]] = None


class MonteCarloAnalysis(BaseModel):
    """Statistical analysis of Monte Carlo simulation results."""
    metric_name: str
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    range_width: float
    percentiles: Dict[int, float]
    histogram_data: Optional[Dict[str, List[float]]] = None


class MonteCarloResults(BaseModel):
    """Complete results from a Monte Carlo simulation."""
    result_id: str = Field(default_factory=lambda : str(uuid.uuid4()))
    strategy_id: str
    config: MonteCarloConfig
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0
    original_metrics: Dict[str, float]
    simulation_results: List[SimulationResult]
    analyses: Dict[str, MonteCarloAnalysis] = Field(default_factory=dict)
    risk_metrics: Dict[str, float] = Field(default_factory=dict)
    plots: Optional[Dict[str, str]] = None


class MonteCarloSimulator:
    """
    Monte Carlo simulator for testing robustness of trading strategies.
    """

    def __init__(self, data_provider: Callable, strategy_evaluator:
        Callable, random_seed: Optional[int]=None):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            data_provider: Function that provides market data
            strategy_evaluator: Function that evaluates a strategy on data
            random_seed: Optional seed for random number generator
        """
        self.data_provider = data_provider
        self.evaluate_strategy = strategy_evaluator
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    @with_exception_handling
    def run_simulation(self, strategy_id: str, instruments: List[str],
        start_date: datetime, end_date: datetime, config: MonteCarloConfig,
        base_parameters: Dict[str, Any]=None) ->MonteCarloResults:
        """
        Run Monte Carlo simulation for a strategy.
        
        Args:
            strategy_id: ID of the strategy to simulate
            instruments: List of instruments to use
            start_date: Start date for data
            end_date: End date for data
            config: Monte Carlo simulation configuration
            base_parameters: Base parameters for the strategy
            
        Returns:
            MonteCarloResults: Results of the Monte Carlo simulation
        """
        start_time = datetime.utcnow()
        logger.info(
            f'Starting Monte Carlo simulation for strategy {strategy_id}')
        try:
            if config.random_seed is not None:
                np.random.seed(config.random_seed)
            results = MonteCarloResults(strategy_id=strategy_id, config=config)
            data = self._load_data(instruments, start_date, end_date)
            if not data:
                raise ValueError('Failed to load data for simulation')
            original_metrics = self.evaluate_strategy(strategy_id=
                strategy_id, data=data, parameters=base_parameters)
            results.original_metrics = original_metrics
            if config.simulation_type == SimulationType.PRICE_SERIES:
                simulation_results = self._run_price_series_simulation(
                    strategy_id, data, config, base_parameters)
            elif config.simulation_type == SimulationType.TRADE_OUTCOMES:
                simulation_results = self._run_trade_outcomes_simulation(
                    strategy_id, data, config, base_parameters)
            elif config.simulation_type == SimulationType.PARAMETER_VARIATION:
                simulation_results = self._run_parameter_variation_simulation(
                    strategy_id, data, config, base_parameters)
            elif config.simulation_type == SimulationType.EXECUTION_QUALITY:
                simulation_results = self._run_execution_quality_simulation(
                    strategy_id, data, config, base_parameters)
            elif config.simulation_type == SimulationType.SLIPPAGE_IMPACT:
                simulation_results = self._run_slippage_simulation(strategy_id,
                    data, config, base_parameters)
            else:
                raise ValueError(
                    f'Unsupported simulation type: {config.simulation_type}')
            results.simulation_results = simulation_results
            results.analyses = self._analyze_results(simulation_results, config
                )
            results.risk_metrics = self._calculate_risk_metrics(
                simulation_results, original_metrics, config)
            results.plots = self._create_visualizations(simulation_results,
                results.analyses, original_metrics, config)
            end_time = datetime.utcnow()
            results.end_time = end_time
            results.duration_seconds = (end_time - start_time).total_seconds()
            logger.info(
                f'Completed Monte Carlo simulation for strategy {strategy_id}')
            return results
        except Exception as e:
            logger.error(f'Error during Monte Carlo simulation: {str(e)}')
            end_time = datetime.utcnow()
            return MonteCarloResults(strategy_id=strategy_id, config=config,
                end_time=end_time, duration_seconds=(end_time - start_time)
                .total_seconds(), original_metrics={'error': str(e)},
                simulation_results=[])

    @with_exception_handling
    def _load_data(self, instruments: List[str], start_date: datetime,
        end_date: datetime) ->Dict[str, pd.DataFrame]:
        """Load market data for simulation."""
        try:
            data = {}
            for instrument in instruments:
                instrument_data = self.data_provider(instrument=instrument,
                    start_date=start_date, end_date=end_date)
                if instrument_data is not None and not instrument_data.empty:
                    data[instrument] = instrument_data
            return data
        except Exception as e:
            logger.error(f'Error loading data: {str(e)}')
            return {}

    @with_exception_handling
    def _run_price_series_simulation(self, strategy_id: str, original_data:
        Dict[str, pd.DataFrame], config: MonteCarloConfig, base_parameters:
        Dict[str, Any]=None) ->List[SimulationResult]:
        """
        Run Monte Carlo simulation by generating alternative price series.
        Uses Geometric Brownian Motion with varying parameters.
        """
        results = []
        num_sims = config.num_simulations
        vol_range = config.volatility_scale_range or (0.8, 1.2)
        drift_range = config.drift_scale_range or (0.9, 1.1)
        logger.info(f'Running {num_sims} price series simulations')
        for sim_id in range(num_sims):
            try:
                simulated_data = {}
                for instrument, original_df in original_data.items():
                    vol_scale = np.random.uniform(vol_range[0], vol_range[1])
                    drift_scale = np.random.uniform(drift_range[0],
                        drift_range[1])
                    simulated_df = self._simulate_price_series(original_df,
                        vol_scale, drift_scale)
                    simulated_data[instrument] = simulated_df
                metrics = self.evaluate_strategy(strategy_id=strategy_id,
                    data=simulated_data, parameters=base_parameters)
                equity_curve = None
                if 'equity_curve' in metrics:
                    equity_curve = metrics.pop('equity_curve')
                trades = None
                if 'trades' in metrics:
                    trades = metrics.pop('trades')
                result = SimulationResult(simulation_id=sim_id, metrics=
                    metrics, equity_curve=equity_curve, trades=trades)
                results.append(result)
            except Exception as e:
                logger.warning(f'Error in simulation {sim_id}: {str(e)}')
                results.append(SimulationResult(simulation_id=sim_id,
                    metrics={'error': str(e)}))
        return results

    def _simulate_price_series(self, original_df: pd.DataFrame,
        volatility_scale: float, drift_scale: float) ->pd.DataFrame:
        """
        Simulate an alternative price series using Geometric Brownian Motion.
        
        Args:
            original_df: Original price data
            volatility_scale: Factor to scale the volatility
            drift_scale: Factor to scale the drift
            
        Returns:
            DataFrame: Simulated price data
        """
        simulated_df = original_df.copy()
        if 'close' not in original_df.columns:
            return original_df
        original_prices = original_df['close'].values
        returns = np.log(original_prices[1:] / original_prices[:-1])
        mu = returns.mean() * drift_scale
        sigma = returns.std() * volatility_scale
        simulated_returns = np.random.normal(mu, sigma, len(returns))
        simulated_prices = np.zeros_like(original_prices)
        simulated_prices[0] = original_prices[0]
        for i in range(1, len(simulated_prices)):
            simulated_prices[i] = simulated_prices[i - 1] * np.exp(
                simulated_returns[i - 1])
        for col in ['open', 'high', 'low', 'close']:
            if col in simulated_df.columns:
                ratio = simulated_prices / original_prices
                simulated_df[col] = original_df[col] * ratio
        return simulated_df

    @with_exception_handling
    def _run_trade_outcomes_simulation(self, strategy_id: str,
        original_data: Dict[str, pd.DataFrame], config: MonteCarloConfig,
        base_parameters: Dict[str, Any]=None) ->List[SimulationResult]:
        """
        Run Monte Carlo simulation by resampling trade outcomes.
        Uses bootstrap resampling of historical trades.
        """
        results = []
        num_sims = config.num_simulations
        bootstrap_method = config.bootstrap_method or 'simple'
        block_size = config.block_size or 5
        logger.info(
            f'Running {num_sims} trade outcome simulations with {bootstrap_method} bootstrap'
            )
        original_metrics = self.evaluate_strategy(strategy_id=strategy_id,
            data=original_data, parameters=base_parameters, return_trades=True)
        original_trades = original_metrics.get('trades', [])
        if not original_trades:
            raise ValueError('No trades found in original strategy evaluation')
        trades_df = pd.DataFrame(original_trades)
        for sim_id in range(num_sims):
            try:
                if bootstrap_method == 'simple':
                    sampled_indices = np.random.choice(len(trades_df), size
                        =len(trades_df), replace=True)
                    resampled_trades = trades_df.iloc[sampled_indices].to_dict(
                        'records')
                elif bootstrap_method == 'block':
                    resampled_trades = self._block_bootstrap(trades_df,
                        block_size)
                else:
                    raise ValueError(
                        f'Unsupported bootstrap method: {bootstrap_method}')
                metrics = self._calculate_metrics_from_trades(resampled_trades)
                equity_curve = self._calculate_equity_curve(resampled_trades)
                result = SimulationResult(simulation_id=sim_id, metrics=
                    metrics, trades=resampled_trades, equity_curve=equity_curve
                    )
                results.append(result)
            except Exception as e:
                logger.warning(f'Error in simulation {sim_id}: {str(e)}')
                results.append(SimulationResult(simulation_id=sim_id,
                    metrics={'error': str(e)}))
        return results

    def _block_bootstrap(self, trades_df: pd.DataFrame, block_size: int
        ) ->List[Dict[str, Any]]:
        """
        Perform block bootstrap resampling of trades.
        
        Args:
            trades_df: DataFrame of original trades
            block_size: Size of blocks to sample
            
        Returns:
            list: Resampled trades
        """
        resampled_trades = []
        n_trades = len(trades_df)
        n_blocks = int(np.ceil(n_trades / block_size))
        for _ in range(n_blocks):
            if n_trades <= block_size:
                start_idx = 0
            else:
                start_idx = np.random.randint(0, n_trades - block_size + 1)
            block = trades_df.iloc[start_idx:start_idx + block_size]
            resampled_trades.extend(block.to_dict('records'))
        return resampled_trades[:n_trades]

    def _calculate_metrics_from_trades(self, trades: List[Dict[str, Any]]
        ) ->Dict[str, float]:
        """
        Calculate key performance metrics from a list of trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            dict: Calculated metrics
        """
        metrics = {}
        if not trades:
            return {'trade_count': 0}
        metrics['trade_count'] = len(trades)
        pnl_list = [trade.get('profit', 0) for trade in trades]
        metrics['net_profit'] = sum(pnl_list)
        metrics['win_count'] = sum(1 for p in pnl_list if p > 0)
        metrics['loss_count'] = sum(1 for p in pnl_list if p <= 0)
        metrics['win_rate'] = metrics['win_count'] / len(pnl_list
            ) if pnl_list else 0
        winning_trades = [p for p in pnl_list if p > 0]
        losing_trades = [p for p in pnl_list if p <= 0]
        metrics['avg_profit'] = sum(winning_trades) / len(winning_trades
            ) if winning_trades else 0
        metrics['avg_loss'] = sum(losing_trades) / len(losing_trades
            ) if losing_trades else 0
        total_profit = sum(winning_trades)
        total_loss = abs(sum(losing_trades))
        metrics['profit_factor'
            ] = total_profit / total_loss if total_loss > 0 else float('inf')
        equity_curve = self._calculate_equity_curve(trades)
        metrics['max_drawdown'] = self._calculate_max_drawdown(equity_curve)
        return metrics

    def _calculate_equity_curve(self, trades: List[Dict[str, Any]]) ->List[
        float]:
        """
        Calculate equity curve from a list of trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            list: Equity curve values
        """
        equity = [0]
        sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', 0))
        for trade in sorted_trades:
            profit = trade.get('profit', 0)
            equity.append(equity[-1] + profit)
        return equity

    def _calculate_max_drawdown(self, equity_curve: List[float]) ->float:
        """
        Calculate maximum drawdown from an equity curve.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            float: Maximum drawdown as a percentage
        """
        max_dd = 0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        return max_dd

    @with_exception_handling
    def _run_parameter_variation_simulation(self, strategy_id: str,
        original_data: Dict[str, pd.DataFrame], config: MonteCarloConfig,
        base_parameters: Dict[str, Any]=None) ->List[SimulationResult]:
        """
        Run Monte Carlo simulation by varying strategy parameters.
        """
        results = []
        num_sims = config.num_simulations
        parameter_ranges = config.parameter_ranges or {}
        distribution = config.parameter_distribution or 'uniform'
        if not parameter_ranges:
            raise ValueError(
                'No parameter ranges specified for parameter variation simulation'
                )
        logger.info(f'Running {num_sims} parameter variation simulations')
        for sim_id in range(num_sims):
            try:
                parameters = self._generate_parameters(base_parameters or {
                    }, parameter_ranges, distribution)
                metrics = self.evaluate_strategy(strategy_id=strategy_id,
                    data=original_data, parameters=parameters)
                equity_curve = None
                if 'equity_curve' in metrics:
                    equity_curve = metrics.pop('equity_curve')
                trades = None
                if 'trades' in metrics:
                    trades = metrics.pop('trades')
                result = SimulationResult(simulation_id=sim_id, metrics=
                    metrics, parameters=parameters, equity_curve=
                    equity_curve, trades=trades)
                results.append(result)
            except Exception as e:
                logger.warning(f'Error in simulation {sim_id}: {str(e)}')
                results.append(SimulationResult(simulation_id=sim_id,
                    metrics={'error': str(e)}))
        return results

    def _generate_parameters(self, base_params: Dict[str, Any],
        param_ranges: Dict[str, Tuple[float, float]], distribution: str
        ) ->Dict[str, Any]:
        """
        Generate a set of parameters by varying from base parameters.
        
        Args:
            base_params: Base parameter values
            param_ranges: Ranges for parameters to vary (min, max)
            distribution: Distribution to use ("uniform" or "normal")
            
        Returns:
            dict: Generated parameters
        """
        params = base_params.copy()
        for param_name, (min_val, max_val) in param_ranges.items():
            if distribution == 'uniform':
                params[param_name] = np.random.uniform(min_val, max_val)
            elif distribution == 'normal':
                mean = (min_val + max_val) / 2
                std = (max_val - min_val) / 6
                value = np.random.normal(mean, std)
                params[param_name] = np.clip(value, min_val, max_val)
            else:
                raise ValueError(f'Unsupported distribution: {distribution}')
            if param_name in base_params and isinstance(base_params[
                param_name], int):
                params[param_name] = int(round(params[param_name]))
        return params

    def _run_execution_quality_simulation(self, strategy_id: str,
        original_data: Dict[str, pd.DataFrame], config: MonteCarloConfig,
        base_parameters: Dict[str, Any]=None) ->List[SimulationResult]:
        """
        Run Monte Carlo simulation by varying execution quality parameters.
        """
        logger.warning(
            'Execution quality simulation using parameter variation as approximation'
            )
        param_ranges = {}
        if config.fill_rate_range:
            param_ranges['fill_rate'] = config.fill_rate_range
        if config.execution_delay_ms_range:
            param_ranges['execution_delay_ms'
                ] = config.execution_delay_ms_range
        if not param_ranges:
            param_ranges = {'execution_quality': (0.8, 1.0), 'delay_factor':
                (1.0, 2.0)}
        config_copy = config.copy()
        config_copy.parameter_ranges = param_ranges
        config_copy.parameter_distribution = 'uniform'
        return self._run_parameter_variation_simulation(strategy_id,
            original_data, config_copy, base_parameters)

    def _run_slippage_simulation(self, strategy_id: str, original_data:
        Dict[str, pd.DataFrame], config: MonteCarloConfig, base_parameters:
        Dict[str, Any]=None) ->List[SimulationResult]:
        """
        Run Monte Carlo simulation with varying slippage models.
        """
        logger.warning(
            'Slippage simulation using parameter variation as approximation')
        param_ranges = {}
        if config.slippage_range:
            if config.slippage_model == 'fixed':
                param_ranges['slippage_pips'] = config.slippage_range
            elif config.slippage_model == 'percentage':
                param_ranges['slippage_percent'] = config.slippage_range
            else:
                param_ranges['slippage_factor'] = config.slippage_range
        if not param_ranges:
            param_ranges = {'slippage_pips': (1.0, 5.0)}
        config_copy = config.copy()
        config_copy.parameter_ranges = param_ranges
        config_copy.parameter_distribution = 'uniform'
        return self._run_parameter_variation_simulation(strategy_id,
            original_data, config_copy, base_parameters)

    def _analyze_results(self, simulation_results: List[SimulationResult],
        config: MonteCarloConfig) ->Dict[str, MonteCarloAnalysis]:
        """
        Analyze the results of Monte Carlo simulations to produce statistical insights.
        
        Args:
            simulation_results: List of simulation results
            config: Monte Carlo configuration
            
        Returns:
            dict: Analysis results by metric name
        """
        analyses = {}
        metric_counts = {}
        for result in simulation_results:
            for metric in result.metrics:
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
        threshold = len(simulation_results) * 0.5
        for metric, count in metric_counts.items():
            if count >= threshold:
                values = [result.metrics[metric] for result in
                    simulation_results if metric in result.metrics and
                    isinstance(result.metrics[metric], (int, float))]
                if values:
                    percentiles = {(1): float(np.percentile(values, 1)), (5
                        ): float(np.percentile(values, 5)), (10): float(np.
                        percentile(values, 10)), (25): float(np.percentile(
                        values, 25)), (50): float(np.percentile(values, 50)
                        ), (75): float(np.percentile(values, 75)), (90):
                        float(np.percentile(values, 90)), (95): float(np.
                        percentile(values, 95)), (99): float(np.percentile(
                        values, 99))}
                    hist, bin_edges = np.histogram(values, bins=20)
                    histogram_data = {'counts': hist.tolist(), 'bin_edges':
                        bin_edges.tolist(), 'bin_centers': [((bin_edges[i] +
                        bin_edges[i + 1]) / 2) for i in range(len(bin_edges
                        ) - 1)]}
                    analysis = MonteCarloAnalysis(metric_name=metric, mean=
                        float(np.mean(values)), median=float(np.median(
                        values)), std_dev=float(np.std(values)), min_value=
                        float(np.min(values)), max_value=float(np.max(
                        values)), range_width=float(np.max(values) - np.min
                        (values)), percentiles=percentiles, histogram_data=
                        histogram_data)
                    analyses[metric] = analysis
        return analyses

    def _calculate_risk_metrics(self, simulation_results: List[
        SimulationResult], original_metrics: Dict[str, float], config:
        MonteCarloConfig) ->Dict[str, float]:
        """
        Calculate risk metrics based on Monte Carlo simulation results.
        
        Args:
            simulation_results: List of simulation results
            original_metrics: Original strategy metrics
            config: Monte Carlo configuration
            
        Returns:
            dict: Risk metrics
        """
        risk_metrics = {}
        for metric_name in ['net_profit', 'sharpe_ratio', 'max_drawdown']:
            if metric_name not in original_metrics:
                continue
            values = [result.metrics[metric_name] for result in
                simulation_results if metric_name in result.metrics and
                isinstance(result.metrics[metric_name], (int, float))]
            if not values:
                continue
            values_array = np.array(values)
            if metric_name == 'net_profit':
                risk_metrics['probability_of_profit'] = np.mean(
                    values_array > 0)
                confidence_level = config.confidence_level
                var_percentile = 100 * (1 - confidence_level)
                var = np.percentile(values_array, var_percentile)
                risk_metrics[f'var_{int(confidence_level * 100)}'] = float(var)
                cvar_values = values_array[values_array <= var]
                if len(cvar_values) > 0:
                    cvar = float(np.mean(cvar_values))
                    risk_metrics[f'cvar_{int(confidence_level * 100)}'] = cvar
            risk_metrics[f'probability_of_improvement_{metric_name}'
                ] = np.mean(values_array > original_metrics[metric_name])
            if metric_name != 'max_drawdown':
                threshold = original_metrics[metric_name] * 0.8
                risk_metrics[
                    f'probability_of_significant_deterioration_{metric_name}'
                    ] = np.mean(values_array < threshold)
            else:
                threshold = original_metrics[metric_name] * 1.2
                risk_metrics[
                    f'probability_of_significant_deterioration_{metric_name}'
                    ] = np.mean(values_array > threshold)
            if np.std(values_array) > 0:
                risk_metrics[f'robustness_ratio_{metric_name}'] = abs(np.
                    mean(values_array) / np.std(values_array))
        key_ratios = [risk_metrics.get(f'robustness_ratio_{m}', 0) for m in
            ['net_profit', 'sharpe_ratio']]
        if key_ratios:
            normalized_ratios = [min(1.0, r / 2.0) for r in key_ratios if r > 0
                ]
            if normalized_ratios:
                risk_metrics['overall_robustness_score'] = float(np.mean(
                    normalized_ratios))
        return risk_metrics

    @with_exception_handling
    def _create_visualizations(self, simulation_results: List[
        SimulationResult], analyses: Dict[str, MonteCarloAnalysis],
        original_metrics: Dict[str, float], config: MonteCarloConfig) ->Dict[
        str, str]:
        """
        Create visualizations of Monte Carlo simulation results.
        
        Args:
            simulation_results: List of simulation results
            analyses: Analysis results by metric
            original_metrics: Original strategy metrics
            config: Monte Carlo configuration
            
        Returns:
            dict: Base64 encoded plot images
        """
        plots = {}
        try:
            for metric_name in ['net_profit', 'sharpe_ratio', 'max_drawdown']:
                if metric_name not in analyses:
                    continue
                analysis = analyses[metric_name]
                plt.figure(figsize=(10, 6))
                hist_data = analysis.histogram_data
                if hist_data:
                    plt.bar(hist_data['bin_centers'], hist_data['counts'],
                        width=hist_data['bin_edges'][1] - hist_data[
                        'bin_edges'][0], alpha=0.7)
                    if metric_name in original_metrics:
                        plt.axvline(x=original_metrics[metric_name], color=
                            'red', linestyle='--', label=
                            f'Original: {original_metrics[metric_name]:.2f}')
                    lower_ci = analysis.percentiles[5]
                    upper_ci = analysis.percentiles[95]
                    plt.axvline(x=lower_ci, color='green', linestyle='--',
                        label=f'5th %: {lower_ci:.2f}')
                    plt.axvline(x=upper_ci, color='green', linestyle='--',
                        label=f'95th %: {upper_ci:.2f}')
                    plt.title(
                        f"Monte Carlo Distribution: {metric_name.replace('_', ' ').title()}"
                        )
                    plt.xlabel(metric_name.replace('_', ' ').title())
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plots[f'distribution_{metric_name}'] = base64.b64encode(buf
                        .read()).decode('utf-8')
                    plt.close()
            equity_curve_data = []
            for result in simulation_results:
                if result.equity_curve:
                    equity_curve_data.append(result.equity_curve)
            if equity_curve_data:
                lengths = [len(ec) for ec in equity_curve_data]
                median_length = int(np.median(lengths))
                filtered_curves = [ec for ec in equity_curve_data if len(ec
                    ) == median_length]
                if filtered_curves:
                    curves_array = np.array(filtered_curves)
                    percentiles = {(5): np.percentile(curves_array, 5, axis
                        =0), (25): np.percentile(curves_array, 25, axis=0),
                        (50): np.percentile(curves_array, 50, axis=0), (75):
                        np.percentile(curves_array, 75, axis=0), (95): np.
                        percentile(curves_array, 95, axis=0)}
                    plt.figure(figsize=(12, 6))
                    x = range(median_length)
                    plt.fill_between(x, percentiles[5], percentiles[95],
                        alpha=0.2, color='blue', label='5-95%')
                    plt.fill_between(x, percentiles[25], percentiles[75],
                        alpha=0.3, color='blue', label='25-75%')
                    plt.plot(x, percentiles[50], color='blue', label='Median')
                    if 'equity_curve' in original_metrics:
                        original_curve = original_metrics['equity_curve']
                        if len(original_curve) >= median_length:
                            plt.plot(x, original_curve[:median_length],
                                color='red', linestyle='--', label='Original')
                    plt.title('Monte Carlo Equity Curve Distribution')
                    plt.xlabel('Trade Number')
                    plt.ylabel('Equity')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plots['equity_curve_fan'] = base64.b64encode(buf.read()
                        ).decode('utf-8')
                    plt.close()
            for metric_name, analysis in analyses.items():
                if metric_name in ['net_profit', 'sharpe_ratio']:
                    plt.figure(figsize=(10, 6))
                    thresholds = []
                    percentages = []
                    threshold_factors = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
                    if metric_name in original_metrics:
                        original_value = original_metrics[metric_name]
                        for factor in threshold_factors:
                            threshold = original_value * factor
                            thresholds.append(f'{int(factor * 100)}%')
                            values = [result.metrics[metric_name] for
                                result in simulation_results if metric_name in
                                result.metrics and isinstance(result.
                                metrics[metric_name], (int, float))]
                            if values:
                                percentage = sum(1 for v in values if v >=
                                    threshold) / len(values)
                                percentages.append(percentage * 100)
                        plt.bar(thresholds, percentages, alpha=0.7)
                        plt.title(
                            f"Probability of Exceeding Percentage of Original {metric_name.replace('_', ' ').title()}"
                            )
                        plt.xlabel(
                            f"Percentage of Original {metric_name.replace('_', ' ').title()}"
                            )
                        plt.ylabel('Probability (%)')
                        plt.ylim(0, 100)
                        plt.grid(True, alpha=0.3)
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        plots[f'probability_{metric_name}'] = base64.b64encode(
                            buf.read()).decode('utf-8')
                        plt.close()
            return plots
        except Exception as e:
            logger.error(f'Error creating visualizations: {str(e)}')
            return {}
