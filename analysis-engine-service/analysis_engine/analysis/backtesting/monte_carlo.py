"""
Monte Carlo Simulation for Backtesting.

This module provides functionality for robust Monte Carlo simulations of trading strategies
to evaluate statistical reliability of backtest results and estimate the range of possible outcomes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
from scipy import stats
import datetime as dt

from analysis_engine.analysis.backtesting.core import BacktestResults, TradeStats

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations."""
    
    # Number of simulations to run
    num_simulations: int = 1000
    
    # Type of resampling to use
    resampling_method: str = "bootstrap"  # Options: bootstrap, block_bootstrap, random_order
    
    # Block size for block bootstrap (in days)
    block_size: int = 20
    
    # Confidence level for statistical measurements (e.g. 0.95 for 95% confidence)
    confidence_level: float = 0.95
    
    # Initial capital for simulations
    initial_capital: float = 100000.0
    
    # Whether to adjust for autocorrelation in returns
    adjust_autocorrelation: bool = True
    
    # Risk-free rate (annual) for risk metrics calculation
    risk_free_rate: float = 0.02
    
    # Whether to use antithetic sampling (variance reduction technique)
    use_antithetic_sampling: bool = True
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulations."""
    
    # Raw simulation equity curves (one array per simulation)
    equity_curves: np.ndarray
    
    # Final equity distribution
    final_equity_distribution: np.ndarray
    
    # Maximum drawdown distribution
    max_drawdown_distribution: np.ndarray
    
    # Sharpe ratio distribution
    sharpe_ratio_distribution: np.ndarray
    
    # Trade win rate distribution
    win_rate_distribution: np.ndarray
    
    # Statistical summary
    statistics: Dict[str, Dict[str, float]]
    
    # Confidence intervals for key metrics
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Probability of profit
    profit_probability: float
    
    # Probability of ruin (reach zero capital)
    ruin_probability: float
    
    # Original backtest results for reference
    original_results: BacktestResults
    
    # Configuration used for this simulation
    config: MonteCarloConfig
    
    def get_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) at given confidence level."""
        return np.percentile(self.final_equity_distribution, (1 - confidence) * 100)
    
    def get_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR) at given confidence level."""
        var = self.get_var(confidence)
        return self.final_equity_distribution[self.final_equity_distribution <= var].mean()


class MonteCarloSimulator:
    """Monte Carlo simulator for backtesting results."""
    
    def __init__(self, config: MonteCarloConfig = None):
        """Initialize the Monte Carlo simulator.
        
        Args:
            config: Configuration for Monte Carlo simulations. If None, default config is used.
        """
        self.config = config or MonteCarloConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def run_simulation(self, backtest_results: BacktestResults) -> MonteCarloResults:
        """Run Monte Carlo simulations based on backtest results.
        
        Args:
            backtest_results: Original backtest results to use as base for simulations
            
        Returns:
            Results of Monte Carlo simulations
        """
        logger.info(f"Running {self.config.num_simulations} Monte Carlo simulations")
        
        # Extract trade data and returns from backtest results
        trade_returns = self._extract_trade_returns(backtest_results)
        
        # Run simulations
        simulated_returns = self._generate_simulations(trade_returns)
        
        # Calculate equity curves and statistics
        equity_curves = self._calculate_equity_curves(simulated_returns)
        
        # Calculate performance metrics for each simulation
        metrics = self._calculate_simulation_metrics(equity_curves, simulated_returns)
        
        # Compute confidence intervals and statistics
        ci_stats = self._compute_statistics(metrics)
        
        # Calculate probabilities
        profit_prob = np.mean(metrics["final_equity"] > self.config.initial_capital)
        ruin_prob = np.mean(np.min(equity_curves, axis=1) <= 0)
        
        # Create results object
        results = MonteCarloResults(
            equity_curves=equity_curves,
            final_equity_distribution=metrics["final_equity"],
            max_drawdown_distribution=metrics["max_drawdown"],
            sharpe_ratio_distribution=metrics["sharpe_ratio"],
            win_rate_distribution=metrics["win_rate"],
            statistics=ci_stats["statistics"],
            confidence_intervals=ci_stats["confidence_intervals"],
            profit_probability=profit_prob,
            ruin_probability=ruin_prob,
            original_results=backtest_results,
            config=self.config
        )
        
        return results
    
    def _extract_trade_returns(self, backtest_results: BacktestResults) -> np.ndarray:
        """Extract returns from individual trades in the backtest results.
        
        Args:
            backtest_results: Original backtest results
            
        Returns:
            Array of trade returns (profit/loss as percentage of account at trade time)
        """
        trades = backtest_results.trades
        
        # If no trades, raise an error
        if len(trades) == 0:
            raise ValueError("Cannot run Monte Carlo simulation with no trades")
        
        # Extract returns as percentage of account at the time of the trade
        returns = np.array([t.profit_loss_pct for t in trades])
        
        # Adjust for autocorrelation if configured
        if self.config.adjust_autocorrelation:
            returns = self._adjust_autocorrelation(returns)
        
        return returns
    
    def _adjust_autocorrelation(self, returns: np.ndarray) -> np.ndarray:
        """Adjust returns for autocorrelation.
        
        Args:
            returns: Original trade returns
        
        Returns:
            Adjusted returns with similar autocorrelation properties
        """
        # Calculate autocorrelation
        acf = np.correlate(returns, returns, mode='full')[len(returns)-1:] / np.var(returns) / len(returns)
        
        # If significant autocorrelation is detected, adjust returns
        if np.abs(acf[1]) > 1.96 / np.sqrt(len(returns)):  # 95% confidence
            logger.info(f"Detected significant autocorrelation: {acf[1]:.4f}. Adjusting simulation method.")
            if self.config.resampling_method == "bootstrap":
                # Switch to block bootstrap
                self.config.resampling_method = "block_bootstrap"
                
        return returns
    
    def _generate_simulations(self, trade_returns: np.ndarray) -> List[np.ndarray]:
        """Generate simulated trade returns using the configured resampling method.
        
        Args:
            trade_returns: Original trade returns
            
        Returns:
            List of simulated trade return sequences
        """
        num_trades = len(trade_returns)
        simulated_returns = []
        
        for i in range(self.config.num_simulations):
            if self.config.resampling_method == "bootstrap":
                # Simple bootstrap: sample with replacement
                indices = np.random.choice(num_trades, size=num_trades, replace=True)
                sampled_returns = trade_returns[indices]
                
            elif self.config.resampling_method == "block_bootstrap":
                # Block bootstrap: sample blocks to preserve autocorrelation
                sampled_returns = self._block_bootstrap(trade_returns)
                
            elif self.config.resampling_method == "random_order":
                # Random order: shuffle the returns
                indices = np.random.permutation(num_trades)
                sampled_returns = trade_returns[indices]
                
            else:
                raise ValueError(f"Unknown resampling method: {self.config.resampling_method}")
            
            # Apply antithetic sampling if configured (variance reduction technique)
            if self.config.use_antithetic_sampling and i % 2 == 1:
                # For odd-numbered simulations, use the negative of the previous simulation
                # This reduces variance in the Monte Carlo estimates
                sampled_returns = -simulated_returns[-1]
            
            simulated_returns.append(sampled_returns)
        
        return simulated_returns
    
    def _block_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Perform block bootstrap resampling.
        
        Args:
            returns: Original trade returns
            
        Returns:
            Block bootstrapped returns
        """
        num_trades = len(returns)
        block_size = min(self.config.block_size, num_trades // 3)
        
        # Number of blocks needed to cover the original number of trades
        num_blocks = int(np.ceil(num_trades / block_size))
        
        # Sample block starting indices
        start_indices = np.random.choice(num_trades - block_size + 1, size=num_blocks, replace=True)
        
        # Construct the resampled returns from blocks
        resampled = []
        for start in start_indices:
            block = returns[start:start + block_size]
            resampled.extend(block)
        
        # Trim to original size
        resampled = np.array(resampled[:num_trades])
        
        return resampled
    
    def _calculate_equity_curves(self, simulated_returns: List[np.ndarray]) -> np.ndarray:
        """Calculate equity curves from simulated returns.
        
        Args:
            simulated_returns: List of simulated trade return sequences
            
        Returns:
            Array of equity curves, shape (num_simulations, num_trades + 1)
        """
        num_sims = len(simulated_returns)
        num_trades = len(simulated_returns[0])
        
        # Initialize equity curves with initial capital, shape: (num_simulations, num_trades + 1)
        equity_curves = np.ones((num_sims, num_trades + 1)) * self.config.initial_capital
        
        # Calculate equity for each simulation
        for i in range(num_sims):
            for j in range(num_trades):
                # Update equity based on return percentage
                equity_curves[i, j+1] = equity_curves[i, j] * (1 + simulated_returns[i][j])
        
        return equity_curves
    
    def _calculate_simulation_metrics(self, 
                                    equity_curves: np.ndarray, 
                                    simulated_returns: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate performance metrics for each simulation.
        
        Args:
            equity_curves: Array of equity curves
            simulated_returns: List of simulated trade return sequences
            
        Returns:
            Dictionary of arrays with metrics for each simulation
        """
        num_sims = equity_curves.shape[0]
        metrics = {
            "final_equity": np.zeros(num_sims),
            "max_drawdown": np.zeros(num_sims),
            "sharpe_ratio": np.zeros(num_sims),
            "win_rate": np.zeros(num_sims),
            "avg_win": np.zeros(num_sims),
            "avg_loss": np.zeros(num_sims),
            "profit_factor": np.zeros(num_sims),
        }
        
        # Calculate metrics for each simulation
        for i in range(num_sims):
            # Final equity
            metrics["final_equity"][i] = equity_curves[i, -1]
            
            # Max drawdown
            peak = np.maximum.accumulate(equity_curves[i])
            drawdown = (peak - equity_curves[i]) / peak
            metrics["max_drawdown"][i] = drawdown.max() * 100  # as percentage
            
            # Win rate
            returns = simulated_returns[i]
            wins = returns > 0
            metrics["win_rate"][i] = np.mean(wins) * 100  # as percentage
            
            # Average win/loss
            if np.sum(wins) > 0:
                metrics["avg_win"][i] = np.mean(returns[wins])
            else:
                metrics["avg_win"][i] = 0
                
            losses = returns < 0
            if np.sum(losses) > 0:
                metrics["avg_loss"][i] = np.abs(np.mean(returns[losses]))
            else:
                metrics["avg_loss"][i] = 0
                
            # Profit factor
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = np.abs(np.sum(returns[returns < 0]))
            metrics["profit_factor"][i] = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Sharpe ratio (annualized)
            # Assume 252 trading days in a year and that the returns represent daily returns
            returns_for_sharpe = np.diff(equity_curves[i]) / equity_curves[i, :-1]
            annual_factor = 252 / len(returns_for_sharpe)  # Adjust based on trading frequency
            excess_returns = returns_for_sharpe - (self.config.risk_free_rate / 252)  # daily risk-free rate
            
            if np.std(returns_for_sharpe) > 0:
                metrics["sharpe_ratio"][i] = np.sqrt(annual_factor) * np.mean(excess_returns) / np.std(returns_for_sharpe)
            else:
                metrics["sharpe_ratio"][i] = 0
        
        return metrics
    
    def _compute_statistics(self, metrics: Dict[str, np.ndarray]) -> Dict:
        """Compute statistics and confidence intervals for simulation metrics.
        
        Args:
            metrics: Dictionary of arrays with metrics for each simulation
            
        Returns:
            Dictionary with statistics and confidence intervals
        """
        # Confidence level for intervals
        cl = self.config.confidence_level
        alpha = 1 - cl
        
        result = {
            "statistics": {},
            "confidence_intervals": {}
        }
        
        # Calculate statistics and CIs for each metric
        for metric_name, values in metrics.items():
            # Basic statistics
            stats_dict = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
            
            # Confidence interval
            # Using percentile method
            lower = np.percentile(values, alpha/2 * 100)
            upper = np.percentile(values, (1 - alpha/2) * 100)
            
            result["statistics"][metric_name] = stats_dict
            result["confidence_intervals"][metric_name] = (lower, upper)
        
        return result

    def plot_results(self, results: MonteCarloResults, 
                    include_original: bool = True,
                    plot_type: str = "all") -> plt.Figure:
        """Plot Monte Carlo simulation results.
        
        Args:
            results: Monte Carlo simulation results
            include_original: Whether to highlight the original backtest result
            plot_type: Type of plot to create ("equity", "distribution", "metrics", or "all")
            
        Returns:
            Matplotlib Figure object
        """
        if plot_type == "all":
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            self._plot_equity_curves(results, include_original, axes[0, 0])
            self._plot_distributions(results, include_original, axes[0, 1], "final_equity")
            self._plot_distributions(results, include_original, axes[1, 0], "max_drawdown")
            self._plot_distributions(results, include_original, axes[1, 1], "sharpe_ratio")
            
            plt.tight_layout()
            return fig
        
        elif plot_type == "equity":
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_equity_curves(results, include_original, ax)
            return fig
        
        elif plot_type == "distribution":
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            self._plot_distributions(results, include_original, axes[0], "final_equity")
            self._plot_distributions(results, include_original, axes[1], "max_drawdown")
            self._plot_distributions(results, include_original, axes[2], "sharpe_ratio")
            plt.tight_layout()
            return fig
        
        elif plot_type == "metrics":
            fig, ax = plt.subplots(figsize=(12, 8))
            self._plot_metrics_comparison(results, ax)
            return fig
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _plot_equity_curves(self, results: MonteCarloResults, 
                           include_original: bool, 
                           ax: plt.Axes) -> None:
        """Plot equity curves from Monte Carlo simulations.
        
        Args:
            results: Monte Carlo simulation results
            include_original: Whether to highlight the original backtest result
            ax: Matplotlib axes to plot on
        """
        # Plot all equity curves with low alpha for transparency
        time_points = np.arange(results.equity_curves.shape[1])
        
        # Sample a subset of curves if there are too many
        max_curves_to_plot = 200
        if results.equity_curves.shape[0] > max_curves_to_plot:
            indices = np.random.choice(
                results.equity_curves.shape[0], 
                size=max_curves_to_plot, 
                replace=False
            )
            equity_subset = results.equity_curves[indices]
        else:
            equity_subset = results.equity_curves
            
        for i in range(len(equity_subset)):
            ax.plot(time_points, equity_subset[i], color='blue', alpha=0.05)
        
        # Plot confidence bands
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'orange', 'red']
        equity_percentiles = np.percentile(results.equity_curves, percentiles, axis=0)
        
        for i, (p, c) in enumerate(zip(percentiles, colors)):
            ax.plot(time_points, equity_percentiles[i], 
                   color=c, 
                   linewidth=2, 
                   label=f"{p}th percentile")
        
        # Plot original equity curve if requested
        if include_original and hasattr(results.original_results, 'equity_curve'):
            original_equity = results.original_results.equity_curve
            # Make sure original equity has same length, truncate or extend if needed
            if len(original_equity) > len(time_points):
                original_equity = original_equity[:len(time_points)]
            elif len(original_equity) < len(time_points):
                # Pad with last value
                padding = np.full(len(time_points) - len(original_equity), original_equity[-1])
                original_equity = np.concatenate([original_equity, padding])
                
            ax.plot(time_points, original_equity, 
                   color='black', 
                   linewidth=3, 
                   label='Original backtest')
        
        # Add labels and legend
        ax.set_title('Monte Carlo Equity Curves Simulation', fontsize=14)
        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Account Equity', fontsize=12)
        ax.legend()
        ax.grid(True)
        
        # Add confidence level information
        conf_level_text = f"Confidence Level: {results.config.confidence_level * 100}%"
        ax.text(0.02, 0.02, conf_level_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom')
    
    def _plot_distributions(self, results: MonteCarloResults,
                           include_original: bool,
                           ax: plt.Axes,
                           metric_name: str) -> None:
        """Plot distribution of a Monte Carlo simulation metric.
        
        Args:
            results: Monte Carlo simulation results
            include_original: Whether to highlight the original backtest result
            ax: Matplotlib axes to plot on
            metric_name: Name of the metric to plot
        """
        if metric_name == "final_equity":
            values = results.final_equity_distribution
            title = "Final Equity Distribution"
            xlabel = "Final Equity"
            original_value = results.original_results.final_equity if hasattr(results.original_results, 'final_equity') else None
            
        elif metric_name == "max_drawdown":
            values = results.max_drawdown_distribution
            title = "Maximum Drawdown Distribution"
            xlabel = "Maximum Drawdown (%)"
            original_value = results.original_results.max_drawdown_pct if hasattr(results.original_results, 'max_drawdown_pct') else None
            
        elif metric_name == "sharpe_ratio":
            values = results.sharpe_ratio_distribution
            title = "Sharpe Ratio Distribution"
            xlabel = "Sharpe Ratio"
            original_value = results.original_results.sharpe_ratio if hasattr(results.original_results, 'sharpe_ratio') else None
            
        elif metric_name == "win_rate":
            values = results.win_rate_distribution
            title = "Win Rate Distribution"
            xlabel = "Win Rate (%)"
            original_value = results.original_results.win_rate_pct if hasattr(results.original_results, 'win_rate_pct') else None
            
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        
        # Plot histogram
        ax.hist(values, bins=50, alpha=0.7, color='blue', density=True)
        
        # Add kernel density estimate
        if len(values) > 10:  # Ensure enough data points
            kde_x = np.linspace(min(values), max(values), 1000)
            kde = stats.gaussian_kde(values)
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2)
        
        # Plot confidence interval
        conf_interval = results.confidence_intervals[metric_name]
        ax.axvline(x=conf_interval[0], color='green', linestyle='--', 
                  linewidth=2, label=f"{results.config.confidence_level*100:.0f}% CI")
        ax.axvline(x=conf_interval[1], color='green', linestyle='--', linewidth=2)
        
        # Plot original backtest value if available
        if include_original and original_value is not None:
            ax.axvline(x=original_value, color='black', linestyle='-', 
                      linewidth=3, label='Original backtest')
        
        # Add mean and median
        stats_dict = results.statistics[metric_name]
        ax.axvline(x=stats_dict["mean"], color='purple', linestyle='-', 
                  linewidth=2, label='Mean')
        ax.axvline(x=stats_dict["median"], color='orange', linestyle='-', 
                  linewidth=2, label='Median')
        
        # Add labels and legend
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.legend()
        ax.grid(True)
        
        # Add statistics text
        stats_text = (
            f"Mean: {stats_dict['mean']:.2f}\n"
            f"Median: {stats_dict['median']:.2f}\n"
            f"Std Dev: {stats_dict['std']:.2f}\n"
            f"{results.config.confidence_level*100:.0f}% CI: "
            f"[{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]"
        )
        
        # Position the stats text in the upper right corner
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    def _plot_metrics_comparison(self, results: MonteCarloResults, ax: plt.Axes) -> None:
        """Plot comparison of original and Monte Carlo metrics.
        
        Args:
            results: Monte Carlo simulation results
            ax: Matplotlib axes to plot on
        """
        # Metrics to compare
        metrics = [
            "final_equity", 
            "max_drawdown", 
            "sharpe_ratio", 
            "win_rate", 
            "profit_factor"
        ]
        
        # Extract statistics
        mc_means = [results.statistics[m]["mean"] for m in metrics]
        mc_lower = [results.confidence_intervals[m][0] for m in metrics]
        mc_upper = [results.confidence_intervals[m][1] for m in metrics]
        
        # Original values, if available
        original_values = []
        for m in metrics:
            attr_name = m.lower()
            if hasattr(results.original_results, attr_name):
                original_values.append(getattr(results.original_results, attr_name))
            elif hasattr(results.original_results, f"{attr_name}_pct"):
                original_values.append(getattr(results.original_results, f"{attr_name}_pct"))
            else:
                original_values.append(None)
        
        # Filter out metrics with no original values
        valid_indices = [i for i, v in enumerate(original_values) if v is not None]
        if not valid_indices:
            ax.text(0.5, 0.5, "No comparable metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        metrics = [metrics[i] for i in valid_indices]
        mc_means = [mc_means[i] for i in valid_indices]
        mc_lower = [mc_lower[i] for i in valid_indices]
        mc_upper = [mc_upper[i] for i in valid_indices]
        original_values = [original_values[i] for i in valid_indices]
        
        # Normalize values for comparison (0-1 scale)
        normalized_means = []
        normalized_lower = []
        normalized_upper = []
        normalized_original = []
        
        for i, m in enumerate(metrics):
            min_val = min(mc_lower[i], original_values[i])
            max_val = max(mc_upper[i], original_values[i])
            range_val = max_val - min_val if max_val > min_val else 1.0
            
            normalized_means.append((mc_means[i] - min_val) / range_val)
            normalized_lower.append((mc_lower[i] - min_val) / range_val)
            normalized_upper.append((mc_upper[i] - min_val) / range_val)
            normalized_original.append((original_values[i] - min_val) / range_val)
        
        # Plot setup
        x = np.arange(len(metrics))
        width = 0.35
        
        # Plot
        ax.bar(x, normalized_means, width, color='blue', alpha=0.6, label='Monte Carlo Mean')
        ax.bar(x + width, normalized_original, width, color='green', alpha=0.6, label='Original Backtest')
        
        # Add error bars for Monte Carlo
        ax.errorbar(x, normalized_means, 
                   yerr=np.array([
                       np.array(normalized_means) - np.array(normalized_lower),
                       np.array(normalized_upper) - np.array(normalized_means)
                   ]), 
                   fmt='none', color='black', capsize=5)
        
        # Customize the plot
        ax.set_title('Comparison of Original vs. Monte Carlo Metrics', fontsize=14)
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax.legend()
        ax.grid(True, axis='y')
        
        # Add annotations with actual values
        for i in range(len(metrics)):
            # Monte Carlo mean
            ax.text(x[i], normalized_means[i] + 0.02, 
                   f"{mc_means[i]:.2f}", ha='center', va='bottom', 
                   color='blue', fontweight='bold')
            
            # Original
            ax.text(x[i] + width, normalized_original[i] + 0.02, 
                   f"{original_values[i]:.2f}", ha='center', va='bottom', 
                   color='green', fontweight='bold')
