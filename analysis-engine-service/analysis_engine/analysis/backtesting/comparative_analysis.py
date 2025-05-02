"""
Comparative Analysis Tools for Backtesting.

This module provides functionality for comparing different trading strategies,
performing attribution analysis, and evaluating strategy performance across
different market conditions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import logging
from scipy import stats
import datetime as dt
from enum import Enum, auto

from analysis_engine.analysis.backtesting.core import BacktestResults, TradeStats
from analysis_engine.analysis.market_regime import MarketRegimeType

logger = logging.getLogger(__name__)


class AttributionFactorType(Enum):
    """Types of factors for attribution analysis."""
    MARKET_REGIME = auto()
    TIME_OF_DAY = auto()
    DAY_OF_WEEK = auto()
    VOLATILITY = auto()
    TREND_STRENGTH = auto()
    NEWS_IMPACT = auto()
    LIQUIDITY = auto()
    CUSTOM = auto()


@dataclass
class AttributionFactor:
    """Represents a factor for attribution analysis."""
    name: str
    type: AttributionFactorType
    description: str
    values: List[Any] = field(default_factory=list)
    
    # Function to categorize trades based on this factor
    categorize_fn: Optional[Callable] = None


@dataclass
class StrategyComparisonConfig:
    """Configuration for strategy comparison."""
    
    # Base capital for comparison (to normalize results)
    base_capital: float = 100000.0
    
    # Whether to normalize returns for comparison
    normalize_returns: bool = True
    
    # Time period for comparison (e.g., '1y', '2y', 'all')
    time_period: str = 'all'
    
    # Whether to include transaction costs in comparison
    include_transaction_costs: bool = True
    
    # Factors to use for attribution analysis
    attribution_factors: List[AttributionFactor] = field(default_factory=list)
    
    # Benchmark strategy name (if any)
    benchmark_name: Optional[str] = None
    

@dataclass
class PerformanceAttribution:
    """Results of performance attribution analysis."""
    
    # Factor name
    factor_name: str
    
    # Factor type
    factor_type: AttributionFactorType
    
    # Category values for this factor
    categories: List[Any]
    
    # Performance metrics for each category
    metrics_by_category: Dict[str, Dict[str, float]]
    
    # Contribution of each category to overall performance
    contributions: Dict[str, float]
    
    # Statistical significance (p-values) of differences between categories
    significance: Optional[Dict[str, float]] = None


@dataclass
class ComparisonResults:
    """Results from strategy comparison."""
    
    # Names of strategies being compared
    strategy_names: List[str]
    
    # Original backtest results for each strategy
    backtest_results: Dict[str, BacktestResults]
    
    # Normalized equity curves for comparison (same starting capital)
    equity_curves: Dict[str, np.ndarray]
    
    # Normalized returns series for each strategy
    returns: Dict[str, np.ndarray]
    
    # Performance metrics for each strategy
    metrics: Dict[str, Dict[str, float]]
    
    # Correlations between strategy returns
    correlations: pd.DataFrame
    
    # Performance attribution results (if attribution factors were provided)
    attribution: Dict[str, Dict[str, PerformanceAttribution]] = field(default_factory=dict)
    
    # Statistical significance of differences between strategies
    significance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Ranking of strategies by different metrics
    rankings: Dict[str, Dict[str, int]] = field(default_factory=dict)


class ComparativeAnalyzer:
    """Tool for comparing trading strategies and performing attribution analysis."""
    
    def __init__(self, config: StrategyComparisonConfig = None):
        """Initialize the comparative analyzer.
        
        Args:
            config: Configuration for strategy comparison. If None, default config is used.
        """
        self.config = config or StrategyComparisonConfig()
    
    def compare_strategies(self, 
                          results_dict: Dict[str, BacktestResults], 
                          benchmark_results: Optional[BacktestResults] = None) -> ComparisonResults:
        """Compare multiple trading strategies.
        
        Args:
            results_dict: Dictionary mapping strategy names to backtest results
            benchmark_results: Optional backtest results for a benchmark strategy
            
        Returns:
            Results of the comparison
        """
        logger.info(f"Comparing {len(results_dict)} strategies")
        
        # Add benchmark if provided
        if benchmark_results is not None and self.config.benchmark_name:
            results_dict[self.config.benchmark_name] = benchmark_results
            
        strategy_names = list(results_dict.keys())
        
        # Normalize equity curves and compute returns
        equity_curves = {}
        returns = {}
        
        for name, result in results_dict.items():
            # Extract and normalize equity curve
            if hasattr(result, 'equity_curve') and len(result.equity_curve) > 0:
                normalized_equity = self._normalize_equity_curve(result.equity_curve)
                equity_curves[name] = normalized_equity
                
                # Calculate returns from equity curve
                strategy_returns = np.diff(normalized_equity) / normalized_equity[:-1]
                returns[name] = strategy_returns
        
        # Calculate performance metrics for each strategy
        metrics = self._calculate_metrics(results_dict, returns)
        
        # Calculate correlations between strategy returns
        correlations = self._calculate_correlations(returns)
        
        # Create base comparison results
        results = ComparisonResults(
            strategy_names=strategy_names,
            backtest_results=results_dict,
            equity_curves=equity_curves,
            returns=returns,
            metrics=metrics,
            correlations=correlations
        )
        
        # Calculate rankings
        results.rankings = self._calculate_rankings(metrics)
        
        # Calculate statistical significance of differences
        results.significance = self._calculate_significance(returns)
        
        # Perform attribution analysis if factors were provided
        if self.config.attribution_factors:
            results.attribution = self._perform_attribution(results_dict)
        
        return results
    
    def _normalize_equity_curve(self, equity_curve: np.ndarray) -> np.ndarray:
        """Normalize an equity curve to start with the base capital.
        
        Args:
            equity_curve: Original equity curve
            
        Returns:
            Normalized equity curve
        """
        if not self.config.normalize_returns:
            return equity_curve
            
        scale_factor = self.config.base_capital / equity_curve[0]
        return equity_curve * scale_factor
    
    def _calculate_metrics(self, 
                          results_dict: Dict[str, BacktestResults],
                          returns_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each strategy.
        
        Args:
            results_dict: Dictionary of backtest results
            returns_dict: Dictionary of strategy returns
            
        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        metrics = {}
        
        for name, result in results_dict.items():
            strategy_metrics = {}
            
            # Extract metrics from backtest results
            if hasattr(result, 'final_equity'):
                strategy_metrics['final_equity'] = result.final_equity
                
            if hasattr(result, 'total_return_pct'):
                strategy_metrics['total_return_pct'] = result.total_return_pct
                
            if hasattr(result, 'max_drawdown_pct'):
                strategy_metrics['max_drawdown_pct'] = result.max_drawdown_pct
                
            if hasattr(result, 'sharpe_ratio'):
                strategy_metrics['sharpe_ratio'] = result.sharpe_ratio
                
            if hasattr(result, 'sortino_ratio'):
                strategy_metrics['sortino_ratio'] = result.sortino_ratio
                
            if hasattr(result, 'win_rate_pct'):
                strategy_metrics['win_rate_pct'] = result.win_rate_pct
                
            if hasattr(result, 'profit_factor'):
                strategy_metrics['profit_factor'] = result.profit_factor
                
            if hasattr(result, 'expectancy'):
                strategy_metrics['expectancy'] = result.expectancy
                
            if hasattr(result, 'trades'):
                strategy_metrics['num_trades'] = len(result.trades)
            
            # Calculate additional metrics from returns if available
            if name in returns_dict and len(returns_dict[name]) > 0:
                returns = returns_dict[name]
                
                # Annualized return (assuming daily returns)
                annual_factor = 252  # Trading days in a year
                strategy_metrics['annualized_return_pct'] = (
                    (1 + np.mean(returns)) ** annual_factor - 1
                ) * 100
                
                # Annualized volatility
                strategy_metrics['annualized_volatility_pct'] = (
                    np.std(returns) * np.sqrt(annual_factor) * 100
                )
                
                # Calmar ratio
                if 'max_drawdown_pct' in strategy_metrics and strategy_metrics['max_drawdown_pct'] > 0:
                    strategy_metrics['calmar_ratio'] = (
                        strategy_metrics['annualized_return_pct'] / strategy_metrics['max_drawdown_pct']
                    )
                
                # Skewness and Kurtosis
                strategy_metrics['return_skewness'] = stats.skew(returns)
                strategy_metrics['return_kurtosis'] = stats.kurtosis(returns)
                
                # Value at Risk (95%)
                strategy_metrics['var_95'] = np.percentile(returns, 5)
                
                # Modified Sharpe ratio (adjusting for skewness and kurtosis)
                # This accounts for non-normal return distributions
                if strategy_metrics['return_skewness'] != 0 and strategy_metrics['return_kurtosis'] != 0:
                    sr = strategy_metrics.get('sharpe_ratio', 0)
                    skew = strategy_metrics['return_skewness']
                    kurt = strategy_metrics['return_kurtosis']
                    strategy_metrics['modified_sharpe_ratio'] = sr * (
                        1 + (skew / 6) * sr - ((kurt - 3) / 24) * (sr ** 2)
                    )
            
            metrics[name] = strategy_metrics
        
        return metrics
    
    def _calculate_correlations(self, returns_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate correlations between strategy returns.
        
        Args:
            returns_dict: Dictionary mapping strategy names to returns
            
        Returns:
            DataFrame with correlation matrix
        """
        # Convert returns to DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate correlation matrix
        return returns_df.corr()
    
    def _calculate_rankings(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, int]]:
        """Rank strategies by different performance metrics.
        
        Args:
            metrics: Dictionary of strategy metrics
            
        Returns:
            Dictionary mapping metrics to strategy rankings
        """
        rankings = {}
        strategy_names = list(metrics.keys())
        
        # For each metric, create a ranking of strategies
        all_metrics = set()
        for strategy_metrics in metrics.values():
            all_metrics.update(strategy_metrics.keys())
        
        for metric in all_metrics:
            # Extract values for strategies that have this metric
            values = []
            names = []
            
            for name in strategy_names:
                if metric in metrics[name]:
                    values.append(metrics[name][metric])
                    names.append(name)
            
            if not values:
                continue
                
            # Sort strategy names by metric value
            # Higher is better for most metrics, but lower is better for some
            reverse = not (
                'drawdown' in metric.lower() or 
                'volatility' in metric.lower() or
                'var' == metric.lower() or
                'var_' in metric.lower()
            )
            
            # Sort and assign rankings
            sorted_indices = np.argsort(values)
            if reverse:
                sorted_indices = sorted_indices[::-1]
                
            metric_rankings = {}
            for rank, idx in enumerate(sorted_indices):
                metric_rankings[names[idx]] = rank + 1
            
            rankings[metric] = metric_rankings
        
        return rankings
    
    def _calculate_significance(self, returns_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate statistical significance of differences between strategy returns.
        
        Args:
            returns_dict: Dictionary mapping strategy names to returns
            
        Returns:
            Dictionary mapping strategy pairs to p-values
        """
        significance = {}
        strategy_names = list(returns_dict.keys())
        
        for i, name1 in enumerate(strategy_names):
            significance[name1] = {}
            
            for name2 in strategy_names:
                if name1 == name2:
                    significance[name1][name2] = 1.0  # Same strategy
                    continue
                    
                # Perform t-test to compare mean returns
                t_stat, p_value = stats.ttest_ind(
                    returns_dict[name1],
                    returns_dict[name2],
                    equal_var=False  # Welch's t-test (don't assume equal variances)
                )
                
                significance[name1][name2] = p_value
        
        return significance
    
    def _perform_attribution(self, results_dict: Dict[str, BacktestResults]) -> Dict[str, Dict[str, PerformanceAttribution]]:
        """Perform attribution analysis for each strategy.
        
        Args:
            results_dict: Dictionary mapping strategy names to backtest results
            
        Returns:
            Dictionary mapping strategy names to attribution results for each factor
        """
        attribution_results = {}
        
        for strategy_name, backtest_results in results_dict.items():
            if not hasattr(backtest_results, 'trades') or not backtest_results.trades:
                continue
                
            strategy_attribution = {}
            
            for factor in self.config.attribution_factors:
                factor_attribution = self._analyze_factor(backtest_results, factor)
                strategy_attribution[factor.name] = factor_attribution
                
            attribution_results[strategy_name] = strategy_attribution
        
        return attribution_results
    
    def _analyze_factor(self, results: BacktestResults, factor: AttributionFactor) -> PerformanceAttribution:
        """Analyze the impact of a factor on strategy performance.
        
        Args:
            results: Backtest results
            factor: Attribution factor to analyze
            
        Returns:
            Performance attribution for this factor
        """
        trades = results.trades
        
        # Define categories for each trade
        categories = []
        for trade in trades:
            if factor.categorize_fn:
                # Use custom categorization function if provided
                category = factor.categorize_fn(trade)
            else:
                # Default categorization based on factor type
                category = self._default_categorize(trade, factor)
                
            categories.append(category)
        
        # Get unique categories
        unique_categories = sorted(set(categories))
        
        # Calculate metrics for each category
        metrics_by_category = {}
        contributions = {}
        
        for category in unique_categories:
            # Filter trades for this category
            category_trades = [
                trade for trade, cat in zip(trades, categories) if cat == category
            ]
            
            if not category_trades:
                continue
                
            # Calculate basic metrics
            total_pnl = sum(t.profit_loss for t in category_trades)
            win_count = sum(1 for t in category_trades if t.profit_loss > 0)
            category_metrics = {
                'count': len(category_trades),
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / len(category_trades),
                'win_rate': win_count / len(category_trades) if category_trades else 0,
                'pct_of_trades': len(category_trades) / len(trades),
            }
            
            # Calculate contribution to overall performance
            overall_pnl = sum(t.profit_loss for t in trades)
            contribution = total_pnl / overall_pnl if overall_pnl != 0 else 0
            
            metrics_by_category[str(category)] = category_metrics
            contributions[str(category)] = contribution
        
        # Calculate statistical significance of differences between categories
        significance = {}
        if len(unique_categories) > 1:
            for i, cat1 in enumerate(unique_categories):
                for j, cat2 in enumerate(unique_categories):
                    if i >= j:
                        continue
                        
                    # Extract PnL for each category
                    pnl1 = [
                        trade.profit_loss 
                        for trade, cat in zip(trades, categories) 
                        if cat == cat1
                    ]
                    
                    pnl2 = [
                        trade.profit_loss 
                        for trade, cat in zip(trades, categories) 
                        if cat == cat2
                    ]
                    
                    if not pnl1 or not pnl2:
                        continue
                        
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(pnl1, pnl2, equal_var=False)
                    significance[f"{cat1}_vs_{cat2}"] = p_value
        
        return PerformanceAttribution(
            factor_name=factor.name,
            factor_type=factor.type,
            categories=unique_categories,
            metrics_by_category=metrics_by_category,
            contributions=contributions,
            significance=significance
        )
    
    def _default_categorize(self, trade, factor: AttributionFactor) -> Any:
        """Default categorization of trades based on factor type.
        
        Args:
            trade: Trade to categorize
            factor: Attribution factor
            
        Returns:
            Category value
        """
        if factor.type == AttributionFactorType.MARKET_REGIME:
            # Attempt to get market regime from trade metadata
            return trade.metadata.get('market_regime', 'unknown') if hasattr(trade, 'metadata') else 'unknown'
            
        elif factor.type == AttributionFactorType.TIME_OF_DAY:
            # Categorize by time of day
            if hasattr(trade, 'entry_time'):
                hour = trade.entry_time.hour
                if 0 <= hour < 8:
                    return "night"
                elif 8 <= hour < 12:
                    return "morning"
                elif 12 <= hour < 16:
                    return "afternoon"
                else:
                    return "evening"
            return 'unknown'
            
        elif factor.type == AttributionFactorType.DAY_OF_WEEK:
            # Categorize by day of week
            if hasattr(trade, 'entry_time'):
                return trade.entry_time.strftime('%A')  # Monday, Tuesday, etc.
            return 'unknown'
            
        elif factor.type == AttributionFactorType.VOLATILITY:
            # Categorize by volatility level
            if hasattr(trade, 'metadata') and 'volatility' in trade.metadata:
                vol = trade.metadata['volatility']
                if vol < 0.5:
                    return "low"
                elif vol < 1.5:
                    return "medium"
                else:
                    return "high"
            return 'unknown'
            
        elif factor.type == AttributionFactorType.TREND_STRENGTH:
            # Categorize by trend strength
            if hasattr(trade, 'metadata') and 'trend_strength' in trade.metadata:
                trend = trade.metadata['trend_strength']
                if abs(trend) < 0.3:
                    return "ranging"
                elif trend > 0:
                    return "uptrend"
                else:
                    return "downtrend"
            return 'unknown'
            
        elif factor.type == AttributionFactorType.NEWS_IMPACT:
            # Categorize by news impact
            if hasattr(trade, 'metadata') and 'news_impact' in trade.metadata:
                impact = trade.metadata['news_impact']
                if impact < 0.3:
                    return "low"
                elif impact < 0.7:
                    return "medium"
                else:
                    return "high"
            return 'unknown'
            
        elif factor.type == AttributionFactorType.LIQUIDITY:
            # Categorize by market liquidity
            if hasattr(trade, 'metadata') and 'liquidity' in trade.metadata:
                liquidity = trade.metadata['liquidity']
                if liquidity < 0.3:
                    return "low"
                elif liquidity < 0.7:
                    return "medium"
                else:
                    return "high"
            return 'unknown'
            
        else:
            # Unknown factor type
            return 'unknown'
    
    def plot_equity_comparison(self, results: ComparisonResults) -> plt.Figure:
        """Plot equity curves for all strategies.
        
        Args:
            results: Results from strategy comparison
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for name, equity in results.equity_curves.items():
            # Plot equity curve
            ax.plot(equity, label=name)
        
        ax.set_title('Strategy Equity Comparison', fontsize=14)
        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Equity', fontsize=12)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_drawdown_comparison(self, results: ComparisonResults) -> plt.Figure:
        """Plot drawdown curves for all strategies.
        
        Args:
            results: Results from strategy comparison
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for name, equity in results.equity_curves.items():
            # Calculate drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak * 100  # as percentage
            
            # Plot drawdown curve
            ax.plot(drawdown, label=name)
        
        ax.set_title('Strategy Drawdown Comparison', fontsize=14)
        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.invert_yaxis()  # Invert y-axis so the largest drawdowns are at the bottom
        
        return fig
    
    def plot_metrics_comparison(self, results: ComparisonResults) -> plt.Figure:
        """Plot a comparison of key metrics across strategies.
        
        Args:
            results: Results from strategy comparison
            
        Returns:
            Matplotlib Figure object
        """
        # Select key metrics to compare
        key_metrics = [
            'sharpe_ratio', 
            'max_drawdown_pct', 
            'win_rate_pct', 
            'profit_factor',
            'annualized_return_pct'
        ]
        
        # Filter metrics that exist for all strategies
        available_metrics = []
        for metric in key_metrics:
            is_available = all(metric in results.metrics[name] for name in results.strategy_names)
            if is_available:
                available_metrics.append(metric)
        
        if not available_metrics:
            # No common metrics found
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No common metrics available for comparison", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set the positions for the bars
        x = np.arange(len(available_metrics))
        width = 0.8 / len(results.strategy_names)
        
        # Plot bars for each strategy
        for i, name in enumerate(results.strategy_names):
            values = [results.metrics[name].get(metric, 0) for metric in available_metrics]
            position = x + (i - len(results.strategy_names)/2 + 0.5) * width
            ax.bar(position, values, width, label=name)
        
        # Customize the plot
        ax.set_title('Strategy Metrics Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics], rotation=45)
        ax.legend()
        ax.grid(True, axis='y')
        
        # Add metric values as text on top of each bar
        for i, name in enumerate(results.strategy_names):
            for j, metric in enumerate(available_metrics):
                if metric in results.metrics[name]:
                    value = results.metrics[name][metric]
                    position = x[j] + (i - len(results.strategy_names)/2 + 0.5) * width
                    ax.text(position, value, f"{value:.2f}", ha='center', va='bottom',
                            fontsize=8, rotation=90)
        
        # Set a sensible y-range based on the data
        max_val = max(
            results.metrics[name].get(metric, 0)
            for name in results.strategy_names
            for metric in available_metrics
            if metric in results.metrics[name]
        )
        ax.set_ylim(0, max_val * 1.2)
        
        return fig
    
    def plot_correlation_heatmap(self, results: ComparisonResults) -> plt.Figure:
        """Plot a heatmap of strategy return correlations.
        
        Args:
            results: Results from strategy comparison
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(results.correlations, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(len(results.strategy_names)))
        ax.set_yticks(np.arange(len(results.strategy_names)))
        ax.set_xticklabels(results.strategy_names)
        ax.set_yticklabels(results.strategy_names)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(results.strategy_names)):
            for j in range(len(results.strategy_names)):
                text = ax.text(j, i, f"{results.correlations.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black")
        
        ax.set_title("Strategy Return Correlations", fontsize=14)
        fig.tight_layout()
        
        return fig
    
    def plot_attribution_results(self, 
                                results: ComparisonResults, 
                                strategy_name: str,
                                factor_name: str) -> plt.Figure:
        """Plot attribution results for a specific strategy and factor.
        
        Args:
            results: Results from strategy comparison
            strategy_name: Name of the strategy to plot attribution for
            factor_name: Name of the attribution factor to plot
            
        Returns:
            Matplotlib Figure object
        """
        if (strategy_name not in results.attribution or 
            factor_name not in results.attribution[strategy_name]):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No attribution results available for strategy '{strategy_name}' and factor '{factor_name}'",
                   ha='center', va='center', fontsize=12)
            return fig
        
        attr = results.attribution[strategy_name][factor_name]
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Attribution Analysis: {strategy_name} - {factor_name}", fontsize=16)
        
        # 1. Bar chart of trade counts by category
        ax = axs[0, 0]
        categories = list(attr.metrics_by_category.keys())
        counts = [attr.metrics_by_category[cat]['count'] for cat in categories]
        ax.bar(categories, counts)
        ax.set_title('Trade Count by Category')
        ax.set_ylabel('Number of Trades')
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # 2. Bar chart of average P&L by category
        ax = axs[0, 1]
        avg_pnl = [attr.metrics_by_category[cat]['avg_pnl'] for cat in categories]
        ax.bar(categories, avg_pnl)
        ax.set_title('Average P&L by Category')
        ax.set_ylabel('P&L')
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # 3. Pie chart of contribution to overall performance
        ax = axs[1, 0]
        contributions = [attr.contributions[cat] for cat in categories]
        ax.pie(contributions, labels=categories, autopct='%1.1f%%', startangle=90)
        ax.set_title('Contribution to Overall Performance')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # 4. Bar chart of win rate by category
        ax = axs[1, 1]
        win_rates = [attr.metrics_by_category[cat]['win_rate'] * 100 for cat in categories]
        ax.bar(categories, win_rates)
        ax.set_title('Win Rate by Category')
        ax.set_ylabel('Win Rate (%)')
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig


# Sample attribution factors
def create_standard_attribution_factors() -> List[AttributionFactor]:
    """Create a standard set of attribution factors for analysis."""
    
    factors = [
        AttributionFactor(
            name="Market Regime",
            type=AttributionFactorType.MARKET_REGIME,
            description="Analysis by different market regimes (trending, ranging, volatile)"
        ),
        
        AttributionFactor(
            name="Time of Day",
            type=AttributionFactorType.TIME_OF_DAY,
            description="Analysis by different times of the trading day"
        ),
        
        AttributionFactor(
            name="Day of Week",
            type=AttributionFactorType.DAY_OF_WEEK,
            description="Analysis by different days of the week"
        ),
        
        AttributionFactor(
            name="Volatility",
            type=AttributionFactorType.VOLATILITY,
            description="Analysis by market volatility levels"
        ),
        
        AttributionFactor(
            name="Trend Strength",
            type=AttributionFactorType.TREND_STRENGTH,
            description="Analysis by trend strength"
        )
    ]
    
    return factors
