import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

class MarketRegime(str, Enum):
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"

class ToolEffectivenessMetric(str, Enum):
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    EXPECTANCY = "expectancy"
    AVERAGE_TRADE = "average_trade"
    AVERAGE_WINNER = "average_winner"
    AVERAGE_LOSER = "average_loser"
    PROFIT_PER_DAY = "profit_per_day"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    COVERAGE = "coverage"

logger = logging.getLogger(__name__)

class RLEffectivenessFramework:
    """
    Enhanced framework for evaluating RL model performance across different market conditions
    and trading scenarios. This framework provides detailed analytics to understand when and
    why RL models perform well or poorly.
    """
    
    def __init__(self, 
                 baseline_strategies: Dict[str, Any] = None,
                 significance_level: float = 0.05,
                 bootstrap_samples: int = 1000):
        """
        Initialize the RL Effectiveness Framework
        
        Parameters:
        -----------
        baseline_strategies : Dict[str, Any]
            Dictionary of baseline strategies to compare against
        significance_level : float
            Statistical significance level for hypothesis tests
        bootstrap_samples : int
            Number of bootstrap samples for statistical tests
        """
        self.baseline_strategies = baseline_strategies or {}
        self.significance_level = significance_level
        self.bootstrap_samples = bootstrap_samples
        
        # Initialize the analyzer components
        self.regime_analyzer = MarketRegimeEffectivenessAnalyzer()
        self.action_quality_analyzer = ActionQualityAnalyzer()
        self.decay_analyzer = EffectivenessDecayAnalyzer()
        
        # Storage for model results
        self.models_results = {}
        self.regime_results = {}
    
    def add_model_results(self, 
                          model_name: str, 
                          trades: pd.DataFrame,
                          actions: Optional[pd.DataFrame] = None,
                          market_data: Optional[pd.DataFrame] = None,
                          model_confidence: Optional[pd.DataFrame] = None):
        """
        Add trading results from an RL model for analysis
        
        Parameters:
        -----------
        model_name : str
            Unique identifier for the model
        trades : pd.DataFrame
            DataFrame containing trade information with columns:
            - entry_time: timestamp of trade entry
            - exit_time: timestamp of trade exit
            - direction: 1 for long, -1 for short
            - entry_price: price at entry
            - exit_price: price at exit
            - profit_loss: P&L of the trade
            - market_regime: (optional) market regime label
        actions : pd.DataFrame, optional
            DataFrame containing all model actions (not just trades) with columns:
            - timestamp: when the action was taken
            - action: the action taken (buy, sell, hold, etc.)
            - confidence: model's confidence in the action
            - predicted_value: predicted value of the action
        market_data : pd.DataFrame, optional
            Market data during the testing period
        model_confidence : pd.DataFrame, optional
            Model's confidence scores over time
        """
        self.models_results[model_name] = {
            'trades': trades,
            'actions': actions,
            'market_data': market_data,
            'model_confidence': model_confidence,
            'metrics': self._calculate_metrics(trades)
        }
        
        # Add to regime analyzer
        if 'market_regime' in trades.columns:
            regime_metrics = self.regime_analyzer.analyze_regimes(trades)
            self.regime_results[model_name] = regime_metrics
    
    def _calculate_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a set of trades
        
        Parameters:
        -----------
        trades : pd.DataFrame
            DataFrame containing trade information
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of performance metrics
        """
        metrics = {}
        
        if trades.empty:
            logger.warning("No trades provided for metrics calculation")
            return metrics
        
        # Basic metrics
        winning_trades = trades[trades['profit_loss'] > 0]
        losing_trades = trades[trades['profit_loss'] <= 0]
        
        # Win rate
        metrics[ToolEffectivenessMetric.WIN_RATE] = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        # Profit factor
        total_profit = winning_trades['profit_loss'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0
        metrics[ToolEffectivenessMetric.PROFIT_FACTOR] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average trade
        metrics[ToolEffectivenessMetric.AVERAGE_TRADE] = trades['profit_loss'].mean()
        metrics[ToolEffectivenessMetric.AVERAGE_WINNER] = winning_trades['profit_loss'].mean() if not winning_trades.empty else 0
        metrics[ToolEffectivenessMetric.AVERAGE_LOSER] = losing_trades['profit_loss'].mean() if not losing_trades.empty else 0
        
        # Expectancy
        metrics[ToolEffectivenessMetric.EXPECTANCY] = (
            metrics[ToolEffectivenessMetric.WIN_RATE] * metrics[ToolEffectivenessMetric.AVERAGE_WINNER] + 
            (1 - metrics[ToolEffectivenessMetric.WIN_RATE]) * metrics[ToolEffectivenessMetric.AVERAGE_LOSER]
        )
        
        # Risk-adjusted returns
        if 'cumulative_returns' not in trades.columns:
            # Create a Series of cumulative returns
            trades_sorted = trades.sort_values('entry_time')
            trades_sorted['cumulative_returns'] = (1 + trades_sorted['profit_loss']).cumprod()
        
            # Calculate daily returns if we have timestamp information
            if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
                # Get the total trading period
                start_date = trades['entry_time'].min()
                end_date = trades['exit_time'].max()
                
                if pd.notna(start_date) and pd.notna(end_date):
                    trading_days = (end_date - start_date).days
                    metrics[ToolEffectivenessMetric.PROFIT_PER_DAY] = (
                        (trades_sorted['cumulative_returns'].iloc[-1] - 1) / trading_days
                        if trading_days > 0 else 0
                    )
        
        # Maximum drawdown
        cumulative_returns = trades_sorted['cumulative_returns']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        metrics[ToolEffectivenessMetric.MAX_DRAWDOWN] = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Sharpe and Sortino ratios
        if len(trades) > 1:
            returns = trades['profit_loss']
            metrics[ToolEffectivenessMetric.SHARPE_RATIO] = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0 else 0
            )
            
            # Sortino ratio using only negative returns for denominator
            downside_returns = returns[returns < 0]
            metrics[ToolEffectivenessMetric.SORTINO_RATIO] = (
                returns.mean() / downside_returns.std() * np.sqrt(252)
                if not downside_returns.empty and downside_returns.std() > 0 else 0
            )
            
            # Calmar ratio
            metrics[ToolEffectivenessMetric.CALMAR_RATIO] = (
                metrics[ToolEffectivenessMetric.PROFIT_PER_DAY] * 252 / metrics[ToolEffectivenessMetric.MAX_DRAWDOWN]
                if metrics[ToolEffectivenessMetric.MAX_DRAWDOWN] > 0 else 0
            )
        
        return metrics
    
    def compare_models(self, metrics: List[ToolEffectivenessMetric] = None) -> pd.DataFrame:
        """
        Compare all models across specified metrics
        
        Parameters:
        -----------
        metrics : List[ToolEffectivenessMetric], optional
            List of metrics to include in the comparison
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with metrics for each model
        """
        if not self.models_results:
            return pd.DataFrame()
        
        if metrics is None:
            metrics = [
                ToolEffectivenessMetric.WIN_RATE,
                ToolEffectivenessMetric.PROFIT_FACTOR,
                ToolEffectivenessMetric.SHARPE_RATIO,
                ToolEffectivenessMetric.MAX_DRAWDOWN,
                ToolEffectivenessMetric.EXPECTANCY
            ]
        
        comparison = {}
        for model_name, results in self.models_results.items():
            comparison[model_name] = {
                metric.value: results['metrics'].get(metric, float('nan'))
                for metric in metrics
            }
        
        return pd.DataFrame.from_dict(comparison, orient='index')
    
    def compare_regimes(self, models: List[str] = None, 
                       regimes: List[MarketRegime] = None) -> pd.DataFrame:
        """
        Compare model performance across different market regimes
        
        Parameters:
        -----------
        models : List[str], optional
            List of model names to include in the comparison
        regimes : List[MarketRegime], optional
            List of market regimes to include in the comparison
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with metrics for each model and regime
        """
        if not self.regime_results:
            return pd.DataFrame()
        
        models = models or list(self.regime_results.keys())
        
        # Collect all unique regimes if not specified
        if regimes is None:
            regimes = set()
            for model_results in self.regime_results.values():
                regimes.update(model_results.keys())
            regimes = list(regimes)
        
        # Create a multi-index DataFrame
        data = []
        index = []
        
        for model in models:
            if model not in self.regime_results:
                continue
                
            for regime in regimes:
                if regime not in self.regime_results[model]:
                    continue
                    
                regime_data = self.regime_results[model][regime]
                data.append([
                    regime_data.get(ToolEffectivenessMetric.WIN_RATE, float('nan')),
                    regime_data.get(ToolEffectivenessMetric.PROFIT_FACTOR, float('nan')),
                    regime_data.get(ToolEffectivenessMetric.EXPECTANCY, float('nan')),
                    regime_data.get('trade_count', 0)
                ])
                index.append((model, regime))
                
        columns = ['Win Rate', 'Profit Factor', 'Expectancy', 'Trade Count']
        return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=['Model', 'Regime']), 
                           columns=columns)
    
    def statistical_comparison(self, 
                              baseline_model: str, 
                              test_model: str,
                              metric: ToolEffectivenessMetric = ToolEffectivenessMetric.PROFIT_FACTOR) -> Dict[str, Any]:
        """
        Perform statistical comparison between two models
        
        Parameters:
        -----------
        baseline_model : str
            Name of the baseline model
        test_model : str
            Name of the model to test
        metric : ToolEffectivenessMetric, optional
            Metric to use for comparison
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with statistical test results
        """
        if baseline_model not in self.models_results or test_model not in self.models_results:
            return {'error': 'One or both models not found'}
        
        baseline_trades = self.models_results[baseline_model]['trades']
        test_trades = self.models_results[test_model]['trades']
        
        if baseline_trades.empty or test_trades.empty:
            return {'error': 'One or both models have no trades'}
        
        # Extract the relevant metric from trades
        if metric == ToolEffectivenessMetric.PROFIT_FACTOR:
            baseline_values = baseline_trades['profit_loss']
            test_values = test_trades['profit_loss']
        else:
            # Default to profit_loss for now
            baseline_values = baseline_trades['profit_loss']
            test_values = test_trades['profit_loss']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(baseline_values, test_values, equal_var=False)
        
        # Bootstrap test
        bootstrap_means = []
        for _ in range(self.bootstrap_samples):
            baseline_sample = baseline_values.sample(len(baseline_values), replace=True).mean()
            test_sample = test_values.sample(len(test_values), replace=True).mean()
            bootstrap_means.append(test_sample - baseline_sample)
        
        bootstrap_means = np.array(bootstrap_means)
        confidence_interval = (
            np.percentile(bootstrap_means, 2.5),
            np.percentile(bootstrap_means, 97.5)
        )
        
        return {
            'baseline_mean': baseline_values.mean(),
            'test_mean': test_values.mean(),
            'difference': test_values.mean() - baseline_values.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'confidence_interval': confidence_interval,
            'sample_sizes': (len(baseline_values), len(test_values))
        }
    
    def plot_regime_performance(self, model: str = None, 
                               metric: ToolEffectivenessMetric = ToolEffectivenessMetric.WIN_RATE,
                               figsize: Tuple[int, int] = (10, 6)):
        """
        Plot performance across different market regimes
        
        Parameters:
        -----------
        model : str, optional
            Model name to plot, if None plots all models
        metric : ToolEffectivenessMetric, optional
            Metric to plot
        figsize : Tuple[int, int], optional
            Figure size
        """
        if not self.regime_results:
            print("No regime analysis results available")
            return
        
        plt.figure(figsize=figsize)
        
        if model is not None and model in self.regime_results:
            # Plot single model across regimes
            regimes = []
            values = []
            
            for regime, metrics in self.regime_results[model].items():
                if metric in metrics:
                    regimes.append(regime)
                    values.append(metrics[metric])
            
            sns.barplot(x=regimes, y=values)
            plt.title(f"{metric.value.replace('_', ' ').title()} by Market Regime\nModel: {model}")
            
        else:
            # Compare all models across regimes
            data = []
            for model_name, regime_data in self.regime_results.items():
                for regime, metrics in regime_data.items():
                    if metric in metrics:
                        data.append({
                            'Model': model_name,
                            'Regime': regime,
                            metric.value: metrics[metric]
                        })
            
            if data:
                df = pd.DataFrame(data)
                sns.barplot(x='Regime', y=metric.value, hue='Model', data=df)
                plt.title(f"{metric.value.replace('_', ' ').title()} by Market Regime")
                plt.legend(title='Model')
            else:
                print(f"No data available for metric {metric}")
        
        plt.xlabel('Market Regime')
        plt.ylabel(metric.value.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_confidence_calibration(self, model: str):
        """
        Plot calibration curve for model confidence vs. actual outcomes
        
        Parameters:
        -----------
        model : str
            Model name to analyze
        """
        if model not in self.models_results or self.models_results[model]['actions'] is None:
            print(f"No action data available for model {model}")
            return
        
        actions = self.models_results[model]['actions']
        
        if 'confidence' not in actions.columns or 'correct' not in actions.columns:
            print("Action data must contain 'confidence' and 'correct' columns")
            return
        
        self.action_quality_analyzer.plot_calibration_curve(actions)
    
    def analyze_model_decay(self, model: str, window_size: int = 20):
        """
        Analyze how model performance decays over time
        
        Parameters:
        -----------
        model : str
            Model name to analyze
        window_size : int, optional
            Rolling window size for calculating metrics
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with decay metrics over time
        """
        if model not in self.models_results:
            print(f"Model {model} not found")
            return None
        
        trades = self.models_results[model]['trades']
        
        if trades.empty:
            print(f"No trades available for model {model}")
            return None
        
        return self.decay_analyzer.analyze_decay(trades, window_size)
    
    def generate_effectiveness_report(self, model: str) -> Dict[str, Any]:
        """
        Generate a comprehensive effectiveness report for a model
        
        Parameters:
        -----------
        model : str
            Model name to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with comprehensive report data
        """
        if model not in self.models_results:
            return {'error': f"Model {model} not found"}
        
        results = self.models_results[model]
        trades = results['trades']
        
        if trades.empty:
            return {'error': f"No trades available for model {model}"}
        
        # Overall performance metrics
        overall_metrics = results['metrics']
        
        # Regime-specific performance
        regime_metrics = self.regime_results.get(model, {})
        
        # Decay analysis
        decay_analysis = self.decay_analyzer.analyze_decay(trades)
        
        # Action quality analysis
        action_quality = {}
        if results['actions'] is not None:
            action_quality = self.action_quality_analyzer.analyze_actions(results['actions'])
        
        return {
            'model_name': model,
            'trade_count': len(trades),
            'overall_metrics': overall_metrics,
            'regime_metrics': regime_metrics,
            'decay_analysis': decay_analysis,
            'action_quality': action_quality,
            'trading_period': {
                'start': trades['entry_time'].min(),
                'end': trades['exit_time'].max(),
                'days': (trades['exit_time'].max() - trades['entry_time'].min()).days
            }
        }


class MarketRegimeEffectivenessAnalyzer:
    """
    Analyzes trading model effectiveness across different market regimes
    """
    
    def analyze_regimes(self, trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze trade performance broken down by market regime
        
        Parameters:
        -----------
        trades : pd.DataFrame
            DataFrame containing trade information with a market_regime column
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of performance metrics for each market regime
        """
        if 'market_regime' not in trades.columns:
            logger.warning("No market regime information in trades data")
            return {}
        
        regime_metrics = {}
        
        for regime, regime_trades in trades.groupby('market_regime'):
            # Skip if there are too few trades in this regime
            if len(regime_trades) < 5:
                continue
                
            winning_trades = regime_trades[regime_trades['profit_loss'] > 0]
            
            # Calculate regime-specific metrics
            win_rate = len(winning_trades) / len(regime_trades) if len(regime_trades) > 0 else 0
            
            total_profit = winning_trades['profit_loss'].sum() if not winning_trades.empty else 0
            total_loss = abs(regime_trades[regime_trades['profit_loss'] <= 0]['profit_loss'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_trade = regime_trades['profit_loss'].mean()
            avg_win = winning_trades['profit_loss'].mean() if not winning_trades.empty else 0
            avg_loss = regime_trades[regime_trades['profit_loss'] <= 0]['profit_loss'].mean() if len(regime_trades[regime_trades['profit_loss'] <= 0]) > 0 else 0
            
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
            
            regime_metrics[regime] = {
                ToolEffectivenessMetric.WIN_RATE: win_rate,
                ToolEffectivenessMetric.PROFIT_FACTOR: profit_factor,
                ToolEffectivenessMetric.AVERAGE_TRADE: avg_trade,
                ToolEffectivenessMetric.EXPECTANCY: expectancy,
                'trade_count': len(regime_trades)
            }
        
        return regime_metrics
    
    def compare_regime_effectiveness(self, 
                                     trades_by_model: Dict[str, pd.DataFrame]
                                    ) -> pd.DataFrame:
        """
        Compare effectiveness across regimes between multiple models
        
        Parameters:
        -----------
        trades_by_model : Dict[str, pd.DataFrame]
            Dictionary mapping model names to their trades
            
        Returns:
        --------
        pd.DataFrame
            DataFrame comparing models across regimes
        """
        comparison_data = []
        
        for model_name, trades in trades_by_model.items():
            regime_metrics = self.analyze_regimes(trades)
            
            for regime, metrics in regime_metrics.items():
                row = {'model': model_name, 'regime': regime}
                row.update({metric.value: value for metric, value in metrics.items()})
                comparison_data.append(row)
        
        if not comparison_data:
            return pd.DataFrame()
            
        return pd.DataFrame(comparison_data)


class ActionQualityAnalyzer:
    """
    Analyzes the quality of individual actions and predictions made by the model
    """
    
    def analyze_actions(self, actions: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze action quality metrics
        
        Parameters:
        -----------
        actions : pd.DataFrame
            DataFrame containing model actions
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of action quality metrics
        """
        if actions.empty:
            return {}
        
        results = {}
        
        # Check required columns
        required_columns = ['action', 'confidence']
        if not all(col in actions.columns for col in required_columns):
            logger.warning(f"Actions dataframe missing required columns: {required_columns}")
            return results
        
        # Action distribution
        action_counts = actions['action'].value_counts().to_dict()
        results['action_distribution'] = action_counts
        
        # Confidence distribution
        if 'confidence' in actions.columns:
            results['avg_confidence'] = actions['confidence'].mean()
            results['confidence_distribution'] = {
                'min': actions['confidence'].min(),
                'q1': actions['confidence'].quantile(0.25),
                'median': actions['confidence'].median(),
                'q3': actions['confidence'].quantile(0.75),
                'max': actions['confidence'].max()
            }
        
        # Accuracy metrics
        if 'correct' in actions.columns:
            correct_actions = actions[actions['correct'] == True]
            results['accuracy'] = len(correct_actions) / len(actions) if len(actions) > 0 else 0
            
            # Calibration metrics
            if 'confidence' in actions.columns:
                results['calibration'] = self._calculate_calibration_metrics(actions)
        
        return results
    
    def _calculate_calibration_metrics(self, actions: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate calibration metrics to see if model's confidence matches its accuracy
        
        Parameters:
        -----------
        actions : pd.DataFrame
            DataFrame containing model actions with confidence and correctness
            
        Returns:
        --------
        Dict[str, Any]
            Calibration metrics
        """
        # Group by confidence bins
        actions['confidence_bin'] = pd.cut(actions['confidence'], bins=10)
        
        calibration_data = []
        for bin_name, group in actions.groupby('confidence_bin'):
            accuracy = group['correct'].mean() if len(group) > 0 else 0
            avg_confidence = group['confidence'].mean()
            
            calibration_data.append({
                'bin': bin_name,
                'bin_center': bin_name.mid,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'count': len(group),
                'calibration_error': abs(accuracy - avg_confidence)
            })
        
        calibration_df = pd.DataFrame(calibration_data)
        
        return {
            'calibration_data': calibration_data,
            'mean_calibration_error': calibration_df['calibration_error'].mean() if not calibration_df.empty else 0
        }
    
    def plot_calibration_curve(self, actions: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot calibration curve showing model confidence vs. actual accuracy
        
        Parameters:
        -----------
        actions : pd.DataFrame
            DataFrame containing model actions with confidence and correctness
        figsize : Tuple[int, int], optional
            Figure size
        """
        if 'confidence' not in actions.columns or 'correct' not in actions.columns:
            print("Actions dataframe must have 'confidence' and 'correct' columns")
            return
        
        # Create confidence bins
        actions['confidence_bin'] = pd.cut(actions['confidence'], bins=10)
        
        # Calculate accuracy for each bin
        bin_data = actions.groupby('confidence_bin').agg({
            'correct': 'mean',
            'confidence': 'mean',
            'action': 'count'
        }).reset_index()
        bin_data = bin_data.rename(columns={'correct': 'accuracy', 'action': 'count'})
        
        plt.figure(figsize=figsize)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot actual calibration
        plt.scatter(bin_data['confidence'], bin_data['accuracy'], 
                   s=bin_data['count'] / bin_data['count'].max() * 100, 
                   alpha=0.7)
        
        for _, row in bin_data.iterrows():
            plt.annotate(f"{row['count']}", 
                        (row['confidence'], row['accuracy']),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center')
        
        plt.xlabel('Predicted Probability (Confidence)')
        plt.ylabel('Observed Frequency (Accuracy)')
        plt.title('Calibration Curve')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()


class EffectivenessDecayAnalyzer:
    """
    Analyzes how model effectiveness degrades over time 
    and identifies when retraining might be needed
    """
    
    def analyze_decay(self, trades: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
        """
        Analyze performance decay over time
        
        Parameters:
        -----------
        trades : pd.DataFrame
            DataFrame containing trade information with timestamps
        window_size : int, optional
            Rolling window size for calculating metrics
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with decay metrics over time
        """
        if trades.empty or 'entry_time' not in trades.columns:
            logger.warning("Cannot analyze decay without timestamped trades")
            return pd.DataFrame()
        
        # Sort trades by time
        trades_sorted = trades.sort_values('entry_time')
        
        # Create rolling window metrics
        rolling_metrics = pd.DataFrame(index=trades_sorted.index)
        
        # Add entry time for reference
        rolling_metrics['entry_time'] = trades_sorted['entry_time']
        
        # Calculate rolling win rate
        rolling_metrics['win_rate'] = (
            trades_sorted['profit_loss'].apply(lambda x: 1 if x > 0 else 0)
            .rolling(window_size, min_periods=5)
            .mean()
        )
        
        # Calculate rolling average trade
        rolling_metrics['avg_trade'] = (
            trades_sorted['profit_loss']
            .rolling(window_size, min_periods=5)
            .mean()
        )
        
        # Calculate rolling Sharpe ratio
        rolling_metrics['sharpe_ratio'] = (
            trades_sorted['profit_loss']
            .rolling(window_size, min_periods=10)
            .apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)
        )
        
        # Calculate profit factor
        def rolling_profit_factor(window):
            profits = window[window > 0].sum()
            losses = abs(window[window < 0].sum())
            return profits / losses if losses > 0 else float('inf')
            
        rolling_metrics['profit_factor'] = (
            trades_sorted['profit_loss']
            .rolling(window_size, min_periods=10)
            .apply(rolling_profit_factor)
        )
        
        # Add trade count by time period (e.g., day or week)
        if pd.api.types.is_datetime64_any_dtype(trades_sorted['entry_time']):
            trades_sorted['date'] = trades_sorted['entry_time'].dt.date
            trade_counts = trades_sorted.groupby('date').size()
            trade_dates = pd.DataFrame({'trade_count': trade_counts})
            trade_dates.index = pd.to_datetime(trade_dates.index)
            
            # Merge back to rolling metrics on the nearest date
            # This creates a daily trade count that can be analyzed for patterns
            date_index = pd.date_range(start=trades_sorted['entry_time'].min().date(), 
                                      end=trades_sorted['entry_time'].max().date())
            daily_metrics = pd.DataFrame(index=date_index)
            daily_metrics['trade_count'] = trade_dates['trade_count']
            daily_metrics['trade_count'] = daily_metrics['trade_count'].fillna(0)
            
            # Forward fill metrics for days without trades
            for metric in ['win_rate', 'avg_trade', 'sharpe_ratio', 'profit_factor']:
                # Get last value of each day
                daily_values = rolling_metrics.groupby(rolling_metrics['entry_time'].dt.date)[metric].last()
                daily_values.index = pd.to_datetime(daily_values.index)
                
                # Add to daily metrics
                daily_metrics[metric] = daily_values
                daily_metrics[metric] = daily_metrics[metric].fillna(method='ffill')
            
            # Calculate decay indicators
            daily_metrics['win_rate_trend'] = (
                daily_metrics['win_rate']
                .rolling(10, min_periods=5)
                .apply(lambda x: stats.linregress(range(len(x)), x)[0])
            )
            
            daily_metrics['avg_trade_trend'] = (
                daily_metrics['avg_trade']
                .rolling(10, min_periods=5)
                .apply(lambda x: stats.linregress(range(len(x)), x)[0])
            )
            
            # Detect decay warning - negative trends for N consecutive days
            daily_metrics['decay_warning'] = (
                (daily_metrics['win_rate_trend'] < 0) & 
                (daily_metrics['avg_trade_trend'] < 0)
            )
            
            daily_metrics['decay_warning_count'] = (
                daily_metrics['decay_warning']
                .rolling(7)
                .sum()
            )
            
            daily_metrics['needs_retraining'] = daily_metrics['decay_warning_count'] >= 5
            
            return daily_metrics
            
        return rolling_metrics
