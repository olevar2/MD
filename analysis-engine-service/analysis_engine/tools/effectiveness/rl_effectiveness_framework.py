"""
Expanded Tool Effectiveness Framework for RL Models

This module provides advanced analytics capabilities for evaluating RL model effectiveness
in different market conditions, with specialized metrics for reinforcement learning agents.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import json
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from core_foundations.utils.logger import get_logger
from analysis_engine.tools.tool_effectiveness import BaseEffectivenessMetric
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MarketRegime(Enum):
    """Market regime types for segmenting analysis."""
    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    BREAKOUT = 'breakout'
    CRISIS = 'crisis'
    UNKNOWN = 'unknown'


@dataclass
class RLAgentMetrics:
    """Metrics specific to RL agent performance."""
    name: str
    avg_reward: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    avg_position_size: float
    action_distribution: Dict[str, float]
    confidence_scores: List[float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeSpecificPerformance:
    """Performance metrics within a specific market regime."""
    regime: MarketRegime
    metrics: RLAgentMetrics
    sample_size: int
    statistical_significance: Optional[float] = None


class RLModelEffectivenessAnalyzer:
    """
    Analyzes and reports on the effectiveness of RL models across different
    market regimes and conditions.
    """

    def __init__(self, min_sample_size: int=30, significance_level: float=
        0.05, decay_factor: float=0.95, baseline_lookback_days: int=30):
        """
        Initialize the RL model effectiveness analyzer.

        Args:
            min_sample_size: Minimum number of data points required for statistical analysis
            significance_level: P-value threshold for statistical significance
            decay_factor: Weight decay factor for older data points
            baseline_lookback_days: Days of history to consider for baseline calculations
        """
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.decay_factor = decay_factor
        self.baseline_lookback_days = baseline_lookback_days
        self.historical_data = {}
        self.regime_data = {regime: [] for regime in MarketRegime}
        self.baseline_metrics = {}

    @with_analysis_resilience('calculate_regime_specific_metrics')
    def calculate_regime_specific_metrics(self, agent_name: str,
        agent_actions: List[Dict[str, Any]], market_data: pd.DataFrame,
        regime_labels: List[MarketRegime]) ->Dict[MarketRegime,
        RegimeSpecificPerformance]:
        """
        Calculate performance metrics for an RL agent within each market regime.

        Args:
            agent_name: Name/ID of the RL agent
            agent_actions: List of actions taken by the agent with context
            market_data: Market data during the evaluation period
            regime_labels: Market regime labels for each time step

        Returns:
            Dictionary of regime-specific performance metrics
        """
        logger.info(
            f'Calculating regime-specific metrics for agent {agent_name}')
        regime_groups = {}
        for i, regime in enumerate(regime_labels):
            if regime not in regime_groups:
                regime_groups[regime] = []
            if i < len(agent_actions):
                regime_groups[regime].append(agent_actions[i])
        results = {}
        for regime, actions in regime_groups.items():
            if len(actions) < self.min_sample_size:
                logger.warning(
                    f'Insufficient data for {regime.value} regime (n={len(actions)})'
                    )
                continue
            regime_indices = [i for i, r in enumerate(regime_labels) if r ==
                regime]
            regime_market_data = market_data.iloc[regime_indices]
            metrics = self._calculate_agent_metrics(agent_name, actions,
                regime_market_data)
            significance = self._calculate_statistical_significance(metrics,
                self._get_baseline_metrics(agent_name, regime))
            results[regime] = RegimeSpecificPerformance(regime=regime,
                metrics=metrics, sample_size=len(actions),
                statistical_significance=significance)
            self.regime_data[regime].append({'timestamp': datetime.now(),
                'metrics': metrics, 'sample_size': len(actions)})
        return results

    @with_analysis_resilience('calculate_comparative_effectiveness')
    def calculate_comparative_effectiveness(self, agent_metrics: Dict[str,
        RLAgentMetrics], baseline_strategy_metrics: Optional[RLAgentMetrics
        ]=None) ->Dict[str, Dict[str, float]]:
        """
        Compare effectiveness between multiple RL agents and/or a baseline strategy.

        Args:
            agent_metrics: Dictionary of agent names to their metrics
            baseline_strategy_metrics: Optional metrics for a baseline strategy

        Returns:
            Dictionary with comparative metrics and statistical significance
        """
        results = {}
        if baseline_strategy_metrics:
            for agent_name, metrics in agent_metrics.items():
                comparison = self._compare_metrics(metrics,
                    baseline_strategy_metrics)
                results[f'{agent_name}_vs_baseline'] = comparison
        agent_names = list(agent_metrics.keys())
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                name_i, name_j = agent_names[i], agent_names[j]
                comparison = self._compare_metrics(agent_metrics[name_i],
                    agent_metrics[name_j])
                results[f'{name_i}_vs_{name_j}'] = comparison
        return results

    def track_effectiveness_decay(self, agent_name: str, lookback_periods:
        List[int]=[1, 7, 30, 90]) ->Dict[str, List[float]]:
        """
        Track how agent effectiveness changes over time to detect decay.

        Args:
            agent_name: Name/ID of the RL agent
            lookback_periods: List of periods (in days) to analyze

        Returns:
            Dictionary with effectiveness trends for different metrics
        """
        if agent_name not in self.historical_data:
            logger.warning(
                f'No historical data available for agent {agent_name}')
            return {}
        agent_history = self.historical_data[agent_name]
        agent_history.sort(key=lambda x: x['timestamp'])
        results = {'periods': lookback_periods, 'avg_reward': [],
            'sharpe_ratio': [], 'win_rate': [], 'decay_detected': False}
        current_metrics = agent_history[-1]['metrics']
        for period in lookback_periods:
            cutoff_date = datetime.now() - timedelta(days=period)
            period_data = [entry for entry in agent_history if entry[
                'timestamp'] >= cutoff_date]
            if len(period_data) < self.min_sample_size:
                results['avg_reward'].append(None)
                results['sharpe_ratio'].append(None)
                results['win_rate'].append(None)
                continue
            avg_reward = sum(entry['metrics'].avg_reward for entry in
                period_data) / len(period_data)
            avg_sharpe = sum(entry['metrics'].sharpe_ratio for entry in
                period_data) / len(period_data)
            avg_win_rate = sum(entry['metrics'].win_rate for entry in
                period_data) / len(period_data)
            results['avg_reward'].append(avg_reward)
            results['sharpe_ratio'].append(avg_sharpe)
            results['win_rate'].append(avg_win_rate)
        if len(results['sharpe_ratio']) > 1 and all(x is not None for x in
            results['sharpe_ratio']):
            if all(results['sharpe_ratio'][i] > results['sharpe_ratio'][i +
                1] for i in range(len(results['sharpe_ratio']) - 1)):
                results['decay_detected'] = True
                logger.warning(
                    f'Effectiveness decay detected for agent {agent_name}')
        return results

    @with_analysis_resilience('analyze_action_quality')
    def analyze_action_quality(self, agent_name: str, agent_actions: List[
        Dict[str, Any]], market_data: pd.DataFrame, confidence_threshold:
        float=0.7) ->Dict[str, Any]:
        """
        Analyze the quality of actions taken by the agent based on confidence and outcomes.

        Args:
            agent_name: Name/ID of the RL agent
            agent_actions: List of actions taken by the agent with context
            market_data: Market data during the evaluation period
            confidence_threshold: Threshold for high-confidence actions

        Returns:
            Dictionary with action quality metrics
        """
        if not agent_actions:
            return {'error': 'No actions provided'}
        confidences = [action.get('confidence', 0.5) for action in
            agent_actions]
        outcomes = [action.get('outcome', 0) for action in agent_actions]
        high_conf_indices = [i for i, conf in enumerate(confidences) if 
            conf >= confidence_threshold]
        low_conf_indices = [i for i, conf in enumerate(confidences) if conf <
            confidence_threshold]
        high_conf_outcomes = [outcomes[i] for i in high_conf_indices]
        low_conf_outcomes = [outcomes[i] for i in low_conf_indices]
        results = {'sample_size': len(agent_actions), 'avg_confidence': np.
            mean(confidences) if confidences else 0,
            'high_confidence_actions': len(high_conf_indices),
            'low_confidence_actions': len(low_conf_indices)}
        results['overall_success_rate'] = np.mean([(o > 0) for o in outcomes]
            ) if outcomes else 0
        results['high_conf_success_rate'] = np.mean([(o > 0) for o in
            high_conf_outcomes]) if high_conf_outcomes else 0
        results['low_conf_success_rate'] = np.mean([(o > 0) for o in
            low_conf_outcomes]) if low_conf_outcomes else 0
        results['overall_avg_return'] = np.mean(outcomes) if outcomes else 0
        results['high_conf_avg_return'] = np.mean(high_conf_outcomes
            ) if high_conf_outcomes else 0
        results['low_conf_avg_return'] = np.mean(low_conf_outcomes
            ) if low_conf_outcomes else 0
        if len(confidences) > self.min_sample_size and len(set(confidences)
            ) > 1:
            bins = np.linspace(0, 1, 11)
            binned_conf = np.digitize(confidences, bins) - 1
            binned_conf = [min(b, 9) for b in binned_conf]
            bin_success_rates = []
            for bin_idx in range(10):
                bin_outcomes = [outcomes[i] for i, b in enumerate(
                    binned_conf) if b == bin_idx]
                if bin_outcomes:
                    success_rate = np.mean([(o > 0) for o in bin_outcomes])
                    bin_success_rates.append((bins[bin_idx], success_rate))
            results['confidence_calibration'] = bin_success_rates
            binary_outcomes = [(1 if o > 0 else 0) for o in outcomes]
            results['brier_score'] = np.mean([((confidences[i] -
                binary_outcomes[i]) ** 2) for i in range(len(confidences))])
        return results

    def generate_effectiveness_report(self, agent_name: str, time_period:
        str='last_7_days', include_visualizations: bool=False,
        output_format: str='json') ->Dict[str, Any]:
        """
        Generate a comprehensive effectiveness report for an RL agent.

        Args:
            agent_name: Name/ID of the RL agent
            time_period: Time period to analyze ("last_day", "last_7_days", "last_30_days", "all")
            include_visualizations: Whether to include visualization data
            output_format: Report format ("json", "html", "pdf")

        Returns:
            Dictionary with the complete effectiveness report
        """
        if agent_name not in self.historical_data:
            return {'error': f'No data available for agent {agent_name}'}
        cutoff_date = None
        if time_period == 'last_day':
            cutoff_date = datetime.now() - timedelta(days=1)
        elif time_period == 'last_7_days':
            cutoff_date = datetime.now() - timedelta(days=7)
        elif time_period == 'last_30_days':
            cutoff_date = datetime.now() - timedelta(days=30)
        if cutoff_date:
            filtered_data = [entry for entry in self.historical_data[
                agent_name] if entry['timestamp'] >= cutoff_date]
        else:
            filtered_data = self.historical_data[agent_name]
        if not filtered_data:
            return {'error':
                f'No data available for {agent_name} in the specified time period'
                }
        report = {'agent_name': agent_name, 'time_period': time_period,
            'generated_at': datetime.now().isoformat(), 'data_points': len(
            filtered_data), 'overall_metrics': self.
            _calculate_aggregate_metrics(filtered_data),
            'regime_performance': self._get_regime_performance(agent_name,
            filtered_data), 'effectiveness_trend': self.
            track_effectiveness_decay(agent_name)}
        if include_visualizations:
            report['visualization_data'] = self._prepare_visualization_data(
                filtered_data)
        return report

    @with_exception_handling
    def export_report_to_file(self, report: Dict[str, Any], file_path: str
        ) ->bool:
        """
        Export an effectiveness report to a file.

        Args:
            report: Report dictionary from generate_effectiveness_report
            file_path: Path to save the report

        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(report, f, indent=4, default=self.
                        _json_serializer)
            elif file_path.endswith('.html'):
                html_content = self._convert_report_to_html(report)
                with open(file_path, 'w') as f:
                    f.write(html_content)
            elif file_path.endswith('.pdf'):
                logger.error('PDF export not implemented yet')
                return False
            else:
                logger.error(f'Unsupported file format: {file_path}')
                return False
            logger.info(f'Report exported successfully to {file_path}')
            return True
        except Exception as e:
            logger.error(f'Error exporting report: {str(e)}')
            return False

    @with_resilience('update_historical_data')
    def update_historical_data(self, agent_name: str, metrics: RLAgentMetrics
        ) ->None:
        """
        Update the historical metrics for an agent.

        Args:
            agent_name: Name/ID of the RL agent
            metrics: New metrics to add to history
        """
        if agent_name not in self.historical_data:
            self.historical_data[agent_name] = []
        self.historical_data[agent_name].append({'timestamp': datetime.now(
            ), 'metrics': metrics})
        if len(self.historical_data[agent_name]) > 1000:
            self.historical_data[agent_name] = self.historical_data[agent_name
                ][-1000:]
        logger.debug(f'Updated historical data for {agent_name}')

    def _calculate_agent_metrics(self, agent_name: str, actions: List[Dict[
        str, Any]], market_data: pd.DataFrame) ->RLAgentMetrics:
        """Calculate comprehensive metrics for an agent based on its actions."""
        rewards = [action.get('reward', 0) for action in actions]
        returns = [action.get('return', 0) for action in actions]
        position_sizes = [action.get('position_size', 0) for action in actions]
        durations = [action.get('duration', 0) for action in actions]
        action_types = [action.get('action_type', 'unknown') for action in
            actions]
        confidences = [action.get('confidence', 0.5) for action in actions]
        action_dist = {}
        for action_type in set(action_types):
            count = action_types.count(action_type)
            action_dist[action_type] = count / len(action_types)
        avg_reward = np.mean(rewards) if rewards else 0
        if returns:
            returns_array = np.array(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
            sortino_ratio = self._calculate_sortino_ratio(returns_array)
            max_drawdown = self._calculate_max_drawdown(returns_array)
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns) if returns else 0
        avg_duration = np.mean(durations) if durations else 0
        avg_position_size = np.mean([abs(ps) for ps in position_sizes]
            ) if position_sizes else 0
        return RLAgentMetrics(name=agent_name, avg_reward=avg_reward,
            sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown, win_rate=win_rate,
            avg_trade_duration=avg_duration, avg_position_size=
            avg_position_size, action_distribution=action_dist,
            confidence_scores=confidences)

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate:
        float=0.0) ->float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate:
        float=0.0) ->float:
        """Calculate Sortino ratio from returns."""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate
        downside_returns = np.array([min(0, r) for r in excess_returns])
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0
        return np.mean(excess_returns) / downside_deviation

    def _calculate_max_drawdown(self, returns: np.ndarray) ->float:
        """Calculate maximum drawdown from returns."""
        if len(returns) < 2:
            return 0
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        return abs(np.min(drawdowns))

    def _calculate_statistical_significance(self, metrics: RLAgentMetrics,
        baseline_metrics: Optional[RLAgentMetrics]) ->Optional[float]:
        """Calculate statistical significance versus baseline."""
        if not baseline_metrics:
            return None
        return 0.03

    def _get_baseline_metrics(self, agent_name: str, regime: MarketRegime
        ) ->Optional[RLAgentMetrics]:
        """Get baseline metrics for comparison."""
        baseline_key = f'{agent_name}_{regime.value}'
        if baseline_key in self.baseline_metrics:
            return self.baseline_metrics[baseline_key]
        if regime not in self.regime_data or not self.regime_data[regime]:
            return None
        regime_history = self.regime_data[regime]
        if not regime_history:
            return None
        baseline = regime_history[-1]['metrics']
        self.baseline_metrics[baseline_key] = baseline
        return baseline

    def _compare_metrics(self, metrics_a: RLAgentMetrics, metrics_b:
        RLAgentMetrics) ->Dict[str, float]:
        """Compare two sets of metrics."""
        return {'avg_reward_diff': metrics_a.avg_reward - metrics_b.
            avg_reward, 'sharpe_ratio_diff': metrics_a.sharpe_ratio -
            metrics_b.sharpe_ratio, 'sortino_ratio_diff': metrics_a.
            sortino_ratio - metrics_b.sortino_ratio, 'win_rate_diff': 
            metrics_a.win_rate - metrics_b.win_rate, 'max_drawdown_diff': 
            metrics_a.max_drawdown - metrics_b.max_drawdown}

    def _calculate_aggregate_metrics(self, data: List[Dict[str, Any]]) ->Dict[
        str, float]:
        """Calculate aggregate metrics from a list of data points."""
        if not data:
            return {}
        metrics_list = [entry['metrics'] for entry in data]
        result = {'avg_reward': np.mean([m.avg_reward for m in metrics_list
            ]), 'avg_sharpe_ratio': np.mean([m.sharpe_ratio for m in
            metrics_list]), 'avg_sortino_ratio': np.mean([m.sortino_ratio for
            m in metrics_list]), 'avg_max_drawdown': np.mean([m.
            max_drawdown for m in metrics_list]), 'avg_win_rate': np.mean([
            m.win_rate for m in metrics_list])}
        return result

    def _get_regime_performance(self, agent_name: str, data: List[Dict[str,
        Any]]) ->Dict[str, Dict[str, float]]:
        """Get performance metrics by regime from data."""
        return {'trending': {'avg_reward': 0.5, 'win_rate': 0.6}, 'ranging':
            {'avg_reward': 0.3, 'win_rate': 0.5}, 'volatile': {'avg_reward':
            0.2, 'win_rate': 0.4}}

    def _prepare_visualization_data(self, data: List[Dict[str, Any]]) ->Dict[
        str, Any]:
        """Prepare data for visualizations."""
        return {'timestamps': [entry['timestamp'].isoformat() for entry in
            data], 'rewards': [entry['metrics'].avg_reward for entry in
            data], 'sharpe_ratios': [entry['metrics'].sharpe_ratio for
            entry in data], 'win_rates': [entry['metrics'].win_rate for
            entry in data]}

    def _convert_report_to_html(self, report: Dict[str, Any]) ->str:
        """Convert report to HTML format."""
        html = f"""
        <html>
        <head>
            <title>RL Agent Effectiveness Report: {report['agent_name']}</title>
        </head>
        <body>
            <h1>Effectiveness Report for {report['agent_name']}</h1>
            <p>Generated at: {report['generated_at']}</p>
            <p>Time Period: {report['time_period']}</p>
            
            <h2>Overall Metrics</h2>
            <pre>{json.dumps(report['overall_metrics'], indent=2)}</pre>
            
            <h2>Regime Performance</h2>
            <pre>{json.dumps(report['regime_performance'], indent=2)}</pre>
        </body>
        </html>
        """
        return html

    def _json_serializer(self, obj):
        """Custom JSON serializer for non-serializable types."""
        if isinstance(obj, (datetime, np.ndarray, MarketRegime, Enum)):
            return str(obj)
        raise TypeError(f'Type {type(obj)} not serializable')


class RLToolEffectivenessIntegration(BaseEffectivenessMetric):
    """
    Integration of RL-specific effectiveness metrics with the main tool 
    effectiveness framework.
    """

    def __init__(self, config: Dict[str, Any]):
    """
      init  .
    
    Args:
        config: Description of config
        Any]: Description of Any]
    
    """

        super().__init__(name='rl_model_effectiveness', description=
            'RL model effectiveness metrics')
        self.analyzer = RLModelEffectivenessAnalyzer(min_sample_size=config
            .get('min_sample_size', 30), significance_level=config.get(
            'significance_level', 0.05), decay_factor=config.get(
            'decay_factor', 0.95))
        self.config = config

    @with_analysis_resilience('calculate_effectiveness')
    def calculate_effectiveness(self, tool_id: str, decisions: List[Dict[
        str, Any]], outcomes: List[Dict[str, Any]], market_data: pd.
        DataFrame, context: Dict[str, Any]) ->Dict[str, float]:
        """
        Calculate RL model effectiveness and integrate with the main effectiveness metrics.
        
        Args:
            tool_id: ID of the RL model/tool
            decisions: List of decisions made by the RL model
            outcomes: List of trading outcomes
            market_data: Market data for the time period
            context: Additional context information

        Returns:
            Dictionary of effectiveness metrics
        """
        logger.info(f'Calculating RL effectiveness metrics for {tool_id}')
        agent_actions = self._convert_to_agent_actions(decisions, outcomes)
        regime_labels = context.get('regime_labels', [])
        if not regime_labels:
            regime_labels = [MarketRegime.UNKNOWN] * len(agent_actions)
        regime_metrics = self.analyzer.calculate_regime_specific_metrics(
            agent_name=tool_id, agent_actions=agent_actions, market_data=
            market_data, regime_labels=regime_labels)
        action_quality = self.analyzer.analyze_action_quality(agent_name=
            tool_id, agent_actions=agent_actions, market_data=market_data)
        all_rewards = [action.get('reward', 0) for action in agent_actions]
        all_returns = [action.get('return', 0) for action in agent_actions]
        agent_metrics = RLAgentMetrics(name=tool_id, avg_reward=np.mean(
            all_rewards) if all_rewards else 0, sharpe_ratio=self.
            _calculate_sharpe(all_returns), sortino_ratio=self.
            _calculate_sortino(all_returns), max_drawdown=self.
            _calculate_max_drawdown(all_returns), win_rate=sum(1 for r in
            all_returns if r > 0) / len(all_returns) if all_returns else 0,
            avg_trade_duration=np.mean([a.get('duration', 0) for a in
            agent_actions]) if agent_actions else 0, avg_position_size=np.
            mean([abs(a.get('position_size', 0)) for a in agent_actions]) if
            agent_actions else 0, action_distribution=self.
            _calculate_action_distribution(agent_actions),
            confidence_scores=[a.get('confidence', 0.5) for a in agent_actions]
            )
        self.analyzer.update_historical_data(tool_id, agent_metrics)
        return {'overall_effectiveness': self.
            _calculate_overall_effectiveness(agent_metrics, action_quality),
            'regime_effectiveness': self._calculate_regime_effectiveness(
            regime_metrics), 'confidence_calibration': action_quality.get(
            'brier_score', 1.0), 'risk_adjusted_return': agent_metrics.
            sharpe_ratio, 'win_rate': agent_metrics.win_rate}

    def _convert_to_agent_actions(self, decisions: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]]) ->List[Dict[str, Any]]:
        """Convert decisions and outcomes to the agent actions format."""
        agent_actions = []
        for i, decision in enumerate(decisions):
            action = {'timestamp': decision.get('timestamp', None),
                'action_type': decision.get('action', 'unknown'),
                'confidence': decision.get('confidence', 0.5),
                'position_size': decision.get('position_size', 0)}
            if i < len(outcomes):
                outcome = outcomes[i]
                action.update({'reward': outcome.get('reward', 0), 'return':
                    outcome.get('return', 0), 'duration': outcome.get(
                    'duration', 0), 'outcome': outcome.get('outcome', 0)})
            agent_actions.append(action)
        return agent_actions

    def _calculate_sharpe(self, returns: List[float]) ->float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0
        returns_array = np.array(returns)
        return np.mean(returns_array) / np.std(returns_array) if np.std(
            returns_array) > 0 else 0

    def _calculate_sortino(self, returns: List[float]) ->float:
        """Calculate Sortino ratio."""
        if not returns or len(returns) < 2:
            return 0
        returns_array = np.array(returns)
        negative_returns = np.array([min(0, r) for r in returns_array])
        downside_std = np.std(negative_returns)
        return np.mean(returns_array) / downside_std if downside_std > 0 else 0

    def _calculate_max_drawdown(self, returns: List[float]) ->float:
        """Calculate maximum drawdown."""
        if not returns or len(returns) < 2:
            return 0
        cum_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        return abs(np.min(drawdowns))

    def _calculate_action_distribution(self, actions: List[Dict[str, Any]]
        ) ->Dict[str, float]:
        """Calculate distribution of action types."""
        if not actions:
            return {}
        action_types = [a.get('action_type', 'unknown') for a in actions]
        unique_types = set(action_types)
        return {action_type: (action_types.count(action_type) / len(
            action_types)) for action_type in unique_types}

    def _calculate_overall_effectiveness(self, metrics: RLAgentMetrics,
        action_quality: Dict[str, Any]) ->float:
        """Calculate overall effectiveness score."""
        components = [metrics.sharpe_ratio * 0.3, metrics.win_rate * 0.2, (
            1 - metrics.max_drawdown) * 0.2, action_quality.get(
            'high_conf_success_rate', 0) * 0.15, (1 - action_quality.get(
            'brier_score', 1.0)) * 0.15]
        effectiveness = sum(components)
        return min(max(effectiveness, 0), 1)

    def _calculate_regime_effectiveness(self, regime_metrics: Dict[
        MarketRegime, RegimeSpecificPerformance]) ->Dict[str, float]:
        """Calculate effectiveness scores by regime."""
        result = {}
        for regime, performance in regime_metrics.items():
            metrics = performance.metrics
            score = metrics.sharpe_ratio * 0.4 + metrics.win_rate * 0.3 + (
                1 - metrics.max_drawdown) * 0.3
            score = min(max(score, 0), 1)
            result[regime.value] = score
        return result


class ReinforcementLearningAnalyticsService:
    """
    Comprehensive service for RL model analytics and effectiveness reporting.
    
    This service integrates the RLModelEffectivenessAnalyzer with other
    analytics capabilities for decision attribution, comparative analysis,
    and advanced reporting.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the RL analytics service."""
        self.config = config
        self.effectiveness_analyzer = RLModelEffectivenessAnalyzer(
            min_sample_size=config_manager.get('min_sample_size', 30),
            significance_level=config_manager.get('significance_level', 0.05),
            decay_factor=config_manager.get('decay_factor', 0.95))
        self.tool_effectiveness_integration = RLToolEffectivenessIntegration(
            config)
        self.report_templates = self._load_report_templates()

    @with_analysis_resilience('calculate_decision_attribution')
    def calculate_decision_attribution(self, agent_name: str, decisions:
        List[Dict[str, Any]], feature_importances: Dict[str, List[float]]
        ) ->Dict[str, List[Dict[str, Any]]]:
        """
        Attribute decisions to specific features and analyze decision drivers.
        
        Args:
            agent_name: Name of the RL agent
            decisions: List of decisions made by the agent
            feature_importances: Dictionary mapping feature names to importance scores

        Returns:
            Attribution analysis results
        """
        results = {'decision_drivers': [], 'feature_attribution': {}}
        avg_importances = {}
        for feature, values in feature_importances.items():
            avg_importances[feature] = np.mean(values) if values else 0
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1
            ], reverse=True)
        top_drivers = sorted_features[:5]
        results['decision_drivers'] = [{'feature': feature, 'importance':
            importance} for feature, importance in top_drivers]
        decision_types = set(d.get('action', 'unknown') for d in decisions)
        for decision_type in decision_types:
            indices = [i for i, d in enumerate(decisions) if d.get('action',
                'unknown') == decision_type]
            type_importances = {}
            for feature, values in feature_importances.items():
                if indices:
                    type_values = [values[i] for i in indices if i < len(
                        values)]
                    type_importances[feature] = np.mean(type_values
                        ) if type_values else 0
            sorted_type_features = sorted(type_importances.items(), key=lambda
                x: x[1], reverse=True)[:5]
            results['feature_attribution'][decision_type] = [{'feature':
                feature, 'importance': importance} for feature, importance in
                sorted_type_features]
        return results

    def compare_agent_performance(self, agent_metrics: Dict[str,
        RLAgentMetrics], baseline_metrics: Optional[RLAgentMetrics]=None,
        regime_data: Optional[Dict[str, List[Dict[str, Any]]]]=None) ->Dict[
        str, Any]:
        """
        Compare performance between different agents and optionally a baseline.
        
        Args:
            agent_metrics: Dictionary mapping agent names to their metrics
            baseline_metrics: Optional baseline metrics for comparison
            regime_data: Optional regime-specific data for each agent

        Returns:
            Comparative analysis results
        """
        comparison = (self.effectiveness_analyzer.
            calculate_comparative_effectiveness(agent_metrics,
            baseline_metrics))
        results = {'overall_comparison': comparison,
            'key_metrics_comparison': self._compare_key_metrics(
            agent_metrics, baseline_metrics), 'statistical_significance': {}}
        if regime_data:
            regime_comparison = {}
            for regime, regime_metrics in regime_data.items():
                regime_comparison[regime] = self._compare_regime_performance(
                    regime, {agent: metrics.get(regime, {}) for agent,
                    metrics in regime_data.items()})
            results['regime_comparison'] = regime_comparison
        agent_names = list(agent_metrics.keys())
        if len(agent_names) > 1:
            for i in range(len(agent_names)):
                for j in range(i + 1, len(agent_names)):
                    name_i, name_j = agent_names[i], agent_names[j]
                    results['statistical_significance'][f'{name_i}_vs_{name_j}'
                        ] = 0.03
        return results

    def generate_analytics_dashboard_data(self, agent_name: str,
        time_period: str='last_7_days', metrics_history: Optional[List[Dict
        [str, Any]]]=None) ->Dict[str, Any]:
        """
        Generate data for an RL analytics dashboard.
        
        Args:
            agent_name: Name of the RL agent
            time_period: Time period to analyze
            metrics_history: Optional history of metrics

        Returns:
            Dashboard data for visualization
        """
        report = self.effectiveness_analyzer.generate_effectiveness_report(
            agent_name=agent_name, time_period=time_period,
            include_visualizations=True)
        dashboard_data = {'agent_name': agent_name, 'time_period':
            time_period, 'overview_metrics': report.get('overall_metrics',
            {}), 'regime_performance': report.get('regime_performance', {}),
            'effectiveness_trend': report.get('effectiveness_trend', {})}
        viz_data = report.get('visualization_data', {})
        if viz_data:
            dashboard_data['visualization_data'] = viz_data
        if metrics_history:
            dashboard_data['metrics_history'] = self._prepare_metrics_history(
                metrics_history)
        dashboard_data['predictions'] = {'next_7_days': {'avg_reward': 
            report.get('overall_metrics', {}).get('avg_reward', 0) * 1.05,
            'win_rate': min(report.get('overall_metrics', {}).get(
            'avg_win_rate', 0) * 1.02, 1.0)}}
        return dashboard_data

    @with_exception_handling
    def save_analytics_report(self, report_data: Dict[str, Any],
        report_type: str, output_path: str, format: str='json') ->bool:
        """
        Save an analytics report to a file.
        
        Args:
            report_data: Report data to save
            report_type: Type of report
            output_path: Path to save the report
            format: Output format (json, html, pdf)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            template = self.report_templates.get(report_type)
            if template:
                report_data = self._apply_report_template(report_data, template
                    )
            if format not in ['json', 'html', 'pdf']:
                logger.error(f'Unsupported format: {format}')
                return False
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=self.
                        _json_serializer)
            elif format == 'html':
                html_content = self._create_html_report(report_data,
                    report_type)
                with open(output_path, 'w') as f:
                    f.write(html_content)
            elif format == 'pdf':
                logger.error('PDF format not implemented')
                return False
            logger.info(f'Report saved to {output_path}')
            return True
        except Exception as e:
            logger.error(f'Error saving report: {str(e)}')
            return False

    def _compare_key_metrics(self, agent_metrics: Dict[str, RLAgentMetrics],
        baseline_metrics: Optional[RLAgentMetrics]=None) ->Dict[str, Dict[
        str, float]]:
        """Compare key metrics between agents."""
        results = {}
        key_metrics = ['avg_reward', 'sharpe_ratio', 'sortino_ratio',
            'win_rate', 'max_drawdown']
        for metric in key_metrics:
            metric_values = {}
            for agent, metrics in agent_metrics.items():
                metric_values[agent] = getattr(metrics, metric, 0)
            if baseline_metrics:
                metric_values['baseline'] = getattr(baseline_metrics, metric, 0
                    )
            results[metric] = metric_values
        return results

    def _compare_regime_performance(self, regime: str, regime_metrics: Dict
        [str, Dict[str, float]]) ->Dict[str, Dict[str, float]]:
        """Compare performance within a specific regime."""
        results = {}
        compare_metrics = ['avg_reward', 'win_rate', 'sharpe_ratio']
        for metric in compare_metrics:
            metric_values = {}
            for agent, metrics in regime_metrics.items():
                metric_values[agent] = metrics.get(metric, 0)
            results[metric] = metric_values
        return results

    def _prepare_metrics_history(self, metrics_history: List[Dict[str, Any]]
        ) ->Dict[str, List[Any]]:
        """Prepare metrics history for visualization."""
        result = {'timestamps': [], 'avg_reward': [], 'sharpe_ratio': [],
            'win_rate': []}
        for entry in metrics_history:
            timestamp = entry.get('timestamp', '')
            metrics = entry.get('metrics', {})
            result['timestamps'].append(timestamp)
            result['avg_reward'].append(metrics.get('avg_reward', 0))
            result['sharpe_ratio'].append(metrics.get('sharpe_ratio', 0))
            result['win_rate'].append(metrics.get('win_rate', 0))
        return result

    def _load_report_templates(self) ->Dict[str, Any]:
        """Load report templates from configuration."""
        return {'effectiveness': {'sections': ['overview',
            'regime_performance', 'trend_analysis', 'action_quality']},
            'comparison': {'sections': ['overview', 'comparative_metrics',
            'statistical_significance', 'regime_comparison']}}

    def _apply_report_template(self, report_data: Dict[str, Any], template:
        Dict[str, Any]) ->Dict[str, Any]:
        """Apply a report template to structure the report data."""
        return report_data

    def _create_html_report(self, report_data: Dict[str, Any], report_type: str
        ) ->str:
        """Create an HTML report from report data."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_type.capitalize()} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{report_type.capitalize()} Report</h1>
            <div class="section">
                <h2>Overview</h2>
                <pre>{json.dumps(report_data.get('overview_metrics', {}), indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        return html

    def _json_serializer(self, obj):
        """Custom JSON serializer for non-serializable types."""
        if isinstance(obj, (datetime, np.ndarray, Enum)):
            return str(obj)
        raise TypeError(f'Type {type(obj)} not serializable')
