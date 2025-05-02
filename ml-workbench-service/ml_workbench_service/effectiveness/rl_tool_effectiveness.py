"""
Tool Effectiveness Framework for RL Models

This module provides tools and analysis for evaluating and tracking the effectiveness
of RL models in forex trading. It extends the basic tool effectiveness framework
with RL-specific metrics and insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from scipy import stats

from core_foundations.utils.logger import get_logger
from trading_gateway_service.simulation.forex_broker_simulator import MarketRegimeType
from risk_management_service.models.risk_metrics import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_value_at_risk, calculate_expected_shortfall
)

logger = get_logger(__name__)


class EffectivenessMetricType(Enum):
    """Types of effectiveness metrics for RL models."""
    REWARD = "reward"
    PNL = "pnl"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    AVERAGE_TRADE = "average_trade"
    TRADE_FREQUENCY = "trade_frequency"
    VAR = "value_at_risk"
    REGIME_ADAPTABILITY = "regime_adaptability"
    NEWS_SENSITIVITY = "news_sensitivity"


@dataclass
class RLModelAction:
    """Represents an action taken by an RL model."""
    timestamp: datetime
    state: Dict[str, float]
    action_values: Dict[str, float]
    action_taken: Any
    reward: float
    next_state: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionQualityAssessment:
    """Assessment of a single RL action quality."""
    action: RLModelAction
    optimal_action: Optional[Any] = None
    optimality_score: float = 0.0  # 0.0 (worst) to 1.0 (optimal)
    opportunity_cost: float = 0.0  # How much reward was missed
    explanation: str = ""
    factors: Dict[str, float] = field(default_factory=dict)


class RLToolEffectiveness:
    """
    Framework for evaluating the effectiveness of RL trading models.
    This extends the general tool effectiveness framework with RL-specific
    metrics and analysis capabilities.
    """
    
    def __init__(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        baseline_model_id: Optional[str] = None
    ):
        """
        Initialize the RL Tool Effectiveness framework.
        
        Args:
            model_id: Unique identifier for the RL model
            model_name: Human-readable name for the model
            model_version: Version of the model
            baseline_model_id: Optional baseline model for comparison
        """
        self.model_id = model_id
        self.model_name = model_name
        self.model_version = model_version
        self.baseline_model_id = baseline_model_id
        
        # Storage for effectiveness metrics
        self.effectiveness_metrics = {}
        self.effectiveness_history = []
        
        # Storage for action histories and assessments
        self.action_history = []
        self.action_assessments = []
        
        # Decay factors for effectiveness over time
        self.metric_halflife = {
            EffectivenessMetricType.REWARD: timedelta(days=30),
            EffectivenessMetricType.PNL: timedelta(days=30),
            EffectivenessMetricType.SHARPE_RATIO: timedelta(days=60),
            EffectivenessMetricType.REGIME_ADAPTABILITY: timedelta(days=90),
        }
        
        # Regime-specific performance tracking
        self.regime_performance = {regime: {} for regime in MarketRegimeType}
        
    def record_action(self, action: RLModelAction) -> None:
        """
        Record an action taken by the RL model for later assessment.
        
        Args:
            action: The RL model action details
        """
        self.action_history.append(action)
        
        # If we have too many actions, remove oldest ones
        if len(self.action_history) > 10000:
            self.action_history = self.action_history[-10000:]
            
    def assess_action_quality(self, action: RLModelAction) -> ActionQualityAssessment:
        """
        Assess the quality of a single RL model action.
        
        Args:
            action: The RL model action to assess
            
        Returns:
            Assessment of the action quality
        """
        # Determine if this was the optimal action by analyzing the action values
        if 'action_values' in action.metadata and len(action.metadata['action_values']) > 0:
            optimal_action_value = max(action.metadata['action_values'].items(), key=lambda x: x[1])
            optimal_action = optimal_action_value[0]
            max_value = optimal_action_value[1]
            
            # Get the value of the action actually taken
            taken_value = action.metadata['action_values'].get(
                str(action.action_taken),
                list(action.metadata['action_values'].values())[0]  # Default to first value
            )
            
            # Calculate optimality score (how close to optimal was the action)
            value_range = max(1e-6, max_value - min(action.metadata['action_values'].values()))
            optimality_score = 1.0 - abs(max_value - taken_value) / value_range
            
            # Calculate opportunity cost
            opportunity_cost = max_value - taken_value
            
            # Create explanation
            explanation = f"Action {action.action_taken} chosen. "
            if optimal_action != str(action.action_taken):
                explanation += f"Optimal would have been {optimal_action} with value {max_value:.4f} vs {taken_value:.4f}."
            else:
                explanation += "This was the optimal action."
                
            # Create assessment with factors
            assessment = ActionQualityAssessment(
                action=action,
                optimal_action=optimal_action,
                optimality_score=optimality_score,
                opportunity_cost=opportunity_cost,
                explanation=explanation,
                factors={
                    'value_confidence': taken_value / max(1e-6, max_value),
                    'exploration_ratio': action.metadata.get('exploration_ratio', 0.0),
                    'action_entropy': action.metadata.get('action_entropy', 0.0),
                }
            )
        else:
            # Limited assessment without action values
            assessment = ActionQualityAssessment(
                action=action,
                optimality_score=0.5,  # Neutral score
                explanation="Limited assessment: no action values available."
            )
            
        # Add assessment to history
        self.action_assessments.append(assessment)
        
        # Trim history if needed
        if len(self.action_assessments) > 10000:
            self.action_assessments = self.action_assessments[-10000:]
            
        return assessment
        
    def update_effectiveness_metrics(
        self,
        episode_metrics: Dict[str, float],
        market_regime: MarketRegimeType,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update effectiveness metrics with new episode results.
        
        Args:
            episode_metrics: Metrics from an episode or evaluation period
            market_regime: The market regime during this episode
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Process the metrics
        for key, value in episode_metrics.items():
            try:
                metric_type = EffectivenessMetricType(key)
            except ValueError:
                # Skip metrics that don't match our enum types
                continue
                
            # Update overall effectiveness metrics
            if key not in self.effectiveness_metrics:
                self.effectiveness_metrics[key] = {
                    'current': value,
                    'history': [],
                    'decay_factor': self._get_decay_factor(metric_type)
                }
            else:
                # Apply exponential decay to old value
                current = self.effectiveness_metrics[key]['current']
                decay = self.effectiveness_metrics[key]['decay_factor']
                self.effectiveness_metrics[key]['current'] = current * (1 - decay) + value * decay
                
            # Add to history
            self.effectiveness_metrics[key]['history'].append({
                'timestamp': timestamp,
                'value': value,
                'decayed_value': self.effectiveness_metrics[key]['current']
            })
            
            # Update regime-specific metrics
            regime_key = market_regime.name
            if regime_key not in self.regime_performance:
                self.regime_performance[regime_key] = {}
                
            if key not in self.regime_performance[regime_key]:
                self.regime_performance[regime_key][key] = {
                    'values': [],
                    'mean': value,
                    'std': 0.0
                }
                
            self.regime_performance[regime_key][key]['values'].append(value)
            
            # Update statistics for this regime
            values = self.regime_performance[regime_key][key]['values']
            self.regime_performance[regime_key][key]['mean'] = np.mean(values)
            self.regime_performance[regime_key][key]['std'] = np.std(values) if len(values) > 1 else 0.0
        
        # Add to effectiveness history
        self.effectiveness_history.append({
            'timestamp': timestamp,
            'metrics': {k: v['current'] for k, v in self.effectiveness_metrics.items()},
            'market_regime': market_regime.name
        })
        
    def _get_decay_factor(self, metric_type: EffectivenessMetricType) -> float:
        """Calculate decay factor for a metric based on its half-life."""
        if metric_type in self.metric_halflife:
            # Use configured half-life
            halflife = self.metric_halflife[metric_type]
            # Convert to decay factor (assuming daily updates)
            days = halflife.total_seconds() / (24 * 3600)
            return 1.0 - 0.5 ** (1.0 / days)
        else:
            # Default decay factor (approximately 30-day half-life)
            return 0.023  # ln(2)/30
    
    def analyze_action_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in the RL model's actions to identify strengths and weaknesses.
        
        Returns:
            Dictionary of pattern analysis results
        """
        if not self.action_assessments:
            return {'error': 'No action assessments available for analysis'}
            
        # Convert assessments to DataFrame for analysis
        data = []
        for assessment in self.action_assessments:
            row = {
                'timestamp': assessment.action.timestamp,
                'optimality_score': assessment.optimality_score,
                'opportunity_cost': assessment.opportunity_cost,
                'reward': assessment.action.reward
            }
            
            # Add action metadata
            if assessment.action.metadata:
                for k, v in assessment.action.metadata.items():
                    if isinstance(v, (int, float, str, bool)):
                        row[f'meta_{k}'] = v
            
            # Add state information (flattened)
            if assessment.action.state:
                for k, v in assessment.action.state.items():
                    if isinstance(v, (int, float)):
                        row[f'state_{k}'] = v
                        
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Skip analysis if dataframe is empty
        if len(df) == 0:
            return {'error': 'No valid data for analysis'}
            
        # Basic effectiveness metrics
        mean_optimality = df['optimality_score'].mean()
        mean_opportunity_cost = df['opportunity_cost'].mean()
        mean_reward = df['reward'].mean()
        
        # Time series analysis
        df['date'] = df['timestamp'].dt.date
        daily_optimality = df.groupby('date')['optimality_score'].mean()
        
        # Detect trend in optimality (learning progress)
        trend = 'stable'
        if len(daily_optimality) > 10:
            try:
                slope, _, _, p_value, _ = stats.linregress(range(len(daily_optimality)), daily_optimality)
                if p_value < 0.05:
                    trend = 'improving' if slope > 0 else 'degrading'
            except:
                pass
                
        # State impact analysis - which state features correlate with optimal decisions?
        state_correlations = {}
        state_columns = [col for col in df.columns if col.startswith('state_')]
        for col in state_columns:
            if df[col].nunique() > 1:  # Skip constant features
                corr = df[col].corr(df['optimality_score'])
                if not pd.isna(corr):
                    state_correlations[col.replace('state_', '')] = corr
        
        # Top positive and negative correlations
        top_positive = {k: v for k, v in sorted(
            state_correlations.items(), key=lambda x: x[1], reverse=True)[:5]}
        top_negative = {k: v for k, v in sorted(
            state_correlations.items(), key=lambda x: x[1])[:5]}
            
        return {
            'mean_optimality_score': mean_optimality,
            'mean_opportunity_cost': mean_opportunity_cost,
            'mean_reward': mean_reward,
            'trend': trend,
            'top_positive_correlations': top_positive,
            'top_negative_correlations': top_negative,
            'sample_size': len(df),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
    def analyze_regime_performance(self) -> Dict[str, Any]:
        """
        Analyze how the RL model performs across different market regimes.
        
        Returns:
            Dictionary of regime-specific performance analysis
        """
        results = {}
        
        for regime, metrics in self.regime_performance.items():
            if not metrics:
                continue
                
            regime_data = {
                'sample_size': len(metrics.get('REWARD', {}).get('values', [])),
            }
            
            # Add key metrics for this regime
            for key in ['REWARD', 'PNL', 'SHARPE_RATIO', 'MAX_DRAWDOWN', 'WIN_RATE']:
                if key in metrics:
                    regime_data[key.lower()] = {
                        'mean': metrics[key]['mean'],
                        'std': metrics[key]['std'],
                        'sample_size': len(metrics[key]['values'])
                    }
            
            results[regime] = regime_data
            
        return {
            'regime_analysis': results,
            'best_regime': self._identify_best_regime(),
            'worst_regime': self._identify_worst_regime()
        }
        
    def _identify_best_regime(self) -> str:
        """Identify the market regime where the model performs best."""
        best_regime = None
        best_score = float('-inf')
        
        for regime, metrics in self.regime_performance.items():
            if 'SHARPE_RATIO' not in metrics or 'PNL' not in metrics:
                continue
                
            # Simple scoring: combine Sharpe and PnL
            sharpe = metrics['SHARPE_RATIO']['mean']
            pnl = metrics['PNL']['mean']
            
            # Ensure sufficient sample size
            sample_size = len(metrics['SHARPE_RATIO']['values'])
            if sample_size < 5:
                continue
                
            # Calculate score (normalized)
            score = 0.7 * (sharpe / 2.0) + 0.3 * (pnl / 100.0)
            
            if score > best_score:
                best_score = score
                best_regime = regime
                
        return best_regime
        
    def _identify_worst_regime(self) -> str:
        """Identify the market regime where the model struggles most."""
        worst_regime = None
        worst_score = float('inf')
        
        for regime, metrics in self.regime_performance.items():
            if 'SHARPE_RATIO' not in metrics or 'PNL' not in metrics:
                continue
                
            # Simple scoring: combine Sharpe and PnL
            sharpe = metrics['SHARPE_RATIO']['mean']
            pnl = metrics['PNL']['mean']
            
            # Ensure sufficient sample size
            sample_size = len(metrics['SHARPE_RATIO']['values'])
            if sample_size < 5:
                continue
                
            # Calculate score (normalized)
            score = 0.7 * (sharpe / 2.0) + 0.3 * (pnl / 100.0)
            
            if score < worst_score:
                worst_score = score
                worst_regime = regime
                
        return worst_regime
        
    def calculate_effectiveness_decay(self) -> Dict[str, float]:
        """
        Calculate how effectiveness has decayed over time for each metric.
        
        Returns:
            Dictionary mapping metric names to decay percentages
        """
        decay_rates = {}
        
        for metric_key, metric_data in self.effectiveness_metrics.items():
            history = metric_data['history']
            if len(history) < 10:  # Need sufficient history
                continue
                
            # Get first 10 values and last 10 values
            first_values = [item['value'] for item in history[:10]]
            last_values = [item['value'] for item in history[-10:]]
            
            # Calculate averages
            first_avg = sum(first_values) / len(first_values)
            last_avg = sum(last_values) / len(last_values)
            
            # Calculate decay as percentage change
            if first_avg != 0:
                decay_pct = (last_avg - first_avg) / abs(first_avg) * 100
            else:
                decay_pct = 0
                
            decay_rates[metric_key] = decay_pct
            
        return decay_rates
        
    def generate_effectiveness_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive effectiveness report for the RL model.
        
        Returns:
            Dictionary containing the effectiveness report
        """
        # Action quality analysis
        action_analysis = self.analyze_action_patterns()
        
        # Regime performance analysis
        regime_analysis = self.analyze_regime_performance()
        
        # Effectiveness decay analysis
        decay_analysis = self.calculate_effectiveness_decay()
        
        # Get current effectiveness metrics
        current_metrics = {k: v['current'] for k, v in self.effectiveness_metrics.items()}
        
        # Generate insights
        insights = self._generate_insights(
            action_analysis, regime_analysis, decay_analysis, current_metrics)
        
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'report_generated_at': datetime.now(),
            'current_metrics': current_metrics,
            'action_analysis': action_analysis,
            'regime_analysis': regime_analysis,
            'effectiveness_decay': decay_analysis,
            'insights': insights,
            'sample_sizes': {
                'actions_recorded': len(self.action_history),
                'actions_assessed': len(self.action_assessments),
                'metrics_history': len(self.effectiveness_history)
            }
        }
        
    def _generate_insights(
        self, 
        action_analysis: Dict[str, Any],
        regime_analysis: Dict[str, Any],
        decay_analysis: Dict[str, float],
        current_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate insights based on the effectiveness analysis."""
        insights = []
        
        # Action quality insights
        if 'mean_optimality_score' in action_analysis:
            score = action_analysis['mean_optimality_score']
            if score > 0.8:
                insights.append(f"Model consistently makes near-optimal decisions with a score of {score:.2f}")
            elif score < 0.5:
                insights.append(f"Model frequently makes suboptimal decisions with a score of {score:.2f}")
                
        if 'trend' in action_analysis:
            if action_analysis['trend'] == 'improving':
                insights.append("Model decision quality is improving over time, indicating successful learning")
            elif action_analysis['trend'] == 'degrading':
                insights.append("Model decision quality is degrading over time, suggesting model drift or changing market dynamics")
                
        # Regime-specific insights
        if 'best_regime' in regime_analysis and regime_analysis['best_regime']:
            insights.append(f"Model performs best in {regime_analysis['best_regime']} market conditions")
            
        if 'worst_regime' in regime_analysis and regime_analysis['worst_regime']:
            insights.append(f"Model struggles most in {regime_analysis['worst_regime']} market conditions")
            
        # Decay insights
        significant_decay = False
        for metric, decay_pct in decay_analysis.items():
            if decay_pct < -20:  # More than 20% decay
                if not significant_decay:
                    insights.append("Model effectiveness shows significant decay in some metrics:")
                    significant_decay = True
                insights.append(f"  - {metric} has declined by {abs(decay_pct):.1f}%")
                
        # Performance insights
        if 'SHARPE_RATIO' in current_metrics:
            sharpe = current_metrics['SHARPE_RATIO']
            if sharpe > 2.0:
                insights.append(f"Excellent risk-adjusted returns with Sharpe ratio of {sharpe:.2f}")
            elif sharpe < 0.5:
                insights.append(f"Poor risk-adjusted returns with Sharpe ratio of {sharpe:.2f}")
                
        if 'WIN_RATE' in current_metrics:
            win_rate = current_metrics['WIN_RATE']
            if win_rate > 0.6:
                insights.append(f"High win rate of {win_rate:.1%}, indicating strong predictive ability")
            elif win_rate < 0.4:
                insights.append(f"Low win rate of {win_rate:.1%}, suggesting refinement needed in trade entry/exit logic")
                
        return insights
        
    def plot_effectiveness_trends(self, metric_keys: List[str] = None) -> None:
        """
        Plot effectiveness trends over time for selected metrics.
        
        Args:
            metric_keys: List of metric keys to plot (defaults to all)
        """
        if not self.effectiveness_history:
            print("No effectiveness history available for plotting.")
            return
            
        # Convert history to DataFrame
        data = []
        for item in self.effectiveness_history:
            row = {'timestamp': item['timestamp'], 'market_regime': item['market_regime']}
            row.update(item['metrics'])
            data.append(row)
            
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        
        # Select metrics to plot
        if metric_keys is None:
            metric_keys = [m.value for m in EffectivenessMetricType]
            
        # Filter to available metrics that exist in the data
        available_metrics = [m for m in metric_keys if m in df.columns]
        
        if not available_metrics:
            print("No metrics available for plotting")
            return
            
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(available_metrics):
            plt.subplot(len(available_metrics), 1, i+1)
            df[metric].plot()
            plt.title(f"{metric} Trend")
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert effectiveness data to a dictionary for serialization."""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'baseline_model_id': self.baseline_model_id,
            'effectiveness_metrics': self.effectiveness_metrics,
            'regime_performance': {
                regime.name if isinstance(regime, MarketRegimeType) else regime: metrics
                for regime, metrics in self.regime_performance.items()
            },
            'effectiveness_history': [
                {
                    'timestamp': item['timestamp'].isoformat(),
                    'metrics': item['metrics'],
                    'market_regime': item['market_regime']
                }
                for item in self.effectiveness_history
            ],
            'last_updated': datetime.now().isoformat()
        }
        
    def save_to_file(self, filepath: str) -> None:
        """Save effectiveness data to a JSON file."""
        data = self.to_dict()
        
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=2)
            
    @classmethod
    def load_from_file(cls, filepath: str) -> 'RLToolEffectiveness':
        """Load effectiveness data from a JSON file."""
        with open(filepath, 'r') as file:
            data = json.load(file)
            
        instance = cls(
            model_id=data['model_id'],
            model_name=data['model_name'],
            model_version=data['model_version'],
            baseline_model_id=data.get('baseline_model_id')
        )
        
        # Restore effectiveness metrics
        instance.effectiveness_metrics = data['effectiveness_metrics']
        
        # Restore regime performance
        instance.regime_performance = {}
        for regime_name, metrics in data['regime_performance'].items():
            try:
                regime = MarketRegimeType(regime_name)
            except ValueError:
                regime = regime_name
                
            instance.regime_performance[regime] = metrics
            
        # Restore effectiveness history with datetime objects
        instance.effectiveness_history = []
        for item in data.get('effectiveness_history', []):
            instance.effectiveness_history.append({
                'timestamp': datetime.fromisoformat(item['timestamp']),
                'metrics': item['metrics'],
                'market_regime': item['market_regime']
            })
            
        return instance
"""
