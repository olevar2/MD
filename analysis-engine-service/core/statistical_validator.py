"""
Statistical validation module for multi-timeframe prediction feedback.

This module provides functions to validate and analyze prediction feedback
across multiple timeframes, including correlation analysis and statistical validation.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging
from enum import Enum
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CorrelationStrength(Enum):
    """Enum for correlation strength classification."""
    NONE = 'none'
    WEAK = 'weak'
    MODERATE = 'moderate'
    STRONG = 'strong'
    VERY_STRONG = 'very_strong'


class StatisticalValidator:
    """
    Provides statistical validation for prediction feedback across timeframes.
    Used to validate parameter adjustments and ensure changes are statistically significant.
    """

    def __init__(self, min_samples: int=20, significance_level: float=0.05,
        correlation_thresholds: Optional[Dict[str, float]]=None):
        """
        Initialize the statistical validator.
        
        Args:
            min_samples: Minimum number of samples required for validation
            significance_level: Statistical significance level (p-value threshold)
            correlation_thresholds: Dictionary defining thresholds for correlation strength classification
        """
        self.min_samples = min_samples
        self.significance_level = significance_level
        self.correlation_thresholds = correlation_thresholds or {'weak': 
            0.2, 'moderate': 0.4, 'strong': 0.7, 'very_strong': 1.0}
        self.logger = logger

    @with_analysis_resilience('validate_prediction_improvements')
    def validate_prediction_improvements(self, baseline_errors: List[float],
        new_errors: List[float]) ->Dict[str, Any]:
        """
        Validate whether a new set of prediction errors shows statistically significant improvement.
        
        Args:
            baseline_errors: List of prediction errors from baseline model
            new_errors: List of prediction errors from improved model
            
        Returns:
            Dictionary with validation results
        """
        if len(baseline_errors) < self.min_samples or len(new_errors
            ) < self.min_samples:
            return {'is_significant': False, 'reason':
                f'Insufficient samples (need at least {self.min_samples})',
                'p_value': None, 'mean_improvement': None}
        baseline_mean = np.mean(baseline_errors)
        new_mean = np.mean(new_errors)
        mean_improvement = baseline_mean - new_mean
        improvement_percentage = (mean_improvement / baseline_mean * 100 if
            baseline_mean != 0 else 0)
        if len(baseline_errors) == len(new_errors):
            t_stat, p_value = stats.ttest_rel(baseline_errors, new_errors)
        else:
            t_stat, p_value = stats.ttest_ind(baseline_errors, new_errors,
                equal_var=False)
        is_significant = (p_value < self.significance_level and 
            mean_improvement > 0)
        return {'is_significant': is_significant, 'p_value': p_value,
            'mean_improvement': mean_improvement, 'improvement_percentage':
            improvement_percentage, 't_statistic': t_stat,
            'baseline_mean_error': baseline_mean, 'new_mean_error': new_mean}

    @with_analysis_resilience('analyze_timeframe_correlations')
    def analyze_timeframe_correlations(self, timeframe_data: Dict[str, List
        [float]]) ->Dict[str, Any]:
        """
        Analyze correlations between prediction errors across different timeframes.
        
        Args:
            timeframe_data: Dictionary mapping timeframes to lists of prediction errors
            
        Returns:
            Dictionary with correlation analysis results
        """
        if not timeframe_data or len(timeframe_data) < 2:
            return {'error':
                'Need at least two timeframes for correlation analysis'}
        for timeframe, errors in timeframe_data.items():
            if len(errors) < self.min_samples:
                return {'error':
                    f'Insufficient samples for timeframe {timeframe} (has {len(errors)}, need {self.min_samples})'
                    }
        df = pd.DataFrame(timeframe_data)
        correlation_matrix = df.corr(method='pearson')
        pairwise_correlations = []
        timeframes = list(timeframe_data.keys())
        for i in range(len(timeframes)):
            for j in range(i + 1, len(timeframes)):
                tf1 = timeframes[i]
                tf2 = timeframes[j]
                corr_value = correlation_matrix.loc[tf1, tf2]
                _, p_value = stats.pearsonr(df[tf1].dropna(), df[tf2].dropna())
                strength = self._classify_correlation_strength(abs(corr_value))
                pairwise_correlations.append({'timeframe1': tf1,
                    'timeframe2': tf2, 'correlation': corr_value,
                    'abs_correlation': abs(corr_value), 'p_value': p_value,
                    'is_significant': p_value < self.significance_level,
                    'strength': strength.value, 'direction': 'positive' if 
                    corr_value > 0 else 'negative'})
        pairwise_correlations.sort(key=lambda x: x['abs_correlation'],
            reverse=True)
        strongest_pairs = pairwise_correlations[:3] if len(
            pairwise_correlations) >= 3 else pairwise_correlations
        significant_correlations = [pc for pc in pairwise_correlations if
            pc['is_significant']]
        positive_correlations = [pc for pc in pairwise_correlations if pc[
            'correlation'] > 0]
        negative_correlations = [pc for pc in pairwise_correlations if pc[
            'correlation'] < 0]
        return {'correlation_matrix': correlation_matrix.to_dict(),
            'pairwise_correlations': pairwise_correlations,
            'strongest_correlations': strongest_pairs, 'summary': {
            'total_pairs': len(pairwise_correlations), 'significant_pairs':
            len(significant_correlations), 'positive_pairs': len(
            positive_correlations), 'negative_pairs': len(
            negative_correlations), 'avg_correlation_strength': np.mean([pc
            ['abs_correlation'] for pc in pairwise_correlations])}}

    @with_resilience('validate_parameter_adjustment')
    @with_exception_handling
    def validate_parameter_adjustment(self, before_performance: List[Dict[
        str, Any]], after_performance: List[Dict[str, Any]], metric_name: str
        ) ->Dict[str, Any]:
        """
        Validate whether a parameter adjustment resulted in statistically significant improvement.
        
        Args:
            before_performance: List of performance metrics before adjustment
            after_performance: List of performance metrics after adjustment
            metric_name: Name of the metric to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            before_values = [item[metric_name] for item in
                before_performance if metric_name in item]
            after_values = [item[metric_name] for item in after_performance if
                metric_name in item]
        except (KeyError, TypeError) as e:
            return {'error':
                f"Error extracting metric '{metric_name}': {str(e)}",
                'is_valid': False}
        if len(before_values) < self.min_samples or len(after_values
            ) < self.min_samples:
            return {'is_valid': False, 'reason':
                f'Insufficient samples (need at least {self.min_samples})',
                'before_count': len(before_values), 'after_count': len(
                after_values)}
        before_mean = np.mean(before_values)
        after_mean = np.mean(after_values)
        before_std = np.std(before_values)
        after_std = np.std(after_values)
        higher_is_better = metric_name in ['profit_factor', 'win_rate',
            'profit', 'return', 'sharpe_ratio']
        if higher_is_better:
            improvement = after_mean - before_mean
            improvement_pct = (improvement / before_mean * 100 if 
                before_mean != 0 else 0)
            is_improvement = improvement > 0
        else:
            improvement = before_mean - after_mean
            improvement_pct = (improvement / before_mean * 100 if 
                before_mean != 0 else 0)
            is_improvement = improvement > 0
        t_stat, p_value = stats.ttest_ind(before_values, after_values,
            equal_var=False)
        pooled_std = np.sqrt((before_std ** 2 + after_std ** 2) / 2)
        effect_size = abs(before_mean - after_mean
            ) / pooled_std if pooled_std != 0 else 0
        effect_size_category = 'none'
        if effect_size >= 0.2:
            effect_size_category = 'small'
        if effect_size >= 0.5:
            effect_size_category = 'medium'
        if effect_size >= 0.8:
            effect_size_category = 'large'
        return {'is_valid': p_value < self.significance_level and
            is_improvement, 'metric_name': metric_name, 'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'is_improvement': is_improvement, 'before_mean': before_mean,
            'after_mean': after_mean, 'improvement': improvement,
            'improvement_percentage': improvement_pct, 'effect_size':
            effect_size, 'effect_size_category': effect_size_category,
            'sample_counts': {'before': len(before_values), 'after': len(
            after_values)}}

    def _classify_correlation_strength(self, corr_value: float
        ) ->CorrelationStrength:
        """
        Classify the strength of a correlation coefficient.
        
        Args:
            corr_value: Absolute correlation coefficient value (0-1)
            
        Returns:
            CorrelationStrength enum indicating the strength category
        """
        if corr_value < self.correlation_thresholds['weak']:
            return CorrelationStrength.NONE
        elif corr_value < self.correlation_thresholds['moderate']:
            return CorrelationStrength.WEAK
        elif corr_value < self.correlation_thresholds['strong']:
            return CorrelationStrength.MODERATE
        elif corr_value < self.correlation_thresholds['very_strong']:
            return CorrelationStrength.STRONG
        else:
            return CorrelationStrength.VERY_STRONG
