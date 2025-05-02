"""
Parameter Statistical Validator

This module implements statistical validation for parameter changes in the feedback system,
providing confidence metrics and statistical significance testing.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import scipy.stats as stats

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class ParameterStatisticalValidator:
    """
    Provides statistical validation for parameter changes.
    
    This class analyzes the effectiveness of parameter changes using statistical
    methods to determine if changes have significant impact on performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the parameter statistical validator.
        
        Args:
            config: Configuration dictionary with statistical thresholds
        """
        self.config = config or {}
        
        # Default statistical settings
        self.significance_level = self.config.get("significance_level", 0.05)
        self.min_sample_size = self.config.get("min_sample_size", 10)
        self.power_threshold = self.config.get("power_threshold", 0.8)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.effect_size_threshold = self.config.get("effect_size_threshold", 0.2)
        
        logger.info("ParameterStatisticalValidator initialized with significance level: %.3f", 
                   self.significance_level)

    def validate_parameter_change(
        self, 
        baseline_metrics: List[Dict[str, Any]],
        new_metrics: List[Dict[str, Any]],
        metric_key: str = "profit_loss",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate if a parameter change produced statistically significant results.
        
        Args:
            baseline_metrics: Performance metrics before parameter change
            new_metrics: Performance metrics after parameter change
            metric_key: The metric to compare (default: 'profit_loss')
            metadata: Additional context about the parameter change
            
        Returns:
            Dict with statistical validation results
        """
        if not baseline_metrics or not new_metrics:
            return {
                "is_significant": False,
                "confidence": 0.0,
                "reason": "Insufficient data for analysis",
                "sample_sizes": (len(baseline_metrics), len(new_metrics)),
                "p_value": None
            }
            
        # Extract values for the metric we're comparing
        baseline_values = [m.get(metric_key, 0) for m in baseline_metrics if metric_key in m]
        new_values = [m.get(metric_key, 0) for m in new_metrics if metric_key in m]
        
        # Check minimum sample sizes
        if len(baseline_values) < self.min_sample_size or len(new_values) < self.min_sample_size:
            return {
                "is_significant": False,
                "confidence": self._calculate_confidence_from_samples(baseline_values, new_values),
                "reason": "Sample size below threshold",
                "sample_sizes": (len(baseline_values), len(new_values)),
                "min_required": self.min_sample_size,
                "p_value": None
            }
            
        # Perform statistical test (t-test for independent samples)
        try:
            t_stat, p_value = stats.ttest_ind(baseline_values, new_values, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_cohens_d(baseline_values, new_values)
            
            # Determine if change is statistically significant
            is_significant = p_value < self.significance_level and abs(effect_size) > self.effect_size_threshold
            
            # Calculate statistical power
            statistical_power = self._calculate_power(baseline_values, new_values, self.significance_level)
            
            # Calculate confidence score based on p-value, effect size and sample size
            confidence = self._calculate_confidence(
                p_value, effect_size, len(baseline_values), len(new_values)
            )
            
            return {
                "is_significant": is_significant,
                "confidence": confidence,
                "p_value": p_value,
                "effect_size": effect_size,
                "effect_size_interpretation": self._interpret_effect_size(effect_size),
                "statistical_power": statistical_power,
                "sample_sizes": (len(baseline_values), len(new_values)),
                "baseline_mean": np.mean(baseline_values),
                "new_mean": np.mean(new_values),
                "percent_change": (np.mean(new_values) - np.mean(baseline_values)) / abs(np.mean(baseline_values))
                if np.mean(baseline_values) != 0 else 0,
                "reason": "Statistically significant improvement" if is_significant and t_stat > 0
                          else "Statistically significant degradation" if is_significant and t_stat < 0
                          else "Not statistically significant"
            }
            
        except Exception as e:
            logger.error("Error performing statistical validation: %s", str(e))
            return {
                "is_significant": False,
                "confidence": 0.0,
                "reason": f"Statistical test error: {str(e)}",
                "sample_sizes": (len(baseline_values), len(new_values)),
                "p_value": None
            }
    
    def estimate_required_samples(
        self,
        observed_effect: float,
        baseline_std: float,
        new_std: Optional[float] = None,
        alpha: float = None,
        power: float = None
    ) -> int:
        """
        Estimate the number of samples needed to detect the effect with given power.
        
        Args:
            observed_effect: The effect size observed in initial samples
            baseline_std: Standard deviation of the baseline data
            new_std: Standard deviation of the new data (use baseline if None)
            alpha: Significance level (uses instance default if None)
            power: Statistical power (uses instance default if None)
            
        Returns:
            Estimated number of samples needed
        """
        if alpha is None:
            alpha = self.significance_level
            
        if power is None:
            power = self.power_threshold
            
        if new_std is None:
            new_std = baseline_std
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt((baseline_std**2 + new_std**2) / 2)
        
        # Calculate standardized effect size
        if pooled_std == 0:
            return self.min_sample_size  # Avoid division by zero
            
        standardized_effect = abs(observed_effect) / pooled_std
        
        # Get critical values
        z_alpha = stats.norm.ppf(1 - alpha/2)  # two-tailed test
        z_power = stats.norm.ppf(power)
        
        # Calculate required sample size (per group)
        n = 2 * ((z_alpha + z_power) / standardized_effect)**2
        
        # Round up and apply minimum
        return max(int(np.ceil(n)), self.min_sample_size)
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Avoid division by zero
        if pooled_sd == 0:
            return 0
            
        return (mean2 - mean1) / pooled_sd
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """
        Interpret the magnitude of Cohen's d effect size.
        
        Args:
            effect_size: Cohen's d value
            
        Returns:
            String interpretation of effect size magnitude
        """
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_power(
        self,
        group1: List[float],
        group2: List[float],
        alpha: float
    ) -> float:
        """
        Calculate the statistical power of the test.
        
        Args:
            group1: First group of values
            group2: Second group of values
            alpha: Significance level
            
        Returns:
            Statistical power (0-1)
        """
        n1, n2 = len(group1), len(group2)
        effect_size = self._calculate_cohens_d(group1, group2)
        
        # Simplified power calculation for t-test
        df = n1 + n2 - 2
        lambda_param = abs(effect_size) * np.sqrt((n1 * n2) / (n1 + n2))
        
        critical_t = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(critical_t, df, lambda_param) + stats.nct.cdf(-critical_t, df, lambda_param)
        
        return power
    
    def _calculate_confidence(
        self,
        p_value: float,
        effect_size: float,
        n1: int,
        n2: int
    ) -> float:
        """
        Calculate confidence score based on p-value, effect size and sample sizes.
        
        Args:
            p_value: P-value from statistical test
            effect_size: Cohen's d effect size
            n1: Sample size of first group
            n2: Sample size of second group
            
        Returns:
            Confidence score (0-1)
        """
        # Convert p-value to confidence component (1 when p=0, 0 when p>=0.2)
        p_confidence = max(0, 1 - (p_value / 0.2))
        
        # Convert effect size to confidence component
        effect_confidence = min(1, abs(effect_size) / 0.8)
        
        # Sample size confidence (reaches 1.0 at 3x minimum sample size)
        target_n = self.min_sample_size * 3
        n_confidence = min(1, (min(n1, n2) / target_n))
        
        # Weighted combination
        weights = {
            "p_value": 0.5,      # Most important
            "effect_size": 0.3,  # Second most important
            "sample_size": 0.2   # Still relevant but less weight
        }
        
        confidence = (
            weights["p_value"] * p_confidence +
            weights["effect_size"] * effect_confidence +
            weights["sample_size"] * n_confidence
        )
        
        return confidence
    
    def _calculate_confidence_from_samples(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate a confidence score based only on available samples.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Simple confidence score based on available data
        """
        # If we don't have enough data for a proper test, calculate a reduced confidence
        n1, n2 = len(group1), len(group2)
        
        # Sample size confidence (scales linearly up to min_sample_size)
        n_confidence = min(1, (min(n1, n2) / self.min_sample_size))
        
        # If we have at least some samples, try to factor in the raw difference
        if n1 > 0 and n2 > 0:
            try:
                mean1 = np.mean(group1)
                mean2 = np.mean(group2)
                
                # Calculate a rough effect magnitude (not proper Cohen's d without enough samples)
                pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2) 
                
                # Avoid division by zero
                if pooled_std > 0:
                    effect_magnitude = abs(mean2 - mean1) / pooled_std
                    effect_confidence = min(1, effect_magnitude / 0.8)
                else:
                    effect_confidence = 0.5
                
                # Combine with reduced weight on effect since sample is small
                return 0.7 * n_confidence + 0.3 * effect_confidence
                
            except Exception:
                # Fall back to just sample confidence if calculation fails
                return n_confidence
        
        return n_confidence * 0.5  # Further reduce confidence if we can't calculate difference
