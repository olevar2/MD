"""
Enhanced Tool Effectiveness Framework

This module provides advanced analytics and effectiveness evaluation for trading tools,
including regime-specific analysis, cross-timeframe performance, and statistical significance.
"""

from typing import Dict, List, Any, Union, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import json

logger = logging.getLogger(__name__)

class EnhancedEffectivenessAnalyzer:
    """
    Enhanced analyzer for tool effectiveness across different market regimes and timeframes.
    
    This analyzer provides advanced effectiveness metrics, statistical significance testing,
    and detailed performance breakdowns to evaluate and compare trading tools.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced effectiveness analyzer
        
        Args:
            config: Configuration parameters for the analyzer
        """
        self.config = config or {}
        self.significance_threshold = self.config.get("significance_threshold", 0.05)
        self.min_sample_size = self.config.get("min_sample_size", 30)
        self.baseline_lookback = self.config.get("baseline_lookback_days", 90)
        self.performance_decay_window = self.config.get("performance_decay_window_days", 30)
    
    def calculate_regime_specific_performance(
        self, 
        tool_results: List[Dict[str, Any]], 
        market_regimes: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate tool effectiveness metrics specific to each market regime
        
        Args:
            tool_results: List of tool result records with timestamp and effectiveness data
            market_regimes: List of market regime records with timestamp and regime data
            
        Returns:
            Dictionary mapping regime types to performance metrics
        """
        # Create a DataFrame from tool results
        if not tool_results or not market_regimes:
            logger.warning("Insufficient data for regime-specific analysis")
            return {}
            
        try:
            results_df = pd.DataFrame(tool_results)
            regimes_df = pd.DataFrame(market_regimes)
            
            # Convert timestamps to datetime
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
            regimes_df['timestamp'] = pd.to_datetime(regimes_df['timestamp'])
            
            # Merge datasets on timestamp (nearest match)
            merged_df = pd.merge_asof(
                results_df.sort_values('timestamp'),
                regimes_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            
            # Group by regime and calculate metrics
            regime_metrics = {}
            
            for regime, group in merged_df.groupby('regime'):
                if len(group) < self.min_sample_size:
                    logger.info(f"Insufficient samples for regime {regime}: {len(group)} < {self.min_sample_size}")
                    continue
                    
                metrics = {
                    'accuracy': group['accuracy'].mean(),
                    'precision': group['precision'].mean() if 'precision' in group.columns else None,
                    'recall': group['recall'].mean() if 'recall' in group.columns else None,
                    'f1_score': group['f1_score'].mean() if 'f1_score' in group.columns else None,
                    'profit_factor': group['profit'].sum() / abs(group['loss'].sum()) if 'profit' in group.columns and 'loss' in group.columns else None,
                    'average_return': group['return'].mean() if 'return' in group.columns else None,
                    'win_rate': (group['outcome'] == 'win').mean() if 'outcome' in group.columns else None,
                    'sample_size': len(group),
                    'statistical_significance': self._calculate_statistical_significance(
                        group['accuracy'], 
                        0.5  # Compare against random chance
                    )
                }
                
                # Filter out None values
                metrics = {k: v for k, v in metrics.items() if v is not None}
                regime_metrics[regime] = metrics
            
            return regime_metrics
            
        except Exception as e:
            logger.error(f"Error in regime-specific performance calculation: {e}")
            return {}
    
    def analyze_cross_timeframe_consistency(
        self, 
        tool_results_by_timeframe: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze consistency of tool performance across different timeframes
        
        Args:
            tool_results_by_timeframe: Dictionary mapping timeframes to tool result lists
            
        Returns:
            Dictionary with cross-timeframe analysis results
        """
        if not tool_results_by_timeframe or len(tool_results_by_timeframe) < 2:
            logger.warning("Insufficient timeframes for cross-timeframe analysis")
            return {
                "consistency_score": None,
                "timeframe_metrics": {},
                "variance_analysis": {},
                "statistical_tests": {}
            }
            
        try:
            # Calculate metrics for each timeframe
            timeframe_metrics = {}
            accuracy_values = []
            
            for timeframe, results in tool_results_by_timeframe.items():
                if not results or len(results) < self.min_sample_size:
                    logger.info(f"Insufficient samples for timeframe {timeframe}: {len(results) if results else 0} < {self.min_sample_size}")
                    continue
                    
                results_df = pd.DataFrame(results)
                
                metrics = {
                    'accuracy': results_df['accuracy'].mean() if 'accuracy' in results_df.columns else None,
                    'precision': results_df['precision'].mean() if 'precision' in results_df.columns else None,
                    'recall': results_df['recall'].mean() if 'recall' in results_df.columns else None,
                    'sample_size': len(results_df)
                }
                
                # Filter out None values
                metrics = {k: v for k, v in metrics.items() if v is not None}
                timeframe_metrics[timeframe] = metrics
                
                if 'accuracy' in results_df.columns:
                    accuracy_values.append(results_df['accuracy'].values)
            
            # Calculate consistency score
            consistency_score = None
            if accuracy_values and len(accuracy_values) >= 2:
                # Use coefficient of variation across timeframe accuracies
                mean_accuracies = [np.mean(acc) for acc in accuracy_values]
                mean_of_means = np.mean(mean_accuracies)
                std_of_means = np.std(mean_accuracies)
                
                if mean_of_means > 0:
                    # Lower CV = higher consistency
                    cv = std_of_means / mean_of_means
                    consistency_score = 1.0 - min(cv, 1.0)  # Invert and bound to [0,1]
            
            # Perform statistical tests for significant differences
            statistical_tests = {}
            
            if len(accuracy_values) >= 2:
                # ANOVA test to see if there are significant differences between timeframes
                f_stat, p_value = f_oneway(*accuracy_values)
                statistical_tests['anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant_difference': p_value < self.significance_threshold
                }
            
            # Calculate variance analysis
            variance_analysis = {}
            if accuracy_values and len(accuracy_values) >= 2:
                variances = [np.var(acc) for acc in accuracy_values]
                variance_analysis = {
                    'mean_variance': float(np.mean(variances)),
                    'max_variance': float(np.max(variances)),
                    'min_variance': float(np.min(variances)),
                    'variance_ratio': float(np.max(variances) / np.min(variances)) if np.min(variances) > 0 else float('inf')
                }
            
            return {
                "consistency_score": consistency_score,
                "timeframe_metrics": timeframe_metrics,
                "variance_analysis": variance_analysis,
                "statistical_tests": statistical_tests
            }
            
        except Exception as e:
            logger.error(f"Error in cross-timeframe consistency analysis: {e}")
            return {
                "error": str(e),
                "consistency_score": None,
                "timeframe_metrics": {},
                "variance_analysis": {},
                "statistical_tests": {}
            }
    
    def detect_performance_decay(
        self, 
        tool_results: List[Dict[str, Any]], 
        window_size: int = None
    ) -> Dict[str, Any]:
        """
        Detect if tool performance is decaying over time
        
        Args:
            tool_results: List of tool result records with timestamp and effectiveness data
            window_size: Rolling window size in days for decay detection
            
        Returns:
            Dictionary with performance decay analysis results
        """
        if not tool_results or len(tool_results) < self.min_sample_size * 2:  # Need enough data for comparison
            logger.warning("Insufficient data for performance decay analysis")
            return {
                "decay_detected": False,
                "decay_metrics": {}
            }
            
        try:
            # Use configured window size or default
            window = window_size if window_size is not None else self.performance_decay_window
            
            # Create DataFrame and sort by timestamp
            results_df = pd.DataFrame(tool_results)
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
            results_df.sort_values('timestamp', inplace=True)
            
            # Split data into time windows
            total_days = (results_df['timestamp'].max() - results_df['timestamp'].min()).days
            
            # If total timespan is too short, use smaller windows
            if total_days < window * 2:
                window = max(total_days // 4, 7)  # At least 7 days, but aim for 4 windows
            
            results_df['time_window'] = pd.cut(
                results_df['timestamp'], 
                bins=pd.date_range(
                    start=results_df['timestamp'].min(),
                    end=results_df['timestamp'].max() + pd.Timedelta(days=1),
                    freq=f'{window}D'
                )
            )
            
            # Calculate metrics by window
            window_metrics = results_df.groupby('time_window').agg({
                'accuracy': 'mean',
                'timestamp': 'count'  # Count as sample size
            }).rename(columns={'timestamp': 'sample_size'})
            
            # If we have at least 2 windows with sufficient data
            if len(window_metrics) >= 2 and all(window_metrics['sample_size'] >= self.min_sample_size / 2):
                # Calculate trend using linear regression
                y = window_metrics['accuracy'].values
                x = np.arange(len(y))
                
                # Simple linear regression
                slope, intercept = np.polyfit(x, y, 1)
                
                # Calculate R-squared
                y_pred = slope * x + intercept
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                ss_res = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Statistical test: compare first and last window
                first_window_data = results_df[results_df['time_window'] == window_metrics.index[0]]['accuracy']
                last_window_data = results_df[results_df['time_window'] == window_metrics.index[-1]]['accuracy']
                
                if len(first_window_data) >= self.min_sample_size / 2 and len(last_window_data) >= self.min_sample_size / 2:
                    t_stat, p_value = ttest_ind(first_window_data, last_window_data)
                    significant_difference = p_value < self.significance_threshold
                else:
                    t_stat, p_value, significant_difference = None, None, False
                
                # Determine if decay is present
                decay_detected = (
                    slope < -0.005 and  # Negative slope of meaningful magnitude
                    (significant_difference if significant_difference is not None else False) and  # Statistically significant
                    r_squared > 0.5  # Reasonable fit
                )
                
                decay_metrics = {
                    "trend_slope": float(slope),
                    "trend_intercept": float(intercept),
                    "r_squared": float(r_squared),
                    "statistical_test": {
                        "t_statistic": float(t_stat) if t_stat is not None else None,
                        "p_value": float(p_value) if p_value is not None else None,
                        "significant_difference": significant_difference
                    },
                    "window_size_days": window,
                    "window_count": len(window_metrics),
                    "earliest_accuracy": float(window_metrics['accuracy'].iloc[0]),
                    "latest_accuracy": float(window_metrics['accuracy'].iloc[-1]),
                    "accuracy_change": float(window_metrics['accuracy'].iloc[-1] - window_metrics['accuracy'].iloc[0]),
                    "accuracy_change_percent": float((window_metrics['accuracy'].iloc[-1] / window_metrics['accuracy'].iloc[0] - 1) * 100) 
                        if window_metrics['accuracy'].iloc[0] > 0 else 0.0
                }
                
                return {
                    "decay_detected": decay_detected,
                    "decay_metrics": decay_metrics
                }
            else:
                return {
                    "decay_detected": False,
                    "decay_metrics": {},
                    "message": "Insufficient data in time windows for decay analysis"
                }
                
        except Exception as e:
            logger.error(f"Error in performance decay detection: {e}")
            return {
                "decay_detected": False,
                "decay_metrics": {},
                "error": str(e)
            }
    
    def compare_against_baseline(
        self, 
        tool_results: List[Dict[str, Any]], 
        baseline_results: List[Dict[str, Any]] = None, 
        baseline_accuracy: float = None
    ) -> Dict[str, Any]:
        """
        Compare tool performance against a baseline or historical performance
        
        Args:
            tool_results: List of tool result records
            baseline_results: Optional list of baseline result records
            baseline_accuracy: Optional fixed baseline accuracy value
            
        Returns:
            Dictionary with comparison analysis results
        """
        if not tool_results or len(tool_results) < self.min_sample_size:
            logger.warning("Insufficient data for baseline comparison")
            return {
                "statistically_significant": False,
                "comparison_metrics": {}
            }
            
        try:
            # Create DataFrame from current results
            current_df = pd.DataFrame(tool_results)
            
            # Calculate current metrics
            current_accuracy = current_df['accuracy'].mean() if 'accuracy' in current_df.columns else None
            current_precision = current_df['precision'].mean() if 'precision' in current_df.columns else None
            current_recall = current_df['recall'].mean() if 'recall' in current_df.columns else None
            
            if current_accuracy is None:
                return {
                    "statistically_significant": False,
                    "comparison_metrics": {},
                    "error": "Accuracy data not available in current results"
                }
            
            # Determine baseline
            if baseline_results and len(baseline_results) >= self.min_sample_size:
                # Use provided baseline results
                baseline_df = pd.DataFrame(baseline_results)
                baseline_accuracy_value = baseline_df['accuracy'].mean() if 'accuracy' in baseline_df.columns else 0.5
                baseline_precision = baseline_df['precision'].mean() if 'precision' in baseline_df.columns else None
                baseline_recall = baseline_df['recall'].mean() if 'recall' in baseline_df.columns else None
                baseline_sample_size = len(baseline_df)
                
                # Statistical test
                if 'accuracy' in baseline_df.columns:
                    t_stat, p_value = ttest_ind(current_df['accuracy'], baseline_df['accuracy'])
                else:
                    t_stat, p_value = None, None
                    
            elif baseline_accuracy is not None:
                # Use provided baseline accuracy value
                baseline_accuracy_value = baseline_accuracy
                baseline_precision = None
                baseline_recall = None
                baseline_sample_size = None
                
                # One-sample t-test against fixed value
                t_stat, p_value = ttest_ind(
                    current_df['accuracy'], 
                    np.ones(len(current_df['accuracy'])) * baseline_accuracy_value
                )
                
            else:
                # Use random chance as baseline
                baseline_accuracy_value = 0.5  # Assuming binary classification
                baseline_precision = None
                baseline_recall = None
                baseline_sample_size = None
                
                # One-sample t-test against random chance
                t_stat, p_value = ttest_ind(
                    current_df['accuracy'], 
                    np.ones(len(current_df['accuracy'])) * baseline_accuracy_value
                )
            
            # Calculate improvement metrics
            absolute_improvement = current_accuracy - baseline_accuracy_value
            relative_improvement = (current_accuracy / baseline_accuracy_value - 1) * 100 if baseline_accuracy_value > 0 else 0
            
            # Determine statistical significance
            statistically_significant = p_value < self.significance_threshold if p_value is not None else False
            
            comparison_metrics = {
                "current_accuracy": float(current_accuracy),
                "baseline_accuracy": float(baseline_accuracy_value),
                "absolute_improvement": float(absolute_improvement),
                "relative_improvement_percent": float(relative_improvement),
                "current_sample_size": int(len(current_df)),
                "baseline_sample_size": int(baseline_sample_size) if baseline_sample_size is not None else None,
                "statistical_test": {
                    "t_statistic": float(t_stat) if t_stat is not None else None,
                    "p_value": float(p_value) if p_value is not None else None,
                }
            }
            
            # Add other metrics if available
            if current_precision is not None and baseline_precision is not None:
                comparison_metrics["current_precision"] = float(current_precision)
                comparison_metrics["baseline_precision"] = float(baseline_precision)
                comparison_metrics["precision_improvement"] = float(current_precision - baseline_precision)
                
            if current_recall is not None and baseline_recall is not None:
                comparison_metrics["current_recall"] = float(current_recall)
                comparison_metrics["baseline_recall"] = float(baseline_recall)
                comparison_metrics["recall_improvement"] = float(current_recall - baseline_recall)
            
            return {
                "statistically_significant": statistically_significant,
                "comparison_metrics": comparison_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in baseline comparison: {e}")
            return {
                "statistically_significant": False,
                "comparison_metrics": {},
                "error": str(e)
            }
    
    def _calculate_statistical_significance(self, values: Union[List, np.ndarray], baseline: float) -> Dict[str, Any]:
        """
        Calculate statistical significance of values compared to baseline
        
        Args:
            values: List or array of values to test
            baseline: Baseline value for comparison
            
        Returns:
            Dictionary with statistical test results
        """
        if not isinstance(values, (list, np.ndarray)) or len(values) < self.min_sample_size:
            return {
                "significant": False,
                "p_value": None,
                "test_statistic": None,
                "confidence_level": None
            }
            
        try:
            # One-sample t-test
            t_stat, p_value = ttest_ind(values, np.ones(len(values)) * baseline)
            
            # Determine confidence level
            if p_value <= 0.01:
                confidence_level = "very high"
            elif p_value <= 0.05:
                confidence_level = "high"
            elif p_value <= 0.1:
                confidence_level = "moderate"
            else:
                confidence_level = "low"
                
            return {
                "significant": p_value < self.significance_threshold,
                "p_value": float(p_value),
                "test_statistic": float(t_stat),
                "confidence_level": confidence_level
            }
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {e}")
            return {
                "significant": False,
                "p_value": None,
                "test_statistic": None,
                "confidence_level": None,
                "error": str(e)
            }
    
    def generate_comprehensive_report(
        self, 
        tool_id: str,
        tool_results: List[Dict[str, Any]],
        market_regimes: List[Dict[str, Any]] = None,
        timeframe_results: Dict[str, List[Dict[str, Any]]] = None,
        baseline_results: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive effectiveness report combining all analyses
        
        Args:
            tool_id: Identifier for the tool being analyzed
            tool_results: List of tool result records
            market_regimes: Optional list of market regime records
            timeframe_results: Optional dictionary mapping timeframes to results
            baseline_results: Optional list of baseline result records
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        report = {
            "tool_id": tool_id,
            "generated_at": datetime.now().isoformat(),
            "overall_metrics": {},
            "regime_specific_performance": {},
            "cross_timeframe_consistency": {},
            "performance_decay": {},
            "baseline_comparison": {}
        }
        
        # Skip if insufficient data
        if not tool_results or len(tool_results) < self.min_sample_size:
            report["error"] = f"Insufficient data: {len(tool_results) if tool_results else 0} samples (minimum {self.min_sample_size})"
            return report
            
        # Calculate overall metrics
        try:
            results_df = pd.DataFrame(tool_results)
            
            report["overall_metrics"] = {
                "accuracy": float(results_df['accuracy'].mean()) if 'accuracy' in results_df.columns else None,
                "precision": float(results_df['precision'].mean()) if 'precision' in results_df.columns else None,
                "recall": float(results_df['recall'].mean()) if 'recall' in results_df.columns else None,
                "f1_score": float(results_df['f1_score'].mean()) if 'f1_score' in results_df.columns else None,
                "sample_size": int(len(results_df)),
                "date_range": {
                    "start": results_df['timestamp'].min().isoformat() if 'timestamp' in results_df.columns else None,
                    "end": results_df['timestamp'].max().isoformat() if 'timestamp' in results_df.columns else None
                }
            }
            
            # Filter None values
            report["overall_metrics"] = {k: v for k, v in report["overall_metrics"].items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            report["overall_metrics"] = {"error": str(e)}
        
        # Calculate regime-specific performance
        if market_regimes:
            try:
                report["regime_specific_performance"] = self.calculate_regime_specific_performance(
                    tool_results, market_regimes
                )
            except Exception as e:
                logger.error(f"Error in regime-specific analysis: {e}")
                report["regime_specific_performance"] = {"error": str(e)}
        
        # Calculate cross-timeframe consistency
        if timeframe_results:
            try:
                report["cross_timeframe_consistency"] = self.analyze_cross_timeframe_consistency(
                    timeframe_results
                )
            except Exception as e:
                logger.error(f"Error in cross-timeframe analysis: {e}")
                report["cross_timeframe_consistency"] = {"error": str(e)}
        
        # Detect performance decay
        try:
            report["performance_decay"] = self.detect_performance_decay(tool_results)
        except Exception as e:
            logger.error(f"Error in performance decay analysis: {e}")
            report["performance_decay"] = {"error": str(e)}
        
        # Compare against baseline
        try:
            report["baseline_comparison"] = self.compare_against_baseline(
                tool_results, baseline_results
            )
        except Exception as e:
            logger.error(f"Error in baseline comparison: {e}")
            report["baseline_comparison"] = {"error": str(e)}
        
        return report
