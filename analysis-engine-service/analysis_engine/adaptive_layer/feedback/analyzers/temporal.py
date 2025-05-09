"""
Temporal Feedback Analyzer

This module implements temporal analysis of feedback across timeframes.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.feedback_exceptions import TimeframeFeedbackError

from ..models import TimeframeInsight

logger = get_logger(__name__)


class TemporalFeedbackAnalyzer:
    """Analyzes temporal patterns in feedback across timeframes."""
    
    def __init__(self, min_sample_size: int = 20):
        """
        Initialize the temporal feedback analyzer.
        
        Args:
            min_sample_size: Minimum sample size for temporal analysis
        """
        self.min_sample_size = min_sample_size
        logger.info("TemporalFeedbackAnalyzer initialized")
    
    def analyze_temporal_patterns(
        self,
        instrument: str,
        timeframe_data: Dict[str, List[Dict[str, Any]]],
        window_days: int = 30
    ) -> TimeframeInsight:
        """
        Analyze temporal patterns in feedback across timeframes.
        
        Args:
            instrument: The instrument being analyzed
            timeframe_data: Dictionary mapping timeframes to lists of feedback data
            window_days: Number of days to analyze
            
        Returns:
            TimeframeInsight: Insights from temporal analysis
        """
        try:
            # Initialize insight object
            insight = TimeframeInsight(
                instrument=instrument,
                timeframes_analyzed=list(timeframe_data.keys()),
                analysis_time=datetime.utcnow()
            )
            
            # Calculate cutoff time
            cutoff_time = datetime.utcnow() - timedelta(days=window_days)
            
            # Process each timeframe
            accuracy_scores = {}
            error_trends = {}
            
            for timeframe, feedback_list in timeframe_data.items():
                # Filter by time window
                filtered_feedback = []
                for fb in feedback_list:
                    try:
                        fb_time = datetime.fromisoformat(fb["timestamp"])
                        if fb_time >= cutoff_time:
                            filtered_feedback.append(fb)
                    except:
                        continue
                
                # Skip if not enough data
                if len(filtered_feedback) < self.min_sample_size:
                    logger.debug(f"Insufficient data for {timeframe} temporal analysis: {len(filtered_feedback)}")
                    continue
                
                # Sort by timestamp
                filtered_feedback.sort(key=lambda fb: fb["timestamp"])
                
                # Extract error magnitudes
                error_magnitudes = []
                timestamps = []
                
                for fb in filtered_feedback:
                    if "error_magnitude" in fb and fb["error_magnitude"] is not None:
                        error_magnitudes.append(fb["error_magnitude"])
                        timestamps.append(fb["timestamp"])
                
                # Skip if not enough error data
                if len(error_magnitudes) < self.min_sample_size:
                    logger.debug(f"Insufficient error data for {timeframe} temporal analysis: {len(error_magnitudes)}")
                    continue
                
                # Calculate accuracy score (inverse of average error)
                avg_error = sum(error_magnitudes) / len(error_magnitudes)
                accuracy_score = 1.0 / (1.0 + avg_error)  # Normalize to 0-1 range
                accuracy_scores[timeframe] = accuracy_score
                
                # Calculate error trend
                trend = self._calculate_trend(error_magnitudes)
                error_trends[timeframe] = trend
                
                # Detect error patterns
                patterns = self._detect_error_patterns(error_magnitudes, timestamps)
                if patterns:
                    for pattern in patterns:
                        pattern["timeframe"] = timeframe
                        insight.error_patterns.append(pattern)
            
            # Identify most and least accurate timeframes
            if accuracy_scores:
                sorted_accuracy = sorted(
                    accuracy_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Most accurate (top 3 or fewer)
                for tf, score in sorted_accuracy[:min(3, len(sorted_accuracy))]:
                    insight.most_accurate_timeframes.append({
                        "timeframe": tf,
                        "accuracy_score": score,
                        "error_trend": error_trends.get(tf, 0.0)
                    })
                
                # Least accurate (bottom 3 or fewer)
                for tf, score in sorted_accuracy[-min(3, len(sorted_accuracy)):]:
                    insight.least_accurate_timeframes.append({
                        "timeframe": tf,
                        "accuracy_score": score,
                        "error_trend": error_trends.get(tf, 0.0)
                    })
            
            # Generate recommendations
            self._generate_recommendations(insight, accuracy_scores, error_trends)
            
            return insight
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to analyze temporal patterns: {str(e)}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate the trend in a series of values.
        
        Args:
            values: List of values
            
        Returns:
            float: Trend coefficient
        """
        if len(values) < 2:
            return 0.0
        
        try:
            # Use linear regression to calculate trend
            x = np.array(range(len(values)))
            y = np.array(values)
            
            # Calculate slope
            n = len(values)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_xx = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            
            # Normalize by average value
            avg_value = np.mean(values)
            if avg_value != 0:
                normalized_slope = slope / avg_value
            else:
                normalized_slope = slope
            
            return normalized_slope
            
        except:
            # Fallback to simple trend calculation
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            if not first_half or not second_half:
                return 0.0
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if first_avg != 0:
                return (second_avg - first_avg) / first_avg
            else:
                return 0.0
    
    def _detect_error_patterns(
        self, error_magnitudes: List[float], timestamps: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns in error magnitudes.
        
        Args:
            error_magnitudes: List of error magnitudes
            timestamps: List of corresponding timestamps
            
        Returns:
            List[Dict[str, Any]]: Detected patterns
        """
        patterns = []
        
        if len(error_magnitudes) < self.min_sample_size:
            return patterns
        
        try:
            # Convert timestamps to datetime objects
            datetime_timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            
            # Check for cyclical patterns
            cyclical_pattern = self._detect_cyclical_pattern(error_magnitudes, datetime_timestamps)
            if cyclical_pattern:
                patterns.append(cyclical_pattern)
            
            # Check for sudden spikes
            spike_pattern = self._detect_spikes(error_magnitudes, datetime_timestamps)
            if spike_pattern:
                patterns.append(spike_pattern)
            
            # Check for consistent improvement or degradation
            trend_pattern = self._detect_trend_pattern(error_magnitudes, datetime_timestamps)
            if trend_pattern:
                patterns.append(trend_pattern)
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error detecting patterns: {str(e)}")
            return patterns
    
    def _detect_cyclical_pattern(
        self, error_magnitudes: List[float], timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect cyclical patterns in error magnitudes.
        
        Args:
            error_magnitudes: List of error magnitudes
            timestamps: List of corresponding timestamps
            
        Returns:
            Optional[Dict[str, Any]]: Cyclical pattern if detected
        """
        # This is a simplified implementation
        # A more sophisticated approach would use FFT or autocorrelation
        
        # Check for daily pattern
        daily_scores = [0] * 24  # 24 hours
        daily_counts = [0] * 24
        
        for error, ts in zip(error_magnitudes, timestamps):
            hour = ts.hour
            daily_scores[hour] += error
            daily_counts[hour] += 1
        
        # Calculate average error by hour
        hourly_avgs = []
        for hour in range(24):
            if daily_counts[hour] > 0:
                hourly_avgs.append(daily_scores[hour] / daily_counts[hour])
            else:
                hourly_avgs.append(0)
        
        # Check if there's significant variation by hour
        if hourly_avgs and max(hourly_avgs) > 0:
            max_hour = hourly_avgs.index(max(hourly_avgs))
            min_hour = hourly_avgs.index(min(filter(lambda x: x > 0, hourly_avgs)))
            
            # Calculate variation
            variation = max(hourly_avgs) / min(filter(lambda x: x > 0, hourly_avgs))
            
            if variation > 1.5:  # Significant variation
                return {
                    "type": "cyclical",
                    "cycle_type": "daily",
                    "highest_error_hour": max_hour,
                    "lowest_error_hour": min_hour,
                    "variation_factor": variation,
                    "confidence": "medium"
                }
        
        return None
    
    def _detect_spikes(
        self, error_magnitudes: List[float], timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect spikes in error magnitudes.
        
        Args:
            error_magnitudes: List of error magnitudes
            timestamps: List of corresponding timestamps
            
        Returns:
            Optional[Dict[str, Any]]: Spike pattern if detected
        """
        if len(error_magnitudes) < 10:
            return None
        
        # Calculate mean and standard deviation
        mean = sum(error_magnitudes) / len(error_magnitudes)
        std_dev = (sum((x - mean) ** 2 for x in error_magnitudes) / len(error_magnitudes)) ** 0.5
        
        # Identify spikes (values more than 2 standard deviations from mean)
        threshold = mean + 2 * std_dev
        spikes = []
        
        for i, (error, ts) in enumerate(zip(error_magnitudes, timestamps)):
            if error > threshold:
                spikes.append({
                    "index": i,
                    "timestamp": ts.isoformat(),
                    "error": error,
                    "deviation": (error - mean) / std_dev if std_dev > 0 else 0
                })
        
        if spikes:
            return {
                "type": "spikes",
                "count": len(spikes),
                "mean_error": mean,
                "threshold": threshold,
                "examples": spikes[:3],  # Include up to 3 examples
                "confidence": "high" if len(spikes) > 3 else "medium"
            }
        
        return None
    
    def _detect_trend_pattern(
        self, error_magnitudes: List[float], timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect trend patterns in error magnitudes.
        
        Args:
            error_magnitudes: List of error magnitudes
            timestamps: List of corresponding timestamps
            
        Returns:
            Optional[Dict[str, Any]]: Trend pattern if detected
        """
        if len(error_magnitudes) < 10:
            return None
        
        # Calculate trend
        trend = self._calculate_trend(error_magnitudes)
        
        # Determine if trend is significant
        if abs(trend) > 0.01:  # 1% change per data point
            direction = "improving" if trend < 0 else "degrading"
            
            return {
                "type": "trend",
                "direction": direction,
                "magnitude": abs(trend),
                "confidence": "high" if abs(trend) > 0.05 else "medium"
            }
        
        return None
    
    def _generate_recommendations(
        self,
        insight: TimeframeInsight,
        accuracy_scores: Dict[str, float],
        error_trends: Dict[str, float]
    ) -> None:
        """
        Generate recommendations based on analysis.
        
        Args:
            insight: TimeframeInsight object to update
            accuracy_scores: Accuracy scores by timeframe
            error_trends: Error trends by timeframe
        """
        # Recommend focusing on most accurate timeframes
        if insight.most_accurate_timeframes:
            best_timeframes = [tf["timeframe"] for tf in insight.most_accurate_timeframes]
            insight.recommendations.append({
                "type": "focus",
                "timeframes": best_timeframes,
                "reason": "These timeframes have shown the highest prediction accuracy",
                "priority": "high"
            })
        
        # Recommend caution with least accurate timeframes
        if insight.least_accurate_timeframes:
            worst_timeframes = [tf["timeframe"] for tf in insight.least_accurate_timeframes]
            insight.recommendations.append({
                "type": "caution",
                "timeframes": worst_timeframes,
                "reason": "These timeframes have shown the lowest prediction accuracy",
                "priority": "high"
            })
        
        # Recommend investigating timeframes with degrading performance
        degrading_timeframes = []
        for timeframe, trend in error_trends.items():
            if trend > 0.05:  # Significant degradation
                degrading_timeframes.append(timeframe)
        
        if degrading_timeframes:
            insight.recommendations.append({
                "type": "investigate",
                "timeframes": degrading_timeframes,
                "reason": "These timeframes show degrading performance over time",
                "priority": "medium"
            })
        
        # Recommend model retraining for timeframes with consistent errors
        retraining_candidates = []
        for pattern in insight.error_patterns:
            if pattern["type"] == "trend" and pattern["direction"] == "degrading":
                retraining_candidates.append(pattern["timeframe"])
        
        if retraining_candidates:
            insight.recommendations.append({
                "type": "retrain",
                "timeframes": retraining_candidates,
                "reason": "These timeframes show consistent degradation in performance",
                "priority": "high"
            })
        
        # Recommend time-based adjustments for cyclical patterns
        cyclical_timeframes = []
        for pattern in insight.error_patterns:
            if pattern["type"] == "cyclical":
                cyclical_timeframes.append(pattern["timeframe"])
        
        if cyclical_timeframes:
            insight.recommendations.append({
                "type": "time_adjust",
                "timeframes": cyclical_timeframes,
                "reason": "These timeframes show cyclical patterns in prediction errors",
                "priority": "medium"
            })