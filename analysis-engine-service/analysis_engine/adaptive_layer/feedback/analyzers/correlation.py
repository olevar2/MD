"""
Timeframe Correlation Analyzer

This module implements correlation analysis between different timeframes.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.feedback_exceptions import TimeframeFeedbackError

from ..models import TimeframeCorrelation

logger = get_logger(__name__)


class TimeframeCorrelationAnalyzer:
    """Analyzes correlations between different timeframes."""
    
    def __init__(self, min_sample_size: int = 10):
        """
        Initialize the timeframe correlation analyzer.
        
        Args:
            min_sample_size: Minimum sample size for correlation analysis
        """
        self.min_sample_size = min_sample_size
        logger.info("TimeframeCorrelationAnalyzer initialized")
    
    def analyze_correlations(
        self,
        instrument: str,
        timeframe_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[TimeframeCorrelation]:
        """
        Analyze correlations between timeframes.
        
        Args:
            instrument: The instrument being analyzed
            timeframe_data: Dictionary mapping timeframes to lists of feedback data
            
        Returns:
            List[TimeframeCorrelation]: Correlation results
        """
        try:
            results = []
            timeframes = list(timeframe_data.keys())
            
            # Need at least two timeframes to calculate correlations
            if len(timeframes) < 2:
                logger.warning(f"Not enough timeframes for correlation analysis: {len(timeframes)}")
                return results
            
            # Extract error magnitudes for each timeframe
            error_data = {}
            for timeframe, feedback_list in timeframe_data.items():
                errors = []
                timestamps = []
                
                for feedback in feedback_list:
                    if "error_magnitude" in feedback and feedback["error_magnitude"] is not None:
                        errors.append(feedback["error_magnitude"])
                        timestamps.append(feedback["timestamp"])
                
                if errors:
                    error_data[timeframe] = {
                        "errors": errors,
                        "timestamps": timestamps
                    }
            
            # Calculate correlations between all timeframe pairs
            for i, tf1 in enumerate(timeframes):
                if tf1 not in error_data:
                    continue
                    
                for j in range(i + 1, len(timeframes)):
                    tf2 = timeframes[j]
                    if tf2 not in error_data:
                        continue
                    
                    # Get error data for both timeframes
                    errors1 = error_data[tf1]["errors"]
                    errors2 = error_data[tf2]["errors"]
                    
                    # Check sample size
                    sample_size = min(len(errors1), len(errors2))
                    if sample_size < self.min_sample_size:
                        logger.debug(f"Insufficient sample size for {tf1}-{tf2} correlation: {sample_size}")
                        continue
                    
                    # Calculate correlation
                    correlation, significance = self._calculate_correlation(errors1[:sample_size], errors2[:sample_size])
                    
                    # Create correlation object
                    corr_obj = TimeframeCorrelation(
                        instrument=instrument,
                        timeframe1=tf1,
                        timeframe2=tf2,
                        correlation=correlation,
                        significance=significance,
                        sample_size=sample_size,
                        calculation_time=datetime.utcnow()
                    )
                    
                    results.append(corr_obj)
                    
                    logger.debug(
                        f"Correlation between {tf1} and {tf2}: {correlation:.2f} "
                        f"({significance}, n={sample_size})"
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe correlations: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to analyze timeframe correlations: {str(e)}")
    
    def _calculate_correlation(
        self, errors1: List[float], errors2: List[float]
    ) -> Tuple[float, str]:
        """
        Calculate correlation coefficient and significance.
        
        Args:
            errors1: Error magnitudes from first timeframe
            errors2: Error magnitudes from second timeframe
            
        Returns:
            Tuple[float, str]: Correlation coefficient and significance
        """
        # Ensure equal length
        min_len = min(len(errors1), len(errors2))
        errors1 = errors1[:min_len]
        errors2 = errors2[:min_len]
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(errors1, errors2)[0, 1]
        except:
            # Fallback to manual calculation
            mean1 = sum(errors1) / len(errors1)
            mean2 = sum(errors2) / len(errors2)
            
            numerator = sum((x - mean1) * (y - mean2) for x, y in zip(errors1, errors2))
            denom1 = sum((x - mean1) ** 2 for x in errors1) ** 0.5
            denom2 = sum((y - mean2) ** 2 for y in errors2) ** 0.5
            
            if denom1 == 0 or denom2 == 0:
                correlation = 0.0
            else:
                correlation = numerator / (denom1 * denom2)
        
        # Determine significance
        abs_corr = abs(correlation)
        if abs_corr < 0.3:
            significance = "weak"
        elif abs_corr < 0.7:
            significance = "moderate"
        else:
            significance = "strong"
        
        return correlation, significance
    
    def get_leading_timeframes(
        self, correlations: List[TimeframeCorrelation]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify leading timeframes based on correlations.
        
        Args:
            correlations: List of timeframe correlations
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Leading timeframes by instrument
        """
        results = {}
        
        # Group correlations by instrument
        by_instrument = {}
        for corr in correlations:
            if corr.instrument not in by_instrument:
                by_instrument[corr.instrument] = []
            by_instrument[corr.instrument].append(corr)
        
        # Process each instrument
        for instrument, corr_list in by_instrument.items():
            # Count how many times each timeframe leads others
            timeframe_scores = {}
            
            for corr in corr_list:
                if corr.significance == "weak":
                    continue
                
                # Positive correlation means they move together
                # Negative correlation means they move in opposite directions
                # Either way, we're looking for patterns
                
                # Initialize timeframes if needed
                if corr.timeframe1 not in timeframe_scores:
                    timeframe_scores[corr.timeframe1] = {"leads": 0, "lags": 0, "correlations": []}
                
                if corr.timeframe2 not in timeframe_scores:
                    timeframe_scores[corr.timeframe2] = {"leads": 0, "lags": 0, "correlations": []}
                
                # For now, we're using a simple heuristic:
                # Smaller timeframes tend to lead larger ones
                # This could be replaced with more sophisticated analysis
                
                # Extract numeric part of timeframe (e.g., "1h" -> 1)
                try:
                    tf1_value = int(''.join(filter(str.isdigit, corr.timeframe1)))
                    tf2_value = int(''.join(filter(str.isdigit, corr.timeframe2)))
                    
                    if tf1_value < tf2_value:
                        # Timeframe 1 is smaller, likely leads
                        timeframe_scores[corr.timeframe1]["leads"] += 1
                        timeframe_scores[corr.timeframe2]["lags"] += 1
                    else:
                        # Timeframe 2 is smaller, likely leads
                        timeframe_scores[corr.timeframe2]["leads"] += 1
                        timeframe_scores[corr.timeframe1]["lags"] += 1
                except:
                    # Skip if we can't parse timeframe values
                    continue
                
                # Record correlation
                timeframe_scores[corr.timeframe1]["correlations"].append({
                    "with": corr.timeframe2,
                    "value": corr.correlation,
                    "significance": corr.significance
                })
                
                timeframe_scores[corr.timeframe2]["correlations"].append({
                    "with": corr.timeframe1,
                    "value": corr.correlation,
                    "significance": corr.significance
                })
            
            # Identify leading timeframes
            leading = []
            for timeframe, scores in timeframe_scores.items():
                if scores["leads"] > scores["lags"]:
                    leading.append({
                        "timeframe": timeframe,
                        "lead_score": scores["leads"] - scores["lags"],
                        "correlations": scores["correlations"]
                    })
            
            # Sort by lead score
            leading.sort(key=lambda x: x["lead_score"], reverse=True)
            
            results[instrument] = leading
        
        return results