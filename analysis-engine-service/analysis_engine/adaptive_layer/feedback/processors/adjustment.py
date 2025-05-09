"""
Timeframe Adjustment Processor

This module implements the calculation of timeframe adjustments based on feedback analysis.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.feedback_exceptions import TimeframeFeedbackError

from ..models import TimeframeAdjustment, TimeframeCorrelation, TimeframeInsight

logger = get_logger(__name__)


class TimeframeAdjustmentProcessor:
    """Calculates adjustments for timeframes based on feedback analysis."""
    
    def __init__(self, max_adjustment: float = 0.5):
        """
        Initialize the timeframe adjustment processor.
        
        Args:
            max_adjustment: Maximum adjustment factor
        """
        self.max_adjustment = max_adjustment
        logger.info("TimeframeAdjustmentProcessor initialized")
    
    def calculate_adjustments(
        self,
        instrument: str,
        correlations: List[TimeframeCorrelation],
        insights: TimeframeInsight
    ) -> Dict[str, TimeframeAdjustment]:
        """
        Calculate adjustments for timeframes.
        
        Args:
            instrument: The instrument
            correlations: Timeframe correlations
            insights: Timeframe insights
            
        Returns:
            Dict[str, TimeframeAdjustment]: Adjustments by timeframe
        """
        try:
            adjustments = {}
            
            # Process each timeframe from insights
            for timeframe in insights.timeframes_analyzed:
                # Initialize adjustment
                adjustment = TimeframeAdjustment(
                    instrument=instrument,
                    target_timeframe=timeframe,
                    calculation_time=datetime.utcnow()
                )
                
                # Apply confidence adjustment based on accuracy
                confidence_adj = self._calculate_confidence_adjustment(timeframe, insights)
                adjustment.confidence_adjustment = confidence_adj
                
                # Apply error magnitude adjustment based on error patterns
                error_adj = self._calculate_error_adjustment(timeframe, insights)
                adjustment.error_magnitude_adjustment = error_adj
                
                # Apply prediction bias adjustment based on correlations
                bias_adj = self._calculate_bias_adjustment(timeframe, correlations)
                adjustment.prediction_bias_adjustment = bias_adj
                
                # Generate recommended actions
                actions = self._generate_recommended_actions(
                    timeframe, confidence_adj, error_adj, bias_adj, insights
                )
                adjustment.recommended_actions = actions
                
                # Add to results
                adjustments[timeframe] = adjustment
                
                logger.debug(
                    f"Calculated adjustment for {timeframe}: "
                    f"confidence={confidence_adj:.2f}, error={error_adj:.2f}, bias={bias_adj:.2f}"
                )
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating timeframe adjustments: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to calculate timeframe adjustments: {str(e)}")
    
    def _calculate_confidence_adjustment(
        self, timeframe: str, insights: TimeframeInsight
    ) -> float:
        """
        Calculate confidence adjustment based on accuracy.
        
        Args:
            timeframe: The timeframe
            insights: Timeframe insights
            
        Returns:
            float: Confidence adjustment factor
        """
        # Default: no adjustment
        adjustment = 0.0
        
        # Check if timeframe is in most accurate
        for tf_data in insights.most_accurate_timeframes:
            if tf_data["timeframe"] == timeframe:
                # Increase confidence for accurate timeframes
                adjustment = 0.2
                break
        
        # Check if timeframe is in least accurate
        for tf_data in insights.least_accurate_timeframes:
            if tf_data["timeframe"] == timeframe:
                # Decrease confidence for inaccurate timeframes
                adjustment = -0.3
                break
        
        # Check for error patterns
        for pattern in insights.error_patterns:
            if pattern["timeframe"] == timeframe:
                if pattern["type"] == "spikes":
                    # Spikes indicate unpredictability
                    adjustment -= 0.1
                elif pattern["type"] == "trend" and pattern["direction"] == "degrading":
                    # Degrading performance
                    adjustment -= 0.2
                elif pattern["type"] == "trend" and pattern["direction"] == "improving":
                    # Improving performance
                    adjustment += 0.1
        
        # Ensure adjustment is within bounds
        return max(-self.max_adjustment, min(self.max_adjustment, adjustment))
    
    def _calculate_error_adjustment(
        self, timeframe: str, insights: TimeframeInsight
    ) -> float:
        """
        Calculate error magnitude adjustment based on error patterns.
        
        Args:
            timeframe: The timeframe
            insights: Timeframe insights
            
        Returns:
            float: Error magnitude adjustment factor
        """
        # Default: no adjustment
        adjustment = 0.0
        
        # Check for error patterns
        for pattern in insights.error_patterns:
            if pattern["timeframe"] == timeframe:
                if pattern["type"] == "spikes":
                    # Increase error estimates for spikey timeframes
                    spike_count = pattern.get("count", 0)
                    if spike_count > 5:
                        adjustment = 0.3
                    elif spike_count > 2:
                        adjustment = 0.2
                    else:
                        adjustment = 0.1
                
                elif pattern["type"] == "cyclical":
                    # Moderate adjustment for cyclical patterns
                    adjustment = 0.15
                
                elif pattern["type"] == "trend":
                    if pattern["direction"] == "degrading":
                        # Increase error estimates for degrading performance
                        magnitude = pattern.get("magnitude", 0.0)
                        adjustment = min(0.4, magnitude * 5)
                    elif pattern["direction"] == "improving":
                        # Decrease error estimates for improving performance
                        magnitude = pattern.get("magnitude", 0.0)
                        adjustment = max(-0.3, -magnitude * 4)
        
        # Ensure adjustment is within bounds
        return max(-self.max_adjustment, min(self.max_adjustment, adjustment))
    
    def _calculate_bias_adjustment(
        self, timeframe: str, correlations: List[TimeframeCorrelation]
    ) -> float:
        """
        Calculate prediction bias adjustment based on correlations.
        
        Args:
            timeframe: The timeframe
            correlations: Timeframe correlations
            
        Returns:
            float: Prediction bias adjustment factor
        """
        # Default: no adjustment
        adjustment = 0.0
        
        # Count strong positive and negative correlations
        pos_count = 0
        neg_count = 0
        
        for corr in correlations:
            if corr.timeframe1 == timeframe or corr.timeframe2 == timeframe:
                if corr.significance == "strong":
                    if corr.correlation > 0:
                        pos_count += 1
                    else:
                        neg_count += 1
        
        # Calculate bias based on correlation imbalance
        if pos_count + neg_count > 0:
            bias = (pos_count - neg_count) / (pos_count + neg_count)
            adjustment = bias * 0.2
        
        # Ensure adjustment is within bounds
        return max(-self.max_adjustment, min(self.max_adjustment, adjustment))
    
    def _generate_recommended_actions(
        self,
        timeframe: str,
        confidence_adj: float,
        error_adj: float,
        bias_adj: float,
        insights: TimeframeInsight
    ) -> List[str]:
        """
        Generate recommended actions based on adjustments.
        
        Args:
            timeframe: The timeframe
            confidence_adj: Confidence adjustment
            error_adj: Error magnitude adjustment
            bias_adj: Prediction bias adjustment
            insights: Timeframe insights
            
        Returns:
            List[str]: Recommended actions
        """
        actions = []
        
        # Confidence adjustment actions
        if confidence_adj <= -0.2:
            actions.append(f"Reduce weight of {timeframe} predictions in decision making")
        elif confidence_adj >= 0.2:
            actions.append(f"Increase weight of {timeframe} predictions in decision making")
        
        # Error adjustment actions
        if error_adj >= 0.2:
            actions.append(f"Widen stop loss levels for {timeframe} based trades")
            actions.append(f"Reduce position sizes for {timeframe} based trades")
        elif error_adj <= -0.2:
            actions.append(f"Consider tighter stop loss levels for {timeframe} based trades")
        
        # Bias adjustment actions
        if abs(bias_adj) >= 0.1:
            direction = "upward" if bias_adj > 0 else "downward"
            actions.append(f"Apply {direction} bias correction to {timeframe} predictions")
        
        # Check for specific recommendations in insights
        for rec in insights.recommendations:
            if timeframe in rec.get("timeframes", []):
                if rec["type"] == "focus" and rec["priority"] == "high":
                    actions.append(f"Prioritize {timeframe} signals in trading decisions")
                elif rec["type"] == "caution" and rec["priority"] == "high":
                    actions.append(f"Use {timeframe} signals only with confirmation from other timeframes")
                elif rec["type"] == "investigate" and rec["priority"] == "medium":
                    actions.append(f"Investigate degrading performance in {timeframe}")
                elif rec["type"] == "retrain" and rec["priority"] == "high":
                    actions.append(f"Retrain models for {timeframe}")
                elif rec["type"] == "time_adjust" and rec["priority"] == "medium":
                    actions.append(f"Apply time-of-day adjustments to {timeframe} predictions")
        
        return actions