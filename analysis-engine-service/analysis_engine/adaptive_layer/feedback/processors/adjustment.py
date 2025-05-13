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
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class TimeframeAdjustmentProcessor:
    """Calculates adjustments for timeframes based on feedback analysis."""

    def __init__(self, max_adjustment: float=0.5):
        """
        Initialize the timeframe adjustment processor.
        
        Args:
            max_adjustment: Maximum adjustment factor
        """
        self.max_adjustment = max_adjustment
        logger.info('TimeframeAdjustmentProcessor initialized')

    @with_analysis_resilience('calculate_adjustments')
    @with_exception_handling
    def calculate_adjustments(self, instrument: str, correlations: List[
        TimeframeCorrelation], insights: TimeframeInsight) ->Dict[str,
        TimeframeAdjustment]:
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
            for timeframe in insights.timeframes_analyzed:
                adjustment = TimeframeAdjustment(instrument=instrument,
                    target_timeframe=timeframe, calculation_time=datetime.
                    utcnow())
                confidence_adj = self._calculate_confidence_adjustment(
                    timeframe, insights)
                adjustment.confidence_adjustment = confidence_adj
                error_adj = self._calculate_error_adjustment(timeframe,
                    insights)
                adjustment.error_magnitude_adjustment = error_adj
                bias_adj = self._calculate_bias_adjustment(timeframe,
                    correlations)
                adjustment.prediction_bias_adjustment = bias_adj
                actions = self._generate_recommended_actions(timeframe,
                    confidence_adj, error_adj, bias_adj, insights)
                adjustment.recommended_actions = actions
                adjustments[timeframe] = adjustment
                logger.debug(
                    f'Calculated adjustment for {timeframe}: confidence={confidence_adj:.2f}, error={error_adj:.2f}, bias={bias_adj:.2f}'
                    )
            return adjustments
        except Exception as e:
            logger.error(f'Error calculating timeframe adjustments: {str(e)}',
                exc_info=True)
            raise TimeframeFeedbackError(
                f'Failed to calculate timeframe adjustments: {str(e)}')

    def _calculate_confidence_adjustment(self, timeframe: str, insights:
        TimeframeInsight) ->float:
        """
        Calculate confidence adjustment based on accuracy.
        
        Args:
            timeframe: The timeframe
            insights: Timeframe insights
            
        Returns:
            float: Confidence adjustment factor
        """
        adjustment = 0.0
        for tf_data in insights.most_accurate_timeframes:
            if tf_data['timeframe'] == timeframe:
                adjustment = 0.2
                break
        for tf_data in insights.least_accurate_timeframes:
            if tf_data['timeframe'] == timeframe:
                adjustment = -0.3
                break
        for pattern in insights.error_patterns:
            if pattern['timeframe'] == timeframe:
                if pattern['type'] == 'spikes':
                    adjustment -= 0.1
                elif pattern['type'] == 'trend' and pattern['direction'
                    ] == 'degrading':
                    adjustment -= 0.2
                elif pattern['type'] == 'trend' and pattern['direction'
                    ] == 'improving':
                    adjustment += 0.1
        return max(-self.max_adjustment, min(self.max_adjustment, adjustment))

    def _calculate_error_adjustment(self, timeframe: str, insights:
        TimeframeInsight) ->float:
        """
        Calculate error magnitude adjustment based on error patterns.
        
        Args:
            timeframe: The timeframe
            insights: Timeframe insights
            
        Returns:
            float: Error magnitude adjustment factor
        """
        adjustment = 0.0
        for pattern in insights.error_patterns:
            if pattern['timeframe'] == timeframe:
                if pattern['type'] == 'spikes':
                    spike_count = pattern.get('count', 0)
                    if spike_count > 5:
                        adjustment = 0.3
                    elif spike_count > 2:
                        adjustment = 0.2
                    else:
                        adjustment = 0.1
                elif pattern['type'] == 'cyclical':
                    adjustment = 0.15
                elif pattern['type'] == 'trend':
                    if pattern['direction'] == 'degrading':
                        magnitude = pattern.get('magnitude', 0.0)
                        adjustment = min(0.4, magnitude * 5)
                    elif pattern['direction'] == 'improving':
                        magnitude = pattern.get('magnitude', 0.0)
                        adjustment = max(-0.3, -magnitude * 4)
        return max(-self.max_adjustment, min(self.max_adjustment, adjustment))

    def _calculate_bias_adjustment(self, timeframe: str, correlations: List
        [TimeframeCorrelation]) ->float:
        """
        Calculate prediction bias adjustment based on correlations.
        
        Args:
            timeframe: The timeframe
            correlations: Timeframe correlations
            
        Returns:
            float: Prediction bias adjustment factor
        """
        adjustment = 0.0
        pos_count = 0
        neg_count = 0
        for corr in correlations:
            if corr.timeframe1 == timeframe or corr.timeframe2 == timeframe:
                if corr.significance == 'strong':
                    if corr.correlation > 0:
                        pos_count += 1
                    else:
                        neg_count += 1
        if pos_count + neg_count > 0:
            bias = (pos_count - neg_count) / (pos_count + neg_count)
            adjustment = bias * 0.2
        return max(-self.max_adjustment, min(self.max_adjustment, adjustment))

    def _generate_recommended_actions(self, timeframe: str, confidence_adj:
        float, error_adj: float, bias_adj: float, insights: TimeframeInsight
        ) ->List[str]:
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
        if confidence_adj <= -0.2:
            actions.append(
                f'Reduce weight of {timeframe} predictions in decision making')
        elif confidence_adj >= 0.2:
            actions.append(
                f'Increase weight of {timeframe} predictions in decision making'
                )
        if error_adj >= 0.2:
            actions.append(
                f'Widen stop loss levels for {timeframe} based trades')
            actions.append(
                f'Reduce position sizes for {timeframe} based trades')
        elif error_adj <= -0.2:
            actions.append(
                f'Consider tighter stop loss levels for {timeframe} based trades'
                )
        if abs(bias_adj) >= 0.1:
            direction = 'upward' if bias_adj > 0 else 'downward'
            actions.append(
                f'Apply {direction} bias correction to {timeframe} predictions'
                )
        for rec in insights.recommendations:
            if timeframe in rec.get('timeframes', []):
                if rec['type'] == 'focus' and rec['priority'] == 'high':
                    actions.append(
                        f'Prioritize {timeframe} signals in trading decisions')
                elif rec['type'] == 'caution' and rec['priority'] == 'high':
                    actions.append(
                        f'Use {timeframe} signals only with confirmation from other timeframes'
                        )
                elif rec['type'] == 'investigate' and rec['priority'
                    ] == 'medium':
                    actions.append(
                        f'Investigate degrading performance in {timeframe}')
                elif rec['type'] == 'retrain' and rec['priority'] == 'high':
                    actions.append(f'Retrain models for {timeframe}')
                elif rec['type'] == 'time_adjust' and rec['priority'
                    ] == 'medium':
                    actions.append(
                        f'Apply time-of-day adjustments to {timeframe} predictions'
                        )
        return actions
