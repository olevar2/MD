"""
Timeframe Feedback Service

This module implements a service for collecting, analyzing, and applying timeframe-specific
feedback to improve prediction accuracy across different timeframes.

The service coordinates the collection of feedback from various sources, analyzes patterns
and correlations between timeframes, and calculates adjustments to improve prediction accuracy.

Note: This is a facade module that re-exports the refactored implementation from the feedback package.
"""

# Re-export from the refactored package
from analysis_engine.adaptive_layer.feedback import (
    TimeframeCorrelation,
    TimeframeAdjustment,
    TimeframeInsight,
    extract_timeframe_from_feedback,
    TimeframeFeedbackService
)

# For backward compatibility
from analysis_engine.adaptive_layer.feedback.collectors.timeframe_collector import TimeframeFeedbackCollector
from analysis_engine.adaptive_layer.feedback.analyzers.correlation import TimeframeCorrelationAnalyzer
from analysis_engine.adaptive_layer.feedback.analyzers.temporal import TemporalFeedbackAnalyzer
from analysis_engine.adaptive_layer.feedback.processors.adjustment import TimeframeAdjustmentProcessor

__all__ = [
    'TimeframeCorrelation',
    'TimeframeAdjustment',
    'TimeframeInsight',
    'extract_timeframe_from_feedback',
    'TimeframeFeedbackService',
    'TimeframeFeedbackCollector',
    'TimeframeCorrelationAnalyzer',
    'TemporalFeedbackAnalyzer',
    'TimeframeAdjustmentProcessor'
]