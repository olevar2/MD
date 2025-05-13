"""
Timeframe Feedback Models

This module defines data models for timeframe feedback analysis.
"""
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import json


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TimeframeCorrelation:
    """Represents correlation between two timeframes."""

    def __init__(self, instrument: str, timeframe1: str, timeframe2: str,
        correlation: float=0.0, significance: str='none', sample_size: int=
        0, calculation_time: Optional[datetime]=None):
        """
        Initialize a timeframe correlation.
        
        Args:
            instrument: The instrument
            timeframe1: First timeframe
            timeframe2: Second timeframe
            correlation: Correlation coefficient (-1.0 to 1.0)
            significance: Correlation significance (none, weak, moderate, strong)
            sample_size: Number of data points used for correlation
            calculation_time: When the correlation was calculated
        """
        self.instrument = instrument
        self.timeframe1 = timeframe1
        self.timeframe2 = timeframe2
        self.correlation = correlation
        self.significance = significance
        self.sample_size = sample_size
        self.calculation_time = calculation_time or datetime.utcnow()

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary representation."""
        return {'instrument': self.instrument, 'timeframe1': self.
            timeframe1, 'timeframe2': self.timeframe2, 'correlation': self.
            correlation, 'significance': self.significance, 'sample_size':
            self.sample_size, 'calculation_time': self.calculation_time.
            isoformat()}

    @classmethod
    @with_exception_handling
    def from_dict(cls, data: Dict[str, Any]) ->'TimeframeCorrelation':
        """Create from dictionary representation."""
        calculation_time = None
        if 'calculation_time' in data:
            try:
                calculation_time = datetime.fromisoformat(data[
                    'calculation_time'])
            except (ValueError, TypeError):
                calculation_time = datetime.utcnow()
        return cls(instrument=data.get('instrument', ''), timeframe1=data.
            get('timeframe1', ''), timeframe2=data.get('timeframe2', ''),
            correlation=data.get('correlation', 0.0), significance=data.get
            ('significance', 'none'), sample_size=data.get('sample_size', 0
            ), calculation_time=calculation_time)


class TimeframeAdjustment:
    """Represents temporal adjustments for a timeframe."""

    def __init__(self, instrument: str, target_timeframe: str,
        confidence_adjustment: float=0.0, error_magnitude_adjustment: float
        =0.0, prediction_bias_adjustment: float=0.0, recommended_actions:
        List[str]=None, calculation_time: Optional[datetime]=None):
        """
        Initialize a timeframe adjustment.
        
        Args:
            instrument: The instrument
            target_timeframe: The target timeframe
            confidence_adjustment: Adjustment to confidence level (-1.0 to 1.0)
            error_magnitude_adjustment: Adjustment to error magnitude
            prediction_bias_adjustment: Adjustment to prediction bias
            recommended_actions: List of recommended actions
            calculation_time: When the adjustment was calculated
        """
        self.instrument = instrument
        self.target_timeframe = target_timeframe
        self.confidence_adjustment = confidence_adjustment
        self.error_magnitude_adjustment = error_magnitude_adjustment
        self.prediction_bias_adjustment = prediction_bias_adjustment
        self.recommended_actions = recommended_actions or []
        self.calculation_time = calculation_time or datetime.utcnow()

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary representation."""
        return {'instrument': self.instrument, 'target_timeframe': self.
            target_timeframe, 'confidence_adjustment': self.
            confidence_adjustment, 'error_magnitude_adjustment': self.
            error_magnitude_adjustment, 'prediction_bias_adjustment': self.
            prediction_bias_adjustment, 'recommended_actions': self.
            recommended_actions, 'calculation_time': self.calculation_time.
            isoformat()}

    @classmethod
    @with_exception_handling
    def from_dict(cls, data: Dict[str, Any]) ->'TimeframeAdjustment':
        """Create from dictionary representation."""
        calculation_time = None
        if 'calculation_time' in data:
            try:
                calculation_time = datetime.fromisoformat(data[
                    'calculation_time'])
            except (ValueError, TypeError):
                calculation_time = datetime.utcnow()
        return cls(instrument=data.get('instrument', ''), target_timeframe=
            data.get('target_timeframe', ''), confidence_adjustment=data.
            get('confidence_adjustment', 0.0), error_magnitude_adjustment=
            data.get('error_magnitude_adjustment', 0.0),
            prediction_bias_adjustment=data.get(
            'prediction_bias_adjustment', 0.0), recommended_actions=data.
            get('recommended_actions', []), calculation_time=calculation_time)


class TimeframeInsight:
    """Represents insights from timeframe analysis."""

    def __init__(self, instrument: str, timeframes_analyzed: List[str]=None,
        error_patterns: List[Dict[str, Any]]=None, leading_timeframes: List
        [Dict[str, Any]]=None, lagging_timeframes: List[Dict[str, Any]]=
        None, most_accurate_timeframes: List[Dict[str, Any]]=None,
        least_accurate_timeframes: List[Dict[str, Any]]=None,
        recommendations: List[Dict[str, Any]]=None, analysis_time: Optional
        [datetime]=None):
        """
        Initialize timeframe insights.
        
        Args:
            instrument: The instrument
            timeframes_analyzed: List of timeframes analyzed
            error_patterns: Patterns in prediction errors
            leading_timeframes: Timeframes that lead others
            lagging_timeframes: Timeframes that lag others
            most_accurate_timeframes: Most accurate timeframes
            least_accurate_timeframes: Least accurate timeframes
            recommendations: Recommendations based on analysis
            analysis_time: When the analysis was performed
        """
        self.instrument = instrument
        self.timeframes_analyzed = timeframes_analyzed or []
        self.error_patterns = error_patterns or []
        self.leading_timeframes = leading_timeframes or []
        self.lagging_timeframes = lagging_timeframes or []
        self.most_accurate_timeframes = most_accurate_timeframes or []
        self.least_accurate_timeframes = least_accurate_timeframes or []
        self.recommendations = recommendations or []
        self.analysis_time = analysis_time or datetime.utcnow()

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary representation."""
        return {'instrument': self.instrument, 'timeframes_analyzed': self.
            timeframes_analyzed, 'error_patterns': self.error_patterns,
            'leading_timeframes': self.leading_timeframes,
            'lagging_timeframes': self.lagging_timeframes,
            'most_accurate_timeframes': self.most_accurate_timeframes,
            'least_accurate_timeframes': self.least_accurate_timeframes,
            'recommendations': self.recommendations, 'analysis_time': self.
            analysis_time.isoformat()}

    @classmethod
    @with_exception_handling
    def from_dict(cls, data: Dict[str, Any]) ->'TimeframeInsight':
        """Create from dictionary representation."""
        analysis_time = None
        if 'analysis_time' in data:
            try:
                analysis_time = datetime.fromisoformat(data['analysis_time'])
            except (ValueError, TypeError):
                analysis_time = datetime.utcnow()
        return cls(instrument=data.get('instrument', ''),
            timeframes_analyzed=data.get('timeframes_analyzed', []),
            error_patterns=data.get('error_patterns', []),
            leading_timeframes=data.get('leading_timeframes', []),
            lagging_timeframes=data.get('lagging_timeframes', []),
            most_accurate_timeframes=data.get('most_accurate_timeframes', [
            ]), least_accurate_timeframes=data.get(
            'least_accurate_timeframes', []), recommendations=data.get(
            'recommendations', []), analysis_time=analysis_time)


@with_exception_handling
def extract_timeframe_from_feedback(feedback: Any) ->Optional[str]:
    """
    Extract timeframe information from feedback metadata.
    
    Args:
        feedback: Feedback object with metadata
        
    Returns:
        Timeframe string or None if not found
    """
    timeframe = None
    if hasattr(feedback, 'metadata'):
        metadata = feedback.metadata
        if isinstance(metadata, str):
            try:
                metadata_dict = json.loads(metadata)
                timeframe = metadata_dict.get('timeframe')
            except:
                pass
        elif isinstance(metadata, dict):
            timeframe = metadata.get('timeframe')
    return timeframe
