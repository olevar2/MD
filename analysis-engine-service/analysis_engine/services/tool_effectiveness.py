"""
Tool Effectiveness Tracking Service

This module provides functionality to track and analyze the effectiveness
of different analysis tools across various market conditions.
"""
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
from dataclasses import dataclass
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    CONSOLIDATING = 'consolidating'
    BREAKOUT = 'breakout'
    REVERSAL = 'reversal'
    UNKNOWN = 'unknown'


@dataclass
class PredictionResult:
    """Container for prediction results"""
    tool_id: str
    timestamp: datetime
    market_regime: MarketRegime
    prediction: str
    actual_outcome: Optional[str] = None
    confidence: float = 0.0
    impact: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def is_correct(self) ->bool:
        """Check if prediction was correct"""
        return (self.actual_outcome is not None and self.prediction == self
            .actual_outcome)

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary representation"""
        return {'tool_id': self.tool_id, 'timestamp': self.timestamp.
            isoformat(), 'market_regime': self.market_regime.value,
            'prediction': self.prediction, 'actual_outcome': self.
            actual_outcome, 'confidence': self.confidence, 'impact': self.
            impact, 'metadata': self.metadata or {}}


class ToolEffectivenessTracker:
    """
    Tracks and analyzes the effectiveness of analysis tools
    
    Features:
    - Performance tracking by market regime
    - Confidence-weighted effectiveness scores
    - Impact-adjusted metrics
    - Adaptive tool selection
    - Historical performance analysis
    """

    def __init__(self, window_size: int=100):
        """
        Initialize effectiveness tracker
        
        Args:
            window_size: Number of predictions to keep in rolling window
        """
        self.window_size = window_size
        self.predictions: Dict[str, List[PredictionResult]] = {}
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {regime
            : {} for regime in MarketRegime}
        self.logger = logging.getLogger(f'{__name__}.ToolEffectivenessTracker')

    @with_exception_handling
    def record_prediction(self, result: PredictionResult):
        """
        Record a new prediction result
        
        Args:
            result: Prediction result to record
        """
        try:
            if result.tool_id not in self.predictions:
                self.predictions[result.tool_id] = []
            self.predictions[result.tool_id].append(result)
            if len(self.predictions[result.tool_id]) > self.window_size:
                self.predictions[result.tool_id].pop(0)
            if result.actual_outcome is not None:
                self._update_regime_performance(result)
        except Exception as e:
            self.logger.error(f'Error recording prediction: {str(e)}')

    @with_exception_handling
    def _update_regime_performance(self, result: PredictionResult):
        """Update performance metrics for market regime"""
        try:
            regime = result.market_regime
            tool_id = result.tool_id
            if tool_id not in self.regime_performance[regime]:
                self.regime_performance[regime][tool_id] = 0.0
            score = result.impact * (1.0 if result.is_correct() else -1.0)
            alpha = 0.1
            current_score = self.regime_performance[regime][tool_id]
            self.regime_performance[regime][tool_id] = alpha * score + (1 -
                alpha) * current_score
        except Exception as e:
            self.logger.error(f'Error updating regime performance: {str(e)}',
                exc_info=True)

    @with_resilience('get_tool_effectiveness')
    @with_exception_handling
    def get_tool_effectiveness(self, tool_id: str, market_regime: Optional[
        MarketRegime]=None, lookback: Optional[timedelta]=None) ->Dict[str, Any
        ]:
        """
        Get effectiveness metrics for a tool
        
        Args:
            tool_id: Tool identifier
            market_regime: Optional market regime filter
            lookback: Optional time window to consider
            
        Returns:
            Dictionary containing effectiveness metrics
        """
        try:
            if tool_id not in self.predictions:
                return {'tool_id': tool_id, 'prediction_count': 0,
                    'accuracy': 0.0, 'average_impact': 0.0,
                    'regime_performance': {}}
            predictions = self.predictions[tool_id]
            if market_regime:
                predictions = [p for p in predictions if p.market_regime ==
                    market_regime]
            if lookback:
                cutoff = datetime.now() - lookback
                predictions = [p for p in predictions if p.timestamp >= cutoff]
            completed_predictions = [p for p in predictions if p.
                actual_outcome is not None]
            if not completed_predictions:
                return {'tool_id': tool_id, 'prediction_count': 0,
                    'accuracy': 0.0, 'average_impact': 0.0,
                    'regime_performance': {}}
            accuracy = np.mean([p.is_correct() for p in completed_predictions])
            average_impact = np.mean([p.impact for p in completed_predictions])
            regime_performance = {regime.value: self.regime_performance[
                regime].get(tool_id, 0.0) for regime in MarketRegime if 
                tool_id in self.regime_performance[regime]}
            return {'tool_id': tool_id, 'prediction_count': len(
                completed_predictions), 'accuracy': accuracy,
                'average_impact': average_impact, 'regime_performance':
                regime_performance}
        except Exception as e:
            self.logger.error(f'Error getting tool effectiveness: {str(e)}',
                exc_info=True)
            return {'tool_id': tool_id, 'error': str(e)}

    @with_resilience('get_best_tools')
    @with_exception_handling
    def get_best_tools(self, market_regime: MarketRegime, min_predictions:
        int=10, top_n: Optional[int]=None) ->List[Dict[str, Any]]:
        """
        Get best performing tools for a market regime
        
        Args:
            market_regime: Market regime to analyze
            min_predictions: Minimum number of predictions required
            top_n: Optional limit on number of tools to return
            
        Returns:
            List of tool performance metrics, sorted by effectiveness
        """
        try:
            tool_metrics = []
            for tool_id in self.predictions:
                metrics = self.get_tool_effectiveness(tool_id,
                    market_regime=market_regime)
                if metrics['prediction_count'] >= min_predictions:
                    tool_metrics.append(metrics)
            sorted_metrics = sorted(tool_metrics, key=lambda x: (x[
                'regime_performance'].get(market_regime.value, 0.0), x[
                'accuracy']), reverse=True)
            if top_n:
                sorted_metrics = sorted_metrics[:top_n]
            return sorted_metrics
        except Exception as e:
            self.logger.error(f'Error getting best tools: {str(e)}',
                exc_info=True)
            return []

    @with_resilience('get_tool_history')
    @with_exception_handling
    def get_tool_history(self, tool_id: str, limit: Optional[int]=None) ->List[
        Dict[str, Any]]:
        """
        Get prediction history for a tool
        
        Args:
            tool_id: Tool identifier
            limit: Optional limit on number of predictions to return
            
        Returns:
            List of prediction results
        """
        try:
            if tool_id not in self.predictions:
                return []
            history = [pred.to_dict() for pred in reversed(self.predictions
                [tool_id])]
            if limit:
                history = history[:limit]
            return history
        except Exception as e:
            self.logger.error(f'Error getting tool history: {str(e)}',
                exc_info=True)
            return []

    @with_resilience('get_performance_summary')
    @with_exception_handling
    def get_performance_summary(self) ->Dict[str, Any]:
        """Get overall performance summary for all tools"""
        try:
            summary = {'total_tools': len(self.predictions),
                'total_predictions': sum(len(preds) for preds in self.
                predictions.values()), 'regime_performance': {},
                'tool_rankings': {}}
            for regime in MarketRegime:
                best_tools = self.get_best_tools(regime, min_predictions=10,
                    top_n=5)
                summary['regime_performance'][regime.value] = {'top_tools':
                    [{'tool_id': tool['tool_id'], 'accuracy': tool[
                    'accuracy'], 'impact': tool['average_impact']} for tool in
                    best_tools]}
            all_tools = []
            for tool_id in self.predictions:
                metrics = self.get_tool_effectiveness(tool_id)
                if metrics['prediction_count'] > 0:
                    all_tools.append(metrics)
            sorted_tools = sorted(all_tools, key=lambda x: (x['accuracy'],
                x['average_impact']), reverse=True)
            summary['tool_rankings'] = {'by_accuracy': [{'tool_id': tool[
                'tool_id'], 'accuracy': tool['accuracy']} for tool in
                sorted(all_tools, key=lambda x: x['accuracy'], reverse=True
                )[:5]], 'by_impact': [{'tool_id': tool['tool_id'], 'impact':
                tool['average_impact']} for tool in sorted(all_tools, key=
                lambda x: x['average_impact'], reverse=True)[:5]]}
            return summary
        except Exception as e:
            self.logger.error(f'Error getting performance summary: {str(e)}',
                exc_info=True)
            return {'error': str(e)}
