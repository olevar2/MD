"""
Tool Effectiveness Adapter Module

This module provides adapters for tool effectiveness functionality,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid
from common_lib.effectiveness.interfaces import MarketRegimeEnum, TimeFrameEnum, SignalEvent, SignalOutcome, IToolEffectivenessTracker, IToolEffectivenessRepository
from analysis_engine.services.tool_effectiveness import ToolEffectivenessTracker, MarketRegime, PredictionResult
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ToolEffectivenessTrackerAdapter(IToolEffectivenessTracker):
    """
    Adapter for ToolEffectivenessTracker that implements the common interface.
    
    This adapter wraps the actual ToolEffectivenessTracker implementation
    and adapts it to the common interface.
    """

    def __init__(self, tracker_instance=None):
        """
        Initialize the adapter.
        
        Args:
            tracker_instance: Optional tracker instance to wrap
        """
        self.tracker = tracker_instance or ToolEffectivenessTracker()
        self.logger = logging.getLogger(__name__)
        self.signal_id_map = {}

    @with_exception_handling
    def record_signal(self, signal: SignalEvent) ->str:
        """Record a new trading signal."""
        try:
            signal_id = str(uuid.uuid4())
            prediction = PredictionResult(tool_id=signal.tool_id, timestamp
                =signal.timestamp, market_regime=self.
                _convert_market_regime(signal.market_regime), prediction=
                signal.direction, confidence=signal.confidence, impact=1.0,
                metadata=signal.metadata)
            self.tracker.record_prediction(prediction)
            self.signal_id_map[signal_id] = prediction
            return signal_id
        except Exception as e:
            self.logger.error(f'Error recording signal: {str(e)}')
            return str(uuid.uuid4())

    @with_exception_handling
    def record_outcome(self, outcome: SignalOutcome) ->bool:
        """Record the outcome of a signal."""
        try:
            prediction = self.signal_id_map.get(outcome.signal_id)
            if not prediction:
                self.logger.warning(
                    f'No prediction found for signal ID: {outcome.signal_id}')
                return False
            prediction.actual_outcome = outcome.result
            prediction.impact = abs(outcome.pips)
            if prediction.metadata is None:
                prediction.metadata = {}
            prediction.metadata.update({'profit_loss': outcome.profit_loss,
                'exit_timestamp': outcome.exit_timestamp.isoformat()})
            self.tracker.record_prediction(prediction)
            return True
        except Exception as e:
            self.logger.error(f'Error recording outcome: {str(e)}')
            return False

    @with_resilience('get_tool_effectiveness')
    @with_exception_handling
    def get_tool_effectiveness(self, tool_id: str, market_regime: Optional[
        str]=None, timeframe: Optional[str]=None, symbol: Optional[str]=
        None, start_date: Optional[datetime]=None, end_date: Optional[
        datetime]=None) ->Dict[str, Any]:
        """Get effectiveness metrics for a tool."""
        try:
            regime = self._convert_market_regime(market_regime
                ) if market_regime else None
            lookback = None
            if start_date and end_date:
                lookback = end_date - start_date
            metrics = self.tracker.get_tool_effectiveness(tool_id=tool_id,
                market_regime=regime, lookback=lookback)
            metrics['timeframe'] = timeframe
            metrics['symbol'] = symbol
            return metrics
        except Exception as e:
            self.logger.error(f'Error getting tool effectiveness: {str(e)}')
            return {'tool_id': tool_id, 'prediction_count': 0, 'accuracy': 
                0.0, 'average_impact': 0.0, 'regime_performance': {},
                'timeframe': timeframe, 'symbol': symbol}

    @with_resilience('get_best_tools')
    @with_exception_handling
    def get_best_tools(self, market_regime: str, timeframe: Optional[str]=
        None, symbol: Optional[str]=None, min_signals: int=10, top_n:
        Optional[int]=None) ->List[Dict[str, Any]]:
        """Get best performing tools for a market regime."""
        try:
            regime = self._convert_market_regime(market_regime)
            tools = self.tracker.get_best_tools(market_regime=regime,
                min_predictions=min_signals, top_n=top_n)
            for tool in tools:
                tool['timeframe'] = timeframe
                tool['symbol'] = symbol
            return tools
        except Exception as e:
            self.logger.error(f'Error getting best tools: {str(e)}')
            return []

    @with_resilience('get_tool_history')
    @with_exception_handling
    def get_tool_history(self, tool_id: str, limit: Optional[int]=None,
        start_date: Optional[datetime]=None, end_date: Optional[datetime]=None
        ) ->List[Dict[str, Any]]:
        """Get signal history for a tool."""
        try:
            history = self.tracker.get_tool_history(tool_id=tool_id, limit=
                limit)
            if start_date or end_date:
                filtered_history = []
                for item in history:
                    timestamp = datetime.fromisoformat(item['timestamp'])
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    filtered_history.append(item)
                return filtered_history
            return history
        except Exception as e:
            self.logger.error(f'Error getting tool history: {str(e)}')
            return []

    @with_resilience('get_performance_summary')
    @with_exception_handling
    def get_performance_summary(self, timeframe: Optional[str]=None, symbol:
        Optional[str]=None, start_date: Optional[datetime]=None, end_date:
        Optional[datetime]=None) ->Dict[str, Any]:
        """Get overall performance summary for all tools."""
        try:
            summary = self.tracker.get_performance_summary()
            summary['timeframe'] = timeframe
            summary['symbol'] = symbol
            return summary
        except Exception as e:
            self.logger.error(f'Error getting performance summary: {str(e)}')
            return {'total_tools': 0, 'total_predictions': 0,
                'regime_performance': {}, 'tool_rankings': {}, 'timeframe':
                timeframe, 'symbol': symbol}

    @with_exception_handling
    def _convert_market_regime(self, regime_str: str) ->MarketRegime:
        """Convert string market regime to enum."""
        try:
            regime_enum = MarketRegimeEnum(regime_str)
            regime_map = {MarketRegimeEnum.TRENDING_UP: MarketRegime.
                TRENDING_UP, MarketRegimeEnum.TRENDING_DOWN: MarketRegime.
                TRENDING_DOWN, MarketRegimeEnum.RANGING: MarketRegime.
                RANGING, MarketRegimeEnum.VOLATILE: MarketRegime.VOLATILE,
                MarketRegimeEnum.CONSOLIDATING: MarketRegime.CONSOLIDATING,
                MarketRegimeEnum.BREAKOUT: MarketRegime.BREAKOUT,
                MarketRegimeEnum.REVERSAL: MarketRegime.REVERSAL,
                MarketRegimeEnum.UNKNOWN: MarketRegime.UNKNOWN}
            return regime_map.get(regime_enum, MarketRegime.UNKNOWN)
        except (ValueError, KeyError):
            try:
                return MarketRegime(regime_str)
            except ValueError:
                return MarketRegime.UNKNOWN
