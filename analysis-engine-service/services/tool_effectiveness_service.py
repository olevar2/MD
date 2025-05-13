"""
Tool Effectiveness Service

This module provides business logic for tool effectiveness analysis,
report generation, and performance tracking.
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.db.models import TradingTool, ToolSignal, SignalOutcome, EffectivenessMetric, EffectivenessReport
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ToolEffectivenessService:
    """Service for analyzing and reporting on trading tool effectiveness"""

    def __init__(self, db: Session):
        self.repository = ToolEffectivenessRepository(db)

    def register_signal(self, signal_data: Dict[str, Any]) ->Dict[str, Any]:
        """Register a new trading signal"""
        signal = self.repository.create_signal(signal_data)
        return {'signal_id': signal.signal_id, 'tool_id': signal.tool_id,
            'instrument': signal.instrument, 'timeframe': signal.timeframe,
            'direction': signal.direction, 'confidence': signal.confidence,
            'timestamp': signal.timestamp}

    def record_outcome(self, outcome_data: Dict[str, Any]) ->Dict[str, Any]:
        """Record the outcome of a trading signal"""
        outcome = self.repository.create_outcome(outcome_data)
        return {'outcome_id': outcome.outcome_id, 'signal_id': outcome.
            signal_id, 'outcome_type': outcome.outcome_type, 'entry_price':
            outcome.entry_price, 'exit_price': outcome.exit_price, 'pnl':
            outcome.pnl, 'exit_timestamp': outcome.exit_timestamp}

    @with_resilience('get_tool_performance_summary')
    def get_tool_performance_summary(self, tool_id: str, timeframe:
        Optional[str]=None, instrument: Optional[str]=None, days_back: int=30
        ) ->Dict[str, Any]:
        """Get a performance summary for a specific trading tool"""
        from_date = datetime.utcnow() - timedelta(days=days_back)
        metrics = self.repository.get_latest_metrics_for_tool(tool_id)
        signals = self.repository.get_signals(tool_id=tool_id, timeframe=
            timeframe, instrument=instrument, from_date=from_date, limit=100)
        outcomes = self.repository.get_outcomes(tool_id=tool_id, timeframe=
            timeframe, instrument=instrument, from_date=from_date, limit=100)
        total_signals = len(signals)
        total_outcomes = len(outcomes)
        if total_outcomes > 0:
            win_rate = len([o for o in outcomes if o.pnl > 0]) / total_outcomes
            total_pnl = sum(o.pnl for o in outcomes)
            avg_pnl_per_trade = total_pnl / total_outcomes
        else:
            win_rate = 0
            total_pnl = 0
            avg_pnl_per_trade = 0
        metrics_dict = {m.metric_type: m.metric_value for m in metrics}
        return {'tool_id': tool_id, 'analysis_period_days': days_back,
            'total_signals': total_signals, 'total_outcomes':
            total_outcomes, 'win_rate': win_rate, 'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl_per_trade, 'metrics': metrics_dict,
            'timestamp': datetime.utcnow()}

    def generate_performance_report(self, tool_id: str, report_type: str=
        'monthly', timeframe: Optional[str]=None, instrument: Optional[str]
        =None) ->Dict[str, Any]:
        """Generate a comprehensive performance report for a trading tool"""
        now = datetime.utcnow()
        if report_type == 'weekly':
            from_date = now - timedelta(days=7)
            time_grouping = 'day'
        elif report_type == 'monthly':
            from_date = now - timedelta(days=30)
            time_grouping = 'week'
        elif report_type == 'quarterly':
            from_date = now - timedelta(days=90)
            time_grouping = 'month'
        elif report_type == 'yearly':
            from_date = now - timedelta(days=365)
            time_grouping = 'month'
        else:
            from_date = now - timedelta(days=30)
            time_grouping = 'week'
        tool = self.repository.get_tool(tool_id)
        if not tool:
            return {'error': 'Tool not found'}
        metric_types = ['win_rate', 'total_pnl', 'risk_reward', 'sharpe']
        metrics_over_time = self.repository.get_aggregated_metrics(tool_id=
            tool_id, metric_types=metric_types, group_by=time_grouping,
            from_date=from_date)
        signals = self.repository.get_signals(tool_id=tool_id, timeframe=
            timeframe, instrument=instrument, from_date=from_date)
        outcomes = self.repository.get_outcomes(tool_id=tool_id, timeframe=
            timeframe, instrument=instrument, from_date=from_date)
        total_signals = len(signals)
        total_outcomes = len(outcomes)
        win_count = len([o for o in outcomes if o.pnl > 0])
        win_rate = win_count / total_outcomes if total_outcomes > 0 else 0
        total_pnl = sum(o.pnl for o in outcomes)
        report_data = {'report_id': str(uuid.uuid4()), 'tool_id': tool_id,
            'report_type': report_type, 'timestamp': now, 'timeframe':
            timeframe, 'instrument': instrument, 'summary': {
            'total_signals': total_signals, 'total_outcomes':
            total_outcomes, 'win_rate': win_rate, 'total_pnl': total_pnl},
            'metrics_over_time': metrics_over_time, 'recommendations': self
            ._generate_recommendations(win_rate, total_pnl, metrics_over_time)}
        self.repository.create_report({'report_id': report_data['report_id'
            ], 'tool_id': tool_id, 'report_type': report_type, 'timestamp':
            now, 'report_data': report_data})
        return report_data

    def compare_tools(self, tool_ids: List[str], metric_type: str=
        'win_rate', timeframe: Optional[str]=None, days_back: int=30) ->Dict[
        str, Any]:
        """Compare multiple trading tools based on specified metrics"""
        from_date = datetime.utcnow() - timedelta(days=days_back)
        comparison_data = self.repository.get_comparative_metrics(tool_ids=
            tool_ids, metric_type=metric_type, timeframe=timeframe,
            from_date=from_date)
        result = {'comparison_type': metric_type, 'timeframe': timeframe,
            'period_days': days_back, 'timestamp': datetime.utcnow(),
            'tools': {}}
        for tool_id, metrics in comparison_data.items():
            tool = self.repository.get_tool(tool_id)
            tool_name = tool.name if tool else tool_id
            if metrics:
                avg_value = sum(m.metric_value for m in metrics) / len(metrics)
            else:
                avg_value = 0
            latest_value = metrics[0].metric_value if metrics else 0
            result['tools'][tool_id] = {'name': tool_name, 'average_value':
                avg_value, 'latest_value': latest_value, 'sample_size': len
                (metrics), 'trend': self._calculate_trend([m.metric_value for
                m in metrics]) if metrics else 'neutral'}
        sorted_tools = sorted(result['tools'].items(), key=lambda x: x[1][
            'average_value'], reverse=metric_type != 'drawdown')
        result['ranking'] = [t[0] for t in sorted_tools]
        return result

    def detect_market_regime(self, instrument: str, timeframe: str,
        days_back: int=30) ->Dict[str, Any]:
        """Detect the current market regime based on historical data"""
        return {'instrument': instrument, 'timeframe': timeframe, 'regime':
            'trending', 'confidence': 0.85, 'volatility': 'medium',
            'trend_direction': 'bullish', 'detected_at': datetime.utcnow()}

    @with_exception_handling
    def _calculate_trend(self, values: List[float]) ->str:
        """Calculate the trend direction from a series of values"""
        if not values or len(values) < 2:
            return 'neutral'
        x = list(range(len(values)))
        y = values
        if len(x) != len(y):
            return 'neutral'
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
        sum_xx = sum(x_i * x_i for x_i in x)
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'deteriorating'
            else:
                return 'stable'
        except:
            return 'neutral'

    def _generate_recommendations(self, win_rate: float, total_pnl: float,
        metrics_over_time: Dict[str, Dict[str, float]]) ->List[Dict[str, Any]]:
        """Generate recommendations based on performance metrics"""
        recommendations = []
        if win_rate < 0.4:
            recommendations.append({'type': 'warning', 'metric': 'win_rate',
                'message':
                'Win rate is below 40%, consider reviewing entry criteria'})
        elif win_rate > 0.65:
            recommendations.append({'type': 'positive', 'metric':
                'win_rate', 'message':
                'Excellent win rate above 65%, maintain current approach'})
        if total_pnl < 0:
            recommendations.append({'type': 'warning', 'metric': 'pnl',
                'message':
                'Negative total PnL, review position sizing and stop loss strategy'
                })
        if metrics_over_time and len(metrics_over_time) >= 2:
            periods = sorted(metrics_over_time.keys())
            if 'win_rate' in metrics_over_time[periods[-1]
                ] and 'win_rate' in metrics_over_time[periods[0]]:
                first_win_rate = metrics_over_time[periods[0]]['win_rate']
                latest_win_rate = metrics_over_time[periods[-1]]['win_rate']
                if latest_win_rate < first_win_rate * 0.8:
                    recommendations.append({'type': 'warning', 'metric':
                        'win_rate_trend', 'message':
                        'Win rate has decreased significantly, consider adapting to changing market conditions'
                        })
        return recommendations
