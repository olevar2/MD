"""
Visualization Data API

This module provides data structures and endpoints optimized for visualization,
generating chart-ready data formats for UI components.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService
from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class VisualizationDataAPI:
    """
    API for generating data structures optimized for visualization
    """

    def __init__(self, db: Session):
    """
      init  .
    
    Args:
        db: Description of db
    
    """

        self.repository = ToolEffectivenessRepository(db)
        self.service = ToolEffectivenessService(db)

    @with_resilience('get_tool_performance_chart_data')
    @cached(ttl=300)
    def get_tool_performance_chart_data(self, tool_id: str, metric_type:
        str='win_rate', timeframe: Optional[str]=None, days_back: int=90
        ) ->Dict[str, Any]:
        """
        Get time-series data for charting tool performance
        Returns data formatted for easy chart integration
        """
        from_date = datetime.utcnow() - timedelta(days=days_back)
        metrics = self.repository.get_metrics_history(tool_id=tool_id,
            metric_type=metric_type, timeframe=timeframe, from_date=
            from_date, limit=1000)
        dates = []
        values = []
        for metric in metrics:
            dates.append(metric.timestamp.strftime('%Y-%m-%d'))
            values.append(round(metric.metric_value, 4))
        if len(dates) > 1:
            filled_dates, filled_values = self._fill_missing_dates(dates,
                values)
        else:
            filled_dates, filled_values = dates, values
        avg = sum(filled_values) / len(filled_values) if filled_values else 0
        min_val = min(filled_values) if filled_values else 0
        max_val = max(filled_values) if filled_values else 0
        tool = self.repository.get_tool(tool_id)
        tool_name = tool.name if tool else tool_id
        return {'tool_id': tool_id, 'tool_name': tool_name, 'metric_type':
            metric_type, 'timeframe': timeframe, 'days_back': days_back,
            'labels': filled_dates, 'data': filled_values, 'stats': {'avg':
            round(avg, 4), 'min': round(min_val, 4), 'max': round(max_val, 4)}}

    @with_resilience('get_comparative_chart_data')
    @cached(ttl=300)
    def get_comparative_chart_data(self, tool_ids: List[str], metric_type:
        str='win_rate', days_back: int=30) ->Dict[str, Any]:
        """Get data for comparing multiple tools in a single chart"""
        comparison = self.service.compare_tools(tool_ids=tool_ids,
            metric_type=metric_type, days_back=days_back)
        datasets = []
        for tool_id in tool_ids:
            if tool_id not in comparison['tools']:
                continue
            tool_data = self.get_tool_performance_chart_data(tool_id=
                tool_id, metric_type=metric_type, days_back=days_back)
            datasets.append({'tool_id': tool_id, 'tool_name': tool_data[
                'tool_name'], 'data': tool_data['data']})
            if not comparison.get('labels') and tool_data.get('labels'):
                comparison['labels'] = tool_data['labels']
        comparison['datasets'] = datasets
        return comparison

    @with_resilience('get_win_rate_by_market_regime')
    @cached(ttl=600)
    def get_win_rate_by_market_regime(self, tool_id: str, days_back: int=90
        ) ->Dict[str, Any]:
        """
        Get win rate broken down by market regime
        Returns data formatted for pie/bar charts
        """
        from_date = datetime.utcnow() - timedelta(days=days_back)
        signals = self.repository.get_signals(tool_id=tool_id, from_date=
            from_date, limit=10000)
        regimes = {}
        for signal in signals:
            regime = signal.market_regime or 'unknown'
            if regime not in regimes:
                regimes[regime] = {'total': 0, 'wins': 0}
            outcomes = self.repository.get_outcomes_for_signal(signal.signal_id
                )
            if outcomes:
                regimes[regime]['total'] += len(outcomes)
                regimes[regime]['wins'] += len([o for o in outcomes if o.
                    pnl > 0])
        labels = []
        win_rates = []
        for regime, stats in regimes.items():
            labels.append(regime)
            win_rate = stats['wins'] / stats['total'] if stats['total'
                ] > 0 else 0
            win_rates.append(round(win_rate * 100, 2))
        tool = self.repository.get_tool(tool_id)
        tool_name = tool.name if tool else tool_id
        return {'tool_id': tool_id, 'tool_name': tool_name, 'days_back':
            days_back, 'labels': labels, 'data': win_rates}

    @with_resilience('get_pnl_distribution')
    @cached(ttl=300)
    def get_pnl_distribution(self, tool_id: str, days_back: int=90) ->Dict[
        str, Any]:
        """
        Get PnL distribution data for histogram visualization
        """
        from_date = datetime.utcnow() - timedelta(days=days_back)
        outcomes = self.repository.get_outcomes(tool_id=tool_id, from_date=
            from_date, limit=10000)
        if not outcomes:
            return {'tool_id': tool_id, 'days_back': days_back, 'bins': [],
                'frequencies': [], 'stats': {'mean': 0, 'median': 0,
                'std_dev': 0}}
        pnl_values = [o.pnl for o in outcomes]
        min_val = min(pnl_values)
        max_val = max(pnl_values)
        if min_val == max_val:
            bins = [min_val]
            frequencies = [len(pnl_values)]
        else:
            bin_width = (max_val - min_val) / 10
            bins = [round(min_val + i * bin_width, 2) for i in range(11)]
            frequencies = [0] * 10
            for val in pnl_values:
                for i in range(10):
                    if bins[i] <= val < bins[i + 1] or i == 9 and val == bins[
                        i + 1]:
                        frequencies[i] += 1
                        break
        mean = sum(pnl_values) / len(pnl_values)
        sorted_values = sorted(pnl_values)
        n = len(sorted_values)
        if n % 2 == 0:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            median = sorted_values[n // 2]
        variance = sum((x - mean) ** 2 for x in pnl_values) / len(pnl_values)
        std_dev = variance ** 0.5
        tool = self.repository.get_tool(tool_id)
        tool_name = tool.name if tool else tool_id
        return {'tool_id': tool_id, 'tool_name': tool_name, 'days_back':
            days_back, 'bins': bins, 'frequencies': frequencies, 'stats': {
            'mean': round(mean, 2), 'median': round(median, 2), 'std_dev':
            round(std_dev, 2)}}

    def _fill_missing_dates(self, dates: List[str], values: List[float]
        ) ->Tuple[List[str], List[float]]:
        """
        Fill missing dates in time series with interpolated values
        """
        if not dates or len(dates) < 2:
            return dates, values
        dt_dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        min_date = min(dt_dates)
        max_date = max(dt_dates)
        complete_dates = []
        current = min_date
        while current <= max_date:
            complete_dates.append(current)
            current += timedelta(days=1)
        str_complete_dates = [d.strftime('%Y-%m-%d') for d in complete_dates]
        complete_values = []
        date_to_value = dict(zip(dates, values))
        for date_str in str_complete_dates:
            if date_str in date_to_value:
                complete_values.append(date_to_value[date_str])
            else:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                prev_dates = [d for d in dt_dates if d < date]
                prev_date = max(prev_dates) if prev_dates else None
                next_dates = [d for d in dt_dates if d > date]
                next_date = min(next_dates) if next_dates else None
                if prev_date and next_date:
                    prev_value = date_to_value[prev_date.strftime('%Y-%m-%d')]
                    next_value = date_to_value[next_date.strftime('%Y-%m-%d')]
                    days_between = (next_date - prev_date).days
                    days_from_prev = (date - prev_date).days
                    if days_between > 0:
                        weight = days_from_prev / days_between
                        interpolated = prev_value + (next_value - prev_value
                            ) * weight
                        complete_values.append(round(interpolated, 4))
                    else:
                        complete_values.append(prev_value)
                elif prev_date:
                    complete_values.append(date_to_value[prev_date.strftime
                        ('%Y-%m-%d')])
                elif next_date:
                    complete_values.append(date_to_value[next_date.strftime
                        ('%Y-%m-%d')])
                else:
                    complete_values.append(0)
        return str_complete_dates, complete_values
