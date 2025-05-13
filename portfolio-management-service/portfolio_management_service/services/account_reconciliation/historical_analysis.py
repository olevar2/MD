"""
Historical reconciliation analysis implementation.

This module provides functionality for analyzing historical reconciliation data,
identifying patterns, trends, and correlations.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
from collections import defaultdict
from portfolio_management_service.services.account_reconciliation.base import ReconciliationBase
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from portfolio_management_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class HistoricalAnalysis(ReconciliationBase):
    """
    Historical reconciliation analysis implementation.
    """

    def _generate_time_points(self, start_date: datetime, end_date:
        datetime, interval: str) ->List[datetime]:
        """Generate time points for historical reconciliation based on interval"""
        time_points = []
        current = start_date
        if interval == 'hourly':
            delta = timedelta(hours=1)
        elif interval == 'daily':
            delta = timedelta(days=1)
        elif interval == 'weekly':
            delta = timedelta(weeks=1)
        else:
            raise ValueError(f'Unsupported interval: {interval}')
        while current <= end_date:
            time_points.append(current)
            current += delta
        return time_points

    async def analyze_historical_reconciliation(self,
        reconciliation_results: List[Dict[str, Any]]) ->Dict[str, Any]:
        """Analyze historical reconciliation results to identify patterns and trends"""
        successful_results = [r for r in reconciliation_results if r.get(
            'status') != 'failed']
        if not successful_results:
            return {'status': 'insufficient_data', 'message':
                'No successful reconciliations available for analysis'}
        time_points = [r.get('time_point') for r in successful_results]
        discrepancy_counts = [r.get('discrepancies', {}).get('total_count',
            0) for r in successful_results]
        trend_analysis = self._calculate_trend_statistics(time_points,
            discrepancy_counts)
        recurring_fields = self._identify_recurring_discrepancy_fields(
            successful_results)
        return {'trend': trend_analysis, 'recurring_fields':
            recurring_fields, 'summary': {'average_discrepancies': sum(
            discrepancy_counts) / len(discrepancy_counts) if
            discrepancy_counts else 0, 'max_discrepancies': max(
            discrepancy_counts) if discrepancy_counts else 0,
            'min_discrepancies': min(discrepancy_counts) if
            discrepancy_counts else 0, 'std_deviation': self.
            _calculate_std_deviation(discrepancy_counts),
            'total_points_analyzed': len(successful_results)}}

    @with_exception_handling
    def _calculate_trend_statistics(self, time_points: List[datetime],
        discrepancy_counts: List[int]) ->Dict[str, Any]:
        """Calculate trend statistics from time series data"""
        if not time_points or not discrepancy_counts or len(time_points
            ) != len(discrepancy_counts):
            return {'status': 'error', 'message': 'Invalid time series data'}
        if len(time_points) < 3:
            return {'status': 'insufficient_data', 'message':
                'Need at least 3 data points for trend analysis'}
        base_date = min(time_points)
        x_values = [((t - base_date).total_seconds() / 86400) for t in
            time_points]
        y_values = discrepancy_counts
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            intercept = (sum_y - slope * sum_x) / n
            y_mean = sum_y / n
            ss_total = sum((y - y_mean) ** 2 for y in y_values)
            ss_residual = sum((y - (slope * x + intercept)) ** 2 for x, y in
                zip(x_values, y_values))
            r_squared = 1 - ss_residual / ss_total if ss_total != 0 else 0
            trend_direction = ('increasing' if slope > 0.01 else 
                'decreasing' if slope < -0.01 else 'stable')
            trend_strength = 'strong' if abs(r_squared
                ) > 0.7 else 'moderate' if abs(r_squared) > 0.3 else 'weak'
            return {'direction': trend_direction, 'strength':
                trend_strength, 'slope': slope, 'r_squared': r_squared,
                'status': 'success'}
        except Exception as e:
            logger.error(f'Error calculating trend statistics: {str(e)}',
                exc_info=True)
            return {'status': 'error', 'message':
                f'Error calculating trend: {str(e)}'}

    def _identify_recurring_discrepancy_fields(self, reconciliation_results:
        List[Dict[str, Any]]) ->List[Dict[str, Any]]:
        """Identify fields that frequently have discrepancies"""
        field_counts = defaultdict(int)
        field_severity = defaultdict(list)
        for result in reconciliation_results:
            for disc in result.get('discrepancies', {}).get('details', []):
                field = disc.get('field')
                severity = disc.get('severity', 'low')
                if field:
                    field_counts[field] += 1
                    field_severity[field].append(severity)
        total_results = len(reconciliation_results)
        recurring_fields = []
        for field, count in field_counts.items():
            frequency = count / total_results if total_results > 0 else 0
            severity_map = {'high': 3, 'medium': 2, 'low': 1}
            severity_score = sum(severity_map.get(s, 1) for s in
                field_severity[field])
            avg_severity = severity_score / count if count > 0 else 0
            if frequency > 0.1:
                recurring_fields.append({'field': field, 'frequency':
                    frequency, 'frequency_percentage': frequency * 100,
                    'occurrence_count': count, 'average_severity_score':
                    avg_severity, 'criticality': 'high' if frequency > 0.5 and
                    avg_severity > 2 else 'medium' if frequency > 0.3 or 
                    avg_severity > 2 else 'low'})
        return sorted(recurring_fields, key=lambda x: (0 if x['criticality'
            ] == 'high' else 1 if x['criticality'] == 'medium' else 2, -x[
            'frequency']))

    def _calculate_std_deviation(self, values: List[float]) ->float:
        """Calculate standard deviation of a list of values"""
        if not values:
            return 0
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return variance ** 0.5

    def _process_reconciliation_data(self, reconciliations: List[Dict[str,
        Any]]) ->Dict[str, Any]:
        """Process raw reconciliation data into a format suitable for analysis"""
        data_points = []
        for recon in reconciliations:
            timestamp = recon.get('start_time')
            if not timestamp:
                continue
            discrepancy_count = recon.get('discrepancies', {}).get(
                'total_count', 0)
            field_discrepancies = {}
            for disc in recon.get('discrepancies', {}).get('details', []):
                field = disc.get('field')
                if field:
                    field_discrepancies[f'field_{field}'] = disc.get(
                        'percentage_difference', 0)
            data_point = {'timestamp': timestamp, 'discrepancy_count':
                discrepancy_count, 'reconciliation_id': recon.get(
                'reconciliation_id'), **field_discrepancies}
            data_points.append(data_point)
        return {'data_points': data_points, 'total_records': len(data_points)}

    def _detect_recurring_discrepancies(self, processed_data: Dict[str, Any]
        ) ->List[Dict[str, Any]]:
        """Detect recurring patterns in discrepancy data"""
        data_points = processed_data.get('data_points', [])
        if not data_points or len(data_points) < 5:
            return []
        field_occurrences = defaultdict(int)
        for point in data_points:
            for key in point.keys():
                if key.startswith('field_'):
                    field_occurrences[key] += 1
        patterns = []
        for field, occurrences in field_occurrences.items():
            if occurrences >= 3:
                field_name = field[6:]
                frequency = occurrences / len(data_points)
                values = [point.get(field, 0) for point in data_points if 
                    field in point]
                avg_value = sum(values) / len(values) if values else 0
                std_dev = self._calculate_std_deviation(values)
                patterns.append({'field': field_name, 'occurrences':
                    occurrences, 'frequency': frequency,
                    'average_discrepancy': avg_value, 'std_deviation':
                    std_dev, 'consistency': 'high' if std_dev < avg_value *
                    0.1 else 'medium' if std_dev < avg_value * 0.25 else 'low'}
                    )
        return sorted(patterns, key=lambda x: x['occurrences'], reverse=True)

    def _detect_discrepancy_trends(self, processed_data: Dict[str, Any]
        ) ->List[Dict[str, Any]]:
        """Detect trends in discrepancy data over time"""
        data_points = processed_data.get('data_points', [])
        if not data_points or len(data_points) < 5:
            return []
        sorted_points = sorted(data_points, key=lambda x: x.get('timestamp',
            datetime.min))
        timestamps = [p.get('timestamp') for p in sorted_points]
        counts = [p.get('discrepancy_count', 0) for p in sorted_points]
        trends = []
        overall_trend = self._calculate_trend_statistics(timestamps, counts)
        if overall_trend.get('status') == 'success':
            trends.append({'field': 'overall_discrepancies', 'direction':
                overall_trend.get('direction', 'stable'), 'strength':
                overall_trend.get('strength', 'weak'), 'r_squared':
                overall_trend.get('r_squared', 0), 'description':
                f"{overall_trend.get('strength', 'weak').capitalize()} {overall_trend.get('direction', 'stable')} trend in overall discrepancies"
                })
        field_keys = [key for key in sorted_points[0].keys() if key.
            startswith('field_') and key in sorted_points[1]]
        for field in field_keys:
            field_values = [p.get(field, 0) for p in sorted_points if field in
                p]
            field_timestamps = [p.get('timestamp') for p in sorted_points if
                field in p]
            if len(field_values) >= 3 and len(field_timestamps) == len(
                field_values):
                field_trend = self._calculate_trend_statistics(field_timestamps
                    , field_values)
                if field_trend.get('status') == 'success' and field_trend.get(
                    'direction') != 'stable':
                    trends.append({'field': field[6:], 'direction':
                        field_trend.get('direction', 'stable'), 'strength':
                        field_trend.get('strength', 'weak'), 'r_squared':
                        field_trend.get('r_squared', 0), 'description':
                        f"{field_trend.get('strength', 'weak').capitalize()} {field_trend.get('direction', 'stable')} trend in {field[6:]} discrepancies"
                        })
        return trends

    async def _detect_external_correlations(self, account_id: str,
        processed_data: Dict[str, Any]) ->List[Dict[str, Any]]:
        """
        Detect correlations between discrepancies and external factors.
        
        This would typically involve analysis against trading volume, market volatility,
        system load, etc. Simplified implementation here.
        """
        return []
