"""
Monitoring and diagnostics service for error tracking and analysis.

Provides advanced monitoring, analysis, and reporting capabilities for
tracking error patterns and system health in the indicator calculations.
"""
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

from feature_store_service.error.error_manager import IndicatorError
from feature_store_service.logging.indicator_logging import IndicatorLogger

logger = logging.getLogger(__name__)

class ErrorPattern:
    """Represents a detected error pattern."""
    
    def __init__(
        self,
        pattern_id: str,
        error_type: str,
        frequency: int,
        first_seen: datetime,
        last_seen: datetime,
        affected_indicators: List[str],
        common_params: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        pattern_id: Description of pattern_id
        error_type: Description of error_type
        frequency: Description of frequency
        first_seen: Description of first_seen
        last_seen: Description of last_seen
        affected_indicators: Description of affected_indicators
        common_params: Description of common_params
        Any]]: Description of Any]]
    
    """

        self.pattern_id = pattern_id
        self.error_type = error_type
        self.frequency = frequency
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.affected_indicators = affected_indicators
        self.common_params = common_params or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary format."""
        return {
            'pattern_id': self.pattern_id,
            'error_type': self.error_type,
            'frequency': self.frequency,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'affected_indicators': self.affected_indicators,
            'common_params': self.common_params
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorPattern':
        """Create pattern from dictionary format."""
        return cls(
            pattern_id=data['pattern_id'],
            error_type=data['error_type'],
            frequency=data['frequency'],
            first_seen=datetime.fromisoformat(data['first_seen']),
            last_seen=datetime.fromisoformat(data['last_seen']),
            affected_indicators=data['affected_indicators'],
            common_params=data.get('common_params')
        )


class DiagnosticMetric:
    """Represents a diagnostic metric for error analysis."""
    
    def __init__(
        self,
        name: str,
        value: float,
        threshold: float,
        severity: str,
        timestamp: datetime
    ):
    """
      init  .
    
    Args:
        name: Description of name
        value: Description of value
        threshold: Description of threshold
        severity: Description of severity
        timestamp: Description of timestamp
    
    """

        self.name = name
        self.value = value
        self.threshold = threshold
        self.severity = severity
        self.timestamp = timestamp
        
    def is_threshold_exceeded(self) -> bool:
        """Check if metric exceeds its threshold."""
        return self.value > self.threshold
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            'name': self.name,
            'value': self.value,
            'threshold': self.threshold,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'threshold_exceeded': self.is_threshold_exceeded()
        }


class ErrorMonitoringService:
    """
    Service for monitoring and analyzing error patterns.
    
    This service tracks error occurrences, identifies patterns, and
    provides diagnostic information about system health.
    """
    
    def __init__(self, storage_dir: str = "monitoring"):
    """
      init  .
    
    Args:
        storage_dir: Description of storage_dir
    
    """

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.error_history: List[Dict[str, Any]] = []
        self.patterns: Dict[str, ErrorPattern] = {}
        self.metrics: Dict[str, List[DiagnosticMetric]] = defaultdict(list)
        
        self.logger = IndicatorLogger(log_dir=storage_dir)
        
        # Load existing data
        self._load_history()
        self._load_patterns()

    def record_error(self, error: IndicatorError, indicator_name: str) -> None:
        """
        Record an error occurrence for analysis.
        
        Args:
            error: The error that occurred
            indicator_name: Name of the affected indicator
        """
        error_data = {
            'timestamp': datetime.utcnow(),
            'error_type': error.error_type,
            'indicator': indicator_name,
            'message': str(error),
            'details': error.details
        }
        
        self.error_history.append(error_data)
        self._save_history()
        
        # Update patterns
        self._analyze_patterns()
        
        # Log the error
        self.logger.log_error(
            indicator_name=indicator_name,
            error_type=error.error_type,
            message=str(error),
            details=error.details
        )

    def get_error_patterns(
        self,
        time_window: Optional[timedelta] = None
    ) -> List[ErrorPattern]:
        """
        Get identified error patterns.
        
        Args:
            time_window: Optional time window to filter patterns
            
        Returns:
            List of identified error patterns
        """
        if not time_window:
            return list(self.patterns.values())
            
        cutoff = datetime.utcnow() - time_window
        return [
            pattern for pattern in self.patterns.values()
            if pattern.last_seen >= cutoff
        ]

    def get_diagnostic_metrics(
        self,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, List[DiagnosticMetric]]:
        """
        Get diagnostic metrics for system health analysis.
        
        Args:
            metric_names: Optional list of specific metrics to retrieve
            
        Returns:
            Dictionary mapping metric names to their values
        """
        if metric_names:
            return {
                name: metrics 
                for name, metrics in self.metrics.items()
                if name in metric_names
            }
        return self.metrics

    def analyze_trends(
        self,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """
        Analyze error trends over time.
        
        Args:
            time_window: Time window for trend analysis
            
        Returns:
            Dictionary containing trend analysis results
        """
        cutoff = datetime.utcnow() - time_window
        recent_errors = [
            error for error in self.error_history
            if error['timestamp'] >= cutoff
        ]
        
        if not recent_errors:
            return {'message': 'No errors in specified time window'}
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(recent_errors)
        
        # Analyze by error type
        error_type_counts = df['error_type'].value_counts().to_dict()
        
        # Analyze by indicator
        indicator_counts = df['indicator'].value_counts().to_dict()
        
        # Calculate daily error rates
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size()
        
        # Calculate trend
        if len(daily_counts) > 1:
            trend = np.polyfit(range(len(daily_counts)), daily_counts, 1)[0]
        else:
            trend = 0
            
        return {
            'total_errors': len(recent_errors),
            'by_error_type': error_type_counts,
            'by_indicator': indicator_counts,
            'daily_average': daily_counts.mean(),
            'trend': trend,  # Positive means increasing errors
            'time_window': str(time_window)
        }

    def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report.
        
        Returns:
            Dictionary containing health report data
        """
        now = datetime.utcnow()
        
        # Analyze different time windows
        windows = {
            'last_hour': timedelta(hours=1),
            'last_day': timedelta(days=1),
            'last_week': timedelta(days=7)
        }
        
        trends = {
            window: self.analyze_trends(delta)
            for window, delta in windows.items()
        }
        
        # Get current patterns
        patterns = self.get_error_patterns(timedelta(days=7))
        
        # Get latest metrics
        metrics = {
            name: metrics[-1].to_dict()
            for name, metrics in self.metrics.items()
            if metrics
        }
        
        return {
            'timestamp': now.isoformat(),
            'trends': trends,
            'active_patterns': [p.to_dict() for p in patterns],
            'current_metrics': metrics,
            'system_status': self._determine_system_status(metrics)
        }

    def _analyze_patterns(self) -> None:
        """Analyze error history to identify patterns."""
        # Group recent errors
        recent = [
            e for e in self.error_history
            if e['timestamp'] >= datetime.utcnow() - timedelta(days=7)
        ]
        
        if not recent:
            return
            
        # Group by error type and analyze
        by_type = defaultdict(list)
        for error in recent:
            by_type[error['error_type']].append(error)
            
        for error_type, errors in by_type.items():
            if len(errors) >= 3:  # Minimum frequency for pattern
                # Group by indicator
                by_indicator = defaultdict(list)
                for error in errors:
                    by_indicator[error['indicator']].append(error)
                    
                # Look for common parameters in errors
                common_params = self._find_common_parameters(errors)
                
                if common_params:
                    pattern_id = f"{error_type}_{sorted(common_params.keys())[0]}"
                    
                    self.patterns[pattern_id] = ErrorPattern(
                        pattern_id=pattern_id,
                        error_type=error_type,
                        frequency=len(errors),
                        first_seen=min(e['timestamp'] for e in errors),
                        last_seen=max(e['timestamp'] for e in errors),
                        affected_indicators=list(by_indicator.keys()),
                        common_params=common_params
                    )
                    
        self._save_patterns()

    def _find_common_parameters(
        self,
        errors: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find common parameters across errors."""
        if not errors:
            return None
            
        # Extract parameters from each error
        all_params = []
        for error in errors:
            params = error.get('details', {}).get('parameters', {})
            if params:
                all_params.append(params)
                
        if not all_params:
            return None
            
        # Find intersection of parameter keys
        common_keys = set.intersection(*(set(p.keys()) for p in all_params))
        
        # Find common values
        common_params = {}
        for key in common_keys:
            values = [p[key] for p in all_params]
            if len(set(values)) == 1:  # All values are the same
                common_params[key] = values[0]
                
        return common_params if common_params else None

    def _determine_system_status(
        self,
        metrics: Dict[str, Dict[str, Any]]
    ) -> str:
        """Determine overall system status based on metrics."""
        if not metrics:
            return "UNKNOWN"
            
        # Count exceeded thresholds by severity
        exceeded = defaultdict(int)
        for metric in metrics.values():
            if metric['threshold_exceeded']:
                exceeded[metric['severity']] += 1
                
        if exceeded.get('critical', 0) > 0:
            return "CRITICAL"
        elif exceeded.get('error', 0) > 0:
            return "ERROR"
        elif exceeded.get('warning', 0) > 0:
            return "WARNING"
        return "HEALTHY"

    def _load_history(self) -> None:
        """Load error history from storage."""
        history_file = self.storage_dir / "error_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.error_history = [
                    {**item, 'timestamp': datetime.fromisoformat(item['timestamp'])}
                    for item in data
                ]

    def _save_history(self) -> None:
        """Save error history to storage."""
        history_file = self.storage_dir / "error_history.json"
        with open(history_file, 'w') as f:
            json.dump(
                [
                    {**item, 'timestamp': item['timestamp'].isoformat()}
                    for item in self.error_history
                ],
                f,
                indent=2
            )

    def _load_patterns(self) -> None:
        """Load error patterns from storage."""
        patterns_file = self.storage_dir / "error_patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                self.patterns = {
                    item['pattern_id']: ErrorPattern.from_dict(item)
                    for item in data
                }

    def _save_patterns(self) -> None:
        """Save error patterns to storage."""
        patterns_file = self.storage_dir / "error_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(
                [pattern.to_dict() for pattern in self.patterns.values()],
                f,
                indent=2
            )
