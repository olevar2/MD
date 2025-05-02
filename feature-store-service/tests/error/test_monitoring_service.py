"""
Tests for the error monitoring service.
"""
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json

from feature_store_service.error.error_manager import (
    CalculationError,
    DataError,
    ParameterError
)
from feature_store_service.error.monitoring_service import (
    ErrorMonitoringService,
    ErrorPattern,
    DiagnosticMetric
)

class TestErrorMonitoringService(unittest.TestCase):
    """Test suite for the ErrorMonitoringService."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for monitoring data
        self.temp_dir = tempfile.mkdtemp()
        self.monitoring_service = ErrorMonitoringService(storage_dir=self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and its contents
        shutil.rmtree(self.temp_dir)

    def test_record_error(self):
        """Test error recording and storage."""
        error = CalculationError(
            message="Division by zero",
            details={'parameters': {'period': 14}}
        )
        
        self.monitoring_service.record_error(error, "RSI")
        
        # Verify error was recorded
        self.assertEqual(len(self.monitoring_service.error_history), 1)
        
        # Check if history file was created
        history_file = Path(self.temp_dir) / "error_history.json"
        self.assertTrue(history_file.exists())
        
        with open(history_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]['error_type'], 'calculation_error')

    def test_error_pattern_detection(self):
        """Test error pattern detection."""
        # Create repeated errors with common parameters
        common_params = {'period': 14, 'price_source': 'close'}
        
        for _ in range(3):  # Need at least 3 occurrences for pattern
            error = CalculationError(
                message="Calculation failed",
                details={'parameters': common_params}
            )
            self.monitoring_service.record_error(error, "RSI")
            
        patterns = self.monitoring_service.get_error_patterns()
        
        self.assertGreater(len(patterns), 0)
        pattern = patterns[0]
        self.assertEqual(pattern.error_type, 'calculation_error')
        self.assertEqual(pattern.frequency, 3)
        self.assertIn('RSI', pattern.affected_indicators)
        self.assertEqual(pattern.common_params['period'], 14)

    def test_trend_analysis(self):
        """Test error trend analysis."""
        # Create errors at different times
        now = datetime.utcnow()
        
        errors = [
            (now - timedelta(days=6), CalculationError("Error 1")),
            (now - timedelta(days=4), DataError("Error 2")),
            (now - timedelta(days=2), CalculationError("Error 3")),
            (now - timedelta(days=1), ParameterError("Error 4"))
        ]
        
        for timestamp, error in errors:
            with unittest.mock.patch('datetime.datetime') as mock_datetime:
                mock_datetime.utcnow.return_value = timestamp
                self.monitoring_service.record_error(error, "RSI")
                
        trends = self.monitoring_service.analyze_trends(timedelta(days=7))
        
        self.assertEqual(trends['total_errors'], 4)
        self.assertIn('by_error_type', trends)
        self.assertIn('daily_average', trends)
        self.assertIn('trend', trends)

    def test_health_report_generation(self):
        """Test health report generation."""
        # Record some errors
        errors = [
            (CalculationError("Error 1"), "RSI"),
            (DataError("Error 2"), "MACD"),
            (CalculationError("Error 3"), "RSI")
        ]
        
        for error, indicator in errors:
            self.monitoring_service.record_error(error, indicator)
            
        report = self.monitoring_service.generate_health_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('trends', report)
        self.assertIn('active_patterns', report)
        self.assertIn('system_status', report)

    def test_diagnostic_metrics(self):
        """Test diagnostic metrics handling."""
        # Add some diagnostic metrics
        now = datetime.utcnow()
        metrics = [
            DiagnosticMetric("error_rate", 0.05, 0.1, "warning", now),
            DiagnosticMetric("recovery_rate", 0.95, 0.8, "info", now)
        ]
        
        for metric in metrics:
            self.monitoring_service.metrics[metric.name].append(metric)
            
        retrieved_metrics = self.monitoring_service.get_diagnostic_metrics()
        
        self.assertEqual(len(retrieved_metrics), 2)
        self.assertIn("error_rate", retrieved_metrics)
        self.assertIn("recovery_rate", retrieved_metrics)

    def test_error_pattern_persistence(self):
        """Test error pattern persistence."""
        # Create a pattern
        error = CalculationError(
            message="Common error",
            details={'parameters': {'period': 14}}
        )
        
        # Record multiple occurrences
        for _ in range(3):
            self.monitoring_service.record_error(error, "RSI")
            
        # Create new service instance (should load existing patterns)
        new_service = ErrorMonitoringService(storage_dir=self.temp_dir)
        patterns = new_service.get_error_patterns()
        
        self.assertGreater(len(patterns), 0)
        self.assertEqual(patterns[0].error_type, 'calculation_error')

    def test_time_filtered_patterns(self):
        """Test pattern retrieval with time filtering."""
        now = datetime.utcnow()
        
        # Create old and recent errors
        with unittest.mock.patch('datetime.datetime') as mock_datetime:
            # Old error
            mock_datetime.utcnow.return_value = now - timedelta(days=10)
            self.monitoring_service.record_error(
                CalculationError("Old error"),
                "RSI"
            )
            
            # Recent errors
            mock_datetime.utcnow.return_value = now - timedelta(hours=1)
            for _ in range(3):
                self.monitoring_service.record_error(
                    DataError("Recent error"),
                    "MACD"
                )
                
        # Get patterns from last week
        recent_patterns = self.monitoring_service.get_error_patterns(
            time_window=timedelta(days=7)
        )
        
        self.assertTrue(all(
            pattern.last_seen >= now - timedelta(days=7)
            for pattern in recent_patterns
        ))

    def test_system_status_determination(self):
        """Test system status determination from metrics."""
        now = datetime.utcnow()
        metrics = {
            'critical_metric': {
                'value': 0.9,
                'threshold': 0.8,
                'severity': 'critical',
                'threshold_exceeded': True
            },
            'warning_metric': {
                'value': 0.7,
                'threshold': 0.8,
                'severity': 'warning',
                'threshold_exceeded': False
            }
        }
        
        status = self.monitoring_service._determine_system_status(metrics)
        self.assertEqual(status, "CRITICAL")

    def test_common_parameter_detection(self):
        """Test detection of common parameters in errors."""
        errors = [
            {
                'details': {
                    'parameters': {
                        'period': 14,
                        'price_source': 'close',
                        'random': 1
                    }
                }
            },
            {
                'details': {
                    'parameters': {
                        'period': 14,
                        'price_source': 'close',
                        'random': 2
                    }
                }
            }
        ]
        
        common_params = self.monitoring_service._find_common_parameters(errors)
        
        self.assertIsNotNone(common_params)
        self.assertEqual(common_params['period'], 14)
        self.assertEqual(common_params['price_source'], 'close')
        self.assertNotIn('random', common_params)

if __name__ == '__main__':
    unittest.main()
