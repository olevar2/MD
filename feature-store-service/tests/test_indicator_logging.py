"""
Tests for the indicator logging and reporting system.
"""
import unittest
import json
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from core.indicator_logging import (
    IndicatorLogger,
    IndicatorReport
)

class TestIndicatorLogging(unittest.TestCase):
    """Test suite for the IndicatorLogger and IndicatorReport classes."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        self.logger = IndicatorLogger(log_dir=self.temp_dir)
        self.reporter = IndicatorReport(log_dir=self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and its contents
        shutil.rmtree(self.temp_dir)

    def test_log_validation(self):
        """Test validation event logging."""
        self.logger.log_validation(
            indicator_name="RSI",
            validation_type="input_data",
            is_valid=True,
            details={"period": 14}
        )

        # Verify log file exists and contains the entry
        log_path = Path(self.temp_dir) / "validation.log"
        self.assertTrue(log_path.exists())
        
        with open(log_path, 'r') as f:
            log_content = f.read()
            self.assertIn("RSI", log_content)
            self.assertIn("input_data", log_content)
            self.assertIn("true", log_content.lower())

    def test_log_error(self):
        """Test error event logging."""
        self.logger.log_error(
            indicator_name="MACD",
            error_type="calculation_error",
            message="Division by zero",
            details={"line": 42}
        )

        # Verify log file exists and contains the entry
        log_path = Path(self.temp_dir) / "error.log"
        self.assertTrue(log_path.exists())
        
        with open(log_path, 'r') as f:
            log_content = f.read()
            self.assertIn("MACD", log_content)
            self.assertIn("Division by zero", log_content)
            self.assertIn("calculation_error", log_content)

    def test_log_performance(self):
        """Test performance metrics logging."""
        self.logger.log_performance(
            indicator_name="SMA",
            execution_time=0.15,
            data_points=1000,
            details={"period": 20}
        )

        # Verify log file exists and contains the entry
        log_path = Path(self.temp_dir) / "performance.log"
        self.assertTrue(log_path.exists())
        
        with open(log_path, 'r') as f:
            log_content = f.read()
            self.assertIn("SMA", log_content)
            self.assertIn("0.15", log_content)
            self.assertIn("1000", log_content)

    def test_generate_validation_report(self):
        """Test validation report generation."""
        # Log some validation events
        indicators = ["RSI", "MACD", "RSI"]
        types = ["input", "calculation", "output"]
        valid = [True, False, True]
        
        for ind, typ, val in zip(indicators, types, valid):
            self.logger.log_validation(ind, typ, val)

        # Generate report
        report = self.reporter.generate_validation_report()
        
        self.assertEqual(report['total_validations'], 3)
        self.assertEqual(len(report['by_indicator']), 2)  # RSI and MACD
        self.assertEqual(len(report['by_type']), 3)  # input, calculation, output

    def test_generate_error_report(self):
        """Test error report generation."""
        # Log some errors
        errors = [
            ("RSI", "calculation_error", "Division by zero"),
            ("MACD", "data_error", "Missing values"),
            ("RSI", "parameter_error", "Invalid period")
        ]
        
        for ind, err_type, msg in errors:
            self.logger.log_error(ind, err_type, msg)

        # Generate report
        report = self.reporter.generate_error_report()
        
        self.assertEqual(report['total_errors'], 3)
        self.assertEqual(len(report['by_indicator']), 2)  # RSI and MACD
        self.assertEqual(len(report['by_type']), 3)  # Three different error types

    def test_generate_performance_report(self):
        """Test performance report generation."""
        # Log some performance metrics
        metrics = [
            ("RSI", 0.1, 1000),
            ("MACD", 0.2, 1500),
            ("RSI", 0.15, 1200)
        ]
        
        for ind, time, points in metrics:
            self.logger.log_performance(ind, time, points)

        # Generate report
        report = self.reporter.generate_performance_report()
        
        self.assertEqual(report['total_executions'], 3)
        self.assertEqual(len(report['by_indicator']), 2)  # RSI and MACD
        
        # Check RSI statistics
        rsi_stats = report['by_indicator']['RSI']
        self.assertEqual(rsi_stats['count'], 2)
        self.assertAlmostEqual(rsi_stats['min_time'], 0.1)
        self.assertAlmostEqual(rsi_stats['max_time'], 0.15)

    def test_generate_summary_report(self):
        """Test summary report generation."""
        # Log various events
        self.logger.log_validation("RSI", "input", True)
        self.logger.log_error("MACD", "calculation_error", "Error")
        self.logger.log_performance("SMA", 0.1, 1000)

        # Generate summary report
        report = self.reporter.generate_summary_report()
        
        self.assertIn('validation_summary', report)
        self.assertIn('error_summary', report)
        self.assertIn('performance_summary', report)
        self.assertIn('generated_at', report)

    def test_time_filtered_reports(self):
        """Test report generation with time filters."""
        # Log events at different times
        now = datetime.utcnow()
        
        # Backdate some events
        with unittest.mock.patch('datetime.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = now - timedelta(hours=2)
            self.logger.log_validation("RSI", "input", True)
            
            mock_datetime.utcnow.return_value = now - timedelta(hours=1)
            self.logger.log_error("MACD", "calculation_error", "Error")
            
            mock_datetime.utcnow.return_value = now
            self.logger.log_performance("SMA", 0.1, 1000)

        # Generate reports for last hour
        start_time = now - timedelta(hours=1)
        
        validation_report = self.reporter.generate_validation_report(start_time)
        self.assertEqual(validation_report.get('total_validations', 0), 0)
        
        error_report = self.reporter.generate_error_report(start_time)
        self.assertEqual(error_report.get('total_errors', 0), 1)
        
        performance_report = self.reporter.generate_performance_report(start_time)
        self.assertEqual(performance_report.get('total_executions', 0), 1)

if __name__ == '__main__':
    unittest.main()
