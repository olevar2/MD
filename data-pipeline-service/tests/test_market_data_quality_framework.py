"""
Tests for the Market Data Quality Framework.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
import os

import pandas as pd
import numpy as np
import pytest

# Add the mocks directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mocks')))

# Import the mock monitoring module
from common_lib.monitoring import MetricsCollector, AlertManager

from core.market_data_quality_framework import (
    MarketDataQualityFramework,
    DataQualityLevel,
    DataQualitySLA,
    DataQualityMetrics,
    DataQualityReport
)
from core.validation_engine import (
    ValidationResult,
    ValidationSeverity
)


class TestMarketDataQualityFramework(unittest.TestCase):
    """Tests for the Market Data Quality Framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()

        self.framework = MarketDataQualityFramework(
            metrics_collector=self.metrics_collector,
            alert_manager=self.alert_manager
        )

    def test_initialization(self):
        """Test initialization of the framework."""
        # Check that validators are registered
        self.assertIn("ohlcv_basic", self.framework.validation_engine.validators)
        self.assertIn("ohlcv_standard", self.framework.validation_engine.validators)
        self.assertIn("ohlcv_comprehensive", self.framework.validation_engine.validators)
        self.assertIn("ohlcv_strict", self.framework.validation_engine.validators)

        self.assertIn("tick_basic", self.framework.validation_engine.validators)
        self.assertIn("tick_standard", self.framework.validation_engine.validators)
        self.assertIn("tick_comprehensive", self.framework.validation_engine.validators)
        self.assertIn("tick_strict", self.framework.validation_engine.validators)

        self.assertIn("news", self.framework.validation_engine.validators)
        self.assertIn("economic", self.framework.validation_engine.validators)
        self.assertIn("sentiment", self.framework.validation_engine.validators)

        # Check that SLAs are initialized
        self.assertIn("default", self.framework.slas)
        self.assertIn("ohlcv", self.framework.slas)
        self.assertIn("tick", self.framework.slas)
        self.assertIn("forex", self.framework.slas)
        self.assertIn("crypto", self.framework.slas)
        self.assertIn("stocks", self.framework.slas)

    def test_validate_ohlcv_data_valid(self):
        """Test validation of valid OHLCV data."""
        # Create valid OHLCV data
        data = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="1h"),
            "instrument": ["EUR_USD"] * 10,
            "open": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "high": [1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05],
            "low": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95],
            "close": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "volume": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        })

        # Mock the validator to return a valid result
        mock_validator = MagicMock()
        mock_validator.validate.return_value = ValidationResult(
            is_valid=True,
            message="Validation passed",
            severity=ValidationSeverity.INFO
        )

        self.framework.validation_engine.validators["ohlcv_standard"] = mock_validator

        # Validate data
        result = self.framework.validate_ohlcv_data(
            data=data,
            instrument_type="forex",
            quality_level=DataQualityLevel.STANDARD,
            generate_report=False
        )

        # Check result
        self.assertTrue(result)

        # Check that the validator was called
        mock_validator.validate.assert_called_once_with(data)

        # Check that metrics were recorded
        self.metrics_collector.record_gauge.assert_called()

    def test_validate_ohlcv_data_invalid(self):
        """Test validation of invalid OHLCV data."""
        # Create invalid OHLCV data (missing required columns)
        data = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="1h"),
            "instrument": ["EUR_USD"] * 10,
            # Missing open, high, low, close, volume
        })

        # Mock the validator to return an invalid result
        mock_validator = MagicMock()
        mock_validator.validate.return_value = ValidationResult(
            is_valid=False,
            message="Missing required columns",
            severity=ValidationSeverity.ERROR
        )

        self.framework.validation_engine.validators["ohlcv_standard"] = mock_validator

        # Validate data
        result = self.framework.validate_ohlcv_data(
            data=data,
            instrument_type="forex",
            quality_level=DataQualityLevel.STANDARD,
            generate_report=False
        )

        # Check result
        self.assertFalse(result)

        # Check that the validator was called
        mock_validator.validate.assert_called_once_with(data)

        # Check that metrics were recorded
        self.metrics_collector.record_gauge.assert_called()

        # Check that an alert was sent
        self.alert_manager.send_alert.assert_called_once()

    def test_validate_ohlcv_data_with_report(self):
        """Test validation of OHLCV data with report generation."""
        # Create valid OHLCV data
        data = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="1h"),
            "instrument": ["EUR_USD"] * 10,
            "open": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "high": [1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05],
            "low": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95],
            "close": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "volume": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        })

        # Mock the validator to return a valid result
        mock_validator = MagicMock()
        mock_validator.validate.return_value = ValidationResult(
            is_valid=True,
            message="Validation passed",
            severity=ValidationSeverity.INFO
        )

        self.framework.validation_engine.validators["ohlcv_standard"] = mock_validator

        # Mock the helper methods
        self.framework._detect_anomalies = MagicMock(return_value=[])
        self.framework._check_sla_breaches = MagicMock(return_value=[])
        self.framework._generate_recommendations = MagicMock(return_value=[])

        # Validate data with report
        result = self.framework.validate_ohlcv_data(
            data=data,
            instrument_type="forex",
            quality_level=DataQualityLevel.STANDARD,
            generate_report=True
        )

        # Check result
        self.assertIsInstance(result, DataQualityReport)
        self.assertEqual(result.data_type, "ohlcv")
        self.assertEqual(result.instrument, "EUR_USD")
        self.assertEqual(result.timeframe, "1h")
        self.assertTrue(result.is_valid)

        # Check that the validator was called
        mock_validator.validate.assert_called_once_with(data)

        # Check that metrics were recorded
        self.metrics_collector.record_gauge.assert_called()

        # Check that helper methods were called
        self.framework._detect_anomalies.assert_called_once()
        self.framework._check_sla_breaches.assert_called_once()
        self.framework._generate_recommendations.assert_called_once()

    def test_validate_tick_data(self):
        """Test validation of tick data."""
        # Create valid tick data
        data = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="1s"),
            "instrument": ["EUR_USD"] * 10,
            "bid": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "ask": [1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91, 2.01],
            "bid_volume": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            "ask_volume": [1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100, 10100]
        })

        # Mock the validator to return a valid result
        mock_validator = MagicMock()
        mock_validator.validate.return_value = ValidationResult(
            is_valid=True,
            message="Validation passed",
            severity=ValidationSeverity.INFO
        )

        self.framework.validation_engine.validators["tick_standard"] = mock_validator

        # Validate data
        result = self.framework.validate_tick_data(
            data=data,
            instrument_type="forex",
            quality_level=DataQualityLevel.STANDARD,
            generate_report=False
        )

        # Check result
        self.assertTrue(result)

        # Check that the validator was called
        mock_validator.validate.assert_called_once_with(data)

        # Check that metrics were recorded
        self.metrics_collector.record_gauge.assert_called()

    def test_validate_alternative_data(self):
        """Test validation of alternative data."""
        # Create valid news data
        data = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="1h"),
            "source": ["Reuters"] * 10,
            "title": [f"News {i}" for i in range(10)],
            "content": [f"Content {i}" for i in range(10)]
        })

        # Mock the validator to return a valid result
        mock_validator = MagicMock()
        mock_validator.validate.return_value = ValidationResult(
            is_valid=True,
            message="Validation passed",
            severity=ValidationSeverity.INFO
        )

        self.framework.validation_engine.validators["news"] = mock_validator

        # Validate data
        result = self.framework.validate_alternative_data(
            data=data,
            data_type="news",
            generate_report=False
        )

        # Check result
        self.assertTrue(result)

        # Check that the validator was called
        mock_validator.validate.assert_called_once_with(data)

        # Check that metrics were recorded
        self.metrics_collector.record_gauge.assert_called()

    def test_get_data_quality_sla(self):
        """Test getting data quality SLA."""
        # Get SLA for forex OHLCV
        sla = self.framework.get_data_quality_sla(
            instrument_type="forex",
            data_type="ohlcv"
        )

        # Check result
        self.assertIsInstance(sla, DataQualitySLA)
        self.assertEqual(sla.completeness, 99.95)

        # Get SLA for non-existent combination
        sla = self.framework.get_data_quality_sla(
            instrument_type="nonexistent",
            data_type="nonexistent"
        )

        # Check result (should fall back to default)
        self.assertIsInstance(sla, DataQualitySLA)
        self.assertEqual(sla.completeness, 99.5)

    def test_set_data_quality_sla(self):
        """Test setting data quality SLA."""
        # Create new SLA
        new_sla = DataQualitySLA(
            completeness=99.0,
            timeliness=98.0,
            accuracy=99.0,
            consistency=98.0,
            max_allowed_gaps_per_day=1,
            max_allowed_spikes_per_day=1,
            max_latency_seconds=2.0
        )

        # Set SLA
        self.framework.set_data_quality_sla(
            sla=new_sla,
            key="test"
        )

        # Check that SLA was set
        self.assertIn("test", self.framework.slas)
        self.assertEqual(self.framework.slas["test"].completeness, 99.0)

        # Get SLA
        sla = self.framework.get_data_quality_sla(
            instrument_type="test",
            data_type="ohlcv"
        )

        # Check result
        self.assertIsInstance(sla, DataQualitySLA)
        self.assertEqual(sla.completeness, 99.0)

    def test_get_data_quality_metrics(self):
        """Test getting data quality metrics."""
        # Create mock validation results
        result1 = ValidationResult(
            is_valid=True,
            message="Validation passed",
            severity=ValidationSeverity.INFO
        )

        result2 = ValidationResult(
            is_valid=False,
            message="Validation failed",
            severity=ValidationSeverity.ERROR
        )

        result3 = ValidationResult(
            is_valid=False,
            message="Validation warning",
            severity=ValidationSeverity.WARNING
        )

        # Add results to cache
        self.framework.validation_cache = {
            "ohlcv_EUR_USD": [result1, result2, result3],
            "tick_EUR_USD": [result1, result1],
            "ohlcv_GBP_USD": [result1]
        }

        # Set cache expiry
        now = datetime.utcnow()
        self.framework.cache_expiry = {
            "ohlcv_EUR_USD": now + timedelta(hours=1),
            "tick_EUR_USD": now + timedelta(hours=1),
            "ohlcv_GBP_USD": now + timedelta(hours=1)
        }

        # Get metrics for all OHLCV data
        metrics = self.framework.get_data_quality_metrics(
            data_type="ohlcv",
            lookback_hours=24
        )

        # Check result
        self.assertEqual(len(metrics), 2)  # EUR_USD and GBP_USD

        # Check EUR_USD metrics
        eur_usd_metrics = next(m for m in metrics if m.instrument == "EUR_USD")
        self.assertEqual(eur_usd_metrics.validation_count, 3)
        self.assertEqual(eur_usd_metrics.error_count, 1)
        self.assertEqual(eur_usd_metrics.warning_count, 1)

        # Get metrics for specific instrument
        metrics = self.framework.get_data_quality_metrics(
            instrument="EUR_USD",
            data_type="tick",
            lookback_hours=24
        )

        # Check result
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].instrument, "EUR_USD")
        self.assertEqual(metrics[0].data_type, "tick")
        self.assertEqual(metrics[0].validation_count, 2)
        self.assertEqual(metrics[0].error_count, 0)

        # Test expired cache
        self.framework.cache_expiry["ohlcv_GBP_USD"] = now - timedelta(hours=1)

        metrics = self.framework.get_data_quality_metrics(
            data_type="ohlcv",
            lookback_hours=24
        )

        # Check result (GBP_USD should be removed)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].instrument, "EUR_USD")


if __name__ == "__main__":
    unittest.main()
