"""
Market Data Quality Framework

This module provides a comprehensive framework for validating, monitoring, and ensuring
the quality of market data throughout the trading platform. It implements multi-layered
validation for all market data inputs with specialized validators for different data types,
configurable severity levels, and detailed validation reporting.

Key features:
- Multi-layered validation pipeline for all market data types
- Detection systems for false ticks, gaps, spikes, and other anomalies
- Real-time alerting for data quality issues
- Configurable validation rules based on instrument and data type
- Comprehensive reporting and metrics for data quality
- Integration with monitoring and alerting systems
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from common_lib.exceptions import DataValidationError, DataQualityError
from common_lib.schemas import OHLCVData, TickData
from common_lib.monitoring import MetricsCollector, AlertManager
from data_pipeline_service.validation.validation_engine import (
    ValidationResult, ValidationSeverity, DataValidationEngine,
    CompositeValidator, SchemaValidator, NullValidator, OutlierValidator,
    TimeSeriesContinuityValidator
)
from data_pipeline_service.validation.ohlcv_validators import (
    CandlestickPatternValidator, GapDetectionValidator, VolumeChangeValidator
)
from data_pipeline_service.validation.tick_validators import (
    QuoteSequenceValidator, TickFrequencyValidator, TickVolumeConsistencyValidator
)

logger = logging.getLogger(__name__)


class DataQualityLevel(str, Enum):
    """Data quality validation levels"""
    BASIC = "basic"  # Schema, nulls, basic range checks
    STANDARD = "standard"  # Basic + continuity, outliers, pattern checks
    COMPREHENSIVE = "comprehensive"  # Standard + advanced anomaly detection
    STRICT = "strict"  # Comprehensive with stricter thresholds


class DataQualitySLA(BaseModel):
    """Service Level Agreement for data quality"""
    completeness: float = Field(99.5, description="Percentage of required fields that must be present")
    timeliness: float = Field(99.0, description="Percentage of data that must arrive within SLA timeframe")
    accuracy: float = Field(99.9, description="Percentage of data that must pass validation checks")
    consistency: float = Field(99.5, description="Percentage of data that must be consistent across sources")
    max_allowed_gaps_per_day: int = Field(0, description="Maximum number of gaps allowed per day")
    max_allowed_spikes_per_day: int = Field(0, description="Maximum number of price spikes allowed per day")
    max_latency_seconds: float = Field(1.0, description="Maximum latency for real-time data in seconds")


class DataQualityMetrics(BaseModel):
    """Metrics for data quality monitoring"""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    validation_time_ms: float = 0.0
    error_rate: float = 0.0
    completeness_score: float = 100.0
    timeliness_score: float = 100.0
    accuracy_score: float = 100.0
    consistency_score: float = 100.0
    anomalies_detected: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DataQualityReport(BaseModel):
    """Comprehensive data quality report"""
    report_id: str
    data_type: str
    instrument: str
    timeframe: Optional[str] = None
    start_time: datetime
    end_time: datetime
    quality_level: DataQualityLevel
    metrics: DataQualityMetrics
    validation_results: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    sla_breaches: List[Dict[str, Any]]
    is_valid: bool
    recommendations: List[str] = []


class MarketDataQualityFramework:
    """
    Comprehensive framework for ensuring market data quality.

    This framework provides multi-layered validation for all market data types,
    with configurable validation rules, severity levels, and detailed reporting.
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        """
        Initialize the Market Data Quality Framework.

        Args:
            metrics_collector: Optional metrics collector for monitoring
            alert_manager: Optional alert manager for notifications
        """
        self.validation_engine = DataValidationEngine()
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.slas: Dict[str, DataQualitySLA] = {}

        # Initialize default SLAs
        self._initialize_default_slas()

        # Initialize validation rules
        self._initialize_validation_rules()

        # Cache for recent validation results
        self.validation_cache: Dict[str, List[ValidationResult]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(hours=24)

        logger.info("Market Data Quality Framework initialized")

    def _initialize_default_slas(self) -> None:
        """Initialize default SLAs for different data types and instruments"""
        # Default SLA for all data
        self.slas["default"] = DataQualitySLA()

        # SLAs for specific data types
        self.slas["ohlcv"] = DataQualitySLA(
            completeness=99.9,
            timeliness=99.5,
            accuracy=99.95,
            consistency=99.8,
            max_allowed_gaps_per_day=0,
            max_allowed_spikes_per_day=0,
            max_latency_seconds=5.0
        )

        self.slas["tick"] = DataQualitySLA(
            completeness=99.8,
            timeliness=99.9,
            accuracy=99.9,
            consistency=99.7,
            max_allowed_gaps_per_day=0,
            max_allowed_spikes_per_day=0,
            max_latency_seconds=0.5
        )

        # SLAs for specific instrument types
        self.slas["forex"] = DataQualitySLA(
            completeness=99.95,
            timeliness=99.9,
            accuracy=99.99,
            consistency=99.9,
            max_allowed_gaps_per_day=0,
            max_allowed_spikes_per_day=0,
            max_latency_seconds=0.2
        )

        self.slas["crypto"] = DataQualitySLA(
            completeness=99.8,
            timeliness=99.5,
            accuracy=99.9,
            consistency=99.5,
            max_allowed_gaps_per_day=2,
            max_allowed_spikes_per_day=5,
            max_latency_seconds=1.0
        )

        self.slas["stocks"] = DataQualitySLA(
            completeness=99.9,
            timeliness=99.7,
            accuracy=99.95,
            consistency=99.8,
            max_allowed_gaps_per_day=0,
            max_allowed_spikes_per_day=2,
            max_latency_seconds=2.0
        )

    def _initialize_validation_rules(self) -> None:
        """Initialize validation rules for different data types"""
        # Register OHLCV validators for different quality levels
        self._register_ohlcv_validators()

        # Register tick data validators for different quality levels
        self._register_tick_validators()

        # Register alternative data validators
        self._register_alternative_data_validators()

    def _register_ohlcv_validators(self) -> None:
        """Register validators for OHLCV data at different quality levels"""
        # Basic OHLCV validator
        basic_ohlcv = CompositeValidator("basic-ohlcv-validator")

        # Schema validation
        schema_validator = SchemaValidator(
            required_columns=["timestamp", "instrument", "open", "high", "low", "close", "volume"],
            dtypes={
                "timestamp": "datetime64[ns]",
                "instrument": "object",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64"
            }
        )
        basic_ohlcv.add_validator(schema_validator)

        # Null validation
        null_validator = NullValidator(
            non_nullable_columns=["timestamp", "instrument", "open", "high", "low", "close"],
            max_null_percentage={"volume": 5.0},  # Allow up to 5% null volume
            severity=ValidationSeverity.ERROR
        )
        basic_ohlcv.add_validator(null_validator)

        # Register basic validator
        self.validation_engine.register_validator("ohlcv_basic", basic_ohlcv)

        # Standard OHLCV validator (includes basic + more)
        standard_ohlcv = CompositeValidator("standard-ohlcv-validator")

        # Include all basic validators
        standard_ohlcv.add_validator(schema_validator)
        standard_ohlcv.add_validator(null_validator)

        # Add continuity validation
        continuity_validator = TimeSeriesContinuityValidator(
            timestamp_column="timestamp",
            max_gap=pd.Timedelta(hours=24),  # Maximum allowed gap
            severity=ValidationSeverity.WARNING
        )
        standard_ohlcv.add_validator(continuity_validator)

        # Add outlier validation
        outlier_validator = OutlierValidator(
            columns_to_check=["open", "high", "low", "close", "volume"],
            method="zscore",
            threshold=4.0,  # 4 standard deviations
            severity=ValidationSeverity.WARNING
        )
        standard_ohlcv.add_validator(outlier_validator)

        # Register standard validator
        self.validation_engine.register_validator("ohlcv_standard", standard_ohlcv)

        # Comprehensive OHLCV validator (includes standard + more)
        comprehensive_ohlcv = CompositeValidator("comprehensive-ohlcv-validator")

        # Include all standard validators
        comprehensive_ohlcv.add_validator(schema_validator)
        comprehensive_ohlcv.add_validator(null_validator)
        comprehensive_ohlcv.add_validator(continuity_validator)
        comprehensive_ohlcv.add_validator(outlier_validator)

        # Add gap detection
        gap_validator = GapDetectionValidator(
            max_gap_percentage=3.0,  # 3% maximum gap
            severity=ValidationSeverity.WARNING
        )
        comprehensive_ohlcv.add_validator(gap_validator)

        # Add volume change validation
        volume_validator = VolumeChangeValidator(
            max_volume_spike_factor=10.0,
            min_volume_drop_factor=0.1,
            severity=ValidationSeverity.WARNING
        )
        comprehensive_ohlcv.add_validator(volume_validator)

        # Add candlestick pattern validation
        pattern_validator = CandlestickPatternValidator(
            severity=ValidationSeverity.INFO
        )
        comprehensive_ohlcv.add_validator(pattern_validator)

        # Register comprehensive validator
        self.validation_engine.register_validator("ohlcv_comprehensive", comprehensive_ohlcv)

        # Strict OHLCV validator (includes comprehensive with stricter thresholds)
        strict_ohlcv = CompositeValidator("strict-ohlcv-validator")

        # Include schema and null validators with stricter settings
        strict_schema_validator = SchemaValidator(
            required_columns=["timestamp", "instrument", "open", "high", "low", "close", "volume"],
            dtypes={
                "timestamp": "datetime64[ns]",
                "instrument": "object",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64"
            }
        )
        strict_ohlcv.add_validator(strict_schema_validator)

        strict_null_validator = NullValidator(
            non_nullable_columns=["timestamp", "instrument", "open", "high", "low", "close", "volume"],
            max_null_percentage={},  # No nulls allowed
            severity=ValidationSeverity.ERROR
        )
        strict_ohlcv.add_validator(strict_null_validator)

        # Add stricter continuity validation
        strict_continuity_validator = TimeSeriesContinuityValidator(
            timestamp_column="timestamp",
            max_gap=pd.Timedelta(minutes=30),  # Stricter maximum allowed gap
            severity=ValidationSeverity.ERROR
        )
        strict_ohlcv.add_validator(strict_continuity_validator)

        # Add stricter outlier validation
        strict_outlier_validator = OutlierValidator(
            columns_to_check=["open", "high", "low", "close", "volume"],
            method="zscore",
            threshold=3.0,  # 3 standard deviations (stricter)
            severity=ValidationSeverity.ERROR
        )
        strict_ohlcv.add_validator(strict_outlier_validator)

        # Add stricter gap detection
        strict_gap_validator = GapDetectionValidator(
            max_gap_percentage=1.0,  # 1% maximum gap (stricter)
            severity=ValidationSeverity.ERROR
        )
        strict_ohlcv.add_validator(strict_gap_validator)

        # Add stricter volume change validation
        strict_volume_validator = VolumeChangeValidator(
            max_volume_spike_factor=5.0,  # Stricter
            min_volume_drop_factor=0.2,  # Stricter
            severity=ValidationSeverity.ERROR
        )
        strict_ohlcv.add_validator(strict_volume_validator)

        # Register strict validator
        self.validation_engine.register_validator("ohlcv_strict", strict_ohlcv)

    def _register_tick_validators(self) -> None:
        """Register validators for tick data at different quality levels"""
        # Basic tick validator
        basic_tick = CompositeValidator("basic-tick-validator")

        # Schema validation
        schema_validator = SchemaValidator(
            required_columns=["timestamp", "instrument", "bid", "ask"],
            dtypes={
                "timestamp": "datetime64[ns]",
                "instrument": "object",
                "bid": "float64",
                "ask": "float64"
            }
        )
        basic_tick.add_validator(schema_validator)

        # Null validation
        null_validator = NullValidator(
            non_nullable_columns=["timestamp", "instrument", "bid", "ask"],
            max_null_percentage={},  # No nulls allowed in required fields
            severity=ValidationSeverity.ERROR
        )
        basic_tick.add_validator(null_validator)

        # Register basic validator
        self.validation_engine.register_validator("tick_basic", basic_tick)

        # Standard tick validator (includes basic + more)
        standard_tick = CompositeValidator("standard-tick-validator")

        # Include all basic validators
        standard_tick.add_validator(schema_validator)
        standard_tick.add_validator(null_validator)

        # Add bid-ask validation
        quote_validator = QuoteSequenceValidator(
            bid_column="bid",
            ask_column="ask",
            severity=ValidationSeverity.WARNING
        )
        standard_tick.add_validator(quote_validator)

        # Add outlier validation
        outlier_validator = OutlierValidator(
            columns_to_check=["bid", "ask"],
            method="zscore",
            threshold=4.0,  # 4 standard deviations
            severity=ValidationSeverity.WARNING
        )
        standard_tick.add_validator(outlier_validator)

        # Register standard validator
        self.validation_engine.register_validator("tick_standard", standard_tick)

        # Comprehensive tick validator (includes standard + more)
        comprehensive_tick = CompositeValidator("comprehensive-tick-validator")

        # Include all standard validators
        comprehensive_tick.add_validator(schema_validator)
        comprehensive_tick.add_validator(null_validator)
        comprehensive_tick.add_validator(quote_validator)
        comprehensive_tick.add_validator(outlier_validator)

        # Add frequency validation
        frequency_validator = TickFrequencyValidator(
            max_gap_seconds=5,  # Maximum 5 seconds between ticks
            severity=ValidationSeverity.WARNING
        )
        comprehensive_tick.add_validator(frequency_validator)

        # Add volume consistency validation
        volume_validator = TickVolumeConsistencyValidator(
            bid_volume_column="bid_volume",
            ask_volume_column="ask_volume",
            severity=ValidationSeverity.WARNING
        )
        comprehensive_tick.add_validator(volume_validator)

        # Register comprehensive validator
        self.validation_engine.register_validator("tick_comprehensive", comprehensive_tick)

        # Strict tick validator (includes comprehensive with stricter thresholds)
        strict_tick = CompositeValidator("strict-tick-validator")

        # Include schema and null validators with stricter settings
        strict_tick.add_validator(schema_validator)
        strict_tick.add_validator(null_validator)

        # Add stricter bid-ask validation
        strict_quote_validator = QuoteSequenceValidator(
            bid_column="bid",
            ask_column="ask",
            severity=ValidationSeverity.ERROR
        )
        strict_tick.add_validator(strict_quote_validator)

        # Add stricter outlier validation
        strict_outlier_validator = OutlierValidator(
            columns_to_check=["bid", "ask"],
            method="zscore",
            threshold=3.0,  # 3 standard deviations (stricter)
            severity=ValidationSeverity.ERROR
        )
        strict_tick.add_validator(strict_outlier_validator)

        # Add stricter frequency validation
        strict_frequency_validator = TickFrequencyValidator(
            max_gap_seconds=1,  # Maximum 1 second between ticks (stricter)
            severity=ValidationSeverity.ERROR
        )
        strict_tick.add_validator(strict_frequency_validator)

        # Add stricter volume consistency validation
        strict_volume_validator = TickVolumeConsistencyValidator(
            bid_volume_column="bid_volume",
            ask_volume_column="ask_volume",
            severity=ValidationSeverity.ERROR
        )
        strict_tick.add_validator(strict_volume_validator)

        # Register strict validator
        self.validation_engine.register_validator("tick_strict", strict_tick)

    def _register_alternative_data_validators(self) -> None:
        """Register validators for alternative data sources"""
        # News data validator
        news_validator = CompositeValidator("news-validator")

        # Schema validation for news data
        news_schema_validator = SchemaValidator(
            required_columns=["timestamp", "source", "title", "content"],
            dtypes={
                "timestamp": "datetime64[ns]",
                "source": "object",
                "title": "object",
                "content": "object"
            }
        )
        news_validator.add_validator(news_schema_validator)

        # Null validation for news data
        news_null_validator = NullValidator(
            non_nullable_columns=["timestamp", "source", "title"],
            max_null_percentage={"content": 5.0},  # Allow up to 5% null content
            severity=ValidationSeverity.WARNING
        )
        news_validator.add_validator(news_null_validator)

        # Register news validator
        self.validation_engine.register_validator("news", news_validator)

        # Economic indicator validator
        economic_validator = CompositeValidator("economic-indicator-validator")

        # Schema validation for economic indicators
        economic_schema_validator = SchemaValidator(
            required_columns=["timestamp", "country", "indicator", "value", "previous_value", "forecast_value"],
            dtypes={
                "timestamp": "datetime64[ns]",
                "country": "object",
                "indicator": "object",
                "value": "float64",
                "previous_value": "float64",
                "forecast_value": "float64"
            }
        )
        economic_validator.add_validator(economic_schema_validator)

        # Null validation for economic indicators
        economic_null_validator = NullValidator(
            non_nullable_columns=["timestamp", "country", "indicator", "value"],
            max_null_percentage={
                "previous_value": 10.0,  # Allow up to 10% null previous values
                "forecast_value": 20.0   # Allow up to 20% null forecast values
            },
            severity=ValidationSeverity.WARNING
        )
        economic_validator.add_validator(economic_null_validator)

        # Register economic validator
        self.validation_engine.register_validator("economic", economic_validator)

        # Social sentiment validator
        sentiment_validator = CompositeValidator("social-sentiment-validator")

        # Schema validation for sentiment data
        sentiment_schema_validator = SchemaValidator(
            required_columns=["timestamp", "instrument", "sentiment_score", "volume"],
            dtypes={
                "timestamp": "datetime64[ns]",
                "instrument": "object",
                "sentiment_score": "float64",
                "volume": "float64"
            }
        )
        sentiment_validator.add_validator(sentiment_schema_validator)

        # Range validation for sentiment scores
        class SentimentRangeValidator:
            def validate(self, data: pd.DataFrame) -> ValidationResult:
                if "sentiment_score" not in data.columns:
                    return ValidationResult(
                        is_valid=False,
                        message="Missing sentiment_score column",
                        severity=ValidationSeverity.ERROR
                    )

                # Check if sentiment scores are in valid range [-1, 1]
                invalid_scores = data[(data["sentiment_score"] < -1) | (data["sentiment_score"] > 1)]

                if not invalid_scores.empty:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Found {len(invalid_scores)} sentiment scores outside valid range [-1, 1]",
                        details={"invalid_scores": invalid_scores.to_dict("records")[:10]},
                        severity=ValidationSeverity.ERROR
                    )

                return ValidationResult(is_valid=True, message="Sentiment scores are in valid range")

        sentiment_validator.add_validator(SentimentRangeValidator())

        # Register sentiment validator
        self.validation_engine.register_validator("sentiment", sentiment_validator)

    def validate_ohlcv_data(
        self,
        data: pd.DataFrame,
        instrument_type: str = "forex",
        quality_level: DataQualityLevel = DataQualityLevel.STANDARD,
        generate_report: bool = False
    ) -> Union[bool, DataQualityReport]:
        """
        Validate OHLCV data against quality rules.

        Args:
            data: DataFrame with OHLCV data
            instrument_type: Type of instrument (forex, crypto, stocks)
            quality_level: Quality level to apply
            generate_report: Whether to generate a detailed report

        Returns:
            Boolean indicating if data is valid, or a detailed report if requested
        """
        start_time = datetime.utcnow()

        # Select the appropriate validator based on quality level
        validator_key = f"ohlcv_{quality_level.value}"

        if validator_key not in self.validation_engine.validators:
            logger.error(f"No validator found for {validator_key}")
            raise ValueError(f"No validator found for {validator_key}")

        # Get the SLA for this instrument type
        sla = self.slas.get(instrument_type, self.slas["default"])

        # Validate the data
        validation_result = self.validation_engine.validators[validator_key].validate(data)

        # Calculate validation time
        validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # in ms

        # Update metrics
        metrics = DataQualityMetrics(
            total_records=len(data),
            valid_records=len(data) if validation_result.is_valid else 0,
            invalid_records=0 if validation_result.is_valid else len(data),
            validation_time_ms=validation_time,
            error_rate=0.0 if validation_result.is_valid else 100.0,
            timestamp=datetime.utcnow()
        )

        # Publish metrics if collector is available
        if self.metrics_collector:
            self.metrics_collector.record_gauge("market_data.quality.total_records", metrics.total_records,
                                               {"data_type": "ohlcv", "instrument_type": instrument_type})
            self.metrics_collector.record_gauge("market_data.quality.valid_records", metrics.valid_records,
                                               {"data_type": "ohlcv", "instrument_type": instrument_type})
            self.metrics_collector.record_gauge("market_data.quality.invalid_records", metrics.invalid_records,
                                               {"data_type": "ohlcv", "instrument_type": instrument_type})
            self.metrics_collector.record_gauge("market_data.quality.validation_time_ms", metrics.validation_time_ms,
                                               {"data_type": "ohlcv", "instrument_type": instrument_type})
            self.metrics_collector.record_gauge("market_data.quality.error_rate", metrics.error_rate,
                                               {"data_type": "ohlcv", "instrument_type": instrument_type})

        # Send alerts if necessary
        if not validation_result.is_valid and self.alert_manager:
            self.alert_manager.send_alert(
                alert_name="market_data_quality_issue",
                severity="warning" if validation_result.severity == ValidationSeverity.WARNING else "error",
                message=f"OHLCV data quality issue: {validation_result.message}",
                details={
                    "data_type": "ohlcv",
                    "instrument_type": instrument_type,
                    "quality_level": quality_level.value,
                    "validation_result": validation_result.dict() if hasattr(validation_result, "dict") else str(validation_result)
                }
            )

        # Update validation cache
        if "instrument" in data.columns and not data.empty:
            instrument = data["instrument"].iloc[0]
            cache_key = f"ohlcv_{instrument}"

            if cache_key not in self.validation_cache:
                self.validation_cache[cache_key] = []

            self.validation_cache[cache_key].append(validation_result)
            self.cache_expiry[cache_key] = datetime.utcnow() + self.cache_ttl

            # Limit cache size
            if len(self.validation_cache[cache_key]) > 100:
                self.validation_cache[cache_key] = self.validation_cache[cache_key][-100:]

        # Generate report if requested
        if generate_report:
            # Get instrument from data if available
            instrument = data["instrument"].iloc[0] if "instrument" in data.columns and not data.empty else "unknown"

            # Get timeframe from data if available (assuming consistent timeframe)
            timeframe = None
            if "timestamp" in data.columns and len(data) >= 2:
                timestamps = data["timestamp"].sort_values()
                diff = timestamps.diff().dropna().median()

                # Determine timeframe based on median difference
                if diff <= pd.Timedelta(minutes=1):
                    timeframe = "1m"
                elif diff <= pd.Timedelta(minutes=5):
                    timeframe = "5m"
                elif diff <= pd.Timedelta(minutes=15):
                    timeframe = "15m"
                elif diff <= pd.Timedelta(minutes=30):
                    timeframe = "30m"
                elif diff <= pd.Timedelta(hours=1):
                    timeframe = "1h"
                elif diff <= pd.Timedelta(hours=4):
                    timeframe = "4h"
                elif diff <= pd.Timedelta(days=1):
                    timeframe = "1d"
                else:
                    timeframe = "unknown"

            # Detect anomalies
            anomalies = self._detect_anomalies(data, "ohlcv")

            # Check SLA breaches
            sla_breaches = self._check_sla_breaches(data, sla, "ohlcv")

            # Generate recommendations
            recommendations = self._generate_recommendations(validation_result, anomalies, sla_breaches)

            # Create report
            report = DataQualityReport(
                report_id=str(uuid.uuid4()),
                data_type="ohlcv",
                instrument=instrument,
                timeframe=timeframe,
                start_time=start_time,
                end_time=datetime.utcnow(),
                quality_level=quality_level,
                metrics=metrics,
                validation_results=[validation_result.dict() if hasattr(validation_result, "dict") else {"message": str(validation_result)}],
                anomalies=anomalies,
                sla_breaches=sla_breaches,
                is_valid=validation_result.is_valid,
                recommendations=recommendations
            )

            return report

        return validation_result.is_valid

    def validate_tick_data(
        self,
        data: pd.DataFrame,
        instrument_type: str = "forex",
        quality_level: DataQualityLevel = DataQualityLevel.STANDARD,
        generate_report: bool = False
    ) -> Union[bool, DataQualityReport]:
        """
        Validate tick data against quality rules.

        Args:
            data: DataFrame with tick data
            instrument_type: Type of instrument (forex, crypto, stocks)
            quality_level: Quality level to apply
            generate_report: Whether to generate a detailed report

        Returns:
            Boolean indicating if data is valid, or a detailed report if requested
        """
        start_time = datetime.utcnow()

        # Select the appropriate validator based on quality level
        validator_key = f"tick_{quality_level.value}"

        if validator_key not in self.validation_engine.validators:
            logger.error(f"No validator found for {validator_key}")
            raise ValueError(f"No validator found for {validator_key}")

        # Get the SLA for this instrument type
        sla = self.slas.get(instrument_type, self.slas["default"])

        # Validate the data
        validation_result = self.validation_engine.validators[validator_key].validate(data)

        # Calculate validation time
        validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # in ms

        # Update metrics
        metrics = DataQualityMetrics(
            total_records=len(data),
            valid_records=len(data) if validation_result.is_valid else 0,
            invalid_records=0 if validation_result.is_valid else len(data),
            validation_time_ms=validation_time,
            error_rate=0.0 if validation_result.is_valid else 100.0,
            timestamp=datetime.utcnow()
        )

        # Publish metrics if collector is available
        if self.metrics_collector:
            self.metrics_collector.record_gauge("market_data.quality.total_records", metrics.total_records,
                                               {"data_type": "tick", "instrument_type": instrument_type})
            self.metrics_collector.record_gauge("market_data.quality.valid_records", metrics.valid_records,
                                               {"data_type": "tick", "instrument_type": instrument_type})
            self.metrics_collector.record_gauge("market_data.quality.invalid_records", metrics.invalid_records,
                                               {"data_type": "tick", "instrument_type": instrument_type})
            self.metrics_collector.record_gauge("market_data.quality.validation_time_ms", metrics.validation_time_ms,
                                               {"data_type": "tick", "instrument_type": instrument_type})
            self.metrics_collector.record_gauge("market_data.quality.error_rate", metrics.error_rate,
                                               {"data_type": "tick", "instrument_type": instrument_type})

        # Send alerts if necessary
        if not validation_result.is_valid and self.alert_manager:
            self.alert_manager.send_alert(
                alert_name="market_data_quality_issue",
                severity="warning" if validation_result.severity == ValidationSeverity.WARNING else "error",
                message=f"Tick data quality issue: {validation_result.message}",
                details={
                    "data_type": "tick",
                    "instrument_type": instrument_type,
                    "quality_level": quality_level.value,
                    "validation_result": validation_result.dict() if hasattr(validation_result, "dict") else str(validation_result)
                }
            )

        # Update validation cache
        if "instrument" in data.columns and not data.empty:
            instrument = data["instrument"].iloc[0]
            cache_key = f"tick_{instrument}"

            if cache_key not in self.validation_cache:
                self.validation_cache[cache_key] = []

            self.validation_cache[cache_key].append(validation_result)
            self.cache_expiry[cache_key] = datetime.utcnow() + self.cache_ttl

            # Limit cache size
            if len(self.validation_cache[cache_key]) > 100:
                self.validation_cache[cache_key] = self.validation_cache[cache_key][-100:]

        # Generate report if requested
        if generate_report:
            # Get instrument from data if available
            instrument = data["instrument"].iloc[0] if "instrument" in data.columns and not data.empty else "unknown"

            # Detect anomalies
            anomalies = self._detect_anomalies(data, "tick")

            # Check SLA breaches
            sla_breaches = self._check_sla_breaches(data, sla, "tick")

            # Generate recommendations
            recommendations = self._generate_recommendations(validation_result, anomalies, sla_breaches)

            # Create report
            report = DataQualityReport(
                report_id=str(uuid.uuid4()),
                data_type="tick",
                instrument=instrument,
                timeframe=None,  # Tick data doesn't have a timeframe
                start_time=start_time,
                end_time=datetime.utcnow(),
                quality_level=quality_level,
                metrics=metrics,
                validation_results=[validation_result.dict() if hasattr(validation_result, "dict") else {"message": str(validation_result)}],
                anomalies=anomalies,
                sla_breaches=sla_breaches,
                is_valid=validation_result.is_valid,
                recommendations=recommendations
            )

            return report

        return validation_result.is_valid

    def validate_alternative_data(
        self,
        data: pd.DataFrame,
        data_type: str,
        generate_report: bool = False
    ) -> Union[bool, DataQualityReport]:
        """
        Validate alternative data against quality rules.

        Args:
            data: DataFrame with alternative data
            data_type: Type of alternative data (news, economic, sentiment)
            generate_report: Whether to generate a detailed report

        Returns:
            Boolean indicating if data is valid, or a detailed report if requested
        """
        start_time = datetime.utcnow()

        # Check if we have a validator for this data type
        if data_type not in self.validation_engine.validators:
            logger.error(f"No validator found for {data_type}")
            raise ValueError(f"No validator found for {data_type}")

        # Get the SLA for this data type
        sla = self.slas.get("default")

        # Validate the data
        validation_result = self.validation_engine.validators[data_type].validate(data)

        # Calculate validation time
        validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # in ms

        # Update metrics
        metrics = DataQualityMetrics(
            total_records=len(data),
            valid_records=len(data) if validation_result.is_valid else 0,
            invalid_records=0 if validation_result.is_valid else len(data),
            validation_time_ms=validation_time,
            error_rate=0.0 if validation_result.is_valid else 100.0,
            timestamp=datetime.utcnow()
        )

        # Publish metrics if collector is available
        if self.metrics_collector:
            self.metrics_collector.record_gauge("market_data.quality.total_records", metrics.total_records,
                                               {"data_type": data_type})
            self.metrics_collector.record_gauge("market_data.quality.valid_records", metrics.valid_records,
                                               {"data_type": data_type})
            self.metrics_collector.record_gauge("market_data.quality.invalid_records", metrics.invalid_records,
                                               {"data_type": data_type})
            self.metrics_collector.record_gauge("market_data.quality.validation_time_ms", metrics.validation_time_ms,
                                               {"data_type": data_type})
            self.metrics_collector.record_gauge("market_data.quality.error_rate", metrics.error_rate,
                                               {"data_type": data_type})

        # Send alerts if necessary
        if not validation_result.is_valid and self.alert_manager:
            self.alert_manager.send_alert(
                alert_name="market_data_quality_issue",
                severity="warning" if validation_result.severity == ValidationSeverity.WARNING else "error",
                message=f"{data_type.capitalize()} data quality issue: {validation_result.message}",
                details={
                    "data_type": data_type,
                    "validation_result": validation_result.dict() if hasattr(validation_result, "dict") else str(validation_result)
                }
            )

        # Generate report if requested
        if generate_report:
            # Detect anomalies
            anomalies = self._detect_anomalies(data, data_type)

            # Check SLA breaches
            sla_breaches = self._check_sla_breaches(data, sla, data_type)

            # Generate recommendations
            recommendations = self._generate_recommendations(validation_result, anomalies, sla_breaches)

            # Create report
            report = DataQualityReport(
                report_id=str(uuid.uuid4()),
                data_type=data_type,
                instrument="N/A",
                timeframe=None,
                start_time=start_time,
                end_time=datetime.utcnow(),
                quality_level=DataQualityLevel.STANDARD,  # Default for alternative data
                metrics=metrics,
                validation_results=[validation_result.dict() if hasattr(validation_result, "dict") else {"message": str(validation_result)}],
                anomalies=anomalies,
                sla_breaches=sla_breaches,
                is_valid=validation_result.is_valid,
                recommendations=recommendations
            )

            return report

        return validation_result.is_valid

    def _detect_anomalies(self, data: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the data.

        Args:
            data: DataFrame with data
            data_type: Type of data

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if data_type == "ohlcv":
            # Check for price gaps
            if all(col in data.columns for col in ["open", "close", "high", "low"]):
                # Sort data by timestamp if available
                if "timestamp" in data.columns:
                    data = data.sort_values("timestamp")

                # Check for gaps between close and next open
                if len(data) > 1:
                    close_prices = data["close"].iloc[:-1].values
                    next_open_prices = data["open"].iloc[1:].values

                    # Calculate percentage gaps
                    gaps = np.abs(next_open_prices - close_prices) / close_prices * 100

                    # Find significant gaps (> 0.5%)
                    significant_gaps = np.where(gaps > 0.5)[0]

                    for idx in significant_gaps:
                        anomalies.append({
                            "type": "price_gap",
                            "severity": "medium" if gaps[idx] < 2.0 else "high",
                            "details": {
                                "index": int(idx),
                                "timestamp": data["timestamp"].iloc[idx+1].isoformat() if "timestamp" in data.columns else None,
                                "close_price": float(close_prices[idx]),
                                "next_open_price": float(next_open_prices[idx]),
                                "gap_percentage": float(gaps[idx])
                            }
                        })

                # Check for price spikes
                for col in ["high", "low"]:
                    # Calculate rolling median and standard deviation
                    rolling_median = data[col].rolling(window=20, min_periods=5).median()
                    rolling_std = data[col].rolling(window=20, min_periods=5).std()

                    # Calculate z-scores
                    z_scores = np.abs((data[col] - rolling_median) / rolling_std)

                    # Find spikes (z-score > 3)
                    spikes = np.where(z_scores > 3)[0]

                    for idx in spikes:
                        if not np.isnan(z_scores[idx]):
                            anomalies.append({
                                "type": f"{col}_spike",
                                "severity": "medium" if z_scores[idx] < 5 else "high",
                                "details": {
                                    "index": int(idx),
                                    "timestamp": data["timestamp"].iloc[idx].isoformat() if "timestamp" in data.columns else None,
                                    "value": float(data[col].iloc[idx]),
                                    "median": float(rolling_median.iloc[idx]),
                                    "z_score": float(z_scores[idx])
                                }
                            })

        elif data_type == "tick":
            # Check for bid-ask spread anomalies
            if all(col in data.columns for col in ["bid", "ask"]):
                # Calculate spread
                data["spread"] = data["ask"] - data["bid"]
                data["spread_pct"] = data["spread"] / data["bid"] * 100

                # Calculate rolling median and standard deviation of spread
                rolling_median = data["spread_pct"].rolling(window=100, min_periods=20).median()
                rolling_std = data["spread_pct"].rolling(window=100, min_periods=20).std()

                # Calculate z-scores
                z_scores = np.abs((data["spread_pct"] - rolling_median) / rolling_std)

                # Find anomalous spreads (z-score > 3)
                anomalous_spreads = np.where(z_scores > 3)[0]

                for idx in anomalous_spreads:
                    if not np.isnan(z_scores[idx]):
                        anomalies.append({
                            "type": "spread_anomaly",
                            "severity": "medium" if z_scores[idx] < 5 else "high",
                            "details": {
                                "index": int(idx),
                                "timestamp": data["timestamp"].iloc[idx].isoformat() if "timestamp" in data.columns else None,
                                "bid": float(data["bid"].iloc[idx]),
                                "ask": float(data["ask"].iloc[idx]),
                                "spread_pct": float(data["spread_pct"].iloc[idx]),
                                "z_score": float(z_scores[idx])
                            }
                        })

        # Limit the number of anomalies to return
        return anomalies[:100]

    def _check_sla_breaches(self, data: pd.DataFrame, sla: DataQualitySLA, data_type: str) -> List[Dict[str, Any]]:
        """
        Check for SLA breaches in the data.

        Args:
            data: DataFrame with data
            sla: SLA to check against
            data_type: Type of data

        Returns:
            List of SLA breaches
        """
        sla_breaches = []

        # Check for completeness (missing values)
        for col in data.columns:
            null_count = data[col].isnull().sum()
            null_percentage = null_count / len(data) * 100 if len(data) > 0 else 0
            completeness = 100 - null_percentage

            if completeness < sla.completeness:
                sla_breaches.append({
                    "type": "completeness",
                    "field": col,
                    "severity": "high" if (sla.completeness - completeness) > 5 else "medium",
                    "details": {
                        "actual": float(completeness),
                        "required": float(sla.completeness),
                        "missing_count": int(null_count)
                    }
                })

        # Check for timeliness (if timestamp is available)
        if "timestamp" in data.columns and not data.empty:
            # Calculate data freshness
            now = pd.Timestamp(datetime.utcnow())
            latest_timestamp = pd.Timestamp(data["timestamp"].max())

            # Calculate age in seconds
            age_seconds = (now - latest_timestamp).total_seconds()

            if age_seconds > sla.max_latency_seconds:
                sla_breaches.append({
                    "type": "timeliness",
                    "severity": "high" if age_seconds > sla.max_latency_seconds * 5 else "medium",
                    "details": {
                        "actual_latency": float(age_seconds),
                        "max_allowed_latency": float(sla.max_latency_seconds),
                        "latest_timestamp": latest_timestamp.isoformat()
                    }
                })

        # Check for gaps (if timestamp is available)
        if "timestamp" in data.columns and len(data) > 1:
            # Sort by timestamp
            sorted_data = data.sort_values("timestamp")

            # Calculate time differences
            time_diffs = sorted_data["timestamp"].diff().dropna()

            # Count gaps
            if data_type == "ohlcv":
                # For OHLCV data, check for missing bars
                # Determine the expected frequency
                median_diff = time_diffs.median()

                # Count gaps that are significantly larger than the median
                gaps = time_diffs[time_diffs > median_diff * 1.5]
                gap_count = len(gaps)

                if gap_count > sla.max_allowed_gaps_per_day:
                    sla_breaches.append({
                        "type": "continuity",
                        "severity": "high" if gap_count > sla.max_allowed_gaps_per_day * 2 else "medium",
                        "details": {
                            "gap_count": int(gap_count),
                            "max_allowed_gaps": int(sla.max_allowed_gaps_per_day),
                            "largest_gap": float(gaps.max().total_seconds()) if not gaps.empty else 0
                        }
                    })

            elif data_type == "tick":
                # For tick data, check for periods of inactivity
                large_gaps = time_diffs[time_diffs.dt.total_seconds() > sla.max_latency_seconds]
                gap_count = len(large_gaps)

                if gap_count > 0:
                    sla_breaches.append({
                        "type": "tick_frequency",
                        "severity": "high" if gap_count > 10 else "medium",
                        "details": {
                            "gap_count": int(gap_count),
                            "max_gap_seconds": float(large_gaps.max().total_seconds()) if not large_gaps.empty else 0,
                            "avg_gap_seconds": float(large_gaps.mean().total_seconds()) if not large_gaps.empty else 0
                        }
                    })

        return sla_breaches

    def get_data_quality_metrics(
        self,
        instrument: Optional[str] = None,
        data_type: str = "ohlcv",
        lookback_hours: int = 24
    ) -> List[DataQualityMetrics]:
        """
        Get data quality metrics.

        Args:
            instrument: Instrument to get metrics for (optional)
            data_type: Type of data (ohlcv, tick)
            lookback_hours: Hours to look back

        Returns:
            List of data quality metrics
        """
        # Get metrics from cache
        metrics = []
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=lookback_hours)

        # Filter metrics by instrument and data type
        for cache_key, results in self.validation_cache.items():
            # Check if cache is expired
            if cache_key in self.cache_expiry and self.cache_expiry[cache_key] < now:
                # Remove expired cache
                del self.validation_cache[cache_key]
                del self.cache_expiry[cache_key]
                continue

            # Check if cache matches instrument and data type
            key_parts = cache_key.split('_')
            if len(key_parts) < 2:
                continue

            cache_data_type = key_parts[0]
            cache_instrument = '_'.join(key_parts[1:])

            if data_type != cache_data_type:
                continue

            if instrument is not None and instrument != cache_instrument:
                continue

            # Calculate metrics
            total_validations = len(results)
            error_count = sum(1 for r in results if r.severity == ValidationSeverity.ERROR and not r.is_valid)
            warning_count = sum(1 for r in results if r.severity == ValidationSeverity.WARNING and not r.is_valid)

            # Create metrics object
            metrics.append(DataQualityMetrics(
                instrument=cache_instrument,
                data_type=cache_data_type,
                timestamp=now,
                completeness=100.0,  # Default values
                timeliness=100.0,
                accuracy=100.0 * (total_validations - error_count) / total_validations if total_validations > 0 else 100.0,
                consistency=100.0,
                gaps_per_day=0,
                spikes_per_day=0,
                latency_seconds=0.0,
                validation_count=total_validations,
                error_count=error_count,
                warning_count=warning_count
            ))

        return metrics

    def get_data_quality_sla(
        self,
        instrument_type: str,
        data_type: str
    ) -> DataQualitySLA:
        """
        Get data quality SLA.

        Args:
            instrument_type: Type of instrument (forex, crypto, stocks)
            data_type: Type of data (ohlcv, tick)

        Returns:
            Data quality SLA
        """
        # Try to get SLA for instrument type
        if instrument_type in self.slas:
            return self.slas[instrument_type]

        # Try to get SLA for data type
        if data_type in self.slas:
            return self.slas[data_type]

        # Fall back to default SLA
        return self.slas["default"]

    def set_data_quality_sla(
        self,
        sla: DataQualitySLA,
        key: str
    ) -> None:
        """
        Set data quality SLA.

        Args:
            sla: SLA to set
            key: SLA key (default, ohlcv, tick, forex, crypto, stocks, etc.)
        """
        self.slas[key] = sla

    def _generate_recommendations(
        self,
        validation_result: ValidationResult,
        anomalies: List[Dict[str, Any]],
        sla_breaches: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate recommendations based on validation results, anomalies, and SLA breaches.

        Args:
            validation_result: Validation result
            anomalies: Detected anomalies
            sla_breaches: SLA breaches

        Returns:
            List of recommendations
        """
        recommendations = []

        # Add recommendations based on validation result
        if not validation_result.is_valid:
            recommendations.append(f"Address validation issues: {validation_result.message}")

        # Add recommendations based on anomalies
        if anomalies:
            anomaly_types = set(a["type"] for a in anomalies)

            if "price_gap" in anomaly_types:
                recommendations.append("Investigate price gaps between trading sessions")

            if "high_spike" in anomaly_types or "low_spike" in anomaly_types:
                recommendations.append("Review price spikes for potential data quality issues")

            if "spread_anomaly" in anomaly_types:
                recommendations.append("Investigate unusual bid-ask spread patterns")

        # Add recommendations based on SLA breaches
        for breach in sla_breaches:
            if breach["type"] == "completeness":
                recommendations.append(f"Improve data completeness for field '{breach['field']}'")

            elif breach["type"] == "timeliness":
                recommendations.append("Reduce data latency to meet SLA requirements")

            elif breach["type"] == "continuity":
                recommendations.append("Address data gaps to ensure continuous data flow")

            elif breach["type"] == "tick_frequency":
                recommendations.append("Improve tick data frequency to meet SLA requirements")

        return recommendations

    def get_data_quality_metrics(
        self,
        instrument: Optional[str] = None,
        data_type: str = "ohlcv",
        lookback_hours: int = 24
    ) -> List[DataQualityMetrics]:
        """
        Get data quality metrics for a specific instrument and data type.

        Args:
            instrument: Instrument to get metrics for (optional)
            data_type: Type of data (ohlcv, tick)
            lookback_hours: Hours to look back

        Returns:
            List of data quality metrics
        """
        # Filter cache keys based on data type and instrument
        cache_keys = []

        for key in self.validation_cache.keys():
            key_parts = key.split("_")
            key_data_type = key_parts[0]

            if key_data_type == data_type:
                if instrument is None or (len(key_parts) > 1 and key_parts[1] == instrument):
                    cache_keys.append(key)

        # Get metrics from cache
        metrics = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        for key in cache_keys:
            # Get validation results from cache
            validation_results = self.validation_cache.get(key, [])

            # Calculate metrics
            total_records = 0
            valid_records = 0
            invalid_records = 0
            validation_times = []

            for result in validation_results:
                # Skip results older than cutoff time
                if hasattr(result, "timestamp") and result.timestamp < cutoff_time:
                    continue

                # Update counts
                if hasattr(result, "record_count"):
                    count = result.record_count
                else:
                    count = 1

                total_records += count
                if result.is_valid:
                    valid_records += count
                else:
                    invalid_records += count

                # Update validation times
                if hasattr(result, "validation_time_ms"):
                    validation_times.append(result.validation_time_ms)

            # Create metrics
            if total_records > 0:
                metrics.append(DataQualityMetrics(
                    total_records=total_records,
                    valid_records=valid_records,
                    invalid_records=invalid_records,
                    validation_time_ms=sum(validation_times) / len(validation_times) if validation_times else 0.0,
                    error_rate=invalid_records / total_records * 100 if total_records > 0 else 0.0,
                    timestamp=datetime.utcnow()
                ))

        return metrics

    def get_data_quality_sla(self, instrument_type: str = "forex", data_type: str = "ohlcv") -> DataQualitySLA:
        """
        Get the SLA for a specific instrument type and data type.

        Args:
            instrument_type: Type of instrument (forex, crypto, stocks)
            data_type: Type of data (ohlcv, tick)

        Returns:
            Data quality SLA
        """
        # Try to get SLA for specific combination
        key = f"{instrument_type}_{data_type}"
        if key in self.slas:
            return self.slas[key]

        # Try to get SLA for instrument type
        if instrument_type in self.slas:
            return self.slas[instrument_type]

        # Try to get SLA for data type
        if data_type in self.slas:
            return self.slas[data_type]

        # Fall back to default SLA
        return self.slas["default"]

    def set_data_quality_sla(self, sla: DataQualitySLA, key: str = "default") -> None:
        """
        Set the SLA for a specific key.

        Args:
            sla: Data quality SLA
            key: SLA key (default, ohlcv, tick, forex, crypto, stocks, etc.)
        """
        self.slas[key] = sla
        logger.info(f"Updated SLA for {key}")

    def cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        now = datetime.utcnow()

        # Find expired keys
        expired_keys = [key for key, expiry in self.cache_expiry.items() if expiry < now]

        # Remove expired entries
        for key in expired_keys:
            if key in self.validation_cache:
                del self.validation_cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")