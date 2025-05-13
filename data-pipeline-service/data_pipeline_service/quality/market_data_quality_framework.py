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
from data_pipeline_service.validation.validation_engine import ValidationResult, ValidationSeverity, DataValidationEngine, CompositeValidator, SchemaValidator, NullValidator, OutlierValidator, TimeSeriesContinuityValidator
from data_pipeline_service.validation.ohlcv_validators import CandlestickPatternValidator, GapDetectionValidator, VolumeChangeValidator
from data_pipeline_service.validation.tick_validators import QuoteSequenceValidator, TickFrequencyValidator, TickVolumeConsistencyValidator
logger = logging.getLogger(__name__)

class DataQualityLevel(str, Enum):
    """Data quality validation levels"""
    BASIC = 'basic'
    STANDARD = 'standard'
    COMPREHENSIVE = 'comprehensive'
    STRICT = 'strict'

class DataQualitySLA(BaseModel):
    """Service Level Agreement for data quality"""
    completeness: float = Field(99.5, description='Percentage of required fields that must be present')
    timeliness: float = Field(99.0, description='Percentage of data that must arrive within SLA timeframe')
    accuracy: float = Field(99.9, description='Percentage of data that must pass validation checks')
    consistency: float = Field(99.5, description='Percentage of data that must be consistent across sources')
    max_allowed_gaps_per_day: int = Field(0, description='Maximum number of gaps allowed per day')
    max_allowed_spikes_per_day: int = Field(0, description='Maximum number of price spikes allowed per day')
    max_latency_seconds: float = Field(1.0, description='Maximum latency for real-time data in seconds')

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

    def __init__(self, metrics_collector: Optional[MetricsCollector]=None, alert_manager: Optional[AlertManager]=None):
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
        self._initialize_default_slas()
        self._initialize_validation_rules()
        self.validation_cache: Dict[str, List[ValidationResult]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(hours=24)
        logger.info('Market Data Quality Framework initialized')

    def _initialize_default_slas(self) -> None:
        """Initialize default SLAs for different data types and instruments"""
        self.slas['default'] = DataQualitySLA()
        self.slas['ohlcv'] = DataQualitySLA(completeness=99.9, timeliness=99.5, accuracy=99.95, consistency=99.8, max_allowed_gaps_per_day=0, max_allowed_spikes_per_day=0, max_latency_seconds=5.0)
        self.slas['tick'] = DataQualitySLA(completeness=99.8, timeliness=99.9, accuracy=99.9, consistency=99.7, max_allowed_gaps_per_day=0, max_allowed_spikes_per_day=0, max_latency_seconds=0.5)
        self.slas['forex'] = DataQualitySLA(completeness=99.95, timeliness=99.9, accuracy=99.99, consistency=99.9, max_allowed_gaps_per_day=0, max_allowed_spikes_per_day=0, max_latency_seconds=0.2)
        self.slas['crypto'] = DataQualitySLA(completeness=99.8, timeliness=99.5, accuracy=99.9, consistency=99.5, max_allowed_gaps_per_day=2, max_allowed_spikes_per_day=5, max_latency_seconds=1.0)
        self.slas['stocks'] = DataQualitySLA(completeness=99.9, timeliness=99.7, accuracy=99.95, consistency=99.8, max_allowed_gaps_per_day=0, max_allowed_spikes_per_day=2, max_latency_seconds=2.0)

    def _initialize_validation_rules(self) -> None:
        """Initialize validation rules for different data types"""
        self._register_ohlcv_validators()
        self._register_tick_validators()
        self._register_alternative_data_validators()

    def _register_ohlcv_validators(self) -> None:
        """Register validators for OHLCV data at different quality levels"""
        basic_ohlcv = CompositeValidator('basic-ohlcv-validator')
        schema_validator = SchemaValidator(required_columns=['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume'], dtypes={'timestamp': 'datetime64[ns]', 'instrument': 'object', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
        basic_ohlcv.add_validator(schema_validator)
        null_validator = NullValidator(non_nullable_columns=['timestamp', 'instrument', 'open', 'high', 'low', 'close'], max_null_percentage={'volume': 5.0}, severity=ValidationSeverity.ERROR)
        basic_ohlcv.add_validator(null_validator)
        self.validation_engine.register_validator('ohlcv_basic', basic_ohlcv)
        standard_ohlcv = CompositeValidator('standard-ohlcv-validator')
        standard_ohlcv.add_validator(schema_validator)
        standard_ohlcv.add_validator(null_validator)
        continuity_validator = TimeSeriesContinuityValidator(timestamp_column='timestamp', max_gap=pd.Timedelta(hours=24), severity=ValidationSeverity.WARNING)
        standard_ohlcv.add_validator(continuity_validator)
        outlier_validator = OutlierValidator(columns_to_check=['open', 'high', 'low', 'close', 'volume'], method='zscore', threshold=4.0, severity=ValidationSeverity.WARNING)
        standard_ohlcv.add_validator(outlier_validator)
        self.validation_engine.register_validator('ohlcv_standard', standard_ohlcv)
        comprehensive_ohlcv = CompositeValidator('comprehensive-ohlcv-validator')
        comprehensive_ohlcv.add_validator(schema_validator)
        comprehensive_ohlcv.add_validator(null_validator)
        comprehensive_ohlcv.add_validator(continuity_validator)
        comprehensive_ohlcv.add_validator(outlier_validator)
        gap_validator = GapDetectionValidator(max_gap_percentage=3.0, severity=ValidationSeverity.WARNING)
        comprehensive_ohlcv.add_validator(gap_validator)
        volume_validator = VolumeChangeValidator(max_volume_spike_factor=10.0, min_volume_drop_factor=0.1, severity=ValidationSeverity.WARNING)
        comprehensive_ohlcv.add_validator(volume_validator)
        pattern_validator = CandlestickPatternValidator(severity=ValidationSeverity.INFO)
        comprehensive_ohlcv.add_validator(pattern_validator)
        self.validation_engine.register_validator('ohlcv_comprehensive', comprehensive_ohlcv)
        strict_ohlcv = CompositeValidator('strict-ohlcv-validator')
        strict_schema_validator = SchemaValidator(required_columns=['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume'], dtypes={'timestamp': 'datetime64[ns]', 'instrument': 'object', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
        strict_ohlcv.add_validator(strict_schema_validator)
        strict_null_validator = NullValidator(non_nullable_columns=['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume'], max_null_percentage={}, severity=ValidationSeverity.ERROR)
        strict_ohlcv.add_validator(strict_null_validator)
        strict_continuity_validator = TimeSeriesContinuityValidator(timestamp_column='timestamp', max_gap=pd.Timedelta(minutes=30), severity=ValidationSeverity.ERROR)
        strict_ohlcv.add_validator(strict_continuity_validator)
        strict_outlier_validator = OutlierValidator(columns_to_check=['open', 'high', 'low', 'close', 'volume'], method='zscore', threshold=3.0, severity=ValidationSeverity.ERROR)
        strict_ohlcv.add_validator(strict_outlier_validator)
        strict_gap_validator = GapDetectionValidator(max_gap_percentage=1.0, severity=ValidationSeverity.ERROR)
        strict_ohlcv.add_validator(strict_gap_validator)
        strict_volume_validator = VolumeChangeValidator(max_volume_spike_factor=5.0, min_volume_drop_factor=0.2, severity=ValidationSeverity.ERROR)
        strict_ohlcv.add_validator(strict_volume_validator)
        self.validation_engine.register_validator('ohlcv_strict', strict_ohlcv)

    def _register_tick_validators(self) -> None:
        """Register validators for tick data at different quality levels"""
        basic_tick = CompositeValidator('basic-tick-validator')
        schema_validator = SchemaValidator(required_columns=['timestamp', 'instrument', 'bid', 'ask'], dtypes={'timestamp': 'datetime64[ns]', 'instrument': 'object', 'bid': 'float64', 'ask': 'float64'})
        basic_tick.add_validator(schema_validator)
        null_validator = NullValidator(non_nullable_columns=['timestamp', 'instrument', 'bid', 'ask'], max_null_percentage={}, severity=ValidationSeverity.ERROR)
        basic_tick.add_validator(null_validator)
        self.validation_engine.register_validator('tick_basic', basic_tick)
        standard_tick = CompositeValidator('standard-tick-validator')
        standard_tick.add_validator(schema_validator)
        standard_tick.add_validator(null_validator)
        quote_validator = QuoteSequenceValidator(bid_column='bid', ask_column='ask', severity=ValidationSeverity.WARNING)
        standard_tick.add_validator(quote_validator)
        outlier_validator = OutlierValidator(columns_to_check=['bid', 'ask'], method='zscore', threshold=4.0, severity=ValidationSeverity.WARNING)
        standard_tick.add_validator(outlier_validator)
        self.validation_engine.register_validator('tick_standard', standard_tick)
        comprehensive_tick = CompositeValidator('comprehensive-tick-validator')
        comprehensive_tick.add_validator(schema_validator)
        comprehensive_tick.add_validator(null_validator)
        comprehensive_tick.add_validator(quote_validator)
        comprehensive_tick.add_validator(outlier_validator)
        frequency_validator = TickFrequencyValidator(max_gap_seconds=5, severity=ValidationSeverity.WARNING)
        comprehensive_tick.add_validator(frequency_validator)
        volume_validator = TickVolumeConsistencyValidator(bid_volume_column='bid_volume', ask_volume_column='ask_volume', severity=ValidationSeverity.WARNING)
        comprehensive_tick.add_validator(volume_validator)
        self.validation_engine.register_validator('tick_comprehensive', comprehensive_tick)
        strict_tick = CompositeValidator('strict-tick-validator')
        strict_tick.add_validator(schema_validator)
        strict_tick.add_validator(null_validator)
        strict_quote_validator = QuoteSequenceValidator(bid_column='bid', ask_column='ask', severity=ValidationSeverity.ERROR)
        strict_tick.add_validator(strict_quote_validator)
        strict_outlier_validator = OutlierValidator(columns_to_check=['bid', 'ask'], method='zscore', threshold=3.0, severity=ValidationSeverity.ERROR)
        strict_tick.add_validator(strict_outlier_validator)
        strict_frequency_validator = TickFrequencyValidator(max_gap_seconds=1, severity=ValidationSeverity.ERROR)
        strict_tick.add_validator(strict_frequency_validator)
        strict_volume_validator = TickVolumeConsistencyValidator(bid_volume_column='bid_volume', ask_volume_column='ask_volume', severity=ValidationSeverity.ERROR)
        strict_tick.add_validator(strict_volume_validator)
        self.validation_engine.register_validator('tick_strict', strict_tick)

    def _register_alternative_data_validators(self) -> None:
        """Register validators for alternative data sources"""
        news_validator = CompositeValidator('news-validator')
        news_schema_validator = SchemaValidator(required_columns=['timestamp', 'source', 'title', 'content'], dtypes={'timestamp': 'datetime64[ns]', 'source': 'object', 'title': 'object', 'content': 'object'})
        news_validator.add_validator(news_schema_validator)
        news_null_validator = NullValidator(non_nullable_columns=['timestamp', 'source', 'title'], max_null_percentage={'content': 5.0}, severity=ValidationSeverity.WARNING)
        news_validator.add_validator(news_null_validator)
        self.validation_engine.register_validator('news', news_validator)
        economic_validator = CompositeValidator('economic-indicator-validator')
        economic_schema_validator = SchemaValidator(required_columns=['timestamp', 'country', 'indicator', 'value', 'previous_value', 'forecast_value'], dtypes={'timestamp': 'datetime64[ns]', 'country': 'object', 'indicator': 'object', 'value': 'float64', 'previous_value': 'float64', 'forecast_value': 'float64'})
        economic_validator.add_validator(economic_schema_validator)
        economic_null_validator = NullValidator(non_nullable_columns=['timestamp', 'country', 'indicator', 'value'], max_null_percentage={'previous_value': 10.0, 'forecast_value': 20.0}, severity=ValidationSeverity.WARNING)
        economic_validator.add_validator(economic_null_validator)
        self.validation_engine.register_validator('economic', economic_validator)
        sentiment_validator = CompositeValidator('social-sentiment-validator')
        sentiment_schema_validator = SchemaValidator(required_columns=['timestamp', 'instrument', 'sentiment_score', 'volume'], dtypes={'timestamp': 'datetime64[ns]', 'instrument': 'object', 'sentiment_score': 'float64', 'volume': 'float64'})
        sentiment_validator.add_validator(sentiment_schema_validator)

        class SentimentRangeValidator:
    """
    SentimentRangeValidator class.
    
    Attributes:
        Add attributes here
    """


            def validate(self, data: pd.DataFrame) -> ValidationResult:
    """
    Validate.
    
    Args:
        data: Description of data
    
    Returns:
        ValidationResult: Description of return value
    
    """

                if 'sentiment_score' not in data.columns:
                    return ValidationResult(is_valid=False, message='Missing sentiment_score column', severity=ValidationSeverity.ERROR)
                invalid_scores = data[(data['sentiment_score'] < -1) | (data['sentiment_score'] > 1)]
                if not invalid_scores.empty:
                    return ValidationResult(is_valid=False, message=f'Found {len(invalid_scores)} sentiment scores outside valid range [-1, 1]', details={'invalid_scores': invalid_scores.to_dict('records')[:10]}, severity=ValidationSeverity.ERROR)
                return ValidationResult(is_valid=True, message='Sentiment scores are in valid range')
        sentiment_validator.add_validator(SentimentRangeValidator())
        self.validation_engine.register_validator('sentiment', sentiment_validator)

    def validate_ohlcv_data(self, data: pd.DataFrame, instrument_type: str='forex', quality_level: DataQualityLevel=DataQualityLevel.STANDARD, generate_report: bool=False) -> Union[bool, DataQualityReport]:
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
        validator_key = f'ohlcv_{quality_level.value}'
        if validator_key not in self.validation_engine.validators:
            logger.error(f'No validator found for {validator_key}')
            raise ValueError(f'No validator found for {validator_key}')
        sla = self.slas.get(instrument_type, self.slas['default'])
        validation_result = self.validation_engine.validators[validator_key].validate(data)
        validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        metrics = DataQualityMetrics(total_records=len(data), valid_records=len(data) if validation_result.is_valid else 0, invalid_records=0 if validation_result.is_valid else len(data), validation_time_ms=validation_time, error_rate=0.0 if validation_result.is_valid else 100.0, timestamp=datetime.utcnow())
        if self.metrics_collector:
            self.metrics_collector.record_gauge('market_data.quality.total_records', metrics.total_records, {'data_type': 'ohlcv', 'instrument_type': instrument_type})
            self.metrics_collector.record_gauge('market_data.quality.valid_records', metrics.valid_records, {'data_type': 'ohlcv', 'instrument_type': instrument_type})
            self.metrics_collector.record_gauge('market_data.quality.invalid_records', metrics.invalid_records, {'data_type': 'ohlcv', 'instrument_type': instrument_type})
            self.metrics_collector.record_gauge('market_data.quality.validation_time_ms', metrics.validation_time_ms, {'data_type': 'ohlcv', 'instrument_type': instrument_type})
            self.metrics_collector.record_gauge('market_data.quality.error_rate', metrics.error_rate, {'data_type': 'ohlcv', 'instrument_type': instrument_type})
        if not validation_result.is_valid and self.alert_manager:
            self.alert_manager.send_alert(alert_name='market_data_quality_issue', severity='warning' if validation_result.severity == ValidationSeverity.WARNING else 'error', message=f'OHLCV data quality issue: {validation_result.message}', details={'data_type': 'ohlcv', 'instrument_type': instrument_type, 'quality_level': quality_level.value, 'validation_result': validation_result.dict() if hasattr(validation_result, 'dict') else str(validation_result)})
        if 'instrument' in data.columns and (not data.empty):
            instrument = data['instrument'].iloc[0]
            cache_key = f'ohlcv_{instrument}'
            if cache_key not in self.validation_cache:
                self.validation_cache[cache_key] = []
            self.validation_cache[cache_key].append(validation_result)
            self.cache_expiry[cache_key] = datetime.utcnow() + self.cache_ttl
            if len(self.validation_cache[cache_key]) > 100:
                self.validation_cache[cache_key] = self.validation_cache[cache_key][-100:]
        if generate_report:
            instrument = data['instrument'].iloc[0] if 'instrument' in data.columns and (not data.empty) else 'unknown'
            timeframe = None
            if 'timestamp' in data.columns and len(data) >= 2:
                timestamps = data['timestamp'].sort_values()
                diff = timestamps.diff().dropna().median()
                if diff <= pd.Timedelta(minutes=1):
                    timeframe = '1m'
                elif diff <= pd.Timedelta(minutes=5):
                    timeframe = '5m'
                elif diff <= pd.Timedelta(minutes=15):
                    timeframe = '15m'
                elif diff <= pd.Timedelta(minutes=30):
                    timeframe = '30m'
                elif diff <= pd.Timedelta(hours=1):
                    timeframe = '1h'
                elif diff <= pd.Timedelta(hours=4):
                    timeframe = '4h'
                elif diff <= pd.Timedelta(days=1):
                    timeframe = '1d'
                else:
                    timeframe = 'unknown'
            anomalies = self._detect_anomalies(data, 'ohlcv')
            sla_breaches = self._check_sla_breaches(data, sla, 'ohlcv')
            recommendations = self._generate_recommendations(validation_result, anomalies, sla_breaches)
            report = DataQualityReport(report_id=str(uuid.uuid4()), data_type='ohlcv', instrument=instrument, timeframe=timeframe, start_time=start_time, end_time=datetime.utcnow(), quality_level=quality_level, metrics=metrics, validation_results=[validation_result.dict() if hasattr(validation_result, 'dict') else {'message': str(validation_result)}], anomalies=anomalies, sla_breaches=sla_breaches, is_valid=validation_result.is_valid, recommendations=recommendations)
            return report
        return validation_result.is_valid

    def validate_tick_data(self, data: pd.DataFrame, instrument_type: str='forex', quality_level: DataQualityLevel=DataQualityLevel.STANDARD, generate_report: bool=False) -> Union[bool, DataQualityReport]:
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
        validator_key = f'tick_{quality_level.value}'
        if validator_key not in self.validation_engine.validators:
            logger.error(f'No validator found for {validator_key}')
            raise ValueError(f'No validator found for {validator_key}')
        sla = self.slas.get(instrument_type, self.slas['default'])
        validation_result = self.validation_engine.validators[validator_key].validate(data)
        validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        metrics = DataQualityMetrics(total_records=len(data), valid_records=len(data) if validation_result.is_valid else 0, invalid_records=0 if validation_result.is_valid else len(data), validation_time_ms=validation_time, error_rate=0.0 if validation_result.is_valid else 100.0, timestamp=datetime.utcnow())
        if self.metrics_collector:
            self.metrics_collector.record_gauge('market_data.quality.total_records', metrics.total_records, {'data_type': 'tick', 'instrument_type': instrument_type})
            self.metrics_collector.record_gauge('market_data.quality.valid_records', metrics.valid_records, {'data_type': 'tick', 'instrument_type': instrument_type})
            self.metrics_collector.record_gauge('market_data.quality.invalid_records', metrics.invalid_records, {'data_type': 'tick', 'instrument_type': instrument_type})
            self.metrics_collector.record_gauge('market_data.quality.validation_time_ms', metrics.validation_time_ms, {'data_type': 'tick', 'instrument_type': instrument_type})
            self.metrics_collector.record_gauge('market_data.quality.error_rate', metrics.error_rate, {'data_type': 'tick', 'instrument_type': instrument_type})
        if not validation_result.is_valid and self.alert_manager:
            self.alert_manager.send_alert(alert_name='market_data_quality_issue', severity='warning' if validation_result.severity == ValidationSeverity.WARNING else 'error', message=f'Tick data quality issue: {validation_result.message}', details={'data_type': 'tick', 'instrument_type': instrument_type, 'quality_level': quality_level.value, 'validation_result': validation_result.dict() if hasattr(validation_result, 'dict') else str(validation_result)})
        if 'instrument' in data.columns and (not data.empty):
            instrument = data['instrument'].iloc[0]
            cache_key = f'tick_{instrument}'
            if cache_key not in self.validation_cache:
                self.validation_cache[cache_key] = []
            self.validation_cache[cache_key].append(validation_result)
            self.cache_expiry[cache_key] = datetime.utcnow() + self.cache_ttl
            if len(self.validation_cache[cache_key]) > 100:
                self.validation_cache[cache_key] = self.validation_cache[cache_key][-100:]
        if generate_report:
            instrument = data['instrument'].iloc[0] if 'instrument' in data.columns and (not data.empty) else 'unknown'
            anomalies = self._detect_anomalies(data, 'tick')
            sla_breaches = self._check_sla_breaches(data, sla, 'tick')
            recommendations = self._generate_recommendations(validation_result, anomalies, sla_breaches)
            report = DataQualityReport(report_id=str(uuid.uuid4()), data_type='tick', instrument=instrument, timeframe=None, start_time=start_time, end_time=datetime.utcnow(), quality_level=quality_level, metrics=metrics, validation_results=[validation_result.dict() if hasattr(validation_result, 'dict') else {'message': str(validation_result)}], anomalies=anomalies, sla_breaches=sla_breaches, is_valid=validation_result.is_valid, recommendations=recommendations)
            return report
        return validation_result.is_valid

    def validate_alternative_data(self, data: pd.DataFrame, data_type: str, generate_report: bool=False) -> Union[bool, DataQualityReport]:
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
        if data_type not in self.validation_engine.validators:
            logger.error(f'No validator found for {data_type}')
            raise ValueError(f'No validator found for {data_type}')
        sla = self.slas.get('default')
        validation_result = self.validation_engine.validators[data_type].validate(data)
        validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        metrics = DataQualityMetrics(total_records=len(data), valid_records=len(data) if validation_result.is_valid else 0, invalid_records=0 if validation_result.is_valid else len(data), validation_time_ms=validation_time, error_rate=0.0 if validation_result.is_valid else 100.0, timestamp=datetime.utcnow())
        if self.metrics_collector:
            self.metrics_collector.record_gauge('market_data.quality.total_records', metrics.total_records, {'data_type': data_type})
            self.metrics_collector.record_gauge('market_data.quality.valid_records', metrics.valid_records, {'data_type': data_type})
            self.metrics_collector.record_gauge('market_data.quality.invalid_records', metrics.invalid_records, {'data_type': data_type})
            self.metrics_collector.record_gauge('market_data.quality.validation_time_ms', metrics.validation_time_ms, {'data_type': data_type})
            self.metrics_collector.record_gauge('market_data.quality.error_rate', metrics.error_rate, {'data_type': data_type})
        if not validation_result.is_valid and self.alert_manager:
            self.alert_manager.send_alert(alert_name='market_data_quality_issue', severity='warning' if validation_result.severity == ValidationSeverity.WARNING else 'error', message=f'{data_type.capitalize()} data quality issue: {validation_result.message}', details={'data_type': data_type, 'validation_result': validation_result.dict() if hasattr(validation_result, 'dict') else str(validation_result)})
        if generate_report:
            anomalies = self._detect_anomalies(data, data_type)
            sla_breaches = self._check_sla_breaches(data, sla, data_type)
            recommendations = self._generate_recommendations(validation_result, anomalies, sla_breaches)
            report = DataQualityReport(report_id=str(uuid.uuid4()), data_type=data_type, instrument='N/A', timeframe=None, start_time=start_time, end_time=datetime.utcnow(), quality_level=DataQualityLevel.STANDARD, metrics=metrics, validation_results=[validation_result.dict() if hasattr(validation_result, 'dict') else {'message': str(validation_result)}], anomalies=anomalies, sla_breaches=sla_breaches, is_valid=validation_result.is_valid, recommendations=recommendations)
            return report
        return validation_result.is_valid

    def detect_anomalies(self, data: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the data.

        Args:
            data: DataFrame with data
            data_type: Type of data

        Returns:
            List of detected anomalies
        """
        anomalies = []
        if data_type == 'ohlcv':
            if all((col in data.columns for col in ['open', 'close', 'high', 'low'])):
                if 'timestamp' in data.columns:
                    data = data.sort_values('timestamp')
                if len(data) > 1:
                    close_prices = data['close'].iloc[:-1].values
                    next_open_prices = data['open'].iloc[1:].values
                    gaps = np.abs(next_open_prices - close_prices) / close_prices * 100
                    significant_gaps = np.where(gaps > 0.5)[0]
                    for idx in significant_gaps:
                        anomalies.append({'type': 'price_gap', 'severity': 'medium' if gaps[idx] < 2.0 else 'high', 'details': {'index': int(idx), 'timestamp': data['timestamp'].iloc[idx + 1].isoformat() if 'timestamp' in data.columns else None, 'close_price': float(close_prices[idx]), 'next_open_price': float(next_open_prices[idx]), 'gap_percentage': float(gaps[idx])}})
                for col in ['high', 'low']:
                    rolling_median = data[col].rolling(window=20, min_periods=5).median()
                    rolling_std = data[col].rolling(window=20, min_periods=5).std()
                    z_scores = np.abs((data[col] - rolling_median) / rolling_std)
                    spikes = np.where(z_scores > 3)[0]
                    for idx in spikes:
                        if not np.isnan(z_scores[idx]):
                            anomalies.append({'type': f'{col}_spike', 'severity': 'medium' if z_scores[idx] < 5 else 'high', 'details': {'index': int(idx), 'timestamp': data['timestamp'].iloc[idx].isoformat() if 'timestamp' in data.columns else None, 'value': float(data[col].iloc[idx]), 'median': float(rolling_median.iloc[idx]), 'z_score': float(z_scores[idx])}})
        elif data_type == 'tick':
            if all((col in data.columns for col in ['bid', 'ask'])):
                data['spread'] = data['ask'] - data['bid']
                data['spread_pct'] = data['spread'] / data['bid'] * 100
                rolling_median = data['spread_pct'].rolling(window=100, min_periods=20).median()
                rolling_std = data['spread_pct'].rolling(window=100, min_periods=20).std()
                z_scores = np.abs((data['spread_pct'] - rolling_median) / rolling_std)
                anomalous_spreads = np.where(z_scores > 3)[0]
                for idx in anomalous_spreads:
                    if not np.isnan(z_scores[idx]):
                        anomalies.append({'type': 'spread_anomaly', 'severity': 'medium' if z_scores[idx] < 5 else 'high', 'details': {'index': int(idx), 'timestamp': data['timestamp'].iloc[idx].isoformat() if 'timestamp' in data.columns else None, 'bid': float(data['bid'].iloc[idx]), 'ask': float(data['ask'].iloc[idx]), 'spread_pct': float(data['spread_pct'].iloc[idx]), 'z_score': float(z_scores[idx])}})
        return anomalies[:100]

    def check_sla_breaches(self, data: pd.DataFrame, sla: DataQualitySLA, data_type: str) -> List[Dict[str, Any]]:
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
        for col in data.columns:
            null_count = data[col].isnull().sum()
            null_percentage = null_count / len(data) * 100 if len(data) > 0 else 0
            completeness = 100 - null_percentage
            if completeness < sla.completeness:
                sla_breaches.append({'type': 'completeness', 'field': col, 'severity': 'high' if sla.completeness - completeness > 5 else 'medium', 'details': {'actual': float(completeness), 'required': float(sla.completeness), 'missing_count': int(null_count)}})
        if 'timestamp' in data.columns and (not data.empty):
            now = pd.Timestamp(datetime.utcnow())
            latest_timestamp = pd.Timestamp(data['timestamp'].max())
            age_seconds = (now - latest_timestamp).total_seconds()
            if age_seconds > sla.max_latency_seconds:
                sla_breaches.append({'type': 'timeliness', 'severity': 'high' if age_seconds > sla.max_latency_seconds * 5 else 'medium', 'details': {'actual_latency': float(age_seconds), 'max_allowed_latency': float(sla.max_latency_seconds), 'latest_timestamp': latest_timestamp.isoformat()}})
        if 'timestamp' in data.columns and len(data) > 1:
            sorted_data = data.sort_values('timestamp')
            time_diffs = sorted_data['timestamp'].diff().dropna()
            if data_type == 'ohlcv':
                median_diff = time_diffs.median()
                gaps = time_diffs[time_diffs > median_diff * 1.5]
                gap_count = len(gaps)
                if gap_count > sla.max_allowed_gaps_per_day:
                    sla_breaches.append({'type': 'continuity', 'severity': 'high' if gap_count > sla.max_allowed_gaps_per_day * 2 else 'medium', 'details': {'gap_count': int(gap_count), 'max_allowed_gaps': int(sla.max_allowed_gaps_per_day), 'largest_gap': float(gaps.max().total_seconds()) if not gaps.empty else 0}})
            elif data_type == 'tick':
                large_gaps = time_diffs[time_diffs.dt.total_seconds() > sla.max_latency_seconds]
                gap_count = len(large_gaps)
                if gap_count > 0:
                    sla_breaches.append({'type': 'tick_frequency', 'severity': 'high' if gap_count > 10 else 'medium', 'details': {'gap_count': int(gap_count), 'max_gap_seconds': float(large_gaps.max().total_seconds()) if not large_gaps.empty else 0, 'avg_gap_seconds': float(large_gaps.mean().total_seconds()) if not large_gaps.empty else 0}})
        return sla_breaches

    def get_data_quality_metrics(self, instrument: Optional[str]=None, data_type: str='ohlcv', lookback_hours: int=24) -> List[DataQualityMetrics]:
        """
        Get data quality metrics.

        Args:
            instrument: Instrument to get metrics for (optional)
            data_type: Type of data (ohlcv, tick)
            lookback_hours: Hours to look back

        Returns:
            List of data quality metrics
        """
        metrics = []
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=lookback_hours)
        for cache_key, results in self.validation_cache.items():
            if cache_key in self.cache_expiry and self.cache_expiry[cache_key] < now:
                del self.validation_cache[cache_key]
                del self.cache_expiry[cache_key]
                continue
            key_parts = cache_key.split('_')
            if len(key_parts) < 2:
                continue
            cache_data_type = key_parts[0]
            cache_instrument = '_'.join(key_parts[1:])
            if data_type != cache_data_type:
                continue
            if instrument is not None and instrument != cache_instrument:
                continue
            total_validations = len(results)
            error_count = sum((1 for r in results if r.severity == ValidationSeverity.ERROR and (not r.is_valid)))
            warning_count = sum((1 for r in results if r.severity == ValidationSeverity.WARNING and (not r.is_valid)))
            metrics.append(DataQualityMetrics(instrument=cache_instrument, data_type=cache_data_type, timestamp=now, completeness=100.0, timeliness=100.0, accuracy=100.0 * (total_validations - error_count) / total_validations if total_validations > 0 else 100.0, consistency=100.0, gaps_per_day=0, spikes_per_day=0, latency_seconds=0.0, validation_count=total_validations, error_count=error_count, warning_count=warning_count))
        return metrics

    def get_data_quality_sla(self, instrument_type: str, data_type: str) -> DataQualitySLA:
        """
        Get data quality SLA.

        Args:
            instrument_type: Type of instrument (forex, crypto, stocks)
            data_type: Type of data (ohlcv, tick)

        Returns:
            Data quality SLA
        """
        if instrument_type in self.slas:
            return self.slas[instrument_type]
        if data_type in self.slas:
            return self.slas[data_type]
        return self.slas['default']

    def set_data_quality_sla(self, sla: DataQualitySLA, key: str) -> None:
        """
        Set data quality SLA.

        Args:
            sla: SLA to set
            key: SLA key (default, ohlcv, tick, forex, crypto, stocks, etc.)
        """
        self.slas[key] = sla

    def generate_recommendations(self, validation_result: ValidationResult, anomalies: List[Dict[str, Any]], sla_breaches: List[Dict[str, Any]]) -> List[str]:
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
        if not validation_result.is_valid:
            recommendations.append(f'Address validation issues: {validation_result.message}')
        if anomalies:
            anomaly_types = set((a['type'] for a in anomalies))
            if 'price_gap' in anomaly_types:
                recommendations.append('Investigate price gaps between trading sessions')
            if 'high_spike' in anomaly_types or 'low_spike' in anomaly_types:
                recommendations.append('Review price spikes for potential data quality issues')
            if 'spread_anomaly' in anomaly_types:
                recommendations.append('Investigate unusual bid-ask spread patterns')
        for breach in sla_breaches:
            if breach['type'] == 'completeness':
                recommendations.append(f"Improve data completeness for field '{breach['field']}'")
            elif breach['type'] == 'timeliness':
                recommendations.append('Reduce data latency to meet SLA requirements')
            elif breach['type'] == 'continuity':
                recommendations.append('Address data gaps to ensure continuous data flow')
            elif breach['type'] == 'tick_frequency':
                recommendations.append('Improve tick data frequency to meet SLA requirements')
        return recommendations

    def get_data_quality_metrics(self, instrument: Optional[str]=None, data_type: str='ohlcv', lookback_hours: int=24) -> List[DataQualityMetrics]:
        """
        Get data quality metrics for a specific instrument and data type.

        Args:
            instrument: Instrument to get metrics for (optional)
            data_type: Type of data (ohlcv, tick)
            lookback_hours: Hours to look back

        Returns:
            List of data quality metrics
        """
        cache_keys = []
        for key in self.validation_cache.keys():
            key_parts = key.split('_')
            key_data_type = key_parts[0]
            if key_data_type == data_type:
                if instrument is None or (len(key_parts) > 1 and key_parts[1] == instrument):
                    cache_keys.append(key)
        metrics = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        for key in cache_keys:
            validation_results = self.validation_cache.get(key, [])
            total_records = 0
            valid_records = 0
            invalid_records = 0
            validation_times = []
            for result in validation_results:
                if hasattr(result, 'timestamp') and result.timestamp < cutoff_time:
                    continue
                if hasattr(result, 'record_count'):
                    count = result.record_count
                else:
                    count = 1
                total_records += count
                if result.is_valid:
                    valid_records += count
                else:
                    invalid_records += count
                if hasattr(result, 'validation_time_ms'):
                    validation_times.append(result.validation_time_ms)
            if total_records > 0:
                metrics.append(DataQualityMetrics(total_records=total_records, valid_records=valid_records, invalid_records=invalid_records, validation_time_ms=sum(validation_times) / len(validation_times) if validation_times else 0.0, error_rate=invalid_records / total_records * 100 if total_records > 0 else 0.0, timestamp=datetime.utcnow()))
        return metrics

    def get_data_quality_sla(self, instrument_type: str='forex', data_type: str='ohlcv') -> DataQualitySLA:
        """
        Get the SLA for a specific instrument type and data type.

        Args:
            instrument_type: Type of instrument (forex, crypto, stocks)
            data_type: Type of data (ohlcv, tick)

        Returns:
            Data quality SLA
        """
        key = f'{instrument_type}_{data_type}'
        if key in self.slas:
            return self.slas[key]
        if instrument_type in self.slas:
            return self.slas[instrument_type]
        if data_type in self.slas:
            return self.slas[data_type]
        return self.slas['default']

    def set_data_quality_sla(self, sla: DataQualitySLA, key: str='default') -> None:
        """
        Set the SLA for a specific key.

        Args:
            sla: Data quality SLA
            key: SLA key (default, ohlcv, tick, forex, crypto, stocks, etc.)
        """
        self.slas[key] = sla
        logger.info(f'Updated SLA for {key}')

    def cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        now = datetime.utcnow()
        expired_keys = [key for key, expiry in self.cache_expiry.items() if expiry < now]
        for key in expired_keys:
            if key in self.validation_cache:
                del self.validation_cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]
        if expired_keys:
            logger.debug(f'Cleaned up {len(expired_keys)} expired cache entries')