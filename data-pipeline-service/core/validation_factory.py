"""
Validation Factory.

Provides factory methods to create common validator configurations for Forex data.
"""
from typing import Dict, List, Optional, Any

import pandas as pd

from core.ohlcv_validators import (
    CandlestickPatternValidator,
    GapDetectionValidator,
    VolumeChangeValidator
)
from core.tick_validators import (
    QuoteSequenceValidator,
    TickFrequencyValidator,
    TickVolumeConsistencyValidator
)
from core.validation_engine import (
    CompositeValidator, 
    DataValidationEngine, 
    ForexPriceValidator, 
    ForexSpreadValidator, 
    NullValidator, 
    OutlierValidator, 
    SchemaValidator, 
    TimeSeriesContinuityValidator, 
    ValidationSeverity,
    ValidationStrategy
)


class ValidationFactory:
    """Factory class to create common validator configurations."""
    
    @staticmethod
    def create_ohlcv_validator(
        instrument_config: Dict[str, Dict[str, Any]],
        expected_interval: Optional[pd.Timedelta] = None,
        include_advanced_validation: bool = True
    ) -> CompositeValidator:
        """
        Create a validator for OHLCV data.
        
        Args:
            instrument_config: Configuration for instruments with expected price ranges
            expected_interval: Expected interval between candles (e.g., pd.Timedelta(minutes=1))
                               If None, continuity validation is skipped
            include_advanced_validation: Whether to include advanced validation strategies
            
        Returns:
            CompositeValidator configured for OHLCV data
        """
        ohlcv_validator = CompositeValidator("ohlcv-validator")
        
        # Schema validation
        required_columns = ["timestamp", "instrument", "open", "high", "low", "close", "volume"]
        schema_validator = SchemaValidator(
            required_columns=required_columns,
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
        ohlcv_validator.add_validator(schema_validator)
        
        # Null validation
        null_validator = NullValidator(
            non_nullable_columns=["timestamp", "instrument", "open", "high", "low", "close"],
            max_null_percentage={"volume": 5.0},  # Allow up to 5% null volume
            severity=ValidationSeverity.ERROR
        )
        ohlcv_validator.add_validator(null_validator)
        
        # Price validation
        price_validator = ForexPriceValidator(
            instrument_config=instrument_config,
            check_open_range=True,
            check_high_low=True,
            check_close_range=True,
            severity=ValidationSeverity.ERROR
        )
        ohlcv_validator.add_validator(price_validator)
        
        # Outlier validation for price and volume
        outlier_validator = OutlierValidator(
            columns_to_check=["open", "high", "low", "close", "volume"],
            method="zscore",
            threshold=4.0,  # Use higher threshold for financial data
            severity=ValidationSeverity.WARNING
        )
        ohlcv_validator.add_validator(outlier_validator)
        
        # Time series continuity validation (optional)
        if expected_interval is not None:
            continuity_validator = TimeSeriesContinuityValidator(
                timestamp_column="timestamp",
                expected_interval=expected_interval,
                max_gap=expected_interval * 3,  # Allow up to 3x interval as gap
                severity=ValidationSeverity.WARNING
            )
            ohlcv_validator.add_validator(continuity_validator)
        
        # Advanced validation strategies
        if include_advanced_validation:
            # Candlestick pattern validation
            candlestick_validator = CandlestickPatternValidator(
                severity=ValidationSeverity.ERROR
            )
            ohlcv_validator.add_validator(candlestick_validator)
            
            # Volume change validation
            volume_change_validator = VolumeChangeValidator(
                max_relative_change=10.0,  # 1000% change
                window_size=20,
                severity=ValidationSeverity.WARNING
            )
            ohlcv_validator.add_validator(volume_change_validator)
            
            # Gap detection validation
            gap_validator = GapDetectionValidator(
                instrument_config=instrument_config,
                max_gap_multiplier=5.0,
                window_size=20,
                severity=ValidationSeverity.WARNING
            )
            ohlcv_validator.add_validator(gap_validator)
            
        return ohlcv_validator
    
    @staticmethod
    def create_tick_data_validator(
        instrument_config: Dict[str, Dict[str, Any]],
        include_advanced_validation: bool = True
    ) -> CompositeValidator:
        """
        Create a validator for tick data.
        
        Args:
            instrument_config: Configuration for instruments with expected price ranges
            include_advanced_validation: Whether to include advanced validation strategies
            
        Returns:
            CompositeValidator configured for tick data
        """
        tick_validator = CompositeValidator("tick-validator")
        
        # Schema validation
        required_columns = ["timestamp", "instrument", "bid", "ask", "bid_volume", "ask_volume"]
        schema_validator = SchemaValidator(
            required_columns=required_columns,
            dtypes={
                "timestamp": "datetime64[ns]",
                "instrument": "object",
                "bid": "float64",
                "ask": "float64",
                "bid_volume": "float64",
                "ask_volume": "float64"
            }
        )
        tick_validator.add_validator(schema_validator)
        
        # Null validation
        null_validator = NullValidator(
            non_nullable_columns=["timestamp", "instrument", "bid", "ask"],
            max_null_percentage={
                "bid_volume": 1.0,  # Stricter for tick data
                "ask_volume": 1.0
            },
            severity=ValidationSeverity.ERROR
        )
        tick_validator.add_validator(null_validator)
        
        # Spread validation
        spread_validator = ForexSpreadValidator(
            instrument_config=instrument_config,
            bid_column="bid",
            ask_column="ask",
            severity=ValidationSeverity.WARNING
        )
        tick_validator.add_validator(spread_validator)
        
        # Outlier validation for prices and volumes
        outlier_validator = OutlierValidator(
            columns_to_check=["bid", "ask", "bid_volume", "ask_volume"],
            method="zscore",
            threshold=4.0,
            severity=ValidationSeverity.WARNING
        )
        tick_validator.add_validator(outlier_validator)
        
        # Advanced validation strategies
        if include_advanced_validation:
            # Quote sequence validation
            quote_sequence_validator = QuoteSequenceValidator(
                bid_column="bid",
                ask_column="ask",
                severity=ValidationSeverity.WARNING
            )
            tick_validator.add_validator(quote_sequence_validator)
            
            # Configure expected tick frequencies based on instrument
            min_expected_ticks = {}
            max_expected_ticks = {}
            
            # Set reasonable defaults for major pairs
            for instrument in instrument_config:
                if instrument in ["EUR/USD", "GBP/USD", "USD/JPY"]:
                    min_expected_ticks[instrument] = 0.5  # At least 0.5 ticks/second during active hours
                    max_expected_ticks[instrument] = 100  # No more than 100 ticks/second
                else:
                    min_expected_ticks[instrument] = 0.1  # At least 0.1 ticks/second for less liquid pairs
                    max_expected_ticks[instrument] = 50   # No more than 50 ticks/second
            
            # Tick frequency validation
            tick_frequency_validator = TickFrequencyValidator(
                min_expected_ticks_per_second=min_expected_ticks,
                max_expected_ticks_per_second=max_expected_ticks,
                analysis_window=pd.Timedelta(minutes=1),
                min_analysis_windows=5,
                severity=ValidationSeverity.WARNING
            )
            tick_validator.add_validator(tick_frequency_validator)
            
            # Configure price move per volume thresholds
            move_per_volume_thresholds = {}
            for instrument, config in instrument_config.items():
                # Calculate based on decimals - lower threshold for JPY pairs
                if "JPY" in instrument:
                    move_per_volume_thresholds[instrument] = 0.05  # Allow 0.05 pip move per volume unit
                else:
                    move_per_volume_thresholds[instrument] = 0.0005  # Allow 0.0005 pip move per volume unit
            
            # Tick volume consistency validation
            volume_consistency_validator = TickVolumeConsistencyValidator(
                max_price_move_per_volume_unit=move_per_volume_thresholds,
                window_size=100,
                severity=ValidationSeverity.WARNING
            )
            tick_validator.add_validator(volume_consistency_validator)
        
        return tick_validator
    
    @staticmethod
    def create_engine_with_default_validators(
        instrument_config: Dict[str, Dict[str, Any]],
        ohlcv_interval: Optional[pd.Timedelta] = None,
        include_advanced_validation: bool = True
    ) -> DataValidationEngine:
        """
        Create a DataValidationEngine with default validators for OHLCV and tick data.
        
        Args:
            instrument_config: Configuration for instruments with expected price ranges
            ohlcv_interval: Expected interval between candles for OHLCV data
            include_advanced_validation: Whether to include advanced validation strategies
            
        Returns:
            DataValidationEngine configured with default validators
        """
        engine = DataValidationEngine()
        
        # Create and register OHLCV validator
        ohlcv_validator = ValidationFactory.create_ohlcv_validator(
            instrument_config=instrument_config,
            expected_interval=ohlcv_interval,
            include_advanced_validation=include_advanced_validation
        )
        engine.register_validator("ohlcv", ohlcv_validator)
        
        # Create and register tick data validator
        tick_validator = ValidationFactory.create_tick_data_validator(
            instrument_config=instrument_config,
            include_advanced_validation=include_advanced_validation
        )
        engine.register_validator("tick", tick_validator)
        
        return engine


# Default instrument configurations for common Forex pairs
DEFAULT_FOREX_INSTRUMENT_CONFIG = {
    "EUR/USD": {
        "decimals": 5,
        "min_price": 0.9,
        "max_price": 1.5,
        "max_spread_pips": 3
    },
    "GBP/USD": {
        "decimals": 5,
        "min_price": 1.1,
        "max_price": 1.8,
        "max_spread_pips": 5
    },
    "USD/JPY": {
        "decimals": 3,
        "min_price": 80,
        "max_price": 170,
        "max_spread_pips": 3
    },
    "USD/CHF": {
        "decimals": 5,
        "min_price": 0.8,
        "max_price": 1.2,
        "max_spread_pips": 4
    },
    "AUD/USD": {
        "decimals": 5,
        "min_price": 0.5,
        "max_price": 1.1,
        "max_spread_pips": 4
    },
    "USD/CAD": {
        "decimals": 5,
        "min_price": 1.1,
        "max_price": 1.6,
        "max_spread_pips": 5
    },
    "NZD/USD": {
        "decimals": 5,
        "min_price": 0.5,
        "max_price": 0.9,
        "max_spread_pips": 6
    }
}