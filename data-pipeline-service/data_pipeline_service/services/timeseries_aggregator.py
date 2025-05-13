"""
TimeseriesAggregator class for resampling and aggregating OHLCV data.
"""
import pandas as pd
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from common_lib.schemas import OHLCVData, TimeframeEnum, AggregationMethodEnum


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TimeseriesAggregator:
    """
    Service for aggregating timeseries data between different timeframes.
    """

    def _convert_timeframe_to_pandas_freq(self, timeframe: str) ->str:
        """Convert trading timeframe string to pandas frequency string."""
        match = re.match('(\\d+)([mhdw])', timeframe)
        if not match:
            raise ValueError(f'Invalid timeframe format: {timeframe}')
        value, unit = match.groups()
        if unit == 'm':
            return f'{value}min'
        elif unit == 'h':
            return f'{value}H'
        elif unit == 'd':
            return f'{value}D'
        elif unit == 'w':
            return f'{value}W'
        else:
            raise ValueError(f'Unsupported timeframe unit: {unit}')

    @with_exception_handling
    def aggregate(self, data: List[OHLCVData], source_timeframe:
        TimeframeEnum, target_timeframe: TimeframeEnum, method:
        AggregationMethodEnum=AggregationMethodEnum.OHLCV) ->List[OHLCVData]:
        """Aggregate OHLCV data from source timeframe to target timeframe."""
        if not data:
            return []
        try:
            df = pd.DataFrame([d.model_dump() for d in data])
        except AttributeError:
            df = pd.DataFrame([d.dict() for d in data])
        df.set_index('timestamp', inplace=True)
        if method == AggregationMethodEnum.OHLCV:
            result_df = self._standard_ohlcv_aggregation(df,
                target_timeframe.value)
        elif method == AggregationMethodEnum.VWAP:
            result_df = self._vwap_aggregation(df, target_timeframe.value)
        elif method == AggregationMethodEnum.TWAP:
            result_df = self._twap_aggregation(df, target_timeframe.value)
        else:
            raise ValueError(f'Unsupported aggregation method: {method}')
        result = []
        for timestamp, row in result_df.iterrows():
            result.append(OHLCVData(timestamp=timestamp, open=row['open'],
                high=row['high'], low=row['low'], close=row['close'],
                volume=row['volume']))
        return result

    def _standard_ohlcv_aggregation(self, df: pd.DataFrame,
        target_timeframe: str) ->pd.DataFrame:
        """Standard OHLCV aggregation."""
        freq = self._convert_timeframe_to_pandas_freq(target_timeframe)
        resampled = df.resample(freq).agg({'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last', 'volume': 'sum'})
        return resampled.dropna()

    def _vwap_aggregation(self, df: pd.DataFrame, target_timeframe: str
        ) ->pd.DataFrame:
        """Volume-weighted average price aggregation."""
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_volume'] = df['typical_price'] * df['volume']
        freq = self._convert_timeframe_to_pandas_freq(target_timeframe)
        resampled = df.resample(freq).agg({'open': 'first', 'high': 'max',
            'low': 'min', 'price_volume': 'sum', 'volume': 'sum'})
        resampled['close'] = resampled['price_volume'] / resampled['volume'
            ].replace(0, float('nan'))
        return resampled.drop(['price_volume'], axis=1).dropna()

    def _twap_aggregation(self, df: pd.DataFrame, target_timeframe: str
        ) ->pd.DataFrame:
        """Time-weighted average price aggregation."""
        df['twap'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        freq = self._convert_timeframe_to_pandas_freq(target_timeframe)
        resampled = df.resample(freq).agg({'open': 'first', 'high': 'max',
            'low': 'min', 'twap': 'mean', 'volume': 'sum'})
        resampled['close'] = resampled['twap']
        return resampled.drop(['twap'], axis=1).dropna()
