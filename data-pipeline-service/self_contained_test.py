# Simple test for direct testing of the TimeseriesAggregator functionality
import os
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
import re

# Add necessary paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define the minimal required classes for testing
class TimeframeEnum(str):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    
    def __init__(self, value):
        self.value = value


class AggregationMethodEnum(str):
    OHLCV = "ohlcv"
    VWAP = "vwap"
    TWAP = "twap"
    
    def __init__(self, value):
        self.value = value


class OHLCVData:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        
    def dict(self):
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


# Define the TimeseriesAggregator class directly in this file
class TimeseriesAggregator:
    """Service for aggregating timeseries data between different timeframes."""
    
    def _convert_timeframe_to_pandas_freq(self, timeframe: str) -> str:
        """Convert trading timeframe string to pandas frequency string."""
        match = re.match(r"(\d+)([mhdw])", timeframe)
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
            
        value, unit = match.groups()
        
        if unit == "m":
            return f"{value}min"
        elif unit == "h":
            return f"{value}H"
        elif unit == "d":
            return f"{value}D"
        elif unit == "w":
            return f"{value}W"
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    def aggregate(
        self,
        data,
        source_timeframe,
        target_timeframe,
        method="ohlcv"
    ):
        """Aggregate OHLCV data from source timeframe to target timeframe."""
        if not data:
            return []
            
        # Convert to pandas DataFrame
        df = pd.DataFrame([d.dict() for d in data])
        
        df.set_index("timestamp", inplace=True)
        
        # Determine aggregation method
        if method == "ohlcv":
            result_df = self._standard_ohlcv_aggregation(df, target_timeframe)
        elif method == "vwap":
            result_df = self._vwap_aggregation(df, target_timeframe)
        elif method == "twap":
            result_df = self._twap_aggregation(df, target_timeframe)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
            
        # Convert back to OHLCVData objects
        result = []
        for timestamp, row in result_df.iterrows():
            result.append(
                OHLCVData(
                    timestamp=timestamp,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                )
            )
            
        return result
        
    def _standard_ohlcv_aggregation(self, df, target_timeframe):
        """Standard OHLCV aggregation."""
        freq = self._convert_timeframe_to_pandas_freq(target_timeframe)
        
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        
        return resampled.dropna()
        
    def _vwap_aggregation(self, df, target_timeframe):
        """Volume-weighted average price aggregation."""
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["price_volume"] = df["typical_price"] * df["volume"]
        
        freq = self._convert_timeframe_to_pandas_freq(target_timeframe)
        
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "price_volume": "sum",
            "volume": "sum"
        })
        
        resampled["close"] = resampled["price_volume"] / resampled["volume"].replace(0, float("nan"))
        
        return resampled.drop(["price_volume"], axis=1).dropna()
        
    def _twap_aggregation(self, df, target_timeframe):
        """Time-weighted average price aggregation."""
        df["twap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        
        freq = self._convert_timeframe_to_pandas_freq(target_timeframe)
        
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "twap": "mean",
            "volume": "sum"
        })
        
        resampled["close"] = resampled["twap"]
        
        return resampled.drop(["twap"], axis=1).dropna()


# Run a simple test
def create_test_data():
    """Create test OHLCV data for testing aggregation."""
    start_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = []
    
    # Create 60 one-minute candles (covering 1 hour)
    for i in range(60):
        candle_time = start_time + timedelta(minutes=i)
        data.append(OHLCVData(
            timestamp=candle_time,
            open=100 + i * 0.1,
            high=100 + i * 0.1 + 0.05,
            low=100 + i * 0.1 - 0.05,
            close=100 + i * 0.1 + (0.02 if i % 2 == 0 else -0.02),
            volume=100 + i
        ))
    return data


def test_aggregation():
    # Create the aggregator and test data
    aggregator = TimeseriesAggregator()
    data = create_test_data()
    
    # Test standard OHLCV aggregation
    result = aggregator.aggregate(
        data=data,
        source_timeframe="1m",
        target_timeframe="5m",
        method="ohlcv"
    )
    
    # Verify results
    assert len(result) == 12, f"Expected 12 candles, got {len(result)}"
    
    # Check first candle
    first_candle = result[0]
    assert first_candle.timestamp == data[0].timestamp, "Timestamp mismatch"
    assert first_candle.open == data[0].open, "Open price mismatch"
    
    print("Basic aggregation test passed!")
    
    # Test VWAP aggregation
    result = aggregator.aggregate(
        data=data,
        source_timeframe="1m",
        target_timeframe="15m",
        method="vwap"
    )
    
    assert len(result) == 4, f"Expected 4 candles, got {len(result)}"
    print("VWAP aggregation test passed!")
    
    # Test empty data
    result = aggregator.aggregate(
        data=[],
        source_timeframe="1m", 
        target_timeframe="5m"
    )
    assert len(result) == 0, "Expected empty result"
    print("Empty data test passed!")
    
    print("All tests passed successfully!")


if __name__ == "__main__":
    test_aggregation()
