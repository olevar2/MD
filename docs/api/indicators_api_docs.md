"""
Technical Indicators API Documentation

This documentation provides a comprehensive guide to the technical indicators
available in the forex_trading_platform. It includes detailed descriptions,
parameters, return types, and usage examples for each indicator.
"""

# API Documentation for Technical Indicators

## Table of Contents

1. [Introduction](#introduction)
2. [Moving Averages](#moving-averages)
3. [Oscillators](#oscillators)
4. [Volume Indicators](#volume-indicators)
5. [Volatility Indicators](#volatility-indicators)
6. [Advanced Indicators](#advanced-indicators)
7. [Multi-Timeframe Indicators](#multi-timeframe-indicators)

## Introduction

The technical indicators in this platform are organized into different categories based on their functionality and application in trading analysis. Each indicator follows a consistent interface design and can be used individually or combined for more complex analysis.

### Base Indicator Interface

All indicators inherit from a base interface that provides common functionality:

```python
from feature_store_service.indicators.base_indicator import BaseIndicator

# Example of accessing an indicator through the common interface
indicator = BaseIndicator.get_indicator("SMA")
result = indicator.calculate(data, window=20)
```

## Moving Averages

Moving averages smooth price data to create a single flowing line, making it easier to identify the direction of the trend.

### Simple Moving Average (SMA)

Calculates the arithmetic mean of a given set of prices over a specified number of periods.

**Function Signature:**
```python
def simple_moving_average(data, window=20, price_column='close'):
    """
    Calculate Simple Moving Average.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    window : int
        Number of periods to calculate the SMA
    price_column : str
        Column name to use for price data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with SMA values
    """
```

**Example Usage:**
```python
import pandas as pd
from feature_store_service.indicators.moving_averages import simple_moving_average

# Load your price data
data = pd.read_csv('price_data.csv')

# Calculate SMA with 20-period window
sma_20 = simple_moving_average(data, window=20)

# Calculate SMA with 50-period window on high prices
sma_50_high = simple_moving_average(data, window=50, price_column='high')
```

**Default Values:**
- `window`: 20
- `price_column`: 'close'

### Exponential Moving Average (EMA)

Gives more weight to recent prices, making it more responsive to new information.

**Function Signature:**
```python
def exponential_moving_average(data, window=20, price_column='close', smoothing=2):
    """
    Calculate Exponential Moving Average.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    window : int
        Number of periods to calculate the EMA
    price_column : str
        Column name to use for price data
    smoothing : int
        Smoothing factor
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with EMA values
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.moving_averages import exponential_moving_average

# Calculate EMA with 20-period window
ema_20 = exponential_moving_average(data, window=20)
```

**Default Values:**
- `window`: 20
- `price_column`: 'close'
- `smoothing`: 2

### Weighted Moving Average (WMA)

Assigns a higher weighting to more recent data points and lower weighting to data points in the distant past.

**Function Signature:**
```python
def weighted_moving_average(data, window=20, price_column='close'):
    """
    Calculate Weighted Moving Average.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    window : int
        Number of periods to calculate the WMA
    price_column : str
        Column name to use for price data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with WMA values
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.moving_averages import weighted_moving_average

# Calculate WMA with 20-period window
wma_20 = weighted_moving_average(data, window=20)
```

**Default Values:**
- `window`: 20
- `price_column`: 'close'

## Oscillators

Oscillators are indicators that fluctuate above and below a centerline, or between set levels, helping to identify overbought and oversold conditions.

### Relative Strength Index (RSI)

Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.

**Function Signature:**
```python
def relative_strength_index(data, window=14, price_column='close'):
    """
    Calculate Relative Strength Index.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    window : int
        Number of periods to calculate the RSI
    price_column : str
        Column name to use for price data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with RSI values
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.oscillators import relative_strength_index

# Calculate RSI with 14-period window
rsi = relative_strength_index(data, window=14)
```

**Default Values:**
- `window`: 14
- `price_column`: 'close'

### Moving Average Convergence Divergence (MACD)

Shows the relationship between two moving averages of a security's price.

**Function Signature:**
```python
def macd(data, fast_period=12, slow_period=26, signal_period=9, price_column='close'):
    """
    Calculate Moving Average Convergence Divergence.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    fast_period : int
        Number of periods for fast EMA
    slow_period : int
        Number of periods for slow EMA
    signal_period : int
        Number of periods for signal line
    price_column : str
        Column name to use for price data
        
    Returns:
    --------
    tuple
        (MACD line, Signal line, Histogram)
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.oscillators import macd

# Calculate MACD with default parameters
macd_line, signal_line, histogram = macd(data)
```

**Default Values:**
- `fast_period`: 12
- `slow_period`: 26
- `signal_period`: 9
- `price_column`: 'close'

### Stochastic Oscillator

Compares a security's closing price to its price range over a given time period.

**Function Signature:**
```python
def stochastic(data, k_window=14, d_window=3, d_method='sma'):
    """
    Calculate Stochastic Oscillator.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data (requires 'high', 'low', 'close')
    k_window : int
        Number of periods for %K line
    d_window : int
        Number of periods for %D line
    d_method : str
        Method for calculating %D ('sma', 'ema')
        
    Returns:
    --------
    tuple
        (%K line, %D line)
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.oscillators import stochastic

# Calculate Stochastic Oscillator
k_line, d_line = stochastic(data, k_window=14, d_window=3)
```

**Default Values:**
- `k_window`: 14
- `d_window`: 3
- `d_method`: 'sma'

## Volume Indicators

Volume indicators use volume data to provide insights into the strength of price movements.

### On-Balance Volume (OBV)

Relates volume to price change to detect momentum.

**Function Signature:**
```python
def on_balance_volume(data, price_column='close', volume_column='volume'):
    """
    Calculate On-Balance Volume.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    price_column : str
        Column name to use for price data
    volume_column : str
        Column name to use for volume data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with OBV values
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.volume import on_balance_volume

# Calculate OBV
obv = on_balance_volume(data)
```

**Default Values:**
- `price_column`: 'close'
- `volume_column`: 'volume'

### Volume Profile

Shows the trading activity at specific price levels.

**Function Signature:**
```python
def volume_profile(data, price_column='close', volume_column='volume', bins=10):
    """
    Calculate Volume Profile.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    price_column : str
        Column name to use for price data
    volume_column : str
        Column name to use for volume data
    bins : int
        Number of price bins to create
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with volume profile values
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.volume import volume_profile

# Calculate Volume Profile with 10 bins
profile = volume_profile(data, bins=10)
```

**Default Values:**
- `price_column`: 'close'
- `volume_column`: 'volume'
- `bins`: 10

## Volatility Indicators

Volatility indicators measure the rate and magnitude of price changes.

### Average True Range (ATR)

Measures market volatility by calculating the average range of price bars.

**Function Signature:**
```python
def average_true_range(data, window=14):
    """
    Calculate Average True Range.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data (requires 'high', 'low', 'close')
    window : int
        Number of periods to calculate ATR
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with ATR values
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.volatility import average_true_range

# Calculate ATR with 14-period window
atr = average_true_range(data, window=14)
```

**Default Values:**
- `window`: 14

### Bollinger Bands

Consists of a middle band (SMA) with upper and lower bands that represent volatility.

**Function Signature:**
```python
def bollinger_bands(data, window=20, num_std=2, price_column='close'):
    """
    Calculate Bollinger Bands.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    window : int
        Number of periods for moving average
    num_std : float
        Number of standard deviations for bands
    price_column : str
        Column name to use for price data
        
    Returns:
    --------
    tuple
        (Upper band, Middle band, Lower band)
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.volatility import bollinger_bands

# Calculate Bollinger Bands
upper, middle, lower = bollinger_bands(data, window=20, num_std=2)
```

**Default Values:**
- `window`: 20
- `num_std`: 2
- `price_column`: 'close'

## Advanced Indicators

Advanced indicators combine multiple methodologies for more complex analysis.

### Ichimoku Cloud

A collection of technical indicators that show support and resistance, momentum, and trend direction.

**Function Signature:**
```python
def ichimoku_cloud(data, conversion_line_period=9, base_line_period=26, 
                  leading_span_b_period=52, displacement=26):
    """
    Calculate Ichimoku Cloud.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data (requires 'high', 'low', 'close')
    conversion_line_period : int
        Period for Tenkan-sen (Conversion Line)
    base_line_period : int
        Period for Kijun-sen (Base Line)
    leading_span_b_period : int
        Period for Senkou Span B
    displacement : int
        Period for displacement
        
    Returns:
    --------
    tuple
        (Conversion Line, Base Line, Leading Span A, Leading Span B, Lagging Span)
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.advanced_price_indicators import ichimoku_cloud

# Calculate Ichimoku Cloud
conversion, base, span_a, span_b, lagging = ichimoku_cloud(data)
```

**Default Values:**
- `conversion_line_period`: 9
- `base_line_period`: 26
- `leading_span_b_period`: 52
- `displacement`: 26

### Kaufman's Adaptive Moving Average (KAMA)

Adapts to price fluctuations, providing a better signal by removing market noise.

**Function Signature:**
```python
def kama(data, er_window=10, fast_window=2, slow_window=30, price_column='close'):
    """
    Calculate Kaufman's Adaptive Moving Average.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    er_window : int
        Window for efficiency ratio
    fast_window : int
        Fast EMA window
    slow_window : int
        Slow EMA window
    price_column : str
        Column name to use for price data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with KAMA values
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.advanced_moving_averages import kama

# Calculate KAMA
kama_values = kama(data, er_window=10, fast_window=2, slow_window=30)
```

**Default Values:**
- `er_window`: 10
- `fast_window`: 2
- `slow_window`: 30
- `price_column`: 'close'

## Multi-Timeframe Indicators

Indicators that analyze data across multiple timeframes.

### Multi-Timeframe RSI

Calculates RSI across different timeframes for a broader market perspective.

**Function Signature:**
```python
def multi_timeframe_rsi(data, timeframes=[5, 14, 30], price_column='close'):
    """
    Calculate RSI across multiple timeframes.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    timeframes : list
        List of timeframes (periods) to calculate RSI for
    price_column : str
        Column name to use for price data
        
    Returns:
    --------
    dict
        Dictionary of DataFrames with RSI values for each timeframe
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.multi_timeframe import multi_timeframe_rsi

# Calculate RSI for 5, 14, and 30 periods
multi_rsi = multi_timeframe_rsi(data, timeframes=[5, 14, 30])

# Access specific timeframe RSI
rsi_5 = multi_rsi[5]
rsi_14 = multi_rsi[14]
```

**Default Values:**
- `timeframes`: [5, 14, 30]
- `price_column`: 'close'

### Multi-Timeframe Moving Average

Calculates moving averages across different timeframes.

**Function Signature:**
```python
def multi_timeframe_ma(data, ma_type='sma', timeframes=[10, 20, 50, 200], price_column='close'):
    """
    Calculate moving averages across multiple timeframes.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    ma_type : str
        Type of moving average ('sma', 'ema', 'wma')
    timeframes : list
        List of timeframes (periods) to calculate MAs for
    price_column : str
        Column name to use for price data
        
    Returns:
    --------
    dict
        Dictionary of DataFrames with MA values for each timeframe
    """
```

**Example Usage:**
```python
from feature_store_service.indicators.multi_timeframe import multi_timeframe_ma

# Calculate SMAs for 10, 20, 50, and 200 periods
multi_sma = multi_timeframe_ma(data, ma_type='sma')

# Calculate EMAs for 10, 20, and 50 periods
multi_ema = multi_timeframe_ma(data, ma_type='ema', timeframes=[10, 20, 50])
```

**Default Values:**
- `ma_type`: 'sma'
- `timeframes`: [10, 20, 50, 200]
- `price_column`: 'close'
