# Technical Indicators Tutorial

This tutorial provides practical examples and guidance for using the technical indicators available in the forex trading platform.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started with Indicators](#getting-started-with-indicators)
3. [Basic Trend Analysis](#basic-trend-analysis)
4. [Momentum Trading Strategies](#momentum-trading-strategies)
5. [Volatility-Based Trading](#volatility-based-trading)
6. [Volume Analysis](#volume-analysis)
7. [Advanced Indicator Combinations](#advanced-indicator-combinations)
8. [Building Custom Indicators](#building-custom-indicators)
9. [Best Practices](#best-practices)

## Introduction

Technical indicators are mathematical calculations based on a security's price, volume, or open interest. They can help identify trends, momentum, volatility, and potential trading opportunities. This tutorial will guide you through practical applications of the indicators available in our forex trading platform.

## Getting Started with Indicators

### Setting Up Your Environment

Before using technical indicators, make sure you have the proper data structure. The platform expects OHLCV (Open, High, Low, Close, Volume) data in a pandas DataFrame format.

```python
import pandas as pd
from feature_store_service.indicators import moving_averages, oscillators, volatility

# Load historical data
data = pd.read_csv('your_forex_data.csv')

# Ensure columns are properly formatted
required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")
        
# Convert timestamp to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
print(f"Data loaded successfully with {len(data)} rows")
```

### Basic Indicator Calculation

Let's calculate a simple moving average (SMA) and plot it:

```python
# Calculate 20-period SMA
sma_20 = moving_averages.simple_moving_average(data, window=20)

# Plot the price and SMA
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['close'], label='Price')
plt.plot(data['timestamp'], sma_20, label='20-period SMA')
plt.title('Price with 20-period Simple Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
```

## Basic Trend Analysis

### Identifying Trends with Moving Averages

Moving averages are one of the most common indicators for identifying market trends.

```python
# Calculate multiple moving averages
sma_20 = moving_averages.simple_moving_average(data, window=20)
sma_50 = moving_averages.simple_moving_average(data, window=50)
sma_200 = moving_averages.simple_moving_average(data, window=200)

# Generate trend signals
data['trend'] = 'neutral'

# Bullish when short-term MA > long-term MA
data.loc[sma_20 > sma_50, 'trend'] = 'bullish'

# Bearish when short-term MA < long-term MA
data.loc[sma_20 < sma_50, 'trend'] = 'bearish'

# Golden Cross (50 crosses above 200)
golden_cross = (sma_50.shift(1) <= sma_200.shift(1)) & (sma_50 > sma_200)
data.loc[golden_cross, 'trend'] = 'strong_bullish'

# Death Cross (50 crosses below 200)
death_cross = (sma_50.shift(1) >= sma_200.shift(1)) & (sma_50 < sma_200)
data.loc[death_cross, 'trend'] = 'strong_bearish'

# Print trend distribution
print(data['trend'].value_counts())

# Plot the moving averages and price
plt.figure(figsize=(14, 7))
plt.plot(data['timestamp'], data['close'], label='Price', alpha=0.5)
plt.plot(data['timestamp'], sma_20, label='20-period SMA')
plt.plot(data['timestamp'], sma_50, label='50-period SMA')
plt.plot(data['timestamp'], sma_200, label='200-period SMA')

# Highlight golden and death crosses
golden_cross_points = data[golden_cross]
death_cross_points = data[death_cross]

plt.scatter(golden_cross_points['timestamp'], golden_cross_points['close'], 
           color='green', s=100, marker='^', label='Golden Cross')
plt.scatter(death_cross_points['timestamp'], death_cross_points['close'], 
           color='red', s=100, marker='v', label='Death Cross')

plt.title('Trend Analysis with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
```

### Advanced Trend Detection

Using multiple indicator types can provide more robust trend identification:

```python
# Calculate ADX (Average Directional Index) for trend strength
from feature_store_service.indicators.advanced_price_indicators import adx

adx_values, plus_di, minus_di = adx(data, window=14)

# Identify strong trends (ADX > 25 generally indicates a strong trend)
data['trend_strength'] = 'weak'
data.loc[adx_values > 25, 'trend_strength'] = 'moderate'
data.loc[adx_values > 40, 'trend_strength'] = 'strong'
data.loc[adx_values > 60, 'trend_strength'] = 'very_strong'

# Identify trend direction with DMI (Directional Movement Index)
data['trend_direction'] = 'neutral'
data.loc[(plus_di > minus_di) & (adx_values > 25), 'trend_direction'] = 'bullish'
data.loc[(plus_di < minus_di) & (adx_values > 25), 'trend_direction'] = 'bearish'

# Plot the ADX and DMI lines
plt.figure(figsize=(14, 10))

# First subplot for price
plt.subplot(2, 1, 1)
plt.plot(data['timestamp'], data['close'], label='Price')
plt.title('Price Chart')
plt.grid(True)
plt.legend()

# Second subplot for ADX and DMI
plt.subplot(2, 1, 2)
plt.plot(data['timestamp'], adx_values, label='ADX', color='black')
plt.plot(data['timestamp'], plus_di, label='+DI', color='green')
plt.plot(data['timestamp'], minus_di, label='-DI', color='red')
plt.axhline(y=25, color='grey', linestyle='--', label='Trend Threshold')
plt.title('ADX and DMI Indicators')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Momentum Trading Strategies

### RSI-Based Strategy

The Relative Strength Index (RSI) helps identify overbought and oversold conditions.

```python
# Calculate RSI
rsi = oscillators.relative_strength_index(data, window=14)

# Define overbought and oversold thresholds
overbought_threshold = 70
oversold_threshold = 30

# Generate trading signals
data['rsi_signal'] = 'hold'
data.loc[rsi > overbought_threshold, 'rsi_signal'] = 'sell'
data.loc[rsi < oversold_threshold, 'rsi_signal'] = 'buy'

# Find RSI divergence (price making new high but RSI isn't)
data['price_high'] = data['close'] > data['close'].shift(1)
data['rsi_high'] = rsi > rsi.shift(1)
data['bearish_divergence'] = data['price_high'] & ~data['rsi_high']

# Plot RSI with signals
plt.figure(figsize=(14, 10))

# Price subplot
plt.subplot(2, 1, 1)
plt.plot(data['timestamp'], data['close'])
buy_signals = data[data['rsi_signal'] == 'buy']
sell_signals = data[data['rsi_signal'] == 'sell']
plt.scatter(buy_signals['timestamp'], buy_signals['close'], 
           color='green', marker='^', label='Buy Signal')
plt.scatter(sell_signals['timestamp'], sell_signals['close'], 
           color='red', marker='v', label='Sell Signal')
plt.title('Price with RSI Signals')
plt.legend()
plt.grid(True)

# RSI subplot
plt.subplot(2, 1, 2)
plt.plot(data['timestamp'], rsi, color='blue', label='RSI')
plt.axhline(y=overbought_threshold, color='red', linestyle='--', label='Overbought')
plt.axhline(y=oversold_threshold, color='green', linestyle='--', label='Oversold')
plt.axhline(y=50, color='grey', linestyle='-', label='Centerline')
plt.title('RSI Indicator')
plt.ylim(0, 100)
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### MACD Strategy

The Moving Average Convergence Divergence (MACD) indicator helps identify momentum changes.

```python
# Calculate MACD
macd_line, signal_line, histogram = oscillators.macd(
    data, fast_period=12, slow_period=26, signal_period=9)

# Generate signals
data['macd_signal'] = 'hold'
# Buy when MACD crosses above signal line
data.loc[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 
        'macd_signal'] = 'buy'
# Sell when MACD crosses below signal line
data.loc[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), 
        'macd_signal'] = 'sell'

# Plot MACD with signals
plt.figure(figsize=(14, 10))

# Price subplot
plt.subplot(2, 1, 1)
plt.plot(data['timestamp'], data['close'])
buy_signals = data[data['macd_signal'] == 'buy']
sell_signals = data[data['macd_signal'] == 'sell']
plt.scatter(buy_signals['timestamp'], buy_signals['close'], 
           color='green', marker='^', label='Buy Signal')
plt.scatter(sell_signals['timestamp'], sell_signals['close'], 
           color='red', marker='v', label='Sell Signal')
plt.title('Price with MACD Signals')
plt.legend()
plt.grid(True)

# MACD subplot
plt.subplot(2, 1, 2)
plt.plot(data['timestamp'], macd_line, color='blue', label='MACD')
plt.plot(data['timestamp'], signal_line, color='red', label='Signal')
plt.bar(data['timestamp'], histogram, color=np.where(histogram >= 0, 'green', 'red'), label='Histogram')
plt.title('MACD Indicator')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Volatility-Based Trading

### Bollinger Bands Strategy

Bollinger Bands help identify volatility and potential price targets.

```python
# Calculate Bollinger Bands
upper, middle, lower = volatility.bollinger_bands(data, window=20, num_std=2)

# Calculate percent B (position within the bands)
data['percent_b'] = (data['close'] - lower) / (upper - lower)

# Generate trading signals
data['bb_signal'] = 'hold'
# Buy when price touches lower band (percent_b near 0)
data.loc[data['percent_b'] < 0.05, 'bb_signal'] = 'buy'
# Sell when price touches upper band (percent_b near 1)
data.loc[data['percent_b'] > 0.95, 'bb_signal'] = 'sell'

# Calculate bandwidth (volatility indicator)
data['bb_width'] = (upper - lower) / middle

# Plot Bollinger Bands with signals
plt.figure(figsize=(14, 10))

# Price and Bollinger Bands
plt.subplot(2, 1, 1)
plt.plot(data['timestamp'], data['close'], label='Price', alpha=0.7)
plt.plot(data['timestamp'], upper, label='Upper Band', color='red', linestyle='--')
plt.plot(data['timestamp'], middle, label='20-period SMA', color='blue')
plt.plot(data['timestamp'], lower, label='Lower Band', color='green', linestyle='--')
buy_signals = data[data['bb_signal'] == 'buy']
sell_signals = data[data['bb_signal'] == 'sell']
plt.scatter(buy_signals['timestamp'], buy_signals['close'], 
           color='green', marker='^', s=80, label='Buy Signal')
plt.scatter(sell_signals['timestamp'], sell_signals['close'], 
           color='red', marker='v', s=80, label='Sell Signal')
plt.title('Price with Bollinger Bands')
plt.legend()
plt.grid(True)

# Bandwidth subplot (volatility)
plt.subplot(2, 1, 2)
plt.plot(data['timestamp'], data['bb_width'], color='purple', label='Bandwidth')
plt.title('Bollinger Bandwidth (Volatility)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### ATR for Stop Loss Placement

Average True Range (ATR) helps determine appropriate stop loss levels based on market volatility.

```python
# Calculate ATR
atr = volatility.average_true_range(data, window=14)

# Define ATR multiple for stop loss
atr_multiple = 2.0

# Calculate stop loss levels
data['long_stop_loss'] = data['close'] - (atr * atr_multiple)
data['short_stop_loss'] = data['close'] + (atr * atr_multiple)

# Plot price with ATR-based stop losses
plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['close'], label='Price', color='blue')
plt.plot(data['timestamp'], data['long_stop_loss'], label='Long Stop Loss', 
        color='red', linestyle='--')
plt.title('Price with ATR-based Stop Loss Levels (Long Positions)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
```

## Volume Analysis

### On-Balance Volume (OBV) for Trend Confirmation

OBV helps confirm price trends by analyzing volume flow.

```python
# Calculate OBV
obv = volume.on_balance_volume(data)

# Calculate OBV moving average for signal line
obv_ma = pd.DataFrame(obv).rolling(window=20).mean()

# Generate signals based on OBV and price
data['obv_signal'] = 'hold'

# Bullish when price and OBV are rising
price_up = data['close'] > data['close'].shift(5)
obv_up = obv > obv.shift(5)
data.loc[price_up & obv_up, 'obv_signal'] = 'buy'

# Bearish when price is rising but OBV is falling (negative divergence)
data.loc[price_up & ~obv_up, 'obv_signal'] = 'sell'

# Plot price and OBV
plt.figure(figsize=(14, 10))

# Price subplot
plt.subplot(2, 1, 1)
plt.plot(data['timestamp'], data['close'], label='Price')
buy_signals = data[data['obv_signal'] == 'buy']
sell_signals = data[data['obv_signal'] == 'sell']
plt.scatter(buy_signals['timestamp'], buy_signals['close'], 
           color='green', marker='^', label='Buy Signal')
plt.scatter(sell_signals['timestamp'], sell_signals['close'], 
           color='red', marker='v', label='Sell Signal')
plt.title('Price with OBV Signals')
plt.legend()
plt.grid(True)

# OBV subplot
plt.subplot(2, 1, 2)
plt.plot(data['timestamp'], obv, label='OBV', color='blue')
plt.plot(data['timestamp'], obv_ma, label='OBV 20-period MA', color='orange')
plt.title('On-Balance Volume (OBV)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Advanced Indicator Combinations

### Triple Screen Trading System

This strategy uses multiple indicators and timeframes to filter trading signals.

```python
from feature_store_service.indicators.multi_timeframe import multi_timeframe_ma, multi_timeframe_rsi

# Get indicators for different timeframes
# Assuming data_weekly and data_daily are available
weekly_macd_line, weekly_signal, weekly_hist = oscillators.macd(data_weekly)
daily_rsi = oscillators.relative_strength_index(data_daily, window=14)

# First screen: Identify trend on weekly timeframe
data_weekly['trend'] = 'neutral'
data_weekly.loc[weekly_macd_line > weekly_signal, 'trend'] = 'bullish'
data_weekly.loc[weekly_macd_line < weekly_signal, 'trend'] = 'bearish'

# Merge weekly trend to daily data
data_merged = pd.merge_asof(
    data_daily, 
    data_weekly[['timestamp', 'trend']], 
    on='timestamp',
    direction='backward'
)

# Second screen: Look for pullbacks with RSI
data_merged['signal'] = 'hold'

# In bullish weekly trend, buy on daily RSI oversold conditions
data_merged.loc[(data_merged['trend'] == 'bullish') & (daily_rsi < 30), 'signal'] = 'buy'

# In bearish weekly trend, sell on daily RSI overbought conditions
data_merged.loc[(data_merged['trend'] == 'bearish') & (daily_rsi > 70), 'signal'] = 'sell'

# Third screen: Entry timing with intraday data
# (Would typically use hourly or 4-hour data)
# For demonstration, we'll use daily data
data_merged['entry_confirmed'] = False

# Buy when signal is buy and price breaks above yesterday's high
data_merged.loc[(data_merged['signal'] == 'buy') & 
                (data_merged['close'] > data_merged['high'].shift(1)), 
                'entry_confirmed'] = True

# Sell when signal is sell and price breaks below yesterday's low
data_merged.loc[(data_merged['signal'] == 'sell') & 
                (data_merged['close'] < data_merged['low'].shift(1)), 
                'entry_confirmed'] = True

# Plot the triple screen system
plt.figure(figsize=(14, 15))

# Weekly MACD subplot
plt.subplot(3, 1, 1)
plt.plot(data_weekly['timestamp'], weekly_macd_line, label='MACD', color='blue')
plt.plot(data_weekly['timestamp'], weekly_signal, label='Signal', color='red')
plt.bar(data_weekly['timestamp'], weekly_hist, 
       color=np.where(weekly_hist >= 0, 'green', 'red'))
plt.title('First Screen: Weekly MACD')
plt.legend()
plt.grid(True)

# Daily RSI subplot
plt.subplot(3, 1, 2)
plt.plot(data_daily['timestamp'], daily_rsi, color='purple')
plt.axhline(y=70, color='red', linestyle='--')
plt.axhline(y=30, color='green', linestyle='--')
plt.title('Second Screen: Daily RSI')
plt.ylim(0, 100)
plt.grid(True)

# Daily price with signals subplot
plt.subplot(3, 1, 3)
plt.plot(data_merged['timestamp'], data_merged['close'])
entry_signals = data_merged[data_merged['entry_confirmed']]
buy_entries = entry_signals[entry_signals['signal'] == 'buy']
sell_entries = entry_signals[entry_signals['signal'] == 'sell']
plt.scatter(buy_entries['timestamp'], buy_entries['close'], 
           color='green', marker='^', s=100, label='Buy Entry')
plt.scatter(sell_entries['timestamp'], sell_entries['close'], 
           color='red', marker='v', s=100, label='Sell Entry')
plt.title('Third Screen: Entry Signals')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Building Custom Indicators

### Creating a Custom Indicator

You can combine existing indicators or create entirely new ones to suit your trading style:

```python
def custom_trend_strength_indicator(data, short_window=20, long_window=50, rsi_window=14):
    """
    A custom indicator combining moving averages and RSI for trend strength identification.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with OHLCV data
    short_window : int
        Window for short-term MA
    long_window : int
        Window for long-term MA
    rsi_window : int
        Window for RSI calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with trend strength indicator values
    """
    from feature_store_service.indicators import moving_averages, oscillators
    
    # Calculate components
    short_ma = moving_averages.exponential_moving_average(data, window=short_window)
    long_ma = moving_averages.exponential_moving_average(data, window=long_window)
    rsi = oscillators.relative_strength_index(data, window=rsi_window)
    
    # Calculate MA spread as percentage
    ma_spread = 100 * (short_ma - long_ma) / long_ma
    
    # Normalize RSI to -50 to +50 range (from 0-100)
    rsi_normalized = rsi - 50
    
    # Combine signals (equal weight)
    trend_strength = 0.5 * ma_spread + 0.5 * rsi_normalized / 50
    
    # Create result DataFrame
    result = pd.DataFrame({
        'trend_strength': trend_strength,
        'ma_spread': ma_spread,
        'rsi_normalized': rsi_normalized
    }, index=data.index)
    
    return result

# Use the custom indicator
trend_strength_data = custom_trend_strength_indicator(data)

# Plot the custom indicator
plt.figure(figsize=(14, 10))

# Price subplot
plt.subplot(2, 1, 1)
plt.plot(data['timestamp'], data['close'])
plt.title('Price Chart')
plt.grid(True)

# Custom indicator subplot
plt.subplot(2, 1, 2)
plt.plot(data['timestamp'], trend_strength_data['trend_strength'], label='Trend Strength', color='blue')
plt.axhline(y=0, color='grey', linestyle='-')
plt.axhline(y=0.5, color='green', linestyle='--', label='Strong Bullish')
plt.axhline(y=-0.5, color='red', linestyle='--', label='Strong Bearish')
plt.title('Custom Trend Strength Indicator')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Best Practices

### Avoiding Common Pitfalls

1. **Lagging Indicators**: Most indicators are lagging by nature. Combine leading and lagging indicators to balance this effect.

2. **Overoptimization**: Don't tweak parameters to perfectly fit historical data, as this often leads to poor future performance.

3. **Indicator Redundancy**: Using multiple indicators that measure the same market aspect (e.g., three different momentum indicators) adds little value and can lead to analysis paralysis.

4. **Ignoring the Trend**: Always consider the overall market trend when interpreting indicator signals.

5. **Confirmation Bias**: Don't selectively look at indicators that confirm your existing bias. Consider all relevant indicators.

### Effective Indicator Combinations

Combining indicators that measure different aspects of the market often provides the most reliable signals:

1. **Trend + Momentum + Volume**: Use moving averages to identify the trend, RSI or MACD for momentum, and OBV for volume confirmation.

2. **Multiple Timeframes**: Analyze indicators across different timeframes for more robust signals.

3. **Price Action + Indicators**: Always confirm indicator signals with actual price action (support/resistance breaks, chart patterns).

### Performance Evaluation

Regularly evaluate the performance of your indicator-based strategies:

```python
def evaluate_strategy(data, signal_column='signal'):
    """Simple strategy evaluation"""
    # Calculate returns for buy/sell signals
    data['return'] = data['close'].pct_change()
    
    # For buy signals, calculate forward returns
    data['strategy_return'] = 0.0
    data.loc[data[signal_column] == 'buy', 'strategy_return'] = data['return'].shift(-1)
    
    # For sell signals (shorts), calculate inverse of forward returns
    data.loc[data[signal_column] == 'sell', 'strategy_return'] = -data['return'].shift(-1)
    
    # Calculate cumulative returns
    data['cum_market_return'] = (1 + data['return']).cumprod() - 1
    data['cum_strategy_return'] = (1 + data['strategy_return']).cumprod() - 1
    
    # Calculate metrics
    total_trades = len(data[data[signal_column].isin(['buy', 'sell'])])
    winning_trades = len(data[(data[signal_column].isin(['buy', 'sell'])) & 
                            (data['strategy_return'] > 0)])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['cum_market_return'], 
            label='Buy & Hold', color='blue', alpha=0.7)
    plt.plot(data['timestamp'], data['cum_strategy_return'], 
            label='Strategy', color='green')
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Final Strategy Return: {data['cum_strategy_return'].iloc[-1]:.2%}")
    print(f"Final Market Return: {data['cum_market_return'].iloc[-1]:.2%}")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'strategy_return': data['cum_strategy_return'].iloc[-1],
        'market_return': data['cum_market_return'].iloc[-1]
    }

# Evaluate the RSI strategy from earlier
evaluate_strategy(data, signal_column='rsi_signal')
```

By implementing these best practices and continuously refining your approach, you can develop effective trading strategies using the technical indicators available in the platform.
