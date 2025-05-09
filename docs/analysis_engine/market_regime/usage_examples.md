# Market Regime Analysis Usage Examples

This document provides examples of how to use the Market Regime Analysis component in various scenarios.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Analyzing Historical Regimes](#analyzing-historical-regimes)
3. [Subscribing to Regime Changes](#subscribing-to-regime-changes)
4. [Custom Configuration](#custom-configuration)
5. [Integration with Trading Strategies](#integration-with-trading-strategies)
6. [Visualization](#visualization)

## Basic Usage

The most basic usage is to analyze the current market regime based on price data:

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
import pandas as pd

# Load price data
price_data = pd.read_csv('price_data.csv')

# Create analyzer
analyzer = MarketRegimeAnalyzer()

# Analyze current regime
result = analyzer.analyze(price_data)

# Print results
print(f"Current regime: {result.regime.name}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Direction: {result.direction.name}")
print(f"Volatility: {result.volatility.name}")

# Access features used for classification
print("\nFeatures:")
for feature, value in result.features.items():
    print(f"  {feature}: {value:.4f}")
```

Example output:
```
Current regime: TRENDING_BULLISH
Confidence: 0.85
Direction: BULLISH
Volatility: MEDIUM

Features:
  volatility: 0.8000
  trend_strength: 0.7000
  momentum: 0.5000
  mean_reversion: -0.1000
  range_width: 0.0200
  price_velocity: 0.0200
  volume_trend: 0.3000
  swing_strength: 0.0100
```

## Analyzing Historical Regimes

To analyze how market regimes have changed over time:

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import numpy as np

# Load historical price data
price_data = pd.read_csv('historical_price_data.csv', index_col='date', parse_dates=True)

# Create analyzer
analyzer = MarketRegimeAnalyzer()

# Get historical regimes
window_size = 20  # Use 20-day windows for analysis
historical_regimes = analyzer.get_historical_regimes(price_data, window_size=window_size)

# Extract regime types and timestamps
timestamps = [r.timestamp for r in historical_regimes]
regimes = [r.regime.name for r in historical_regimes]
confidences = [r.confidence for r in historical_regimes]

# Create a mapping of regime names to integers for coloring
unique_regimes = list(set(regimes))
regime_to_int = {regime: i for i, regime in enumerate(unique_regimes)}
regime_ints = [regime_to_int[r] for r in regimes]

# Create a colormap
cmap = plt.cm.get_cmap('tab10', len(unique_regimes))

# Plot price and regimes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot price
ax1.plot(price_data.index, price_data['close'], 'k-')
ax1.set_title('Price Chart')
ax1.grid(True)
ax1.set_ylabel('Price')

# Plot regimes
scatter = ax2.scatter(timestamps, [1] * len(timestamps), c=regime_ints, cmap=cmap, s=100 * np.array(confidences))
ax2.set_yticks([])
ax2.set_title('Market Regimes')
ax2.grid(True)
ax2.set_xlabel('Date')

# Add legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             label=regime,
                             markerfacecolor=cmap(regime_to_int[regime]), 
                             markersize=10)
                  for regime in unique_regimes]
ax2.legend(handles=legend_elements, loc='upper right')

# Format x-axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.tight_layout()
plt.show()
```

## Subscribing to Regime Changes

To be notified when the market regime changes:

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
import pandas as pd
import time

# Create analyzer
analyzer = MarketRegimeAnalyzer()

# Define callback for regime changes
def on_regime_change(new_classification, old_classification):
    print(f"[{new_classification.timestamp}] Regime changed from "
          f"{old_classification.regime.name if old_classification else 'None'} "
          f"to {new_classification.regime.name}")
    print(f"  Direction: {new_classification.direction.name}")
    print(f"  Volatility: {new_classification.volatility.name}")
    print(f"  Confidence: {new_classification.confidence:.2f}")
    
    # You could trigger trading actions here
    if new_classification.regime.name.startswith('TRENDING'):
        print("  Action: Adjust for trending market")
    elif new_classification.regime.name.startswith('RANGING'):
        print("  Action: Adjust for ranging market")
    elif new_classification.regime.name.startswith('VOLATILE'):
        print("  Action: Reduce position sizes due to volatility")

# Subscribe to regime changes
analyzer.subscribe_to_regime_changes(on_regime_change)

# Simulate real-time analysis
def simulate_real_time_analysis():
    # Load historical data
    full_data = pd.read_csv('historical_price_data.csv')
    
    # Process data in chunks to simulate real-time updates
    window_size = 100
    for i in range(window_size, len(full_data), 10):
        # Get the current window of data
        current_data = full_data.iloc[i-window_size:i]
        
        # Analyze the current window
        analyzer.analyze(current_data)
        
        # Wait a bit to simulate real-time
        time.sleep(1)

# Run the simulation
simulate_real_time_analysis()
```

Example output:
```
[2023-01-15 00:00:00] Regime changed from None to RANGING_NEUTRAL
  Direction: NEUTRAL
  Volatility: LOW
  Confidence: 0.53
  Action: Adjust for ranging market

[2023-01-25 00:00:00] Regime changed from RANGING_NEUTRAL to TRENDING_BULLISH
  Direction: BULLISH
  Volatility: MEDIUM
  Confidence: 0.78
  Action: Adjust for trending market

[2023-02-10 00:00:00] Regime changed from TRENDING_BULLISH to VOLATILE_BULLISH
  Direction: BULLISH
  Volatility: EXTREME
  Confidence: 0.90
  Action: Reduce position sizes due to volatility
```

## Custom Configuration

To customize the behavior of the analyzer:

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
import pandas as pd

# Define custom configuration
config = {
    'detector': {
        'lookback_periods': {
            'volatility': 10,        # Shorter period for volatility detection
            'trend_strength': 14,    # Standard period for trend detection
            'momentum': 10,          # Shorter period for momentum
            'mean_reversion': 20,    # Longer period for mean reversion
            'range_width': 30        # Longer period for range detection
        },
        'atr_period': 10,            # Period for ATR calculation
        'adx_period': 14,            # Period for ADX calculation
        'rsi_period': 10,            # Period for RSI calculation
        'volatility_threshold': {
            'low': 0.4,              # Threshold for low volatility
            'medium': 0.8,           # Threshold for medium volatility
            'high': 1.5              # Threshold for high volatility
        }
    },
    'classifier': {
        'trend_threshold': 0.3,      # Higher threshold for trend detection
        'volatility_thresholds': {
            'low': 0.4,              # Threshold for low volatility
            'medium': 0.8,           # Threshold for medium volatility
            'high': 1.5              # Threshold for high volatility
        },
        'momentum_threshold': 0.25,  # Threshold for directional momentum
        'hysteresis': {
            'trend_to_range': 0.05,  # Hysteresis for trend to range transition
            'range_to_trend': 0.05,  # Hysteresis for range to trend transition
            'volatility_increase': 0.1,  # Hysteresis for volatility increase
            'volatility_decrease': 0.1   # Hysteresis for volatility decrease
        }
    },
    'cache_size': 256               # Size of the LRU cache
}

# Create analyzer with custom configuration
analyzer = MarketRegimeAnalyzer(config=config)

# Analyze price data
price_data = pd.read_csv('price_data.csv')
result = analyzer.analyze(price_data)

print(f"Current regime: {result.regime.name}")
print(f"Confidence: {result.confidence:.2f}")
```

## Integration with Trading Strategies

Here's an example of how to integrate market regime analysis with a trading strategy:

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer, RegimeType
import pandas as pd

class RegimeAwareStrategy:
    """A trading strategy that adapts to different market regimes."""
    
    def __init__(self):
        self.analyzer = MarketRegimeAnalyzer()
        self.current_regime = None
        self.position_size = 1.0  # Base position size
    
    def analyze_regime(self, price_data):
        """Analyze the current market regime."""
        self.current_regime = self.analyzer.analyze(price_data)
        return self.current_regime
    
    def calculate_position_size(self):
        """Calculate position size based on market regime."""
        if self.current_regime is None:
            return self.position_size
            
        # Adjust position size based on volatility
        volatility_adjustments = {
            'LOW': 1.2,      # Increase size in low volatility
            'MEDIUM': 1.0,   # Normal size in medium volatility
            'HIGH': 0.7,     # Reduce size in high volatility
            'EXTREME': 0.5   # Significantly reduce size in extreme volatility
        }
        
        volatility_factor = volatility_adjustments[self.current_regime.volatility.name]
        
        # Adjust position size based on confidence
        confidence_factor = self.current_regime.confidence
        
        # Calculate adjusted position size
        adjusted_size = self.position_size * volatility_factor * confidence_factor
        
        return adjusted_size
    
    def generate_signals(self, price_data, indicators):
        """Generate trading signals based on market regime and indicators."""
        # Analyze regime if not already done
        if self.current_regime is None:
            self.analyze_regime(price_data)
        
        signals = []
        
        # Different signal generation based on regime
        if self.current_regime.regime in [RegimeType.TRENDING_BULLISH, RegimeType.TRENDING_BEARISH]:
            # In trending markets, use trend-following indicators
            signals = self._generate_trend_following_signals(indicators)
            
        elif self.current_regime.regime in [RegimeType.RANGING_NEUTRAL, RegimeType.RANGING_BULLISH, RegimeType.RANGING_BEARISH]:
            # In ranging markets, use mean-reversion indicators
            signals = self._generate_mean_reversion_signals(indicators)
            
        elif self.current_regime.regime in [RegimeType.VOLATILE_BULLISH, RegimeType.VOLATILE_BEARISH, RegimeType.VOLATILE_NEUTRAL]:
            # In volatile markets, use volatility-based indicators
            signals = self._generate_volatility_based_signals(indicators)
        
        # Adjust position sizes
        position_size = self.calculate_position_size()
        for signal in signals:
            signal['size'] = position_size
        
        return signals
    
    def _generate_trend_following_signals(self, indicators):
        """Generate signals for trending markets."""
        # Implementation depends on your specific indicators
        # This is just a placeholder
        signals = []
        
        if indicators.get('macd_histogram', 0) > 0 and indicators.get('adx', 0) > 25:
            signals.append({
                'type': 'BUY',
                'reason': 'Strong trend detected by MACD and ADX'
            })
        elif indicators.get('macd_histogram', 0) < 0 and indicators.get('adx', 0) > 25:
            signals.append({
                'type': 'SELL',
                'reason': 'Strong trend detected by MACD and ADX'
            })
        
        return signals
    
    def _generate_mean_reversion_signals(self, indicators):
        """Generate signals for ranging markets."""
        # Implementation depends on your specific indicators
        # This is just a placeholder
        signals = []
        
        if indicators.get('rsi', 50) < 30:
            signals.append({
                'type': 'BUY',
                'reason': 'Oversold condition in ranging market'
            })
        elif indicators.get('rsi', 50) > 70:
            signals.append({
                'type': 'SELL',
                'reason': 'Overbought condition in ranging market'
            })
        
        return signals
    
    def _generate_volatility_based_signals(self, indicators):
        """Generate signals for volatile markets."""
        # Implementation depends on your specific indicators
        # This is just a placeholder
        signals = []
        
        # In volatile markets, we might want to be more conservative
        if indicators.get('rsi', 50) < 20:
            signals.append({
                'type': 'BUY',
                'reason': 'Extremely oversold in volatile market'
            })
        elif indicators.get('rsi', 50) > 80:
            signals.append({
                'type': 'SELL',
                'reason': 'Extremely overbought in volatile market'
            })
        
        return signals

# Example usage
strategy = RegimeAwareStrategy()

# Load price data and calculate indicators
price_data = pd.read_csv('price_data.csv')
indicators = {
    'macd_histogram': 0.5,  # Positive MACD histogram
    'adx': 30,              # Strong trend
    'rsi': 45               # Neutral RSI
}

# Analyze regime
regime = strategy.analyze_regime(price_data)
print(f"Current regime: {regime.regime.name}")
print(f"Confidence: {regime.confidence:.2f}")
print(f"Direction: {regime.direction.name}")
print(f"Volatility: {regime.volatility.name}")

# Generate signals
signals = strategy.generate_signals(price_data, indicators)
for signal in signals:
    print(f"Signal: {signal['type']}")
    print(f"Reason: {signal['reason']}")
    print(f"Size: {signal['size']:.2f}")
```

## Visualization

Here's an example of how to visualize market regimes:

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

# Load historical price data
price_data = pd.read_csv('historical_price_data.csv', index_col='date', parse_dates=True)

# Create analyzer
analyzer = MarketRegimeAnalyzer()

# Get historical regimes
window_size = 20  # Use 20-day windows for analysis
historical_regimes = analyzer.get_historical_regimes(price_data, window_size=window_size)

# Create a DataFrame with regime information
regime_data = pd.DataFrame({
    'timestamp': [r.timestamp for r in historical_regimes],
    'regime': [r.regime.name for r in historical_regimes],
    'direction': [r.direction.name for r in historical_regimes],
    'volatility': [r.volatility.name for r in historical_regimes],
    'confidence': [r.confidence for r in historical_regimes]
})
regime_data.set_index('timestamp', inplace=True)

# Resample price data to match regime data
price_resampled = price_data.reindex(regime_data.index, method='nearest')

# Create a comprehensive visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

# Plot 1: Price chart with regime background
ax1 = axes[0]
ax1.plot(price_resampled.index, price_resampled['close'], 'k-', linewidth=1.5)
ax1.set_title('Price Chart with Market Regimes', fontsize=14)
ax1.set_ylabel('Price', fontsize=12)
ax1.grid(True, alpha=0.3)

# Color the background based on regime
regime_colors = {
    'TRENDING_BULLISH': 'lightgreen',
    'TRENDING_BEARISH': 'lightcoral',
    'RANGING_NEUTRAL': 'lightskyblue',
    'RANGING_BULLISH': 'palegreen',
    'RANGING_BEARISH': 'lightsalmon',
    'VOLATILE_BULLISH': 'yellowgreen',
    'VOLATILE_BEARISH': 'salmon',
    'VOLATILE_NEUTRAL': 'khaki',
    'UNDEFINED': 'lightgray'
}

# Create background colors
prev_date = price_resampled.index[0]
prev_regime = regime_data.iloc[0]['regime']
for i in range(1, len(regime_data)):
    curr_date = regime_data.index[i]
    curr_regime = regime_data.iloc[i]['regime']
    
    if curr_regime != prev_regime:
        ax1.axvspan(prev_date, curr_date, alpha=0.3, color=regime_colors[prev_regime])
        prev_date = curr_date
        prev_regime = curr_regime

# Add the last segment
ax1.axvspan(prev_date, regime_data.index[-1], alpha=0.3, color=regime_colors[prev_regime])

# Create legend for regimes
legend_patches = [mpatches.Patch(color=color, alpha=0.3, label=regime) 
                 for regime, color in regime_colors.items() if regime in regime_data['regime'].unique()]
ax1.legend(handles=legend_patches, loc='upper left', fontsize=10)

# Plot 2: Direction and confidence
ax2 = axes[1]
direction_colors = {'BULLISH': 'green', 'BEARISH': 'red', 'NEUTRAL': 'blue'}
for direction in direction_colors:
    mask = regime_data['direction'] == direction
    ax2.scatter(regime_data.index[mask], regime_data['confidence'][mask], 
               color=direction_colors[direction], alpha=0.7, label=direction)
ax2.set_ylim(0, 1.1)
ax2.set_ylabel('Confidence', fontsize=12)
ax2.set_title('Direction and Confidence', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=10)

# Plot 3: Volatility
ax3 = axes[2]
volatility_colors = {'LOW': 'green', 'MEDIUM': 'blue', 'HIGH': 'orange', 'EXTREME': 'red'}
for vol_level in volatility_colors:
    mask = regime_data['volatility'] == vol_level
    ax3.scatter(regime_data.index[mask], [1] * sum(mask), 
               color=volatility_colors[vol_level], alpha=0.7, s=100, label=vol_level)
ax3.set_yticks([])
ax3.set_title('Volatility Levels', fontsize=14)
ax3.set_xlabel('Date', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=10)

# Format x-axis
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.tight_layout()
plt.show()
```

This visualization includes:
1. A price chart with colored backgrounds indicating different market regimes
2. A plot showing the direction (bullish, bearish, neutral) and confidence level
3. A plot showing the volatility levels over time

This comprehensive visualization helps traders understand how market regimes have evolved and how they relate to price movements.