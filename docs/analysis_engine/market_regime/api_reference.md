# Market Regime Analysis API Reference

## Overview

The Market Regime Analysis component provides tools for analyzing and classifying market regimes based on price data. It includes functionality for feature extraction, regime detection, and classification.

## Modules

### `analysis_engine.analysis.market_regime`

This is the main package for market regime analysis. It provides the following public API:

- `MarketRegimeAnalyzer`: Main class for performing market regime analysis
- `RegimeType`: Enum representing different market regime types
- `RegimeClassification`: Data model for regime classification results

## Classes

### `MarketRegimeAnalyzer`

The main analyzer class for market regime analysis. This class coordinates the detection and classification of market regimes based on price data.

#### Constructor

```python
MarketRegimeAnalyzer(config=None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary with parameters for detector and classifier components.

#### Methods

##### `analyze`

```python
analyze(price_data, timestamp=None)
```

Analyze price data to determine the current market regime.

**Parameters:**
- `price_data` (pandas.DataFrame): DataFrame with OHLCV data. Required columns: 'open', 'high', 'low', 'close'. Optional: 'volume'.
- `timestamp` (datetime, optional): Timestamp for the analysis. Defaults to current time.

**Returns:**
- `RegimeClassification`: Classification result with regime type, direction, volatility level, and confidence.

##### `analyze_cached`

```python
analyze_cached(instrument, timeframe, price_data_key, timestamp=None)
```

Cached version of analyze for repeated calls with the same data.

**Parameters:**
- `instrument` (str): Instrument symbol
- `timeframe` (str): Timeframe string (e.g., 'H1', 'D1')
- `price_data_key` (str): A unique key representing the price data
- `timestamp` (str, optional): Timestamp string for the analysis

**Returns:**
- `dict`: Classification result as a dictionary

##### `subscribe_to_regime_changes`

```python
subscribe_to_regime_changes(callback)
```

Subscribe to regime change events.

**Parameters:**
- `callback` (callable): Function to call when regime changes. The callback will receive the new classification and the old one.

##### `unsubscribe_from_regime_changes`

```python
unsubscribe_from_regime_changes(callback)
```

Unsubscribe from regime change events.

**Parameters:**
- `callback` (callable): Function to remove from subscribers

##### `get_historical_regimes`

```python
get_historical_regimes(price_data, window_size=1)
```

Analyze historical price data to determine regime changes over time.

**Parameters:**
- `price_data` (pandas.DataFrame): DataFrame with OHLCV data
- `window_size` (int, optional): Size of the rolling window for analysis. Defaults to 1.

**Returns:**
- `List[RegimeClassification]`: List of regime classifications over time

### `RegimeDetector`

Extracts features from price data to detect market regimes.

#### Constructor

```python
RegimeDetector(config=None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary with parameters:
  - `lookback_periods`: Dict mapping feature names to lookback periods
  - `atr_period`: Period for ATR calculation (default: 14)
  - `adx_period`: Period for ADX calculation (default: 14)
  - `rsi_period`: Period for RSI calculation (default: 14)
  - `volatility_threshold`: Dict with 'low', 'medium', 'high' thresholds

#### Methods

##### `extract_features`

```python
extract_features(price_data)
```

Extract features from price data for regime detection.

**Parameters:**
- `price_data` (pandas.DataFrame): DataFrame with OHLCV data. Required columns: 'open', 'high', 'low', 'close'.

**Returns:**
- `FeatureSet`: A set of features for regime classification

### `RegimeClassifier`

Classifies market regimes based on extracted features.

#### Constructor

```python
RegimeClassifier(config=None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary with parameters:
  - `trend_threshold`: Threshold for trend strength (default: 0.25)
  - `volatility_thresholds`: Dict with thresholds for volatility levels
  - `momentum_threshold`: Threshold for momentum (default: 0.2)
  - `hysteresis`: Dict with hysteresis values for regime changes

#### Methods

##### `classify`

```python
classify(features, timestamp=None)
```

Classify market regime based on extracted features.

**Parameters:**
- `features` (FeatureSet): FeatureSet containing extracted features
- `timestamp` (datetime, optional): Timestamp for the classification. Defaults to current time.

**Returns:**
- `RegimeClassification`: Classification result with regime type, direction, volatility level, and confidence

## Data Models

### `RegimeType` (Enum)

Enum representing different market regime types.

- `TRENDING_BULLISH`: Strong uptrend
- `TRENDING_BEARISH`: Strong downtrend
- `RANGING_NEUTRAL`: Sideways market with no clear direction
- `RANGING_BULLISH`: Sideways market with bullish bias
- `RANGING_BEARISH`: Sideways market with bearish bias
- `VOLATILE_BULLISH`: Highly volatile market with bullish bias
- `VOLATILE_BEARISH`: Highly volatile market with bearish bias
- `VOLATILE_NEUTRAL`: Highly volatile market with no clear direction
- `UNDEFINED`: Undefined or unknown regime

### `DirectionType` (Enum)

Enum representing market direction.

- `BULLISH`: Upward direction
- `BEARISH`: Downward direction
- `NEUTRAL`: No clear direction

### `VolatilityLevel` (Enum)

Enum representing different volatility levels.

- `LOW`: Low volatility
- `MEDIUM`: Medium volatility
- `HIGH`: High volatility
- `EXTREME`: Extreme volatility

### `RegimeClassification`

Data model for regime classification results.

#### Constructor

```python
RegimeClassification(regime, confidence, direction, volatility, timestamp, features, metadata=None)
```

**Parameters:**
- `regime` (RegimeType): The classified regime type
- `confidence` (float): Confidence level of the classification (0-1)
- `direction` (DirectionType): Market direction
- `volatility` (VolatilityLevel): Volatility level
- `timestamp` (datetime): Timestamp of the classification
- `features` (dict): Dictionary of features used for classification
- `metadata` (dict, optional): Additional metadata

#### Methods

##### `to_dict`

```python
to_dict()
```

Convert the classification to a dictionary.

**Returns:**
- `dict`: Dictionary representation of the classification

##### `from_dict` (classmethod)

```python
from_dict(data)
```

Create a classification from a dictionary.

**Parameters:**
- `data` (dict): Dictionary representation of the classification

**Returns:**
- `RegimeClassification`: Classification object

### `FeatureSet`

Data model for feature sets used in regime analysis.

#### Constructor

```python
FeatureSet(volatility, trend_strength, momentum, mean_reversion, range_width, additional_features=None)
```

**Parameters:**
- `volatility` (float): Market volatility
- `trend_strength` (float): Strength of the trend
- `momentum` (float): Market momentum
- `mean_reversion` (float): Mean reversion tendency
- `range_width` (float): Width of the trading range
- `additional_features` (dict, optional): Additional features

#### Methods

##### `to_dict`

```python
to_dict()
```

Convert the feature set to a dictionary.

**Returns:**
- `dict`: Dictionary representation of the feature set

##### `from_dict` (classmethod)

```python
from_dict(data)
```

Create a feature set from a dictionary.

**Parameters:**
- `data` (dict): Dictionary representation of the feature set

**Returns:**
- `FeatureSet`: Feature set object

## Usage Examples

### Basic Usage

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
```

### Subscribing to Regime Changes

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
import pandas as pd

# Create analyzer
analyzer = MarketRegimeAnalyzer()

# Define callback for regime changes
def on_regime_change(new_classification, old_classification):
    print(f"Regime changed from {old_classification.regime.name if old_classification else 'None'} "
          f"to {new_classification.regime.name}")
    print(f"New direction: {new_classification.direction.name}")
    print(f"New volatility: {new_classification.volatility.name}")

# Subscribe to regime changes
analyzer.subscribe_to_regime_changes(on_regime_change)

# Analyze price data (this will trigger the callback if regime changes)
price_data = pd.read_csv('price_data.csv')
analyzer.analyze(price_data)
```

### Analyzing Historical Regimes

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

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

# Plot price and regimes
plt.figure(figsize=(12, 8))

# Plot price
plt.subplot(2, 1, 1)
plt.plot(price_data.index, price_data['close'])
plt.title('Price Chart')
plt.grid(True)

# Plot regimes
plt.subplot(2, 1, 2)
plt.scatter(timestamps, [1] * len(timestamps), c=[hash(r) % 256 for r in regimes], s=100)
plt.yticks([])
plt.title('Market Regimes')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Custom Configuration

```python
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
import pandas as pd

# Define custom configuration
config = {
    'detector': {
        'lookback_periods': {
            'volatility': 10,
            'trend_strength': 14,
            'momentum': 10,
            'mean_reversion': 20,
            'range_width': 30
        },
        'atr_period': 10,
        'adx_period': 14,
        'rsi_period': 10,
        'volatility_threshold': {
            'low': 0.4,
            'medium': 0.8,
            'high': 1.5
        }
    },
    'classifier': {
        'trend_threshold': 0.3,
        'volatility_thresholds': {
            'low': 0.4,
            'medium': 0.8,
            'high': 1.5
        },
        'momentum_threshold': 0.25,
        'hysteresis': {
            'trend_to_range': 0.05,
            'range_to_trend': 0.05,
            'volatility_increase': 0.1,
            'volatility_decrease': 0.1
        }
    },
    'cache_size': 256
}

# Create analyzer with custom configuration
analyzer = MarketRegimeAnalyzer(config=config)

# Analyze price data
price_data = pd.read_csv('price_data.csv')
result = analyzer.analyze(price_data)

print(f"Current regime: {result.regime.name}")
print(f"Confidence: {result.confidence:.2f}")
```