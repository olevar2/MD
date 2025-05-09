# Market Regime Analysis

## Overview

The Market Regime Analysis component provides tools for analyzing and classifying market regimes based on price data. It includes functionality for feature extraction, regime detection, and classification.

## Features

- **Feature Extraction**: Extract features from price data, including volatility, trend strength, momentum, mean reversion, and range width.
- **Regime Classification**: Classify market regimes based on extracted features, including trending, ranging, and volatile regimes.
- **Direction Detection**: Determine market direction (bullish, bearish, neutral) based on momentum and other indicators.
- **Volatility Assessment**: Assess market volatility levels (low, medium, high, extreme) based on price action.
- **Regime Change Detection**: Detect and notify subscribers of regime changes.
- **Historical Analysis**: Analyze historical price data to identify regime changes over time.

## Architecture

The component follows a modular architecture with clear separation of concerns:

- **RegimeDetector**: Extracts features from price data.
- **RegimeClassifier**: Classifies market regimes based on extracted features.
- **MarketRegimeAnalyzer**: Coordinates detection and classification, provides a simple API for clients.

## Usage

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

## Configuration

The component can be configured with custom parameters:

```python
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

analyzer = MarketRegimeAnalyzer(config=config)
```

## Documentation

For more detailed documentation, see:

- [API Reference](../../docs/analysis_engine/market_regime/api_reference.md)
- [Usage Examples](../../docs/analysis_engine/market_regime/usage_examples.md)
- [Architecture](../../docs/analysis_engine/market_regime/architecture.md)

## Testing

The component includes comprehensive unit tests, integration tests, and characterization tests. To run the tests:

```bash
python -m unittest discover -s tests/analysis_engine/analysis/market_regime
```