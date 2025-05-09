# Advanced Pattern Recognition

This module provides advanced pattern recognition capabilities for forex market analysis.

## Overview

The Advanced Pattern Recognition module is designed to identify complex chart patterns that go beyond simple technical indicators. It includes pattern recognition for various chart types and analysis methodologies:

- Renko Patterns
- Ichimoku Cloud Patterns
- Wyckoff Patterns
- Heikin-Ashi Patterns
- Volume Spread Analysis (VSA) Patterns
- Market Profile Analysis
- Point and Figure Analysis
- Wolfe Wave Patterns
- Pitchfork Analysis
- Divergence Detection

## Architecture

The module is organized with a facade pattern to provide a unified interface to all pattern recognition systems:

- `AdvancedPatternFacade`: Main entry point that coordinates all pattern recognizers
- Individual pattern recognizers for each chart type/methodology
- Base classes and utilities for pattern detection

## Usage

### Basic Usage with Facade

```python
from feature_store_service.indicators.advanced_patterns import AdvancedPatternFacade

# Initialize the facade with default settings
facade = AdvancedPatternFacade()

# Calculate patterns on OHLCV data
result = facade.calculate(data)

# Check for patterns
has_patterns = result['has_advanced_pattern'].any()

# Find specific patterns
patterns = facade.find_patterns(data)
```

### Using Specific Pattern Recognizers

```python
from feature_store_service.indicators.advanced_patterns import (
    RenkoPatternRecognizer,
    IchimokuPatternRecognizer
)

# Initialize a specific recognizer
renko_recognizer = RenkoPatternRecognizer(
    brick_size=0.5,
    min_trend_length=3,
    sensitivity=0.8
)

# Calculate Renko patterns
renko_result = renko_recognizer.calculate(data)

# Check for specific patterns
has_reversal = renko_result['pattern_renko_reversal'].any()
```

## Pattern Types

### Renko Patterns

- `renko_reversal`: Reversal patterns in Renko charts
- `renko_breakout`: Breakout patterns after consolidation
- `renko_double_top`: Double top formations
- `renko_double_bottom`: Double bottom formations

### Ichimoku Patterns

- `ichimoku_tk_cross`: Tenkan-Kijun (TK) crosses
- `ichimoku_kumo_breakout`: Price breaking through the Kumo (cloud)
- `ichimoku_kumo_twist`: Senkou Span A and B crossing (cloud twist)
- `ichimoku_chikou_cross`: Chikou Span crossing the price

### Wyckoff Patterns

- `wyckoff_accumulation`: Wyckoff accumulation phases
- `wyckoff_distribution`: Wyckoff distribution phases
- `wyckoff_spring`: Spring pattern (false breakout downward)
- `wyckoff_upthrust`: Upthrust pattern (false breakout upward)

### Heikin-Ashi Patterns

- `heikin_ashi_reversal`: Reversal patterns in Heikin-Ashi charts
- `heikin_ashi_continuation`: Continuation patterns in Heikin-Ashi charts

### VSA Patterns

- `vsa_no_demand`: No demand bars (bearish)
- `vsa_no_supply`: No supply bars (bullish)
- `vsa_stopping_volume`: Stopping volume patterns
- `vsa_climactic_volume`: Climactic volume patterns
- `vsa_effort_vs_result`: Effort vs. result divergence

## Pattern Information

Each detected pattern includes:

- Direction (bullish, bearish, neutral)
- Strength (0.0-1.0)
- Target price projection
- Suggested stop loss level

## Examples

See the `examples/advanced_patterns_example.py` file for complete usage examples.

## Integration with Indicator Service

The Advanced Pattern Recognition module is fully integrated with the Feature Store Service's indicator system. You can use it through the `IndicatorService` or `EnhancedIndicatorService` just like any other indicator:

```python
from feature_store_service.services.indicator_service import IndicatorService

service = IndicatorService()
result = service.calculate_indicator(data, "AdvancedPatternFacade", lookback_period=50, sensitivity=0.75)
```