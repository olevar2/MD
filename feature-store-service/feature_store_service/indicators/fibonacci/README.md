# Fibonacci Analysis Tools

This package provides comprehensive Fibonacci-based technical analysis tools for the Forex Trading Platform.

## Overview

Fibonacci analysis is a popular technical analysis method that uses Fibonacci ratios to identify potential support, resistance, and price targets. This package implements various Fibonacci tools including:

- **Fibonacci Retracements**: Identify potential support/resistance levels during price corrections
- **Fibonacci Extensions**: Project potential price targets beyond the initial trend
- **Fibonacci Fans**: Create diagonal support/resistance lines based on Fibonacci ratios
- **Fibonacci Time Zones**: Identify potential time points for market reversals
- **Fibonacci Circles**: Create circular support/resistance zones based on Fibonacci ratios
- **Fibonacci Clusters**: Identify areas where multiple Fibonacci levels converge

## Usage Examples

### Fibonacci Retracement

```python
from feature_store_service.indicators.fibonacci import FibonacciRetracement

# Create a Fibonacci Retracement indicator with auto-detection
fib_retracement = FibonacciRetracement(
    levels=[0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
    swing_lookback=30,
    auto_detect_swings=True
)

# Calculate retracement levels
result = fib_retracement.calculate(data)
```

### Fibonacci Extension

```python
from feature_store_service.indicators.fibonacci import FibonacciExtension

# Create a Fibonacci Extension indicator with auto-detection
fib_extension = FibonacciExtension(
    levels=[0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618],
    swing_lookback=30,
    auto_detect_swings=True
)

# Calculate extension levels
result = fib_extension.calculate(data)
```

### Fibonacci Fan

```python
from feature_store_service.indicators.fibonacci import FibonacciFan

# Create a Fibonacci Fan indicator with auto-detection
fib_fan = FibonacciFan(
    levels=[0.236, 0.382, 0.5, 0.618, 0.786],
    swing_lookback=30,
    auto_detect_swings=True,
    projection_bars=50
)

# Calculate fan levels
result = fib_fan.calculate(data)
```

### Fibonacci Time Zones

```python
from feature_store_service.indicators.fibonacci import FibonacciTimeZones

# Create a Fibonacci Time Zones indicator
fib_time_zones = FibonacciTimeZones(
    fib_sequence=[1, 2, 3, 5, 8, 13, 21, 34],
    auto_detect_start=True,
    max_zones=5
)

# Calculate time zones
result = fib_time_zones.calculate(data)
```

### Fibonacci Circles

```python
from feature_store_service.indicators.fibonacci import FibonacciCircles

# Create a Fibonacci Circles indicator
fib_circles = FibonacciCircles(
    levels=[0.382, 0.5, 0.618, 1.0, 1.618],
    swing_lookback=30,
    auto_detect_points=True,
    projection_bars=20,
    time_price_ratio=1.0
)

# Calculate circle levels
result = fib_circles.calculate(data)
```

### Fibonacci Clusters

```python
from feature_store_service.indicators.fibonacci import FibonacciClusters

# First calculate various Fibonacci levels
data = fib_retracement.calculate(data)
data = fib_extension.calculate(data)

# Create a Fibonacci Clusters indicator
fib_clusters = FibonacciClusters(
    price_column='close',
    cluster_threshold=3,
    price_tolerance=0.5
)

# Calculate clusters
result = fib_clusters.calculate(data)
```

## Utility Functions

The package also provides utility functions for Fibonacci analysis:

```python
from feature_store_service.indicators.fibonacci import (
    generate_fibonacci_sequence,
    fibonacci_ratios,
    calculate_fibonacci_retracement_levels,
    calculate_fibonacci_extension_levels,
    is_golden_ratio,
    is_fibonacci_ratio
)

# Generate Fibonacci sequence
sequence = generate_fibonacci_sequence(10)

# Get standard Fibonacci ratios
ratios = fibonacci_ratios()

# Calculate retracement levels
levels = calculate_fibonacci_retracement_levels(100, 200)

# Check if a ratio is close to the golden ratio
is_golden = is_golden_ratio(0.62)
```

## Architecture

The package is organized into the following modules:

- `base.py`: Base classes and common functionality
- `retracements.py`: Fibonacci retracement implementation
- `extensions.py`: Fibonacci extension implementation
- `fans.py`: Fibonacci fan implementation
- `time_zones.py`: Fibonacci time zone implementation
- `circles.py`: Fibonacci circle implementation
- `clusters.py`: Fibonacci cluster implementation
- `utils.py`: Utility functions for Fibonacci analysis
- `facade.py`: Facade for all Fibonacci tools

All classes and functions are re-exported from the package root for easy access.

## Migration Guide

If you were using the original `fibonacci.py` module, you can continue to use it as before. The original module now imports from the new package structure, maintaining backward compatibility.

To migrate to the new package structure, simply update your imports:

```python
# Old import
from feature_store_service.indicators.fibonacci import FibonacciRetracement

# New import (same as old)
from feature_store_service.indicators.fibonacci import FibonacciRetracement
```

The new package structure provides better organization, more functionality, and improved performance.