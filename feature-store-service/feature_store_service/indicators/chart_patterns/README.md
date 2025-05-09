# Chart Patterns Module

This module provides implementations of various chart pattern recognition algorithms for technical analysis.

## Overview

The chart patterns module has been refactored to improve maintainability, extensibility, and performance. The original monolithic implementation has been split into smaller, more focused components organized by pattern type.

## Directory Structure

```
chart_patterns/
├── __init__.py                  # Exports the facade classes
├── base.py                      # Common base classes and utilities
├── classic/                     # Classic chart patterns
│   ├── __init__.py
│   ├── head_and_shoulders.py    # Head and Shoulders pattern
│   ├── double_formations.py     # Double Top/Bottom patterns
│   ├── triple_formations.py     # Triple Top/Bottom patterns
│   ├── triangles.py             # Triangle patterns (ascending, descending, symmetric)
│   ├── flags_pennants.py        # Flag and Pennant patterns
│   ├── wedges.py                # Wedge patterns (rising, falling)
│   └── rectangles.py            # Rectangle patterns
├── harmonic/                    # Harmonic patterns
│   ├── __init__.py
│   ├── gartley.py               # Gartley pattern
│   ├── butterfly.py             # Butterfly pattern
│   ├── bat.py                   # Bat pattern
│   └── crab.py                  # Crab pattern
├── candlestick/                 # Candlestick patterns
│   ├── __init__.py
│   ├── base.py                  # Base class for candlestick patterns
│   ├── doji.py                  # Doji pattern
│   ├── hammer.py                # Hammer and Hanging Man patterns
│   └── engulfing.py             # Engulfing pattern
├── visualization.py             # Visualization utilities for patterns
└── facade.py                    # Facade that maintains the original API
```

## Usage

The module provides three main classes for pattern recognition:

1. `ChartPatternRecognizer`: Identifies classic chart patterns like Head and Shoulders, Double Tops/Bottoms, etc.
2. `HarmonicPatternFinder`: Identifies harmonic patterns like Gartley, Butterfly, Bat, etc.
3. `CandlestickPatterns`: Identifies candlestick patterns like Doji, Hammer, Engulfing, etc.

### Examples

#### Classic Chart Patterns

```python
from feature_store_service.indicators.chart_patterns import ChartPatternRecognizer

# Initialize the recognizer
recognizer = ChartPatternRecognizer(
    lookback_period=100,
    pattern_types=["head_and_shoulders", "double_top", "triangle"],
    min_pattern_size=10,
    max_pattern_size=50,
    sensitivity=0.75
)

# Calculate patterns
result = recognizer.calculate(data)

# Access pattern columns
head_and_shoulders = result["pattern_head_and_shoulders"]
double_top = result["pattern_double_top"]
triangle = result["pattern_triangle"]

# Check if any patterns were detected
has_pattern = result["has_pattern"]

# Get pattern strength
pattern_strength = result["pattern_strength"]
```

#### Harmonic Patterns

```python
from feature_store_service.indicators.chart_patterns import HarmonicPatternFinder

# Initialize the harmonic pattern finder
finder = HarmonicPatternFinder(
    lookback_period=100,
    pattern_types=["gartley", "butterfly", "bat", "crab"],
    tolerance=0.05
)

# Calculate patterns
result = finder.calculate(data)

# Access pattern columns
gartley_bullish = result["harmonic_gartley_bullish"]
gartley_bearish = result["harmonic_gartley_bearish"]
butterfly_bullish = result["harmonic_butterfly_bullish"]

# Find detailed pattern information
patterns = finder.find_patterns(data)
gartley_patterns = patterns["gartley"]  # List of Gartley patterns with points and ratios
```

#### Candlestick Patterns

```python
from feature_store_service.indicators.chart_patterns import CandlestickPatterns

# Initialize the candlestick pattern recognizer
candlestick = CandlestickPatterns(
    pattern_types=["doji", "hammer", "engulfing"]
)

# Calculate patterns
result = candlestick.calculate(data)

# Access pattern columns
doji = result["candle_doji"]
hammer_bullish = result["candle_hammer_bullish"]
engulfing_bearish = result["candle_engulfing_bearish"]

# Find detailed pattern information
patterns = candlestick.find_patterns(data)
doji_patterns = patterns["doji"]  # List of Doji patterns
```

#### Visualization

```python
from feature_store_service.indicators.chart_patterns import ChartPatternRecognizer
from feature_store_service.indicators.chart_patterns.visualization import plot_chart_with_patterns

# Initialize the recognizer and find patterns
recognizer = ChartPatternRecognizer()
result = recognizer.calculate(data)
patterns = recognizer.find_patterns(data)

# Plot chart with patterns
fig = plot_chart_with_patterns(data, patterns, title="Chart Patterns")

# Show the figure
import matplotlib.pyplot as plt
plt.show()
```

## Backward Compatibility

The refactored implementation maintains backward compatibility with the original API through facade classes. The original methods like `find_patterns()` are still available, but they now use the new implementation internally.

## Implemented Features

1. **Classic Chart Patterns**:
   - Head and Shoulders (and Inverse)
   - Double Top/Bottom
   - Triple Top/Bottom
   - Triangle (Ascending, Descending, Symmetric)
   - Flag and Pennant
   - Wedge (Rising, Falling)
   - Rectangle

2. **Harmonic Patterns**:
   - Gartley
   - Butterfly
   - Bat
   - Crab

3. **Candlestick Patterns**:
   - Doji (Standard, Long-Legged, Dragonfly, Gravestone)
   - Hammer and Hanging Man
   - Engulfing (Bullish, Bearish)

4. **Visualization Utilities**:
   - Chart with patterns
   - Harmonic pattern with Fibonacci ratios
   - Candlestick pattern with annotations

## Future Work

1. Implement additional harmonic patterns:
   - Shark
   - Cypher
   - Three Drives

2. Implement additional candlestick patterns:
   - Morning Star/Evening Star
   - Three White Soldiers/Three Black Crows
   - Harami
   - Piercing Line/Dark Cloud Cover
   - Spinning Top

3. Add more pattern types to the classic patterns module

4. Improve pattern detection accuracy:
   - Add machine learning-based pattern recognition
   - Implement adaptive parameters based on market conditions

5. Add performance benchmarking and optimization
