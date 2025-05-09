# Gann Analysis Tools

This package provides implementations of W.D. Gann's analytical methods including Gann angles, squares, fans, and other geometric tools.

## Overview

W.D. Gann (1878-1955) was a legendary trader who developed a unique approach to market analysis based on geometric, mathematical, and astronomical relationships. His methods combine price and time analysis to identify potential market turning points and trend directions.

### Key Concepts in Gann Analysis

1. **Price-Time Relationship**: Gann believed that price and time are interconnected, and markets move in predictable patterns when both dimensions are considered together.

2. **Geometric Angles**: Gann used angles (particularly the 1x1 or 45° angle) to determine the strength and direction of trends. These angles represent specific rates of price change over time.

3. **Square of 9**: A spiral arrangement of numbers used to identify important price levels and time cycles. The Square of 9 helps identify potential support, resistance, and reversal points.

4. **Natural Cycles**: Gann observed that markets tend to move in cycles based on natural numbers (especially 90, 180, 270, and 360) and their divisions.

5. **Fibonacci Relationships**: While not exclusive to Gann, his methods often incorporate Fibonacci ratios to identify potential reversal points.

### Applications in Trading

Gann tools can be used for:

- Identifying potential support and resistance levels
- Predicting potential market turning points
- Determining trend strength and direction
- Setting price targets and stop-loss levels
- Timing market entries and exits

## Directory Structure

```
feature-store-service/feature_store_service/indicators/gann/
├── __init__.py                # Package exports
├── base.py                    # Base classes and common utilities
├── angles.py                  # Gann angles implementation
├── fans.py                    # Gann fan implementation
├── squares/                   # Gann square implementations
│   ├── __init__.py
│   ├── square_of_9.py         # Square of 9 implementation
│   └── square_of_144.py       # Square of 144 implementation
├── time_cycles.py             # Gann time cycles implementation
├── grid.py                    # Gann grid implementation
├── box.py                     # Gann box implementation
├── hexagon.py                 # Gann hexagon implementation
└── facade.py                  # Facade for backward compatibility
```

## Available Tools

### Gann Angles

Gann Angles are diagonal lines that represent different rates of price movement over time. The primary Gann angle is the 1x1 (45°) line, which represents one unit of price for one unit of time.

```python
from feature_store_service.indicators.gann import GannAngles

# Initialize Gann Angles
gann_angles = GannAngles(
    pivot_type="swing_low",  # Use the lowest low as pivot
    angle_types=["1x1", "1x2", "2x1"],  # Specify which angles to calculate
    lookback_period=100,  # Look back 100 bars to find pivot
    price_scaling=1.0,  # Scale factor for price units
    projection_bars=50  # Project angles 50 bars into the future
)

# Calculate Gann Angles
result = gann_angles.calculate(data)

# Access angle values
angle_1x1_up = result["gann_angle_up_1x1"]  # 45° upward angle
angle_1x1_down = result["gann_angle_down_1x1"]  # 45° downward angle
angle_1x2_up = result["gann_angle_up_1x2"]  # 26.57° upward angle (1 price unit per 2 time units)
angle_2x1_up = result["gann_angle_up_2x1"]  # 63.43° upward angle (2 price units per 1 time unit)

# Find the pivot point
pivot_idx = result.loc[result["gann_angle_pivot_idx"]].index[0]
pivot_price = result.loc[pivot_idx, "gann_angle_pivot_price"]
print(f"Pivot point: {pivot_idx}, Price: {pivot_price}")
```

#### Understanding Gann Angles

Gann angles represent different rates of price movement over time:

| Angle Name | Degrees | Description |
|------------|---------|-------------|
| 1x8 | 7.5° | 1 unit of price per 8 units of time |
| 1x4 | 15° | 1 unit of price per 4 units of time |
| 1x3 | 18.75° | 1 unit of price per 3 units of time |
| 1x2 | 26.57° | 1 unit of price per 2 units of time |
| 1x1 | 45° | 1 unit of price per 1 unit of time |
| 2x1 | 63.43° | 2 units of price per 1 unit of time |
| 3x1 | 71.25° | 3 units of price per 1 unit of time |
| 4x1 | 75° | 4 units of price per 1 unit of time |
| 8x1 | 82.5° | 8 units of price per 1 unit of time |

The 1x1 (45°) angle is considered the most important and represents a balanced market. When price stays above the 1x1 angle, the market is considered strong; when it stays below, the market is considered weak.

#### Practical Usage

1. **Trend Strength**: If price stays above the 1x1 angle, the trend is strong. If it breaks below, the trend may be weakening.

2. **Support/Resistance**: Gann angles often act as support or resistance levels. When price approaches an angle, watch for potential bounces or breakouts.

3. **Price Targets**: Project Gann angles forward to identify potential price targets at specific dates.

4. **Trend Changes**: When price breaks through multiple Gann angles in succession, it may signal a significant trend change.

### Gann Fan

Gann Fan consists of a set of Gann angles drawn from a significant pivot point (high or low). These lines act as potential support and resistance levels.

```python
from feature_store_service.indicators.gann import GannFan

# Initialize Gann Fan
gann_fan = GannFan(
    pivot_type="swing_low",
    fan_angles=["1x8", "1x4", "1x2", "1x1", "2x1", "4x1", "8x1"],
    lookback_period=100,
    price_scaling=1.0,
    projection_bars=50
)

# Calculate Gann Fan
result = gann_fan.calculate(data)
```

### Gann Square

Gann Squares are used to identify potential support and resistance levels based on geometric relationships derived from price and time squares. The Square of 9 is the most common.

```python
from feature_store_service.indicators.gann import GannSquare

# Initialize Gann Square
gann_square = GannSquare(
    square_type="square_of_9",
    pivot_price=None,  # Auto-detect
    auto_detect_pivot=True,
    lookback_period=100,
    num_levels=4
)

# Calculate Gann Square
result = gann_square.calculate(data)

# Access Square of 9 levels
for i in range(1, 5):  # For each level (1-4)
    for angle in [45, 90, 135, 180]:  # For each angle
        support_level = result[f"gann_sq_sup_{angle}_{i}"].iloc[-1]
        resistance_level = result[f"gann_sq_res_{angle}_{i}"].iloc[-1]
        print(f"Level {i}, Angle {angle}°: Support = {support_level:.2f}, Resistance = {resistance_level:.2f}")

# Get the pivot price
pivot_price = result["gann_square_pivot_price"].iloc[-1]
print(f"Pivot price: {pivot_price:.2f}")
```

#### Understanding the Square of 9

The Square of 9 is a spiral of numbers arranged in a square format, starting with a central number (the pivot) and spiraling outward. Each complete rotation around the spiral represents a "square" of the central number.

Here's a simplified visualization of the Square of 9:

```
25 24 23 22 21
10  9  8  7 20
11  2  1  6 19
12  3  4  5 18
13 14 15 16 17
```

In this example, 1 is the central number, and the numbers spiral outward. The Square of 9 helps identify important price levels based on geometric relationships.

#### Key Angles on the Square of 9

The most important angles on the Square of 9 are:

| Angle | Degrees | Description |
|-------|---------|-------------|
| Cardinal | 0°, 90°, 180°, 270° | Major support/resistance levels |
| Diagonal | 45°, 135°, 225°, 315° | Secondary support/resistance levels |

#### Practical Usage

1. **Support/Resistance Levels**: The levels calculated from the Square of 9 often act as support or resistance in the market.

2. **Price Targets**: Use Square of 9 levels as potential price targets for trades.

3. **Reversal Points**: When price reaches a significant Square of 9 level, watch for potential reversals.

4. **Confluence**: Look for areas where multiple Square of 9 levels converge, as these can be particularly significant support/resistance zones.

### Gann Time Cycles

Identifies potential future turning points in time based on Gann's cycle theories, often using significant past highs or lows as starting points and projecting forward using specific time intervals.

```python
from feature_store_service.indicators.gann import GannTimeCycles

# Initialize Gann Time Cycles
gann_time_cycles = GannTimeCycles(
    cycle_lengths=[30, 60, 90, 120, 180, 270, 360],
    starting_point_type="major_low",
    lookback_period=200,
    auto_detect_start=True,
    max_cycles=5
)

# Calculate Gann Time Cycles
result = gann_time_cycles.calculate(data)
```

### Advanced Gann Tools

The package also includes more advanced Gann tools:

- **GannGrid**: Overlays a grid on the chart based on a significant pivot point.
- **GannBox**: Draws a box between two points with price and time divisions.
- **GannSquare144**: Calculates levels based on the Square of 144 concept.
- **GannHexagon**: Calculates points based on hexagonal geometry.

## Parameter Tuning and Performance Considerations

### Parameter Tuning

When using Gann tools, consider the following parameter tuning guidelines:

1. **Pivot Selection**:
   - For uptrends, use significant lows as pivot points
   - For downtrends, use significant highs as pivot points
   - In ranging markets, use both high and low pivots to create a range

2. **Price Scaling**:
   - Adjust the `price_scaling` parameter based on the instrument's volatility
   - Higher volatility instruments may require larger scaling factors
   - Lower volatility instruments may require smaller scaling factors

3. **Lookback Period**:
   - Use longer lookback periods (200+ bars) for longer-term analysis
   - Use shorter lookback periods (50-100 bars) for shorter-term analysis
   - Ensure the lookback period captures at least one significant market cycle

4. **Angle Selection**:
   - The 1x1 (45°) angle is the most important and should always be included
   - Include steeper angles (2x1, 3x1) for volatile markets
   - Include shallower angles (1x2, 1x3) for less volatile markets

### Performance Considerations

Gann tools can be computationally intensive, especially when calculating multiple tools on large datasets. Consider the following performance optimizations:

1. **Selective Calculation**:
   - Only calculate the specific Gann tools and angles you need
   - Limit the number of levels and projections to what's necessary

2. **Caching Results**:
   - Cache calculation results when possible, especially for static pivot points
   - Recalculate only when new data is available or parameters change

3. **Batch Processing**:
   - Process data in batches when dealing with very large datasets
   - Consider downsampling data for initial analysis, then refine on full data

4. **Parallel Processing**:
   - For heavy calculations, consider using parallel processing
   - This is especially useful for calculating multiple Gann tools simultaneously

## Backward Compatibility

For backward compatibility with the original `gann_tools.py` module, you can import from the facade:

```python
from feature_store_service.indicators.gann_tools import GannAngles, GannFan, GannSquare
```

However, for new code, it is recommended to import directly from the `gann` package:

```python
from feature_store_service.indicators.gann import GannAngles, GannFan, GannSquare
```

### Migration Guide

If you're transitioning from the original `gann_tools.py` module to the new structure, follow these steps:

1. **Update Import Statements**:
   ```python
   # Old import
   from feature_store_service.indicators.gann_tools import GannAngles

   # New import
   from feature_store_service.indicators.gann import GannAngles
   ```

2. **Review Parameter Names**:
   - Some parameter names may have changed for clarity
   - Check the docstrings for each class for the updated parameter names

3. **Update Method Calls**:
   - The core calculation methods remain the same
   - Additional utility methods may be available in the new implementation

4. **Test Thoroughly**:
   - Verify that your calculations produce the same results with the new implementation
   - Run tests with both implementations to ensure consistency
