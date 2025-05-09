# Gann Tools Migration Guide

This guide helps you migrate from the original `gann_tools.py` module to the new modular structure.

## Overview

The Gann tools have been refactored from a single monolithic file into a modular structure with separate files for each Gann tool. This improves maintainability, extensibility, and readability of the code.

## Backward Compatibility

For backward compatibility, you can continue to import from the original module:

```python
from feature_store_service.indicators.gann_tools import GannAngles, GannFan, GannSquare
```

However, for new code, it is recommended to import directly from the new structure:

```python
from feature_store_service.indicators.gann import GannAngles, GannFan, GannSquare
```

## API Changes

### Class Name Changes

Some class names have been updated for clarity:

| Original Name | New Name |
|---------------|----------|
| GannSquare9   | GannSquare |

### Method Name Changes

Some method names have been standardized:

| Original Method | New Method |
|-----------------|------------|
| calculate_angles | calculate |
| calculate_fan | calculate |
| calculate_levels | calculate |

### Parameter Changes

Some parameters have been renamed or added:

#### GannAngles

| Original Parameter | New Parameter | Notes |
|-------------------|---------------|-------|
| pivot_date | N/A | Replaced by pivot_type and lookback_period |
| pivot_price | N/A | Replaced by pivot_type and lookback_period |
| auto_detect_pivot | auto_detect_pivot | Same functionality |
| N/A | pivot_type | New parameter for specifying pivot type |
| N/A | lookback_period | New parameter for specifying lookback period |
| N/A | price_scaling | New parameter for scaling price units |
| N/A | projection_bars | New parameter for projecting angles forward |

#### GannFan

| Original Parameter | New Parameter | Notes |
|-------------------|---------------|-------|
| pivot_date | N/A | Replaced by pivot_type and lookback_period |
| pivot_price | N/A | Replaced by pivot_type and lookback_period |
| auto_detect_pivot | auto_detect_pivot | Same functionality |
| N/A | pivot_type | New parameter for specifying pivot type |
| N/A | lookback_period | New parameter for specifying lookback period |
| N/A | price_scaling | New parameter for scaling price units |
| N/A | projection_bars | New parameter for projecting fan lines forward |

#### GannSquare

| Original Parameter | New Parameter | Notes |
|-------------------|---------------|-------|
| base_price | pivot_price | Renamed for consistency |
| N/A | square_type | New parameter for specifying square type |
| N/A | auto_detect_pivot | New parameter for auto-detecting pivot |
| N/A | lookback_period | New parameter for specifying lookback period |
| N/A | num_levels | New parameter for specifying number of levels |

## Migration Steps

### Step 1: Update Import Statements

```python
# Old import
from feature_store_service.indicators.gann_tools import GannAngles, GannFan, GannSquare9

# New import
from feature_store_service.indicators.gann import GannAngles, GannFan, GannSquare
```

### Step 2: Update Class Instantiation

```python
# Old instantiation
gann_angles = GannAngles(pivot_price=100.0, pivot_date=some_date)

# New instantiation
gann_angles = GannAngles(
    pivot_type="manual",
    manual_pivot_price=100.0,
    manual_pivot_time_idx=some_idx,
    auto_detect_pivot=False
)
```

### Step 3: Update Method Calls

```python
# Old method call
angles = gann_angles.calculate_angles(data)

# New method call
result = gann_angles.calculate(data)
angles_1x1 = result["gann_angle_up_1x1"]
```

### Step 4: Update Result Handling

The new implementation returns a DataFrame with all calculated values as columns, rather than separate dictionaries or lists.

```python
# Old result handling
for angle_name, angle_values in angles.items():
    # Process angle values
    pass

# New result handling
for angle_type in ["1x1", "1x2", "2x1"]:
    up_values = result[f"gann_angle_up_{angle_type}"]
    down_values = result[f"gann_angle_down_{angle_type}"]
    # Process angle values
    pass
```

## Example Migrations

### Example 1: GannAngles

```python
# Old code
from feature_store_service.indicators.gann_tools import GannAngles

gann_angles = GannAngles(pivot_price=100.0, pivot_date=some_date)
angles = gann_angles.calculate_angles(data)

# New code
from feature_store_service.indicators.gann import GannAngles

gann_angles = GannAngles(
    pivot_type="manual",
    manual_pivot_price=100.0,
    manual_pivot_time_idx=data.index.get_loc(some_date),
    auto_detect_pivot=False,
    angle_types=["1x1", "1x2", "2x1"]
)
result = gann_angles.calculate(data)
```

### Example 2: GannFan

```python
# Old code
from feature_store_service.indicators.gann_tools import GannFan

gann_fan = GannFan(pivot_price=100.0, pivot_date=some_date)
fan_lines = gann_fan.calculate_fan(data)

# New code
from feature_store_service.indicators.gann import GannFan

gann_fan = GannFan(
    pivot_type="manual",
    manual_pivot_price=100.0,
    manual_pivot_time_idx=data.index.get_loc(some_date),
    auto_detect_pivot=False,
    fan_angles=["1x1", "1x2", "2x1", "1x4", "4x1"]
)
result = gann_fan.calculate(data)
```

### Example 3: GannSquare

```python
# Old code
from feature_store_service.indicators.gann_tools import GannSquare9

gann_square = GannSquare9(base_price=100.0)
levels = gann_square.calculate_levels(n_levels=4)

# New code
from feature_store_service.indicators.gann import GannSquare

gann_square = GannSquare(
    square_type="square_of_9",
    pivot_price=100.0,
    auto_detect_pivot=False,
    num_levels=4
)
result = gann_square.calculate(data)
```

## Advanced Features

The new implementation includes several advanced features not available in the original implementation:

1. **Error Handling**: Improved error handling with detailed error messages and graceful fallbacks.

2. **Parameter Validation**: Comprehensive parameter validation to catch errors early.

3. **Pivot Detection**: More sophisticated pivot point detection with multiple pivot types.

4. **Visualization**: Built-in visualization utilities for all Gann tools.

5. **Documentation**: Comprehensive documentation with examples and usage guidelines.

## Getting Help

If you encounter any issues during migration, please refer to the following resources:

1. **README.md**: Comprehensive documentation with examples and usage guidelines.

2. **Docstrings**: Detailed docstrings for all classes and methods.

3. **Examples**: Example code in the documentation.

4. **Tests**: Test cases demonstrating proper usage.

## Conclusion

The new modular structure provides a more maintainable, extensible, and user-friendly implementation of Gann tools. While backward compatibility is maintained, it is recommended to migrate to the new structure for new code to take advantage of the improved features and documentation.
