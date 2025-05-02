# Technical Indicators Developer Documentation

This document provides detailed information about the internal structure, algorithms, and dependencies of the technical indicators implementation in the forex trading platform. It's intended for developers who need to maintain, extend, or optimize the indicator codebase.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Indicator Code Structure](#indicator-code-structure)
3. [Key Algorithms Explained](#key-algorithms-explained)
4. [Dependency Map](#dependency-map)
5. [Performance Considerations](#performance-considerations)
6. [Adding New Indicators](#adding-new-indicators)
7. [Testing and Validation](#testing-and-validation)
8. [Optimization Guidelines](#optimization-guidelines)

## Architecture Overview

The technical indicators are implemented as part of the feature-store-service, which provides calculation and storage of technical analysis features for the forex trading platform. The indicators follow a layered architecture:

```
┌─────────────────┐
│ Indicator APIs  │ <- External facing interfaces (moving_averages.py, oscillators.py, etc.)
├─────────────────┤
│ Base Indicators │ <- Core algorithm implementations (base_indicator.py)
├─────────────────┤
│ Data Handling   │ <- Input validation and preprocessing
└─────────────────┘
```

### Key Components

1. **Base Indicator** - Abstract base class that defines the common interface and functionality for all indicators
2. **Specialized Indicators** - Concrete implementations for specific indicator types
3. **Indicator Registry** - Central registry for indicator discovery and instantiation
4. **Adapters** - Components that convert between different data formats and interfaces
5. **Optimization Layers** - Performance enhancements for indicator calculations

### Data Flow

The typical flow for indicator calculation is:

1. Client requests an indicator with parameters
2. The indicator validates input data and parameters
3. The core algorithm is executed on the data
4. Results are post-processed (if needed)
5. Results are returned in a standardized format (usually a pandas DataFrame)

## Indicator Code Structure

### Directory Structure

```
feature_store_service/indicators/
├── __init__.py                     # Package initialization
├── base_indicator.py               # Base indicator class
├── moving_averages.py              # Simple, exponential, weighted MAs
├── oscillators.py                  # RSI, MACD, Stochastic
├── volatility.py                   # Bollinger Bands, ATR, etc.
├── volume.py                       # Volume-based indicators
├── fibonacci.py                    # Fibonacci retracement tools
├── advanced_price_indicators.py    # Complex price-based indicators
├── advanced_moving_averages.py     # KAMA, TEMA, etc.
├── advanced_oscillators.py         # Advanced momentum indicators
├── multi_timeframe.py              # Multi-timeframe indicators
├── statistical_regression_indicators.py # Statistical indicators
├── chart_patterns.py               # Chart pattern recognition
├── factory.py                      # Factory for creating indicators
├── indicator_registry.py           # Central registry
├── indicator_selection.py          # Selection algorithms
├── incremental/                    # Incremental calculation optimizations
│   ├── __init__.py
│   └── incremental_indicators.py
├── advanced/                       # Advanced indicator implementations
│   ├── __init__.py
│   └── advanced_indicators_registrar.py
└── testing/                        # Testing frameworks
    └── indicator_tester.py
```

### Class Hierarchy

```
BaseIndicator
├── MovingAverageIndicator
│   ├── SimpleMovingAverage
│   ├── ExponentialMovingAverage
│   └── WeightedMovingAverage
├── OscillatorIndicator
│   ├── RSI
│   ├── MACD
│   └── Stochastic
├── VolatilityIndicator
│   ├── BollingerBands
│   └── AverageTrueRange
└── VolumeIndicator
    ├── OnBalanceVolume
    └── VolumeProfile
```

### Key Files and Their Roles

- **base_indicator.py**: Defines the `BaseIndicator` abstract class that all indicators inherit from
- **factory.py**: Implements the factory pattern for creating indicator instances
- **indicator_registry.py**: Provides a central registry for all available indicators
- **incremental_indicators.py**: Optimized versions of indicators for incremental calculations

## Key Algorithms Explained

This section explains the core algorithms for key indicator types, focusing on implementation details that may not be obvious.

### Moving Averages

#### Simple Moving Average (SMA)

```python
def simple_moving_average(data, window=20, price_column='close'):
    """
    Implementation notes:
    - Uses pandas rolling function for efficient calculation
    - Handles NaN values at the beginning (window-1 periods)
    - Time complexity: O(n) where n is the number of data points
    - Memory complexity: O(n) for input and output arrays
    
    Internal algorithm:
    1. Validate input data and parameters
    2. Extract price series from dataframe
    3. Apply rolling window function
    4. Return result as DataFrame with same index as input
    """
    price_series = data[price_column]
    sma = price_series.rolling(window=window).mean()
    return pd.DataFrame(sma, index=data.index, columns=[f'sma_{window}'])
```

#### Exponential Moving Average (EMA)

```python
def exponential_moving_average(data, window=20, price_column='close', smoothing=2):
    """
    Implementation notes:
    - Uses pandas ewm function with adjustable smoothing factor
    - First window periods use SMA for initialization (industry standard)
    - Optimization: Pre-calculates weights to avoid redundant calculations
    
    Mathematical formula:
    EMA_today = (Price_today * k) + (EMA_yesterday * (1 - k))
    where k = smoothing / (1 + window)
    
    Corner cases handled:
    - Start of series initialization
    - Zero or negative window values
    - NaN or infinite values in price series
    """
    price_series = data[price_column]
    alpha = smoothing / (window + 1.0)  # The smoothing factor
    ema = price_series.ewm(alpha=alpha, min_periods=window).mean()
    return pd.DataFrame(ema, index=data.index, columns=[f'ema_{window}'])
```

### Oscillators

#### Relative Strength Index (RSI)

```python
def relative_strength_index(data, window=14, price_column='close'):
    """
    Implementation notes:
    - Uses specialized algorithm to avoid division by zero
    - Handles edge cases like all identical prices
    - Optimizes calculation by vectorizing operations
    
    Algorithm steps:
    1. Calculate price changes (deltas)
    2. Separate gains (positive) and losses (negative)
    3. Calculate average gain and average loss over window
    4. Calculate relative strength (avg_gain / avg_loss)
    5. Convert to RSI: 100 - (100 / (1 + RS))
    
    Performance bottlenecks:
    - Multiple rolling window calculations
    - Division operations (handled with numpy's safe division)
    """
    price_series = data[price_column]
    delta = price_series.diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return pd.DataFrame(rsi, index=data.index, columns=['rsi'])
```

### Advanced Algorithm: Kaufman's Adaptive Moving Average (KAMA)

This implementation has more complex logic worth explaining in detail:

```python
def kama(data, er_window=10, fast_window=2, slow_window=30, price_column='close'):
    """
    Implementation notes:
    - Dynamically adjusts smoothing based on market efficiency ratio
    - Uses custom calculation loop for precise control
    - Optimized by pre-calculating constants
    
    Algorithm steps:
    1. Calculate price change and volatility
    2. Calculate efficiency ratio (ER)
    3. Calculate smoothing constant using ER
    4. Apply KAMA formula recursively
    
    Mathematical basis:
    - Efficiency Ratio = |Price Change| / Volatility
    - Smoothing Constant = [ER * (fast_SC - slow_SC) + slow_SC]^2
    - KAMA_today = KAMA_yesterday + SC * (Price_today - KAMA_yesterday)
    
    Performance considerations:
    - Uses NumPy vectorized operations where possible
    - Falls back to iterative calculation where required
    """
    price = data[price_column].values
    n = len(price)
    er_window = int(er_window)
    
    # Calculate price change and volatility
    change = np.abs(price[er_window:] - price[:-er_window])
    volatility = np.sum([np.abs(price[i+1] - price[i]) for i in range(n-1)], axis=0)
    
    # Calculate efficiency ratio
    er = np.zeros(n)
    er[er_window:] = change / (volatility + np.finfo(float).eps)
    
    # Calculate smoothing constants
    sc_fast = 2.0 / (fast_window + 1.0)
    sc_slow = 2.0 / (slow_window + 1.0)
    
    # Initialize KAMA with first price
    kama_values = np.zeros(n)
    kama_values[0] = price[0]
    
    # Calculate KAMA values
    for i in range(1, n):
        # Calculate adaptive smoothing factor
        smooth = er[i] * (sc_fast - sc_slow) + sc_slow
        smooth_squared = smooth * smooth
        
        # Apply KAMA formula
        kama_values[i] = kama_values[i-1] + smooth_squared * (price[i] - kama_values[i-1])
    
    return pd.DataFrame(kama_values, index=data.index, columns=['kama'])
```

## Dependency Map

This section outlines the dependencies between different components of the indicator system.

### External Dependencies

The indicator system relies on these external libraries:

- **numpy**: For numerical operations and optimized array calculations
- **pandas**: For data manipulation and time series operations
- **scipy**: For advanced statistical functions
- **statsmodels**: For regression analysis and statistical testing
- **scikit-learn**: For machine learning-based indicators
- **matplotlib** (optional): For visualization capabilities

### Internal Dependencies

```
┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ base_indicator.py   │◄────┤ indicator_registry  │◄────┤ factory.py           │
└─────────────────────┘     └─────────────────────┘     └──────────────────────┘
         ▲                           ▲                           ▲
         │                           │                           │
         │                           │                           │
┌────────┴────────────┐     ┌────────┴────────────┐     ┌────────┴────────────┐
│ moving_averages.py  │     │ oscillators.py      │     │ volatility.py       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
         ▲                           ▲                           ▲
         │                           │                           │
         │                           │                           │
┌────────┴────────────┐     ┌────────┴────────────┐     ┌────────┴────────────┐
│ advanced_moving_    │     │ advanced_           │     │ incremental         │
│ averages.py         │     │ oscillators.py      │     │ indicators.py       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

### Dependency Descriptions

1. **Base Dependencies**:
   - All indicator modules depend on `base_indicator.py`
   - The `indicator_registry.py` maintains references to all registered indicators
   - `factory.py` depends on both to instantiate indicators

2. **Indicator Group Dependencies**:
   - Some advanced indicators depend on basic indicators (e.g., MACD depends on EMA)
   - Multi-timeframe indicators depend on their single timeframe counterparts
   - Pattern recognition depends on multiple indicator types

3. **Optimization Dependencies**:
   - Incremental indicators depend on their standard implementations
   - Parallelized indicators depend on the original algorithms

## Performance Considerations

### Memory Optimization

The indicator system includes several strategies to minimize memory usage:

1. **Lazy Calculation**: Indicators are calculated on-demand rather than pre-calculated
2. **Memory Reuse**: Input arrays are reused where possible to avoid copies
3. **Column Selection**: Only necessary columns are extracted from input DataFrames
4. **In-place Operations**: NumPy/Pandas in-place operations are used where appropriate

### CPU Optimization

For computational efficiency, the system employs:

1. **Vectorization**: Using NumPy/Pandas vectorized operations instead of loops
2. **Caching**: Intermediate results are cached for reuse in complex indicators
3. **Parallelization**: Heavy calculations are parallelized when beneficial
4. **Algorithm Selection**: Carefully selected algorithms for each indicator type

Example of vectorized calculation:

```python
# Inefficient loop-based calculation
def slow_bollinger(data, window=20, num_std=2):
    n = len(data)
    upper = np.zeros(n)
    lower = np.zeros(n)
    for i in range(window-1, n):
        segment = data[i-window+1:i+1]
        mean = segment.mean()
        std = segment.std()
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return upper, lower

# Optimized vectorized calculation
def fast_bollinger(data, window=20, num_std=2):
    rolling = data.rolling(window=window)
    mean = rolling.mean()
    std = rolling.std()
    upper = mean + num_std * std
    lower = mean - num_std * std
    return upper, lower
```

### Incremental Calculation

For streaming data scenarios, optimized incremental calculations are provided:

```python
def update_sma(prev_sma, prev_data, new_value, window):
    """
    Update SMA with a new value without recalculating the entire window.
    
    Args:
        prev_sma: Previous SMA value
        prev_data: Queue/array of previous window values
        new_value: New data point
        window: Window size
        
    Returns:
        (new_sma, updated_data)
    """
    # Remove oldest value and add new value
    oldest_value = prev_data[0]
    updated_data = np.append(prev_data[1:], new_value)
    
    # Update SMA efficiently
    new_sma = prev_sma + (new_value - oldest_value) / window
    
    return new_sma, updated_data
```

## Adding New Indicators

This section provides guidelines for adding new indicators to the system.

### Implementation Checklist

When adding a new indicator:

1. Decide which category it belongs to (moving average, oscillator, etc.)
2. Create class that inherits from appropriate base class
3. Implement required methods (`calculate`, `validate_params`, etc.)
4. Add unit tests covering normal cases and edge cases
5. Document the indicator with proper docstrings
6. Register the indicator with the indicator registry
7. Add to the API documentation

### Code Templates

#### New Indicator Template

```python
from feature_store_service.indicators.base_indicator import BaseIndicator

class NewIndicator(BaseIndicator):
    """
    Description of the new indicator.
    
    Parameters:
    -----------
    param1 : type
        Description of parameter 1
    param2 : type
        Description of parameter 2
        
    Attributes:
    ----------
    attr1 : type
        Description of attribute 1
        
    Methods:
    -------
    calculate(data):
        Calculate the indicator values
    """
    
    def __init__(self, param1=default1, param2=default2):
        super().__init__(name="NewIndicator")
        self.param1 = param1
        self.param2 = param2
        
    def validate_params(self):
        """Validate the parameters."""
        if self.param1 <= 0:
            raise ValueError("param1 must be positive")
        # Additional validation...
        
    def calculate(self, data):
        """
        Calculate the indicator.
        
        Parameters:
        ----------
        data : pd.DataFrame
            Input data with required columns
            
        Returns:
        -------
        pd.DataFrame
            Indicator values
        """
        self.validate_params()
        # Implementation of the indicator calculation
        # ...
        
        return result
        
    @classmethod
    def get_params_info(cls):
        """Return information about parameters."""
        return {
            "param1": {
                "type": "int",
                "description": "Description of param1",
                "default": default1
            },
            "param2": {
                "type": "float",
                "description": "Description of param2",
                "default": default2
            }
        }
```

#### Registering a New Indicator

```python
from feature_store_service.indicators.indicator_registry import register_indicator
from feature_store_service.indicators.your_module import YourIndicator

# Register the indicator
register_indicator('your_indicator_name', YourIndicator)
```

### Best Practices for New Indicators

1. **Consistent Interface**: Follow the established interface patterns
2. **Parameter Validation**: Validate all parameters before calculation
3. **Error Handling**: Provide clear error messages for invalid inputs
4. **Documentation**: Include thorough docstrings with examples
5. **Optimizations**: Consider performance optimizations from the start
6. **Testing**: Create comprehensive unit tests for accuracy and edge cases

## Testing and Validation

### Testing Framework

The indicator testing framework consists of:

1. **Unit Tests**: Test individual indicator calculations
2. **Integration Tests**: Test interactions between indicators
3. **Performance Tests**: Evaluate computational efficiency
4. **Edge Case Tests**: Verify behavior with extreme or invalid inputs

### IndicatorTester Usage

The `IndicatorTester` class provides a comprehensive testing framework:

```python
from feature_store_service.indicators.testing.indicator_tester import IndicatorTester
from feature_store_service.indicators.moving_averages import simple_moving_average
import pandas as pd
import numpy as np

# Create test data
data = pd.DataFrame({
    'close': np.random.random(100) * 100
})

# Initialize tester
tester = IndicatorTester(reference_data=data)

# Test accuracy against known good implementation
expected_results = pd.DataFrame({'sma': data['close'].rolling(window=10).mean()})
accuracy_results = tester.test_indicator_accuracy(
    indicator_func=simple_moving_average,
    expected_results=expected_results,
    params={'window': 10}
)

# Test performance with different parameters
performance_results = tester.benchmark_performance(
    indicator_func=simple_moving_average,
    params_list=[{'window': 10}, {'window': 20}, {'window': 50}],
    data_sizes=[100, 1000, 10000]
)

# Generate report
report = tester.generate_report('sma_test_report.md')
```

### Validation Guidelines

When validating a new indicator implementation:

1. Compare against industry-standard implementations
2. Test with real-world market data
3. Verify behavior in different market conditions (trending, volatile, ranging)
4. Check consistency with expected mathematical properties
5. Validate edge cases (e.g., insufficient data points, extreme values)

## Optimization Guidelines

This section provides guidance on optimizing indicators for performance.

### Profiling and Bottleneck Identification

Use the `IndicatorOptimizer` class to identify performance bottlenecks:

```python
from testing.indicator_optimizer import IndicatorOptimizer
from feature_store_service.indicators.oscillators import relative_strength_index
import pandas as pd

# Load test data
data = pd.read_csv('large_test_data.csv')

# Create optimizer
optimizer = IndicatorOptimizer()

# Identify bottlenecks
bottlenecks = optimizer.identify_bottlenecks(
    indicator_func=relative_strength_index,
    test_data=data,
    params={'window': 14}
)

print(f"Top bottlenecks: {bottlenecks['bottlenecks']}")
```

### Optimization Techniques

1. **Vectorization**: Convert loop-based calculations to vector operations
   ```python
   # Instead of:
   for i in range(len(data)):
       result[i] = some_calculation(data[i])
       
   # Use:
   result = vectorized_calculation(data)
   ```

2. **Caching**: Implement memoization for repeated calculations
   ```python
   # Add caching decorator
   @lru_cache(maxsize=128)
   def expensive_calculation(param1, param2):
       # ...calculation
       return result
   ```

3. **Parallelization**: Use multiprocessing for CPU-bound operations
   ```python
   # Use the optimizer's parallelization wrapper
   optimized_func = optimizer.optimize_with_parallelization(original_func)
   ```

4. **Algorithm Selection**: Choose appropriate algorithms based on data size
   ```python
   def adaptive_algorithm(data, params):
       if len(data) < 1000:
           return simple_algorithm(data, params)
       else:
           return complex_efficient_algorithm(data, params)
   ```

5. **Memory Management**: Minimize object creation and copying
   ```python
   # Instead of creating multiple intermediate DataFrames
   def optimized_calculation(data):
       # Pre-allocate result array
       result = np.zeros(len(data))
       # Fill result in-place
       # ...
       return pd.DataFrame(result, index=data.index, columns=['result'])
   ```

### Incremental Processing

For indicators that need to process streaming data efficiently:

1. Implement dedicated incremental update methods
2. Use sliding window implementations
3. Store minimum necessary state information
4. Avoid recalculating static values

Example with a stateful incremental RSI calculator:

```python
class IncrementalRSI:
    def __init__(self, window=14):
        self.window = window
        self.avg_gain = None
        self.avg_loss = None
        self.prev_price = None
        self.initialized = False
        self.price_history = []
        
    def initialize(self, data):
        """Initialize with historical data"""
        if len(data) < self.window + 1:
            raise ValueError(f"Need at least {self.window + 1} data points to initialize")
            
        # Calculate initial avg_gain and avg_loss
        price = data['close'].values
        deltas = np.diff(price)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        self.avg_gain = np.mean(gains[:self.window])
        self.avg_loss = np.mean(losses[:self.window])
        self.prev_price = price[-1]
        self.initialized = True
        
    def update(self, new_price):
        """Update RSI with a new price point"""
        if not self.initialized:
            self.price_history.append(new_price)
            if len(self.price_history) >= self.window + 1:
                # Convert history to DataFrame for initialization
                hist_df = pd.DataFrame({'close': self.price_history})
                self.initialize(hist_df)
                self.price_history = []
            return None
            
        # Calculate new gain/loss
        delta = new_price - self.prev_price
        gain = max(0, delta)
        loss = max(0, -delta)
        
        # Update averages
        self.avg_gain = ((self.avg_gain * (self.window - 1)) + gain) / self.window
        self.avg_loss = ((self.avg_loss * (self.window - 1)) + loss) / self.window
        
        # Update previous price
        self.prev_price = new_price
        
        # Calculate RSI
        if self.avg_loss == 0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
```

By following these guidelines and utilizing the provided tools, you can efficiently extend the indicators system while maintaining high performance and reliability.
