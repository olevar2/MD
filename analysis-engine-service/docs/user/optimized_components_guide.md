# Optimized Components User Guide

This guide provides detailed information on how to use the optimized components in the forex trading platform.

## Table of Contents

1. [Introduction](#introduction)
2. [OptimizedConfluenceDetector](#optimizedconfluencedetector)
3. [Memory Optimization](#memory-optimization)
4. [Caching Strategies](#caching-strategies)
5. [Parallel Processing](#parallel-processing)
6. [GPU Acceleration](#gpu-acceleration)
7. [Distributed Tracing](#distributed-tracing)
8. [Predictive Caching](#predictive-caching)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

## Introduction

The optimized components in the forex trading platform provide significant performance improvements over the original implementations. These components are designed to be drop-in replacements for the original components, with the same API but better performance characteristics.

Key benefits include:

- 2.5-3x speedup for both confluence and divergence detection
- 60% reduction in memory usage
- Adaptive thread pool and caching for handling large datasets
- Distributed tracing for performance monitoring and troubleshooting
- Optional GPU acceleration for technical indicator calculations

## OptimizedConfluenceDetector

The `OptimizedConfluenceDetector` is a drop-in replacement for the `RelatedPairsConfluenceAnalyzer` with significant performance improvements.

### Basic Usage

```python
from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer

# Create detector
detector = OptimizedConfluenceDetector(
    correlation_service=correlation_service,
    currency_strength_analyzer=CurrencyStrengthAnalyzer(),
    correlation_threshold=0.7,
    lookback_periods=20,
    cache_ttl_minutes=60,
    max_workers=4
)

# Detect confluence
result = detector.detect_confluence_optimized(
    symbol="EURUSD",
    price_data=price_data,
    signal_type="trend",
    signal_direction="bullish"
)

# Analyze divergence
divergence = detector.analyze_divergence_optimized(
    symbol="EURUSD",
    price_data=price_data
)
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| correlation_threshold | Minimum correlation for related pairs | 0.7 |
| lookback_periods | Number of periods to look back for analysis | 20 |
| cache_ttl_minutes | Cache time-to-live in minutes | 60 |
| max_workers | Maximum number of parallel workers | 4 |

### Performance Comparison

| Operation | Original | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Confluence Detection (cold) | ~250ms | ~85ms | ~66% faster |
| Confluence Detection (warm) | ~120ms | ~15ms | ~87% faster |
| Divergence Detection (cold) | ~180ms | ~65ms | ~64% faster |
| Divergence Detection (warm) | ~90ms | ~12ms | ~87% faster |
| Memory Usage (confluence) | ~45MB | ~18MB | ~60% reduction |
| Memory Usage (divergence) | ~38MB | ~15MB | ~61% reduction |

## Memory Optimization

The `MemoryOptimizedDataFrame` provides memory-efficient operations on pandas DataFrames.

### Basic Usage

```python
from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame

# Create memory-optimized DataFrame
df = price_data["EURUSD"]
optimized_df = MemoryOptimizedDataFrame(df)

# Optimize data types
optimized_df.optimize_dtypes()

# Add computed column
def compute_typical_price(df):
    return (df['high'] + df['low'] + df['close']) / 3

optimized_df.add_computed_column('typical_price', compute_typical_price)

# Get view of specific columns and rows
view = optimized_df.get_view(columns=['open', 'close'], rows=slice(0, 10))
```

### Key Features

- Automatic data type optimization to reduce memory usage
- Lazy evaluation of computed columns
- Efficient views without copying data
- Transparent access to underlying DataFrame methods

## Caching Strategies

The platform provides two caching implementations:

1. `AdaptiveCacheManager`: Basic caching with adaptive TTL
2. `PredictiveCacheManager`: Advanced caching with predictive precomputation

### AdaptiveCacheManager

```python
from analysis_engine.utils.adaptive_cache_manager import AdaptiveCacheManager

# Create cache manager
cache = AdaptiveCacheManager(
    default_ttl_seconds=300,
    max_size=1000,
    cleanup_interval_seconds=60,
    adaptive_ttl=True
)

# Set value
cache.set("key1", "value1")

# Get value
hit, value = cache.get("key1")

# Get stats
stats = cache.get_stats()
```

### PredictiveCacheManager

```python
from analysis_engine.utils.predictive_cache_manager import PredictiveCacheManager

# Create predictive cache manager
cache = PredictiveCacheManager(
    default_ttl_seconds=300,
    max_size=1000,
    prediction_threshold=0.7,
    max_precompute_workers=2
)

# Register precomputation function
def precompute_value(key):
    # Expensive computation
    return result

cache.register_precomputation_function(
    key_pattern="confluence_",
    function=precompute_value,
    priority=0
)

# Set and get values (same API as AdaptiveCacheManager)
cache.set("key1", "value1")
hit, value = cache.get("key1")
```

## Parallel Processing

The `OptimizedParallelProcessor` provides efficient parallel processing with adaptive thread pool sizing.

### Basic Usage

```python
from analysis_engine.utils.optimized_parallel_processor import OptimizedParallelProcessor

# Create processor
processor = OptimizedParallelProcessor(
    min_workers=2,
    max_workers=4
)

# Define tasks
def task1(x):
    return x * 2

def task2(x):
    return x * 3

tasks = [
    (0, task1, (1,)),  # (priority, function, args)
    (0, task2, (2,))
]

# Process tasks
results = processor.process(tasks, timeout=5.0)
```

### Key Features

- Priority-based task scheduling
- Adaptive thread pool sizing
- Timeout support
- Exception handling

## GPU Acceleration

The `GPUAccelerator` provides GPU acceleration for technical indicator calculations.

### Basic Usage

```python
from analysis_engine.utils.gpu_accelerator import GPUAccelerator

# Create accelerator
accelerator = GPUAccelerator(
    enable_gpu=True,
    memory_limit_mb=1024,
    batch_size=1000
)

# Calculate technical indicators
indicators = ["sma", "ema", "rsi"]
parameters = {
    "sma": {"period": 14},
    "ema": {"period": 14},
    "rsi": {"period": 14}
}

results = accelerator.calculate_technical_indicators(
    price_data,
    indicators,
    parameters
)
```

### Supported Indicators

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands

## Distributed Tracing

The `DistributedTracer` provides distributed tracing capabilities for performance monitoring.

### Basic Usage

```python
from analysis_engine.utils.distributed_tracing import DistributedTracer

# Create tracer
tracer = DistributedTracer(
    service_name="analysis-engine",
    enable_tracing=True,
    sampling_rate=0.1,
    otlp_endpoint="http://jaeger-collector:4317"
)

# Use context manager for spans
with tracer.start_span("operation_name") as span:
    # Add attributes
    span.set_attribute("attribute_name", "attribute_value")
    
    # Add event
    tracer.add_span_event("event_name")
    
    # Perform operation
    result = perform_operation()

# Use decorator for functions
@tracer.trace()
def traced_function(arg1, arg2):
    return arg1 + arg2
```

### Key Features

- OpenTelemetry integration
- Automatic context propagation
- Span creation and management
- Trace sampling and filtering

## Predictive Caching

The `PredictiveCacheManager` provides predictive caching capabilities for anticipating future requests.

### Access Pattern Analysis

The predictive cache manager analyzes access patterns to predict future requests:

1. When a cache key is accessed, it records the access in the access history.
2. It builds a model of which keys are likely to be accessed after a given key.
3. When a key is accessed, it precomputes the values for keys that are likely to be accessed next.

### Precomputation Functions

To enable predictive caching, you need to register precomputation functions:

```python
# Register precomputation function for confluence detection
def precompute_confluence(key):
    # Parse key to get parameters
    parts = key.split("_")
    symbol = parts[1]
    signal_type = parts[2]
    signal_direction = parts[3]
    
    # Perform computation
    result = detector.detect_confluence_optimized(
        symbol=symbol,
        price_data=price_data,
        signal_type=signal_type,
        signal_direction=signal_direction
    )
    
    return result

cache.register_precomputation_function(
    key_pattern="confluence_",
    function=precompute_confluence,
    priority=0
)
```

## Performance Tuning

### Configuration Guidelines

| Component | Parameter | Recommended Value | Notes |
|-----------|-----------|------------------|-------|
| OptimizedConfluenceDetector | max_workers | CPU cores | Adjust based on workload |
| OptimizedConfluenceDetector | cache_ttl_minutes | 60 | Adjust based on data update frequency |
| AdaptiveCacheManager | max_size | 1000-10000 | Adjust based on memory availability |
| PredictiveCacheManager | prediction_threshold | 0.7 | Higher values reduce false positives |
| GPUAccelerator | batch_size | 1000 | Adjust based on GPU memory |
| DistributedTracer | sampling_rate | 0.1 | Adjust based on traffic volume |

### Memory Optimization Tips

1. Use `MemoryOptimizedDataFrame.optimize_dtypes()` to reduce memory usage.
2. Limit the number of cached items with appropriate `max_size` settings.
3. Use views instead of copies when possible.
4. Consider using GPU acceleration for large datasets.

### CPU Optimization Tips

1. Adjust `max_workers` based on available CPU cores and workload.
2. Use priority-based task scheduling for critical operations.
3. Implement proper caching to avoid redundant calculations.
4. Consider using GPU acceleration for compute-intensive operations.

## Troubleshooting

### Common Issues

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| High memory usage | Too many cached items | Reduce cache max_size |
| | Large DataFrames | Use optimize_dtypes() |
| High CPU usage | Too many parallel workers | Reduce max_workers |
| | Inefficient algorithms | Use optimized implementations |
| Slow response times | Cold cache | Implement predictive caching |
| | Inefficient data structures | Use memory-optimized data structures |
| GPU errors | CUDA not available | Fall back to CPU implementation |
| | Insufficient GPU memory | Reduce batch size |

### Logging and Monitoring

The platform provides comprehensive logging and monitoring capabilities:

1. Use the distributed tracer to identify performance bottlenecks.
2. Monitor cache hit rates to optimize caching strategies.
3. Use Prometheus metrics to track resource usage and performance.
4. Set up alerts for performance degradation.

### Getting Help

If you encounter issues with the optimized components, please:

1. Check the documentation and this guide.
2. Look for similar issues in the GitHub repository.
3. Open a new issue with detailed information about the problem.
4. Contact support at support@forex-platform.com.
