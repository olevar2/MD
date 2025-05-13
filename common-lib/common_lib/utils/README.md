# Utilities Module

This module provides common utilities for the Forex Trading Platform. It includes platform compatibility utilities, memory optimization for DataFrames, data export services, and parallel processing utilities.

## Key Components

1. **Platform Compatibility**: Utilities for platform-specific information and compatibility checks
2. **Memory Optimized DataFrame**: Memory-efficient wrapper for pandas DataFrame with lazy evaluation
3. **Export Service**: Utilities for exporting data to various formats
4. **Parallel Processor**: Utilities for parallel processing of tasks

## Platform Compatibility

The platform compatibility utilities provide information about the current platform and utilities for ensuring cross-platform compatibility.

```python
from common_lib.utils import PlatformInfo, PlatformCompatibility

# Get platform information
platform_info = PlatformInfo.get_platform_info()
print(f"Operating System: {platform_info['os']}")
print(f"Python Version: {platform_info['python_version']}")
print(f"Memory: {platform_info['memory']['total'] / (1024 * 1024 * 1024):.2f} GB")

# Check if a module is available
if PlatformCompatibility.is_module_available("torch"):
    import torch
    print("PyTorch is available")

# Get optimal thread count for parallel processing
thread_count = PlatformCompatibility.get_optimal_thread_count()
print(f"Optimal Thread Count: {thread_count}")
```

## Memory Optimized DataFrame

The memory optimized DataFrame provides a memory-efficient wrapper for pandas DataFrame with lazy evaluation and optimized data types.

```python
from common_lib.utils import MemoryOptimizedDataFrame
import pandas as pd

# Create a memory optimized DataFrame
df = MemoryOptimizedDataFrame(data={
    "id": range(1000),
    "value": [i * 0.1 for i in range(1000)],
    "category": ["A", "B", "C"] * 333 + ["A"]
})

# Add a computed column
df.add_computed_column("squared", lambda df: df["value"] ** 2)

# Access the computed column
print(df["squared"].head())

# Convert to pandas DataFrame
pandas_df = df.to_pandas()

# Check memory usage
print(f"Memory Usage: {df.total_memory_usage() / 1024:.2f} KB")
```

## Export Service

The export service provides utilities for exporting data to various formats.

```python
from common_lib.utils import convert_to_csv, convert_to_json, convert_to_parquet, convert_to_excel
import pandas as pd

# Create sample data
data = [
    {"id": 1, "name": "John", "value": 10.5},
    {"id": 2, "name": "Jane", "value": 20.7},
    {"id": 3, "name": "Bob", "value": 15.2}
]

# Convert to CSV
csv_data = convert_to_csv(data)
print(csv_data)

# Convert to JSON
json_data = convert_to_json(data, indent=2)
print(json_data)

# Convert to Parquet
parquet_data = convert_to_parquet(data)
with open("data.parquet", "wb") as f:
    f.write(parquet_data)

# Convert to Excel
excel_data = convert_to_excel(data)
with open("data.xlsx", "wb") as f:
    f.write(excel_data)
```

## Parallel Processor

The parallel processor provides utilities for parallel processing of tasks.

```python
from common_lib.utils import ParallelProcessor, Task, TaskPriority, ParallelizationMethod
import time

# Define a task function
def process_item(item):
    time.sleep(0.1)  # Simulate work
    return item * 2

# Create a parallel processor
processor = ParallelProcessor(max_workers=4)

# Process items in parallel
items = list(range(10))
results = processor.map(process_item, items)
print(results)

# Create tasks with priorities
tasks = [
    Task(func=process_item, args=(i,), priority=TaskPriority.HIGH if i < 5 else TaskPriority.LOW)
    for i in range(10)
]

# Execute tasks in parallel
results = processor.execute_tasks(tasks)
print(results)

# Close the processor
processor.close()
```

## Async Parallel Processing

The parallel processor also supports asynchronous parallel processing.

```python
import asyncio
from common_lib.utils import ParallelProcessor, Task, TaskPriority, ParallelizationMethod

# Define an async task function
async def process_item_async(item):
    await asyncio.sleep(0.1)  # Simulate async work
    return item * 2

async def main():
    # Create a parallel processor
    processor = ParallelProcessor(max_workers=4)
    
    # Process items in parallel asynchronously
    items = list(range(10))
    results = await processor.map_async(process_item_async, items)
    print(results)
    
    # Create async tasks with priorities
    tasks = [
        Task(func=process_item_async, args=(i,), priority=TaskPriority.HIGH if i < 5 else TaskPriority.LOW)
        for i in range(10)
    ]
    
    # Execute tasks in parallel asynchronously
    results = await processor.execute_tasks_async(tasks)
    print(results)
    
    # Close the processor
    processor.close()

# Run the async main function
asyncio.run(main())
```
