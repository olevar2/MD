# Async Patterns in Analysis Engine Service

## Overview

This document describes the standardized asynchronous programming patterns used throughout the Analysis Engine Service. These patterns ensure consistent, efficient, and maintainable code when dealing with asynchronous operations.

## Key Async Patterns

### 1. Async Service Methods

All service methods that perform I/O operations (database queries, API calls, file operations) should be implemented as async methods using Python's `async/await` syntax.

```python
class AnalysisService:
    async def analyze(self, data: MarketData) -> AnalysisResult:
        """
        Perform analysis asynchronously.
        
        Args:
            data: Market data to analyze
            
        Returns:
            Analysis result
        """
        # Asynchronous implementation
        result = await self._perform_analysis(data)
        return result
```

### 2. Async Analyzer Components

All analyzer components implement async methods for their core functionality:

```python
class BaseAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, data: Any) -> AnalysisResult:
        """
        Analyze the provided data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Analysis result
        """
        pass
        
    async def update_incremental(self, data: Any, previous_result: AnalysisResult) -> AnalysisResult:
        """
        Update analysis incrementally with new data.
        
        Args:
            data: New data
            previous_result: Previous analysis result
            
        Returns:
            Updated analysis result
        """
        # Default implementation falls back to full analysis
        return await self.analyze(data)
```

### 3. Async Schedulers

Background tasks and schedulers use asyncio instead of threading:

```python
class AsyncScheduler:
    def __init__(self):
        self.running = False
        self.task = None
        self.scheduled_tasks = {}
        
    async def start(self):
        """Start the scheduler."""
        self.running = True
        self.task = asyncio.create_task(self._run_scheduler())
        
    async def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
                
    async def _run_scheduler(self):
        """Run the scheduler loop."""
        try:
            while self.running:
                now = datetime.now().timestamp()
                
                # Check each scheduled task
                for task_name, task_info in self.scheduled_tasks.items():
                    if now >= task_info["next_run"]:
                        # Run the task
                        asyncio.create_task(task_info["func"]())
                        
                        # Update next run time
                        task_info["next_run"] = now + task_info["interval"]
                
                # Sleep before checking again
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise
```

### 4. Async API Endpoints

All FastAPI endpoints are implemented as async functions:

```python
@router.post("/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    timeframe: str = Query("1h"),
    db: Session = Depends(get_db_session)
):
    """Analyze a symbol with the specified timeframe."""
    try:
        service = AnalysisService(db)
        result = await service.analyze_symbol(symbol, timeframe)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
```

### 5. Service Container Lifecycle

The service container manages async initialization and cleanup:

```python
class ServiceContainer:
    async def initialize(self):
        """Initialize all services."""
        for service in self._services.values():
            if hasattr(service, 'initialize'):
                await service.initialize()
                
    async def cleanup(self):
        """Clean up all services."""
        for service in self._services.values():
            if hasattr(service, 'cleanup'):
                await service.cleanup()
```

## Best Practices

### 1. Consistent Async/Await Usage

- Use `async def` for all functions that perform I/O operations
- Always `await` async functions
- Don't mix sync and async code in the same function without proper bridging

### 2. Task Management

- Use `asyncio.create_task()` for background tasks
- Always handle task cancellation with try/except for `asyncio.CancelledError`
- Store references to long-running tasks to prevent garbage collection

### 3. Error Handling

- Use try/except blocks around await expressions that might fail
- Propagate errors with appropriate context
- Log errors before re-raising or transforming them

### 4. Resource Management

- Use async context managers (`async with`) for managing resources
- Ensure proper cleanup of resources in finally blocks
- Consider using `asynccontextmanager` for custom resource management

### 5. Testing

- Use `pytest.mark.asyncio` for testing async functions
- Mock async dependencies with AsyncMock
- Test both success and error paths

## Migration from Threading to Asyncio

When migrating code from threading to asyncio:

1. Replace `threading.Thread` with `asyncio.create_task()`
2. Convert blocking functions to async functions
3. Replace `time.sleep()` with `await asyncio.sleep()`
4. Replace thread synchronization primitives with asyncio equivalents
5. Update thread lifecycle management to task lifecycle management

## Performance Considerations

- Avoid CPU-bound operations in async functions
- Use `asyncio.gather()` for parallel execution of independent tasks
- Consider using thread pools (`loop.run_in_executor()`) for CPU-bound operations
- Monitor task execution time and resource usage

## Examples

### Parallel Analysis

```python
async def analyze_multiple(data_list: List[MarketData]) -> List[AnalysisResult]:
    """Analyze multiple data sets in parallel."""
    tasks = [analyze_single(data) for data in data_list]
    results = await asyncio.gather(*tasks)
    return results
```

### Scheduled Task

```python
async def schedule_daily_report():
    """Schedule a daily report generation task."""
    scheduler = ReportScheduler()
    await scheduler.start()
    
    # Run for a day
    await asyncio.sleep(86400)
    
    # Stop the scheduler
    await scheduler.stop()
```

### Error Handling

```python
async def fetch_and_analyze(symbol: str):
    """Fetch data and analyze it with proper error handling."""
    try:
        data = await fetch_market_data(symbol)
        result = await analyze_data(data)
        return result
    except DataFetchError as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        raise AnalysisError(f"Analysis failed due to data fetch error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error analyzing {symbol}: {e}")
        raise
```
