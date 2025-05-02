# Error Handling Refactoring Plan

This document outlines the plan for gradually refactoring the Analysis Engine Service code to use the new standardized exception hierarchy.

## Goals

1. Replace direct imports from `analysis_engine.core.errors` with imports from `analysis_engine.core.exceptions_bridge`
2. Update exception handling to catch both service-specific and common-lib exceptions
3. Ensure that all raised exceptions include appropriate details for debugging
4. Update error responses to use the standardized format with `to_dict()` method
5. Improve error logging with structured context data
6. Maintain backward compatibility during the transition

## Phased Approach

### Phase 1: Core Components (High Priority)

Refactor core components that are used throughout the service:

- [ ] `analysis_engine/core/service_container.py`
- [ ] `analysis_engine/core/config.py`
- [ ] `analysis_engine/api/routes.py`
- [ ] `analysis_engine/services/analysis_service.py`

### Phase 2: Analysis Components (Medium Priority)

Refactor analysis components that implement business logic:

- [ ] `analysis_engine/analysis/base_analyzer.py`
- [ ] `analysis_engine/analysis/confluence_analyzer.py`
- [ ] `analysis_engine/analysis/advanced_ta/market_regime.py`
- [ ] `analysis_engine/analysis/advanced_ta/multi_timeframe.py`

### Phase 3: Integration Components (Medium Priority)

Refactor components that integrate with other services:

- [ ] `analysis_engine/integration/analysis_integration_service.py`
- [ ] `analysis_engine/integration/feature_store_client.py`
- [ ] `analysis_engine/integration/ml_integration_client.py`

### Phase 4: Utility Components (Low Priority)

Refactor utility components:

- [ ] `analysis_engine/utils/validation.py`
- [ ] `analysis_engine/utils/data_transformation.py`

## Refactoring Guidelines

### Import Changes

Replace:
```python
from analysis_engine.core.errors import (
    AnalysisEngineError,
    ValidationError,
    DataFetchError
)
```

With:
```python
from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError as CommonDataFetchError
)
from analysis_engine.core.errors import (
    AnalysisEngineError,
    ValidationError,
    DataFetchError
)
```

### Exception Raising Changes

Replace:
```python
raise DataFetchError("Failed to fetch data")
```

With:
```python
raise DataFetchError(
    message="Failed to fetch data",
    source="feature_store",
    details={"symbol": symbol, "timeframe": timeframe}
)
```

### Exception Handling Changes

Replace:
```python
try:
    # Code that might raise an exception
except DataFetchError as e:
    logger.error(f"Data fetch error: {str(e)}")
    # Handle the error
```

With:
```python
try:
    # Code that might raise an exception
except (DataFetchError, CommonDataFetchError) as e:
    logger.error(
        f"Data fetch error: {getattr(e, 'message', str(e))}",
        extra=getattr(e, "details", {})
    )
    # Handle the error
```

### Error Response Format Changes

Replace:
```python
return JSONResponse(
    status_code=status.HTTP_400_BAD_REQUEST,
    content={
        "error": {
            "type": "ValidationError",
            "message": "Invalid input",
            "details": details
        }
    }
)
```

With:
```python
error = ValidationError(
    message="Invalid input",
    details=details
)
return JSONResponse(
    status_code=status.HTTP_400_BAD_REQUEST,
    content=error.to_dict()
)
```

Or, even better, just raise the exception and let the exception handler take care of it:
```python
raise ValidationError(
    message="Invalid input",
    details=details
)
```

## Testing Strategy

1. Write tests for each refactored component
2. Ensure that both old and new exception types are handled correctly
3. Verify that error responses contain the expected information
4. Check that logging includes all relevant details

## Example Refactorings

### Example 1: Utility Function

#### Before

```python
from analysis_engine.core.errors import DataFetchError

def fetch_market_data(symbol, timeframe):
    try:
        response = requests.get(f"{API_URL}/data/{symbol}?timeframe={timeframe}")
        if response.status_code != 200:
            raise DataFetchError(f"Failed to fetch data for {symbol}, status code: {response.status_code}")
        return response.json()
    except requests.RequestException as e:
        raise DataFetchError(f"Request failed: {str(e)}")
```

#### After

```python
from analysis_engine.core.exceptions_bridge import DataFetchError as CommonDataFetchError
from analysis_engine.core.errors import DataFetchError

def fetch_market_data(symbol, timeframe):
    try:
        response = requests.get(f"{API_URL}/data/{symbol}?timeframe={timeframe}")
        if response.status_code != 200:
            raise DataFetchError(
                message=f"Failed to fetch data for {symbol}",
                source="external_api",
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status_code": response.status_code,
                    "response_text": response.text[:200]  # Include part of the response for debugging
                }
            )
        return response.json()
    except requests.RequestException as e:
        raise DataFetchError(
            message="Request failed",
            source="external_api",
            details={
                "symbol": symbol,
                "timeframe": timeframe,
                "exception": str(e)
            }
        )
```

### Example 2: API Route

#### Before

```python
@router.get("/analysis/{symbol}")
async def get_analysis(symbol: str):
    try:
        # Validate input
        if not symbol:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "type": "ValidationError",
                        "message": "Symbol is required"
                    }
                }
            )

        # Perform analysis
        result = await analysis_service.analyze(symbol)
        return result
    except Exception as e:
        logger.error(f"Error in get_analysis: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred"
                }
            }
        )
```

#### After

```python
@router.get("/analysis/{symbol}")
async def get_analysis(symbol: str):
    try:
        # Validate input
        if not symbol:
            raise ValidationError(
                message="Symbol is required",
                details={"parameter": "symbol"}
            )

        # Perform analysis
        result = await analysis_service.analyze(symbol)
        return result
    except ValidationError:
        # Will be handled by the validation_exception_handler
        raise
    except DataFetchError:
        # Will be handled by the data_fetch_exception_handler
        raise
    except AnalysisError:
        # Will be handled by the analysis_exception_handler
        raise
    except Exception as e:
        # Wrap unexpected exceptions
        logger.error(f"Unexpected error in get_analysis: {str(e)}", exc_info=True)
        raise AnalysisEngineError(
            message="An unexpected error occurred",
            error_code="INTERNAL_SERVER_ERROR",
            details={"exception_type": e.__class__.__name__}
        )
```

### Example 3: Service Method

#### Before

```python
async def analyze(self, symbol: str):
    try:
        # Fetch data
        data = await self.data_client.get_data(symbol)

        # Perform analysis
        result = self.analyzer.analyze(data)

        return result
    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        raise AnalysisError(f"Analysis failed: {str(e)}")
```

#### After

```python
async def analyze(self, symbol: str):
    try:
        # Fetch data
        data = await self.data_client.get_data(symbol)

        # Perform analysis
        result = self.analyzer.analyze(data)

        return result
    except DataFetchError as e:
        # Re-raise with more context
        raise DataFetchError(
            message=f"Failed to fetch data for analysis: {e.message}",
            source="data_client",
            details={
                "symbol": symbol,
                "original_error": str(e),
                **getattr(e, "details", {})
            }
        )
    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}", exc_info=True)
        raise AnalysisError(
            message="Analysis failed",
            details={
                "symbol": symbol,
                "exception_type": e.__class__.__name__,
                "exception_message": str(e)
            }
        )
```

## Timeline

- Phase 1: 1-2 weeks
- Phase 2: 2-3 weeks
- Phase 3: 1-2 weeks
- Phase 4: 1 week

Total estimated time: 5-8 weeks
