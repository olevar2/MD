# Exercise 1: Basic Error Handling

## Overview

In this exercise, you will implement error handling in a trading service method, create domain-specific exceptions, and format error responses for API endpoints.

## Prerequisites

- Python 3.8+
- Access to the Forex Trading Platform codebase
- Understanding of the platform's exception hierarchy

## Tasks

### Task 1: Create Domain-Specific Exceptions

Create the following domain-specific exceptions in `trading_service/exceptions.py`:

1. `OrderValidationError`: Raised when an order fails validation
2. `InsufficientBalanceError`: Raised when an account has insufficient balance
3. `MarketClosedError`: Raised when trying to trade in a closed market

Each exception should:
- Extend the appropriate base class from `common_lib.exceptions`
- Include appropriate constructor parameters
- Provide a default error message
- Set an appropriate error code

### Task 2: Implement Error Handling in a Service Method

Implement error handling in the `execute_order` method in `trading_service/service.py`:

```python
async def execute_order(self, order_request: OrderRequest) -> OrderResult:
    """
    Execute a trading order.
    
    Args:
        order_request: The order request to execute
        
    Returns:
        OrderResult with execution details
        
    Raises:
        OrderValidationError: If order validation fails
        InsufficientBalanceError: If account has insufficient balance
        MarketClosedError: If market is closed
        TradingError: For other trading-related errors
    """
    # TODO: Implement error handling
    
    # 1. Validate the order
    # 2. Check account balance
    # 3. Check market status
    # 4. Execute the order
    # 5. Return the result
```

Your implementation should:
- Validate the order and raise `OrderValidationError` if invalid
- Check account balance and raise `InsufficientBalanceError` if insufficient
- Check market status and raise `MarketClosedError` if closed
- Handle unexpected errors appropriately
- Log errors with appropriate context

### Task 3: Format Error Responses for API Endpoints

Implement an error handler for the trading service API in `trading_service/api/error_handlers.py`:

```python
async def order_validation_error_handler(request: Request, exc: OrderValidationError) -> JSONResponse:
    """
    Handle OrderValidationError exceptions.
    
    Args:
        request: FastAPI request
        exc: The exception to handle
        
    Returns:
        JSONResponse with error details
    """
    # TODO: Implement error handler
    
    # 1. Log the error
    # 2. Format the error response
    # 3. Return the response with appropriate status code
```

Your implementation should:
- Log the error with appropriate context
- Format the error response according to the platform's standard format
- Return the response with the appropriate HTTP status code

### Task 4: Register Error Handlers

Register your error handlers in `trading_service/api/main.py`:

```python
from fastapi import FastAPI
from trading_service.api.error_handlers import (
    order_validation_error_handler,
    insufficient_balance_error_handler,
    market_closed_error_handler
)
from trading_service.exceptions import (
    OrderValidationError,
    InsufficientBalanceError,
    MarketClosedError
)

app = FastAPI()

# TODO: Register error handlers
```

### Task 5: Test Error Scenarios

Write tests for your error handling in `tests/trading_service/test_error_handling.py`:

```python
import pytest
from trading_service.exceptions import (
    OrderValidationError,
    InsufficientBalanceError,
    MarketClosedError
)
from trading_service.service import TradingService

# TODO: Implement tests for error scenarios
```

Your tests should cover:
- Order validation failures
- Insufficient balance scenarios
- Market closed scenarios
- Unexpected errors

## Expected Output

When an error occurs, the API should return a response like:

```json
{
  "error": {
    "code": "ORDER_VALIDATION_ERROR",
    "message": "Order validation failed",
    "details": {
      "order_id": "12345",
      "validation_errors": [
        {"field": "quantity", "message": "Quantity must be positive"}
      ]
    },
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2023-05-22T12:34:56.789Z",
    "service": "trading-service"
  }
}
```

## Submission

Submit your solution by:

1. Creating a branch named `error-handling-exercise-{your-name}`
2. Committing your changes to that branch
3. Creating a pull request to the `main` branch

## Evaluation Criteria

Your solution will be evaluated based on:

1. Correctness: Does it handle errors correctly?
2. Completeness: Are all required tasks implemented?
3. Code quality: Is the code well-structured and maintainable?
4. Documentation: Are the exceptions and handlers well-documented?
5. Testing: Are all error scenarios properly tested?

## Hints

- Use the `async_with_exception_handling` decorator for consistent error handling
- Look at existing error handlers in other services for examples
- Use structured logging with appropriate context
- Remember to propagate correlation IDs for cross-service tracking
