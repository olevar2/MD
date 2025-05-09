# Common Error Scenarios and Recovery Strategies

This document provides guidance on handling common error scenarios in the Forex Trading Platform. For each scenario, we provide:

1. A description of the error
2. Common causes
3. Detection strategies
4. Recovery approaches
5. Code examples

## Table of Contents

1. [Network Errors](#1-network-errors)
2. [Data Validation Errors](#2-data-validation-errors)
3. [Authentication and Authorization Errors](#3-authentication-and-authorization-errors)
4. [Service Unavailability](#4-service-unavailability)
5. [Database Errors](#5-database-errors)
6. [Business Logic Errors](#6-business-logic-errors)
7. [Resource Exhaustion](#7-resource-exhaustion)
8. [Concurrency Issues](#8-concurrency-issues)
9. [External API Errors](#9-external-api-errors)
10. [Model Prediction Errors](#10-model-prediction-errors)

## 1. Network Errors

### Description

Network errors occur when services cannot communicate due to network issues such as timeouts, connection resets, or DNS failures.

### Common Causes

- Network infrastructure problems
- Service overload
- Firewall or security group misconfiguration
- DNS resolution failures
- Proxy or load balancer issues

### Detection

- Connection timeouts
- Connection refused errors
- Socket errors
- DNS resolution failures

### Recovery Strategies

1. **Retry with Backoff**: Implement exponential backoff with jitter
2. **Circuit Breaker**: Prevent cascading failures by stopping calls to failing services
3. **Fallback to Cache**: Use cached data when network requests fail
4. **Alternative Endpoints**: Try alternative service endpoints or regions
5. **Graceful Degradation**: Continue with reduced functionality

### Example

```python
from common_lib.resilience import retry_with_policy
from common_lib.exceptions import NetworkError, ServiceUnavailableError

@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    jitter=True,
    exceptions=[ConnectionError, TimeoutError]
)
async def fetch_market_data(symbol: str) -> MarketData:
    """
    Fetch market data with retry logic for network errors.
    
    Args:
        symbol: The trading symbol to fetch data for
        
    Returns:
        MarketData object with current prices
        
    Raises:
        NetworkError: If network issues persist after retries
        ServiceUnavailableError: If the service is unavailable
    """
    try:
        return await market_data_client.get_price(symbol)
    except (ConnectionError, TimeoutError) as e:
        # This will be caught by the retry decorator and retried
        logger.warning(
            f"Network error fetching market data for {symbol}, retrying...",
            extra={"symbol": symbol, "error": str(e)}
        )
        raise
    except Exception as e:
        # Other exceptions won't be retried
        logger.error(
            f"Error fetching market data for {symbol}",
            extra={"symbol": symbol, "error": str(e)}
        )
        raise NetworkError(
            message=f"Failed to fetch market data for {symbol}",
            details={"symbol": symbol, "original_error": str(e)}
        )
```

## 2. Data Validation Errors

### Description

Data validation errors occur when input data fails to meet the expected format, type, or business rules.

### Common Causes

- User input errors
- Malformed API requests
- Data corruption
- Incompatible data formats between services
- Missing required fields

### Detection

- Schema validation failures
- Type conversion errors
- Business rule violations
- Constraint violations

### Recovery Strategies

1. **Early Validation**: Validate data as early as possible in the request lifecycle
2. **Detailed Error Messages**: Provide specific information about validation failures
3. **Default Values**: Use sensible defaults for non-critical missing fields
4. **Partial Processing**: Process valid parts of the request if possible
5. **Validation Layers**: Implement validation at multiple layers (API, service, database)

### Example

```python
from pydantic import BaseModel, validator, ValidationError
from common_lib.exceptions import DataValidationError

class TradeRequest(BaseModel):
    """Model for trade request validation."""
    
    symbol: str
    quantity: float
    price: float
    side: str  # "buy" or "sell"
    
    @validator('symbol')
    def symbol_must_be_valid(cls, v):
        if not is_valid_symbol(v):
            raise ValueError(f"Invalid trading symbol: {v}")
        return v
    
    @validator('quantity')
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Quantity must be positive, got {v}")
        return v
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Price must be positive, got {v}")
        return v
    
    @validator('side')
    def side_must_be_valid(cls, v):
        if v.lower() not in ["buy", "sell"]:
            raise ValueError(f"Side must be 'buy' or 'sell', got {v}")
        return v.lower()

async def create_trade(trade_data: dict):
    """
    Create a new trade with validation.
    
    Args:
        trade_data: Dictionary containing trade details
        
    Returns:
        Created trade object
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        # Validate the trade request
        trade_request = TradeRequest(**trade_data)
        
        # Process the validated request
        return await trading_service.execute_trade(trade_request)
    except ValidationError as e:
        # Convert Pydantic validation error to platform error
        validation_errors = []
        for error in e.errors():
            validation_errors.append({
                "field": ".".join(error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        raise DataValidationError(
            message="Trade request validation failed",
            details={"validation_errors": validation_errors}
        )
```

## 3. Authentication and Authorization Errors

### Description

Authentication errors occur when users cannot be identified. Authorization errors occur when users lack permission for an operation.

### Common Causes

- Invalid credentials
- Expired tokens
- Missing tokens
- Insufficient permissions
- Role misconfiguration
- Token revocation

### Detection

- Failed token validation
- Missing authentication headers
- Permission checks failing
- Role-based access control denials

### Recovery Strategies

1. **Clear Error Messages**: Provide specific error messages (without revealing sensitive information)
2. **Token Refresh**: Implement token refresh mechanisms for expired tokens
3. **Redirect to Authentication**: Redirect users to login when authentication fails
4. **Permission Guidance**: Inform users about required permissions
5. **Audit Logging**: Log all authentication and authorization failures

### Example

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from common_lib.exceptions import AuthenticationError, AuthorizationError

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Validate JWT token and return the current user.
    
    Args:
        token: JWT token from request
        
    Returns:
        User object for the authenticated user
        
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationError(
                message="Invalid authentication credentials",
                error_code="INVALID_TOKEN"
            )
    except JWTError:
        raise AuthenticationError(
            message="Invalid authentication credentials",
            error_code="INVALID_TOKEN"
        )
    
    user = await user_service.get_user(user_id)
    if user is None:
        raise AuthenticationError(
            message="User not found",
            error_code="USER_NOT_FOUND"
        )
    
    return user

async def get_current_active_user(current_user = Depends(get_current_user)):
    """
    Check if the current user is active.
    
    Args:
        current_user: User object from get_current_user
        
    Returns:
        User object if active
        
    Raises:
        AuthenticationError: If user is inactive
    """
    if not current_user.is_active:
        raise AuthenticationError(
            message="Inactive user",
            error_code="INACTIVE_USER"
        )
    
    return current_user

def has_permission(required_permission: str):
    """
    Dependency to check if user has the required permission.
    
    Args:
        required_permission: Permission string to check
        
    Returns:
        Dependency function
        
    Raises:
        AuthorizationError: If user lacks the required permission
    """
    async def check_permission(current_user = Depends(get_current_active_user)):
        if not current_user.has_permission(required_permission):
            raise AuthorizationError(
                message=f"Not authorized to perform this action",
                error_code="PERMISSION_DENIED",
                details={"required_permission": required_permission}
            )
        return current_user
    
    return check_permission
```

## 4. Service Unavailability

### Description

Service unavailability occurs when a dependent service is down, overloaded, or unreachable.

### Common Causes

- Service deployment or restart
- Service crash
- Infrastructure failure
- Resource exhaustion
- Network partition
- Maintenance window

### Detection

- Connection failures
- Timeout errors
- HTTP 5xx responses
- Health check failures

### Recovery Strategies

1. **Circuit Breaker**: Prevent cascading failures by stopping calls to failing services
2. **Fallback Functionality**: Provide alternative functionality when a service is unavailable
3. **Retry with Backoff**: Implement exponential backoff for transient failures
4. **Service Discovery**: Use service discovery to find healthy instances
5. **Bulkhead Pattern**: Isolate failures to prevent system-wide impact

### Example

```python
from common_lib.resilience import CircuitBreaker, CircuitBreakerConfig
from common_lib.exceptions import ServiceUnavailableError, CircuitBreakerOpenError

# Create a circuit breaker for the trading service
trading_cb = CircuitBreaker(
    service_name="portfolio-service",
    resource_name="trading-gateway",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=30,
        half_open_max_calls=2
    )
)

async def execute_trade(trade_request):
    """
    Execute a trade with circuit breaker protection.
    
    Args:
        trade_request: Validated trade request
        
    Returns:
        Trade execution result
        
    Raises:
        ServiceUnavailableError: If trading service is unavailable
    """
    try:
        # Execute the trade through the circuit breaker
        return await trading_cb.execute(
            lambda: trading_gateway_client.execute_trade(trade_request)
        )
    except CircuitBreakerOpenError:
        # Circuit is open, service is considered unavailable
        logger.warning(
            "Circuit breaker open for trading-gateway",
            extra={"circuit": "trading-gateway"}
        )
        
        # Check if we can provide a fallback
        if trade_request.is_market_order():
            # For market orders, we can't provide a fallback
            raise ServiceUnavailableError(
                message="Trading service is currently unavailable",
                error_code="TRADING_SERVICE_UNAVAILABLE",
                details={
                    "retry_after_seconds": 30,
                    "circuit": "trading-gateway"
                }
            )
        else:
            # For limit orders, we can queue them for later execution
            await order_queue.enqueue(trade_request)
            return {
                "status": "queued",
                "message": "Order queued for later execution",
                "order_id": generate_order_id()
            }
```

## 5. Database Errors

### Description

Database errors occur when database operations fail due to connectivity issues, constraint violations, or other database-related problems.

### Common Causes

- Connection pool exhaustion
- Database server overload
- Constraint violations
- Deadlocks
- Schema changes
- Query timeout

### Detection

- SQL exceptions
- Connection errors
- Constraint violation errors
- Deadlock errors
- Query timeout errors

### Recovery Strategies

1. **Retry Transient Errors**: Retry operations that might succeed on retry
2. **Connection Pool Management**: Properly configure and monitor connection pools
3. **Transaction Management**: Use appropriate transaction isolation levels
4. **Constraint Handling**: Properly handle constraint violations
5. **Deadlock Retry**: Automatically retry transactions that fail due to deadlocks

### Example

```python
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from common_lib.resilience import retry_with_policy
from common_lib.exceptions import (
    DataStorageError,
    DataValidationError,
    ServiceUnavailableError
)

@retry_with_policy(
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0,
    jitter=True,
    exceptions=[OperationalError]
)
async def save_portfolio(portfolio):
    """
    Save a portfolio to the database with retry for transient errors.
    
    Args:
        portfolio: Portfolio object to save
        
    Returns:
        Saved portfolio with ID
        
    Raises:
        DataValidationError: If portfolio violates constraints
        DataStorageError: If database operation fails
        ServiceUnavailableError: If database is unavailable
    """
    try:
        return await repository.save(portfolio)
    except IntegrityError as e:
        # Handle constraint violations (not retried)
        if "unique constraint" in str(e).lower():
            raise DataValidationError(
                message="Portfolio with this name already exists",
                error_code="DUPLICATE_PORTFOLIO",
                details={"portfolio_name": portfolio.name}
            )
        else:
            raise DataValidationError(
                message="Portfolio data violates database constraints",
                error_code="CONSTRAINT_VIOLATION",
                details={"error": str(e)}
            )
    except OperationalError as e:
        # Handle operational errors (will be retried)
        if "deadlock" in str(e).lower():
            logger.warning(
                "Deadlock detected, retrying transaction",
                extra={"portfolio_id": portfolio.id}
            )
        elif "connection" in str(e).lower():
            logger.warning(
                "Database connection error, retrying",
                extra={"portfolio_id": portfolio.id}
            )
        raise
    except SQLAlchemyError as e:
        # Handle other database errors (not retried)
        logger.error(
            f"Database error: {str(e)}",
            extra={"portfolio_id": portfolio.id}
        )
        raise DataStorageError(
            message="Failed to save portfolio",
            error_code="DATABASE_ERROR",
            details={"portfolio_id": portfolio.id, "error": str(e)}
        )
```

## 6. Business Logic Errors

### Description

Business logic errors occur when operations violate business rules or constraints specific to the domain.

### Common Causes

- Insufficient funds or balance
- Exceeding trading limits
- Invalid trading hours
- Risk management violations
- Regulatory constraints
- Business rule violations

### Detection

- Business rule validation failures
- Domain-specific constraint checks
- Risk management checks

### Recovery Strategies

1. **Domain-Specific Exceptions**: Use exceptions that map to business concepts
2. **Actionable Error Messages**: Provide clear guidance on how to resolve the issue
3. **Preventive Validation**: Validate business rules before attempting operations
4. **Compensating Actions**: Suggest alternative actions when a rule is violated
5. **Contextual Information**: Include relevant context in error messages

### Example

```python
from common_lib.exceptions import TradingError
from portfolio_management_service.error import InsufficientBalanceError

async def execute_trade(trade_request, portfolio_id):
    """
    Execute a trade for a portfolio with business rule validation.
    
    Args:
        trade_request: Validated trade request
        portfolio_id: Portfolio ID to trade on
        
    Returns:
        Trade execution result
        
    Raises:
        PortfolioNotFoundError: If portfolio doesn't exist
        InsufficientBalanceError: If portfolio has insufficient balance
        TradingError: If trade violates trading rules
    """
    # Get the portfolio
    portfolio = await portfolio_service.get_portfolio(portfolio_id)
    
    # Check if the portfolio has sufficient balance
    required_amount = calculate_required_amount(trade_request)
    if portfolio.available_balance < required_amount:
        raise InsufficientBalanceError(
            message="Insufficient balance to execute trade",
            error_code="INSUFFICIENT_BALANCE",
            details={
                "portfolio_id": portfolio_id,
                "required_amount": required_amount,
                "available_balance": portfolio.available_balance,
                "currency": portfolio.currency
            }
        )
    
    # Check trading hours
    if not is_valid_trading_time(trade_request.symbol):
        raise TradingError(
            message=f"Trading for {trade_request.symbol} is not available at this time",
            error_code="INVALID_TRADING_HOURS",
            details={
                "symbol": trade_request.symbol,
                "current_time": get_current_time(),
                "trading_hours": get_trading_hours(trade_request.symbol)
            }
        )
    
    # Check risk limits
    risk_check_result = await risk_service.check_trade(trade_request, portfolio)
    if not risk_check_result.is_approved:
        raise TradingError(
            message="Trade exceeds risk limits",
            error_code="RISK_LIMIT_EXCEEDED",
            details={
                "portfolio_id": portfolio_id,
                "risk_check_result": risk_check_result.to_dict()
            }
        )
    
    # Execute the trade
    return await trading_gateway.execute_trade(trade_request)
```

## 7. Resource Exhaustion

### Description

Resource exhaustion occurs when a system runs out of critical resources such as memory, CPU, disk space, or connection pools.

### Common Causes

- Memory leaks
- Inefficient algorithms
- Excessive concurrent requests
- Large data processing
- Unbounded resource usage
- Missing resource limits

### Detection

- Out of memory errors
- High CPU utilization
- Disk space warnings
- Connection pool exhaustion
- Thread pool exhaustion

### Recovery Strategies

1. **Resource Limits**: Set explicit limits on resource usage
2. **Graceful Degradation**: Reduce functionality when resources are low
3. **Load Shedding**: Reject non-critical requests when under heavy load
4. **Bulkhead Pattern**: Isolate critical operations with dedicated resources
5. **Monitoring and Alerting**: Detect resource issues before they cause failures

### Example

```python
from common_lib.resilience import bulkhead
from common_lib.exceptions import BulkheadFullError, ServiceUnavailableError

# Create bulkheads for different operation types
critical_operations = bulkhead(
    name="critical-operations",
    max_concurrent=20,
    max_queue_size=10
)

non_critical_operations = bulkhead(
    name="non-critical-operations",
    max_concurrent=10,
    max_queue_size=20
)

@critical_operations
async def execute_trade(trade_request):
    """
    Execute a trade with resource protection via bulkhead.
    
    Args:
        trade_request: Validated trade request
        
    Returns:
        Trade execution result
        
    Raises:
        ServiceUnavailableError: If system is under heavy load
    """
    try:
        # This operation is protected by the critical_operations bulkhead
        return await trading_gateway.execute_trade(trade_request)
    except BulkheadFullError:
        # Bulkhead is full, system is under heavy load
        logger.warning(
            "Critical operations bulkhead full, rejecting trade request",
            extra={"bulkhead": "critical-operations"}
        )
        raise ServiceUnavailableError(
            message="System is currently under heavy load, please try again later",
            error_code="SYSTEM_OVERLOADED",
            details={"retry_after_seconds": 30}
        )

@non_critical_operations
async def get_market_analysis(symbol):
    """
    Get market analysis with resource protection via bulkhead.
    
    Args:
        symbol: Trading symbol to analyze
        
    Returns:
        Market analysis result
        
    Raises:
        ServiceUnavailableError: If system is under heavy load
    """
    try:
        # This operation is protected by the non_critical_operations bulkhead
        return await analysis_service.analyze_market(symbol)
    except BulkheadFullError:
        # Bulkhead is full, system is under heavy load
        logger.warning(
            "Non-critical operations bulkhead full, rejecting analysis request",
            extra={"bulkhead": "non-critical-operations"}
        )
        raise ServiceUnavailableError(
            message="Analysis service is currently under heavy load, please try again later",
            error_code="ANALYSIS_SERVICE_OVERLOADED",
            details={"retry_after_seconds": 60}
        )
```

## 8. Concurrency Issues

### Description

Concurrency issues occur when multiple operations interfere with each other, leading to race conditions, deadlocks, or inconsistent state.

### Common Causes

- Race conditions
- Deadlocks
- Starvation
- Inconsistent locking
- Missing synchronization
- Optimistic concurrency control failures

### Detection

- Deadlock errors
- Version conflicts
- Inconsistent state
- Timing-dependent failures

### Recovery Strategies

1. **Optimistic Concurrency Control**: Use version numbers to detect conflicts
2. **Distributed Locks**: Use locks for critical operations
3. **Idempotent Operations**: Design operations to be safely retryable
4. **Conflict Resolution**: Implement strategies to resolve conflicts
5. **Transaction Isolation**: Use appropriate transaction isolation levels

### Example

```python
from common_lib.exceptions import DataStorageError, ConcurrencyError

async def update_portfolio(portfolio_id, updates, version):
    """
    Update a portfolio with optimistic concurrency control.
    
    Args:
        portfolio_id: ID of the portfolio to update
        updates: Dictionary of updates to apply
        version: Expected version of the portfolio
        
    Returns:
        Updated portfolio
        
    Raises:
        PortfolioNotFoundError: If portfolio doesn't exist
        ConcurrencyError: If portfolio was modified by another operation
        DataStorageError: If update fails
    """
    try:
        # Get the current portfolio
        portfolio = await portfolio_repository.get_portfolio(portfolio_id)
        
        # Check if the version matches
        if portfolio.version != version:
            raise ConcurrencyError(
                message="Portfolio was modified by another operation",
                error_code="CONCURRENT_MODIFICATION",
                details={
                    "portfolio_id": portfolio_id,
                    "expected_version": version,
                    "actual_version": portfolio.version
                }
            )
        
        # Apply updates
        for key, value in updates.items():
            setattr(portfolio, key, value)
        
        # Increment version
        portfolio.version += 1
        
        # Save the updated portfolio
        return await portfolio_repository.save(portfolio)
    except ConcurrencyError:
        # Rethrow concurrency errors
        raise
    except Exception as e:
        logger.error(
            f"Failed to update portfolio: {str(e)}",
            extra={"portfolio_id": portfolio_id}
        )
        raise DataStorageError(
            message="Failed to update portfolio",
            error_code="UPDATE_FAILED",
            details={"portfolio_id": portfolio_id, "error": str(e)}
        )
```

## 9. External API Errors

### Description

External API errors occur when third-party services or APIs return errors or unexpected responses.

### Common Causes

- API changes
- Rate limiting
- Authentication failures
- Service outages
- Invalid requests
- Unexpected response formats

### Detection

- HTTP error status codes
- Unexpected response formats
- Timeout errors
- Authentication errors

### Recovery Strategies

1. **Response Validation**: Validate responses against expected schemas
2. **Retry with Backoff**: Implement exponential backoff for transient failures
3. **Circuit Breaker**: Prevent cascading failures from external dependencies
4. **Fallback Providers**: Use alternative providers when available
5. **Caching**: Cache responses to reduce dependency on external services

### Example

```python
from common_lib.resilience import retry_with_policy, CircuitBreaker
from common_lib.exceptions import (
    ExternalServiceError,
    CircuitBreakerOpenError,
    ServiceUnavailableError
)

# Create a circuit breaker for the external market data API
market_data_cb = CircuitBreaker(
    service_name="market-data-service",
    resource_name="external-market-api",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=60
    )
)

@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    jitter=True,
    exceptions=[ConnectionError, TimeoutError]
)
async def get_market_data(symbol):
    """
    Get market data from external API with resilience patterns.
    
    Args:
        symbol: Trading symbol to get data for
        
    Returns:
        Market data for the symbol
        
    Raises:
        ExternalServiceError: If external API returns an error
        ServiceUnavailableError: If external API is unavailable
    """
    try:
        # Use circuit breaker to protect against external API failures
        return await market_data_cb.execute(
            lambda: external_market_api.get_price(symbol)
        )
    except CircuitBreakerOpenError:
        # Circuit is open, try to use cached data
        logger.warning(
            "Circuit breaker open for external-market-api, using cached data",
            extra={"symbol": symbol}
        )
        
        cached_data = await market_data_cache.get(symbol)
        if cached_data:
            return {
                **cached_data,
                "source": "cache",
                "cache_time": cached_data.timestamp
            }
        else:
            raise ServiceUnavailableError(
                message="Market data service is currently unavailable",
                error_code="MARKET_DATA_UNAVAILABLE",
                details={"symbol": symbol, "retry_after_seconds": 60}
            )
    except (ConnectionError, TimeoutError) as e:
        # These will be retried by the retry decorator
        logger.warning(
            f"Network error accessing external market API: {str(e)}",
            extra={"symbol": symbol}
        )
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(
            f"Error accessing external market API: {str(e)}",
            extra={"symbol": symbol}
        )
        raise ExternalServiceError(
            message=f"Failed to get market data for {symbol}",
            error_code="EXTERNAL_API_ERROR",
            details={"symbol": symbol, "error": str(e)}
        )
```

## 10. Model Prediction Errors

### Description

Model prediction errors occur when machine learning models fail to generate predictions or generate invalid predictions.

### Common Causes

- Model loading failures
- Input data preprocessing errors
- Feature extraction failures
- Out-of-distribution inputs
- Resource constraints
- Model version incompatibility

### Detection

- Model loading errors
- Input validation failures
- Prediction timeouts
- Unexpected prediction values
- Confidence score thresholds

### Recovery Strategies

1. **Input Validation**: Validate inputs before sending to model
2. **Fallback Models**: Use simpler models when complex models fail
3. **Prediction Validation**: Validate predictions against expected ranges
4. **Confidence Thresholds**: Only use predictions with sufficient confidence
5. **Model Versioning**: Ensure compatibility between model and code versions

### Example

```python
from common_lib.exceptions import (
    ModelError,
    ModelLoadError,
    ModelPredictionError,
    DataValidationError
)

async def predict_market_movement(symbol, timeframe, features):
    """
    Predict market movement using ML model with error handling.
    
    Args:
        symbol: Trading symbol
        timeframe: Prediction timeframe
        features: Input features for the model
        
    Returns:
        Prediction result with confidence score
        
    Raises:
        DataValidationError: If input features are invalid
        ModelError: If model prediction fails
    """
    try:
        # Validate input features
        if not validate_features(features, required_features):
            missing_features = get_missing_features(features, required_features)
            raise DataValidationError(
                message="Missing required features for prediction",
                error_code="MISSING_FEATURES",
                details={"missing_features": missing_features}
            )
        
        # Load the appropriate model
        try:
            model = await model_service.get_model(f"market_movement_{timeframe}")
        except ModelLoadError as e:
            logger.error(
                f"Failed to load market movement model: {str(e)}",
                extra={"timeframe": timeframe}
            )
            
            # Try to use fallback model
            logger.info("Attempting to use fallback model")
            model = await model_service.get_model("market_movement_fallback")
        
        # Make prediction
        prediction = await model.predict(features)
        
        # Validate prediction
        if not is_valid_prediction(prediction):
            raise ModelPredictionError(
                message="Model generated invalid prediction",
                error_code="INVALID_PREDICTION",
                details={"prediction": prediction}
            )
        
        # Check confidence
        if prediction.confidence < MIN_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low confidence prediction",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "confidence": prediction.confidence
                }
            )
            
            # Return prediction with warning
            return {
                "prediction": prediction.value,
                "confidence": prediction.confidence,
                "warning": "Low confidence prediction, use with caution"
            }
        
        # Return valid prediction
        return {
            "prediction": prediction.value,
            "confidence": prediction.confidence
        }
    except DataValidationError:
        # Rethrow validation errors
        raise
    except ModelLoadError as e:
        # Handle model loading errors
        logger.error(
            f"Failed to load any model for prediction: {str(e)}",
            extra={"timeframe": timeframe}
        )
        raise ModelError(
            message="Failed to load prediction model",
            error_code="MODEL_UNAVAILABLE",
            details={"timeframe": timeframe}
        )
    except ModelPredictionError as e:
        # Handle prediction errors
        logger.error(
            f"Model prediction error: {str(e)}",
            extra={"symbol": symbol, "timeframe": timeframe}
        )
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(
            f"Unexpected error during prediction: {str(e)}",
            extra={"symbol": symbol, "timeframe": timeframe}
        )
        raise ModelError(
            message="Failed to generate market movement prediction",
            error_code="PREDICTION_FAILED",
            details={"symbol": symbol, "timeframe": timeframe, "error": str(e)}
        )
```
