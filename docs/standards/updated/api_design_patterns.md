# Forex Trading Platform API Design Patterns

This document defines standard patterns for API design and naming across the Forex Trading Platform. These patterns ensure consistency, maintainability, and alignment with domain concepts.

## Table of Contents

1. [API Design Principles](#api-design-principles)
2. [URL Structure and Naming](#url-structure-and-naming)
3. [HTTP Methods and CRUD Operations](#http-methods-and-crud-operations)
4. [Request and Response Formats](#request-and-response-formats)
5. [Error Handling](#error-handling)
6. [Versioning](#versioning)
7. [Authentication and Authorization](#authentication-and-authorization)
8. [Pagination, Filtering, and Sorting](#pagination-filtering-and-sorting)
9. [Domain-Specific Patterns](#domain-specific-patterns)
10. [Examples](#examples)

## API Design Principles

1. **Domain-Driven Design**: APIs should reflect the ubiquitous language of forex trading
2. **Resource-Oriented**: APIs should be organized around resources
3. **Consistency**: Follow consistent patterns across all services
4. **Backward Compatibility**: Maintain backward compatibility for existing APIs
5. **Documentation**: All APIs should be well-documented with examples

## URL Structure and Naming

### Base URL Structure

```
https://api.example.com/v1/{service}/{resource}/{id}
```

- `{service}`: The service name (e.g., `market-data`, `trading`, `analysis`)
- `{resource}`: The resource name (e.g., `instruments`, `orders`, `positions`)
- `{id}`: The resource identifier (optional)

### Resource Naming Conventions

- Use plural nouns for resource collections (e.g., `/instruments`, `/orders`)
- Use kebab-case for multi-word resource names (e.g., `/currency-pairs`, `/trading-signals`)
- Use lowercase for all URL segments
- Avoid verbs in resource names (use HTTP methods instead)

### Nested Resources

For resources that are logically nested under another resource:

```
/v1/{service}/{parent-resource}/{parent-id}/{child-resource}
```

Example:
```
/v1/trading/accounts/123/positions
```

### Actions on Resources

For actions that don't fit the CRUD model, use a resource-action pattern:

```
/v1/{service}/{resource}/{id}/{action}
```

Example:
```
/v1/trading/orders/456/cancel
```

## HTTP Methods and CRUD Operations

Use standard HTTP methods for CRUD operations:

| Operation | HTTP Method | URL Pattern | Description |
|-----------|-------------|-------------|-------------|
| Create | POST | `/v1/{service}/{resource}` | Create a new resource |
| Read (Collection) | GET | `/v1/{service}/{resource}` | Get a list of resources |
| Read (Single) | GET | `/v1/{service}/{resource}/{id}` | Get a specific resource |
| Update (Full) | PUT | `/v1/{service}/{resource}/{id}` | Replace a resource |
| Update (Partial) | PATCH | `/v1/{service}/{resource}/{id}` | Update parts of a resource |
| Delete | DELETE | `/v1/{service}/{resource}/{id}` | Delete a resource |

### Domain-Specific Operations

| Operation | HTTP Method | URL Pattern | Description |
|-----------|-------------|-------------|-------------|
| Place Order | POST | `/v1/trading/orders` | Place a new trading order |
| Cancel Order | POST | `/v1/trading/orders/{id}/cancel` | Cancel an existing order |
| Get Market Data | GET | `/v1/market-data/instruments/{symbol}/ohlcv` | Get OHLCV data for an instrument |
| Generate Signal | POST | `/v1/analysis/signals/generate` | Generate a trading signal |
| Backtest Strategy | POST | `/v1/strategy/backtest` | Run a backtest for a trading strategy |

## Request and Response Formats

### Request Format

#### Headers

- `Content-Type: application/json`
- `Accept: application/json`
- `Authorization: Bearer {token}`
- `X-Correlation-ID: {correlation-id}`

#### Body (JSON)

```json
{
  "property1": "value1",
  "property2": "value2",
  "nested": {
    "property3": "value3"
  }
}
```

### Response Format

#### Headers

- `Content-Type: application/json`
- `X-Correlation-ID: {correlation-id}`
- `X-Request-ID: {request-id}`

#### Success Response Body

```json
{
  "data": {
    "id": "123",
    "property1": "value1",
    "property2": "value2"
  },
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z",
    "pagination": {
      "page": 1,
      "pageSize": 10,
      "totalPages": 5,
      "totalItems": 42
    }
  }
}
```

#### Collection Response Body

```json
{
  "data": [
    {
      "id": "123",
      "property1": "value1"
    },
    {
      "id": "456",
      "property1": "value2"
    }
  ],
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z",
    "pagination": {
      "page": 1,
      "pageSize": 10,
      "totalPages": 5,
      "totalItems": 42
    }
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field1": "Error details for field1",
      "field2": "Error details for field2"
    },
    "correlationId": "correlation-id",
    "requestId": "request-id",
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

### HTTP Status Codes

| Status Code | Description | Usage |
|-------------|-------------|-------|
| 200 OK | Success | Successful GET, PUT, PATCH, or DELETE |
| 201 Created | Resource created | Successful POST that creates a resource |
| 204 No Content | Success with no content | Successful DELETE or PUT with no response body |
| 400 Bad Request | Invalid request | Malformed request or invalid parameters |
| 401 Unauthorized | Authentication required | Missing or invalid authentication |
| 403 Forbidden | Permission denied | Authenticated but not authorized |
| 404 Not Found | Resource not found | Resource does not exist |
| 409 Conflict | Resource conflict | Resource already exists or version conflict |
| 422 Unprocessable Entity | Validation error | Request validation failed |
| 429 Too Many Requests | Rate limit exceeded | Too many requests in a given time |
| 500 Internal Server Error | Server error | Unexpected server error |
| 503 Service Unavailable | Service unavailable | Service temporarily unavailable |

### Error Codes

Error codes should be domain-specific and follow this pattern:

`{DOMAIN}_{CATEGORY}_{REASON}`

Examples:
- `TRADING_ORDER_INVALID_PRICE`
- `MARKET_DATA_INSTRUMENT_NOT_FOUND`
- `AUTH_TOKEN_EXPIRED`
- `ANALYSIS_SIGNAL_GENERATION_FAILED`

## Versioning

### URL-Based Versioning

Include the API version in the URL:

```
https://api.example.com/v1/market-data/instruments
```

### Version Lifecycle

1. **Development**: Pre-release versions (e.g., `v1-alpha`, `v1-beta`)
2. **Stable**: Released versions (e.g., `v1`, `v2`)
3. **Deprecated**: Versions scheduled for removal
4. **Sunset**: Versions no longer supported

### Version Headers

Include version information in response headers:

- `X-API-Version: v1`
- `X-API-Deprecated: true` (for deprecated versions)
- `X-API-Sunset-Date: 2024-12-31` (for versions scheduled for removal)

## Authentication and Authorization

### Authentication Methods

1. **Bearer Token**: JWT-based authentication
   ```
   Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

2. **API Key**: For service-to-service communication
   ```
   X-API-Key: api_key_value
   ```

### Authorization Scopes

Define scopes for different levels of access:

- `read:market-data`: Read market data
- `write:orders`: Place and modify orders
- `read:account`: Read account information
- `write:account`: Modify account settings
- `admin:system`: Administrative access

## Pagination, Filtering, and Sorting

### Pagination

Use query parameters for pagination:

```
GET /v1/market-data/instruments?page=2&pageSize=10
```

Response should include pagination metadata:

```json
{
  "data": [...],
  "meta": {
    "pagination": {
      "page": 2,
      "pageSize": 10,
      "totalPages": 5,
      "totalItems": 42
    }
  }
}
```

### Filtering

Use query parameters for filtering:

```
GET /v1/market-data/instruments?category=forex&region=europe
```

For more complex filters, use a structured query parameter:

```
GET /v1/market-data/instruments?filter={"category":"forex","region":"europe"}
```

### Sorting

Use query parameters for sorting:

```
GET /v1/market-data/instruments?sort=name&order=asc
```

For multiple sort fields:

```
GET /v1/market-data/instruments?sort=category,name&order=asc,desc
```

## Domain-Specific Patterns

### Market Data API Patterns

#### OHLCV Data

```
GET /v1/market-data/instruments/{symbol}/ohlcv?timeframe=1h&start=2023-01-01T00:00:00Z&end=2023-01-31T23:59:59Z
```

Response:

```json
{
  "data": {
    "symbol": "EUR/USD",
    "timeframe": "1h",
    "ohlcv": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 1.0750,
        "high": 1.0755,
        "low": 1.0748,
        "close": 1.0752,
        "volume": 1250
      },
      // More data points...
    ]
  },
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

#### Technical Indicators

```
GET /v1/analysis/indicators/{indicator}?symbol=EUR/USD&timeframe=1h&start=2023-01-01T00:00:00Z&end=2023-01-31T23:59:59Z&params={"period":14}
```

Response:

```json
{
  "data": {
    "symbol": "EUR/USD",
    "timeframe": "1h",
    "indicator": "RSI",
    "parameters": {
      "period": 14
    },
    "values": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "value": 65.42
      },
      // More data points...
    ]
  },
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

### Trading API Patterns

#### Place Order

```
POST /v1/trading/orders
```

Request:

```json
{
  "symbol": "EUR/USD",
  "side": "BUY",
  "orderType": "LIMIT",
  "quantity": 1.0,
  "price": 1.0750,
  "timeInForce": "GTC"
}
```

Response:

```json
{
  "data": {
    "orderId": "12345",
    "symbol": "EUR/USD",
    "side": "BUY",
    "orderType": "LIMIT",
    "quantity": 1.0,
    "price": 1.0750,
    "timeInForce": "GTC",
    "status": "PENDING",
    "createdAt": "2023-06-01T12:34:56Z"
  },
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

#### Get Positions

```
GET /v1/trading/positions
```

Response:

```json
{
  "data": [
    {
      "positionId": "67890",
      "symbol": "EUR/USD",
      "side": "BUY",
      "quantity": 1.0,
      "entryPrice": 1.0750,
      "currentPrice": 1.0760,
      "unrealizedPnl": 0.0010,
      "openedAt": "2023-06-01T10:30:00Z"
    },
    // More positions...
  ],
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

### Analysis API Patterns

#### Generate Signal

```
POST /v1/analysis/signals/generate
```

Request:

```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "strategy": "MACD_CROSSOVER",
  "parameters": {
    "fastPeriod": 12,
    "slowPeriod": 26,
    "signalPeriod": 9
  }
}
```

Response:

```json
{
  "data": {
    "signalId": "54321",
    "symbol": "EUR/USD",
    "timeframe": "1h",
    "strategy": "MACD_CROSSOVER",
    "parameters": {
      "fastPeriod": 12,
      "slowPeriod": 26,
      "signalPeriod": 9
    },
    "signal": "BUY",
    "strength": 0.75,
    "timestamp": "2023-06-01T12:30:00Z",
    "price": 1.0755,
    "indicators": {
      "macd": -0.0005,
      "signal": -0.0010,
      "histogram": 0.0005
    }
  },
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

## Examples

### Market Data Service API

#### Get Instruments

```
GET /v1/market-data/instruments
```

Response:

```json
{
  "data": [
    {
      "symbol": "EUR/USD",
      "name": "Euro / US Dollar",
      "category": "forex",
      "pipValue": 0.0001,
      "minLotSize": 0.01,
      "tradingHours": "24/5"
    },
    {
      "symbol": "GBP/USD",
      "name": "British Pound / US Dollar",
      "category": "forex",
      "pipValue": 0.0001,
      "minLotSize": 0.01,
      "tradingHours": "24/5"
    }
  ],
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z",
    "pagination": {
      "page": 1,
      "pageSize": 10,
      "totalPages": 5,
      "totalItems": 42
    }
  }
}
```

#### Get OHLCV Data

```
GET /v1/market-data/instruments/EUR-USD/ohlcv?timeframe=1h&start=2023-01-01T00:00:00Z&end=2023-01-01T23:59:59Z
```

Response:

```json
{
  "data": {
    "symbol": "EUR/USD",
    "timeframe": "1h",
    "ohlcv": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 1.0750,
        "high": 1.0755,
        "low": 1.0748,
        "close": 1.0752,
        "volume": 1250
      },
      {
        "timestamp": "2023-01-01T01:00:00Z",
        "open": 1.0752,
        "high": 1.0760,
        "low": 1.0751,
        "close": 1.0758,
        "volume": 1320
      }
    ]
  },
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

### Trading Service API

#### Place Order

```
POST /v1/trading/orders
```

Request:

```json
{
  "symbol": "EUR/USD",
  "side": "BUY",
  "orderType": "LIMIT",
  "quantity": 1.0,
  "price": 1.0750,
  "timeInForce": "GTC"
}
```

Response:

```json
{
  "data": {
    "orderId": "12345",
    "symbol": "EUR/USD",
    "side": "BUY",
    "orderType": "LIMIT",
    "quantity": 1.0,
    "price": 1.0750,
    "timeInForce": "GTC",
    "status": "PENDING",
    "createdAt": "2023-06-01T12:34:56Z"
  },
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

#### Get Order

```
GET /v1/trading/orders/12345
```

Response:

```json
{
  "data": {
    "orderId": "12345",
    "symbol": "EUR/USD",
    "side": "BUY",
    "orderType": "LIMIT",
    "quantity": 1.0,
    "price": 1.0750,
    "timeInForce": "GTC",
    "status": "FILLED",
    "filledQuantity": 1.0,
    "filledPrice": 1.0750,
    "createdAt": "2023-06-01T12:34:56Z",
    "updatedAt": "2023-06-01T12:35:30Z",
    "filledAt": "2023-06-01T12:35:30Z"
  },
  "meta": {
    "timestamp": "2023-06-01T12:36:00Z"
  }
}
```

### Analysis Service API

#### Generate Signal

```
POST /v1/analysis/signals/generate
```

Request:

```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "strategy": "MACD_CROSSOVER",
  "parameters": {
    "fastPeriod": 12,
    "slowPeriod": 26,
    "signalPeriod": 9
  }
}
```

Response:

```json
{
  "data": {
    "signalId": "54321",
    "symbol": "EUR/USD",
    "timeframe": "1h",
    "strategy": "MACD_CROSSOVER",
    "parameters": {
      "fastPeriod": 12,
      "slowPeriod": 26,
      "signalPeriod": 9
    },
    "signal": "BUY",
    "strength": 0.75,
    "timestamp": "2023-06-01T12:30:00Z",
    "price": 1.0755,
    "indicators": {
      "macd": -0.0005,
      "signal": -0.0010,
      "histogram": 0.0005
    }
  },
  "meta": {
    "timestamp": "2023-06-01T12:34:56Z"
  }
}
```

#### Get Signals

```
GET /v1/analysis/signals?symbol=EUR-USD&timeframe=1h&start=2023-06-01T00:00:00Z&end=2023-06-01T23:59:59Z
```

Response:

```json
{
  "data": [
    {
      "signalId": "54321",
      "symbol": "EUR/USD",
      "timeframe": "1h",
      "strategy": "MACD_CROSSOVER",
      "signal": "BUY",
      "strength": 0.75,
      "timestamp": "2023-06-01T12:30:00Z",
      "price": 1.0755
    },
    {
      "signalId": "54322",
      "symbol": "EUR/USD",
      "timeframe": "1h",
      "strategy": "RSI_OVERSOLD",
      "signal": "BUY",
      "strength": 0.65,
      "timestamp": "2023-06-01T14:30:00Z",
      "price": 1.0740
    }
  ],
  "meta": {
    "timestamp": "2023-06-01T15:00:00Z",
    "pagination": {
      "page": 1,
      "pageSize": 10,
      "totalPages": 1,
      "totalItems": 2
    }
  }
}
```