# Enhanced API Gateway API Reference

This document describes the API endpoints provided by the Enhanced API Gateway for the Forex Trading Platform.

## Authentication

The Enhanced API Gateway supports two authentication methods:

- **JWT Authentication**: For user authentication.
- **API Key Authentication**: For service-to-service authentication.

### JWT Authentication

To authenticate using JWT, include an `Authorization` header with a Bearer token:

```
Authorization: Bearer <token>
```

### API Key Authentication

To authenticate using an API key, include an `X-API-Key` header:

```
X-API-Key: <api_key>
```

## Headers

The Enhanced API Gateway supports the following headers:

- **X-Correlation-ID**: A unique identifier for the request. If not provided, a new one will be generated.
- **X-Request-ID**: A unique identifier for the request. If not provided, a new one will be generated.
- **Authorization**: The JWT token for authentication.
- **X-API-Key**: The API key for authentication.
- **X-CSRF-Token**: The CSRF token for CSRF protection.

## Response Format

All responses from the Enhanced API Gateway follow a standardized format:

```json
{
  "status": "success",
  "data": {
    // Response data
  },
  "meta": {
    "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
    "request_id": "123e4567-e89b-12d3-a456-426614174001",
    "timestamp": "2023-01-01T00:00:00Z",
    "version": "1.0",
    "service": "api-gateway"
  },
  "pagination": {
    "total": 100,
    "per_page": 10,
    "page": 1,
    "pages": 10,
    "next_page": "/api/v1/resource?page=2",
    "prev_page": null
  }
}
```

For error responses, the format is:

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message",
    "details": {
      // Error details
    },
    "source": "api-gateway",
    "field": "field_name"
  },
  "meta": {
    "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
    "request_id": "123e4567-e89b-12d3-a456-426614174001",
    "timestamp": "2023-01-01T00:00:00Z",
    "version": "1.0",
    "service": "api-gateway"
  }
}
```

## Endpoints

### Health Check

```
GET /health
```

Returns the health status of the API Gateway.

#### Response

```json
{
  "status": "ok"
}
```

### Proxy Endpoints

```
GET /api/v1/{service_name}/{path}
POST /api/v1/{service_name}/{path}
PUT /api/v1/{service_name}/{path}
DELETE /api/v1/{service_name}/{path}
PATCH /api/v1/{service_name}/{path}
HEAD /api/v1/{service_name}/{path}
OPTIONS /api/v1/{service_name}/{path}
```

Proxies a request to a backend service.

#### Parameters

- **service_name**: The name of the backend service.
- **path**: The path to the endpoint on the backend service.

#### Response

The response from the backend service, formatted according to the standardized response format.

## Error Codes

The Enhanced API Gateway uses the following error codes:

- **AUTHENTICATION_ERROR**: Authentication failed.
- **AUTHORIZATION_ERROR**: Authorization failed.
- **RATE_LIMIT_EXCEEDED**: Rate limit exceeded.
- **SERVICE_NOT_FOUND**: Service not found.
- **SERVICE_UNAVAILABLE**: Service unavailable.
- **SERVICE_TIMEOUT**: Service timed out.
- **INTERNAL_SERVER_ERROR**: Internal server error.
- **BAD_REQUEST**: Bad request.
- **NOT_FOUND**: Resource not found.
- **METHOD_NOT_ALLOWED**: Method not allowed.
- **CONFLICT**: Conflict.
- **UNPROCESSABLE_ENTITY**: Unprocessable entity.
- **TOO_MANY_REQUESTS**: Too many requests.

## Rate Limiting

The Enhanced API Gateway implements rate limiting to prevent abuse. Rate limits are configured per role and per API key.

When a rate limit is exceeded, the API Gateway returns a `429 Too Many Requests` response with a `Retry-After` header indicating how many seconds to wait before retrying.

## CORS

The Enhanced API Gateway supports Cross-Origin Resource Sharing (CORS). CORS is configured in the API Gateway configuration file.

## Security

The Enhanced API Gateway implements security best practices:

- **XSS Protection**: Protects against Cross-Site Scripting attacks.
- **CSRF Protection**: Protects against Cross-Site Request Forgery attacks.
- **Security Headers**: Sets security headers to protect against various attacks.

## Versioning

The Enhanced API Gateway supports API versioning through the URL path. The current version is `v1`.

## Pagination

The Enhanced API Gateway supports pagination for list endpoints. Pagination parameters are passed as query parameters:

- **page**: The page number (default: 1).
- **per_page**: The number of items per page (default: 10).

Pagination information is included in the response metadata.

## Filtering

The Enhanced API Gateway supports filtering for list endpoints. Filter parameters are passed as query parameters.

## Sorting

The Enhanced API Gateway supports sorting for list endpoints. Sort parameters are passed as query parameters:

- **sort**: The field to sort by.
- **order**: The sort order (`asc` or `desc`, default: `asc`).

## Examples

### Get Market Data

```
GET /api/v1/market-data/ohlcv?symbol=EURUSD&timeframe=1h
```

#### Response

```json
{
  "status": "success",
  "data": {
    "symbol": "EURUSD",
    "timeframe": "1h",
    "ohlcv": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.05,
        "volume": 1000
      }
    ]
  },
  "meta": {
    "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
    "request_id": "123e4567-e89b-12d3-a456-426614174001",
    "timestamp": "2023-01-01T00:00:00Z",
    "version": "1.0",
    "service": "api-gateway"
  }
}
```

### Create Order

```
POST /api/v1/trading/orders
```

#### Request Body

```json
{
  "symbol": "EURUSD",
  "type": "market",
  "side": "buy",
  "quantity": 1.0
}
```

#### Response

```json
{
  "status": "success",
  "data": {
    "order_id": "123e4567-e89b-12d3-a456-426614174000",
    "symbol": "EURUSD",
    "type": "market",
    "side": "buy",
    "quantity": 1.0,
    "price": 1.05,
    "status": "filled",
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-01T00:00:00Z"
  },
  "meta": {
    "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
    "request_id": "123e4567-e89b-12d3-a456-426614174001",
    "timestamp": "2023-01-01T00:00:00Z",
    "version": "1.0",
    "service": "api-gateway"
  }
}
```

### Error Response

```
GET /api/v1/non-existent-service/resource
```

#### Response

```json
{
  "status": "error",
  "error": {
    "code": "SERVICE_NOT_FOUND",
    "message": "Service non-existent-service not found",
    "source": "api-gateway"
  },
  "meta": {
    "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
    "request_id": "123e4567-e89b-12d3-a456-426614174001",
    "timestamp": "2023-01-01T00:00:00Z",
    "version": "1.0",
    "service": "api-gateway"
  }
}
```