# Analysis Engine API Documentation

## Overview

The Analysis Engine API provides endpoints for performing technical analysis on forex data, including confluence detection, divergence analysis, and other multi-asset analysis functions.

## Base URL

```
https://api.forex-platform.com/analysis-engine/v1
```

## Authentication

All API requests require authentication using an API key. Include the API key in the `X-API-Key` header:

```
X-API-Key: your_api_key_here
```

## Rate Limiting

The API is rate limited to 100 requests per minute per API key. If you exceed this limit, you will receive a 429 Too Many Requests response.

## Endpoints

### Confluence Detection

Detects confluence signals across multiple currency pairs.

#### Request

```
POST /confluence
```

##### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Primary currency pair (e.g., "EURUSD") |
| signal_type | string | Yes | Type of signal ("trend", "reversal", "breakout") |
| signal_direction | string | Yes | Direction of the signal ("bullish", "bearish") |
| timeframe | string | No | Timeframe for analysis ("M1", "M5", "M15", "M30", "H1", "H4", "D1") |
| use_currency_strength | boolean | No | Whether to include currency strength in analysis (default: true) |
| min_confirmation_strength | float | No | Minimum strength for confirmation signals (0.0 to 1.0, default: 0.3) |
| related_pairs | object | No | Dictionary of related pairs and their correlations |

##### Example Request

```json
{
  "symbol": "EURUSD",
  "signal_type": "trend",
  "signal_direction": "bullish",
  "timeframe": "H1",
  "use_currency_strength": true,
  "min_confirmation_strength": 0.3,
  "related_pairs": {
    "GBPUSD": 0.85,
    "AUDUSD": 0.75,
    "USDCAD": -0.65,
    "USDJPY": -0.55
  }
}
```

#### Response

```json
{
  "symbol": "EURUSD",
  "signal_type": "trend",
  "signal_direction": "bullish",
  "timeframe": "H1",
  "confirmation_count": 2,
  "contradiction_count": 1,
  "neutral_count": 1,
  "confluence_score": 0.65,
  "confirmations": [
    {
      "pair": "GBPUSD",
      "correlation": 0.85,
      "signal_strength": 0.72,
      "expected_direction": "bullish",
      "actual_direction": "bullish"
    },
    {
      "pair": "AUDUSD",
      "correlation": 0.75,
      "signal_strength": 0.58,
      "expected_direction": "bullish",
      "actual_direction": "bullish"
    }
  ],
  "contradictions": [
    {
      "pair": "USDCAD",
      "correlation": -0.65,
      "signal_strength": 0.45,
      "expected_direction": "bearish",
      "actual_direction": "bullish"
    }
  ],
  "neutrals": [
    {
      "pair": "USDJPY",
      "correlation": -0.55,
      "signal_strength": 0.18,
      "expected_direction": "bearish",
      "actual_direction": "bearish",
      "message": "Signal strength below threshold"
    }
  ],
  "execution_time": 0.125,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### Divergence Analysis

Analyzes divergences between correlated currency pairs.

#### Request

```
POST /divergence
```

##### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Primary currency pair (e.g., "EURUSD") |
| timeframe | string | No | Timeframe for analysis ("M1", "M5", "M15", "M30", "H1", "H4", "D1") |
| related_pairs | object | No | Dictionary of related pairs and their correlations |

##### Example Request

```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "related_pairs": {
    "GBPUSD": 0.85,
    "AUDUSD": 0.75,
    "USDCAD": -0.65,
    "USDJPY": -0.55
  }
}
```

#### Response

```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "divergences_found": 2,
  "divergence_score": 0.7,
  "divergences": [
    {
      "pair": "GBPUSD",
      "correlation": 0.85,
      "primary_momentum": 0.65,
      "related_momentum": 0.25,
      "expected_momentum": 0.55,
      "momentum_difference": -0.30,
      "divergence_type": "negative",
      "divergence_strength": 0.75
    },
    {
      "pair": "USDCAD",
      "correlation": -0.65,
      "primary_momentum": 0.65,
      "related_momentum": 0.30,
      "expected_momentum": -0.42,
      "momentum_difference": 0.72,
      "divergence_type": "positive",
      "divergence_strength": 0.65
    }
  ],
  "execution_time": 0.135,
  "request_id": "b2c3d4e5-f6a7-8901-bcde-f23456789012"
}
```

### Currency Strength Analysis

Analyzes the relative strength of major currencies.

#### Request

```
GET /currency-strength
```

##### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| timeframe | string | No | Timeframe for analysis ("M1", "M5", "M15", "M30", "H1", "H4", "D1") |
| method | string | No | Strength calculation method ("momentum", "trend", "combined") |

##### Example Request

```
GET /currency-strength?timeframe=H1&method=combined
```

#### Response

```json
{
  "timeframe": "H1",
  "method": "combined",
  "currencies": {
    "EUR": 0.75,
    "USD": -0.25,
    "GBP": 0.65,
    "JPY": -0.55,
    "AUD": 0.45,
    "CAD": -0.15,
    "CHF": 0.05,
    "NZD": 0.35
  },
  "strongest": "EUR",
  "weakest": "JPY",
  "execution_time": 0.085,
  "request_id": "c3d4e5f6-a7b8-9012-cdef-3456789012ab"
}
```

### Related Pairs

Finds pairs related to a given currency pair based on correlation.

#### Request

```
GET /related-pairs/{symbol}
```

##### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Currency pair (e.g., "EURUSD") |
| min_correlation | float | No | Minimum absolute correlation (0.0 to 1.0, default: 0.5) |
| timeframe | string | No | Timeframe for correlation calculation ("M1", "M5", "M15", "M30", "H1", "H4", "D1") |

##### Example Request

```
GET /related-pairs/EURUSD?min_correlation=0.6&timeframe=H1
```

#### Response

```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "related_pairs": {
    "GBPUSD": 0.85,
    "AUDUSD": 0.75,
    "USDCAD": -0.65,
    "USDJPY": -0.55,
    "EURGBP": 0.62,
    "EURJPY": 0.78
  },
  "execution_time": 0.075,
  "request_id": "d4e5f6a7-b8c9-0123-def4-56789012abcd"
}
```

## Error Responses

### 400 Bad Request

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "symbol": "Required field missing"
    }
  },
  "request_id": "e5f6a7b8-c9d0-1234-ef56-789012abcdef"
}
```

### 401 Unauthorized

```json
{
  "error": {
    "code": "AUTHENTICATION_ERROR",
    "message": "Invalid API key"
  },
  "request_id": "f6a7b8c9-d0e1-2345-f678-9012abcdef01"
}
```

### 429 Too Many Requests

```json
{
  "error": {
    "code": "RATE_LIMIT_ERROR",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "reset_at": "2023-06-01T12:30:00Z"
    }
  },
  "request_id": "a7b8c9d0-e1f2-3456-789a-bcdef0123456"
}
```

### 500 Internal Server Error

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An internal error occurred",
    "correlation_id": "b8c9d0e1-f2a3-4567-89ab-cdef01234567"
  },
  "request_id": "c9d0e1f2-a3b4-5678-9abc-def0123456789"
}
```

## Webhooks

The API supports webhooks for asynchronous notifications of analysis results. To register a webhook:

```
POST /webhooks
```

```json
{
  "url": "https://your-server.com/webhook",
  "events": ["confluence_detected", "divergence_detected"],
  "min_score": 0.7
}
```

## SDK Libraries

Official SDK libraries are available for:

- Python: [forex-platform-python](https://github.com/olevar2/forex-platform-python)
- JavaScript: [forex-platform-js](https://github.com/olevar2/forex-platform-js)
- C#: [forex-platform-dotnet](https://github.com/olevar2/forex-platform-dotnet)

## Rate Limits and Quotas

| Plan | Requests per minute | Requests per day | Supported timeframes |
|------|---------------------|------------------|----------------------|
| Free | 10 | 1,000 | H1, H4, D1 |
| Basic | 60 | 10,000 | M15, M30, H1, H4, D1 |
| Pro | 300 | 100,000 | M1, M5, M15, M30, H1, H4, D1 |
| Enterprise | 1,000 | Unlimited | All |
