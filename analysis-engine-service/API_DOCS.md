# Analysis Engine Service API Documentation

## Overview

The Analysis Engine Service provides advanced technical analysis capabilities for the Forex Trading Platform. It offers a comprehensive suite of analytical tools, including pattern recognition, technical indicators, and data transformations.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## Error Handling

The service uses standard HTTP status codes and returns error responses in the following format:

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable error message",
        "details": {
            "field": "Additional error details"
        }
    }
}
```

Common error codes:
- `400`: Bad Request - Invalid input parameters
- `401`: Unauthorized - Missing or invalid authentication
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource not found
- `422`: Validation Error - Invalid data format
- `500`: Internal Server Error - Server-side error

## Endpoints

### Analysis Endpoints

#### Run Confluence Analysis

Analyzes market data to identify confluence zones where multiple technical factors align.

```http
POST /analysis/confluence
```

Request body:
```json
{
    "symbol": "EURUSD",
    "timeframe": "H1",
    "market_data": {
        "open": [...],
        "high": [...],
        "low": [...],
        "close": [...],
        "volume": [...],
        "timestamp": [...]
    },
    "parameters": {
        "min_tools_for_confluence": 2,
        "effectiveness_threshold": 0.5,
        "sr_proximity_threshold": 0.0015
    }
}
```

Response:
```json
{
    "timestamp": "2024-03-14T12:00:00Z",
    "symbol": "EURUSD",
    "current_price": 1.09234,
    "confluence_zones": [
        {
            "price_level": 1.09200,
            "zone_width": 0.00020,
            "upper_bound": 1.09210,
            "lower_bound": 1.09190,
            "confluence_types": ["support_resistance", "fibonacci_alignment"],
            "strength": 3,
            "strength_name": "STRONG",
            "contributing_tools": ["support_resistance_H1", "fibonacci_H1"],
            "timeframes": ["H1"],
            "direction": "bullish",
            "expected_reaction": "bounce"
        }
    ],
    "market_regime": "TRENDING",
    "effective_tools": {
        "support_resistance_H1": 0.85,
        "fibonacci_H1": 0.75
    }
}
```

#### Run Multi-Timeframe Analysis

Analyzes market data across multiple timeframes to identify significant patterns and levels.

```http
POST /analysis/multi-timeframe
```

Request body:
```json
{
    "symbol": "EURUSD",
    "timeframes": ["M15", "H1", "H4", "D1"],
    "market_data": {
        "M15": {
            "open": [...],
            "high": [...],
            "low": [...],
            "close": [...],
            "volume": [...],
            "timestamp": [...]
        },
        "H1": {
            // Similar structure
        },
        "H4": {
            // Similar structure
        },
        "D1": {
            // Similar structure
        }
    },
    "parameters": {
        "correlation_threshold": 0.7,
        "min_timeframes": 2
    }
}
```

Response:
```json
{
    "timestamp": "2024-03-14T12:00:00Z",
    "symbol": "EURUSD",
    "timeframe_analysis": {
        "M15": {
            "trend": "bullish",
            "strength": 0.8,
            "key_levels": [...]
        },
        "H1": {
            "trend": "bullish",
            "strength": 0.9,
            "key_levels": [...]
        },
        "H4": {
            "trend": "bullish",
            "strength": 0.7,
            "key_levels": [...]
        },
        "D1": {
            "trend": "neutral",
            "strength": 0.5,
            "key_levels": [...]
        }
    },
    "correlation_matrix": {
        "M15_H1": 0.85,
        "M15_H4": 0.75,
        "M15_D1": 0.45,
        "H1_H4": 0.82,
        "H1_D1": 0.48,
        "H4_D1": 0.65
    },
    "overall_assessment": {
        "trend": "bullish",
        "strength": 0.8,
        "confidence": 0.85,
        "key_levels": [...]
    }
}
```

### Feedback System Endpoints

#### Submit Trading Feedback

Submit feedback about trading performance for analysis and improvement.

```http
POST /feedback/trading
```

Request body:
```json
{
    "symbol": "EURUSD",
    "timeframe": "H1",
    "trade_id": "TRADE_123",
    "entry_price": 1.09200,
    "exit_price": 1.09500,
    "position": "long",
    "profit_loss": 30.0,
    "analysis_used": {
        "confluence_zones": ["zone_1", "zone_2"],
        "indicators": ["RSI", "MACD"],
        "patterns": ["double_bottom"]
    },
    "feedback": {
        "accuracy": 0.8,
        "usefulness": 0.9,
        "comments": "Analysis was accurate but entry timing could be improved"
    }
}
```

Response:
```json
{
    "message": "Feedback submitted successfully",
    "feedback_id": "unique_feedback_id"
}
```

#### Get Feedback Insights

Retrieve insights derived from feedback analysis.

```http
GET /feedback/insights
```

*Query Parameters:*
- `source`: Filter by feedback source (e.g., 'manual', 'automated')
- `category`: Filter by feedback category (e.g., 'signal_quality', 'model_performance')
- `instrument_id`: Filter by instrument
- `start_date`: Filter by start date (ISO format)
- `end_date`: Filter by end date (ISO format)
- `limit`: Maximum number of insights to return (default 100)

*Response:* A list of `FeedbackInsight` objects.

#### Get Feedback Statistics

Retrieve aggregated statistics about the collected feedback.

```http
GET /feedback/statistics
```

*Query Parameters:*
- `strategy_id`: Filter by strategy ID
- `model_id`: Filter by model ID
- `instrument`: Filter by instrument
- `start_time`: Filter by start time (ISO format)
- `end_time`: Filter by end time (ISO format)

*Response:* `FeedbackStatistics` object containing counts and distributions.

#### Trigger Model Retraining (Example - May reside elsewhere)

Endpoint to trigger model retraining based on feedback.
*(Note: Actual implementation might be in ML Integration Service)*

```http
POST /feedback/trigger-retraining
```

*Request Body:*
```json
{
    "model_id": "model_to_retrain",
    "feedback_criteria": {
        "min_negative_feedback": 10,
        "time_window_hours": 24
    }
}
```

*Response:*
```json
{
    "message": "Retraining job triggered for model_id",
    "job_id": "retraining_job_123"
}
```

```

### System Endpoints

#### Health Check

Check the health status of the service.

```http
GET /health
```

Response:
```json
{
    "status": "healthy",
    "timestamp": "2024-03-14T12:00:00Z",
    "components": {
        "api": "healthy",
        "database": "healthy",
        "cache": "healthy",
        "message_queue": "healthy"
    },
    "metrics": {
        "uptime": "5d 12h 30m",
        "memory_usage": "45%",
        "cpu_usage": "30%",
        "active_connections": 25
    }
}
```

#### Service Status

Get detailed status information about the service.

```http
GET /status
```

Response:
```json
{
    "version": "1.0.0",
    "environment": "production",
    "start_time": "2024-03-09T00:00:00Z",
    "components": {
        "analysis_engine": {
            "status": "active",
            "active_analyzers": 5,
            "queue_size": 10
        },
        "feedback_system": {
            "status": "active",
            "pending_feedback": 3,
            "processing_rate": "50/min"
        },
        "data_pipeline": {
            "status": "active",
            "last_update": "2024-03-14T11:55:00Z",
            "data_freshness": "5m"
        }
    },
    "performance": {
        "request_rate": "100/min",
        "average_response_time": "150ms",
        "error_rate": "0.1%"
    }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage and system stability:

- Standard tier: 100 requests per minute
- Premium tier: 500 requests per minute

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1615723200
```

## WebSocket Endpoints

### Real-time Analysis Updates

Connect to receive real-time analysis updates.

```http
GET /ws/analysis/updates
```

Query parameters:
- `symbols`: Comma-separated list of symbols to subscribe to
- `timeframes`: Comma-separated list of timeframes to subscribe to

Example message:
```json
{
    "type": "confluence_update",
    "timestamp": "2024-03-14T12:00:00Z",
    "symbol": "EURUSD",
    "timeframe": "H1",
    "data": {
        "confluence_zones": [...],
        "market_regime": "TRENDING",
        "effective_tools": {...}
    }
}
```

## Data Models

### MarketData

```typescript
interface MarketData {
    symbol: string;
    timeframe: string;
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume: number[];
    timestamp: string[];
    metadata?: {
        pip_value?: number;
        contract_size?: number;
        [key: string]: any;
    };
}
```

### AnalysisResult

```typescript
interface AnalysisResult {
    analyzer_name: string;
    timestamp: string;
    symbol: string;
    timeframe: string;
    result: {
        [key: string]: any;
    };
    is_valid: boolean;
    metadata?: {
        processing_time?: number;
        confidence?: number;
        [key: string]: any;
    };
}
```

### FeedbackData

```typescript
interface FeedbackData {
    symbol: string;
    timeframe: string;
    trade_id: string;
    entry_price: number;
    exit_price: number;
    position: "long" | "short";
    profit_loss: number;
    analysis_used: {
        confluence_zones?: string[];
        indicators?: string[];
        patterns?: string[];
        [key: string]: any;
    };
    feedback: {
        accuracy: number;
        usefulness: number;
        comments?: string;
        [key: string]: any;
    };
}
```

## Best Practices

1. **Error Handling**
   - Always check for error responses
   - Implement proper retry logic for transient errors
   - Handle rate limiting appropriately

2. **Performance**
   - Use appropriate timeframes for analysis
   - Implement caching for frequently accessed data
   - Batch requests when possible

3. **Security**
   - Keep API tokens secure
   - Use HTTPS for all requests
   - Validate all input data

4. **Monitoring**
   - Monitor rate limits
   - Track response times
   - Log errors appropriately

## Examples

### Python Example

```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"
TOKEN = "your_jwt_token"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Run confluence analysis
def run_confluence_analysis(symbol, timeframe, market_data):
    url = f"{BASE_URL}/analysis/confluence"
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "market_data": market_data
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Submit feedback
def submit_feedback(feedback_data):
    url = f"{BASE_URL}/feedback/trading"
    response = requests.post(url, headers=headers, json=feedback_data)
    response.raise_for_status()
    return response.json()
```

### JavaScript Example

```javascript
const BASE_URL = 'http://localhost:8000/api/v1';
const TOKEN = 'your_jwt_token';

const headers = {
    'Authorization': `Bearer ${TOKEN}`,
    'Content-Type': 'application/json'
};

// Run confluence analysis
async function runConfluenceAnalysis(symbol, timeframe, marketData) {
    const url = `${BASE_URL}/analysis/confluence`;
    const payload = {
        symbol,
        timeframe,
        market_data: marketData
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

// Submit feedback
async function submitFeedback(feedbackData) {
    const url = `${BASE_URL}/feedback/trading`;
    const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(feedbackData)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}
```

## Support

For API support, please contact:
- Email: api-support@forextradingplatform.com
- Documentation: https://docs.forextradingplatform.com
- Status Page: https://status.forextradingplatform.com