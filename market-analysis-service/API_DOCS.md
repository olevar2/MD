# Market Analysis Service API Documentation

The Market Analysis Service provides comprehensive market analysis capabilities for forex trading, including pattern recognition, support/resistance detection, market regime detection, correlation analysis, volatility analysis, and sentiment analysis.

## Base URL

```
http://localhost:8000/api/v1/market-analysis
```

## Endpoints

### Comprehensive Market Analysis

**Endpoint:** `/analyze`

**Method:** `POST`

**Description:** Perform comprehensive market analysis, including multiple analysis types in a single request.

**Request Body:**

```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "analysis_types": ["technical", "pattern", "support_resistance", "market_regime", "volatility", "sentiment"],
  "additional_parameters": {
    "symbols": ["GBP/USD", "USD/JPY"],
    "window_sizes": [5, 10, 20, 50]
  }
}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "analysis_results": [
    {
      "analysis_type": "technical",
      "result": {
        "indicators": {
          "rsi": [50.1, 52.3, 48.7, ...],
          "macd": [0.001, 0.002, -0.001, ...],
          "macd_signal": [0.0005, 0.001, 0.0015, ...]
        }
      },
      "confidence": 0.8,
      "execution_time_ms": 150
    },
    {
      "analysis_type": "pattern",
      "result": {
        "patterns": [
          {
            "pattern_type": "head_and_shoulders",
            "start_index": 10,
            "end_index": 30,
            "confidence": 0.85,
            "target_price": 1.0850,
            "stop_loss": 1.1050,
            "risk_reward_ratio": 2.0
          }
        ]
      },
      "confidence": 0.85,
      "execution_time_ms": 200
    }
  ],
  "execution_time_ms": 1200,
  "timestamp": "2023-02-01T12:34:56Z"
}
```

### Pattern Recognition

**Endpoint:** `/patterns`

**Method:** `POST`

**Description:** Recognize chart patterns in market data.

**Request Body:**

```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "pattern_types": ["head_and_shoulders", "double_top", "double_bottom"],
  "min_confidence": 0.7
}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440001",
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "patterns": [
    {
      "pattern_type": "head_and_shoulders",
      "start_index": 10,
      "end_index": 30,
      "confidence": 0.85,
      "target_price": 1.0850,
      "stop_loss": 1.1050,
      "risk_reward_ratio": 2.0,
      "metadata": {
        "left_shoulder_idx": 15,
        "head_idx": 20,
        "right_shoulder_idx": 25,
        "neckline": 1.0950
      }
    }
  ],
  "execution_time_ms": 200,
  "timestamp": "2023-02-01T12:34:56Z"
}
```

### Support and Resistance Detection

**Endpoint:** `/support-resistance`

**Method:** `POST`

**Description:** Identify support and resistance levels.

**Request Body:**

```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "methods": ["price_swings", "moving_average", "fibonacci"],
  "levels_count": 5
}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440002",
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "levels": [
    {
      "price": 1.1050,
      "type": "resistance",
      "strength": 0.85,
      "method": "price_swings",
      "touches": 5,
      "last_touch_date": "2023-01-25T14:00:00Z"
    },
    {
      "price": 1.0950,
      "type": "support",
      "strength": 0.90,
      "method": "price_swings",
      "touches": 7,
      "last_touch_date": "2023-01-28T10:00:00Z"
    }
  ],
  "execution_time_ms": 150,
  "timestamp": "2023-02-01T12:34:56Z"
}
```

### Market Regime Detection

**Endpoint:** `/market-regime`

**Method:** `POST`

**Description:** Detect market regime (trending, ranging, volatile).

**Request Body:**

```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "window_size": 20
}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440003",
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "regimes": [
    {
      "regime_type": "trending_up",
      "start_index": 0,
      "end_index": 50,
      "start_date": "2023-01-01T00:00:00Z",
      "end_date": "2023-01-10T02:00:00Z",
      "confidence": 0.85,
      "metadata": {
        "price_change": 0.0150
      }
    }
  ],
  "current_regime": "trending_up",
  "execution_time_ms": 180,
  "timestamp": "2023-02-01T12:34:56Z"
}
```

### Correlation Analysis

**Endpoint:** `/correlation`

**Method:** `POST`

**Description:** Analyze correlations between symbols.

**Request Body:**

```json
{
  "symbols": ["EUR/USD", "GBP/USD", "USD/JPY"],
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "window_size": 20,
  "method": "pearson"
}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440004",
  "symbols": ["EUR/USD", "GBP/USD", "USD/JPY"],
  "timeframe": "1h",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "method": "pearson",
  "correlation_matrix": {
    "EUR/USD": {
      "EUR/USD": 1.0,
      "GBP/USD": 0.85,
      "USD/JPY": -0.65
    },
    "GBP/USD": {
      "EUR/USD": 0.85,
      "GBP/USD": 1.0,
      "USD/JPY": -0.55
    },
    "USD/JPY": {
      "EUR/USD": -0.65,
      "GBP/USD": -0.55,
      "USD/JPY": 1.0
    }
  },
  "correlation_pairs": [
    {
      "symbol1": "EUR/USD",
      "symbol2": "GBP/USD",
      "correlation": 0.85,
      "p_value": 0.001
    }
  ],
  "execution_time_ms": 250,
  "timestamp": "2023-02-01T12:34:56Z"
}
```

### Volatility Analysis

**Endpoint:** `/volatility`

**Method:** `POST`

**Description:** Analyze market volatility.

**Request Parameters:**

- `symbol`: Symbol to analyze
- `timeframe`: Timeframe for analysis (e.g., '1h', '1d')
- `start_date`: Start date for analysis (ISO format)
- `end_date`: End date for analysis (ISO format)
- `window_sizes`: Window sizes for volatility calculation
- `additional_parameters`: Additional parameters for analysis

**Response:**

```json
{
  "volatility": {
    "5": {
      "current": 0.12,
      "average": 0.10,
      "percentile": 75.0
    },
    "20": {
      "current": 0.15,
      "average": 0.12,
      "percentile": 80.0
    }
  },
  "regimes": {
    "current_regime": "medium",
    "regime_thresholds": [0.08, 0.15]
  },
  "forecasts": {
    "forecast": 0.14,
    "confidence_interval": {
      "lower": 0.11,
      "upper": 0.17
    }
  },
  "term_structure": [
    {
      "window": 5,
      "volatility": 0.12
    },
    {
      "window": 20,
      "volatility": 0.15
    }
  ],
  "execution_time_ms": 180
}
```

### Sentiment Analysis

**Endpoint:** `/sentiment`

**Method:** `POST`

**Description:** Analyze market sentiment.

**Request Parameters:**

- `symbol`: Symbol to analyze
- `timeframe`: Timeframe for analysis (e.g., '1h', '1d')
- `start_date`: Start date for analysis (ISO format)
- `end_date`: End date for analysis (ISO format)
- `additional_parameters`: Additional parameters for analysis

**Response:**

```json
{
  "technical_sentiment": {
    "sentiment": 0.65,
    "indicators": {
      "ma_crossover": 1,
      "rsi": {
        "value": 65.0,
        "sentiment": 0.5
      },
      "macd": {
        "value": 0.002,
        "signal": 0.001,
        "sentiment": 1
      }
    }
  },
  "price_sentiment": {
    "sentiment": 0.70,
    "components": {
      "momentum": {
        "value": 0.002,
        "sentiment": 0.8
      },
      "trend": {
        "value": 0.015,
        "sentiment": 0.7
      }
    }
  },
  "combined_sentiment": {
    "sentiment": 0.65,
    "category": "bullish",
    "components": {
      "technical": {
        "value": 0.65,
        "weight": 0.4
      },
      "price": {
        "value": 0.70,
        "weight": 0.4
      },
      "external": {
        "value": 0.55,
        "weight": 0.2
      }
    }
  },
  "execution_time_ms": 200
}
```

### Available Patterns

**Endpoint:** `/available-patterns`

**Method:** `GET`

**Description:** Get available chart patterns for recognition.

**Response:**

```json
[
  {
    "id": "head_and_shoulders",
    "name": "HEAD_AND_SHOULDERS",
    "description": "A reversal pattern with three peaks, the middle one being the highest",
    "min_bars": 30
  },
  {
    "id": "double_top",
    "name": "DOUBLE_TOP",
    "description": "A reversal pattern with two peaks at approximately the same level",
    "min_bars": 20
  }
]
```

### Available Regimes

**Endpoint:** `/available-regimes`

**Method:** `GET`

**Description:** Get available market regimes for detection.

**Response:**

```json
[
  {
    "id": "trending_up",
    "name": "TRENDING_UP",
    "description": "Market is in an uptrend"
  },
  {
    "id": "trending_down",
    "name": "TRENDING_DOWN",
    "description": "Market is in a downtrend"
  }
]
```

### Available Methods

**Endpoint:** `/available-methods`

**Method:** `GET`

**Description:** Get available analysis methods.

**Response:**

```json
{
  "pattern_recognition": ["head_and_shoulders", "double_top", "double_bottom"],
  "support_resistance": ["price_swings", "moving_average", "fibonacci"],
  "market_regime": ["trending_up", "trending_down", "ranging", "volatile"],
  "correlation": ["pearson", "spearman", "kendall"],
  "volatility": ["historical", "garch", "parkinson", "garman_klass"],
  "sentiment": ["technical", "price", "external", "combined"]
}
```
