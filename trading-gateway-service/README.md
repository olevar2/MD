# Trading Gateway Service

## Overview
The Trading Gateway Service serves as the interface between the Forex Trading Platform and various external trading providers and exchanges. It provides a unified API for executing trades, retrieving market data, and managing orders across multiple brokers and liquidity providers.

## Setup

### Prerequisites
- Node.js 18.x or higher (for JavaScript components)
- Python 3.10 or higher (for Python components)
- Poetry (dependency management for Python)
- npm (dependency management for JavaScript)
- Network connectivity to trading providers
- API credentials for supported brokers

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd trading-gateway-service
```
3. Install JavaScript dependencies:
```bash
npm install
```
4. Install Python dependencies using Poetry:
```bash
poetry install
```

### Environment Variables
The following environment variables are required (refer to `.env.example` for a complete list):

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (debug, info, warn, error) | info |
| `PORT` | Service port | 8004 |
| `BROKER_API_KEYS` | JSON string containing broker API keys | - |
| `DEFAULT_BROKER` | Default broker to use | - |
| `API_KEY` | API key for internal authentication | - |
| `JWT_SECRET` | Secret key for JWT authentication | - |
| `ANALYSIS_ENGINE_API_KEY` | API key for analysis-engine-service | - |
| `PORTFOLIO_API_KEY` | API key for portfolio-management-service | - |
| `FEATURE_STORE_API_KEY` | API key for feature-store-service | - |
| `RISK_MANAGEMENT_API_KEY` | API key for risk-management-service | - |
| `ORDER_TIMEOUT_MS` | Order timeout in milliseconds | 5000 |
| `RETRY_ATTEMPTS` | Max number of retry attempts | 3 |
| `REDIS_URL` | Redis connection string for caching | redis://localhost:6379 |

For local development:
1. Copy `.env.example` to `.env` 
2. Update all sensitive values with your actual credentials
3. Never commit `.env` files to version control

### Running the Service
Run the JavaScript components:
```bash
npm start
```

For development with auto-reload:
```bash
npm run dev
```

To run Python components:
```bash
poetry run python -m trading_gateway_service.market_data_processor
```

## API Documentation

### Endpoints

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

#### GET /api/v1/brokers
List all configured brokers.

**Response:**
```json
{
  "brokers": [
    {
      "id": "broker1",
      "name": "Primary FX Provider",
      "status": "connected",
      "instruments": ["EUR/USD", "GBP/USD", "USD/JPY"]
    },
    {
      "id": "broker2",
      "name": "Secondary FX Provider",
      "status": "connected",
      "instruments": ["EUR/USD", "USD/CHF", "AUD/USD"]
    }
  ]
}
```

#### GET /api/v1/quotes/{symbol}
Get real-time quotes for a symbol.

**Parameters:**
- `symbol` (path): The trading symbol (e.g., "EUR/USD")
- `broker` (query, optional): Broker ID to use

**Response:**
```json
{
  "symbol": "EUR/USD",
  "bid": 1.1012,
  "ask": 1.1014,
  "spread": 0.0002,
  "timestamp": "2025-04-29T14:25:36.123Z",
  "broker": "broker1"
}
```

#### POST /api/v1/orders
Submit a new order.

**Request Body:**
```json
{
  "symbol": "EUR/USD",
  "side": "buy",
  "type": "market",
  "quantity": 100000,
  "price": 1.1015,
  "stop_loss": 1.0990,
  "take_profit": 1.1050,
  "broker": "broker1"
}
```

**Response:**
```json
{
  "order_id": "b1_ord123456",
  "status": "filled",
  "symbol": "EUR/USD",
  "side": "buy",
  "type": "market",
  "quantity": 100000,
  "executed_price": 1.1014,
  "timestamp": "2025-04-29T14:25:40.321Z"
}
```

#### GET /api/v1/orders
Get all active orders.

**Response:**
```json
{
  "orders": [
    {
      "order_id": "b1_ord123456",
      "status": "filled",
      "symbol": "EUR/USD",
      "side": "buy",
      "quantity": 100000,
      "executed_price": 1.1014,
      "timestamp": "2025-04-29T14:25:40.321Z"
    },
    {
      "order_id": "b1_ord123457",
      "status": "pending",
      "symbol": "GBP/USD",
      "side": "sell",
      "type": "limit",
      "quantity": 50000,
      "price": 1.2750,
      "timestamp": "2025-04-29T14:27:12.546Z"
    }
  ]
}
```

## Supported Brokers
The Trading Gateway Service supports multiple forex brokers and providers:

1. **Primary FX Providers**: 
   - OANDA
   - FXCM
   - Interactive Brokers

2. **Secondary FX Providers**:
   - Saxo Bank
   - IG Markets
   - Forex.com

## Architecture
The service is built using a hybrid architecture:
- Node.js for the core API and order processing
- Python components for market data processing and analysis
- Shared libraries for security and common functions

## Integration with Other Services
The Trading Gateway Service integrates with:

- Strategy Execution Engine for executing automated trading strategies
- Risk Management Service for pre-trade risk checks
- Portfolio Management Service for position management
- Monitoring & Alerting Service for operation metrics

## Error Handling
The service uses standardized error responses and implements retry logic using `common_lib.resilience.retry_with_policy` for Python components and equivalent patterns from `common-js-lib` for JavaScript components.

## Security
- Authentication is handled using API keys through the `common-js-lib` security module
- All sensitive information (API keys, JWT secrets, broker credentials) is loaded via environment variables
- No hardcoded secrets in the codebase
- CORS is configured to restrict access to approved origins
- Broker credentials are stored securely and not exposed in API responses

For local development:
1. Copy `.env.example` to `.env`
2. Update with your actual API keys and secrets
3. Never commit `.env` files to the repository
