# Portfolio Management Service

## Overview
The Portfolio Management Service is responsible for tracking, analyzing, and optimizing forex trading portfolios within the platform. It provides position tracking, portfolio analytics, risk assessment, and performance reporting to ensure effective capital management and investment strategy evaluation.

## Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (dependency management)
- PostgreSQL database
- Redis (for caching)
- Network connectivity to other platform services

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd portfolio-management-service
```
3. Install dependencies using Poetry:
```bash
poetry install
```

### Environment Variables
The following environment variables are required (see `.env.example` for a complete list):

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `PORT` | Service port | 8006 |
| `DB_CONNECTION_STRING` | Database connection string | postgresql://user:password@localhost:5432/portfolio |
| `DB_USER` | Database username | postgres |
| `DB_PASSWORD` | Database password | - |
| `DB_HOST` | Database host | localhost |
| `DB_PORT` | Database port | 5432 |
| `DB_NAME` | Database name | portfolio |
| `REDIS_URL` | Redis connection string for caching | redis://localhost:6379 |
| `REDIS_PASSWORD` | Redis password | - |
| `TRADING_GATEWAY_URL` | URL to the Trading Gateway Service | http://trading-gateway-service:8004 |
| `RISK_MANAGEMENT_URL` | URL to the Risk Management Service | http://risk-management-service:8007 |
| `API_KEY` | API key for authentication | - |
| `SERVICE_API_KEY` | API key for this service | - |
| `TRADING_GATEWAY_API_KEY` | API key for trading gateway | - |
| `RISK_MANAGEMENT_API_KEY` | API key for risk management | - |
| `MAX_POSITIONS_PER_PORTFOLIO` | Maximum positions per portfolio | 100 |

For local development:
1. Copy `.env.example` to `.env` 
2. Update all sensitive values with your actual credentials
3. Never commit `.env` files to version control

### Running the Service
Run the service using Poetry:
```bash
poetry run python -m portfolio_management_service.main
```

For development with auto-reload:
```bash
poetry run uvicorn portfolio_management_service.main:app --reload
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

#### GET /api/v1/portfolios
Get all portfolios.

**Response:**
```json
{
  "portfolios": [
    {
      "id": "port-001",
      "name": "Main Trading Portfolio",
      "base_currency": "USD",
      "total_value": 125000.50,
      "cash_balance": 25000.50,
      "margin_used": 10000.00,
      "unrealized_pnl": 1250.75,
      "realized_pnl": 5000.25,
      "created_at": "2024-01-15T10:00:00Z"
    },
    {
      "id": "port-002",
      "name": "Test Strategy Portfolio",
      "base_currency": "USD",
      "total_value": 50000.00,
      "cash_balance": 45000.00,
      "margin_used": 5000.00,
      "unrealized_pnl": 250.25,
      "realized_pnl": 750.50,
      "created_at": "2025-02-20T14:30:00Z"
    }
  ]
}
```

#### GET /api/v1/portfolios/{portfolio_id}
Get portfolio details.

**Parameters:**
- `portfolio_id` (path): The portfolio ID

**Response:**
```json
{
  "id": "port-001",
  "name": "Main Trading Portfolio",
  "base_currency": "USD",
  "total_value": 125000.50,
  "cash_balance": 25000.50,
  "margin_used": 10000.00,
  "unrealized_pnl": 1250.75,
  "realized_pnl": 5000.25,
  "creation_date": "2024-01-15T10:00:00Z",
  "positions": [
    {
      "id": "pos-001",
      "symbol": "EUR/USD",
      "direction": "long",
      "size": 100000,
      "open_price": 1.1015,
      "current_price": 1.1055,
      "unrealized_pnl": 400.00,
      "realized_pnl": 0.00,
      "open_date": "2025-04-25T09:15:30Z"
    },
    {
      "id": "pos-002",
      "symbol": "GBP/USD",
      "direction": "short",
      "size": 50000,
      "open_price": 1.2450,
      "current_price": 1.2380,
      "unrealized_pnl": 350.00,
      "realized_pnl": 0.00,
      "open_date": "2025-04-26T11:30:45Z"
    }
  ],
  "performance": {
    "daily_returns": [0.15, 0.22, -0.05, 0.18],
    "monthly_returns": [1.8, 2.1, 1.5],
    "sharpe_ratio": 1.35,
    "drawdown": -2.3,
    "volatility": 0.75
  }
}
```

#### POST /api/v1/portfolios
Create a new portfolio.

**Request Body:**
```json
{
  "name": "New Strategy Portfolio",
  "base_currency": "USD",
  "initial_capital": 100000.00,
  "description": "Portfolio for testing mean reversion strategies",
  "risk_profile": "moderate"
}
```

#### PUT /api/v1/portfolios/{portfolio_id}/allocate
Allocate capital to a strategy.

**Parameters:**
- `portfolio_id` (path): The portfolio ID

**Request Body:**
```json
{
  "strategy_id": "trend_following_v1",
  "amount": 25000.00,
  "max_drawdown": 10.0,
  "auto_rebalance": true
}
```

#### GET /api/v1/portfolios/{portfolio_id}/performance
Get detailed portfolio performance.

**Parameters:**
- `portfolio_id` (path): The portfolio ID
- `start_date` (query): Start date in ISO format
- `end_date` (query): End date in ISO format
- `period` (query): Period for aggregation (daily, weekly, monthly)

**Response:**
```json
{
  "portfolio_id": "port-001",
  "name": "Main Trading Portfolio",
  "period": "daily",
  "start_date": "2025-01-01T00:00:00Z",
  "end_date": "2025-04-29T00:00:00Z",
  "metrics": {
    "total_return": 12.5,
    "annualized_return": 15.8,
    "volatility": 4.2,
    "sharpe_ratio": 1.35,
    "max_drawdown": -5.2,
    "win_rate": 62.5,
    "profit_factor": 1.8
  },
  "returns": [
    {"date": "2025-01-02", "return": 0.15},
    {"date": "2025-01-03", "return": 0.22},
    {"date": "2025-01-04", "return": -0.05},
    // ... more dates
    {"date": "2025-04-29", "return": 0.18}
  ],
  "drawdowns": [
    {"start_date": "2025-02-15", "end_date": "2025-02-28", "depth": -5.2, "recovery_days": 12},
    {"start_date": "2025-03-20", "end_date": "2025-03-25", "depth": -3.1, "recovery_days": 5}
  ]
}
```

## Core Features

### Position Management
- Real-time position tracking
- Position sizing and adjustment
- Position aggregation across instruments
- Margin requirement calculation
- Stop-loss and take-profit management

### Portfolio Analytics
- Performance metrics (returns, Sharpe ratio, drawdown, etc.)
- Risk analysis (VaR, Expected Shortfall)
- Correlation analysis
- Attribution analysis
- Custom reporting periods

### Capital Allocation
- Strategy allocation framework
- Dynamic capital rebalancing
- Risk-based position sizing
- Multi-currency support
- Margin optimization

### Performance Reporting
- Daily/weekly/monthly reports
- Custom performance dashboards
- Export capabilities (CSV, PDF)
- Benchmark comparison
- Tax-related reporting

## Integration with Other Services
The Portfolio Management Service integrates with:

- Trading Gateway Service for order execution and position data
- Risk Management Service for risk controls and limits
- Strategy Execution Engine for strategy allocation
- Monitoring & Alerting Service for performance tracking

## Database Structure
The service uses a PostgreSQL database with the following main tables:
- `portfolios`: Portfolio metadata
- `positions`: Open and closed positions
- `transactions`: Individual transactions
- `allocations`: Strategy allocations
- `performance`: Historical performance data

## Error Handling
The service uses standardized error responses from `common_lib.exceptions` and implements database connection logic using `common_lib.database`.

## Security
- Authentication is handled using API keys through `common_lib.security`
- All sensitive information (database credentials, API keys, Redis passwords) is loaded via environment variables
- No hardcoded secrets in the codebase
- CORS is configured to restrict access to approved origins
- Database connections use encrypted connections when SSL is enabled
- Service-to-service communication requires API key authentication
- Redis connections support authentication via password
