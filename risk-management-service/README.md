# Risk Management Service

This service provides risk assessment, limit enforcement, and monitoring capabilities for the Forex Trading Platform.

## Features

- Risk limit definition and enforcement
- Position-level risk assessment
- Portfolio risk monitoring 
- Dynamic risk adjustment based on market conditions
- Risk alerts and notifications
- API Key authentication for secure access

## Setup and Installation

### Prerequisites

- Python 3.9+
- PostgreSQL database
- Kafka message broker

### Environment Variables

This service requires several environment variables to be set for proper operation. A template is provided in the `.env.example` file. Copy this file to create your own `.env` file:

```bash
cp .env.example .env
```

Then edit the `.env` file to set the appropriate values:

#### Database Configuration

- `RISK_DB_USER`: Database username
- `RISK_DB_PASSWORD`: Database password
- `RISK_DB_HOST`: Database host address
- `RISK_DB_PORT`: Database port (default: 5432)
- `RISK_DB_NAME`: Database name
- `RISK_DB_POOL_SIZE`: Connection pool size
- `RISK_DB_SSL_REQUIRED`: Whether SSL is required for database connection

#### API Security

- `API_KEY`: API key used for service-to-service authentication

#### Service Configuration

- `PORT`: Service port (default: 8004)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `DEBUG_MODE`: Whether to run in debug mode

#### Kafka Configuration

- `KAFKA_BOOTSTRAP_SERVERS`: Comma-separated list of Kafka broker addresses

### Installation

1. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create and configure your `.env` file as described above

4. Run the service:
```bash
python -m risk_management_service.main
```

## API Documentation

The API documentation is available at `/docs` when the service is running.

## Testing

The service includes various test suites:

- **Unit Tests**: Tests individual components in isolation
- **Integration Tests**: Tests components working together
- **Fixtures**: Provides common test data

### Running Tests

To run all tests:

```bash
pytest
```

To run tests with coverage report:

```bash
pytest --cov=risk_management_service tests/
```

To run specific test types:

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# A specific test file
pytest tests/unit/test_risk_manager.py
```

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── __init__.py          # Makes tests importable
├── fixtures/            # Test data
│   ├── __init__.py
│   ├── market_data.json
│   └── test_data.py
├── integration/         # Integration tests
│   ├── __init__.py
│   └── test_risk_integration.py
└── unit/               # Unit tests
    ├── __init__.py
    ├── test_risk_components.py
    └── test_risk_manager.py
```
