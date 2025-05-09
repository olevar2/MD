# Trading Gateway Service Tests

This directory contains tests for the Trading Gateway Service.

## Test Structure

- `test_api.py`: Tests for the Python API endpoints
- `test_error_handling.py`: Tests for the Python error handling functionality
- `test_error_handlers.py`: Tests for the Python error handlers
- `test_error_bridge.js`: Tests for the JavaScript-Python error bridge
- `test_error_handler_middleware.js`: Tests for the JavaScript error handler middleware

## Running Tests

### Python Tests

To run the Python tests:

```bash
cd trading-gateway-service
python -m pytest tests/
```

To run a specific test file:

```bash
python -m pytest tests/test_api.py
```

### JavaScript Tests

To run the JavaScript tests:

```bash
cd trading-gateway-service
npm test
```

To run a specific test file:

```bash
npm test -- tests/test_error_bridge.js
```

## Test Coverage

To generate a test coverage report for Python tests:

```bash
python -m pytest --cov=trading_gateway_service tests/
```

To generate a test coverage report for JavaScript tests:

```bash
npm run test:coverage
```
