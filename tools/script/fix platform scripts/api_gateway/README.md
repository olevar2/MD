# API Gateway Testing Scripts

This directory contains scripts for testing the Enhanced API Gateway.

## Scripts

- `test_api_gateway.py`: Tests the Enhanced API Gateway by sending requests to various endpoints and verifying the responses.

## Usage

### Test API Gateway

```bash
python test_api_gateway.py --base-url http://localhost:8000 --config-path ../../../../api-gateway/config/api-gateway-enhanced.yaml
```

#### Arguments

- `--base-url`: Base URL of the API Gateway (default: http://localhost:8000)
- `--config-path`: Path to the API Gateway configuration file (default: config/api-gateway-enhanced.yaml)

#### Environment Variables

- `JWT_SECRET_KEY`: Secret key for JWT token generation (default: test_secret_key)
- `API_KEY`: API key for testing (default: test_api_key)

#### Tests

The script runs the following tests:

1. **Health Check**: Tests the health check endpoint.
2. **Authentication**: Tests JWT and API key authentication.
3. **Rate Limiting**: Tests rate limiting functionality.
4. **Proxy**: Tests proxy functionality.
5. **Error Handling**: Tests error handling.

#### Output

The script outputs the results of each test and an overall summary.

Example:

```
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Testing health check endpoint...
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Health check test passed
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Testing authentication...
2023-01-01 00:00:00,000 - test_api_gateway - INFO - JWT authentication test passed
2023-01-01 00:00:00,000 - test_api_gateway - INFO - API key authentication test passed
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Authentication failure test passed
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Testing rate limiting...
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Rate limiting triggered after 100 requests
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Testing proxy functionality...
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Proxy test passed
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Testing error handling...
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Error handling test passed
2023-01-01 00:00:00,000 - test_api_gateway - INFO - Test summary:
2023-01-01 00:00:00,000 - test_api_gateway - INFO -   Test 1: PASSED
2023-01-01 00:00:00,000 - test_api_gateway - INFO -   Test 2: PASSED
2023-01-01 00:00:00,000 - test_api_gateway - INFO -   Test 3: PASSED
2023-01-01 00:00:00,000 - test_api_gateway - INFO -   Test 4: PASSED
2023-01-01 00:00:00,000 - test_api_gateway - INFO -   Test 5: PASSED
```

## Requirements

- Python 3.8 or higher
- httpx
- PyJWT
- PyYAML