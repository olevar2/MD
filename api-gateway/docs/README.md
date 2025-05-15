# Enhanced API Gateway

The Enhanced API Gateway is a central entry point for all API requests to the Forex Trading Platform. It provides a unified interface for clients to interact with the platform's services.

## Features

- **Authentication and Authorization**: Supports multiple authentication methods (JWT, API key) and role-based access control.
- **Rate Limiting**: Implements rate limiting with support for different rate limits for different user roles and API keys.
- **Request Routing**: Routes requests to the appropriate backend services.
- **Service Discovery**: Discovers and monitors backend services.
- **Error Handling**: Provides standardized error responses.
- **Logging and Monitoring**: Logs all requests and responses, and monitors backend services.
- **Security**: Implements security best practices (CORS, XSS protection, CSRF protection, security headers).

## Architecture

The Enhanced API Gateway is built using FastAPI and follows a modular architecture:

- **Core**: Contains core functionality (authentication, rate limiting, response formatting).
- **Services**: Contains services for interacting with backend services (proxy, registry).
- **API**: Contains API routes and the main application.
- **Config**: Contains configuration files.
- **Docs**: Contains documentation.

## Configuration

The Enhanced API Gateway is configured using a YAML file (`config/api-gateway-enhanced.yaml`). The configuration includes:

- **Service Information**: Name, version, description.
- **Server Configuration**: Host, port, debug mode.
- **Logging Configuration**: Log level, format, file, rotation, retention.
- **Authentication Configuration**: Secret key, algorithm, public paths, API key paths, API keys, role permissions.
- **Rate Limiting Configuration**: Enabled, limit, window, exempt paths, role limits, API key limits.
- **CORS Configuration**: Allow origins, methods, headers, credentials, max age.
- **XSS Protection Configuration**: Enabled, exempt paths.
- **CSRF Protection Configuration**: Enabled, cookie name, header name, cookie max age, secure, same site, exempt paths.
- **Security Headers Configuration**: Enabled, headers, exempt paths.
- **Services Configuration**: URL, health check URL, health check interval, timeout, retry, circuit breaker, endpoints.

## Usage

### Starting the API Gateway

```bash
uvicorn api.app_enhanced:app --host 0.0.0.0 --port 8000
```

### Authentication

The Enhanced API Gateway supports two authentication methods:

- **JWT Authentication**: For user authentication.
- **API Key Authentication**: For service-to-service authentication.

#### JWT Authentication

To authenticate using JWT, include an `Authorization` header with a Bearer token:

```
Authorization: Bearer <token>
```

#### API Key Authentication

To authenticate using an API key, include an `X-API-Key` header:

```
X-API-Key: <api_key>
```

### Authorization

The Enhanced API Gateway implements role-based access control. Each user has one or more roles, and each role has permissions to access specific endpoints.

### Rate Limiting

The Enhanced API Gateway implements rate limiting to prevent abuse. Rate limits are configured per role and per API key.

### Request Routing

The Enhanced API Gateway routes requests to the appropriate backend services based on the request path.

### Error Handling

The Enhanced API Gateway provides standardized error responses for all errors.

## API Reference

### Health Check

```
GET /health
```

Returns the health status of the API Gateway.

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

## Development

### Prerequisites

- Python 3.8 or higher
- FastAPI
- Uvicorn
- PyYAML
- PyJWT
- httpx

### Installation

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
pytest
```

### Building Documentation

```bash
mkdocs build
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.