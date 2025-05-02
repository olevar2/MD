# Security Service

## Overview
The Security Service is a critical component of the Forex Trading Platform that implements platform-wide authentication, authorization, audit logging, and security best practices. It provides centralized security mechanisms to protect user data, control access, and maintain platform security across all services.

## Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 18.x or higher (for JavaScript components)
- Redis (for session management)
- Certificate management tools
- Network connectivity to all platform services

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd security
```
3. Install dependencies:
```bash
# For Python components
pip install -r requirements.txt

# For JavaScript components
cd api
npm install
```

### Environment Variables
The following environment variables are required:

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `PORT` | API service port | 8008 |
| `JWT_SECRET` | Secret for JWT token signing | - |
| `JWT_EXPIRATION_MINUTES` | JWT token expiration in minutes | 60 |
| `API_KEY_SALT` | Salt for API key hashing | - |
| `REDIS_URL` | Redis connection string for session storage | redis://localhost:6379 |
| `AUTH_PROVIDER_URL` | External auth provider URL (if any) | - |
| `MFA_ENABLED` | Enable multi-factor authentication | true |
| `AUDIT_LOG_PATH` | Path for storing audit logs | ./logs/audit |
| `MAX_LOGIN_ATTEMPTS` | Maximum login attempts before lockout | 5 |
| `LOCKOUT_TIME_MINUTES` | Account lockout duration in minutes | 15 |

### Running the Service
Run the API service:
```bash
cd api
npm start
```

Start monitoring tools:
```bash
cd monitoring
python security_monitor.py
```

## Components

### Authentication
The authentication system provides:

- **User Authentication**: Username/password authentication
- **API Key Management**: API key generation, validation and rotation
- **JWT Tokens**: Token-based authentication for services
- **Multi-Factor Authentication**: Optional 2FA/MFA support
- **SSO Integration**: Support for external identity providers
- **Session Management**: Secure session handling

### Authorization
The authorization system includes:

- **Role-Based Access Control**: User role management
- **Permission System**: Fine-grained permission control
- **Resource Access Control**: Service-level access restrictions
- **IP Whitelisting**: Restricting access by IP address
- **Rate Limiting**: Preventing abuse through rate limits

### Monitoring
Security monitoring capabilities include:

- **Login Attempt Tracking**: Detection of brute force attempts
- **Anomaly Detection**: Identifying unusual access patterns
- **Real-time Alerts**: Notification of security events
- **Audit Logging**: Comprehensive logging of security events
- **Compliance Reporting**: Generating reports for compliance

### API Security
The service provides security middleware for APIs:

- **API Key Validation**: Validating API keys for service-to-service communication
- **Request Signing**: Cryptographic verification of API requests
- **Token Validation**: JWT token validation middleware
- **CORS Configuration**: Secure cross-origin resource sharing settings
- **Security Headers**: Implementation of secure HTTP headers

## API Documentation

### Endpoints

#### POST /auth/login
Authenticate a user and retrieve a JWT token.

**Request Body:**
```json
{
  "username": "trader_user",
  "password": "secure_password",
  "mfa_code": "123456"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_at": "2025-04-29T15:30:00Z",
  "user": {
    "id": "usr_123456",
    "username": "trader_user",
    "roles": ["trader", "analyst"],
    "last_login": "2025-04-28T10:15:22Z"
  }
}
```

#### POST /auth/api-keys
Generate a new API key.

**Request Body:**
```json
{
  "name": "Trading Bot Access",
  "expiration_days": 90,
  "permissions": ["read:market_data", "write:orders"]
}
```

**Response:**
```json
{
  "key_id": "key_789012",
  "api_key": "ft_sk_7a8b9c0d1e2f3g4h5i6j7k8l9m0n",
  "name": "Trading Bot Access",
  "created_at": "2025-04-29T14:30:00Z",
  "expires_at": "2025-07-28T14:30:00Z"
}
```

#### GET /auth/permissions
Get all available permissions.

**Response:**
```json
{
  "permissions": [
    {
      "id": "read:market_data",
      "description": "Read market data from all sources"
    },
    {
      "id": "write:orders",
      "description": "Create and modify orders"
    },
    {
      "id": "admin:users",
      "description": "Manage platform users"
    },
    {
      "id": "execute:strategies",
      "description": "Execute trading strategies"
    }
  ]
}
```

#### POST /auth/verify
Verify a JWT token.

**Request Body:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:**
```json
{
  "valid": true,
  "user_id": "usr_123456",
  "permissions": ["read:market_data", "write:orders"],
  "expires_at": "2025-04-29T15:30:00Z"
}
```

## Centralized Security Libraries

The Security Service provides the following shared libraries:

1. **common_lib.security**: Python library for security functions
   - Authentication helpers
   - Token management
   - Permission checking
   - Secure data handling

2. **common-js-lib/security.js**: JavaScript library for security functions
   - Authentication middleware
   - Token validation
   - Permission guards
   - Secure API request helpers

## Integration with Other Services
The Security Service integrates with all platform services by providing:

- Authentication middleware
- Authorization checks
- Audit logging
- Security monitoring

## Security Best Practices
The service implements the following security best practices:

- Password hashing with industry-standard algorithms
- API key rotation and expiration
- JWT token security with appropriate claims
- Protection against OWASP Top 10 vulnerabilities
- Regular security scanning and penetration testing
- Defense-in-depth strategy

## Audit Logging
The service maintains comprehensive audit logs for:

- Authentication events (login, logout, failed attempts)
- Authorization checks (access granted, access denied)
- API key usage and management
- Administrative actions
- Security configuration changes

## Compliance
The Security Service helps maintain compliance with:

- GDPR requirements
- Financial services regulations
- Information security standards (ISO 27001)
- Data protection requirements
