# Decision Log

## Potential Duplications Identified

### 1. Indicator Calculation Logic
- **Services Affected:**
  - `feature-store-service` (feature_store_service/indicators/)
  - `analysis-engine-service` (analysis_engine/analysis)
- **Description:** Both services contain implementations of technical indicators. The feature store service should be the primary source for indicator calculations, while the analysis engine should consume these indicators rather than implementing them separately.

### 2. Authentication/Security Mechanisms
- **Services Affected:**
  - `security` (authentication/, api_security_manager.py)
  - `trading-gateway-service` (trading_gateway_service/api)
  - `portfolio-management-service` (portfolio_management_service)
  - `ml-integration-service` (ml_integration_service)
- **Description:** Multiple services implement their own authentication mechanisms instead of leveraging a common implementation from the security service.

### 3. Database Connection Logic
- **Services Affected:**
  - `data-pipeline-service` (data_pipeline_service/database/engine.py)
  - `portfolio-management-service` (portfolio_management_service)
  - `feature-store-service` (feature_store_service)
  - `risk-management-service` (risk_management_service)
- **Description:** Multiple services implement their own database connection patterns instead of using a shared database utility module.

### 4. Error Handling & Custom Exceptions
- **Services Affected:**
  - `core-foundations` (core_foundations/exceptions)
  - `data-pipeline-service` (data_pipeline_service/exceptions)
  - `feature-store-service` (feature_store_service)
  - `risk-management-service` (risk_management_service)
- **Description:** Multiple services define similar exception classes instead of using the common exception classes from core-foundations.

### 5. Resilience Patterns
- **Services Affected:**
  - `core-foundations` (core_foundations/resilience)
  - `risk-management-service` (risk_management_service/circuit_breaker.py)
  - `trading-gateway-service`
- **Description:** Circuit breaker and retry logic are implemented in multiple services rather than consistently using the patterns from core-foundations.

### 6. Configuration Management
- **Services Affected:**
  - Most services have their own configuration handling
  - `monitoring-alerting-service` (config/)
  - `strategy-execution-engine` (config/)
  - `data-pipeline-service` (data_pipeline_service/settings.py)
  - `ml-integration-service` (ml_integration_service/settings.py)
- **Description:** Different services use inconsistent approaches to configuration management instead of a common pattern.

### 7. Caching Logic
- **Services Affected:**
  - `feature-store-service` (feature_store_service/caching)
  - `optimization` (caching/)
- **Description:** Multiple implementations of caching functionality instead of a shared caching service or library.

### 8. Monitoring & Metrics Code
- **Services Affected:**
  - `monitoring-alerting-service` (metrics_exporters/)
  - `core-foundations` (core_foundations/monitoring)
  - `risk-management-service` (risk_management_service)
- **Description:** Services implement custom monitoring logic instead of using the centralized monitoring functionality.

### 9. Test Utilities & Fixtures
- **Services Affected:**
  - `testing` (multiple test files)
  - `e2e` (fixtures/, utils/)
  - Individual service test folders contain duplicated test utilities
- **Description:** Test helpers, fixtures, and utilities are duplicated across services instead of being centralized in a shared test utilities package.

## Centralization Decisions

### Error Handling & Custom Exceptions
**Decision:** Move all custom exception classes to `common-lib`. The core-foundations module already has an exceptions package that should be the single source of truth for all custom exceptions used across the platform.
**Rationale:** This ensures consistent error handling patterns across all services and reduces duplication. Exception handling is a cross-cutting concern that should be standardized.
**Implementation Priority:** High
**Action:** Moved custom exception classes from `core-foundations/core_foundations/exceptions/base_exceptions.py` to `common-lib/common_lib/exceptions.py`. This includes the base `ForexTradingPlatformError` class and all specialized exceptions for configuration, data handling, services, authentication, trading, and model operations.

### Authentication/Security Mechanisms
**Decision:** Extract core authentication and security mechanisms from the security service to `common-lib`, including:
- API key validation
- JWT token generation and validation
- Role-based access control primitives
- Security headers utility functions
**Rationale:** Authentication is used by nearly all services and should follow consistent patterns.
**Implementation Priority:** High
**Action:** Created comprehensive security utilities in `common-lib/common_lib/security.py`, based on the implementation from the security service. The module includes:
- API key validation and authentication dependency
- JWT token creation, validation, and authentication dependency
- Security event logging
- Security headers for HTTP responses
- Support for role-based access through token scopes

### Database Connection Logic
**Decision:** Create a standardized database connection module in `common-lib` that handles:
- Connection pooling
- Retries
- Basic ORM integration
- Connection string management
**Rationale:** Each service should not need to implement its own database connection logic.
**Implementation Priority:** High
**Action:** Created comprehensive database connection utilities in `common-lib/common_lib/database.py` with the following features:
- Unified database configuration from environment variables
- Connection pooling for both synchronous and asynchronous database access
- Automatic retries for transient connection failures
- Context managers for easy session management
- Health check functionality
- Singleton pattern for efficient connection management
- Refactored `risk-management-service` database connection logic to use `common-lib/common_lib/database.py`.

### Configuration Management
**Decision:** Implement a unified configuration management system in `common-lib` that supports:
- Environment-specific configuration
- Hierarchical configuration (defaults + overrides)
- Secret management integration
- Runtime configuration updates
**Rationale:** Consistent configuration handling will simplify deployment and operations.
**Implementation Priority:** Medium

### Resilience Patterns
**Decision:** Consolidate all resilience patterns in core-foundations/resilience and ensure it's the only implementation used across services:
- Circuit breaker implementation
- Retry logic with exponential backoff
- Timeout handling
- Bulkhead patterns
**Rationale:** These are critical for system stability and should be consistently implemented.
**Implementation Priority:** Medium
**Action:** Created comprehensive resilience module in `common-lib/common_lib/resilience/` that re-exports and enhances the core resilience patterns from core-foundations. The module includes:
- Enhanced circuit breaker with monitoring integration
- Retry policy with specialized database exception handling
- Timeout handler for both sync and async functions
- Bulkhead pattern implementation for concurrency control
- Examples demonstrating usage of all patterns
- Documentation of integration patterns and best practices
- Refactored services to use centralized retry logic:
  - `analysis-engine-service` clients (ml_pipeline_client.py and execution_engine_client.py) - replaced old async_retry with retry_with_policy
  - `trading-gateway-service` broker adapters - updated imports to use common_lib resilience
  - `strategy-execution-engine` risk and trading clients - updated imports to use common_lib resilience

### Monitoring & Metrics
**Decision:** Extract core monitoring utilities from monitoring-alerting-service to `common-lib`:
- Metrics collection utilities
- Standard instrumentation helpers
- Common metric naming conventions
**Rationale:** Standardized monitoring approach enables consistent observability.
**Implementation Priority:** Medium

### Caching Logic
**Decision:** Create a unified caching module in `common-lib` with:
- In-memory cache implementation
- Redis cache adapter
- Cache invalidation patterns
**Rationale:** Caching strategies should be consistent across services.
**Implementation Priority:** Low

### Indicator Calculation Logic
**Decision:** Feature-store-service will be the canonical source for all indicator implementations. Analysis-engine will not duplicate these, instead consuming from feature-store.
**Rationale:** This is domain-specific logic that should have a single owner.
**Implementation Priority:** Medium

### Test Utilities
**Decision:** Create a shared test utilities package in the testing directory with:
- Common test fixtures
- Mock data generators
- Assertion helpers
- Performance testing utilities
**Rationale:** Test code quality is improved with shared utilities, and test maintenance is reduced.
**Implementation Priority:** Low

### JavaScript/Node.js Centralization
**Decision:** Create a `common-js-lib` JavaScript library to provide standardized functionality for Node.js services, similar to how `common-lib` works for Python services. Initial focus on centralizing:
- Security utilities (API key validation, JWT handling)
- Common middleware for Express applications
**Rationale:** Ensures consistent security patterns across language boundaries and prevents duplication of critical security code.
**Implementation Priority:** High
**Action:** Created `common-js-lib` with security utilities and middleware functions for Express applications. The `trading-gateway-service` was refactored to use these centralized components. Dependencies have been successfully installed.
