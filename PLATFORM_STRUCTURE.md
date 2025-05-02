# Forex Trading Platform Structure Documentation

## Overview

This document provides a comprehensive overview of the forex trading platform's structure, including all services, modules, and key files. This can serve as a reference for developers working on the platform.

## Project Root Files

- **DECISION_LOG.md** - Record of architectural and technical decisions
- **MASTER_CHECKLIST.md** - Master checklist for project management
- **PROJECT_STATUS.md** - Current status of the entire project
- **README.md** - Main project documentation
- **run_test.bat** - Batch script to run tests
- **run_tests.py** - Python script to run tests

## Service Architecture

The platform consists of multiple services, each responsible for specific functionality:

### 1. Analysis Engine Service

This service handles market analysis, indicators, and trading signals.

**Key Components:**
- `analysis_engine/` - Main service code
  - `adaptive_layer/` - Adaptive analysis components
  - `analysis/` - Core analysis functionality
  - `advanced_ta/` - Advanced technical analysis
  - `gann/` - Gann analysis methods
  - `backtesting/` - Backtesting facilities
  - `basic_ta/` - Basic technical analysis
  - `correlation/` - Correlation analysis
  - `manipulation/` - Data manipulation tools
  - `nlp/` - Natural language processing for news analysis
  - `pattern_recognition/` - Chart pattern recognition
  - `api/` - API endpoints
  - `auth/` - Authentication mechanisms
  - `backtesting/` - Additional backtesting capabilities
  - `batch/` - Batch processing tools
  - `causal/` - Causal analysis tools
  - `clients/` - Client interfaces
  - `core/` - Core service functionality
  - `db/` - Database interactions
  - `integration/` - Integration with other services
  - `interfaces/` - Interface definitions
  - `learning_from_mistakes/` - Performance improvement framework
  - `monitoring/` - Monitoring tools
  - `repositories/` - Data repositories
  - `resilience/` - Resilience patterns
  - `scheduling/` - Task scheduling
  - `services/` - Service implementations
  - `tools/` - Utility tools
  - `utils/` - Utility functions
  - `visualization/` - Data visualization tools
- `config/` - Configuration files
- `examples/` - Example implementations
- `tests/` - Test suite
  - `analysis/` - Analysis tests
  - `clients/` - Client tests
  - `integration/` - Integration tests
  - `multi_asset/` - Multi-asset tests
  - `services/` - Service tests

### 2. Common JS Library

Shared JavaScript utilities used across services.

**Key Components:**
- `index.js` - Main entry point
- `package.json` - Package configuration
- `security.js` - Security utilities
- `test/` - Test suite

### 3. Common Library

Shared Python utilities used across services.

**Key Components:**
- `common_lib/` - Library source code
  - `config/` - Shared configurations
  - `resilience/` - Resilience patterns implementation
- `docs/` - Documentation
- `tests/` - Test suite
- `usage_demos/` - Usage examples

### 4. Core Foundations

Core components and frameworks used by all services.

**Key Components:**
- `core_foundations/` - Main library code
  - `api/` - API utilities
  - `config/` - Configuration utilities
  - `events/` - Event handling
  - `exceptions/` - Exception definitions
  - `feedback/` - Feedback mechanisms
  - `interfaces/` - Interface definitions
  - `models/` - Data models
  - `monitoring/` - Monitoring tools
  - `performance/` - Performance tools
  - `resilience/` - Resilience patterns
  - `utils/` - Utility functions
- `docs/` - Documentation
- `tests/` - Test suite

### 5. Data Pipeline Service

Handles data ingestion, processing, and storage.

**Key Components:**
- `data_pipeline_service/` - Main service code
  - `api/` - API endpoints
  - `backup/` - Data backup functionality
  - `cleaning/` - Data cleaning tools
  - `compliance/` - Compliance handling
  - `config/` - Configuration
  - `db/` - Database interactions
  - `exceptions/` - Exception handling
  - `models/` - Data models
  - `repositories/` - Data repositories
  - `services/` - Service implementations
  - `source_adapters/` - Data source adapters
  - `validation/` - Data validation
- `tests/` - Test suite

### 6. End-to-End Tests

End-to-end testing framework for the entire platform.

**Key Components:**
- `fixtures/` - Test fixtures
- `framework/` - Testing framework
- `reporting/` - Test reporting
- `tests/` - Test definitions
- `utils/` - Testing utilities
- `validation/` - Validation tools

### 7. Feature Store Service

Manages and provides access to trading features.

**Key Components:**
- `feature_store_service/` - Main service code
  - `api/` - API endpoints
  - `caching/` - Feature caching
  - `computation/` - Feature computation
  - `config/` - Configuration
  - `core/` - Core functionality
  - `db/` - Database interactions
  - `error/` - Error handling
  - `indicators/` - Trading indicators
  - `interfaces/` - Interface definitions
  - `logging/` - Logging system
  - `models/` - Data models
  - `monitoring/` - Monitoring capabilities
  - `optimization/` - Performance optimization
  - `recovery/` - Error recovery
  - `reliability/` - Reliability features
  - `repositories/` - Data repositories
  - `scheduling/` - Task scheduling
  - `services/` - Service implementations
  - `storage/` - Storage management
  - `utils/` - Utility functions
  - `validation/` - Validation tools
  - `verification/` - Verification tools
- `indicators/` - Additional indicators
- `tests/` - Test suite

### 8. Infrastructure

Infrastructure configuration and management.

**Key Components:**
- `backup/` - Backup configurations
- `config/` - Infrastructure configurations
- `database/` - Database scripts
  - `migrations/` - Database migrations
- `docker/` - Docker configurations
  - `grafana/` - Grafana setup
  - `prometheus/` - Prometheus setup
- `incidents/` - Incident handling
- `scaling/` - Scaling configurations
- `scripts/` - Infrastructure scripts
- `terraform/` - Terraform configurations

### 9. ML Integration Service

Integrates machine learning capabilities with trading strategies.

**Key Components:**
- `ml_integration_service/` - Main service code
  - `api/` - API endpoints
  - `config/` - Configuration
  - `examples/` - Examples of ML integration
  - `feedback/` - Feedback loop for model improvement
  - `monitoring/` - Model monitoring
  - `optimization/` - Model optimization
  - `services/` - Service implementations
  - `strategy_filters/` - ML-based strategy filters
  - `strategy_optimizers/` - ML-based strategy optimization
  - `stress_testing/` - Stress testing for models
  - `visualization/` - Model visualization
- `docs/` - Documentation
- `tests/` - Test suite

### 10. ML Workbench Service

Development environment for machine learning models.

**Key Components:**
- `ml_workbench_service/` - Main service code
  - `api/` - API endpoints
  - `backtesting/` - Model backtesting
  - `clients/` - Client implementations
  - `config/` - Configuration
  - `effectiveness/` - Model effectiveness analysis
  - `explainability/` - Model explainability tools
  - `feedback/` - Feedback mechanisms
  - `model_registry/` - Model versioning and registry
  - `models/` - Model implementations
  - `multitask/` - Multi-task learning
  - `reinforcement/` - Reinforcement learning
  - `optimization/` - Model optimization
  - `performance/` - Performance analysis
  - `repositories/` - Data repositories
  - `scripts/` - Utility scripts
  - `services/` - Service implementations
  - `transfer_learning/` - Transfer learning tools
  - `visualization/` - Model visualization
- `docs/` - Documentation
- `examples/` - Example implementations
- `tests/` - Test suite

### 11. Monitoring & Alerting Service

Handles monitoring and alerting across the platform.

**Key Components:**
- `alerts/` - Alert definitions
- `config/` - Configuration
- `dashboards/` - Monitoring dashboards
- `infrastructure/` - Monitoring infrastructure
  - `docker/` - Docker configurations
  - `grafana/` - Grafana setup
  - `loki/` - Loki setup
  - `prometheus/` - Prometheus setup
  - `tempo/` - Tempo setup
- `metrics_exporters/` - Custom metrics exporters
- `tests/` - Test suite

### 12. Optimization

Performance and resource optimization tools.

**Key Components:**
- `caching/` - Caching strategies
- `ml/` - ML-based optimization
- `resources/` - Resource allocation
- `tests/` - Test suite
- `timeseries/` - Time series optimization

### 13. Portfolio Management Service

Manages trading portfolios.

**Key Components:**
- `portfolio_management_service/` - Main service code
  - `api/` - API endpoints
  - `clients/` - Client implementations
  - `db/` - Database interactions
  - `models/` - Data models
  - `multi_asset/` - Multi-asset portfolio management
  - `repositories/` - Data repositories
  - `services/` - Service implementations
  - `tax_reporting/` - Tax reporting tools
- `tests/` - Test suite

### 14. Risk Management Service

Manages trading risk.

**Key Components:**
- `risk_management_service/` - Main service code
  - `adaptive_risk/` - Adaptive risk management
  - `api/` - API endpoints
  - `calculators/` - Risk calculators
  - `db/` - Database interactions
  - `managers/` - Risk managers
  - `models/` - Risk models
  - `optimization/` - Risk optimization
  - `pipelines/` - Risk pipelines
  - `repositories/` - Data repositories
  - `services/` - Service implementations
- `tests/` - Test suite

### 15. Security

Platform security components.

**Key Components:**
- `api/` - Security APIs
- `authentication/` - Authentication mechanisms
- `authorization/` - Authorization rules
- `monitoring/` - Security monitoring
- `tests/` - Security tests

### 16. Strategy Execution Engine

Executes trading strategies.

**Key Components:**
- `strategy_execution_engine/` - Main service code
  - `adaptive_layer/` - Adaptive execution components
  - `backtesting/` - Strategy backtesting
  - `execution/` - Execution management
  - `factory/` - Strategy factory patterns
  - `integration/` - Integration with other services
  - `models/` - Strategy models
  - `multi_asset/` - Multi-asset strategy support
  - `performance/` - Performance tracking
  - `risk/` - Risk management integration
  - `signal/` - Signal processing
  - `signal_aggregation/` - Signal aggregation
  - `strategies/` - Strategy implementations
  - `trading/` - Trading execution
- `config/` - Configuration
- `strategy_mutation/` - Strategy mutation tools
- `tests/` - Test suite

### 17. Testing

Testing tools and frameworks.

**Key Components:**
- `feedback_system/` - Test feedback system
- `feedback_tests/` - Feedback-driven tests
- `integration_testing/` - Integration testing framework
- `stress_testing/` - Stress testing tools
- `analysis/` - Analysis testing
- `config/` - Testing configuration
- `system_validation/` - System validation tools

### 18. Trading Gateway Service

Interface to trading platforms and brokers.

**Key Components:**
- `trading_gateway_service/` - Main service code
  - `broker_adapters/` - Broker integration adapters
  - `execution_algorithms/` - Execution algorithms
  - `incidents/` - Incident handling
  - `interfaces/` - Interface definitions
  - `monitoring/` - Gateway monitoring
  - `resilience/` - Resilience patterns
  - `services/` - Service implementations
  - `simulation/` - Trading simulation
    - `reinforcement_learning/` - RL-based simulation
- `tests/` - Test suite
  - `broker_adapters/` - Adapter tests
  - `execution_algorithms/` - Algorithm tests
  - `services/` - Service tests

### 19. UI Service

User interface components.

**Key Components:**
- `components/` - UI components
- `public/` - Public assets
- `src/` - Source code
  - `api/` - API clients
  - `components/` - React components
  - `contexts/` - React contexts
  - `hooks/` - React hooks
  - `pages/` - App pages
  - `prototypes/` - UI prototypes
  - `routes/` - Routing configuration
  - `services/` - Frontend services
  - `styles/` - CSS styles
  - `types/` - TypeScript types
- `tests/` - Test suite

## Documentation

Comprehensive documentation for the platform.

**Key Components:**
- `api/` - API documentation
- `architecture/` - Architecture documentation
- `compliance/` - Compliance documentation
- `developer/` - Developer guides
- `knowledge_base/` - Knowledge base
- `operations/` - Operations documentation
- `phase8/` - Phase 8 implementation details
- `refactoring/` - Refactoring guidelines
- `risk_register/` - Risk register
- `user_guides/` - User guides

## Examples

Example implementations and code.

**Key Components:**
- `enhanced_strategy_example.py` - Enhanced strategy example
- `retry_examples.py` - Retry pattern examples

## Summary

The forex trading platform is organized as a microservice architecture with each service responsible for specific functionality. The services communicate with each other through well-defined APIs and share common code through libraries. The platform is designed to be resilient, scalable, and maintainable.

The overall system architecture follows modern software engineering practices with clear separation of concerns, comprehensive testing, proper documentation, and infrastructure as code.
