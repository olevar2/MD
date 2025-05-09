# Forex Trading Platform File and Directory Structure Standards

This document defines standard file and directory structures for services in the Forex Trading Platform. These standards ensure consistency, maintainability, and alignment with domain concepts across all services.

## Table of Contents

1. [General Principles](#general-principles)
2. [Python Service Structure](#python-service-structure)
3. [JavaScript/TypeScript Frontend Structure](#javascripttypescript-frontend-structure)
4. [Common Library Structure](#common-library-structure)
5. [Test Directory Structure](#test-directory-structure)
6. [Documentation Structure](#documentation-structure)
7. [Configuration Files](#configuration-files)
8. [Migration Guide](#migration-guide)

## General Principles

1. **Domain-Driven Structure**: Organize code around domain concepts
2. **Consistency**: Follow consistent patterns across all services
3. **Separation of Concerns**: Separate different responsibilities into different directories
4. **Discoverability**: Make it easy to find code by following predictable patterns
5. **Modularity**: Design for modularity and reusability

## Python Service Structure

### Basic Service Structure

```
service-name/
├── service_name/                # Main package
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Application entry point
│   ├── config.py                # Configuration
│   ├── constants.py             # Constants and enums
│   ├── api/                     # API endpoints
│   │   ├── __init__.py
│   │   ├── routes.py            # Route definitions
│   │   ├── models.py            # API models (request/response)
│   │   └── dependencies.py      # API dependencies
│   ├── domain/                  # Domain models and logic
│   │   ├── __init__.py
│   │   ├── models.py            # Domain entities
│   │   └── services.py          # Domain services
│   ├── adapters/                # External service adapters
│   │   ├── __init__.py
│   │   └── service_adapter.py   # Adapter implementation
│   ├── infrastructure/          # Infrastructure concerns
│   │   ├── __init__.py
│   │   ├── database/            # Database access
│   │   ├── messaging/           # Message bus integration
│   │   └── logging/             # Logging configuration
│   └── utils/                   # Utility functions
├── tests/                       # Test directory
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── conftest.py              # Test fixtures
├── pyproject.toml               # Project configuration
├── README.md                    # Project documentation
└── Makefile                     # Build and development tasks
```

### Domain-Specific Service Structure

#### Market Data Service

```
market-data-service/
├── market_data_service/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── constants.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── models.py
│   │   └── dependencies.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── services.py
│   │   └── validators.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── data_provider_adapter.py
│   │   └── cache_adapter.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── database/
│   │   ├── messaging/
│   │   └── logging/
│   └── utils/
│       ├── __init__.py
│       ├── formatting.py
│       └── validation.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml
├── README.md
└── Makefile
```

#### Trading Service

```
trading-service/
├── trading_service/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── constants.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── models.py
│   │   └── dependencies.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── order.py
│   │   │   ├── position.py
│   │   │   └── account.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── order_service.py
│   │   │   ├── position_service.py
│   │   │   └── account_service.py
│   │   └── validators/
│   │       ├── __init__.py
│   │       ├── order_validator.py
│   │       └── risk_validator.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── broker_adapter.py
│   │   └── market_data_adapter.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── database/
│   │   ├── messaging/
│   │   └── logging/
│   └── utils/
│       ├── __init__.py
│       ├── order_utils.py
│       └── risk_utils.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml
├── README.md
└── Makefile
```

#### Analysis Service

```
analysis-service/
├── analysis_service/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── constants.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── models.py
│   │   └── dependencies.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── indicator.py
│   │   │   ├── pattern.py
│   │   │   └── signal.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── indicator_service.py
│   │   │   ├── pattern_service.py
│   │   │   └── signal_service.py
│   │   └── analyzers/
│   │       ├── __init__.py
│   │       ├── technical_analyzer.py
│   │       ├── pattern_analyzer.py
│   │       └── signal_generator.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── market_data_adapter.py
│   │   └── model_registry_adapter.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── database/
│   │   ├── messaging/
│   │   └── logging/
│   └── utils/
│       ├── __init__.py
│       ├── math_utils.py
│       └── time_utils.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml
├── README.md
└── Makefile
```

### Module Naming Conventions

- **Package Names**: Lowercase with underscores (e.g., `market_data_service`)
- **Module Names**: Lowercase with underscores (e.g., `order_service.py`)
- **Class Names**: CapWords/PascalCase (e.g., `OrderService`)
- **Function Names**: Lowercase with underscores (e.g., `place_order`)
- **Variable Names**: Lowercase with underscores (e.g., `order_id`)
- **Constant Names**: All uppercase with underscores (e.g., `MAX_ORDER_SIZE`)

## JavaScript/TypeScript Frontend Structure

### Basic Frontend Structure

```
ui-service/
├── src/                         # Source code
│   ├── components/              # React components
│   │   ├── common/              # Shared components
│   │   ├── charts/              # Chart components
│   │   └── trading/             # Trading components
│   ├── hooks/                   # Custom React hooks
│   │   ├── useMarketData.ts
│   │   └── useOrderManagement.ts
│   ├── services/                # API services
│   │   ├── marketDataService.ts
│   │   └── orderService.ts
│   ├── store/                   # State management
│   │   ├── slices/              # Redux slices
│   │   └── store.ts             # Store configuration
│   ├── utils/                   # Utility functions
│   │   ├── formatting.ts
│   │   └── calculations.ts
│   ├── types/                   # TypeScript type definitions
│   │   ├── marketData.ts
│   │   └── orders.ts
│   ├── constants/               # Application constants
│   │   ├── timeframes.ts
│   │   └── orderTypes.ts
│   └── App.tsx                  # Application entry point
├── public/                      # Static assets
├── tests/                       # Test directory
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── package.json                 # Project configuration
├── tsconfig.json                # TypeScript configuration
├── .eslintrc.js                 # ESLint configuration
└── README.md                    # Project documentation
```

### Domain-Specific Frontend Structure

#### Trading UI

```
trading-ui/
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   └── Modal.tsx
│   │   ├── charts/
│   │   │   ├── PriceChart.tsx
│   │   │   ├── IndicatorChart.tsx
│   │   │   └── VolumeChart.tsx
│   │   ├── trading/
│   │   │   ├── OrderForm.tsx
│   │   │   ├── OrderList.tsx
│   │   │   ├── PositionList.tsx
│   │   │   └── AccountSummary.tsx
│   │   └── analysis/
│   │       ├── IndicatorPanel.tsx
│   │       ├── PatternDisplay.tsx
│   │       └── SignalAlert.tsx
│   ├── hooks/
│   │   ├── useMarketData.ts
│   │   ├── useOrderManagement.ts
│   │   ├── usePositions.ts
│   │   └── useIndicators.ts
│   ├── services/
│   │   ├── marketDataService.ts
│   │   ├── orderService.ts
│   │   ├── accountService.ts
│   │   └── analysisService.ts
│   ├── store/
│   │   ├── slices/
│   │   │   ├── marketDataSlice.ts
│   │   │   ├── orderSlice.ts
│   │   │   ├── positionSlice.ts
│   │   │   └── accountSlice.ts
│   │   └── store.ts
│   ├── utils/
│   │   ├── formatting.ts
│   │   ├── calculations.ts
│   │   ├── validation.ts
│   │   └── time.ts
│   ├── types/
│   │   ├── marketData.ts
│   │   ├── orders.ts
│   │   ├── positions.ts
│   │   └── account.ts
│   ├── constants/
│   │   ├── timeframes.ts
│   │   ├── orderTypes.ts
│   │   └── indicators.ts
│   └── App.tsx
├── public/
│   ├── index.html
│   ├── favicon.ico
│   └── assets/
├── tests/
│   ├── unit/
│   └── integration/
├── package.json
├── tsconfig.json
├── .eslintrc.js
└── README.md
```

### File Naming Conventions

- **Component Files**: PascalCase with `.tsx` extension (e.g., `OrderForm.tsx`)
- **Hook Files**: camelCase with `use` prefix and `.ts` extension (e.g., `useMarketData.ts`)
- **Service Files**: camelCase with `Service` suffix and `.ts` extension (e.g., `marketDataService.ts`)
- **Utility Files**: camelCase with `.ts` extension (e.g., `formatting.ts`)
- **Type Files**: camelCase with `.ts` extension (e.g., `marketData.ts`)
- **Constant Files**: camelCase with `.ts` extension (e.g., `timeframes.ts`)
- **Store Files**: camelCase with `Slice` suffix and `.ts` extension (e.g., `orderSlice.ts`)

## Common Library Structure

### Basic Common Library Structure

```
common-lib/
├── common_lib/                  # Main package
│   ├── __init__.py              # Package initialization
│   ├── models/                  # Domain models
│   │   ├── __init__.py
│   │   ├── market_data.py
│   │   ├── orders.py
│   │   └── signals.py
│   ├── interfaces/              # Service interfaces
│   │   ├── __init__.py
│   │   ├── market_data.py
│   │   ├── trading.py
│   │   └── analysis.py
│   ├── errors/                  # Error definitions
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── market_data.py
│   │   ├── trading.py
│   │   └── analysis.py
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── formatting.py
│   │   └── time.py
│   └── constants/               # Constants and enums
│       ├── __init__.py
│       ├── market_data.py
│       ├── orders.py
│       └── timeframes.py
├── tests/                       # Test directory
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── pyproject.toml               # Project configuration
├── README.md                    # Project documentation
└── Makefile                     # Build and development tasks
```

### Domain-Specific Common Library Structure

```
common-lib/
├── common_lib/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── market_data/
│   │   │   ├── __init__.py
│   │   │   ├── instrument.py
│   │   │   ├── ohlcv.py
│   │   │   └── tick.py
│   │   ├── trading/
│   │   │   ├── __init__.py
│   │   │   ├── order.py
│   │   │   ├── position.py
│   │   │   └── account.py
│   │   └── analysis/
│   │       ├── __init__.py
│   │       ├── indicator.py
│   │       ├── pattern.py
│   │       └── signal.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── market_data/
│   │   │   ├── __init__.py
│   │   │   ├── data_provider.py
│   │   │   └── instrument_provider.py
│   │   ├── trading/
│   │   │   ├── __init__.py
│   │   │   ├── order_service.py
│   │   │   └── position_service.py
│   │   └── analysis/
│   │       ├── __init__.py
│   │       ├── indicator_service.py
│   │       └── signal_service.py
│   ├── errors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── market_data.py
│   │   ├── trading.py
│   │   └── analysis.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── formatting.py
│   │   ├── time.py
│   │   ├── math.py
│   │   └── logging.py
│   └── constants/
│       ├── __init__.py
│       ├── market_data.py
│       ├── orders.py
│       ├── timeframes.py
│       └── indicators.py
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml
├── README.md
└── Makefile
```

## Test Directory Structure

### Basic Test Structure

```
tests/
├── unit/                        # Unit tests
│   ├── test_models.py
│   ├── test_services.py
│   └── test_utils.py
├── integration/                 # Integration tests
│   ├── test_api.py
│   ├── test_database.py
│   └── test_external_services.py
├── e2e/                         # End-to-end tests
│   ├── test_workflows.py
│   └── test_scenarios.py
├── performance/                 # Performance tests
│   ├── test_throughput.py
│   └── test_latency.py
├── fixtures/                    # Test fixtures
│   ├── market_data_fixtures.py
│   ├── order_fixtures.py
│   └── user_fixtures.py
└── conftest.py                  # Pytest configuration
```

### Domain-Specific Test Structure

```
tests/
├── unit/
│   ├── domain/
│   │   ├── test_order_model.py
│   │   ├── test_position_model.py
│   │   └── test_account_model.py
│   ├── services/
│   │   ├── test_order_service.py
│   │   ├── test_position_service.py
│   │   └── test_account_service.py
│   ├── adapters/
│   │   ├── test_broker_adapter.py
│   │   └── test_market_data_adapter.py
│   └── utils/
│       ├── test_order_utils.py
│       └── test_risk_utils.py
├── integration/
│   ├── api/
│   │   ├── test_order_api.py
│   │   ├── test_position_api.py
│   │   └── test_account_api.py
│   ├── database/
│   │   ├── test_order_repository.py
│   │   └── test_position_repository.py
│   └── external/
│       ├── test_broker_integration.py
│       └── test_market_data_integration.py
├── e2e/
│   ├── workflows/
│   │   ├── test_order_placement.py
│   │   ├── test_position_management.py
│   │   └── test_account_operations.py
│   └── scenarios/
│       ├── test_trading_scenario.py
│       └── test_risk_management_scenario.py
├── performance/
│   ├── test_order_throughput.py
│   ├── test_market_data_latency.py
│   └── test_signal_generation_performance.py
├── fixtures/
│   ├── market_data_fixtures.py
│   ├── order_fixtures.py
│   ├── position_fixtures.py
│   └── account_fixtures.py
└── conftest.py
```

### Test File Naming Conventions

- **Test Files**: Prefix with `test_` and use lowercase with underscores (e.g., `test_order_service.py`)
- **Test Functions**: Prefix with `test_` and use lowercase with underscores (e.g., `test_place_order_success`)
- **Fixture Files**: Suffix with `_fixtures` and use lowercase with underscores (e.g., `order_fixtures.py`)
- **Fixture Functions**: Use descriptive names in lowercase with underscores (e.g., `valid_order`)

## Documentation Structure

### Basic Documentation Structure

```
docs/
├── api/                         # API documentation
│   ├── market_data_api.md
│   ├── trading_api.md
│   └── analysis_api.md
├── architecture/                # Architecture documentation
│   ├── overview.md
│   ├── services.md
│   └── data_flow.md
├── development/                 # Development guides
│   ├── setup.md
│   ├── testing.md
│   └── deployment.md
├── standards/                   # Coding standards
│   ├── python_standards.md
│   ├── typescript_standards.md
│   └── api_standards.md
└── domain/                      # Domain documentation
    ├── market_data.md
    ├── trading.md
    └── analysis.md
```

### Domain-Specific Documentation Structure

```
docs/
├── api/
│   ├── market_data/
│   │   ├── instruments.md
│   │   ├── ohlcv.md
│   │   └── ticks.md
│   ├── trading/
│   │   ├── orders.md
│   │   ├── positions.md
│   │   └── accounts.md
│   └── analysis/
│       ├── indicators.md
│       ├── patterns.md
│       └── signals.md
├── architecture/
│   ├── overview.md
│   ├── services/
│   │   ├── market_data_service.md
│   │   ├── trading_service.md
│   │   └── analysis_service.md
│   ├── data_flow/
│   │   ├── order_flow.md
│   │   ├── market_data_flow.md
│   │   └── signal_flow.md
│   └── infrastructure/
│       ├── database.md
│       ├── messaging.md
│       └── deployment.md
├── development/
│   ├── setup/
│   │   ├── local_environment.md
│   │   ├── docker_environment.md
│   │   └── cloud_environment.md
│   ├── testing/
│   │   ├── unit_testing.md
│   │   ├── integration_testing.md
│   │   └── e2e_testing.md
│   └── deployment/
│       ├── staging.md
│       ├── production.md
│       └── monitoring.md
├── standards/
│   ├── coding/
│   │   ├── python_standards.md
│   │   ├── typescript_standards.md
│   │   └── api_standards.md
│   ├── architecture/
│   │   ├── service_boundaries.md
│   │   ├── error_handling.md
│   │   └── resilience.md
│   └── process/
│       ├── code_review.md
│       ├── release_process.md
│       └── incident_management.md
└── domain/
    ├── market_data/
    │   ├── instruments.md
    │   ├── price_data.md
    │   └── market_events.md
    ├── trading/
    │   ├── orders.md
    │   ├── positions.md
    │   └── risk_management.md
    └── analysis/
        ├── technical_analysis.md
        ├── pattern_recognition.md
        └── signal_generation.md
```

## Configuration Files

### Project Configuration Files

- **pyproject.toml**: Python project configuration
- **package.json**: JavaScript/TypeScript project configuration
- **tsconfig.json**: TypeScript configuration
- **.eslintrc.js**: ESLint configuration
- **.prettierrc.json**: Prettier configuration
- **.pre-commit-config.yaml**: Pre-commit hooks configuration
- **Makefile**: Build and development tasks
- **docker-compose.yml**: Docker Compose configuration
- **Dockerfile**: Docker configuration

### Service Configuration Files

- **config.py**: Python service configuration
- **constants.py**: Constants and enums
- **.env**: Environment variables (not committed to version control)
- **.env.example**: Example environment variables (committed to version control)
- **alembic.ini**: Database migration configuration
- **logging.conf**: Logging configuration

## Migration Guide

### Migrating Existing Services

1. **Assessment Phase**
   - Analyze current structure
   - Identify gaps and inconsistencies
   - Create migration plan

2. **Incremental Migration**
   - Start with one service at a time
   - Begin with new files following the new structure
   - Gradually refactor existing files
   - Update imports and references

3. **Testing and Validation**
   - Ensure tests pass after each change
   - Verify functionality in development environment
   - Get peer review for structural changes

4. **Documentation Update**
   - Update documentation to reflect new structure
   - Create guides for developers

### Migration Checklist

- [ ] Create new directory structure
- [ ] Move files to appropriate locations
- [ ] Update imports and references
- [ ] Update configuration files
- [ ] Run tests to verify functionality
- [ ] Update documentation
- [ ] Get peer review
- [ ] Deploy to development environment
- [ ] Verify functionality in development environment
- [ ] Update CI/CD pipeline if necessary

### Migration Timeline

- **Phase 1**: Create new structure for new services
- **Phase 2**: Migrate common libraries
- **Phase 3**: Migrate core services
- **Phase 4**: Migrate supporting services
- **Phase 5**: Migrate frontend applications