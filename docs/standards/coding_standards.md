# Forex Trading Platform Coding Standards

## Overview

This document defines the comprehensive coding standards for the Forex Trading Platform. These standards are designed to ensure consistency, maintainability, and alignment with domain concepts across all services.

## Core Principles

1. **Domain-Driven Design**: Code should reflect the ubiquitous language of forex trading
2. **Consistency**: Follow consistent patterns across all services
3. **Readability**: Code should be self-documenting and easy to understand
4. **Maintainability**: Design for future changes and extensions
5. **Testability**: Code should be designed for comprehensive testing

## Language-Specific Standards

### Python Standards

#### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `market_data.py` |
| Packages | snake_case | `technical_analysis` |
| Classes | PascalCase | `MarketDataService` |
| Functions/Methods | snake_case | `calculate_moving_average()` |
| Variables | snake_case | `price_data` |
| Constants | UPPER_SNAKE_CASE | `MAX_LEVERAGE` |
| Type Variables | PascalCase | `PriceData` |
| Private Attributes | _leading_underscore | `_internal_state` |

#### Domain-Specific Naming

| Domain Concept | Naming Pattern | Examples |
|----------------|----------------|----------|
| Indicators | `{name}_indicator` | `rsi_indicator`, `macd_indicator` |
| Patterns | `{name}_pattern` | `head_and_shoulders_pattern`, `double_top_pattern` |
| Strategies | `{name}_strategy` | `trend_following_strategy`, `mean_reversion_strategy` |
| Signals | `{name}_signal` | `buy_signal`, `sell_signal` |
| Market Data | `{instrument}_{timeframe}_data` | `eurusd_h1_data`, `btcusd_m5_data` |
| Services | `{domain}Service` | `OrderExecutionService`, `RiskManagementService` |
| Repositories | `{entity}Repository` | `TradeRepository`, `OrderRepository` |

#### Code Structure

1. **Imports**
   - Group imports in the following order:
     1. Standard library imports
     2. Third-party library imports
     3. Local application imports
   - Sort imports alphabetically within each group
   - Use absolute imports for application modules

2. **Class Structure**
   - Order class methods as follows:
     1. `__init__` and other dunder methods
     2. Class methods
     3. Static methods
     4. Public methods
     5. Protected methods (prefixed with `_`)
     6. Private methods (prefixed with `__`)

3. **Function/Method Structure**
   - Include type hints for parameters and return values
   - Document parameters, return values, and exceptions in docstrings
   - Keep functions focused on a single responsibility
   - Limit function length to 50 lines (prefer shorter)

#### Documentation

1. **Docstrings**
   - Use Google-style docstrings
   - Include descriptions for all public classes and methods
   - Document parameters, return values, and exceptions
   - Include examples for complex functions

2. **Comments**
   - Use comments to explain "why", not "what"
   - Keep comments up-to-date with code changes
   - Use TODO comments with issue references for future work

#### Error Handling

1. **Exceptions**
   - Use custom exceptions from `common_lib.exceptions`
   - Include contextual information in exception messages
   - Document exceptions in function docstrings
   - Handle exceptions at appropriate levels of abstraction

2. **Validation**
   - Validate inputs at service boundaries
   - Use Pydantic models for data validation
   - Include clear error messages for validation failures

### JavaScript/TypeScript Standards

#### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Files | kebab-case | `market-data.ts` |
| Interfaces | PascalCase with 'I' prefix | `IMarketData` |
| Classes | PascalCase | `MarketDataService` |
| Functions | camelCase | `calculateMovingAverage()` |
| Variables | camelCase | `priceData` |
| Constants | UPPER_SNAKE_CASE | `MAX_LEVERAGE` |
| Private Properties | _leadingUnderscore | `_internalState` |

#### Domain-Specific Naming

| Domain Concept | Naming Pattern | Examples |
|----------------|----------------|----------|
| Components | `{Name}Component` | `ChartComponent`, `OrderFormComponent` |
| Services | `{domain}Service` | `MarketDataService`, `AuthService` |
| Hooks | `use{Feature}` | `useMarketData`, `useOrderBook` |
| State | `{feature}State` | `orderState`, `chartState` |
| Actions | `{verb}{Noun}` | `fetchMarketData`, `placeOrder` |
| Reducers | `{feature}Reducer` | `orderReducer`, `authReducer` |

#### Code Structure

1. **Imports**
   - Group imports in the following order:
     1. External libraries
     2. Internal modules
     3. Component imports
     4. Style imports
   - Sort imports alphabetically within each group

2. **Component Structure**
   - Keep components focused on a single responsibility
   - Extract complex logic into custom hooks
   - Use functional components with hooks
   - Implement proper prop validation

3. **Function Structure**
   - Include TypeScript type annotations
   - Keep functions focused on a single responsibility
   - Limit function length to 50 lines (prefer shorter)

#### Documentation

1. **JSDoc Comments**
   - Use JSDoc for all public functions and components
   - Document parameters, return values, and exceptions
   - Include examples for complex functions

2. **Comments**
   - Use comments to explain "why", not "what"
   - Keep comments up-to-date with code changes
   - Use TODO comments with issue references for future work

#### Error Handling

1. **Error Boundaries**
   - Implement error boundaries for component trees
   - Include fallback UI for error states
   - Log errors to monitoring service

2. **API Error Handling**
   - Handle API errors consistently
   - Display user-friendly error messages
   - Include retry mechanisms for transient failures

## Domain-Specific Standards

### Market Data Components

1. **Data Validation**
   - Validate all incoming market data for completeness and correctness
   - Handle missing data points gracefully
   - Document data quality assumptions

2. **Timeframe Handling**
   - Use consistent timeframe representations across the platform
   - Implement proper alignment for multi-timeframe analysis
   - Document timeframe conversion logic

### Technical Analysis Components

1. **Indicator Implementation**
   - Implement indicators as pure functions where possible
   - Include validation for input parameters
   - Document the mathematical formula and interpretation
   - Include unit tests with known values

2. **Pattern Recognition**
   - Document pattern criteria clearly
   - Include confidence levels in pattern detection
   - Implement visualization helpers for detected patterns

### Trading Components

1. **Order Management**
   - Validate all order parameters before submission
   - Implement proper error handling for order failures
   - Include audit logging for all order operations
   - Document risk checks and validations

2. **Position Management**
   - Implement proper position tracking and reconciliation
   - Include validation for position updates
   - Document position calculation logic

### Risk Management Components

1. **Risk Calculations**
   - Document all risk calculation formulas
   - Implement validation for risk parameters
   - Include unit tests with known values
   - Log all risk limit violations

2. **Exposure Management**
   - Implement proper exposure tracking and limits
   - Include validation for exposure updates
   - Document exposure calculation logic

## File and Directory Structure

### Python Service Structure

```
service-name/
├── service_name/
│   ├── __init__.py
│   ├── main.py                  # Application entry point
│   ├── config.py                # Configuration management
│   ├── api/                     # API endpoints
│   │   ├── __init__.py
│   │   ├── routes/              # Route definitions
│   │   └── models/              # API models (request/response)
│   ├── domain/                  # Domain models and logic
│   │   ├── __init__.py
│   │   ├── models/              # Domain entities
│   │   └── services/            # Domain services
│   ├── adapters/                # External service adapters
│   │   ├── __init__.py
│   │   └── external_service/    # Specific external service
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

### JavaScript/TypeScript Frontend Structure

```
ui-service/
├── src/
│   ├── components/              # UI components
│   │   ├── common/              # Shared components
│   │   ├── charts/              # Chart components
│   │   ├── orders/              # Order-related components
│   │   └── analysis/            # Analysis components
│   ├── hooks/                   # Custom React hooks
│   │   ├── useMarketData.ts
│   │   └── useOrderBook.ts
│   ├── services/                # API services
│   │   ├── marketDataService.ts
│   │   └── orderService.ts
│   ├── store/                   # State management
│   │   ├── actions/
│   │   ├── reducers/
│   │   └── selectors/
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

## Automated Tools and Enforcement

### Python Tools

1. **Black**: Code formatting with line length of 88
2. **isort**: Import sorting with Black compatibility
3. **flake8**: Linting with custom rules
4. **mypy**: Static type checking
5. **pylint**: Advanced static analysis

### JavaScript/TypeScript Tools

1. **Prettier**: Code formatting
2. **ESLint**: Linting with custom rules
3. **TypeScript**: Static type checking

### CI/CD Integration

1. **Pre-commit Hooks**: Run linting and formatting before commits
2. **CI Checks**: Enforce standards in pull requests
3. **Automated Reports**: Generate code quality reports

## Migration Strategy

1. **Phased Approach**
   - Start with critical services
   - Apply standards to new code first
   - Gradually refactor existing code

2. **Documentation**
   - Provide examples of before/after refactoring
   - Document common patterns and anti-patterns
   - Create cheat sheets for quick reference

3. **Training**
   - Conduct knowledge sharing sessions
   - Provide pair programming support
   - Create self-paced learning resources

## Exceptions and Flexibility

While consistency is important, some flexibility is necessary:

1. **Legacy Code**
   - Document exceptions for legacy code
   - Create migration plans for critical components
   - Apply standards incrementally during refactoring

2. **Domain-Specific Exceptions**
   - Document justified exceptions for domain-specific needs
   - Require approval for non-standard patterns
   - Ensure exceptions don't compromise overall architecture

3. **External Libraries**
   - Document integration patterns for external libraries
   - Create wrappers to maintain consistent interfaces
   - Isolate external dependencies in adapter layers

## Review and Evolution

These standards should evolve with the platform:

1. **Regular Reviews**
   - Schedule quarterly reviews of standards
   - Collect feedback from development teams
   - Update standards based on lessons learned

2. **Metrics and Monitoring**
   - Track adherence to standards
   - Monitor impact on development velocity
   - Measure code quality improvements

3. **Continuous Improvement**
   - Refine standards based on project needs
   - Incorporate new best practices
   - Remove outdated or unnecessary rules