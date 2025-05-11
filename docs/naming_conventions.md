# Forex Trading Platform Naming Conventions

This document defines the standard naming conventions for the forex trading platform.

## General Principles

- Be consistent
- Be descriptive
- Be concise
- Follow language-specific conventions

## File and Directory Naming

### Python Files and Directories

- Use `snake_case` for all Python files and directories
- Example: `market_data_service`, `feature_store.py`, `data_processing_utils.py`

### JavaScript/TypeScript Files and Directories

- Use `kebab-case` for all JavaScript/TypeScript files and directories
- Example: `market-data-client`, `feature-store.ts`, `data-processing-utils.ts`

### Configuration Files

- Use `kebab-case` for configuration files
- Example: `app-config.yaml`, `database-config.json`

### Documentation Files

- Use `Title_Case.md` for documentation files
- Example: `Architecture_Overview.md`, `API_Documentation.md`

## Code Naming

### Python

- **Classes**: Use `PascalCase`
  - Example: `MarketDataProvider`, `FeatureStore`
- **Functions/Methods**: Use `snake_case`
  - Example: `get_market_data()`, `calculate_indicator()`
- **Variables**: Use `snake_case`
  - Example: `market_data`, `indicator_value`
- **Constants**: Use `UPPER_SNAKE_CASE`
  - Example: `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT`
- **Module-level variables**: Use `snake_case`
  - Example: `logger`, `config`

### JavaScript/TypeScript

- **Classes**: Use `PascalCase`
  - Example: `MarketDataClient`, `FeatureStore`
- **Functions/Methods**: Use `camelCase`
  - Example: `getMarketData()`, `calculateIndicator()`
- **Variables**: Use `camelCase`
  - Example: `marketData`, `indicatorValue`
- **Constants**: Use `UPPER_SNAKE_CASE`
  - Example: `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT`

## Service Naming

- Use `kebab-case` for service names in URLs, configuration, and deployment files
  - Example: `market-data-service`, `feature-store-service`
- Use `snake_case` for service names in Python imports and module names
  - Example: `market_data_service`, `feature_store_service`

## Database Naming

- **Tables**: Use `snake_case` and plural form
  - Example: `market_data`, `indicators`, `user_preferences`
- **Columns**: Use `snake_case`
  - Example: `created_at`, `last_updated`, `indicator_value`
- **Indexes**: Use `idx_table_column` format
  - Example: `idx_market_data_timestamp`, `idx_indicators_symbol`
- **Foreign Keys**: Use `fk_table_reference` format
  - Example: `fk_indicators_market_data`

## API Naming

- **Endpoints**: Use `kebab-case` for paths
  - Example: `/api/v1/market-data`, `/api/v1/indicators/calculate`
- **Query Parameters**: Use `camelCase`
  - Example: `?symbolId=EURUSD&timeFrame=1h`
- **Request/Response Fields**: Use `camelCase` for JSON fields
  - Example: `{ "symbolId": "EURUSD", "timeFrame": "1h" }`

## Event Naming

- **Event Names**: Use `past.tense.kebab-case`
  - Example: `market-data.updated`, `indicator.calculated`
- **Event Payload Fields**: Use `camelCase`
  - Example: `{ "symbolId": "EURUSD", "indicatorValue": 1.2345 }`

## Implementation Guidelines

1. When renaming files and directories, ensure all imports are updated
2. Update configuration files to reflect new naming conventions
3. Update documentation to reflect new naming conventions
4. Run tests after each change to ensure functionality is preserved
5. Commit changes in small, logical batches

## Exceptions

In some cases, it may be necessary to deviate from these conventions:

1. When integrating with third-party libraries that use different conventions
2. When maintaining backward compatibility with existing APIs
3. When following specific framework conventions

In these cases, document the exception and the reason for it.