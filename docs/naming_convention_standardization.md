# Forex Trading Platform Naming Convention Standardization

## Overview

This document outlines the standardized naming conventions for the Forex Trading Platform. The goal is to establish consistent naming patterns across the codebase to improve maintainability, readability, and reduce confusion.

## Current Issues

1. **Inconsistent Directory Naming**: The codebase currently uses a mix of kebab-case (e.g., `feature-store-service`) and snake_case (e.g., `feature_store_service`) for directory names.

2. **Duplicate Directories**: Some components have duplicate directories with different naming conventions (e.g., `analysis-engine-service` and `analysis_engine`).

3. **Inconsistent File Naming**: File naming patterns vary across the codebase.

4. **Inconsistent Module Naming**: Module imports and references use different naming conventions.

## Standardized Naming Conventions

### Directory Naming

1. **Service Directories**: Use kebab-case for top-level service directories.
   - Example: `feature-store-service`, `analysis-engine-service`

2. **Module Directories**: Use snake_case for module directories within services.
   - Example: `feature_store_service/api`, `analysis_engine/core`

### File Naming

1. **Python Files**: Use snake_case for Python files.
   - Example: `database_connection.py`, `api_router.py`

2. **JavaScript/TypeScript Files**: Use kebab-case for JavaScript and TypeScript files.
   - Example: `api-client.js`, `data-visualization.tsx`

3. **Configuration Files**: Use kebab-case for configuration files.
   - Example: `docker-compose.yml`, `tsconfig.json`

4. **Documentation Files**: Use kebab-case for documentation files.
   - Example: `api-reference.md`, `developer-guide.md`

### Module Naming

1. **Python Modules**: Use snake_case for Python module names.
   - Example: `from feature_store_service.api import router`

2. **JavaScript/TypeScript Modules**: Use camelCase for JavaScript and TypeScript module names.
   - Example: `import { dataService } from './dataService'`

### Variable and Function Naming

1. **Python Variables and Functions**: Use snake_case for Python variables and functions.
   - Example: `user_id`, `calculate_moving_average()`

2. **JavaScript/TypeScript Variables and Functions**: Use camelCase for JavaScript and TypeScript variables and functions.
   - Example: `userId`, `calculateMovingAverage()`

3. **Constants**: Use UPPER_SNAKE_CASE for constants in all languages.
   - Example: `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT_MS`

### Class Naming

1. **Python Classes**: Use PascalCase for Python classes.
   - Example: `class DatabaseConnection`, `class ApiClient`

2. **JavaScript/TypeScript Classes**: Use PascalCase for JavaScript and TypeScript classes.
   - Example: `class DataService`, `class UserAuthentication`

## Implementation Plan

1. **Phase 1: Documentation and Guidelines**
   - Create and distribute naming convention guidelines (this document)
   - Update coding standards documentation

2. **Phase 2: New Code Compliance**
   - Ensure all new code follows the standardized naming conventions
   - Add linting rules to enforce naming conventions

3. **Phase 3: Gradual Refactoring**
   - Refactor existing code to follow the standardized naming conventions
   - Prioritize high-impact areas and frequently modified code

4. **Phase 4: Automated Checks**
   - Implement automated checks to ensure compliance with naming conventions
   - Add pre-commit hooks to prevent non-compliant code from being committed

## Handling Duplicate Directories

For duplicate directories with different naming conventions (e.g., `analysis-engine-service` and `analysis_engine`), we will:

1. Keep the kebab-case version as the main service directory (e.g., `analysis-engine-service`)
2. Ensure the snake_case version (e.g., `analysis_engine`) is properly imported and referenced
3. Consider consolidating duplicate functionality in the long term

## Conclusion

Standardizing naming conventions across the Forex Trading Platform will improve code quality, reduce confusion, and make the codebase more maintainable. This document serves as a guide for both current and future development efforts.