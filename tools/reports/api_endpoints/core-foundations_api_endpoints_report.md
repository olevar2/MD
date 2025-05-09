# API Endpoint Validation Report: core-foundations

## Compliance Summary

- Total Endpoints: 3
- Compliant Endpoints: 0 (0%)
- Non-Compliant Endpoints: 3 (100%)

## Non-Compliant Endpoints

### GET /health

**File:** D:\MD\forex_trading_platform\core-foundations\core_foundations\api\health_check.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health/live

**File:** D:\MD\forex_trading_platform\core-foundations\core_foundations\api\health_check.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health/ready

**File:** D:\MD\forex_trading_platform\core-foundations\core_foundations\api\health_check.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

## Compliant Endpoints
