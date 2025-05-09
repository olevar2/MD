# API Endpoint Validation Report: chat_interface_template

## Compliance Summary

- Total Endpoints: 4
- Compliant Endpoints: 0 (0%)
- Non-Compliant Endpoints: 4 (100%)

## Non-Compliant Endpoints

### POST /api/v1/chat/execute-action

**File:** D:\MD\forex_trading_platform\chat_interface_template\api_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/chat/history

**File:** D:\MD\forex_trading_platform\chat_interface_template\api_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### DELETE /api/v1/chat/history

**File:** D:\MD\forex_trading_platform\chat_interface_template\api_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/chat/message

**File:** D:\MD\forex_trading_platform\chat_interface_template\api_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

## Compliant Endpoints
