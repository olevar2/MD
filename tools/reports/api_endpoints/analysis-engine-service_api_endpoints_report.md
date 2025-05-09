# API Endpoint Validation Report: analysis-engine-service

## Compliance Summary

- Total Endpoints: 135
- Compliant Endpoints: 5 (3%)
- Non-Compliant Endpoints: 130 (97%)

## Non-Compliant Endpoints

### GET /api/v1/adaptive-layer/adaptations/history

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/adaptive-layer/feedback/insights/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/adaptive-layer/feedback/outcomes

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/adaptive-layer/feedback/performance/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/adaptive-layer/parameters/adjust

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/adaptive-layer/parameters/generate

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/adaptive-layer/parameters/history/{strategy_id}/{instrument}/{timeframe}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/adaptive-layer/strategy/effectiveness-trend

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/adaptive-layer/strategy/recommendations

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/adaptive-layer/strategy/update

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\adaptive_layer.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/analysis/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\analysis_results_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/analysis/confluence/analyze

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\analysis_results_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/analysis/multi_timeframe/analyze

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\analysis_results_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/analysis/{analyzer_name}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\analysis_results_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/analysis/{analyzer_name}/analyze

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\analysis_results_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/analysis/{analyzer_name}/effectiveness

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\analysis_results_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/analysis/{analyzer_name}/effectiveness/record

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\analysis_results_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/analysis/{analyzer_name}/performance

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\analysis_results_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/causal-visualization/available-symbols

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_visualization_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/causal-visualization/available-timeframes

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_visualization_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal-visualization/causal-graph

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_visualization_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal-visualization/counterfactual-scenario

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_visualization_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal-visualization/enhanced-data

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_visualization_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal-visualization/intervention-effect

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_visualization_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal/correlation-breakdown-risk

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\causal_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal/counterfactual-scenarios

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\causal_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal/currency-pair-relationships

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\causal_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal/discover-structure

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal/enhance-trading-signals

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\causal_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal/estimate-effects

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal/generate-counterfactuals

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\causal_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/causal/regime-change-drivers

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\causal_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/chat/execute-action

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\chat\api_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/chat/history

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\chat\api_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### DELETE /api/v1/chat/history

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\chat\api_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/chat/message

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\chat\api_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/correlations/changes/{symbol1}/{symbol2}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/correlations/cross-asset/{symbol}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/correlations/matrix

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/correlations/symbol/{symbol}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/correlations/update

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/correlations/visualization

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/feedback/insights/learning/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/feedback/items

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/feedback/parameters/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/feedback/statistics

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/feedback/strategy/evaluate/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/feedback/strategy/mutate/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/feedback/strategy/mutation-effectiveness/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/feedback/strategy/outcome

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/feedback/strategy/versions/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/feedback/submit

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /api/v1/feedback/system/reset-stats

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/feedback/system/status

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/monitoring/memory

**File:** D:\MD\forex_trading_platform\analysis-engine-service\tests\api\test_memory_monitoring.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /api/v1/monitoring/memory

**File:** D:\MD\forex_trading_platform\analysis-engine-service\tests\api\test_memory_monitoring.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /complementarity

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\effectiveness_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /comprehensive

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\effectiveness_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /correlation/analyze

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /correlation/breakdown-detection

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /correlation/cointegration

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /correlation/lead-lag

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\correlation_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /cross-timeframe

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\effectiveness_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /dashboard

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\effectiveness_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /dashboard-data/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\tool_effectiveness.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/entries

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_router.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/insights

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_router.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /feedback/model/retrain/{model_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/parameters/{strategy_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/recent

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /feedback/retrain-model/{model_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_router.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### PUT /feedback/rules

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/statistics

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_router.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/statistics

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\feedback.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/stats

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/status

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_endpoints.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /feedback/submit

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_router.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /feedback/system/reset-metrics

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_router.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /feedback/system/status

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\feedback_router.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health_migrated.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health_migrated.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health/live

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health/live

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health_migrated.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health/live

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health_migrated.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health/ready

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health/ready

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health_migrated.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /health/ready

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\health_migrated.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /manipulation/detect

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\manipulation_detection.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /manipulation/fake-breakouts

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\manipulation_detection.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /manipulation/stop-hunting

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\manipulation_detection.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /manipulation/volume-anomalies

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\manipulation_detection.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/complementarity/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/detect/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/effectiveness-trends/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/history/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/optimal-conditions/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/performance-report/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/recommend-tools/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/regime-analysis/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /market-regime/underperforming-tools/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\market_regime_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /metrics

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\effectiveness_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /metrics/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\tool_effectiveness.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /monitoring/async-performance

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\monitoring.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /monitoring/async-performance/report

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\monitoring.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /monitoring/feedback-dlq

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\monitoring.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /monitoring/feedback-kafka

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\monitoring.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /monitoring/memory

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\monitoring.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /multi-asset/analysis-parameters/{symbol}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\multi_asset.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /multi-asset/assets

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\multi_asset.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /multi-asset/assets/{symbol}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\multi_asset.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /multi-asset/correlations/{symbol}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\multi_asset.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /multi-asset/groups

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\multi_asset.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /multi-asset/groups/{group_name}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\multi_asset.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /nlp/analyze-economic-report

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\nlp_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /nlp/analyze-news

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\nlp_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /nlp/combined-insights

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\nlp_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /nlp/market-sentiment

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\nlp_analysis.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /optimal-conditions

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\effectiveness_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /outcomes/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\tool_effectiveness.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /quality-analysis

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\signal_quality.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /quality-trends

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\signal_quality.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /regime

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\effectiveness_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /reports/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\tool_effectiveness.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /reports/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\tool_effectiveness.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### GET /reports/{report_id}

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\tool_effectiveness.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /signals/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\tool_effectiveness.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /signals/{signal_id}/quality

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\signal_quality.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### POST /temporal

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\effectiveness_analysis_api.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

### DELETE /tool/{tool_id}/data/

**File:** D:\MD\forex_trading_platform\analysis-engine-service\analysis_engine\api\v1\tool_effectiveness.py
**Framework:** fastapi

**Violations:**

- **url_structure**: URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$`
- **resource_naming**: Resource names should be plural nouns in kebab-case
  - Expected pattern: `^/?v\d+/[a-z-]+/[a-z-]+`

## Compliant Endpoints

- GET /v1/analysis-engine/healths
- GET /v1/analysis-engine/healths/live
- GET /v1/analysis-engine/healths/ready
- GET /v1/analysis/health-checks/liveness
- GET /v1/analysis/health-checks/readiness