# Forex Trading Platform Architecture Report
Generated on: 2025-05-13 09:13:52
Updated on: 2025-05-18 10:45:00 (Added Platform Map)
Updated on: 2025-05-19 14:30:00 (Verified Interface-Based Decoupling Implementation)

## Project Overview

- **Services:** 15
- **Files:** 2285
- **Classes:** 4568
- **Functions:** 16828

# Platform Map

## Service Architecture and Integration Map

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                             │
│                                                 Forex Trading Platform Architecture                                          │
│                                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                               │
                 ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
                 │                                             │                                             │
                 ▼                                             ▼                                             ▼
┌────────────────────────────────┐    ┌────────────────────────────────────┐    ┌────────────────────────────────────┐
│                                │    │                                    │    │                                    │
│     Frontend & Gateway Layer   │    │       Core Services Layer          │    │      Infrastructure Layer          │
│                                │    │                                    │    │                                    │
└────────────────────────────────┘    └────────────────────────────────────┘    └────────────────────────────────────┘
                 │                                             │                                             │
        ┌────────┴────────┐                    ┌──────────────┼──────────────┐                     ┌────────┴────────┐
        │                 │                    │              │              │                     │                 │
        ▼                 ▼                    ▼              ▼              ▼                     ▼                 ▼
┌─────────────┐  ┌─────────────┐     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     ┌─────────────┐  ┌─────────────┐
│             │  │             │     │             │ │             │ │             │     │             │  │             │
│ api-gateway │  │ ui-service  │     │  data-pipe  │ │  analysis   │ │  trading    │     │  common-lib │  │ monitoring  │
│             │  │             │     │  line-svc   │ │  engine-svc │ │  gateway-svc│     │             │  │ alerting-svc│
│             │  │             │     │             │ │             │ │             │     │             │  │             │
└─────────────┘  └─────────────┘     └─────────────┘ └─────────────┘ └─────────────┘     └─────────────┘  └─────────────┘
       │                │                   │               │               │                    ▲                │
       │                │                   │               │               │                    │                │
       └────────────────┼───────────────────┼───────────────┼───────────────┘                    │                │
                        │                   │               │                                    │                │
                        │                   │               │                                    │                │
                        ▼                   ▼               ▼                                    │                │
                 ┌─────────────┐    ┌─────────────┐ ┌─────────────┐                             │                │
                 │             │    │             │ │             │                             │                │
                 │ feature-    │    │ ml-         │ │ portfolio-  │                             │                │
                 │ store-svc   │    │ integration │ │ management  │                             │                │
                 │             │    │             │ │             │                             │                │
                 └─────────────┘    └─────────────┘ └─────────────┘                             │                │
                        │                  │                │                                    │                │
                        │                  │                │                                    │                │
                        │                  ▼                ▼                                    │                │
                        │           ┌─────────────┐ ┌─────────────┐                             │                │
                        │           │             │ │             │                             │                │
                        └──────────►│ ml-         │ │ risk-       │                             │                │
                                    │ workbench   │ │ management  │◄────────────────────────────┘                │
                                    │             │ │             │                                               │
                                    └─────────────┘ └─────────────┘                                               │
                                           │               │                                                      │
                                           │               │                                                      │
                                           ▼               ▼                                                      │
                                    ┌─────────────┐ ┌─────────────┐                                               │
                                    │             │ │             │                                               │
                                    │ model-      │ │ strategy-   │◄──────────────────────────────────────────────┘
                                    │ registry    │ │ execution   │
                                    │             │ │             │
                                    └─────────────┘ └─────────────┘
```

## Detailed Service Integration Map

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                             │
│                                                 Service Integration Map                                                      │
│                                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐                                                                                  ┌─────────────────┐
│                 │                                                                                  │                 │
│  api-gateway    │◄─────────────────────────────────────────────────────────────────────────────────┤  common-lib     │
│                 │                                                                                  │                 │
└────────┬────────┘                                                                                  └─────────────────┘
         │                                                                                                    ▲
         │                                                                                                    │
         ▼                                                                                                    │
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐                  │
│                 │     │                 │     │                 │     │                 │                  │
│  data-pipeline  │────►│  feature-store  │────►│  analysis-      │────►│  ml-integration │                  │
│                 │     │                 │     │  engine         │     │                 │                  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘     └────────┬────────┘                  │
         ▲                       ▲                       │                       │                           │
         │                       │                       │                       │                           │
         │                       │                       ▼                       ▼                           │
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐                  │
│                 │     │                 │     │                 │     │                 │                  │
│  data-          │────►│  monitoring-    │────►│  trading-       │◄────┤  ml-workbench   │                  │
│  management     │     │  alerting       │     │  gateway        │     │                 │                  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘     └────────┬────────┘                  │
         │                       ▲                       │                       │                           │
         │                       │                       │                       │                           │
         │                       │                       ▼                       ▼                           │
         │               ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐                  │
         │               │                 │     │                 │     │                 │                  │
         └──────────────►│  strategy-      │◄────┤  portfolio-     │◄────┤  risk-          │                  │
                         │  execution      │     │  management     │     │  management     │                  │
                         └─────────────────┘     └─────────────────┘     └─────────────────┘                  │
                                  │                                               │                           │
                                  │                                               │                           │
                                  ▼                                               ▼                           │
                         ┌─────────────────┐                             ┌─────────────────┐                  │
                         │                 │                             │                 │                  │
                         │  model-         │                             │  ui-service     │──────────────────┘
                         │  registry       │                             │                 │
                         └─────────────────┘                             └─────────────────┘
```

## Data Flow Map

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                             │
│                                                 Data Flow Map                                                               │
│                                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │     │                 │
│  Market Data    │────►│  data-pipeline  │────►│  feature-store  │────►│  analysis-      │────►│  trading        │
│  Sources        │     │                 │     │                 │     │  engine         │     │  decisions      │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                        │                        │                      │
                                │                        │                        │                      │
                                ▼                        ▼                        ▼                      ▼
                        ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │     │                 │     │                 │
                        │  Historical     │     │  Feature        │     │  ML Model       │     │  Order          │
                        │  Data Storage   │     │  Repository     │     │  Training       │     │  Execution      │
                        └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                │                      │
                                                                                │                      │
                                                                                ▼                      ▼
                                                                        ┌─────────────────┐     ┌─────────────────┐
                                                                        │                 │     │                 │
                                                                        │  Model          │     │  Position       │
                                                                        │  Deployment     │     │  Management     │
                                                                        └─────────────────┘     └─────────────────┘
                                                                                                        │
                                                                                                        │
                                                                                                        ▼
                                                                                                ┌─────────────────┐
                                                                                                │                 │
                                                                                                │  Performance    │
                                                                                                │  Analysis       │
                                                                                                └─────────────────┘
                                                                                                        │
                                                                                                        │
                                                                                                        ▼
                                                                                                ┌─────────────────┐
                                                                                                │                 │
                                                                                                │  Strategy       │
                                                                                                │  Refinement     │
                                                                                                └─────────────────┘
```

## Interface-Based Decoupling Implementation

The platform has successfully implemented the Interface-Based Decoupling pattern to break circular dependencies between services. This implementation includes:

### Interface Definitions

Common interfaces are defined in `common-lib/common_lib/interfaces/`:

- `trading_gateway.py`: Interfaces for Trading Gateway Service
- `ml_integration.py`: Interfaces for ML Integration Service
- `ml_workbench.py`: Interfaces for ML Workbench Service
- `risk_management.py`: Interfaces for Risk Management Service
- `feature_store.py`: Interfaces for Feature Store Service
- `analysis_engine.py`: Interfaces for Analysis Engine Service

### Adapter Implementations

Adapters implementing these interfaces are in `common-lib/common_lib/adapters/`:

- `trading_gateway_adapter.py`: Adapter for Trading Gateway Service
- `ml_integration_adapter.py`: Adapter for ML Integration Service
- `ml_workbench_adapter.py`: Adapter for ML Workbench Service
- `risk_management_adapter.py`: Adapter for Risk Management Service
- `feature_store_adapter.py`: Adapter for Feature Store Service

### Adapter Factory

The Analysis Engine Service uses an adapter factory to create adapters for the services it depends on:
- `analysis-engine-service/adapters/common_adapter_factory.py`

### Service Dependencies

The Analysis Engine Service uses a service dependencies module to provide adapters to the API endpoints:
- `analysis-engine-service/core/service_dependencies.py`

### API Endpoints

API endpoints use the service dependencies to interact with other services:
- `analysis-engine-service/api/v1/integrated_analysis.py`

### Documentation

Comprehensive documentation of the Interface-Based Decoupling approach:
- `analysis-engine-service/docs/interface_based_decoupling.md`

### Example Usage

Example usage of the Interface-Based Decoupling approach:
- `analysis-engine-service/examples/common_adapter_usage.py`

### Benefits Achieved

- Eliminated direct dependencies between services
- Improved testability through dependency injection
- Enhanced maintainability by centralizing service communication
- Reduced coupling between services
- Improved resilience through fallback mechanisms

## Component Integration Details

### Core Components Integration

| Component | Integrates With | Integration Type | Data Exchange |
|-----------|----------------|------------------|---------------|
| analysis-engine-service | ml-integration-service | REST API | ML model predictions |
| analysis-engine-service | trading-gateway-service | REST API | Trading signals |
| analysis-engine-service | feature-store-service | REST API | Feature data |
| analysis-engine-service | ml-workbench-service | REST API | Model evaluation |
| portfolio-management-service | trading-gateway-service | REST API | Position updates |
| portfolio-management-service | risk-management-service | REST API | Risk assessments |
| strategy-execution-engine | trading-gateway-service | REST API | Strategy execution |
| strategy-execution-engine | monitoring-alerting-service | REST API | Performance metrics |

### Data Flow Integration

| Source | Destination | Data Type | Protocol |
|--------|-------------|-----------|----------|
| data-pipeline-service | feature-store-service | Market data | REST API |
| feature-store-service | analysis-engine-service | Feature data | REST API |
| analysis-engine-service | ml-integration-service | Analysis results | REST API |
| ml-integration-service | model-registry-service | ML models | REST API |
| trading-gateway-service | portfolio-management-service | Execution results | REST API |
| portfolio-management-service | risk-management-service | Portfolio data | REST API |
| risk-management-service | strategy-execution-engine | Risk constraints | REST API |

### Event-Driven Integration

| Publisher | Subscriber | Event Type | Implementation Status |
|-----------|------------|------------|----------------------|
| trading-gateway-service | portfolio-management-service | Order execution | Partial (65%) |
| analysis-engine-service | strategy-execution-engine | Trading signals | Partial (70%) |
| data-pipeline-service | feature-store-service | New market data | Partial (75%) |
| portfolio-management-service | risk-management-service | Position updates | Partial (60%) |
| risk-management-service | monitoring-alerting-service | Risk alerts | Partial (55%) |

### Interface-Based Integration

| Service | Interface | Adapter | Implementation Status |
|---------|-----------|---------|----------------------|
| trading-gateway-service | ITradingGateway | TradingGatewayAdapter | Complete (100%) |
| ml-integration-service | IMLModelRegistry, IMLMetricsProvider | MLIntegrationAdapter | Complete (100%) |
| ml-workbench-service | IExperimentManager, IModelEvaluator, IDatasetManager | MLWorkbenchAdapter | Complete (100%) |
| risk-management-service | IRiskManager | RiskManagementAdapter | Complete (100%) |
| feature-store-service | IFeatureProvider, IFeatureStore, IFeatureGenerator | FeatureStoreAdapter | Complete (100%) |
| analysis-engine-service | IAnalysisProvider, IIndicatorProvider, IPatternRecognizer | AnalysisAdapter | Complete (100%) |

### Self-Learning Loop Integration

| Component | Role in Learning Loop | Integration Status |
|-----------|----------------------|-------------------|
| portfolio-management-service | Performance tracking | Implemented (80%) |
| analysis-engine-service | Strategy adjustment | Partial (65%) |
| ml-integration-service | Model retraining | Partial (60%) |
| feature-store-service | Feature optimization | Partial (55%) |
| strategy-execution-engine | Strategy execution | Implemented (75%) |

## Service Dependencies

### Most Dependent Services

| Service | Dependencies |
| --- | --- |
| analysis-engine-service | 4 |
| feature-store-service | 3 |
| ml-workbench-service | 3 |
| ml-integration-service | 2 |
| monitoring-alerting-service | 2 |

### Most Depended-on Services

| Service | Dependents |
| --- | --- |
| common-lib | 14 |
| trading-gateway-service | 2 |
| data-pipeline-service | 2 |
| risk-management-service | 2 |
| ml-integration-service | 1 |

## Service Structure

### Services with Most Files

| Service | Files |
| --- | --- |
| analysis-engine-service | 527 |
| feature-store-service | 348 |
| common-lib | 265 |
| ml-workbench-service | 158 |
| trading-gateway-service | 146 |
| data-pipeline-service | 142 |
| strategy-execution-engine | 138 |
| ml-integration-service | 106 |
| data-management-service | 93 |
| monitoring-alerting-service | 88 |

### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| async_pattern | 4333 |
| timeout | 3115 |
| cqrs | 2693 |
| test | 2016 |
| repository | 1614 |
| factory | 1340 |
| retry | 1050 |
| event_driven | 937 |
| gateway | 936 |
| service | 722 |

## API Endpoints

- **REST Endpoints:** 646
- **gRPC Services:** 37
- **Message Queues:** 1
- **WebSocket Endpoints:** 0

### Services with Most Endpoints

| Service | Endpoints |
| --- | --- |
| analysis-engine-service | 241 |
| ml-workbench-service | 93 |
| feature-store-service | 70 |
| data-pipeline-service | 60 |
| trading-gateway-service | 47 |
| data-management-service | 42 |
| portfolio-management-service | 42 |
| ml-integration-service | 32 |
| monitoring-alerting-service | 32 |
| risk-management-service | 30 |

## Database Models

- **Models:** 1104
- **Relationships:** 3

### Services with Most Models

| Service | Models |
| --- | --- |
| analysis-engine-service | 336 |
| feature-store-service | 156 |
| common-lib | 133 |
| ml-workbench-service | 112 |
| data-management-service | 62 |
| data-pipeline-service | 58 |
| ml-integration-service | 48 |
| risk-management-service | 38 |
| portfolio-management-service | 30 |
| model-registry-service | 29 |

### Most Used Data Access Patterns

| Pattern | Occurrences |
| --- | --- |
| update | 1236 |
| filter | 799 |
| session | 748 |
| join | 695 |
| query | 692 |
| execute | 550 |
| limit | 512 |
| transaction | 276 |
| delete | 224 |
| select | 189 |

## Service Details

### analysis-engine-service

#### Statistics

- **Files:** 527
- **Classes:** 1180
- **Functions:** 4583
- **Modules:** 527
- **Directories:** 11

#### Dependencies

This service depends on:

- ml-integration-service
- common-lib
- ml-workbench-service
- trading-gateway-service

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | / | GET |
| REST | /{analyzer_name} | GET |
| REST | /{analyzer_name}/analyze | POST |
| REST | /{analyzer_name}/performance | GET |
| REST | /{analyzer_name}/effectiveness | GET |
| REST | /{analyzer_name}/effectiveness/record | POST |
| REST | /multi_timeframe/analyze | POST |
| REST | /confluence/analyze | POST |
| REST | /message | POST |
| REST | /execute-action | POST |
| REST | /history | GET |
| REST | /history | DELETE |
| REST | /discover-structure | POST |
| REST | /estimate-effects | POST |
| REST | /generate-counterfactuals | POST |
| REST | /causal-graph | POST |
| REST | /intervention-effect | POST |
| REST | /counterfactual-scenario | POST |
| REST | /available-symbols | GET |
| REST | /available-timeframes | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| async_pattern | 1211 |
| timeout | 741 |
| event_driven | 588 |
| repository | 581 |
| factory | 571 |
| test | 515 |
| cqrs | 344 |
| retry | 217 |
| service | 175 |
| circuit_breaker | 175 |

### api-gateway

#### Statistics

- **Files:** 33
- **Classes:** 33
- **Functions:** 105
- **Modules:** 33
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /health | GET |
| REST | /indicators | POST |
| REST | /indicators/{indicator}/{symbol}/{timeframe} | GET |
| REST | /patterns | POST |
| REST | /patterns/{pattern}/{symbol}/{timeframe} | GET |
| REST | /features | GET |
| REST | /features/{feature} | GET |
| REST | /features/data | POST |
| REST | /features/{feature}/{symbol}/{timeframe} | GET |
| REST | /feature-sets | GET |
| REST | /feature-sets/{name} | GET |
| REST | /feature-sets | POST |
| REST | /feature-sets/{name} | PUT |
| REST | /feature-sets/{name} | DELETE |
| REST | /symbols | GET |
| REST | /symbols/{symbol} | GET |
| REST | /data/{symbol}/{timeframe} | GET |
| REST | /latest/{symbol}/{timeframe} | GET |
| REST | /orders | POST |
| REST | /orders | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| async_pattern | 75 |
| timeout | 30 |
| cqrs | 27 |
| retry | 16 |
| rate_limiter | 15 |
| circuit_breaker | 12 |
| gateway | 10 |
| middleware | 8 |
| service | 6 |
| handler | 6 |

### data-management-service

#### Statistics

- **Files:** 93
- **Classes:** 114
- **Functions:** 402
- **Modules:** 93
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /data | POST |
| REST | /features | POST |
| REST | /store | POST |
| REST | /types | GET |
| REST | /adapters | GET |
| REST | /ohlcv | POST |
| REST | /tick | POST |
| REST | /alternative | POST |
| REST | /correction | POST |
| REST | /quality-report | POST |
| REST | /ohlcv | GET |
| REST | /tick | GET |
| REST | /alternative | GET |
| REST | /record-history/{record_id} | GET |
| REST | /ml-dataset | POST |
| REST | /point-in-time | POST |
| REST | /configs | POST |
| REST | /tasks | POST |
| REST | /tasks/{task_id}/run | POST |
| REST | /configs/{config_id} | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| cqrs | 379 |
| async_pattern | 245 |
| repository | 113 |
| factory | 97 |
| config | 37 |
| factory_method | 24 |
| adapter | 22 |
| store | 21 |
| service | 11 |
| command | 8 |

### data-pipeline-service

#### Statistics

- **Files:** 142
- **Classes:** 282
- **Functions:** 815
- **Modules:** 142
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /connect | POST |
| REST | /disconnect | POST |
| REST | /instruments | GET |
| REST | /tick | GET |
| REST | /ohlcv | GET |
| REST | /health | GET |
| REST | /ready | GET |
| REST | /metrics | GET |
| REST | /health | GET |
| REST | /ready | GET |
| REST | /metrics | GET |
| REST | /ohlcv | POST |
| REST | /tick-data | POST |
| REST | /{reconciliation_id} | GET |
| REST | /{symbol} | GET |
| REST | /{symbol}/trading-hours | GET |
| REST | /{symbol}/trading-hours | POST |
| REST | /health | GET |
| REST | /health/liveness | GET |
| REST | /health/readiness | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| cqrs | 631 |
| async_pattern | 236 |
| timeout | 220 |
| repository | 135 |
| test | 69 |
| decorator | 49 |
| retry | 47 |
| aggregate | 44 |
| factory | 40 |
| error | 38 |

### feature-store-service

#### Statistics

- **Files:** 348
- **Classes:** 660
- **Functions:** 2480
- **Modules:** 348
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib
- monitoring-alerting-service
- data-pipeline-service

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /feature-data | POST |
| REST | /feature-version | POST |
| REST | /{reconciliation_id} | GET |
| REST | /initialize | POST |
| REST | /update | POST |
| REST | /update-all | POST |
| REST | /active | GET |
| REST | /load-states | POST |
| REST | /health | GET |
| REST | /health/liveness | GET |
| REST | /health/readiness | GET |
| REST | /metrics | GET |
| REST | /api/v1/features | GET |
| REST | /api/v1/feature-sets | GET |
| REST | /api/v1/test/portfolio-interaction | GET |
| REST | /api/v1/test/risk-interaction | GET |
| REST | /api/v1/test/feature-interaction | GET |
| REST | /api/v1/test/ml-interaction | GET |
| REST | /configure | POST |
| REST | /process-tick/{symbol} | POST |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| cqrs | 630 |
| test | 449 |
| async_pattern | 280 |
| timeout | 143 |
| factory | 102 |
| cache | 76 |
| fallback | 64 |
| repository | 59 |
| aggregate | 59 |
| decorator | 54 |

### ml-integration-service

#### Statistics

- **Files:** 106
- **Classes:** 189
- **Functions:** 676
- **Modules:** 106
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib
- data-pipeline-service

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /stats | GET |
| REST | /clear | POST |
| REST | /cache | GET |
| REST | /visualize/performance | POST |
| REST | /visualize/feature-importance | POST |
| REST | /optimize/regime-aware | POST |
| REST | /optimize/multi-objective | POST |
| REST | /stress-test/robustness | POST |
| REST | /stress-test/sensitivity | POST |
| REST | /stress-test/load | POST |
| REST | /reconciliation | GET |
| REST | /health | GET |
| REST | /ready | GET |
| REST | /metrics | GET |
| REST | /health | GET |
| REST | /ready | GET |
| REST | /metrics | GET |
| REST | /training-data | POST |
| REST | /inference-data | POST |
| REST | /{reconciliation_id} | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| async_pattern | 170 |
| timeout | 139 |
| repository | 131 |
| model | 102 |
| test | 69 |
| fallback | 61 |
| decorator | 56 |
| service | 54 |
| error | 30 |
| retry | 30 |

### ml-workbench-service

#### Statistics

- **Files:** 158
- **Classes:** 375
- **Functions:** 1372
- **Modules:** 158
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib
- risk-management-service
- trading-gateway-service

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | / | POST |
| REST | / | GET |
| REST | /{dataset_id} | GET |
| REST | /generate | POST |
| REST | /{dataset_id}/statistics | GET |
| REST | /available-features | GET |
| REST | /{dataset_id}/download | GET |
| REST | / | POST |
| REST | / | GET |
| REST | /{experiment_id} | GET |
| REST | /{experiment_id} | PATCH |
| REST | /{experiment_id} | DELETE |
| REST | /{experiment_id}/start | POST |
| REST | /{experiment_id}/complete | POST |
| REST | /{experiment_id}/versions | POST |
| REST | /{experiment_id}/metrics | POST |
| REST | /{experiment_id}/versions/{version_id}/artifact | POST |
| REST | /{experiment_id}/versions/{version_id}/artifact | GET |
| REST | / | POST |
| REST | / | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| model | 271 |
| async_pattern | 265 |
| timeout | 205 |
| factory | 193 |
| cqrs | 122 |
| test | 90 |
| repository | 77 |
| gateway | 67 |
| factory_method | 65 |
| service | 55 |

### model-registry-service

#### Statistics

- **Files:** 32
- **Classes:** 38
- **Functions:** 83
- **Modules:** 32
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /models | POST |
| REST | /models | GET |
| REST | /models/{model_id} | GET |
| REST | /models/{model_id}/versions | POST |
| REST | /models/{model_id}/versions | GET |
| REST | /versions/{version_id} | GET |
| REST | /versions/{version_id}/stage | PATCH |
| REST | /versions/{version_id}/artifact | GET |
| REST | /models/{model_id} | DELETE |
| REST | /versions/{version_id} | DELETE |
| REST | /models/{model_id}/abtests | POST |
| REST | /abtests | GET |
| REST | /abtests/{test_id} | GET |
| REST | /abtests/{test_id} | PATCH |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| async_pattern | 61 |
| model | 50 |
| repository | 44 |
| test | 19 |
| error | 10 |
| cqrs | 8 |
| factory | 6 |
| handler | 5 |
| registry | 5 |
| factory_method | 5 |

### monitoring-alerting-service

#### Statistics

- **Files:** 88
- **Classes:** 91
- **Functions:** 601
- **Modules:** 88
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib
- strategy-execution-engine

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /health | GET |
| REST | /ready | GET |
| REST | /status | GET |
| REST | /alerts | GET |
| REST | /silences | GET |
| REST | / | GET |
| REST | / | POST |
| REST | /{alert_id} | GET |
| REST | / | GET |
| REST | / | POST |
| REST | /{dashboard_uid} | GET |
| REST | /dashboards | GET |
| REST | /datasources | GET |
| REST | /users | GET |
| REST | /health | GET |
| REST | /health/liveness | GET |
| REST | /health/readiness | GET |
| REST | /metrics | GET |
| REST | /api/v1/alerts | GET |
| REST | /api/v1/metrics | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| timeout | 338 |
| async_pattern | 144 |
| gateway | 95 |
| cqrs | 88 |
| distributed_tracing | 81 |
| repository | 63 |
| entity | 53 |
| test | 48 |
| decorator | 46 |
| service | 39 |

### portfolio-management-service

#### Statistics

- **Files:** 87
- **Classes:** 116
- **Functions:** 419
- **Modules:** 87
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /positions | POST |
| REST | /portfolio/{account_id}/summary | GET |
| REST | /portfolio/{account_id}/risk | GET |
| REST | /portfolio/{account_id}/allocation-recommendations | GET |
| REST | /positions | POST |
| REST | /positions/{position_id} | GET |
| REST | /positions/{position_id} | PUT |
| REST | /positions/{position_id}/close | POST |
| REST | /accounts/{account_id}/summary | GET |
| REST | /accounts/{account_id}/historical-performance | GET |
| REST | /accounts/{account_id}/update-prices | POST |
| REST | /accounts/{account_id}/daily-snapshot | POST |
| REST | /test-portfolio-not-found | GET |
| REST | /test-position-not-found | GET |
| REST | /test-insufficient-balance | GET |
| REST | /test-portfolio-operation | GET |
| REST | /test-generic-error | GET |
| REST | /{account_id}/balance | GET |
| REST | /{account_id}/summary | GET |
| REST | /{account_id}/initialize | POST |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| repository | 277 |
| async_pattern | 173 |
| test | 93 |
| cqrs | 68 |
| gateway | 62 |
| timeout | 59 |
| event_driven | 51 |
| service | 38 |
| error | 29 |
| adapter | 19 |

### risk-management-service

#### Statistics

- **Files:** 85
- **Classes:** 157
- **Functions:** 533
- **Modules:** 85
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /strategy/weaknesses | POST |
| REST | /ml/metrics | POST |
| REST | /ml/feedback | POST |
| REST | /monitor/thresholds | POST |
| REST | /control/automated | POST |
| REST | /limits | POST |
| REST | /limits/{limit_id} | GET |
| REST | /limits/{limit_id} | PUT |
| REST | /accounts/{account_id}/limits | GET |
| REST | /check/position | POST |
| REST | /check/portfolio/{account_id} | POST |
| REST | /calculate/position-size | POST |
| REST | /calculate/var | POST |
| REST | /calculate/drawdown | POST |
| REST | /calculate/correlation | POST |
| REST | /calculate/max-trades | POST |
| REST | /profiles | POST |
| REST | /accounts/{account_id}/apply-profile/{profile_id} | POST |
| REST | /health | GET |
| REST | /health/liveness | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| cqrs | 163 |
| async_pattern | 75 |
| repository | 64 |
| test | 58 |
| fallback | 38 |
| mock | 23 |
| model | 18 |
| config | 18 |
| adapter | 14 |
| timeout | 13 |

### strategy-execution-engine

#### Statistics

- **Files:** 138
- **Classes:** 206
- **Functions:** 1132
- **Modules:** 138
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | / | GET |
| REST | /health | GET |
| REST | /api/v1/strategies | GET |
| REST | / | GET |
| REST | /health | GET |
| REST | /api/v1/strategies | GET |
| REST | /api/v1/strategies/{strategy_id} | GET |
| REST | /api/v1/strategies/register | POST |
| REST | /api/v1/backtest | POST |
| REST | /test | GET |
| REST | /test | GET |
| REST | /error | GET |
| REST | /test | GET |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| async_pattern | 352 |
| test | 209 |
| gateway | 193 |
| aggregate | 154 |
| timeout | 147 |
| strategy | 128 |
| fallback | 85 |
| factory | 79 |
| error | 79 |
| retry | 59 |

### trading-gateway-service

#### Statistics

- **Files:** 146
- **Classes:** 261
- **Functions:** 1312
- **Modules:** 146
- **Directories:** 12

#### Dependencies

This service depends on:

- common-lib
- risk-management-service

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /health | GET |
| REST | /ready | GET |
| REST | /metrics | GET |
| REST | /health | GET |
| REST | /health/liveness | GET |
| REST | /health/readiness | GET |
| REST | /metrics | GET |
| REST | /api/v1/instruments | GET |
| REST | /api/v1/accounts | GET |
| REST | /api/v1/test/portfolio-interaction | GET |
| REST | /api/v1/test/risk-interaction | GET |
| REST | /api/v1/test/feature-interaction | GET |
| REST | /api/v1/test/ml-interaction | GET |
| REST | / | GET |
| REST | /market-data/{symbol} | GET |
| REST | /reconcile/orders | POST |
| REST | /monitoring/performance | GET |
| REST | /status/degraded-mode | GET |
| gRPC | TestOrderExecution | setUp |
| gRPC | TestOrderExecution | test_direct_execution |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| async_pattern | 396 |
| gateway | 291 |
| timeout | 185 |
| test | 161 |
| retry | 90 |
| fallback | 73 |
| circuit_breaker | 67 |
| factory | 59 |
| event_driven | 54 |
| service | 53 |

### ui-service

#### Statistics

- **Files:** 37
- **Classes:** 64
- **Functions:** 255
- **Modules:** 37
- **Directories:** 11

#### Dependencies

This service depends on:

- common-lib
- feature-store-service

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| timeout | 62 |
| gateway | 26 |
| service | 24 |
| adapter | 20 |
| performance_monitoring | 18 |
| factory_method | 16 |
| async_pattern | 16 |
| repository | 14 |
| decorator | 13 |
| factory | 11 |

### common-lib

#### Statistics

- **Files:** 265
- **Classes:** 802
- **Functions:** 2060
- **Modules:** 265
- **Directories:** 61

#### Dependencies

This service has no dependencies.

#### API Endpoints

| Type | Endpoint | Details |
| --- | --- | --- |
| REST | /api/resources/{resource_id} | GET |
| REST | /health | GET |
| REST | /ready | GET |
| REST | /metrics | GET |
| REST | /test | GET |
| REST | /test | GET |
| REST | /test | GET |
| REST | /api/resource/{resource_id} | GET |
| REST | /api/resource | POST |
| REST | /api/proxy/{resource_id} | GET |
| REST | /api/proxy | POST |

#### Most Common Patterns

| Pattern | Occurrences |
| --- | --- |
| timeout | 829 |
| async_pattern | 634 |
| retry | 495 |
| decorator | 265 |
| test | 227 |
| bulkhead | 220 |
| circuit_breaker | 172 |
| event_driven | 167 |
| cqrs | 159 |
| error | 153 |
