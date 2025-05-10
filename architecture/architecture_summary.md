# Forex Trading Platform Architecture Summary

Analysis Date: 2025-05-10 03:47:43

## Overall Statistics

- Total Python Files: 1443
- Total Modules: 1443
- Total Classes: 3049
- Total Functions: 8189
- Total Dependencies: 89
- Circular Dependencies: 0

## Modules by Directory

| Directory | Module Count |
|-----------|--------------|
| analysis-engine-service | 426 |
| feature-store-service | 264 |
| strategy-execution-engine | 102 |
| ml-workbench-service | 101 |
| common-lib | 89 |
| trading-gateway-service | 76 |
| data-pipeline-service | 71 |
| data-management-service | 60 |
| risk-management-service | 54 |
| portfolio-management-service | 49 |
| ml-integration-service | 44 |
| core-foundations | 41 |
| monitoring-alerting-service | 36 |
| model-registry-service | 10 |
| analysis_engine | 9 |
| ui-service | 6 |
| feature_store_service | 5 |

## Most Imported Modules

| Module | Import Count |
|--------|--------------|
| feature_store_service.indicators.base_indicator | 49 |
| feature_store_service.indicators.harmonic_patterns.utils | 8 |
| analysis_engine.config.settings | 7 |
| analysis_engine.analysis.advanced_ta.elliott_wave.models | 5 |
| analysis_engine.analysis.market_regime.models | 4 |
| analysis_engine.analysis.advanced_ta.elliott_wave.pattern | 3 |
| analysis_engine.analysis.advanced_ta.elliott_wave.utils | 3 |
| analysis_engine.analysis.market_regime.classifier | 2 |
| analysis_engine.analysis.market_regime.detector | 2 |
| analysis_engine.core.deprecation_monitor | 2 |
| analysis_engine.analysis.market_regime.analyzer | 1 |
| feature_store_service.indicators.harmonic_patterns.models | 1 |
| feature_store_service.indicators.volatility.bands | 1 |
| feature_store_service.indicators.volatility.range | 1 |
| analysis-engine-service.main | 0 |
| analysis-engine-service.test_async_patterns | 0 |
| analysis-engine-service.test_config | 0 |
| analysis-engine-service.analysis_engine.dependencies | 0 |
| analysis-engine-service.analysis_engine.adapters.adaptive_strategy_adapter | 0 |
| analysis-engine-service.analysis_engine.adapters.analysis_engine_adapter | 0 |

## Modules with Most Imports

| Module | Import Count |
|--------|--------------|
| analysis-engine-service.analysis_engine.analysis.market_regime | 4 |
| analysis-engine-service.analysis_engine.analysis.advanced_ta.elliott_wave | 3 |
| analysis-engine-service.analysis_engine.analysis.advanced_ta.elliott_wave.analyzer | 3 |
| analysis_engine.analysis.market_regime.analyzer | 3 |
| feature-store-service.feature_store_service.indicators.harmonic_patterns.screener | 3 |
| analysis-engine-service.analysis_engine.analysis.advanced_ta.elliott_wave.counter | 2 |
| analysis-engine-service.analysis_engine.core.config | 2 |
| feature-store-service.feature_store_service.indicators.volatility | 2 |
| analysis-engine-service.test_config | 1 |
| analysis-engine-service.analysis_engine.analysis.advanced_ta.elliott_wave.pattern | 1 |
| analysis-engine-service.analysis_engine.analysis.advanced_ta.elliott_wave.validators | 1 |
| analysis-engine-service.analysis_engine.api.router | 1 |
| analysis-engine-service.analysis_engine.resilience.database | 1 |
| analysis-engine-service.analysis_engine.resilience.http_client | 1 |
| analysis-engine-service.analysis_engine.resilience.redis_client | 1 |
| analysis-engine-service.analysis_engine.services.feature_store_client | 1 |
| analysis-engine-service.tests.config.test_settings | 1 |
| analysis_engine.analysis.advanced_ta.elliott_wave.pattern | 1 |
| analysis_engine.analysis.market_regime.classifier | 1 |
| analysis_engine.analysis.market_regime.detector | 1 |

## Directory Dependencies

| Directory | Depends On |
|-----------|------------|
| analysis-engine-service | analysis_engine |
| feature-store-service | feature_store_service |

## Modules with Most Classes

| Module | Class Count |
|--------|-------------|
| common-lib.common_lib.exceptions | 23 |
| ml-workbench-service.ml_workbench_service.models.schemas | 22 |
| common-lib.common_lib.error.exceptions | 20 |
| core-foundations.core_foundations.events.event_schema | 20 |
| strategy-execution-engine.strategy_execution_engine.error.exceptions | 18 |
| analysis-engine-service.analysis_engine.analysis.backtesting.core | 15 |
| data-pipeline-service.data_pipeline_service.models.schemas | 15 |
| risk-management-service.risk_management_service.models.risk_limits | 15 |
| analysis-engine-service.analysis_engine.api.v1.standardized.effectiveness | 14 |
| analysis-engine-service.analysis_engine.core.exceptions_bridge | 14 |
| risk-management-service.risk_management_service.models.risk_metrics | 14 |
| analysis-engine-service.analysis_engine.api.v1.standardized.causal | 13 |
| analysis-engine-service.analysis_engine.api.distributed_computing_endpoints | 12 |
| analysis-engine-service.analysis_engine.api.v1.adaptive_layer | 12 |
| analysis-engine-service.analysis_engine.api.v1.standardized.adaptive_layer | 12 |
| analysis-engine-service.analysis_engine.api.v1.standardized.backtesting | 12 |
| data-management-service.data_management_service.historical.models | 12 |
| trading-gateway-service.trading_gateway_service.simulation.forex_broker_simulator | 12 |
| analysis-engine-service.common_lib.exceptions | 11 |
| analysis-engine-service.tests.integration.test_optimized_components_integration | 11 |

## Modules with Most Functions

| Module | Function Count |
|--------|----------------|
| analysis-engine-service.tests.integration.test_optimized_components_integration | 56 |
| analysis-engine-service.analysis_engine.analysis.alert_system | 36 |
| feature-store-service.feature_store_service.computation.incremental_calculator | 35 |
| analysis-engine-service.analysis_engine.config.settings | 32 |
| trading-gateway-service.trading_gateway_service.simulation.forex_broker_simulator | 32 |
| feature-store-service.feature_store_service.optimization.advanced_calculation | 31 |
| ml-integration-service.tests.test_ml_integration | 31 |
| ml-workbench-service.ml_workbench_service.explainability.explainability_module | 30 |
| ui-service.components.custom_dashboards | 30 |
| core-foundations.core_foundations.resilience.service_registry | 29 |
| monitoring-alerting-service.alerts.indicator_alerts | 29 |
| trading-gateway-service.trading_gateway_service.simulation.advanced_news_sentiment_simulator | 29 |
| analysis-engine-service.analysis_engine.analysis.visualization | 28 |
| trading-gateway-service.tests.services.test_refactored_execution_service | 28 |
| ui-service.components.dashboard_components | 28 |
| analysis-engine-service.analysis_engine.analysis.ml_feature_transformers | 27 |
| analysis-engine-service.analysis_engine.analysis.signal_system | 27 |
| feature-store-service.feature_store_service.adapters.fibonacci_adapter | 27 |
| feature-store-service.feature_store_service.indicators.incremental_indicators | 27 |
| trading-gateway-service.trading_gateway_service.resilience.degraded_mode_strategies | 27 |

## File Extensions

| Extension | Count |
|-----------|-------|
| .py | 1615 |
| .md | 204 |
| .json | 106 |
| .tsx | 105 |
| .ts | 58 |
| .yaml | 40 |
| .yml | 38 |
| .js | 32 |
| .toml | 24 |
| .txt | 17 |
| .png | 14 |
| .example | 13 |
| .lock | 12 |
| .jsx | 8 |
| .bat | 5 |
| .ps1 | 5 |
| .html | 5 |
| .ini | 4 |
| .csv | 4 |
| .log | 3 |
| .tf | 3 |
| .pt | 3 |
| .sql | 2 |
| .tfvars | 2 |
| .test | 1 |
| .db | 1 |
| .db-shm | 1 |
| .db-wal | 1 |
| .msi | 1 |
| .tmpl | 1 |
| .jupyter | 1 |
| .id | 1 |
| .sh | 1 |
| .dev | 1 |
| .bak | 1 |
| .scss | 1 |

## Longest Dependency Chains

1. analysis-engine-service.analysis_engine.analysis.market_regime -> analysis_engine.analysis.market_regime.analyzer -> analysis_engine.analysis.market_regime.detector -> analysis_engine.analysis.market_regime.models
2. analysis-engine-service.analysis_engine.analysis.market_regime -> analysis_engine.analysis.market_regime.analyzer -> analysis_engine.analysis.market_regime.classifier -> analysis_engine.analysis.market_regime.models
3. analysis-engine-service.analysis_engine.analysis.market_regime -> analysis_engine.analysis.market_regime.analyzer -> analysis_engine.analysis.market_regime.classifier
4. analysis-engine-service.analysis_engine.analysis.market_regime -> analysis_engine.analysis.market_regime.analyzer -> analysis_engine.analysis.market_regime.detector
5. analysis-engine-service.analysis_engine.analysis.market_regime -> analysis_engine.analysis.market_regime.detector -> analysis_engine.analysis.market_regime.models
6. analysis-engine-service.analysis_engine.analysis.market_regime -> analysis_engine.analysis.market_regime.classifier -> analysis_engine.analysis.market_regime.models
7. analysis-engine-service.analysis_engine.analysis.market_regime -> analysis_engine.analysis.market_regime.analyzer -> analysis_engine.analysis.market_regime.models
8. analysis-engine-service.analysis_engine.analysis.advanced_ta.elliott_wave -> analysis_engine.analysis.advanced_ta.elliott_wave.pattern -> analysis_engine.analysis.advanced_ta.elliott_wave.models
9. analysis-engine-service.analysis_engine.analysis.advanced_ta.elliott_wave.analyzer -> analysis_engine.analysis.advanced_ta.elliott_wave.pattern -> analysis_engine.analysis.advanced_ta.elliott_wave.models
10. analysis-engine-service.analysis_engine.analysis.advanced_ta.elliott_wave.counter -> analysis_engine.analysis.advanced_ta.elliott_wave.pattern -> analysis_engine.analysis.advanced_ta.elliott_wave.models
