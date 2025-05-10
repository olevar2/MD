# Forex Trading Platform Module Dependency Report

## Summary

- Total modules: 1443
- Total dependencies: 89
- Circular dependencies: 0

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

## Circular Dependencies

No circular dependencies found.

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
