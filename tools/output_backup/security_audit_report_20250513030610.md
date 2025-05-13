# Security Audit Report

Date: 2025-05-13 03:06:10

## Summary

- Total files scanned: 1554
- Total findings: 2711

### Findings by Severity

- Critical: 0
- High: 2317
- Medium: 394
- Low: 0

### Findings by Check

- Check for weak cryptographic algorithms: 2315
- Check for insecure HTTP usage: 380
- Check for insecure deserialization: 1
- Check for hardcoded secrets: 1
- Check for containers using latest tag: 1
- Check for containers without resource limits: 13

## Detailed Findings

### High Severity

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\main.py`
- Line: 26
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\update_datetime.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\error\exceptions_bridge.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\error\exceptions_bridge.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\error\exceptions_bridge.py`
- Line: 202
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\monitoring\performance_monitoring.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\adapters\test_analysis_engine_adapter.py`
- Line: 111
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\adapters\test_analysis_engine_adapter.py`
- Line: 118
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 29
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 30
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 61
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 73
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 85
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 88
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 111
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 114
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 131
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 132
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 135
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 152
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 164
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 176
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 188
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 225
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_connectivity_loss_handling.py`
- Line: 354
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_ctrader_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_interactive_brokers_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_interactive_brokers_adapter.py`
- Line: 188
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_metatrader_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\broker_adapters\test_oanda_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\execution_algorithms\test_execution_algorithms.py`
- Line: 32
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\execution_algorithms\test_execution_algorithms_integration.py`
- Line: 31
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\execution_algorithms\test_execution_algorithms_integration.py`
- Line: 170
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\execution_algorithms\test_execution_algorithms_integration.py`
- Line: 214
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\execution_algorithms\test_execution_algorithms_integration.py`
- Line: 215
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\execution_algorithms\test_execution_algorithms_integration.py`
- Line: 216
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\execution_algorithms\test_execution_algorithms_integration.py`
- Line: 246
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_market_data_service.py`
- Line: 13
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_market_data_service.py`
- Line: 18
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 44
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 134
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 135
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 136
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 181
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 182
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 183
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 228
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 229
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 230
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 275
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 276
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 277
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_order_execution_service.py`
- Line: 356
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\tests\services\test_refactored_execution_service.py`
- Line: 96
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 73
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 79
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 86
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 93
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 100
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 106
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 112
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 119
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 125
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 132
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 138
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 145
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 256
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 265
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 265
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 271
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 271
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\monitoring.py`
- Line: 324
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\config\standardized_config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\config\standardized_config.py`
- Line: 6
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\config\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway\config\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\database.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\database.py`
- Line: 28
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error_handling.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\logging_setup.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 121
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 133
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 134
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 135
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 148
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 149
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 150
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 163
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 176
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 177
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 190
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 191
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 192
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 193
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 194
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 207
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 208
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\service_clients.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\service_clients.py`
- Line: 24
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\adapter_factory.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\adapter_factory.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\analysis_engine_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\analysis_engine_adapter.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\order_book_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\risk_adapters.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\risk_management_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\risk_management_adapter.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\rl_environment_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\trading_provider_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\dependencies.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 17
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 33
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 34
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 35
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 37
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 39
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 126
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 164
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 208
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\api\v1\adapter_api.py`
- Line: 209
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\broker_adapters\metatrader_adapter.py`
- Line: 400
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\broker_adapters\metatrader_adapter.py`
- Line: 401
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\broker_adapters\metatrader_adapter.py`
- Line: 402
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 28
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 29
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 33
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 34
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 35
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 38
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 39
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 40
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 44
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 45
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 46
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 47
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 49
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 50
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 51
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 52
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 55
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 56
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 57
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 58
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 59
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 63
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 64
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 65
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 66
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 67
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 68
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 69
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 70
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 71
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 72
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 73
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 74
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 75
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 76
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 77
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 80
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 81
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 82
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 83
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 84
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 87
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 88
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 89
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 92
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 93
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\config.py`
- Line: 94
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\core\logging.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\core\logging.py`
- Line: 22
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\core\service_template.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\core\service_template.py`
- Line: 61
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\core\service_template.py`
- Line: 88
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\core\service_template.py`
- Line: 96
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\core\service_template.py`
- Line: 104
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\core\service_template.py`
- Line: 112
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 102
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 103
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 104
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 105
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 133
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 134
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 135
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 136
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 137
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 166
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 167
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 168
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 169
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 279
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 280
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 337
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exceptions_bridge.py`
- Line: 338
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\exception_handlers.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\error\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\execution_algorithms\base_algorithm.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\execution_algorithms\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 57
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 67
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 67
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 81
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 81
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 215
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 220
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 225
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 231
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 236
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 241
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 246
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 251
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 256
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 261
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 266
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\emergency_action_system.py`
- Line: 271
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 11
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 14
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 47
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 65
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 68
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 70
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 75
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 85
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 123
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 126
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 146
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 165
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 168
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 187
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 194
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 205
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 225
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 228
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 247
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 268
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 342
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\runbooks.py`
- Line: 343
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 85
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 91
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 94
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 94
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 179
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 367
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 520
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 534
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 546
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 559
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 571
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\incidents\trading_incident_manager.py`
- Line: 587
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\performance_monitoring.py`
- Line: 30
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 53
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 68
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 75
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 82
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 89
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 113
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 120
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 127
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 135
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 145
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 152
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 160
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 167
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 174
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 182
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 191
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 198
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 206
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 213
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 220
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\monitoring\service_metrics.py`
- Line: 227
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 10
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 114
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 132
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 166
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 172
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 177
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 183
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 187
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 192
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 197
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 202
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 464
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 465
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 493
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 494
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 497
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 508
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode.py`
- Line: 509
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode_strategies.py`
- Line: 15
- Match: `deS`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode_strategies.py`
- Line: 336
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode_strategies.py`
- Line: 341
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode_strategies.py`
- Line: 341
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode_strategies.py`
- Line: 342
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode_strategies.py`
- Line: 346
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode_strategies.py`
- Line: 346
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\degraded_mode_strategies.py`
- Line: 347
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 91
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 92
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 93
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 94
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 95
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 98
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 146
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 147
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 160
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\resilience\utils.py`
- Line: 161
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution_analytics.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution_analytics.py`
- Line: 52
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution_analytics_fixed.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution_analytics_fixed.py`
- Line: 52
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\market_data_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\market_regime_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_execution_service.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 635
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 647
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 648
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 649
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 661
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 662
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 663
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 676
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 687
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 688
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 701
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 702
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 703
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 704
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 705
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 717
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\order_reconciliation_service.py`
- Line: 718
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\algorithm_execution_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\algorithm_execution_service.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\algorithm_execution_service.py`
- Line: 52
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\base_execution_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\base_execution_service.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\conditional_execution_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\conditional_execution_service.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\conditional_execution_service.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\execution_mode_handler.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\execution_mode_handler.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\execution_mode_handler.py`
- Line: 21
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\execution_mode_handler.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\execution_mode_handler.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\limit_execution_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\limit_execution_service.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\limit_execution_service.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\market_execution_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\market_execution_service.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\market_execution_service.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\stop_execution_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\stop_execution_service.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\stop_execution_service.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\execution\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 78
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 96
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 96
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 110
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 110
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 283
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 283
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 534
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 540
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 545
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 550
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 556
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 563
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 570
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 579
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 586
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 594
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 639
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 643
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 648
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 652
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 657
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 662
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 669
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 675
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 681
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 688
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 689
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 696
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 707
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_market_regime_simulator.py`
- Line: 714
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 77
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 92
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 92
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 94
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 109
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 109
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 170
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 170
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 187
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 187
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 588
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 598
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 640
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 640
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 641
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 641
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 666
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 666
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 716
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 779
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 781
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 787
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 789
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 808
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 845
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 867
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 884
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 896
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 896
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 911
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\advanced_news_sentiment_simulator.py`
- Line: 911
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 73
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 74
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 75
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 76
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 77
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 78
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 79
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 80
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 81
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\broker_simulator.py`
- Line: 82
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 181
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 182
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 183
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 184
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 185
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 188
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 245
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 246
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 247
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 248
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 249
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 252
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 299
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 300
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 301
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 302
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 303
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 306
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 381
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 406
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 407
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 408
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 409
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 411
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 416
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 418
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\dynamic_risk_manager.py`
- Line: 423
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 40
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 40
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 43
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 185
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 186
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 414
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 414
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 642
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 648
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 649
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 651
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 652
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 669
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 669
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 671
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 673
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 678
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 678
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 681
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 681
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 685
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 685
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 687
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 689
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 691
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 699
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 699
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 701
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 702
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 702
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\enhanced_market_condition_generator.py`
- Line: 704
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 126
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 127
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 128
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 129
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 130
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 131
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 132
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 133
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 134
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 135
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 136
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 137
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 138
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 139
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 140
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 189
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 190
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 191
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 192
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 193
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 194
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 195
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 196
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 197
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 272
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 273
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 274
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 275
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 367
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 372
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 373
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 374
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 375
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 375
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 375
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 376
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 383
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 383
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 452
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 578
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 604
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 612
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 612
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 615
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 615
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 619
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 942
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 985
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\forex_broker_simulator.py`
- Line: 986
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\historical_news_data_collector.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\historical_news_data_collector.py`
- Line: 119
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\historical_news_data_collector.py`
- Line: 199
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\historical_news_data_collector.py`
- Line: 210
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\historical_news_data_collector.py`
- Line: 225
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\market_simulator.py`
- Line: 68
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\market_simulator.py`
- Line: 69
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\market_simulator.py`
- Line: 70
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\market_simulator.py`
- Line: 71
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\market_simulator.py`
- Line: 73
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\market_simulator.py`
- Line: 75
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 299
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 299
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 307
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 307
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 308
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 308
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 309
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 310
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 311
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 311
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 316
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 316
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 317
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 317
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 326
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 328
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 328
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 334
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 334
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 335
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_aware_strategy_demo.py`
- Line: 335
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 305
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 305
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 306
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 311
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 312
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 314
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 321
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 321
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 321
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 324
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 324
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 325
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 325
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 336
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 336
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 336
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 337
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 337
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 337
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 352
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 487
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 488
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 492
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 494
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 495
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 496
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 497
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 499
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 512
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 514
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 514
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 515
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 516
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 516
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 517
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 517
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 518
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 518
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 521
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 523
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 524
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 525
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 525
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 525
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 529
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 529
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 530
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 530
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 531
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 531
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 532
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 533
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 534
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 535
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 535
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 535
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 538
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 538
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 539
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 539
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 540
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 540
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 540
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 541
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 541
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_event_backtester.py`
- Line: 541
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 61
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 76
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 76
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 91
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 91
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 130
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 131
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 148
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 149
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\news_sentiment_simulator.py`
- Line: 346
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\paper_trading_system.py`
- Line: 58
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\paper_trading_system.py`
- Line: 59
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\paper_trading_system.py`
- Line: 60
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\paper_trading_system.py`
- Line: 61
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\paper_trading_system.py`
- Line: 62
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\paper_trading_system.py`
- Line: 63
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 56
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 74
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 136
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 143
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 150
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 160
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 167
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 177
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 184
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 194
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 204
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 214
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 221
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 231
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 236
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 244
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 244
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 255
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 256
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 265
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 275
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 275
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 284
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 285
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 288
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 288
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 347
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 356
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 356
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 378
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 380
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 418
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 418
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 424
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 424
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 578
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 586
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 586
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 596
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 596
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 610
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 610
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 733
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 733
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 735
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 735
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 773
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\agent_benchmarking.py`
- Line: 774
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 38
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 102
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 112
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 125
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 136
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 147
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 321
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 321
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 365
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\curriculum_learning_framework.py`
- Line: 365
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 86
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 113
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 164
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 320
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 339
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 580
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 642
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 661
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 661
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 662
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 663
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 666
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 667
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 672
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 673
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 673
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 675
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 683
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 686
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 688
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 688
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 688
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 689
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 689
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 691
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 691
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 695
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\environment_generator.py`
- Line: 695
- Match: `des`

#### Check for insecure deserialization

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 166
- Match: `eval(`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 246
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 252
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 252
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 260
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 284
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 445
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 452
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 452
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 462
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 472
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 476
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 481
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 489
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 489
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 491
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 506
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 515
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 538
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 540
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 546
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 552
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 552
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 560
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 560
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 573
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 573
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 584
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 592
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 592
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 607
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 608
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 769
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\training_module.py`
- Line: 776
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\simulation\reinforcement_learning\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\main.py`
- Line: 26
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\error\exceptions_bridge.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\error\exceptions_bridge.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\error\exceptions_bridge.py`
- Line: 202
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\monitoring\performance_monitoring.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management\config\standardized_config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management\config\standardized_config.py`
- Line: 6
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management\config\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management\config\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\main.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\adapters\multi_asset_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\middleware.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 27
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 71
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 72
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 103
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 104
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 113
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 113
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 119
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 136
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 137
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 138
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 138
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 138
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 146
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 146
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 153
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 170
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 171
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 172
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 197
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 198
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 200
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 201
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 202
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 234
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\accounts.py`
- Line: 234
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\multi_asset_portfolio_api.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\multi_asset_portfolio_api.py`
- Line: 46
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\multi_asset_portfolio_api.py`
- Line: 65
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\multi_asset_portfolio_api.py`
- Line: 85
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 53
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 63
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 65
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 75
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 88
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 89
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 106
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\portfolio_api.py`
- Line: 119
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 15
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 56
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 56
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 83
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 83
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 111
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 112
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 112
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 143
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 171
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\api\v1\positions.py`
- Line: 172
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\clients\risk_management_client.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\clients\risk_management_client.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\clients\risk_management_client.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\clients\risk_management_client.py`
- Line: 222
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\clients\risk_management_client.py`
- Line: 226
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\clients\risk_management_client.py`
- Line: 234
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\clients\risk_management_client.py`
- Line: 239
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\core\logging.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\core\logging.py`
- Line: 22
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\connection.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\connection.py`
- Line: 93
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\models.py`
- Line: 114
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\models.py`
- Line: 130
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\models.py`
- Line: 130
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\models.py`
- Line: 152
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\models.py`
- Line: 168
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\models.py`
- Line: 170
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\models.py`
- Line: 184
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\db\models.py`
- Line: 184
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\error_handlers.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 86
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 87
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 88
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 89
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 113
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 114
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 115
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 116
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 142
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 143
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 144
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 145
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 146
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 147
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 175
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 176
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 177
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 178
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 202
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 203
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 204
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 205
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 238
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 311
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 312
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 361
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\exceptions_bridge.py`
- Line: 362
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\error\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 14
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 15
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 16
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 17
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 18
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 27
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 28
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 29
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 33
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 42
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 44
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 45
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 46
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 47
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 49
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 50
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 51
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 57
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 58
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 64
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\account.py`
- Line: 65
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 14
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 15
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 16
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 17
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 18
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 19
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 20
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 21
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 22
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 33
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 34
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 35
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 36
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 37
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 38
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 38
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 38
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 39
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 49
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 50
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 51
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 52
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 58
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 59
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 61
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 62
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 63
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\historical.py`
- Line: 64
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 39
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 40
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 42
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 44
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 49
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 50
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 51
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 52
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 53
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 54
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 55
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 56
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 59
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 61
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 62
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 63
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 64
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 65
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 68
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 78
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 79
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 80
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 81
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 82
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 83
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 84
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 89
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 90
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 91
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 92
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 93
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 94
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 95
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 96
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 97
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 99
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\models\position.py`
- Line: 100
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\multi_asset\multi_asset_portfolio_manager.py`
- Line: 23
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\repositories\historical_repository.py`
- Line: 80
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\repositories\historical_repository.py`
- Line: 82
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\repositories\historical_repository.py`
- Line: 161
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\repositories\historical_repository.py`
- Line: 187
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\repositories\historical_repository.py`
- Line: 210
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_snapshot_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 56
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 67
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 67
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 75
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 102
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 104
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 124
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 249
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 258
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 258
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 288
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 289
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 289
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 291
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 297
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 315
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 315
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 325
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 432
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 458
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 466
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 474
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 476
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 489
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 496
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 511
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 516
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 527
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 665
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 666
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 666
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 667
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\export_service.py`
- Line: 668
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\historical_tracking.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\historical_tracking.py`
- Line: 90
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\historical_tracking.py`
- Line: 90
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\historical_tracking.py`
- Line: 164
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\historical_tracking.py`
- Line: 164
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\historical_tracking.py`
- Line: 170
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\performance_metrics.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\performance_metrics.py`
- Line: 77
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\performance_metrics.py`
- Line: 95
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\performance_metrics.py`
- Line: 152
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\performance_metrics.py`
- Line: 211
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\portfolio_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 114
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 114
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 117
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 123
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 125
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 132
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 336
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 340
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 343
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 343
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 347
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 351
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 353
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 355
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 360
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 364
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 366
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 367
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 386
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 397
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 397
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 408
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 408
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 411
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 412
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 416
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 417
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 418
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 419
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 419
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 421
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 429
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 538
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 545
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 545
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 602
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 609
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 609
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 615
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 616
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 618
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 619
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 620
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 622
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 622
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 623
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 624
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 625
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 637
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 637
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 638
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 639
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 639
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 640
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 643
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 649
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 649
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 655
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 657
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 698
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 705
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 705
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 714
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\tax_reporting_service.py`
- Line: 715
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\base.py`
- Line: 26
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\basic_reconciliation.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\discrepancy_handling.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\discrepancy_handling.py`
- Line: 134
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\discrepancy_handling.py`
- Line: 138
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\discrepancy_handling.py`
- Line: 149
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\facade.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\facade.py`
- Line: 39
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\full_reconciliation.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\historical_analysis.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\historical_analysis.py`
- Line: 218
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\historical_analysis.py`
- Line: 237
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\position_reconciliation.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\reporting.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\account_reconciliation\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\reconciliation\base.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\reconciliation\cash.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\reconciliation\positions.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\services\reconciliation\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\tax_reporting\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\tests\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\portfolio-management-service\tests\integration\test_performance_analytics.py`
- Line: 368
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\run_tests.py`
- Line: 82
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\database.py`
- Line: 122
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\database.py`
- Line: 147
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\db.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\db.py`
- Line: 73
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 33
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 34
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 35
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 36
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 68
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 69
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 70
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 71
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 88
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 89
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 90
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 108
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 109
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 110
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 111
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 130
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 131
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 132
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 133
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 150
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 151
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 152
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 153
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 154
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 173
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 174
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 175
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 176
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 177
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 197
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 198
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 199
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 200
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 219
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 220
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 221
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 222
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 241
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 242
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 243
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 244
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 263
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 264
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 265
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 266
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 267
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 286
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 287
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 288
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 289
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 290
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 310
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 311
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 312
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 329
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 330
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 331
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 332
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 361
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 362
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 363
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 364
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 365
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 386
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 387
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 388
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 389
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 406
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 407
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 408
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 409
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 427
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 428
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 429
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 430
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 448
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 449
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 450
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 451
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 469
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 470
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 471
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 472
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 473
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 492
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 493
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 494
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 495
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 511
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 512
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 513
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\exceptions.py`
- Line: 514
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\schemas.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\schemas.py`
- Line: 58
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\schemas.py`
- Line: 59
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\schemas.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\schemas.py`
- Line: 61
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\schemas.py`
- Line: 62
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\schemas.py`
- Line: 78
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\schemas.py`
- Line: 79
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security.py`
- Line: 35
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security.py`
- Line: 36
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security.py`
- Line: 37
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security.py`
- Line: 71
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security.py`
- Line: 72
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security.py`
- Line: 235
- Match: `deS`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\analysis_engine_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\factory.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\factory.py`
- Line: 46
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\feature_store_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\market_data_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\ml_integration_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\risk_management_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\trading_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\trading_adapter.py`
- Line: 356
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\trading_adapter.py`
- Line: 376
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\trading_adapter.py`
- Line: 380
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\trading_gateway_adapter.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adaptive\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\analysis\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\api\responses.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\api\versioning.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\api\versioning.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\api\versioning.py`
- Line: 108
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\api\versioning.py`
- Line: 111
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\api\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\adaptive_cache_manager.py`
- Line: 559
- Match: `md5`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\adaptive_cache_manager.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\adaptive_cache_manager.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\adaptive_cache_manager.py`
- Line: 518
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\adaptive_cache_manager.py`
- Line: 521
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\adaptive_cache_manager.py`
- Line: 541
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\adaptive_cache_manager.py`
- Line: 542
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\adaptive_cache_manager.py`
- Line: 545
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_key.py`
- Line: 43
- Match: `md5`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_key.py`
- Line: 89
- Match: `md5`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_key.py`
- Line: 126
- Match: `md5`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_key.py`
- Line: 147
- Match: `md5`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_key.py`
- Line: 171
- Match: `md5`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_key.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_key.py`
- Line: 16
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_monitoring.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_monitoring.py`
- Line: 25
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_monitoring.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_monitoring.py`
- Line: 37
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_monitoring.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_monitoring.py`
- Line: 49
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_monitoring.py`
- Line: 57
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_service.py`
- Line: 49
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_service.py`
- Line: 55
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_service.py`
- Line: 61
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_service.py`
- Line: 67
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_service.py`
- Line: 76
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_service.py`
- Line: 170
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\cache_service.py`
- Line: 171
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\dependency_invalidation.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\event_invalidation.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\invalidation.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\predictive_cache_manager.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\predictive_cache_manager.py`
- Line: 452
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\predictive_cache_manager.py`
- Line: 510
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\secure_serialization.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\secure_serialization.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\secure_serialization.py`
- Line: 20
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\secure_serialization.py`
- Line: 20
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\secure_serialization.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\secure_serialization.py`
- Line: 45
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\secure_serialization.py`
- Line: 51
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\secure_serialization.py`
- Line: 164
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\caching\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\base_client.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\base_client.py`
- Line: 99
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\base_client.py`
- Line: 101
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\base_client.py`
- Line: 365
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\client_factory.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\client_factory.py`
- Line: 90
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\client_factory.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\client_factory.py`
- Line: 102
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\client_factory.py`
- Line: 127
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\templates\service_client_template.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\templates\service_client_template.py`
- Line: 34
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\templates\service_client_template.py`
- Line: 35
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\templates\service_client_template.py`
- Line: 36
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_loader.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_loader.py`
- Line: 23
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_manager.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_manager.py`
- Line: 19
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 16
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 17
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 18
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 19
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 20
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 21
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 22
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 23
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 24
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 25
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 47
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 50
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 52
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 53
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 54
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 70
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 71
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 72
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 73
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 95
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 96
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 97
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 134
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 135
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 138
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 161
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 162
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 163
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 166
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 184
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 188
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 192
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 196
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 205
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 206
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 207
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 233
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 234
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 235
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 236
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 239
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\config_schema.py`
- Line: 243
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 23
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 28
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 36
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 40
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 44
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 52
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 56
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 79
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 84
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 88
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 92
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 96
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 100
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 104
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 108
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 112
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 116
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 120
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 132
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 137
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 141
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 145
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 149
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 153
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 157
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 161
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 165
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 184
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 189
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 193
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 197
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 201
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 205
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 209
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 213
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 217
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 221
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 233
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 238
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 242
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 246
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 250
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 254
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 258
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 262
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 266
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 270
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 282
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 287
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 291
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 295
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 299
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 303
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 307
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 311
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 315
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 319
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 323
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 327
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 339
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 344
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 348
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 352
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 356
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 360
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 364
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 368
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 372
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 376
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 395
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 400
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 404
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 408
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 412
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 416
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 420
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 424
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 428
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 432
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 444
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 449
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 453
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 457
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 461
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 465
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 469
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 473
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 477
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 481
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 485
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 489
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 493
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 505
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 510
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 514
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 518
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 522
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 526
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 530
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 534
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 538
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 542
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 546
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 550
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 16
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 17
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 18
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 19
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 20
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 21
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 22
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 23
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 24
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 25
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 27
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 33
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 34
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 35
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 36
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 37
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 39
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\settings.py`
- Line: 49
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 75
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 85
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 86
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 87
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 90
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 93
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 95
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 101
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 102
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 103
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 104
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 106
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 107
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 108
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 109
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 112
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 113
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 114
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 115
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 118
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 119
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 120
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 121
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 122
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 125
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 126
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 127
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 128
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 129
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 132
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 133
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 136
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 137
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 138
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 139
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 142
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\standardized_config.py`
- Line: 258
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\client_correlation.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\client_correlation.py`
- Line: 74
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\client_correlation.py`
- Line: 75
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\client_correlation.py`
- Line: 111
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\client_correlation.py`
- Line: 112
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\client_correlation.py`
- Line: 133
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\correlation_id.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\correlation_id.py`
- Line: 168
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\correlation_id.py`
- Line: 169
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\correlation_id.py`
- Line: 205
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\correlation_id.py`
- Line: 206
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\event_correlation.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\event_correlation.py`
- Line: 99
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\event_correlation.py`
- Line: 100
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\event_correlation.py`
- Line: 101
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\event_correlation.py`
- Line: 138
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\event_correlation.py`
- Line: 139
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\event_correlation.py`
- Line: 140
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\middleware.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\correlation\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\database\connection_manager.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\database\connection_manager.py`
- Line: 26
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\database\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\base.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\batch.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\exceptions.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\exceptions.py`
- Line: 86
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\exceptions.py`
- Line: 123
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\exceptions.py`
- Line: 160
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\realtime.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\reporting.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\strategies.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\data_reconciliation\__init__.py`
- Line: 7
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 25
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 26
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 37
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 62
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 91
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 91
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 94
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 95
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 96
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 102
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 123
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 123
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 126
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 127
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 128
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 134
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 137
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 142
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 142
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 144
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 147
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 147
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 152
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 153
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 155
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 157
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 158
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 160
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 162
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 163
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 167
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 167
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 169
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 172
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 172
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 177
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 178
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 180
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 182
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 184
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 185
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 188
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 189
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 191
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 193
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 194
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 284
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 284
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 304
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 304
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 326
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 326
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 342
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 342
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 350
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 350
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 352
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 358
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 363
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 366
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\container.py`
- Line: 366
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 33
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 36
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 46
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 47
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 50
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 102
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 105
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 115
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 116
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\decorators.py`
- Line: 119
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\dependency_injection\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\effectiveness\enhanced_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\effectiveness\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\effectiveness\interfaces.py`
- Line: 74
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\effectiveness\interfaces.py`
- Line: 75
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\effectiveness\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\error\error_bridge.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\error\exceptions.py`
- Line: 9
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\api.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\api.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\api.py`
- Line: 35
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\api.py`
- Line: 165
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\api.py`
- Line: 165
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\base.py`
- Line: 13
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\base_exceptions.py`
- Line: 16
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 61
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 64
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 74
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 75
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 78
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 195
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 198
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 208
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 209
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\decorators.py`
- Line: 212
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 117
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 117
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 195
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 198
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 208
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 209
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 212
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 246
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 249
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 259
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 260
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\error_handler.py`
- Line: 263
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 92
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 95
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 105
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 106
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 109
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 236
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 239
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 249
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 250
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 253
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\exceptions_bridge.py`
- Line: 343
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\handler.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\middleware.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\middleware.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\middleware.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\middleware.py`
- Line: 145
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\middleware.py`
- Line: 145
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\__init__.py`
- Line: 6
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\errors\__init__.py`
- Line: 36
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\event_bus.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\event_bus.py`
- Line: 24
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\event_bus.py`
- Line: 169
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\kafka_event_bus.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\message_broker.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\message_broker.py`
- Line: 82
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\message_broker.py`
- Line: 243
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\message_broker.py`
- Line: 306
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\events\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\base_indicator.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\fibonacci_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\indicator_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\moving_averages.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\moving_averages.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\oscillators.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\oscillators.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\volatility.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\volatility.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\moving_averages\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\oscillators\macd.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\oscillators\rsi.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\oscillators\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\volatility\bollinger_bands.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\indicators\volatility\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\interfaces\analysis_engine.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\interfaces\feature_store.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\interfaces\market_data.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\interfaces\risk_management.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\interfaces\trading.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\metrics\reconciliation_metrics.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\metrics\reconciliation_metrics.py`
- Line: 63
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\metrics\reconciliation_metrics.py`
- Line: 73
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\metrics\reconciliation_metrics.py`
- Line: 74
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\feature_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\model_feedback_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\model_feedback_interfaces.py`
- Line: 56
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\prediction_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\prediction_interfaces.py`
- Line: 33
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\rl_effectiveness_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\rl_effectiveness_interfaces.py`
- Line: 151
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\rl_effectiveness_interfaces.py`
- Line: 160
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\rl_effectiveness_interfaces.py`
- Line: 160
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\trading_feedback_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\ml\workbench_interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 60
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 68
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 68
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 84
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 84
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 118
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 240
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 356
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\alerting.py`
- Line: 356
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 34
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 43
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 48
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 101
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 102
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 102
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 114
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 157
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 166
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 166
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 175
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\health.py`
- Line: 175
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\logging.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\logging.py`
- Line: 124
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\logging.py`
- Line: 262
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\logging.py`
- Line: 272
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\logging.py`
- Line: 273
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\logging_config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 91
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 97
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 105
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 111
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 119
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 125
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 133
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 140
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 146
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 153
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 162
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 162
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 177
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 190
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 199
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 199
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 214
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 227
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 237
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 237
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 254
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 268
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 277
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 277
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 292
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 475
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 485
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 485
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 497
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 507
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 508
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 519
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 525
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 531
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics.py`
- Line: 538
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 73
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 100
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 107
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 115
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 123
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 131
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 140
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 147
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 155
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 162
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 169
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 175
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 184
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 191
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 198
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 205
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 212
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 219
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 226
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 236
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 243
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 251
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 258
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 267
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 274
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 281
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 287
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 293
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\metrics_standards.py`
- Line: 300
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\middleware.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 63
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 64
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 369
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 372
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 386
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 387
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 390
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\performance_monitoring.py`
- Line: 507
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\tracing.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\tracing.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\tracing.py`
- Line: 223
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\tracing.py`
- Line: 240
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\tracing.py`
- Line: 241
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\tracing.py`
- Line: 284
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\tracing.py`
- Line: 285
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\monitoring\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\multi_asset\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\parallel_processor.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\parallel_processor.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\parallel_processor.py`
- Line: 632
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\parallel_processor.py`
- Line: 633
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 56
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 151
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 157
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 157
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 179
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 186
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 413
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\specialized_processors.py`
- Line: 414
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\parallel\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\reinforcement\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\bulkhead.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\circuit_breaker.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\circuit_breaker.py`
- Line: 36
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\circuit_breaker.py`
- Line: 37
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\circuit_breaker.py`
- Line: 38
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\circuit_breaker.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 39
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 40
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 41
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 69
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 70
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 71
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 72
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 92
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 103
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 104
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 105
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 106
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 109
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 136
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 159
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 160
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 161
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 190
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 191
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 192
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 193
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 194
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 195
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 196
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 197
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 198
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 199
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 200
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 201
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 240
- Match: `deS`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 248
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 254
- Match: `deS`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 271
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 296
- Match: `deS`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\core_fallback.py`
- Line: 304
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 54
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 57
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 93
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 94
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 97
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 119
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 120
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 123
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 185
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 188
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 207
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 208
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 211
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 233
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 234
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 237
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 291
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 294
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 312
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 313
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 316
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 338
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 339
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 342
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 398
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 401
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 423
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 424
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 427
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 449
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 450
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 453
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 542
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\decorators.py`
- Line: 545
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\fallback.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\fallback.py`
- Line: 50
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\fallback.py`
- Line: 117
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\fallback.py`
- Line: 118
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 226
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 227
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 230
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 340
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 341
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 344
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 354
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 355
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\resilience.py`
- Line: 358
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry.py`
- Line: 41
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry.py`
- Line: 51
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry.py`
- Line: 52
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry.py`
- Line: 91
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry.py`
- Line: 186
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry.py`
- Line: 187
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 41
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 89
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 170
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 171
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 172
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 173
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 174
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 177
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 188
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 189
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 192
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 205
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 206
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\retry_policy.py`
- Line: 209
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout.py`
- Line: 67
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout.py`
- Line: 125
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout.py`
- Line: 126
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 31
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 32
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 134
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 137
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 148
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 149
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 152
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 168
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 169
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\timeout_handler.py`
- Line: 172
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\resilience\__init__.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\adapters.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\client.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\client.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\models.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\models.py`
- Line: 21
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\models.py`
- Line: 32
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\cookie_manager.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\cookie_manager.py`
- Line: 17
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\cookie_manager.py`
- Line: 58
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\cookie_manager.py`
- Line: 59
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\cookie_manager.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\file_upload.py`
- Line: 184
- Match: `md5`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\file_upload.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\file_upload.py`
- Line: 23
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 32
- Match: `SHA1`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 84
- Match: `SHA1`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 36
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 37
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 38
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 45
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 45
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 72
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 74
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 85
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 86
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 98
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 99
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 99
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 107
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 107
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 108
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 108
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 189
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 189
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 191
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 194
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 196
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 198
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 200
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 207
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 209
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 209
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 209
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 211
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 211
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 216
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 216
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 226
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 226
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 229
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 231
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 272
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 307
- Match: `DES`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\mfa.py`
- Line: 307
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\middleware.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\middleware.py`
- Line: 534
- Match: `deS`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\monitoring.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\monitoring.py`
- Line: 68
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\monitoring.py`
- Line: 79
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\monitoring.py`
- Line: 333
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\monitoring.py`
- Line: 333
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\rbac.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\rbac.py`
- Line: 60
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\rbac.py`
- Line: 75
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\rbac.py`
- Line: 96
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\rbac.py`
- Line: 104
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\rbac.py`
- Line: 118
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\rbac.py`
- Line: 134
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\__init__.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\security\__init__.py`
- Line: 92
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\base_client.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\base_client.py`
- Line: 44
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\base_client.py`
- Line: 45
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\base_client.py`
- Line: 104
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\base_client.py`
- Line: 171
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\base_client.py`
- Line: 249
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\base_client.py`
- Line: 316
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 35
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 57
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 58
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 170
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 257
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 279
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 280
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\http_client.py`
- Line: 409
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\resilient_client.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\resilient_client.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\resilient_client.py`
- Line: 82
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\service_client\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 222
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 228
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 229
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 230
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 231
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 232
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 233
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 234
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 235
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 236
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 236
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 236
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 248
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 248
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 261
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 261
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 276
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\simulation\interfaces.py`
- Line: 276
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\strategy\interfaces.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 29
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 34
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 35
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 38
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 39
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 40
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 44
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\config.py`
- Line: 45
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\database.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\database.py`
- Line: 22
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\error_handling.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\logging_setup.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 26
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 46
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 47
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 59
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 60
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 71
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 72
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 83
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 84
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 95
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 96
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\service_clients.py`
- Line: 115
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\standardized_config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\standardized_config.py`
- Line: 25
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\standardized_config.py`
- Line: 28
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\standardized_config.py`
- Line: 29
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\standardized_config.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\standardized_config.py`
- Line: 35
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\standardized_config.py`
- Line: 39
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\enhanced_memory_optimized_dataframe.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\enhanced_memory_optimized_dataframe.py`
- Line: 30
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\enhanced_memory_optimized_dataframe.py`
- Line: 244
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\enhanced_memory_optimized_dataframe.py`
- Line: 245
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\enhanced_memory_optimized_dataframe.py`
- Line: 414
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\export_service.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\export_service.py`
- Line: 129
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\memory_optimized_dataframe.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\memory_optimized_dataframe.py`
- Line: 26
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\parallel_processor.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\parallel_processor.py`
- Line: 148
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\parallel_processor.py`
- Line: 336
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\parallel_processor.py`
- Line: 398
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\platform_compatibility.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\utils\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\events\event_bus.py`
- Line: 51
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\events\event_bus.py`
- Line: 52
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\events\event_bus.py`
- Line: 65
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\events\event_bus.py`
- Line: 66
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\events\event_bus.py`
- Line: 67
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\events\event_bus.py`
- Line: 83
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\events\event_bus.py`
- Line: 84
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\examples\config_example.py`
- Line: 42
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\examples\config_example.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\examples\config_example.py`
- Line: 44
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\examples\config_example.py`
- Line: 47
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\examples\correlation_id_example.py`
- Line: 43
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\examples\correlation_id_example.py`
- Line: 55
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\config.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\config.py`
- Line: 28
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\config.py`
- Line: 29
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\database.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\database.py`
- Line: 22
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\error_handling.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\logging_setup.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 73
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 79
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 86
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 93
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 100
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 106
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 112
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 119
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 125
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 132
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 138
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 145
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 256
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 265
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 265
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 271
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 271
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring.py`
- Line: 324
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\service_clients.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\service_clients.py`
- Line: 21
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\__init__.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\error\exceptions_bridge.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\error\exceptions_bridge.py`
- Line: 5
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\error\exceptions_bridge.py`
- Line: 202
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\monitoring\performance_monitoring.py`
- Line: 4
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\test_resilience.py`
- Line: 250
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\correlation\test_client_correlation.py`
- Line: 134
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\correlation\test_client_correlation.py`
- Line: 145
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\data_reconciliation\test_strategies.py`
- Line: 188
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\data_reconciliation\test_strategies.py`
- Line: 189
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\data_reconciliation\test_strategies.py`
- Line: 190
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\data_reconciliation\test_strategies.py`
- Line: 269
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\data_reconciliation\test_strategies.py`
- Line: 270
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\data_reconciliation\test_strategies.py`
- Line: 271
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\indicators\test_base_indicator.py`
- Line: 116
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\integration\test_correlation_id_propagation.py`
- Line: 33
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\integration\test_correlation_id_propagation.py`
- Line: 45
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\integration\test_data_reconciliation.py`
- Line: 77
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\integration\test_data_reconciliation.py`
- Line: 88
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\integration\test_data_reconciliation.py`
- Line: 157
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\integration\test_data_reconciliation.py`
- Line: 168
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\integration\test_event_correlation.py`
- Line: 31
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\integration\test_event_correlation.py`
- Line: 43
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\resilience\test_decorators.py`
- Line: 60
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\resilience\test_decorators.py`
- Line: 180
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\resilience\test_decorators.py`
- Line: 262
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\service_client\test_service_client.py`
- Line: 36
- Match: `des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\tests\service_client\test_service_client.py`
- Line: 43
- Match: `des`

#### Check for hardcoded secrets

- File: `d:\MD\forex_trading_platform\common-lib\tests\templates\service_template\test_database.py`
- Line: 32
- Match: `password="postgres"`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 54
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 57
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 69
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 72
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 84
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 87
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 107
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 110
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 122
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 125
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\trading\interfaces.py`
- Line: 137
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\usage_demos\bulkhead_test.py`
- Line: 37
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\usage_demos\bulkhead_test.py`
- Line: 38
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\usage_demos\resilience_examples.py`
- Line: 96
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\usage_demos\simple_resilience_test.py`
- Line: 230
- Match: `Des`

#### Check for weak cryptographic algorithms

- File: `d:\MD\forex_trading_platform\common-lib\usage_demos\simple_resilience_test.py`
- Line: 231
- Match: `Des`

### Medium Severity

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\start.py`
- Line: 45
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\@babel\core\node_modules\debug\package.json`
- Line: 22
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\@babel\parser\CHANGELOG.md`
- Line: 418
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\@babel\traverse\node_modules\debug\package.json`
- Line: 22
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\accepts\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\accepts\README.md`
- Line: 125
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ansi-escapes\readme.md`
- Line: 3
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\argparse\CHANGELOG.md`
- Line: 113
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\argparse\README.md`
- Line: 4
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\argparse\README.md`
- Line: 8
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\argparse\README.md`
- Line: 14
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\argparse\README.md`
- Line: 112
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\argparse\README.md`
- Line: 142
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\argparse\README.md`
- Line: 175
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\argparse\README.md`
- Line: 239
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\array-flatten\package.json`
- Line: 26
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\babel-plugin-istanbul\README.md`
- Line: 5
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\babel-plugin-istanbul\README.md`
- Line: 5
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\babel-plugin-istanbul\README.md`
- Line: 7
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\balanced-match\package.json`
- Line: 29
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\balanced-match\README.md`
- Line: 5
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\basic-auth\node_modules\safe-buffer\package.json`
- Line: 8
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\basic-auth\node_modules\safe-buffer\README.md`
- Line: 406
- Match: `http://h`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\basic-auth\node_modules\safe-buffer\README.md`
- Line: 415
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\basic-auth\node_modules\safe-buffer\README.md`
- Line: 573
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\basic-auth\node_modules\safe-buffer\README.md`
- Line: 584
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\body-parser\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\brace-expansion\package.json`
- Line: 28
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\brace-expansion\README.md`
- Line: 6
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\braces\package.json`
- Line: 11
- Match: `http://h`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\braces\package.json`
- Line: 12
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\braces\README.md`
- Line: 427
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\bser\package.json`
- Line: 24
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\bytes\package.json`
- Line: 5
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\caniuse-lite\package.json`
- Line: 16
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\chokidar\README.md`
- Line: 31
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ci-info\README.md`
- Line: 38
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ci-info\README.md`
- Line: 46
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ci-info\README.md`
- Line: 60
- Match: `http://h`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ci-info\README.md`
- Line: 65
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ci-info\README.md`
- Line: 75
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ci-info\README.md`
- Line: 77
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\co\Readme.md`
- Line: 54
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\co\Readme.md`
- Line: 209
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\color-name\README.md`
- Line: 1
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\ecdsa-sig-formatter\README.md`
- Line: 64
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\ecdsa-sig-formatter\README.md`
- Line: 65
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\jsonwebtoken\README.md`
- Line: 5
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\jwa\README.md`
- Line: 4
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\jwa\README.md`
- Line: 74
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\jws\readme.md`
- Line: 1
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.includes\package.json`
- Line: 9
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.includes\package.json`
- Line: 11
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isboolean\package.json`
- Line: 9
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isboolean\package.json`
- Line: 11
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isinteger\package.json`
- Line: 9
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isinteger\package.json`
- Line: 11
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isnumber\package.json`
- Line: 9
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isnumber\package.json`
- Line: 11
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isplainobject\package.json`
- Line: 9
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isplainobject\package.json`
- Line: 11
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isstring\package.json`
- Line: 9
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.isstring\package.json`
- Line: 11
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.once\package.json`
- Line: 9
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\lodash.once\package.json`
- Line: 11
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\safe-buffer\README.md`
- Line: 406
- Match: `http://h`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\safe-buffer\README.md`
- Line: 415
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\safe-buffer\README.md`
- Line: 573
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\common-js-lib\node_modules\safe-buffer\README.md`
- Line: 584
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\content-disposition\README.md`
- Line: 127
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\convert-source-map\package.json`
- Line: 29
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\CONTRIBUTING.md`
- Line: 3
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\CONTRIBUTING.md`
- Line: 3
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\CONTRIBUTING.md`
- Line: 3
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\CONTRIBUTING.md`
- Line: 7
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\CONTRIBUTING.md`
- Line: 7
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\CONTRIBUTING.md`
- Line: 24
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\CONTRIBUTING.md`
- Line: 24
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\CONTRIBUTING.md`
- Line: 33
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 8
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 8
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 8
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 78
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 98
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 98
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 170
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 170
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 193
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 194
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 194
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 196
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 196
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 217
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 221
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 221
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\cors\README.md`
- Line: 230
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\debug\package.json`
- Line: 16
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\debug\README.md`
- Line: 56
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\debug\README.md`
- Line: 58
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\debug\README.md`
- Line: 80
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\debug\README.md`
- Line: 84
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\dedent\package.json`
- Line: 23
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\dedent\README.md`
- Line: 163
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\deepmerge\readme.md`
- Line: 57
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\deepmerge\readme.md`
- Line: 255
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\depd\Readme.md`
- Line: 23
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\destroy\package.json`
- Line: 8
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\destroy\README.md`
- Line: 54
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\destroy\README.md`
- Line: 58
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\destroy\README.md`
- Line: 60
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\diff-sequences\README.md`
- Line: 18
- Match: `http://x`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ee-first\package.json`
- Line: 8
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ee-first\README.md`
- Line: 69
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ee-first\README.md`
- Line: 75
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ee-first\README.md`
- Line: 77
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\error-ex\README.md`
- Line: 143
- Match: `http://o`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\escalade\readme.md`
- Line: 14
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\esprima\package.json`
- Line: 4
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\esprima\package.json`
- Line: 26
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\esprima\README.md`
- Line: 6
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\esprima\README.md`
- Line: 7
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\esprima\README.md`
- Line: 15
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\esprima\README.md`
- Line: 19
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\esprima\README.md`
- Line: 46
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\exit\package.json`
- Line: 8
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\exit\README.md`
- Line: 1
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\exit\README.md`
- Line: 66
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\History.md`
- Line: 3280
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\History.md`
- Line: 3374
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\History.md`
- Line: 3465
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\History.md`
- Line: 3573
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\package.json`
- Line: 17
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\Readme.md`
- Line: 1
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\Readme.md`
- Line: 3
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\Readme.md`
- Line: 57
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\Readme.md`
- Line: 72
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\express\Readme.md`
- Line: 109
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\fast-json-stable-stringify\benchmark\test.json`
- Line: 8
- Match: `http://p`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\fast-json-stable-stringify\benchmark\test.json`
- Line: 53
- Match: `http://p`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\fast-json-stable-stringify\benchmark\test.json`
- Line: 98
- Match: `http://p`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\fb-watchman\package.json`
- Line: 23
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\fill-range\package.json`
- Line: 9
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\fresh\package.json`
- Line: 5
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\fresh\package.json`
- Line: 8
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\fs.realpath\package.json`
- Line: 20
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\glob\package.json`
- Line: 2
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\glob\README.md`
- Line: 354
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\globals\readme.md`
- Line: 9
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\has-symbols\package.json`
- Line: 41
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\has-symbols\package.json`
- Line: 47
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\http-errors\package.json`
- Line: 5
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\human-signals\README.md`
- Line: 158
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\iconv-lite\README.md`
- Line: 3
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\iconv-lite\README.md`
- Line: 5
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\iconv-lite\README.md`
- Line: 5
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\iconv-lite\README.md`
- Line: 5
- Match: `http://y`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\iconv-lite\README.md`
- Line: 86
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\iconv-lite\README.md`
- Line: 107
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\iconv-lite\README.md`
- Line: 107
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\inflight\package.json`
- Line: 23
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\inherits\README.md`
- Line: 2
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ipaddr.js\README.md`
- Line: 12
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\is-arrayish\package.json`
- Line: 5
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\is-arrayish\README.md`
- Line: 15
- Match: `http://o`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\is-extglob\README.md`
- Line: 98
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\is-glob\package.json`
- Line: 10
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\is-number\package.json`
- Line: 8
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\isexe\package.json`
- Line: 20
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\istanbul-lib-source-maps\node_modules\debug\package.json`
- Line: 22
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\jest-each\README.md`
- Line: 8
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\jest-validate\README.md`
- Line: 101
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\jest-validate\README.md`
- Line: 121
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\jest-validate\README.md`
- Line: 141
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\jest-validate\README.md`
- Line: 176
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\jest-validate\README.md`
- Line: 195
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-tokens\CHANGELOG.md`
- Line: 67
- Match: `http://v`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-tokens\README.md`
- Line: 121
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-tokens\README.md`
- Line: 232
- Match: `http://2`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-tokens\README.md`
- Line: 233
- Match: `http://2`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-tokens\README.md`
- Line: 234
- Match: `http://2`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\package.json`
- Line: 14
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\package.json`
- Line: 16
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 7
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 10
- Match: `http://y`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 11
- Match: `http://p`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 110
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 112
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 114
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 117
- Match: `http://y`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 236
- Match: `http://p`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\js-yaml\README.md`
- Line: 237
- Match: `http://y`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\jsesc\README.md`
- Line: 327
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\jsesc\README.md`
- Line: 403
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\json5\package.json`
- Line: 52
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\json5\README.md`
- Line: 74
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\json5\README.md`
- Line: 234
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\json5\README.md`
- Line: 259
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\json5\README.md`
- Line: 261
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\json5\README.md`
- Line: 263
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\json5\README.md`
- Line: 264
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\makeerror\readme.md`
- Line: 1
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\media-typer\README.md`
- Line: 75
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\merge-descriptors\package.json`
- Line: 8
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\merge-stream\README.md`
- Line: 5
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\methods\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\methods\package.json`
- Line: 8
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\micromatch\package.json`
- Line: 12
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\micromatch\package.json`
- Line: 15
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\micromatch\package.json`
- Line: 20
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\micromatch\README.md`
- Line: 787
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime\package.json`
- Line: 4
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime\package.json`
- Line: 16
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime\README.md`
- Line: 7
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime-db\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime-db\package.json`
- Line: 8
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime-db\README.md`
- Line: 14
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime-db\README.md`
- Line: 16
- Match: `http://h`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime-db\README.md`
- Line: 52
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime-db\README.md`
- Line: 53
- Match: `http://h`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime-db\README.md`
- Line: 83
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\mime-types\package.json`
- Line: 8
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\minimatch\package.json`
- Line: 2
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\minimatch\README.md`
- Line: 5
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\morgan\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\morgan\README.md`
- Line: 10
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\morgan\node_modules\on-finished\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\morgan\node_modules\on-finished\README.md`
- Line: 148
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\ms\readme.md`
- Line: 4
- Match: `http://z`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\natural-compare\README.md`
- Line: 2
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\natural-compare\README.md`
- Line: 3
- Match: `http://i`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\natural-compare\README.md`
- Line: 116
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\natural-compare\README.md`
- Line: 123
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\negotiator\package.json`
- Line: 8
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\node-int64\package.json`
- Line: 4
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\nodemon\README.md`
- Line: 16
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\nodemon\README.md`
- Line: 109
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\nodemon\README.md`
- Line: 302
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\nodemon\README.md`
- Line: 389
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\nodemon\README.md`
- Line: 434
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\nodemon\README.md`
- Line: 434
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\nodemon\node_modules\debug\package.json`
- Line: 23
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\normalize-path\package.json`
- Line: 9
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\object-assign\readme.md`
- Line: 3
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\on-finished\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\once\package.json`
- Line: 31
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\p-try\readme.md`
- Line: 5
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\parseurl\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\parseurl\README.md`
- Line: 67
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\path-is-absolute\readme.md`
- Line: 3
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\path-is-absolute\readme.md`
- Line: 44
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\path-parse\package.json`
- Line: 27
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\path-parse\README.md`
- Line: 42
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\path-to-regexp\Readme.md`
- Line: 31
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\picomatch\CHANGELOG.md`
- Line: 5
- Match: `http://k`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\picomatch\CHANGELOG.md`
- Line: 24
- Match: `http://k`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\picomatch\README.md`
- Line: 33
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\pirates\package.json`
- Line: 23
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\prompts\readme.md`
- Line: 31
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\prompts\readme.md`
- Line: 31
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\qs\package.json`
- Line: 19
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\range-parser\package.json`
- Line: 3
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\range-parser\package.json`
- Line: 9
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\raw-body\package.json`
- Line: 5
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\require-directory\package.json`
- Line: 2
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\require-directory\package.json`
- Line: 22
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\require-directory\package.json`
- Line: 27
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\safe-buffer\README.md`
- Line: 406
- Match: `http://h`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\safe-buffer\README.md`
- Line: 415
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\safe-buffer\README.md`
- Line: 573
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\safe-buffer\README.md`
- Line: 584
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map\package.json`
- Line: 47
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map\README.md`
- Line: 75
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map\README.md`
- Line: 82
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map\README.md`
- Line: 83
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map\README.md`
- Line: 89
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map\README.md`
- Line: 95
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map\README.md`
- Line: 456
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map\README.md`
- Line: 535
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map-support\README.md`
- Line: 60
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map-support\README.md`
- Line: 69
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map-support\README.md`
- Line: 277
- Match: `http://1`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map-support\README.md`
- Line: 278
- Match: `http://1`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map-support\README.md`
- Line: 279
- Match: `http://1`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map-support\README.md`
- Line: 280
- Match: `http://1`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\source-map-support\README.md`
- Line: 284
- Match: `http://o`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\sprintf-js\bower.json`
- Line: 8
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\sprintf-js\package.json`
- Line: 5
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\stack-utils\readme.md`
- Line: 143
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\stack-utils\readme.md`
- Line: 143
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\statuses\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\string-length\readme.md`
- Line: 5
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\tmpl\readme.md`
- Line: 1
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\to-regex-range\package.json`
- Line: 8
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\to-regex-range\README.md`
- Line: 58
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\to-regex-range\README.md`
- Line: 58
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\toidentifier\package.json`
- Line: 8
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\touch\package.json`
- Line: 2
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\type-detect\package.json`
- Line: 1
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\type-detect\README.md`
- Line: 2
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\type-detect\README.md`
- Line: 3
- Match: `http://c`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\type-detect\README.md`
- Line: 8
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\type-detect\README.md`
- Line: 87
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\type-detect\README.md`
- Line: 87
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\type-detect\README.md`
- Line: 101
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\type-is\package.json`
- Line: 7
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\unpipe\README.md`
- Line: 37
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\utils-merge\package.json`
- Line: 11
- Match: `http://w`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\utils-merge\package.json`
- Line: 18
- Match: `http://g`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\utils-merge\package.json`
- Line: 24
- Match: `http://o`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\utils-merge\README.md`
- Line: 30
- Match: `http://o`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\utils-merge\README.md`
- Line: 32
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\utils-merge\README.md`
- Line: 32
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\walker\readme.md`
- Line: 1
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\which\package.json`
- Line: 2
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\wrappy\package.json`
- Line: 23
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\wrappy\README.md`
- Line: 13
- Match: `http://n`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\yallist\package.json`
- Line: 27
- Match: `http://b`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\yargs\README.md`
- Line: 166
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\yargs\README.md`
- Line: 200
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\yargs\README.md`
- Line: 201
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\yargs-parser\package.json`
- Line: 26
- Match: `http://1`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\node_modules\yargs-parser\README.md`
- Line: 10
- Match: `http://y`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\main.py`
- Line: 36
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\analysis_engine_adapter.py`
- Line: 47
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\risk_management_adapter.py`
- Line: 44
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\adapters\risk_manager_adapter.py`
- Line: 42
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\default\config.yaml`
- Line: 37
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\default\config.yaml`
- Line: 46
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\default\config.yaml`
- Line: 55
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\config\default\config.yaml`
- Line: 64
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\trading-gateway-service\trading_gateway_service\services\market_regime_service.py`
- Line: 52
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\portfolio-management-service\README.md`
- Line: 41
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\portfolio-management-service\README.md`
- Line: 42
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\portfolio-management-service\start.py`
- Line: 45
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\adapters\multi_asset_adapter.py`
- Line: 48
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\portfolio-management-service\portfolio_management_service\clients\risk_management_client.py`
- Line: 43
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\factory.py`
- Line: 90
- Match: `http://{`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\ml_integration_adapter.py`
- Line: 42
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\ml_integration_adapter.py`
- Line: 191
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\ml_integration_adapter.py`
- Line: 347
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\ml_integration_adapter.py`
- Line: 480
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\risk_management_adapter.py`
- Line: 39
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\adapters\trading_gateway_adapter.py`
- Line: 42
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\base_client.py`
- Line: 206
- Match: `http://'`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\client_factory.py`
- Line: 61
- Match: `http://{`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 21
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 32
- Match: `http://d`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 43
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 54
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 65
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 76
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 87
- Match: `http://p`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 98
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\config.py`
- Line: 120
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\clients\README.md`
- Line: 89
- Match: `http://o`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 253
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 286
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\config\service_settings.py`
- Line: 290
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\risk\client.py`
- Line: 50
- Match: `http://r`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\common_lib\templates\service_template\standardized_config.py`
- Line: 34
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\docs\service_communication.md`
- Line: 152
- Match: `http://o`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\docs\standardized_service_communication.md`
- Line: 129
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\docs\standardized_service_communication.md`
- Line: 137
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\examples\config_example.py`
- Line: 101
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\examples\config_example.py`
- Line: 116
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\examples\config_example.py`
- Line: 120
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\examples\config_example.py`
- Line: 124
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\examples\resilience_example.py`
- Line: 39
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\examples\resilient_client_example.py`
- Line: 31
- Match: `http://e`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\README.md`
- Line: 52
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\config\default_config.yaml`
- Line: 28
- Match: `http://m`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\config\default_config.yaml`
- Line: 42
- Match: `http://f`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\config\default_config.yaml`
- Line: 56
- Match: `http://a`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\templates\service_template\config\default_config.yaml`
- Line: 70
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\tests\service_client\test_service_client.py`
- Line: 68
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\tests\service_client\test_service_client.py`
- Line: 75
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\tests\service_client\test_service_client.py`
- Line: 92
- Match: `http://t`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\common-lib\tests\templates\service_template\test_service_clients.py`
- Line: 39
- Match: `http://l`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\kubernetes\analysis-engine-deployment.yaml`
- Line: 51
- Match: `http://j`

#### Check for insecure HTTP usage

- File: `d:\MD\forex_trading_platform\kubernetes\analysis-engine-deployment.yaml`
- Line: 109
- Match: `http://j`

#### Check for containers using latest tag

- File: `d:\MD\forex_trading_platform\kubernetes\analysis-engine-deployment.yaml`
- Line: 28
- Match: `image: olevar2/forex-platform:latest`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\analysis-engine-deployment.yaml`
- Line: 26
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\analysis-engine-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\data-management-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\data-pipeline-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\feature-store-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\ml-integration-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\ml-workbench-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\model-registry-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\monitoring-alerting-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\portfolio-management-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\risk-management-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\trading-gateway-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`

#### Check for containers without resource limits

- File: `d:\MD\forex_trading_platform\kubernetes\ui-service\templates\deployment.yaml`
- Line: 22
- Match: `containers:
`


## Recommendations

### High Severity Issues

- Address high severity issues as soon as possible
- Implement secure coding practices
- Consider using security linters

### Hardcoded Secrets

- Remove all hardcoded secrets
- Use environment variables or a secrets management solution
- Rotate any exposed secrets

