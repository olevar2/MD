# Current Forex Trading Platform Architecture Report

Generated: 2025-05-10 04:08:01

## Summary

- Total Services: 53
- Total Dependencies: 57

## Services by Layer

### Analysis

- analysis-engine
- analysis-engine-service
- ml-integration-service
- ml-workbench-service

### Cross-cutting

- .pre-commit-config.yaml
- mcp-config.json
- monitoring
- monitoring-alerting-service

### Data

- data-pipeline-service
- feature-store-service

### Execution

- portfolio-management-service
- risk-management-service
- strategy-execution-engine
- trading-gateway-service

### Foundation

- common-js-lib
- common-lib
- core-foundations

### Presentation

- ui-service

### Unknown

- .augment.json
- .copilotrc
- .prettierrc.json
- .pytest-cache
- .vscode
- architecture
- chat-interface-template
- code-quality-report
- commit-message.txt
- data
- data-management-service
- data-reconciliation-commit-message.txt
- desktop-commander.log
- dockerfile
- e2e
- examples
- go-installer.msi
- infrastructure
- install-smithery.bat
- kubernetes
- m.id
- mcp-server
- mcp-servers.log
- model-registry-service
- optimization
- pyproject.toml
- pytest.ini
- reference-servers
- requirements.txt
- security
- sequential-thinking.log
- start-mcp-servers.bat
- test-mcp.txt
- testing
- tests

## Service Details

| Service | Layer | Files | Classes | Functions | Dependencies |
|---------|-------|-------|---------|-----------|--------------|
| .augment.json | unknown | 1 | 0 | 0 |  |
| .copilotrc | unknown | 1 | 0 | 0 |  |
| .pre-commit-config.yaml | cross-cutting | 1 | 0 | 0 |  |
| .prettierrc.json | unknown | 1 | 0 | 0 |  |
| .pytest-cache | unknown | 4 | 0 | 0 |  |
| .vscode | unknown | 3 | 0 | 0 |  |
| analysis-engine | analysis | 17 | 21 | 70 | analysis-engine-service, data |
| analysis-engine-service | analysis | 499 | 889 | 3110 | data, ml-integration-service, ml-workbench-service, risk-management-service, tests, trading-gateway-service |
| architecture | unknown | 214 | 0 | 0 |  |
| chat-interface-template | unknown | 7 | 1 | 8 | analysis-engine-service, ml-integration-service, trading-gateway-service |
| code-quality-report | unknown | 2 | 0 | 0 |  |
| commit-message.txt | unknown | 1 | 0 | 0 |  |
| common-js-lib | foundation | 16 | 0 | 0 |  |
| common-lib | foundation | 124 | 301 | 594 | data |
| core-foundations | foundation | 54 | 135 | 469 | data, kubernetes |
| data | unknown | 3 | 0 | 0 |  |
| data-management-service | unknown | 75 | 97 | 138 | data |
| data-pipeline-service | data | 111 | 154 | 435 | data |
| data-reconciliation-commit-message.txt | unknown | 1 | 0 | 0 |  |
| desktop-commander.log | unknown | 1 | 0 | 0 |  |
| dockerfile | unknown | 1 | 0 | 0 |  |
| e2e | unknown | 32 | 36 | 140 | data |
| examples | unknown | 9 | 2 | 13 | analysis-engine-service, strategy-execution-engine |
| feature-store-service | data | 327 | 449 | 2068 | analysis-engine-service, data, monitoring, tests |
| go-installer.msi | unknown | 1 | 0 | 0 |  |
| infrastructure | unknown | 33 | 7 | 22 | data |
| install-smithery.bat | unknown | 1 | 0 | 0 |  |
| kubernetes | unknown | 1 | 0 | 0 |  |
| m.id | unknown | 1 | 0 | 0 |  |
| mcp-config.json | cross-cutting | 1 | 0 | 0 |  |
| mcp-server | unknown | 8 | 3 | 17 |  |
| mcp-servers.log | unknown | 1 | 0 | 0 |  |
| ml-integration-service | analysis | 57 | 81 | 302 | data |
| ml-workbench-service | analysis | 129 | 281 | 948 | data, risk-management-service, trading-gateway-service |
| model-registry-service | unknown | 15 | 32 | 11 |  |
| monitoring | cross-cutting | 4 | 0 | 0 |  |
| monitoring-alerting-service | cross-cutting | 89 | 42 | 279 | analysis-engine-service, data, ml-integration-service, monitoring, strategy-execution-engine |
| optimization | unknown | 38 | 28 | 121 | data |
| portfolio-management-service | execution | 66 | 68 | 182 | analysis-engine-service |
| pyproject.toml | unknown | 1 | 0 | 0 |  |
| pytest.ini | unknown | 1 | 0 | 0 |  |
| reference-servers | unknown | 149 | 25 | 55 | data |
| requirements.txt | unknown | 1 | 0 | 0 |  |
| risk-management-service | execution | 77 | 138 | 448 | data |
| security | unknown | 13 | 9 | 49 |  |
| sequential-thinking.log | unknown | 1 | 0 | 0 |  |
| start-mcp-servers.bat | unknown | 1 | 0 | 0 |  |
| strategy-execution-engine | execution | 146 | 145 | 732 | analysis-engine-service, optimization, tests |
| test-mcp.txt | unknown | 1 | 0 | 0 |  |
| testing | unknown | 55 | 94 | 364 | analysis-engine-service, data, feature-store-service, ml-integration-service, ml-workbench-service, risk-management-service, strategy-execution-engine |
| tests | unknown | 26 | 13 | 89 | analysis-engine-service, feature-store-service |
| trading-gateway-service | execution | 115 | 181 | 733 | analysis-engine-service, common-js-lib, data, risk-management-service, tests |
| ui-service | presentation | 158 | 36 | 155 | analysis-engine-service, data, feature-store-service |

## Dependencies

### analysis-engine

- analysis-engine-service
- data

### analysis-engine-service

- data
- ml-integration-service
- ml-workbench-service
- risk-management-service
- tests
- trading-gateway-service

### chat-interface-template

- analysis-engine-service
- ml-integration-service
- trading-gateway-service

### common-lib

- data

### core-foundations

- data
- kubernetes

### data-management-service

- data

### data-pipeline-service

- data

### e2e

- data

### examples

- analysis-engine-service
- strategy-execution-engine

### feature-store-service

- analysis-engine-service
- data
- monitoring
- tests

### infrastructure

- data

### ml-integration-service

- data

### ml-workbench-service

- data
- risk-management-service
- trading-gateway-service

### monitoring-alerting-service

- analysis-engine-service
- data
- ml-integration-service
- monitoring
- strategy-execution-engine

### optimization

- data

### portfolio-management-service

- analysis-engine-service

### reference-servers

- data

### risk-management-service

- data

### strategy-execution-engine

- analysis-engine-service
- optimization
- tests

### testing

- analysis-engine-service
- data
- feature-store-service
- ml-integration-service
- ml-workbench-service
- risk-management-service
- strategy-execution-engine

### tests

- analysis-engine-service
- feature-store-service

### trading-gateway-service

- analysis-engine-service
- common-js-lib
- data
- risk-management-service
- tests

### ui-service

- analysis-engine-service
- data
- feature-store-service
