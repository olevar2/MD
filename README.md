# Forex Trading Platform

## Overview

This platform provides a comprehensive suite of services for forex trading, including data pipelines, technical analysis, machine learning integration, and trading execution.

## Project Structure

- **analysis-engine-service**: Technical analysis and indicator calculation
- **core-foundations**: Core libraries and utilities
- **data-pipeline-service**: Data ingestion and processing
- **feature-store-service**: Technical indicator management
- **ml-integration-service**: Machine learning model integration
- **ml-workbench-service**: ML model development environment
- **monitoring-alerting-service**: System monitoring and alerts
- **portfolio-management-service**: Portfolio tracking and management
- **risk-management-service**: Risk assessment and management
- **strategy-execution-engine**: Trading strategy execution
- **trading-gateway-service**: Interface to trading platforms
- **ui-service**: User interface

## Technical Documentation

### Developer Guides

- [Feature Extraction Framework](docs/developer/feature_extraction_framework.md): Guide to the consolidated feature extraction framework

### Architecture Documentation

- See the `docs/architecture` directory for system architecture documentation

## Recent Improvements

- **Consolidated Configuration System**: Implemented a centralized configuration system for the Analysis Engine Service with improved validation and documentation
- **Consolidated Feature Extraction Framework**: Merged duplicate feature extraction functionality between ML Integration Service and Analysis Engine Service
- **Enhanced Technical Indicators**: Implemented missing indicators in the Feature Store Service, including CCI, Williams %R, and ROC

## Important Notices

### Configuration Migration

The Analysis Engine Service has consolidated its configuration system into a single module. The old configuration modules (`analysis_engine.core.config` and `config.config`) are deprecated and will be removed after December 31, 2023.

Please migrate to the new configuration module (`analysis_engine.config`) as soon as possible. See the [Configuration Migration Guide](docs/configuration_migration_guide.md) for details.

Migration tools are available in the `tools` directory:
- `tools/deprecation_report.py`: Generate reports on usage of deprecated modules
- `tools/migrate_config_imports.py`: Automate the migration of imports
- `tools/scheduled_deprecation_report.py`: Generate and distribute reports on a schedule

See the [Tools README](tools/README.md) for more information.

## Contacts

For questions or issues, contact the development team.
