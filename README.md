# Forex Trading Platform

[![CodeQL](https://github.com/olevar2/MD/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/olevar2/MD/actions/workflows/codeql-analysis.yml)
[![Python Security Scan](https://github.com/olevar2/MD/actions/workflows/python-security.yml/badge.svg)](https://github.com/olevar2/MD/actions/workflows/python-security.yml)
[![Deploy](https://github.com/olevar2/MD/actions/workflows/deploy.yml/badge.svg)](https://github.com/olevar2/MD/actions/workflows/deploy.yml)
[![Dependabot Status](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/olevar2/MD/blob/main/.github/dependabot.yml)

A comprehensive forex trading platform with microservice architecture, designed for high-performance trading, analysis, and portfolio management.

## Overview

This platform provides a complete solution for forex trading, including:

- Real-time market data processing
- Advanced technical analysis
- Machine learning-based prediction models
- Automated trading strategies
- Risk management and portfolio optimization
- Performance monitoring and reporting

## Services

The platform is built on a microservice architecture with the following components:

- **Trading Gateway Service**: Handles order execution, broker integration, and trade management
- **Feature Store Service**: Manages technical indicators, patterns, and feature engineering
- **Data Pipeline Service**: Processes and normalizes market data from various sources
- **Analysis Engine Service**: Provides technical analysis, pattern recognition, and strategy evaluation
- **ML Integration Service**: Integrates machine learning models for prediction and optimization
- **Risk Management Service**: Enforces risk limits and provides risk assessment
- **Portfolio Management Service**: Tracks and analyzes trading portfolios
- **Monitoring & Alerting Service**: Monitors system health and performance
- **UI Service**: Provides the user interface for the platform
- **Common Library**: Shared components, interfaces, and utilities

## Architecture

The platform follows an event-driven microservice architecture with:

- Standardized interfaces and adapters to reduce circular dependencies
- Centralized error handling and resilience patterns
- Comprehensive monitoring and observability
- Data reconciliation across services
- Consistent naming conventions and code organization

## Development

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- PostgreSQL
- Redis
- Kafka (optional)

### Getting Started

1. Clone the repository
2. Install dependencies for each service
3. Configure environment variables
4. Run the services using Docker Compose

```bash
git clone https://github.com/olevar2/MD.git
cd MD
docker-compose up -d
```

## Security

This project uses:
- GitHub CodeQL for code scanning
- Dependabot for dependency updates
- Regular security audits
- Comprehensive input validation

## License

This project is proprietary and confidential. All rights reserved.