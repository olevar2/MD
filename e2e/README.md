# End-to-End Testing Framework

## Overview
The End-to-End (E2E) Testing Framework provides a comprehensive solution for testing the entire Forex Trading Platform across service boundaries. It enables automated testing of complete user workflows, API integrations, and system behavior under realistic conditions to ensure platform reliability and correctness.

## Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (dependency management)
- Docker and Docker Compose (for testing environment)
- Network connectivity to all platform services
- Test data generation tools

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd e2e
```
3. Install dependencies using Poetry:
```bash
poetry install
```

### Environment Variables
The following environment variables are required:

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `TEST_ENV` | Test environment (local, development, staging) | local |
| `SERVICE_URLS` | JSON string containing service URLs | - |
| `TEST_ACCOUNT_CREDENTIALS` | Credentials for test accounts | - |
| `REPORT_PATH` | Path for test reports | ./reports |
| `PARALLEL_EXECUTION` | Number of parallel test executions | 3 |
| `SCREENSHOTS_ON_FAILURE` | Take screenshots on test failure | true |
| `TIMEOUT_SECONDS` | Default timeout for test operations | 30 |

### Running Tests
Run all end-to-end tests:
```bash
poetry run pytest
```

Run specific test suite:
```bash
poetry run pytest tests/trading_workflow_test.py
```

Run tests with HTML report:
```bash
poetry run pytest --html=report.html
```

## Framework Structure

### Test Organization
The E2E framework is organized into:

- `tests/`: Test suites for different platform capabilities
- `framework/`: Core test framework components
- `fixtures/`: Reusable test fixtures and data
- `utils/`: Test utilities and helpers
- `reporting/`: Test reporting and visualization
- `validation/`: Test result validation helpers

### Test Types
The framework supports multiple test types:

1. **API Integration Tests**: End-to-end API workflow testing
2. **UI Workflow Tests**: User interface interaction testing
3. **Data Flow Tests**: Testing data consistency across services
4. **Performance Tests**: Load and stress testing
5. **Security Tests**: Authentication and authorization testing
6. **Recovery Tests**: System recovery after failures

## Key Features

### Test Fixtures
- Test account management
- Market data simulation
- Order execution simulation
- Database fixtures for consistent testing

### Test Data Generation
- Dynamic test data generation
- Historical market data replay
- Custom market scenario creation
- Random order generation

### Test Execution
- Parallel test execution
- Cross-service orchestration
- Test retry policies
- Screenshot capture on failure

### Test Reporting
- Detailed test reports
- Test metrics visualization
- Failure analysis tools
- CI/CD integration

## Common Test Scenarios

### Trading Workflow Tests
- Order placement and execution
- Position management
- Portfolio updates
- Strategy execution

### Authentication & Security Tests
- Login/logout flows
- Permission validation
- API key authentication
- Rate limiting

### Data Consistency Tests
- Cross-service data validation
- Event propagation verification
- Cache consistency

### Recovery Tests
- Service restart recovery
- Network interruption handling
- Database failover

## Integration with CI/CD
The E2E framework integrates with CI/CD pipelines:

- Automated test execution on code changes
- Nightly comprehensive test runs
- Pre-deployment validation
- Performance regression detection

## Best Practices
- Tests should be independent and self-contained
- Use unique test data for each test run
- Clean up test data after test execution
- Design for parallel execution
- Implement appropriate wait strategies
- Use explicit assertions for clear failure diagnosis

## Troubleshooting
Common troubleshooting steps:

1. Check service health before test execution
2. Verify environment variables are correctly set
3. Examine detailed test logs
4. Check screenshots for UI-related failures
5. Validate test data setup

## Extending the Framework
Guidelines for extending the framework:

1. Create new page objects for UI components
2. Implement service clients for new APIs
3. Add custom test fixtures for specialized scenarios
4. Extend reporting for specific metrics
