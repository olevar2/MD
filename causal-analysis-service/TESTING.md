# Comprehensive Testing with Docker

This document explains how to use Docker for comprehensive testing of the causal-analysis-service.

## Prerequisites

- Docker and Docker Compose installed
- Basic understanding of Docker concepts

## Available Testing Configurations

The following Docker configurations are available for testing:

1. **Dockerfile.testing**: A multi-purpose Dockerfile for running the service and tests
2. **docker-compose.yml**: For running the service with its dependencies
3. **docker-compose.test.yml**: For running different types of tests

## Running Different Types of Tests

### 1. Unit Tests

Unit tests verify that individual components work correctly in isolation.

```bash
# Run unit tests
docker-compose -f docker-compose.test.yml run --rm unit-tests
```

### 2. Integration Tests

Integration tests verify that components work correctly together.

```bash
# Run integration tests
docker-compose -f docker-compose.test.yml run --rm integration-tests
```

### 3. End-to-End Tests

End-to-end tests verify that the entire service works correctly from a user perspective.

```bash
# Run end-to-end tests
docker-compose -f docker-compose.test.yml run --rm e2e-tests
```

### 4. Load Tests

Load tests verify that the service performs well under load.

```bash
# Run load tests
docker-compose -f docker-compose.test.yml run --rm load-tests
```

### 5. Running All Tests

You can run all tests in sequence:

```bash
# Run all tests
docker-compose -f docker-compose.test.yml run --rm unit-tests && \
docker-compose -f docker-compose.test.yml run --rm integration-tests && \
docker-compose -f docker-compose.test.yml run --rm e2e-tests && \
docker-compose -f docker-compose.test.yml run --rm load-tests
```

## Running the Service

To run the service with its dependencies:

```bash
# Start the service
docker-compose up -d

# Check service logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Development Mode

For development with hot-reload:

```bash
# Start in development mode
docker-compose up -d

# View logs
docker-compose logs -f
```

## Customizing Tests

You can customize the test environment by modifying the Docker Compose files:

- Add more dependencies (databases, message queues, etc.)
- Change environment variables
- Adjust resource limits

## Test Directory Structure

The tests are organized in the following directories:

- `tests/unit/`: Unit tests
- `tests/integration/`: Integration tests
- `tests/e2e/`: End-to-end tests
- `tests/load/`: Load tests

## Troubleshooting

If you encounter issues:

1. Check Docker logs: `docker-compose logs`
2. Verify that all dependencies are running: `docker-compose ps`
3. Check network connectivity: `docker network inspect causal-analysis-network`
4. Ensure volumes are properly mounted: `docker volume ls`
