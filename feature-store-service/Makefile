# Makefile for Feature Store Service

# Variables
PYTEST = python -m pytest
COVERAGE = python -m pytest --cov=feature_store_service
SERVICE_PATH = feature_store_service

# Default target
.PHONY: all
all: test

# Run tests without coverage
.PHONY: test
test:
	$(PYTEST) -xvs tests/

# Run tests with coverage
.PHONY: coverage
coverage:
	$(COVERAGE) --cov-report=term-missing --cov-report=html:coverage_html tests/
	@echo "Coverage report generated in coverage_html directory"

# Run only caching system tests
.PHONY: test-cache
test-cache:
	$(PYTEST) -xvs tests/caching/

# Run caching system tests with coverage
.PHONY: coverage-cache
coverage-cache:
	$(COVERAGE) --cov=feature_store_service/caching --cov-report=term-missing --cov-report=html:cache_coverage_html tests/caching/
	@echo "Cache coverage report generated in cache_coverage_html directory"

# Clean up temporary files and test artifacts
.PHONY: clean
clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf coverage_html
	rm -rf cache_coverage_html
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf tests/*/__pycache__
	rm -rf $(SERVICE_PATH)/__pycache__
	rm -rf $(SERVICE_PATH)/*/__pycache__

# Generate cache documentation
.PHONY: cache-docs
cache-docs:
	pdoc --html --output-dir docs/api $(SERVICE_PATH)/caching
	@echo "Cache documentation generated in docs/api directory"

# Start the service in development mode
.PHONY: dev
dev:
	uvicorn $(SERVICE_PATH).main:app --reload

# Run linting
.PHONY: lint
lint:
	flake8 $(SERVICE_PATH) tests

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all (default) - Run tests"
	@echo "  test          - Run tests without coverage"
	@echo "  coverage      - Run tests with coverage reporting"
	@echo "  test-cache    - Run only caching system tests"
	@echo "  coverage-cache - Run caching tests with coverage"
	@echo "  clean         - Clean up temporary files and test artifacts"
	@echo "  cache-docs    - Generate documentation for the caching system"
	@echo "  dev           - Start the service in development mode"
	@echo "  lint          - Run linting on code"
