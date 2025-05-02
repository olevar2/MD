# Makefile for Core Foundations Module

# Variables
PYTEST = python -m pytest
COVERAGE = python -m pytest --cov=core_foundations
MODULE_PATH = core_foundations

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

# Run only specific test category
.PHONY: test-utils
test-utils:
	$(PYTEST) -xvs tests/utils/

.PHONY: test-config
test-config:
	$(PYTEST) -xvs tests/config/

.PHONY: test-api
test-api:
	$(PYTEST) -xvs tests/api/

# Clean up temporary files and test artifacts
.PHONY: clean
clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf coverage_html
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf tests/*/__pycache__
	rm -rf $(MODULE_PATH)/__pycache__
	rm -rf $(MODULE_PATH)/*/__pycache__

# Generate documentation
.PHONY: docs
docs:
	pdoc --html --output-dir docs/api $(MODULE_PATH)
	@echo "Documentation generated in docs/api directory"

# Run linting
.PHONY: lint
lint:
	flake8 $(MODULE_PATH) tests

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all (default) - Run tests"
	@echo "  test          - Run tests without coverage"
	@echo "  coverage      - Run tests with coverage reporting"
	@echo "  test-utils    - Test only utilities"
	@echo "  test-config   - Test only configuration"
	@echo "  test-api      - Test only API endpoints"
	@echo "  clean         - Clean up temporary files and test artifacts"
	@echo "  docs          - Generate documentation"
	@echo "  lint          - Run linting on code"
