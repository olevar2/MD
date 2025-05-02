# Makefile for Data Pipeline Service

# Variables
PYTEST = python -m pytest
COVERAGE = python -m pytest --cov=data_pipeline_service
MODULE_PATH = data_pipeline_service

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

# Run only specific test categories
.PHONY: test-cleaning
test-cleaning:
	$(PYTEST) -xvs tests/cleaning/

.PHONY: test-validation
test-validation:
	$(PYTEST) -xvs tests/validation/

.PHONY: test-source-adapters
test-source-adapters:
	$(PYTEST) -xvs tests/source_adapters/

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

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all (default) - Run tests"
	@echo "  test          - Run tests without coverage"
	@echo "  coverage      - Run tests with coverage reporting"
	@echo "  test-cleaning - Test only data cleaning components"
	@echo "  test-validation - Test only data validation components"
	@echo "  test-source-adapters - Test only source adapters"
	@echo "  test-api      - Test only API endpoints"
	@echo "  clean         - Clean up temporary files and test artifacts"
