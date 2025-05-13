"""
Domain-Specific Test Strategy

This module implements a domain-driven testing strategy for the Forex Trading Platform,
aligning test boundaries with domain boundaries to ensure comprehensive coverage
while maintaining clear separation of concerns.

Key features:
1. Domain-specific test fixtures and factories
2. Test patterns for each domain context
3. Integration test patterns for domain boundaries
4. Performance test patterns for critical domain operations
"""

import os
import sys
import logging
import inspect
import importlib
from typing import Dict, List, Any, Optional, Callable, Type, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("domain_test_strategy")

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DomainContext(Enum):
    """Enumeration of domain contexts in the Forex Trading Platform."""
    MARKET_DATA = auto()
    TECHNICAL_ANALYSIS = auto()
    FEATURE_ENGINEERING = auto()
    MACHINE_LEARNING = auto()
    STRATEGY = auto()
    PORTFOLIO_MANAGEMENT = auto()
    RISK_MANAGEMENT = auto()
    EXECUTION = auto()
    REPORTING = auto()
    USER_INTERFACE = auto()


@dataclass
class DomainTestConfig:
    """Configuration for domain-specific tests."""
    context: DomainContext
    service_paths: List[str]
    fixture_paths: List[str] = field(default_factory=list)
    mock_paths: List[str] = field(default_factory=list)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    dependencies: List[DomainContext] = field(default_factory=list)
    boundary_tests: Dict[str, List[str]] = field(default_factory=dict)


class DomainTestRegistry:
    """Registry for domain-specific test configurations."""
    
    def __init__(self):
        """Initialize the registry."""
        self.configs: Dict[DomainContext, DomainTestConfig] = {}
        self.fixtures: Dict[str, Callable] = {}
        self.mocks: Dict[str, Any] = {}
        self.factories: Dict[str, Callable] = {}
    
    def register_domain(self, config: DomainTestConfig) -> None:
        """
        Register a domain test configuration.
        
        Args:
            config: Domain test configuration
        """
        self.configs[config.context] = config
        logger.info(f"Registered domain test config for {config.context.name}")
    
    def register_fixture(self, name: str, fixture_func: Callable) -> None:
        """
        Register a test fixture.
        
        Args:
            name: Fixture name
            fixture_func: Fixture function
        """
        self.fixtures[name] = fixture_func
        logger.info(f"Registered fixture: {name}")
    
    def register_mock(self, name: str, mock_obj: Any) -> None:
        """
        Register a mock object.
        
        Args:
            name: Mock name
            mock_obj: Mock object
        """
        self.mocks[name] = mock_obj
        logger.info(f"Registered mock: {name}")
    
    def register_factory(self, name: str, factory_func: Callable) -> None:
        """
        Register a test data factory.
        
        Args:
            name: Factory name
            factory_func: Factory function
        """
        self.factories[name] = factory_func
        logger.info(f"Registered factory: {name}")
    
    def get_domain_config(self, context: DomainContext) -> Optional[DomainTestConfig]:
        """
        Get domain test configuration.
        
        Args:
            context: Domain context
            
        Returns:
            Domain test configuration or None if not found
        """
        return self.configs.get(context)
    
    def get_fixture(self, name: str) -> Optional[Callable]:
        """
        Get a test fixture.
        
        Args:
            name: Fixture name
            
        Returns:
            Fixture function or None if not found
        """
        return self.fixtures.get(name)
    
    def get_mock(self, name: str) -> Optional[Any]:
        """
        Get a mock object.
        
        Args:
            name: Mock name
            
        Returns:
            Mock object or None if not found
        """
        return self.mocks.get(name)
    
    def get_factory(self, name: str) -> Optional[Callable]:
        """
        Get a test data factory.
        
        Args:
            name: Factory name
            
        Returns:
            Factory function or None if not found
        """
        return self.factories.get(name)
    
    def get_boundary_tests(self, context1: DomainContext, context2: DomainContext) -> List[str]:
        """
        Get boundary tests between two domains.
        
        Args:
            context1: First domain context
            context2: Second domain context
            
        Returns:
            List of boundary test paths
        """
        config1 = self.get_domain_config(context1)
        if not config1:
            return []
        
        boundary_key = context2.name.lower()
        return config1.boundary_tests.get(boundary_key, [])


class DomainTestRunner:
    """Runner for domain-specific tests."""
    
    def __init__(self, registry: DomainTestRegistry):
        """
        Initialize the test runner.
        
        Args:
            registry: Domain test registry
        """
        self.registry = registry
    
    def run_domain_tests(self, context: DomainContext) -> Dict[str, Any]:
        """
        Run tests for a specific domain.
        
        Args:
            context: Domain context
            
        Returns:
            Test results
        """
        config = self.registry.get_domain_config(context)
        if not config:
            logger.error(f"No test configuration found for domain: {context.name}")
            return {"success": False, "error": f"No test configuration found for domain: {context.name}"}
        
        logger.info(f"Running tests for domain: {context.name}")
        
        # Collect test paths
        test_paths = []
        for service_path in config.service_paths:
            test_paths.extend(self._collect_test_paths(service_path))
        
        # Run tests
        results = self._run_pytest(test_paths)
        
        return {
            "success": results.get("exitcode", 1) == 0,
            "domain": context.name,
            "test_count": results.get("test_count", 0),
            "passed": results.get("passed", 0),
            "failed": results.get("failed", 0),
            "skipped": results.get("skipped", 0),
            "duration": results.get("duration", 0),
            "test_paths": test_paths
        }
    
    def run_boundary_tests(self, context1: DomainContext, context2: DomainContext) -> Dict[str, Any]:
        """
        Run boundary tests between two domains.
        
        Args:
            context1: First domain context
            context2: Second domain context
            
        Returns:
            Test results
        """
        logger.info(f"Running boundary tests between {context1.name} and {context2.name}")
        
        # Get boundary tests
        test_paths = self.registry.get_boundary_tests(context1, context2)
        if not test_paths:
            logger.warning(f"No boundary tests found between {context1.name} and {context2.name}")
            return {
                "success": True,
                "warning": f"No boundary tests found between {context1.name} and {context2.name}",
                "test_count": 0
            }
        
        # Run tests
        results = self._run_pytest(test_paths)
        
        return {
            "success": results.get("exitcode", 1) == 0,
            "domain_boundary": f"{context1.name}_{context2.name}",
            "test_count": results.get("test_count", 0),
            "passed": results.get("passed", 0),
            "failed": results.get("failed", 0),
            "skipped": results.get("skipped", 0),
            "duration": results.get("duration", 0),
            "test_paths": test_paths
        }
    
    def run_all_domain_tests(self) -> Dict[str, Any]:
        """
        Run tests for all domains.
        
        Returns:
            Test results
        """
        logger.info("Running tests for all domains")
        
        results = {}
        for context in DomainContext:
            domain_results = self.run_domain_tests(context)
            results[context.name] = domain_results
        
        # Calculate overall success
        success = all(result.get("success", False) for result in results.values())
        
        return {
            "success": success,
            "domains": results
        }
    
    def run_all_boundary_tests(self) -> Dict[str, Any]:
        """
        Run all boundary tests.
        
        Returns:
            Test results
        """
        logger.info("Running all boundary tests")
        
        results = {}
        for context1 in DomainContext:
            config1 = self.registry.get_domain_config(context1)
            if not config1:
                continue
                
            for context2 in config1.dependencies:
                boundary_key = f"{context1.name}_{context2.name}"
                boundary_results = self.run_boundary_tests(context1, context2)
                results[boundary_key] = boundary_results
        
        # Calculate overall success
        success = all(result.get("success", False) for result in results.values())
        
        return {
            "success": success,
            "boundaries": results
        }
    
    def _collect_test_paths(self, service_path: str) -> List[str]:
        """
        Collect test paths for a service.
        
        Args:
            service_path: Service path
            
        Returns:
            List of test paths
        """
        test_paths = []
        service_dir = Path(service_path)
        
        # Check if service directory exists
        if not service_dir.exists():
            logger.warning(f"Service directory not found: {service_path}")
            return test_paths
        
        # Look for tests directory
        tests_dir = service_dir / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.glob("**/*.py"):
                if test_file.name.startswith("test_"):
                    test_paths.append(str(test_file))
        
        return test_paths
    
    def _run_pytest(self, test_paths: List[str]) -> Dict[str, Any]:
        """
        Run pytest on test paths.
        
        Args:
            test_paths: List of test paths
            
        Returns:
            Test results
        """
        if not test_paths:
            logger.warning("No test paths provided")
            return {
                "exitcode": 0,
                "test_count": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "duration": 0
            }
        
        import pytest
        
        # Create pytest arguments
        pytest_args = [
            "-v",
            "--no-header",
            "--no-summary",
            "-p", "no:warnings"
        ]
        pytest_args.extend(test_paths)
        
        # Run pytest
        start_time = datetime.now()
        result = pytest.main(pytest_args)
        end_time = datetime.now()
        
        # Calculate duration
        duration = (end_time - start_time).total_seconds()
        
        # Parse pytest result
        return {
            "exitcode": result,
            "test_count": len(test_paths),
            "passed": 0,  # Would need pytest hooks to get actual counts
            "failed": 0,
            "skipped": 0,
            "duration": duration
        }


# Initialize the registry
registry = DomainTestRegistry()

# Register domain test configurations
registry.register_domain(DomainTestConfig(
    context=DomainContext.MARKET_DATA,
    service_paths=[
        "data-pipeline-service",
        "market-data-service"
    ],
    fixture_paths=[
        "testing/fixtures/market_data"
    ],
    performance_thresholds={
        "data_fetch_ms": 100,
        "data_transform_ms": 50
    },
    dependencies=[],
    boundary_tests={
        "technical_analysis": [
            "testing/boundary_tests/market_data_technical_analysis_test.py"
        ]
    }
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.TECHNICAL_ANALYSIS,
    service_paths=[
        "analysis-engine-service"
    ],
    fixture_paths=[
        "testing/fixtures/technical_analysis"
    ],
    performance_thresholds={
        "indicator_calculation_ms": 200,
        "pattern_recognition_ms": 500
    },
    dependencies=[
        DomainContext.MARKET_DATA
    ],
    boundary_tests={
        "feature_engineering": [
            "testing/boundary_tests/technical_analysis_feature_engineering_test.py"
        ]
    }
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.FEATURE_ENGINEERING,
    service_paths=[
        "feature-store-service"
    ],
    fixture_paths=[
        "testing/fixtures/feature_engineering"
    ],
    performance_thresholds={
        "feature_extraction_ms": 300,
        "feature_transformation_ms": 150
    },
    dependencies=[
        DomainContext.TECHNICAL_ANALYSIS,
        DomainContext.MARKET_DATA
    ],
    boundary_tests={
        "machine_learning": [
            "testing/boundary_tests/feature_engineering_machine_learning_test.py"
        ]
    }
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.MACHINE_LEARNING,
    service_paths=[
        "ml-integration-service",
        "ml_workbench-service"
    ],
    fixture_paths=[
        "testing/fixtures/machine_learning"
    ],
    performance_thresholds={
        "model_inference_ms": 500,
        "model_training_s": 300
    },
    dependencies=[
        DomainContext.FEATURE_ENGINEERING
    ],
    boundary_tests={
        "strategy": [
            "testing/boundary_tests/machine_learning_strategy_test.py"
        ]
    }
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.STRATEGY,
    service_paths=[
        "strategy-execution-engine"
    ],
    fixture_paths=[
        "testing/fixtures/strategy"
    ],
    performance_thresholds={
        "strategy_evaluation_ms": 200,
        "signal_generation_ms": 100
    },
    dependencies=[
        DomainContext.TECHNICAL_ANALYSIS,
        DomainContext.MACHINE_LEARNING
    ],
    boundary_tests={
        "portfolio_management": [
            "testing/boundary_tests/strategy_portfolio_management_test.py"
        ],
        "execution": [
            "testing/boundary_tests/strategy_execution_test.py"
        ]
    }
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.PORTFOLIO_MANAGEMENT,
    service_paths=[
        "portfolio-management-service"
    ],
    fixture_paths=[
        "testing/fixtures/portfolio_management"
    ],
    performance_thresholds={
        "position_sizing_ms": 50,
        "portfolio_update_ms": 100
    },
    dependencies=[
        DomainContext.STRATEGY,
        DomainContext.RISK_MANAGEMENT
    ],
    boundary_tests={
        "execution": [
            "testing/boundary_tests/portfolio_management_execution_test.py"
        ]
    }
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.RISK_MANAGEMENT,
    service_paths=[
        "risk-management-service"
    ],
    fixture_paths=[
        "testing/fixtures/risk_management"
    ],
    performance_thresholds={
        "risk_calculation_ms": 100,
        "exposure_update_ms": 50
    },
    dependencies=[
        DomainContext.PORTFOLIO_MANAGEMENT
    ],
    boundary_tests={
        "portfolio_management": [
            "testing/boundary_tests/risk_management_portfolio_management_test.py"
        ]
    }
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.EXECUTION,
    service_paths=[
        "trading-gateway-service"
    ],
    fixture_paths=[
        "testing/fixtures/execution"
    ],
    performance_thresholds={
        "order_submission_ms": 200,
        "order_update_ms": 100
    },
    dependencies=[
        DomainContext.PORTFOLIO_MANAGEMENT,
        DomainContext.STRATEGY
    ],
    boundary_tests={}
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.REPORTING,
    service_paths=[
        "reporting-service"
    ],
    fixture_paths=[
        "testing/fixtures/reporting"
    ],
    performance_thresholds={
        "report_generation_ms": 500,
        "data_aggregation_ms": 300
    },
    dependencies=[
        DomainContext.PORTFOLIO_MANAGEMENT,
        DomainContext.EXECUTION
    ],
    boundary_tests={}
))

registry.register_domain(DomainTestConfig(
    context=DomainContext.USER_INTERFACE,
    service_paths=[
        "ui-service"
    ],
    fixture_paths=[
        "testing/fixtures/user_interface"
    ],
    performance_thresholds={
        "page_load_ms": 1000,
        "chart_rendering_ms": 500
    },
    dependencies=[
        DomainContext.REPORTING,
        DomainContext.TECHNICAL_ANALYSIS
    ],
    boundary_tests={}
))

# Create test runner
runner = DomainTestRunner(registry)


def main():
    """Run domain-specific tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run domain-specific tests")
    parser.add_argument(
        "--domain",
        choices=[domain.name for domain in DomainContext],
        help="Domain to test"
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Run tests for all domains"
    )
    parser.add_argument(
        "--boundary",
        action="store_true",
        help="Run boundary tests"
    )
    parser.add_argument(
        "--all-boundaries",
        action="store_true",
        help="Run all boundary tests"
    )
    
    args = parser.parse_args()
    
    if args.domain:
        # Run tests for specific domain
        domain_context = DomainContext[args.domain]
        results = runner.run_domain_tests(domain_context)
        print(f"Domain test results for {args.domain}:")
        print(f"Success: {results['success']}")
        print(f"Tests: {results['test_count']}")
        
    elif args.all_domains:
        # Run tests for all domains
        results = runner.run_all_domain_tests()
        print("All domain test results:")
        print(f"Overall success: {results['success']}")
        for domain, domain_results in results['domains'].items():
            print(f"  {domain}: {domain_results['success']} ({domain_results['test_count']} tests)")
    
    elif args.boundary:
        # Run boundary tests
        if not args.domain:
            print("Error: --domain is required with --boundary")
            return
            
        domain_context = DomainContext[args.domain]
        config = registry.get_domain_config(domain_context)
        if not config:
            print(f"Error: No configuration found for domain {args.domain}")
            return
            
        for dependency in config.dependencies:
            results = runner.run_boundary_tests(domain_context, dependency)
            print(f"Boundary test results for {domain_context.name} -> {dependency.name}:")
            print(f"Success: {results['success']}")
            print(f"Tests: {results['test_count']}")
    
    elif args.all_boundaries:
        # Run all boundary tests
        results = runner.run_all_boundary_tests()
        print("All boundary test results:")
        print(f"Overall success: {results['success']}")
        for boundary, boundary_results in results['boundaries'].items():
            print(f"  {boundary}: {boundary_results['success']} ({boundary_results['test_count']} tests)")
    
    else:
        # No arguments provided, show help
        parser.print_help()


if __name__ == "__main__":
    main()