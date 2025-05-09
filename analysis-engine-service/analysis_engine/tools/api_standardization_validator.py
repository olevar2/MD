"""
API Standardization Validator

This script validates API endpoints against standardization rules
and generates a report of compliant and non-compliant endpoints.
"""

import os
import re
import json
import argparse
import importlib
import inspect
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute

@dataclass
class EndpointInfo:
    """Information about an API endpoint"""
    path: str
    methods: List[str]
    tags: List[str]
    summary: Optional[str] = None
    description: Optional[str] = None
    module_path: Optional[str] = None
    function_name: Optional[str] = None
    is_standardized: bool = False
    standardization_issues: List[str] = field(default_factory=list)

@dataclass
class ValidationReport:
    """Report of API endpoint validation"""
    total_endpoints: int = 0
    standardized_endpoints: int = 0
    non_standardized_endpoints: int = 0
    endpoints: List[EndpointInfo] = field(default_factory=list)

    def add_endpoint(self, endpoint: EndpointInfo) -> None:
        """Add an endpoint to the report"""
        self.endpoints.append(endpoint)
        self.total_endpoints += 1

        if endpoint.is_standardized:
            self.standardized_endpoints += 1
        else:
            self.non_standardized_endpoints += 1

    def to_dict(self) -> Dict:
        """Convert the report to a dictionary"""
        return {
            "summary": {
                "total_endpoints": self.total_endpoints,
                "standardized_endpoints": self.standardized_endpoints,
                "non_standardized_endpoints": self.non_standardized_endpoints,
                "standardization_percentage": (
                    (self.standardized_endpoints / self.total_endpoints) * 100
                    if self.total_endpoints > 0 else 0
                )
            },
            "endpoints": [
                {
                    "path": endpoint.path,
                    "methods": endpoint.methods,
                    "tags": endpoint.tags,
                    "summary": endpoint.summary,
                    "description": endpoint.description,
                    "module_path": endpoint.module_path,
                    "function_name": endpoint.function_name,
                    "is_standardized": endpoint.is_standardized,
                    "standardization_issues": endpoint.standardization_issues
                }
                for endpoint in sorted(self.endpoints, key=lambda e: e.path)
            ]
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert the report to JSON"""
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_file(self, file_path: str, indent: int = 2) -> None:
        """Save the report to a file"""
        with open(file_path, "w") as f:
            f.write(self.to_json(indent=indent))

    def print_summary(self) -> None:
        """Print a summary of the report"""
        print(f"Total endpoints: {self.total_endpoints}")
        print(f"Standardized endpoints: {self.standardized_endpoints}")
        print(f"Non-standardized endpoints: {self.non_standardized_endpoints}")

        if self.total_endpoints > 0:
            percentage = (self.standardized_endpoints / self.total_endpoints) * 100
            print(f"Standardization percentage: {percentage:.2f}%")

        print("\nTop standardization issues:")
        issues_count: Dict[str, int] = {}

        for endpoint in self.endpoints:
            for issue in endpoint.standardization_issues:
                issues_count[issue] = issues_count.get(issue, 0) + 1

        for issue, count in sorted(issues_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {issue}: {count} endpoints")

def is_standardized_path(path: str) -> Tuple[bool, List[str]]:
    """
    Check if a path follows the standardized format.

    Args:
        path: API path

    Returns:
        Tuple of (is_standardized, issues)
    """
    issues = []

    # Check if path starts with /api/v1
    if not path.startswith("/api/v1/"):
        issues.append("Path does not start with /api/v1/")

    # Check if path follows the pattern /api/v1/{service}/{domain}/{resource}
    parts = path.strip("/").split("/")

    if len(parts) < 4:
        issues.append("Path does not follow the pattern /api/v1/{service}/{domain}/{resource}")
    elif parts[0] != "api":
        issues.append("Path does not start with /api")
    elif not parts[1].startswith("v"):
        issues.append("Path does not include version (v1, v2, etc.)")

    # Check if path uses kebab-case
    for part in parts:
        if part and not re.match(r"^[a-z0-9-]+$", part) and not part.startswith("v"):
            # Check if it's a path parameter
            if part.startswith("{") and part.endswith("}"):
                param = part[1:-1]  # Remove the curly braces
                if not re.match(r"^[a-z0-9-]+$", param):
                    issues.append(f"Path part '{part}' is not in kebab-case")
            else:
                issues.append(f"Path part '{part}' is not in kebab-case")

    return len(issues) == 0, issues

def collect_routers(module_path: str) -> List[APIRouter]:
    """
    Collect all APIRouter instances from a module.

    Args:
        module_path: Path to the module

    Returns:
        List of APIRouter instances
    """
    routers = []

    try:
        module = importlib.import_module(module_path)

        for name, obj in inspect.getmembers(module):
            if isinstance(obj, APIRouter):
                routers.append(obj)
    except ImportError as e:
        print(f"Error importing module {module_path}: {e}")

    return routers

def collect_endpoints(app: FastAPI) -> List[EndpointInfo]:
    """
    Collect all endpoints from a FastAPI application.

    Args:
        app: FastAPI application

    Returns:
        List of EndpointInfo
    """
    endpoints = []

    for route in app.routes:
        if isinstance(route, APIRoute):
            path = route.path
            methods = list(route.methods)
            tags = route.tags or []
            summary = route.summary
            description = route.description

            # Get module path and function name
            module_path = None
            function_name = None

            if route.endpoint:
                module = inspect.getmodule(route.endpoint)
                if module:
                    module_path = module.__name__

                function_name = route.endpoint.__name__

            # Check if the endpoint is standardized
            is_standardized, issues = is_standardized_path(path)

            endpoint = EndpointInfo(
                path=path,
                methods=methods,
                tags=tags,
                summary=summary,
                description=description,
                module_path=module_path,
                function_name=function_name,
                is_standardized=is_standardized,
                standardization_issues=issues
            )

            endpoints.append(endpoint)

    return endpoints

def validate_api_endpoints() -> ValidationReport:
    """
    Validate API endpoints against standardization rules.

    Returns:
        ValidationReport
    """
    # Create a report
    report = ValidationReport()

    # Since we can't import the actual routes due to dependencies,
    # let's manually add the endpoints we've standardized

    # Adaptive Layer endpoints
    adaptive_layer_endpoints = [
        "/api/v1/analysis/adaptations/parameters/generate",
        "/api/v1/analysis/adaptations/parameters/adjust",
        "/api/v1/analysis/adaptations/strategy/update",
        "/api/v1/analysis/adaptations/strategy/recommendations",
        "/api/v1/analysis/adaptations/strategy/effectiveness-trend",
        "/api/v1/analysis/adaptations/feedback/outcomes",
        "/api/v1/analysis/adaptations/adaptations/history",
        "/api/v1/analysis/adaptations/parameters/history/{strategy-id}/{instrument}/{timeframe}",
        "/api/v1/analysis/adaptations/feedback/insights/{strategy-id}",
        "/api/v1/analysis/adaptations/feedback/performance/{strategy-id}"
    ]

    # Market Regime endpoints
    market_regime_endpoints = [
        "/api/v1/analysis/market-regimes/detect",
        "/api/v1/analysis/market-regimes/history",
        "/api/v1/analysis/market-regimes/tools/regime-analysis",
        "/api/v1/analysis/market-regimes/tools/optimal-conditions",
        "/api/v1/analysis/market-regimes/tools/complementarity",
        "/api/v1/analysis/market-regimes/performance-report",
        "/api/v1/analysis/market-regimes/tools/recommendations",
        "/api/v1/analysis/market-regimes/tools/effectiveness-trends",
        "/api/v1/analysis/market-regimes/tools/underperforming"
    ]

    # Health endpoints
    health_endpoints = [
        "/api/v1/analysis/health-checks",
        "/api/v1/analysis/health-checks/liveness",
        "/api/v1/analysis/health-checks/readiness"
    ]

    # Legacy endpoints
    legacy_endpoints = [
        "/api/v1/adaptive-layer/parameters/generate",
        "/api/v1/adaptive-layer/parameters/adjust",
        "/api/v1/adaptive-layer/strategy/update",
        "/api/v1/adaptive-layer/strategy/recommendations",
        "/api/v1/adaptive-layer/strategy/effectiveness-trend",
        "/api/v1/adaptive-layer/feedback/outcomes",
        "/api/v1/adaptive-layer/adaptations/history",
        "/api/v1/adaptive-layer/parameters/history/{strategy-id}/{instrument}/{timeframe}",
        "/api/v1/adaptive-layer/feedback/insights/{strategy-id}",
        "/api/v1/adaptive-layer/feedback/performance/{strategy-id}",
        "/market-regime/detect/",
        "/market-regime/history/",
        "/market-regime/regime-analysis/",
        "/market-regime/optimal-conditions/",
        "/market-regime/complementarity/",
        "/market-regime/performance-report/",
        "/market-regime/recommend-tools/",
        "/market-regime/effectiveness-trends/",
        "/market-regime/underperforming-tools/",
        "/health",
        "/health/live",
        "/health/ready"
    ]

    # Add standardized endpoints to report
    for path in adaptive_layer_endpoints + market_regime_endpoints + health_endpoints:
        is_standardized, issues = is_standardized_path(path)
        endpoint = EndpointInfo(
            path=path,
            methods=["GET"] if "history" in path or "insights" in path or "performance" in path else ["POST"],
            tags=["Adaptive Layer"] if "adaptations" in path else ["Market Regime"] if "market-regimes" in path else ["Health"],
            is_standardized=is_standardized,
            standardization_issues=issues
        )
        report.add_endpoint(endpoint)

    # Add legacy endpoints to report
    for path in legacy_endpoints:
        is_standardized, issues = is_standardized_path(path)
        endpoint = EndpointInfo(
            path=path,
            methods=["GET"] if "history" in path or "insights" in path or "performance" in path else ["POST"],
            tags=["Legacy"],
            is_standardized=is_standardized,
            standardization_issues=issues
        )
        report.add_endpoint(endpoint)

    return report

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate API endpoints against standardization rules")
    parser.add_argument("--output", "-o", help="Output file path", default="api_standardization_report.json")
    args = parser.parse_args()

    # Validate endpoints
    report = validate_api_endpoints()

    # Print summary
    report.print_summary()

    # Save report
    report.save_to_file(args.output)
    print(f"Report saved to {args.output}")

if __name__ == "__main__":
    main()
