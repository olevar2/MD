#!/usr/bin/env python
"""
API Standardization Validator

This script validates that all API endpoints in the Analysis Engine Service
follow the standardization patterns defined in the API Standardization Plan.

It checks for:
- URL structure
- HTTP methods
- Request/response models
- Documentation
- Error handling
- Client libraries
"""

import os
import sys
import re
import importlib
import inspect
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import json
import argparse

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from fastapi import APIRouter, Depends, Path, Query, Body, HTTPException
from pydantic import BaseModel

@dataclass
class EndpointInfo:
    """Information about an API endpoint"""
    path: str
    http_method: str
    function_name: str
    module_path: str
    has_response_model: bool
    has_summary: bool
    has_description: bool
    has_correlation_id: bool
    has_error_handling: bool

@dataclass
class ClientInfo:
    """Information about a client library"""
    name: str
    module_path: str
    has_resilience: bool
    has_error_handling: bool
    has_logging: bool
    has_timeout: bool
    has_type_hints: bool

@dataclass
class ValidationResult:
    """Result of API standardization validation"""
    endpoints: List[EndpointInfo]
    clients: List[ClientInfo]
    compliant_endpoints: int
    non_compliant_endpoints: int
    compliant_clients: int
    non_compliant_clients: int
    issues: List[str]

def find_api_modules() -> List[str]:
    """Find all API modules in the Analysis Engine Service"""
    api_dir = project_root / "analysis_engine" / "api" / "v1" / "standardized"
    api_modules = []
    
    for file_path in api_dir.glob("*.py"):
        if file_path.name.startswith("__"):
            continue
        
        module_name = f"analysis_engine.api.v1.standardized.{file_path.stem}"
        api_modules.append(module_name)
    
    return api_modules

def find_client_modules() -> List[str]:
    """Find all client modules in the Analysis Engine Service"""
    client_dir = project_root / "analysis_engine" / "clients" / "standardized"
    client_modules = []
    
    for file_path in client_dir.glob("*.py"):
        if file_path.name.startswith("__") or file_path.name == "client_factory.py":
            continue
        
        module_name = f"analysis_engine.clients.standardized.{file_path.stem}"
        client_modules.append(module_name)
    
    return client_modules

def extract_endpoints(module_name: str) -> List[EndpointInfo]:
    """Extract API endpoints from a module"""
    endpoints = []
    
    try:
        module = importlib.import_module(module_name)
        
        # Find all router objects
        routers = []
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, APIRouter):
                routers.append(obj)
        
        # If no routers found directly, look for setup function
        if not routers:
            for name, obj in inspect.getmembers(module):
                if name.startswith("setup_") and name.endswith("_routes") and callable(obj):
                    # Create a mock app to capture routers
                    class MockApp:
                        def __init__(self):
                            self.routers = []
                        
                        def include_router(self, router, prefix=None):
                            self.routers.append((router, prefix))
                    
                    mock_app = MockApp()
                    obj(mock_app)
                    
                    for router, prefix in mock_app.routers:
                        if prefix == "/api":
                            routers.append(router)
        
        # Extract endpoints from routers
        for router in routers:
            for route in router.routes:
                # Skip HEAD and OPTIONS methods
                if route.methods and ("HEAD" in route.methods or "OPTIONS" in route.methods):
                    continue
                
                # Get HTTP method
                http_method = "GET"  # Default
                if route.methods:
                    for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        if method in route.methods:
                            http_method = method
                            break
                
                # Get endpoint path
                path = route.path
                if router.prefix:
                    path = router.prefix + path
                
                # Get endpoint function
                endpoint_func = route.endpoint
                function_name = endpoint_func.__name__
                
                # Check for response model
                has_response_model = hasattr(route, "response_model") and route.response_model is not None
                
                # Check for summary and description
                has_summary = hasattr(route, "summary") and route.summary is not None
                has_description = hasattr(route, "description") and route.description is not None
                
                # Check for correlation ID
                has_correlation_id = False
                source_code = inspect.getsource(endpoint_func)
                if "correlation_id" in source_code:
                    has_correlation_id = True
                
                # Check for error handling
                has_error_handling = False
                if "try:" in source_code and "except" in source_code:
                    has_error_handling = True
                
                endpoints.append(EndpointInfo(
                    path=path,
                    http_method=http_method,
                    function_name=function_name,
                    module_path=module_name,
                    has_response_model=has_response_model,
                    has_summary=has_summary,
                    has_description=has_description,
                    has_correlation_id=has_correlation_id,
                    has_error_handling=has_error_handling
                ))
    except Exception as e:
        print(f"Error extracting endpoints from {module_name}: {str(e)}")
    
    return endpoints

def extract_client_info(module_name: str) -> Optional[ClientInfo]:
    """Extract client information from a module"""
    try:
        module = importlib.import_module(module_name)
        
        # Find client class
        client_class = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and "Client" in name and name != "BaseClient":
                client_class = obj
                break
        
        if not client_class:
            return None
        
        # Check for resilience patterns
        has_resilience = False
        has_error_handling = False
        has_logging = False
        has_timeout = False
        has_type_hints = True  # Assume true until proven otherwise
        
        # Check class source code
        source_code = inspect.getsource(client_class)
        if "retry_with_backoff" in source_code or "circuit_breaker" in source_code:
            has_resilience = True
        
        if "except" in source_code and "raise" in source_code:
            has_error_handling = True
        
        if "logger" in source_code and "logger.info" in source_code:
            has_logging = True
        
        if "timeout" in source_code:
            has_timeout = True
        
        # Check type hints
        for name, method in inspect.getmembers(client_class, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            
            signature = inspect.signature(method)
            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue
                
                if param.annotation == inspect.Parameter.empty:
                    has_type_hints = False
                    break
            
            if method.__annotations__.get("return", inspect.Parameter.empty) == inspect.Parameter.empty:
                has_type_hints = False
                break
        
        return ClientInfo(
            name=client_class.__name__,
            module_path=module_name,
            has_resilience=has_resilience,
            has_error_handling=has_error_handling,
            has_logging=has_logging,
            has_timeout=has_timeout,
            has_type_hints=has_type_hints
        )
    except Exception as e:
        print(f"Error extracting client info from {module_name}: {str(e)}")
        return None

def validate_standardization() -> ValidationResult:
    """Validate API standardization"""
    # Find API modules
    api_modules = find_api_modules()
    
    # Extract endpoints
    all_endpoints = []
    for module_name in api_modules:
        endpoints = extract_endpoints(module_name)
        all_endpoints.extend(endpoints)
    
    # Find client modules
    client_modules = find_client_modules()
    
    # Extract client info
    all_clients = []
    for module_name in client_modules:
        client_info = extract_client_info(module_name)
        if client_info:
            all_clients.append(client_info)
    
    # Validate endpoints
    compliant_endpoints = 0
    non_compliant_endpoints = 0
    issues = []
    
    for endpoint in all_endpoints:
        is_compliant = True
        endpoint_issues = []
        
        # Check URL structure
        if not re.match(r"^/v1/analysis/[a-z-]+/.*$", endpoint.path):
            is_compliant = False
            endpoint_issues.append(f"URL structure does not follow pattern: {endpoint.path}")
        
        # Check for response model
        if not endpoint.has_response_model and endpoint.http_method != "DELETE":
            is_compliant = False
            endpoint_issues.append("Missing response model")
        
        # Check for summary and description
        if not endpoint.has_summary:
            is_compliant = False
            endpoint_issues.append("Missing summary")
        
        if not endpoint.has_description:
            is_compliant = False
            endpoint_issues.append("Missing description")
        
        # Check for correlation ID
        if not endpoint.has_correlation_id:
            is_compliant = False
            endpoint_issues.append("Missing correlation ID handling")
        
        # Check for error handling
        if not endpoint.has_error_handling:
            is_compliant = False
            endpoint_issues.append("Missing error handling")
        
        if is_compliant:
            compliant_endpoints += 1
        else:
            non_compliant_endpoints += 1
            issues.append(f"Endpoint {endpoint.http_method} {endpoint.path} ({endpoint.function_name}) has issues:")
            for issue in endpoint_issues:
                issues.append(f"  - {issue}")
    
    # Validate clients
    compliant_clients = 0
    non_compliant_clients = 0
    
    for client in all_clients:
        is_compliant = True
        client_issues = []
        
        # Check for resilience patterns
        if not client.has_resilience:
            is_compliant = False
            client_issues.append("Missing resilience patterns")
        
        # Check for error handling
        if not client.has_error_handling:
            is_compliant = False
            client_issues.append("Missing error handling")
        
        # Check for logging
        if not client.has_logging:
            is_compliant = False
            client_issues.append("Missing logging")
        
        # Check for timeout handling
        if not client.has_timeout:
            is_compliant = False
            client_issues.append("Missing timeout handling")
        
        # Check for type hints
        if not client.has_type_hints:
            is_compliant = False
            client_issues.append("Missing type hints")
        
        if is_compliant:
            compliant_clients += 1
        else:
            non_compliant_clients += 1
            issues.append(f"Client {client.name} has issues:")
            for issue in client_issues:
                issues.append(f"  - {issue}")
    
    return ValidationResult(
        endpoints=all_endpoints,
        clients=all_clients,
        compliant_endpoints=compliant_endpoints,
        non_compliant_endpoints=non_compliant_endpoints,
        compliant_clients=compliant_clients,
        non_compliant_clients=non_compliant_clients,
        issues=issues
    )

def generate_report(result: ValidationResult, output_file: Optional[str] = None) -> None:
    """Generate a report of the validation results"""
    report = []
    
    report.append("# API Standardization Validation Report")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append(f"- Total Endpoints: {len(result.endpoints)}")
    report.append(f"- Compliant Endpoints: {result.compliant_endpoints}")
    report.append(f"- Non-Compliant Endpoints: {result.non_compliant_endpoints}")
    report.append(f"- Compliance Rate: {result.compliant_endpoints / len(result.endpoints) * 100:.2f}%")
    report.append("")
    report.append(f"- Total Clients: {len(result.clients)}")
    report.append(f"- Compliant Clients: {result.compliant_clients}")
    report.append(f"- Non-Compliant Clients: {result.non_compliant_clients}")
    report.append(f"- Compliance Rate: {result.compliant_clients / len(result.clients) * 100:.2f}%")
    report.append("")
    
    if result.issues:
        report.append("## Issues")
        report.append("")
        for issue in result.issues:
            report.append(f"- {issue}")
        report.append("")
    
    report.append("## Endpoints")
    report.append("")
    report.append("| Method | Path | Function | Response Model | Summary | Description | Correlation ID | Error Handling |")
    report.append("|--------|------|----------|---------------|---------|-------------|---------------|----------------|")
    
    for endpoint in sorted(result.endpoints, key=lambda e: e.path):
        report.append(f"| {endpoint.http_method} | {endpoint.path} | {endpoint.function_name} | {'✅' if endpoint.has_response_model else '❌'} | {'✅' if endpoint.has_summary else '❌'} | {'✅' if endpoint.has_description else '❌'} | {'✅' if endpoint.has_correlation_id else '❌'} | {'✅' if endpoint.has_error_handling else '❌'} |")
    
    report.append("")
    report.append("## Clients")
    report.append("")
    report.append("| Client | Resilience | Error Handling | Logging | Timeout | Type Hints |")
    report.append("|--------|------------|---------------|---------|---------|------------|")
    
    for client in sorted(result.clients, key=lambda c: c.name):
        report.append(f"| {client.name} | {'✅' if client.has_resilience else '❌'} | {'✅' if client.has_error_handling else '❌'} | {'✅' if client.has_logging else '❌'} | {'✅' if client.has_timeout else '❌'} | {'✅' if client.has_type_hints else '❌'} |")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        print(f"Report written to {output_file}")
    else:
        print(report_text)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Validate API standardization")
    parser.add_argument("--output", "-o", help="Output file for the report")
    args = parser.parse_args()
    
    result = validate_standardization()
    generate_report(result, args.output)
    
    # Return non-zero exit code if there are issues
    if result.non_compliant_endpoints > 0 or result.non_compliant_clients > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
