#!/usr/bin/env python3
"""
Script to validate API endpoints against the platform's API design standards.

This script:
1. Scans FastAPI and Express/NestJS applications for API endpoints
2. Validates endpoints against the platform's API design standards
3. Reports compliance and violations
4. Optionally generates a report of findings
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


# API endpoint patterns for different frameworks
FASTAPI_ROUTE_PATTERN = r'@(?:router|app)\.(?P<method>get|post|put|patch|delete)\s*\(\s*[\'"](?P<path>[^\'"]+)[\'"]'
EXPRESS_ROUTE_PATTERN = r'(?:router|app)\.(?P<method>get|post|put|patch|delete)\s*\(\s*[\'"](?P<path>[^\'"]+)[\'"]'
NESTJS_ROUTE_PATTERN = r'@(?:Get|Post|Put|Patch|Delete)\s*\(\s*[\'"](?P<path>[^\'"]+)[\'"]'

# API design standards
API_STANDARDS = {
    "url_structure": {
        "pattern": r'^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$',
        "description": "URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}"
    },
    "resource_naming": {
        "pattern": r'^/?v\d+/[a-z-]+/[a-z-]+',
        "description": "Resource names should be plural nouns in kebab-case"
    },
    "http_methods": {
        "get": "Read operations",
        "post": "Create operations",
        "put": "Full update operations",
        "patch": "Partial update operations",
        "delete": "Delete operations"
    },
    "action_pattern": {
        "pattern": r'^/?v\d+/[a-z-]+/[a-z-]+/[a-z0-9-]+/[a-z-]+$',
        "description": "Actions should follow the pattern: /v{version}/{service}/{resource}/{id}/{action}"
    }
}


def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/linting
    return Path(__file__).parent.parent.parent


def find_api_files(service_path: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Find API files in a service directory.
    
    Args:
        service_path: Path to the service directory
        
    Returns:
        Tuple of (fastapi_files, express_files, nestjs_files)
    """
    fastapi_files = []
    express_files = []
    nestjs_files = []
    
    # Find Python files that might contain FastAPI routes
    for py_file in service_path.glob("**/*.py"):
        if py_file.name.startswith("__"):
            continue
        
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            if "fastapi" in content.lower() and ("@app." in content or "@router." in content):
                fastapi_files.append(py_file)
    
    # Find JavaScript/TypeScript files that might contain Express routes
    for js_file in service_path.glob("**/*.js"):
        if "node_modules" in str(js_file):
            continue
        
        with open(js_file, "r", encoding="utf-8") as f:
            content = f.read()
            if ("express" in content.lower() or "router" in content.lower()) and ("app." in content or "router." in content):
                express_files.append(js_file)
    
    # Find TypeScript files that might contain NestJS routes
    for ts_file in service_path.glob("**/*.ts"):
        if "node_modules" in str(ts_file):
            continue
        
        with open(ts_file, "r", encoding="utf-8") as f:
            content = f.read()
            if "@nestjs" in content.lower() and any(decorator in content for decorator in ["@Get", "@Post", "@Put", "@Patch", "@Delete"]):
                nestjs_files.append(ts_file)
    
    return fastapi_files, express_files, nestjs_files


def extract_endpoints(
    fastapi_files: List[Path],
    express_files: List[Path],
    nestjs_files: List[Path]
) -> List[Dict[str, str]]:
    """
    Extract API endpoints from files.
    
    Args:
        fastapi_files: List of FastAPI files
        express_files: List of Express files
        nestjs_files: List of NestJS files
        
    Returns:
        List of endpoints with method, path, and file information
    """
    endpoints = []
    
    # Extract FastAPI endpoints
    for file_path in fastapi_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Find router prefix if any
            prefix = ""
            prefix_match = re.search(r'APIRouter\s*\(\s*prefix\s*=\s*[\'"]([^\'"]+)[\'"]', content)
            if prefix_match:
                prefix = prefix_match.group(1)
            
            # Find routes
            for match in re.finditer(FASTAPI_ROUTE_PATTERN, content):
                method = match.group("method").lower()
                path = match.group("path")
                
                # Combine prefix and path
                if prefix and not path.startswith("/"):
                    path = f"{prefix}/{path}"
                elif prefix:
                    path = f"{prefix}{path}"
                
                endpoints.append({
                    "method": method,
                    "path": path,
                    "file": str(file_path),
                    "framework": "fastapi"
                })
    
    # Extract Express endpoints
    for file_path in express_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Find router prefix if any
            prefix = ""
            prefix_match = re.search(r'router\s*=\s*express\.Router\s*\(\s*\)\s*;\s*app\.use\s*\(\s*[\'"]([^\'"]+)[\'"]', content)
            if prefix_match:
                prefix = prefix_match.group(1)
            
            # Find routes
            for match in re.finditer(EXPRESS_ROUTE_PATTERN, content):
                method = match.group("method").lower()
                path = match.group("path")
                
                # Combine prefix and path
                if prefix and not path.startswith("/"):
                    path = f"{prefix}/{path}"
                elif prefix:
                    path = f"{prefix}{path}"
                
                endpoints.append({
                    "method": method,
                    "path": path,
                    "file": str(file_path),
                    "framework": "express"
                })
    
    # Extract NestJS endpoints
    for file_path in nestjs_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Find controller prefix if any
            prefix = ""
            prefix_match = re.search(r'@Controller\s*\(\s*[\'"]([^\'"]+)[\'"]', content)
            if prefix_match:
                prefix = prefix_match.group(1)
            
            # Find routes
            for decorator in ["Get", "Post", "Put", "Patch", "Delete"]:
                method = decorator.lower()
                pattern = r'@' + decorator + r'\s*\(\s*[\'"](?P<path>[^\'"]*)[\'"]'
                
                for match in re.finditer(pattern, content):
                    path = match.group("path") or ""
                    
                    # Combine prefix and path
                    if prefix and not path.startswith("/"):
                        path = f"{prefix}/{path}"
                    elif prefix:
                        path = f"{prefix}{path}"
                    elif not path:
                        path = "/"
                    
                    endpoints.append({
                        "method": method,
                        "path": path,
                        "file": str(file_path),
                        "framework": "nestjs"
                    })
    
    return endpoints


def validate_endpoints(endpoints: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate API endpoints against the platform's API design standards.
    
    Args:
        endpoints: List of endpoints with method, path, and file information
        
    Returns:
        Tuple of (compliant_endpoints, non_compliant_endpoints)
    """
    compliant_endpoints = []
    non_compliant_endpoints = []
    
    for endpoint in endpoints:
        path = endpoint["path"]
        method = endpoint["method"]
        violations = []
        
        # Check URL structure
        if not re.match(API_STANDARDS["url_structure"]["pattern"], path):
            violations.append({
                "type": "url_structure",
                "description": API_STANDARDS["url_structure"]["description"],
                "expected_pattern": API_STANDARDS["url_structure"]["pattern"]
            })
        
        # Check resource naming
        if not re.match(API_STANDARDS["resource_naming"]["pattern"], path):
            violations.append({
                "type": "resource_naming",
                "description": API_STANDARDS["resource_naming"]["description"],
                "expected_pattern": API_STANDARDS["resource_naming"]["pattern"]
            })
        
        # Check if path ends with an action and method is POST
        action_match = re.match(API_STANDARDS["action_pattern"]["pattern"], path)
        if action_match and method != "post":
            violations.append({
                "type": "action_method",
                "description": "Actions should use POST method",
                "expected_method": "post",
                "actual_method": method
            })
        
        # Add endpoint to appropriate list
        endpoint_info = {
            "method": method,
            "path": path,
            "file": endpoint["file"],
            "framework": endpoint["framework"]
        }
        
        if violations:
            endpoint_info["violations"] = violations
            non_compliant_endpoints.append(endpoint_info)
        else:
            compliant_endpoints.append(endpoint_info)
    
    return compliant_endpoints, non_compliant_endpoints


def generate_report(
    service_name: str,
    compliant_endpoints: List[Dict[str, Any]],
    non_compliant_endpoints: List[Dict[str, Any]]
) -> str:
    """
    Generate a report of the validation results.
    
    Args:
        service_name: Name of the service
        compliant_endpoints: List of compliant endpoints
        non_compliant_endpoints: List of non-compliant endpoints
        
    Returns:
        Report as a string
    """
    total_endpoints = len(compliant_endpoints) + len(non_compliant_endpoints)
    compliance_pct = int(len(compliant_endpoints) / total_endpoints * 100) if total_endpoints > 0 else 100
    
    report = [
        f"# API Endpoint Validation Report: {service_name}",
        "",
        "## Compliance Summary",
        "",
        f"- Total Endpoints: {total_endpoints}",
        f"- Compliant Endpoints: {len(compliant_endpoints)} ({compliance_pct}%)",
        f"- Non-Compliant Endpoints: {len(non_compliant_endpoints)} ({100 - compliance_pct}%)",
        "",
    ]
    
    if non_compliant_endpoints:
        report.extend([
            "## Non-Compliant Endpoints",
            ""
        ])
        
        for endpoint in sorted(non_compliant_endpoints, key=lambda e: e["path"]):
            report.extend([
                f"### {endpoint['method'].upper()} {endpoint['path']}",
                "",
                f"**File:** {endpoint['file']}",
                f"**Framework:** {endpoint['framework']}",
                "",
                "**Violations:**",
                ""
            ])
            
            for violation in endpoint["violations"]:
                report.append(f"- **{violation['type']}**: {violation['description']}")
                if "expected_pattern" in violation:
                    report.append(f"  - Expected pattern: `{violation['expected_pattern']}`")
                if "expected_method" in violation:
                    report.append(f"  - Expected method: `{violation['expected_method']}`, actual: `{violation['actual_method']}`")
            
            report.append("")
    
    report.extend([
        "## Compliant Endpoints",
        ""
    ])
    
    for endpoint in sorted(compliant_endpoints, key=lambda e: e["path"]):
        report.append(f"- {endpoint['method'].upper()} {endpoint['path']}")
    
    return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate API endpoints against platform standards")
    parser.add_argument("--service", help="Path to a specific service to validate")
    parser.add_argument("--all", action="store_true", help="Validate all services in the repository")
    parser.add_argument("--report", action="store_true", help="Generate a report for each service")
    parser.add_argument("--report-dir", default="tools/reports/api_endpoints", help="Directory to store reports")
    args = parser.parse_args()
    
    repo_root = get_repo_root()
    
    if args.service:
        service_path = Path(args.service)
        if not service_path.is_absolute():
            service_path = repo_root / service_path
        
        if not service_path.exists() or not service_path.is_dir():
            print(f"Error: Service path {service_path} does not exist or is not a directory")
            return 1
        
        services = [service_path]
    elif args.all:
        # Find all potential service directories
        services = []
        for item in repo_root.iterdir():
            if item.is_dir() and not item.name.startswith(".") and item.name not in ["tools", "docs", "common-lib", "common-js-lib"]:
                services.append(item)
    else:
        parser.print_help()
        return 1
    
    # Create report directory if needed
    if args.report:
        report_dir = repo_root / args.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate each service
    results = {}
    for service_path in services:
        service_name = service_path.name
        print(f"Validating API endpoints in {service_name}...")
        
        # Find API files
        fastapi_files, express_files, nestjs_files = find_api_files(service_path)
        
        if not fastapi_files and not express_files and not nestjs_files:
            print(f"  No API files found in {service_name}")
            continue
        
        # Extract endpoints
        endpoints = extract_endpoints(fastapi_files, express_files, nestjs_files)
        
        if not endpoints:
            print(f"  No API endpoints found in {service_name}")
            continue
        
        # Validate endpoints
        compliant_endpoints, non_compliant_endpoints = validate_endpoints(endpoints)
        
        # Calculate compliance percentage
        total_endpoints = len(compliant_endpoints) + len(non_compliant_endpoints)
        compliance_pct = int(len(compliant_endpoints) / total_endpoints * 100) if total_endpoints > 0 else 100
        
        # Print summary
        print(f"  Total Endpoints: {total_endpoints}")
        print(f"  Compliant: {len(compliant_endpoints)} ({compliance_pct}%)")
        print(f"  Non-Compliant: {len(non_compliant_endpoints)} ({100 - compliance_pct}%)")
        
        if non_compliant_endpoints:
            print("  Non-compliant endpoints:")
            for endpoint in sorted(non_compliant_endpoints, key=lambda e: e["path"])[:5]:  # Show only first 5
                print(f"    - {endpoint['method'].upper()} {endpoint['path']}")
            if len(non_compliant_endpoints) > 5:
                print(f"    - ... and {len(non_compliant_endpoints) - 5} more")
        
        # Generate report if requested
        if args.report:
            report = generate_report(service_name, compliant_endpoints, non_compliant_endpoints)
            report_path = report_dir / f"{service_name}_api_endpoints_report.md"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"  Report saved to {report_path}")
        
        # Store results
        results[service_name] = {
            "total_endpoints": total_endpoints,
            "compliant_endpoints": len(compliant_endpoints),
            "non_compliant_endpoints": len(non_compliant_endpoints),
            "compliance_pct": compliance_pct
        }
    
    # Print overall summary
    print("\nOverall Summary:")
    total_endpoints = sum(result["total_endpoints"] for result in results.values())
    total_compliant = sum(result["compliant_endpoints"] for result in results.values())
    overall_pct = int(total_compliant / total_endpoints * 100) if total_endpoints > 0 else 100
    
    print(f"Total Endpoints: {total_endpoints}")
    print(f"Compliant Endpoints: {total_compliant} ({overall_pct}%)")
    print(f"Non-Compliant Endpoints: {total_endpoints - total_compliant} ({100 - overall_pct}%)")
    
    for service_name, result in sorted(results.items()):
        print(f"{service_name}: {result['compliance_pct']}% compliant ({result['compliant_endpoints']}/{result['total_endpoints']})")
    
    # Return non-zero exit code if any non-compliant endpoints
    return 1 if total_endpoints - total_compliant > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
