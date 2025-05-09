#!/usr/bin/env python3
"""
Script to validate service directory structures against the platform standards.

This script:
1. Reads the file structure standards from the documentation
2. Validates service directories against these standards
3. Reports compliance and violations
4. Optionally generates a report of findings
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


# Standard service structure patterns
PYTHON_SERVICE_STRUCTURE = {
    "required": [
        "{service_name}",
        "{service_name}/__init__.py",
        "{service_name}/api",
        "{service_name}/api/__init__.py",
        "{service_name}/domain",
        "{service_name}/domain/__init__.py",
        "{service_name}/infrastructure",
        "{service_name}/infrastructure/__init__.py",
        "tests",
        "tests/__init__.py",
        "pyproject.toml",
        "README.md",
    ],
    "recommended": [
        "{service_name}/api/routes",
        "{service_name}/api/models",
        "{service_name}/api/dependencies.py",
        "{service_name}/domain/models",
        "{service_name}/domain/services",
        "{service_name}/domain/repositories",
        "{service_name}/infrastructure/adapters",
        "{service_name}/infrastructure/persistence",
        "{service_name}/infrastructure/clients",
        "tests/unit",
        "tests/integration",
        "tests/conftest.py",
        "docs",
        ".pre-commit-config.yaml",
    ],
    "optional": [
        "{service_name}/api/middlewares",
        "{service_name}/api/websockets",
        "{service_name}/domain/events",
        "{service_name}/domain/exceptions.py",
        "{service_name}/infrastructure/messaging",
        "{service_name}/infrastructure/config.py",
        "tests/e2e",
        "tests/performance",
        "Dockerfile",
        "docker-compose.yml",
        "Makefile",
    ]
}

JS_SERVICE_STRUCTURE = {
    "required": [
        "src",
        "src/index.js",
        "src/api",
        "src/domain",
        "src/infrastructure",
        "tests",
        "package.json",
        "README.md",
    ],
    "recommended": [
        "src/api/routes",
        "src/api/models",
        "src/api/middlewares",
        "src/domain/models",
        "src/domain/services",
        "src/domain/repositories",
        "src/infrastructure/adapters",
        "src/infrastructure/persistence",
        "src/infrastructure/clients",
        "tests/unit",
        "tests/integration",
        "jest.config.js",
        ".eslintrc.js",
        ".prettierrc.json",
        "tsconfig.json",
    ],
    "optional": [
        "src/api/websockets",
        "src/domain/events",
        "src/domain/exceptions.js",
        "src/infrastructure/messaging",
        "src/infrastructure/config.js",
        "tests/e2e",
        "tests/performance",
        "Dockerfile",
        "docker-compose.yml",
        "Makefile",
    ]
}


def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/linting
    return Path(__file__).parent.parent.parent


def detect_service_type(service_path: Path) -> str:
    """
    Detect whether a service is Python or JavaScript/TypeScript based.
    
    Args:
        service_path: Path to the service directory
        
    Returns:
        "python", "javascript", or "unknown"
    """
    if (service_path / "pyproject.toml").exists() or list(service_path.glob("**/*.py")):
        return "python"
    elif (service_path / "package.json").exists() or list(service_path.glob("**/*.js")) or list(service_path.glob("**/*.ts")):
        return "javascript"
    else:
        return "unknown"


def validate_service_structure(
    service_path: Path, 
    service_type: str,
    service_name: Optional[str] = None
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Validate a service directory structure against the standards.
    
    Args:
        service_path: Path to the service directory
        service_type: "python" or "javascript"
        service_name: Optional service name to use in pattern matching
        
    Returns:
        Tuple of (compliance, violations) dictionaries
    """
    if service_name is None:
        service_name = service_path.name.replace("-", "_")
    
    # Select the appropriate structure pattern
    if service_type == "python":
        structure = PYTHON_SERVICE_STRUCTURE
    elif service_type == "javascript":
        structure = JS_SERVICE_STRUCTURE
    else:
        raise ValueError(f"Unknown service type: {service_type}")
    
    # Get all files and directories in the service
    all_files = set()
    for root, dirs, files in os.walk(service_path):
        rel_root = os.path.relpath(root, service_path)
        if rel_root == ".":
            rel_root = ""
        
        for dir_name in dirs:
            if dir_name not in [".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"]:
                all_files.add(os.path.join(rel_root, dir_name))
        
        for file_name in files:
            all_files.add(os.path.join(rel_root, file_name))
    
    # Check compliance and violations
    compliance = {
        "required": [],
        "recommended": [],
        "optional": []
    }
    
    violations = {
        "required": [],
        "recommended": []
    }
    
    # Check required files
    for pattern in structure["required"]:
        pattern = pattern.format(service_name=service_name)
        if any(re.match(f"^{pattern}$", f) for f in all_files):
            compliance["required"].append(pattern)
        else:
            violations["required"].append(pattern)
    
    # Check recommended files
    for pattern in structure["recommended"]:
        pattern = pattern.format(service_name=service_name)
        if any(re.match(f"^{pattern}$", f) for f in all_files):
            compliance["recommended"].append(pattern)
        else:
            violations["recommended"].append(pattern)
    
    # Check optional files
    for pattern in structure["optional"]:
        pattern = pattern.format(service_name=service_name)
        if any(re.match(f"^{pattern}$", f) for f in all_files):
            compliance["optional"].append(pattern)
    
    return compliance, violations


def generate_report(
    service_path: Path,
    service_type: str,
    compliance: Dict[str, List[str]],
    violations: Dict[str, List[str]]
) -> str:
    """
    Generate a report of the validation results.
    
    Args:
        service_path: Path to the service directory
        service_type: "python" or "javascript"
        compliance: Dictionary of compliant files/directories
        violations: Dictionary of missing files/directories
        
    Returns:
        Report as a string
    """
    service_name = service_path.name
    
    report = [
        f"# Service Structure Validation Report: {service_name}",
        "",
        f"**Service Type:** {service_type}",
        f"**Service Path:** {service_path}",
        "",
        "## Compliance Summary",
        "",
        f"- Required: {len(compliance['required'])}/{len(compliance['required']) + len(violations['required'])} ({int(len(compliance['required']) / (len(compliance['required']) + len(violations['required'])) * 100 if len(compliance['required']) + len(violations['required']) > 0 else 100)}%)",
        f"- Recommended: {len(compliance['recommended'])}/{len(compliance['recommended']) + len(violations['recommended'])} ({int(len(compliance['recommended']) / (len(compliance['recommended']) + len(violations['recommended'])) * 100 if len(compliance['recommended']) + len(violations['recommended']) > 0 else 100)}%)",
        f"- Optional: {len(compliance['optional'])}",
        "",
    ]
    
    if violations["required"]:
        report.extend([
            "## Required Files/Directories Missing",
            ""
        ])
        for item in sorted(violations["required"]):
            report.append(f"- `{item}`")
        report.append("")
    
    if violations["recommended"]:
        report.extend([
            "## Recommended Files/Directories Missing",
            ""
        ])
        for item in sorted(violations["recommended"]):
            report.append(f"- `{item}`")
        report.append("")
    
    report.extend([
        "## Compliant Files/Directories",
        "",
        "### Required",
        ""
    ])
    for item in sorted(compliance["required"]):
        report.append(f"- `{item}`")
    report.append("")
    
    report.extend([
        "### Recommended",
        ""
    ])
    for item in sorted(compliance["recommended"]):
        report.append(f"- `{item}`")
    report.append("")
    
    report.extend([
        "### Optional",
        ""
    ])
    for item in sorted(compliance["optional"]):
        report.append(f"- `{item}`")
    
    return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate service directory structures against platform standards")
    parser.add_argument("--service", help="Path to a specific service to validate")
    parser.add_argument("--all", action="store_true", help="Validate all services in the repository")
    parser.add_argument("--report", action="store_true", help="Generate a report for each service")
    parser.add_argument("--report-dir", default="tools/reports/service_structure", help="Directory to store reports")
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
        service_type = detect_service_type(service_path)
        
        if service_type == "unknown":
            print(f"Skipping {service_name}: Unknown service type")
            continue
        
        print(f"Validating {service_name} ({service_type})...")
        compliance, violations = validate_service_structure(service_path, service_type)
        
        # Calculate compliance percentages
        required_total = len(compliance["required"]) + len(violations["required"])
        required_pct = int(len(compliance["required"]) / required_total * 100) if required_total > 0 else 100
        
        recommended_total = len(compliance["recommended"]) + len(violations["recommended"])
        recommended_pct = int(len(compliance["recommended"]) / recommended_total * 100) if recommended_total > 0 else 100
        
        # Print summary
        print(f"  Required: {len(compliance['required'])}/{required_total} ({required_pct}%)")
        print(f"  Recommended: {len(compliance['recommended'])}/{recommended_total} ({recommended_pct}%)")
        print(f"  Optional: {len(compliance['optional'])}")
        
        if violations["required"]:
            print("  Missing required files/directories:")
            for item in sorted(violations["required"]):
                print(f"    - {item}")
        
        # Generate report if requested
        if args.report:
            report = generate_report(service_path, service_type, compliance, violations)
            report_path = report_dir / f"{service_name}_structure_report.md"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"  Report saved to {report_path}")
        
        # Store results
        results[service_name] = {
            "type": service_type,
            "required_pct": required_pct,
            "recommended_pct": recommended_pct,
            "optional_count": len(compliance["optional"]),
            "violations": violations
        }
    
    # Print overall summary
    print("\nOverall Summary:")
    for service_name, result in sorted(results.items()):
        print(f"{service_name}: Required: {result['required_pct']}%, Recommended: {result['recommended_pct']}%")
    
    # Check if any services have required violations
    has_required_violations = any(len(result["violations"]["required"]) > 0 for result in results.values())
    
    return 1 if has_required_violations else 0


if __name__ == "__main__":
    sys.exit(main())
