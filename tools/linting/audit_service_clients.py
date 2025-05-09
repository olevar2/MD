#!/usr/bin/env python3
"""
Script to audit service clients for consistency with the platform's standards.

This script:
1. Scans the codebase for service client implementations
2. Validates clients against the platform's standardized client template
3. Reports compliance and violations
4. Optionally generates a report of findings
"""

import os
import sys
import json
import re
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


# Client implementation patterns
PYTHON_CLIENT_PATTERN = r'class\s+(\w+Client)\s*\('
JS_CLIENT_PATTERN = r'class\s+(\w+Client)\s+extends'

# Required methods and properties for standardized clients
PYTHON_CLIENT_REQUIREMENTS = {
    "base_classes": ["BaseServiceClient"],
    "methods": ["get", "post", "put", "delete", "patch"],
    "properties": ["logger", "config"],
    "error_handling": ["try", "except", "raise"]
}

JS_CLIENT_REQUIREMENTS = {
    "base_classes": ["BaseServiceClient"],
    "methods": ["get", "post", "put", "delete", "patch"],
    "properties": ["logger", "config"],
    "error_handling": ["try", "catch", "throw"]
}


def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/linting
    return Path(__file__).parent.parent.parent


def find_client_files(repo_root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find service client files in the repository.
    
    Args:
        repo_root: Path to the repository root
        
    Returns:
        Tuple of (python_client_files, js_client_files)
    """
    python_client_files = []
    js_client_files = []
    
    # Find Python client files
    for py_file in repo_root.glob("**/*client*.py"):
        if "node_modules" in str(py_file) or "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                if re.search(PYTHON_CLIENT_PATTERN, content) and "BaseServiceClient" in content:
                    python_client_files.append(py_file)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    # Find JavaScript/TypeScript client files
    for js_file in repo_root.glob("**/*[cC]lient*.[jt]s"):
        if "node_modules" in str(js_file):
            continue
        
        try:
            with open(js_file, "r", encoding="utf-8") as f:
                content = f.read()
                if re.search(JS_CLIENT_PATTERN, content) and "BaseServiceClient" in content:
                    js_client_files.append(js_file)
        except Exception as e:
            print(f"Error reading {js_file}: {e}")
    
    return python_client_files, js_client_files


def analyze_python_client(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a Python service client file.
    
    Args:
        file_path: Path to the Python client file
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        "file": str(file_path),
        "language": "python",
        "client_name": None,
        "base_classes": [],
        "methods": [],
        "properties": [],
        "has_error_handling": False,
        "has_logging": False,
        "has_resilience": False,
        "violations": []
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Extract client name
            client_match = re.search(PYTHON_CLIENT_PATTERN, content)
            if client_match:
                result["client_name"] = client_match.group(1)
            
            # Parse the file with ast
            tree = ast.parse(content)
            
            # Find client class
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == result["client_name"]:
                    # Get base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            result["base_classes"].append(base.id)
                        elif isinstance(base, ast.Attribute):
                            result["base_classes"].append(base.attr)
                    
                    # Get methods
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            result["methods"].append(child.name)
                            
                            # Check for error handling
                            for node in ast.walk(child):
                                if isinstance(node, ast.Try):
                                    result["has_error_handling"] = True
                                    break
                    
                    break
            
            # Check for logging
            result["has_logging"] = "logger" in content and ("self.logger" in content or "logging.getLogger" in content)
            
            # Check for resilience patterns
            result["has_resilience"] = any(pattern in content for pattern in ["circuit_breaker", "retry", "timeout", "bulkhead"])
            
            # Check for violations
            for base_class in PYTHON_CLIENT_REQUIREMENTS["base_classes"]:
                if base_class not in result["base_classes"]:
                    result["violations"].append(f"Missing base class: {base_class}")
            
            for method in PYTHON_CLIENT_REQUIREMENTS["methods"]:
                if method not in result["methods"]:
                    result["violations"].append(f"Missing method: {method}")
            
            for property_name in PYTHON_CLIENT_REQUIREMENTS["properties"]:
                if property_name not in content:
                    result["violations"].append(f"Missing property: {property_name}")
            
            if not result["has_error_handling"]:
                result["violations"].append("Missing error handling")
            
            if not result["has_logging"]:
                result["violations"].append("Missing logging")
            
            if not result["has_resilience"]:
                result["violations"].append("Missing resilience patterns")
    
    except Exception as e:
        result["violations"].append(f"Error analyzing file: {e}")
    
    return result


def analyze_js_client(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a JavaScript/TypeScript service client file.
    
    Args:
        file_path: Path to the JavaScript/TypeScript client file
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        "file": str(file_path),
        "language": "javascript" if file_path.suffix == ".js" else "typescript",
        "client_name": None,
        "base_classes": [],
        "methods": [],
        "properties": [],
        "has_error_handling": False,
        "has_logging": False,
        "has_resilience": False,
        "violations": []
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Extract client name
            client_match = re.search(JS_CLIENT_PATTERN, content)
            if client_match:
                result["client_name"] = client_match.group(1)
            
            # Extract base classes
            base_match = re.search(r'extends\s+(\w+)', content)
            if base_match:
                result["base_classes"].append(base_match.group(1))
            
            # Extract methods
            method_matches = re.finditer(r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*{', content)
            for match in method_matches:
                method_name = match.group(1)
                if method_name not in ["constructor", "super"]:
                    result["methods"].append(method_name)
            
            # Check for error handling
            result["has_error_handling"] = "try" in content and "catch" in content
            
            # Check for logging
            result["has_logging"] = "logger" in content and ("this.logger" in content or "console.log" in content)
            
            # Check for resilience patterns
            result["has_resilience"] = any(pattern in content for pattern in ["circuitBreaker", "retry", "timeout", "bulkhead"])
            
            # Check for violations
            for base_class in JS_CLIENT_REQUIREMENTS["base_classes"]:
                if base_class not in result["base_classes"]:
                    result["violations"].append(f"Missing base class: {base_class}")
            
            for method in JS_CLIENT_REQUIREMENTS["methods"]:
                if method not in result["methods"]:
                    result["violations"].append(f"Missing method: {method}")
            
            for property_name in JS_CLIENT_REQUIREMENTS["properties"]:
                if property_name not in content:
                    result["violations"].append(f"Missing property: {property_name}")
            
            if not result["has_error_handling"]:
                result["violations"].append("Missing error handling")
            
            if not result["has_logging"]:
                result["violations"].append("Missing logging")
            
            if not result["has_resilience"]:
                result["violations"].append("Missing resilience patterns")
    
    except Exception as e:
        result["violations"].append(f"Error analyzing file: {e}")
    
    return result


def generate_report(
    client_analyses: List[Dict[str, Any]]
) -> str:
    """
    Generate a report of the audit results.
    
    Args:
        client_analyses: List of client analysis results
        
    Returns:
        Report as a string
    """
    compliant_clients = [c for c in client_analyses if not c["violations"]]
    non_compliant_clients = [c for c in client_analyses if c["violations"]]
    
    total_clients = len(client_analyses)
    compliance_pct = int(len(compliant_clients) / total_clients * 100) if total_clients > 0 else 100
    
    report = [
        "# Service Client Audit Report",
        "",
        "## Compliance Summary",
        "",
        f"- Total Clients: {total_clients}",
        f"- Compliant Clients: {len(compliant_clients)} ({compliance_pct}%)",
        f"- Non-Compliant Clients: {len(non_compliant_clients)} ({100 - compliance_pct}%)",
        "",
    ]
    
    if non_compliant_clients:
        report.extend([
            "## Non-Compliant Clients",
            ""
        ])
        
        for client in sorted(non_compliant_clients, key=lambda c: c["file"]):
            report.extend([
                f"### {client['client_name']}",
                "",
                f"**File:** {client['file']}",
                f"**Language:** {client['language']}",
                "",
                "**Violations:**",
                ""
            ])
            
            for violation in client["violations"]:
                report.append(f"- {violation}")
            
            report.append("")
    
    report.extend([
        "## Compliant Clients",
        ""
    ])
    
    for client in sorted(compliant_clients, key=lambda c: c["file"]):
        report.append(f"- {client['client_name']} ({client['language']}) - {client['file']}")
    
    return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Audit service clients for consistency with platform standards")
    parser.add_argument("--report", action="store_true", help="Generate a report of findings")
    parser.add_argument("--report-path", default="tools/reports/service_client_audit.md", help="Path to save the report")
    args = parser.parse_args()
    
    repo_root = get_repo_root()
    
    # Find client files
    print("Finding service client files...")
    python_client_files, js_client_files = find_client_files(repo_root)
    
    total_clients = len(python_client_files) + len(js_client_files)
    print(f"Found {total_clients} service clients ({len(python_client_files)} Python, {len(js_client_files)} JavaScript/TypeScript)")
    
    # Analyze clients
    print("Analyzing service clients...")
    client_analyses = []
    
    for file_path in python_client_files:
        analysis = analyze_python_client(file_path)
        client_analyses.append(analysis)
    
    for file_path in js_client_files:
        analysis = analyze_js_client(file_path)
        client_analyses.append(analysis)
    
    # Calculate compliance
    compliant_clients = [c for c in client_analyses if not c["violations"]]
    non_compliant_clients = [c for c in client_analyses if c["violations"]]
    
    compliance_pct = int(len(compliant_clients) / total_clients * 100) if total_clients > 0 else 100
    
    # Print summary
    print(f"Compliant Clients: {len(compliant_clients)} ({compliance_pct}%)")
    print(f"Non-Compliant Clients: {len(non_compliant_clients)} ({100 - compliance_pct}%)")
    
    if non_compliant_clients:
        print("\nNon-Compliant Clients:")
        for client in sorted(non_compliant_clients, key=lambda c: c["file"])[:5]:  # Show only first 5
            print(f"  - {client['client_name']} ({client['file']})")
            for violation in client["violations"][:3]:  # Show only first 3 violations
                print(f"    - {violation}")
            if len(client["violations"]) > 3:
                print(f"    - ... and {len(client['violations']) - 3} more violations")
        
        if len(non_compliant_clients) > 5:
            print(f"  - ... and {len(non_compliant_clients) - 5} more non-compliant clients")
    
    # Generate report if requested
    if args.report:
        report = generate_report(client_analyses)
        report_path = repo_root / args.report_path
        
        # Create directory if it doesn't exist
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"\nReport saved to {report_path}")
    
    # Return non-zero exit code if any non-compliant clients
    return 1 if non_compliant_clients else 0


if __name__ == "__main__":
    sys.exit(main())
