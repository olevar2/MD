#!/usr/bin/env python3
"""
Script to validate error handling consistency across language boundaries.

This script:
1. Scans the codebase for error handling implementations
2. Validates error handling against the platform's standards
3. Checks for consistency between Python and JavaScript/TypeScript error handling
4. Reports compliance and violations
5. Optionally generates a report of findings
"""

import os
import sys
import json
import re
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


# Error handling patterns
PYTHON_ERROR_BRIDGE_PATTERN = r'(convert_to_js_error|convert_from_js_error|handle_js_error_response)'
JS_ERROR_BRIDGE_PATTERN = r'(convertToPythonError|convertFromPythonError|handlePythonErrorResponse)'

# Required error classes and methods
PYTHON_ERROR_REQUIREMENTS = {
    "base_classes": ["ForexTradingPlatformError"],
    "domain_errors": [
        "ConfigurationError", 
        "DataError", 
        "ServiceError", 
        "TradingError", 
        "ModelError", 
        "SecurityError", 
        "ResilienceError"
    ],
    "bridge_methods": [
        "convert_to_js_error",
        "convert_from_js_error",
        "handle_js_error_response"
    ]
}

JS_ERROR_REQUIREMENTS = {
    "base_classes": ["ForexTradingPlatformError"],
    "domain_errors": [
        "ConfigurationError", 
        "DataError", 
        "ServiceError", 
        "TradingError", 
        "ModelError", 
        "SecurityError", 
        "ResilienceError"
    ],
    "bridge_methods": [
        "convertToPythonError",
        "convertFromPythonError",
        "handlePythonErrorResponse"
    ]
}


def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/linting
    return Path(__file__).parent.parent.parent


def find_error_handling_files(repo_root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find error handling files in the repository.
    
    Args:
        repo_root: Path to the repository root
        
    Returns:
        Tuple of (python_error_files, js_error_files)
    """
    python_error_files = []
    js_error_files = []
    
    # Find Python error handling files
    for py_file in repo_root.glob("**/*error*.py"):
        if "node_modules" in str(py_file) or "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                if "ForexTradingPlatformError" in content or re.search(PYTHON_ERROR_BRIDGE_PATTERN, content):
                    python_error_files.append(py_file)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    # Find JavaScript/TypeScript error handling files
    for js_file in repo_root.glob("**/*error*.[jt]s"):
        if "node_modules" in str(js_file):
            continue
        
        try:
            with open(js_file, "r", encoding="utf-8") as f:
                content = f.read()
                if "ForexTradingPlatformError" in content or re.search(JS_ERROR_BRIDGE_PATTERN, content):
                    js_error_files.append(js_file)
        except Exception as e:
            print(f"Error reading {js_file}: {e}")
    
    return python_error_files, js_error_files


def analyze_python_error_handling(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a Python error handling file.
    
    Args:
        file_path: Path to the Python error handling file
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        "file": str(file_path),
        "language": "python",
        "error_classes": [],
        "bridge_methods": [],
        "has_error_mapping": False,
        "has_correlation_id": False,
        "violations": []
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Parse the file with ast
            tree = ast.parse(content)
            
            # Find error classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and "Error" in node.name:
                    result["error_classes"].append(node.name)
            
            # Find bridge methods
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and any(method in node.name for method in PYTHON_ERROR_REQUIREMENTS["bridge_methods"]):
                    result["bridge_methods"].append(node.name)
            
            # Check for error mapping
            result["has_error_mapping"] = "PYTHON_TO_JS_ERROR_MAPPING" in content or "JS_TO_PYTHON_ERROR_MAPPING" in content
            
            # Check for correlation ID
            result["has_correlation_id"] = "correlation_id" in content
            
            # Check for violations
            for base_class in PYTHON_ERROR_REQUIREMENTS["base_classes"]:
                if base_class not in result["error_classes"] and base_class not in content:
                    result["violations"].append(f"Missing base error class: {base_class}")
            
            for domain_error in PYTHON_ERROR_REQUIREMENTS["domain_errors"]:
                if domain_error not in result["error_classes"] and domain_error not in content:
                    result["violations"].append(f"Missing domain error class: {domain_error}")
            
            # Only check bridge methods if this is an error bridge file
            if "bridge" in file_path.name.lower() or "convert" in content:
                for method in PYTHON_ERROR_REQUIREMENTS["bridge_methods"]:
                    if method not in result["bridge_methods"] and method not in content:
                        result["violations"].append(f"Missing bridge method: {method}")
                
                if not result["has_error_mapping"]:
                    result["violations"].append("Missing error mapping between Python and JavaScript")
                
                if not result["has_correlation_id"]:
                    result["violations"].append("Missing correlation ID support")
    
    except Exception as e:
        result["violations"].append(f"Error analyzing file: {e}")
    
    return result


def analyze_js_error_handling(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a JavaScript/TypeScript error handling file.
    
    Args:
        file_path: Path to the JavaScript/TypeScript error handling file
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        "file": str(file_path),
        "language": "javascript" if file_path.suffix == ".js" else "typescript",
        "error_classes": [],
        "bridge_methods": [],
        "has_error_mapping": False,
        "has_correlation_id": False,
        "violations": []
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Extract error classes
            class_matches = re.finditer(r'class\s+(\w+Error)\s+extends', content)
            for match in class_matches:
                result["error_classes"].append(match.group(1))
            
            # Extract bridge methods
            method_matches = re.finditer(r'(?:export\s+)?(?:function|const)\s+(\w+)', content)
            for match in method_matches:
                method_name = match.group(1)
                if any(bridge_method in method_name for bridge_method in JS_ERROR_REQUIREMENTS["bridge_methods"]):
                    result["bridge_methods"].append(method_name)
            
            # Check for error mapping
            result["has_error_mapping"] = "JS_TO_PYTHON_ERROR_MAPPING" in content or "PYTHON_TO_JS_ERROR_MAPPING" in content
            
            # Check for correlation ID
            result["has_correlation_id"] = "correlationId" in content
            
            # Check for violations
            for base_class in JS_ERROR_REQUIREMENTS["base_classes"]:
                if base_class not in result["error_classes"] and base_class not in content:
                    result["violations"].append(f"Missing base error class: {base_class}")
            
            for domain_error in JS_ERROR_REQUIREMENTS["domain_errors"]:
                if domain_error not in result["error_classes"] and domain_error not in content:
                    result["violations"].append(f"Missing domain error class: {domain_error}")
            
            # Only check bridge methods if this is an error bridge file
            if "bridge" in file_path.name.lower() or "convert" in content:
                for method in JS_ERROR_REQUIREMENTS["bridge_methods"]:
                    if method not in result["bridge_methods"] and method not in content:
                        result["violations"].append(f"Missing bridge method: {method}")
                
                if not result["has_error_mapping"]:
                    result["violations"].append("Missing error mapping between JavaScript and Python")
                
                if not result["has_correlation_id"]:
                    result["violations"].append("Missing correlation ID support")
    
    except Exception as e:
        result["violations"].append(f"Error analyzing file: {e}")
    
    return result


def check_cross_language_consistency(
    python_analyses: List[Dict[str, Any]],
    js_analyses: List[Dict[str, Any]]
) -> List[str]:
    """
    Check for consistency between Python and JavaScript error handling.
    
    Args:
        python_analyses: List of Python error handling analysis results
        js_analyses: List of JavaScript error handling analysis results
        
    Returns:
        List of consistency issues
    """
    consistency_issues = []
    
    # Extract all error classes
    python_error_classes = set()
    for analysis in python_analyses:
        python_error_classes.update(analysis["error_classes"])
    
    js_error_classes = set()
    for analysis in js_analyses:
        js_error_classes.update(analysis["error_classes"])
    
    # Check for missing error classes in either language
    for error_class in PYTHON_ERROR_REQUIREMENTS["domain_errors"]:
        if error_class in python_error_classes and error_class not in js_error_classes:
            consistency_issues.append(f"Error class {error_class} exists in Python but not in JavaScript")
    
    for error_class in JS_ERROR_REQUIREMENTS["domain_errors"]:
        if error_class in js_error_classes and error_class not in python_error_classes:
            consistency_issues.append(f"Error class {error_class} exists in JavaScript but not in Python")
    
    # Check for bridge methods
    python_bridge_methods = set()
    for analysis in python_analyses:
        python_bridge_methods.update(analysis["bridge_methods"])
    
    js_bridge_methods = set()
    for analysis in js_analyses:
        js_bridge_methods.update(analysis["bridge_methods"])
    
    if not python_bridge_methods:
        consistency_issues.append("No Python error bridge methods found")
    
    if not js_bridge_methods:
        consistency_issues.append("No JavaScript error bridge methods found")
    
    # Check for error mapping
    python_has_mapping = any(analysis["has_error_mapping"] for analysis in python_analyses)
    js_has_mapping = any(analysis["has_error_mapping"] for analysis in js_analyses)
    
    if not python_has_mapping:
        consistency_issues.append("No Python to JavaScript error mapping found")
    
    if not js_has_mapping:
        consistency_issues.append("No JavaScript to Python error mapping found")
    
    # Check for correlation ID support
    python_has_correlation_id = any(analysis["has_correlation_id"] for analysis in python_analyses)
    js_has_correlation_id = any(analysis["has_correlation_id"] for analysis in js_analyses)
    
    if not python_has_correlation_id:
        consistency_issues.append("No correlation ID support in Python error handling")
    
    if not js_has_correlation_id:
        consistency_issues.append("No correlation ID support in JavaScript error handling")
    
    return consistency_issues


def generate_report(
    python_analyses: List[Dict[str, Any]],
    js_analyses: List[Dict[str, Any]],
    consistency_issues: List[str]
) -> str:
    """
    Generate a report of the validation results.
    
    Args:
        python_analyses: List of Python error handling analysis results
        js_analyses: List of JavaScript error handling analysis results
        consistency_issues: List of cross-language consistency issues
        
    Returns:
        Report as a string
    """
    python_compliant = [a for a in python_analyses if not a["violations"]]
    python_non_compliant = [a for a in python_analyses if a["violations"]]
    
    js_compliant = [a for a in js_analyses if not a["violations"]]
    js_non_compliant = [a for a in js_analyses if a["violations"]]
    
    total_files = len(python_analyses) + len(js_analyses)
    compliant_files = len(python_compliant) + len(js_compliant)
    compliance_pct = int(compliant_files / total_files * 100) if total_files > 0 else 100
    
    report = [
        "# Error Handling Validation Report",
        "",
        "## Compliance Summary",
        "",
        f"- Total Files: {total_files}",
        f"- Compliant Files: {compliant_files} ({compliance_pct}%)",
        f"- Non-Compliant Files: {total_files - compliant_files} ({100 - compliance_pct}%)",
        f"- Python Files: {len(python_analyses)} ({len(python_compliant)} compliant, {len(python_non_compliant)} non-compliant)",
        f"- JavaScript/TypeScript Files: {len(js_analyses)} ({len(js_compliant)} compliant, {len(js_non_compliant)} non-compliant)",
        "",
    ]
    
    if consistency_issues:
        report.extend([
            "## Cross-Language Consistency Issues",
            ""
        ])
        
        for issue in consistency_issues:
            report.append(f"- {issue}")
        
        report.append("")
    
    if python_non_compliant:
        report.extend([
            "## Non-Compliant Python Files",
            ""
        ])
        
        for analysis in sorted(python_non_compliant, key=lambda a: a["file"]):
            report.extend([
                f"### {os.path.basename(analysis['file'])}",
                "",
                f"**File:** {analysis['file']}",
                "",
                "**Violations:**",
                ""
            ])
            
            for violation in analysis["violations"]:
                report.append(f"- {violation}")
            
            report.append("")
    
    if js_non_compliant:
        report.extend([
            "## Non-Compliant JavaScript/TypeScript Files",
            ""
        ])
        
        for analysis in sorted(js_non_compliant, key=lambda a: a["file"]):
            report.extend([
                f"### {os.path.basename(analysis['file'])}",
                "",
                f"**File:** {analysis['file']}",
                f"**Language:** {analysis['language']}",
                "",
                "**Violations:**",
                ""
            ])
            
            for violation in analysis["violations"]:
                report.append(f"- {violation}")
            
            report.append("")
    
    return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate error handling consistency across language boundaries")
    parser.add_argument("--report", action="store_true", help="Generate a report of findings")
    parser.add_argument("--report-path", default="tools/reports/error_handling_validation.md", help="Path to save the report")
    args = parser.parse_args()
    
    repo_root = get_repo_root()
    
    # Find error handling files
    print("Finding error handling files...")
    python_error_files, js_error_files = find_error_handling_files(repo_root)
    
    total_files = len(python_error_files) + len(js_error_files)
    print(f"Found {total_files} error handling files ({len(python_error_files)} Python, {len(js_error_files)} JavaScript/TypeScript)")
    
    # Analyze files
    print("Analyzing error handling files...")
    python_analyses = []
    js_analyses = []
    
    for file_path in python_error_files:
        analysis = analyze_python_error_handling(file_path)
        python_analyses.append(analysis)
    
    for file_path in js_error_files:
        analysis = analyze_js_error_handling(file_path)
        js_analyses.append(analysis)
    
    # Check cross-language consistency
    print("Checking cross-language consistency...")
    consistency_issues = check_cross_language_consistency(python_analyses, js_analyses)
    
    # Calculate compliance
    python_compliant = [a for a in python_analyses if not a["violations"]]
    python_non_compliant = [a for a in python_analyses if a["violations"]]
    
    js_compliant = [a for a in js_analyses if not a["violations"]]
    js_non_compliant = [a for a in js_analyses if a["violations"]]
    
    total_compliant = len(python_compliant) + len(js_compliant)
    compliance_pct = int(total_compliant / total_files * 100) if total_files > 0 else 100
    
    # Print summary
    print(f"Compliant Files: {total_compliant} ({compliance_pct}%)")
    print(f"Non-Compliant Files: {total_files - total_compliant} ({100 - compliance_pct}%)")
    print(f"Python Files: {len(python_analyses)} ({len(python_compliant)} compliant, {len(python_non_compliant)} non-compliant)")
    print(f"JavaScript/TypeScript Files: {len(js_analyses)} ({len(js_compliant)} compliant, {len(js_non_compliant)} non-compliant)")
    
    if consistency_issues:
        print("\nCross-Language Consistency Issues:")
        for issue in consistency_issues:
            print(f"  - {issue}")
    
    # Generate report if requested
    if args.report:
        report = generate_report(python_analyses, js_analyses, consistency_issues)
        report_path = repo_root / args.report_path
        
        # Create directory if it doesn't exist
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"\nReport saved to {report_path}")
    
    # Return non-zero exit code if any non-compliant files or consistency issues
    return 1 if (total_files - total_compliant > 0 or consistency_issues) else 0


if __name__ == "__main__":
    sys.exit(main())
