#!/usr/bin/env python3
"""
Script to run all validation tools and generate a comprehensive report.

This script:
1. Runs all validation tools (service structure, API endpoints, service clients, error handling)
2. Collects results from each tool
3. Generates a comprehensive report
4. Provides a summary of all validation results
"""

import os
import sys
import subprocess
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any


def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/linting
    return Path(__file__).parent.parent.parent


def run_validation_tool(tool_path: Path, args: List[str]) -> subprocess.CompletedProcess:
    """
    Run a validation tool and return the result.
    
    Args:
        tool_path: Path to the validation tool
        args: Arguments to pass to the tool
        
    Returns:
        CompletedProcess object with the tool's output
    """
    cmd = [sys.executable, str(tool_path)] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def generate_comprehensive_report(
    results: Dict[str, Any],
    report_dir: Path
) -> str:
    """
    Generate a comprehensive report of all validation results.
    
    Args:
        results: Dictionary of validation results
        report_dir: Directory containing individual reports
        
    Returns:
        Path to the comprehensive report
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = [
        "# Forex Trading Platform Validation Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Summary",
        "",
        "| Validation | Status | Compliance |",
        "|------------|--------|------------|",
    ]
    
    for tool_name, result in results.items():
        status = "✅ PASS" if result["exit_code"] == 0 else "❌ FAIL"
        compliance = result.get("compliance", "N/A")
        report.append(f"| {tool_name} | {status} | {compliance} |")
    
    report.extend([
        "",
        "## Detailed Results",
        ""
    ])
    
    for tool_name, result in results.items():
        report.extend([
            f"### {tool_name}",
            "",
            f"**Status:** {'PASS' if result['exit_code'] == 0 else 'FAIL'}",
            f"**Compliance:** {result.get('compliance', 'N/A')}",
            "",
            "**Details:**",
            "",
            "```",
            result["output"],
            "```",
            "",
            f"**Full Report:** [{tool_name} Report]({result['report_path']})",
            ""
        ])
    
    # Write the report
    report_path = report_dir / "comprehensive_validation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    
    return report_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run all validation tools and generate a comprehensive report")
    parser.add_argument("--report-dir", default="tools/reports", help="Directory to store reports")
    args = parser.parse_args()
    
    repo_root = get_repo_root()
    tools_dir = repo_root / "tools" / "linting"
    report_dir = repo_root / args.report_dir
    
    # Create report directory if it doesn't exist
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Define validation tools and their arguments
    validation_tools = {
        "Service Structure": {
            "tool": tools_dir / "validate_service_structure.py",
            "args": ["--all", "--report", f"--report-dir={args.report_dir}/service_structure"],
            "report_path": f"{args.report_dir}/service_structure"
        },
        "API Endpoints": {
            "tool": tools_dir / "validate_api_endpoints.py",
            "args": ["--all", "--report", f"--report-dir={args.report_dir}/api_endpoints"],
            "report_path": f"{args.report_dir}/api_endpoints"
        },
        "Service Clients": {
            "tool": tools_dir / "audit_service_clients.py",
            "args": ["--report", f"--report-path={args.report_dir}/service_client_audit.md"],
            "report_path": f"{args.report_dir}/service_client_audit.md"
        },
        "Error Handling": {
            "tool": tools_dir / "validate_error_handling.py",
            "args": ["--report", f"--report-path={args.report_dir}/error_handling_validation.md"],
            "report_path": f"{args.report_dir}/error_handling_validation.md"
        }
    }
    
    # Run each validation tool
    results = {}
    for tool_name, tool_info in validation_tools.items():
        print(f"Running {tool_name} validation...")
        result = run_validation_tool(tool_info["tool"], tool_info["args"])
        
        # Extract compliance percentage from output
        compliance = "N/A"
        for line in result.stdout.splitlines():
            if "Compliant" in line and "%" in line:
                compliance = line.strip()
                break
        
        results[tool_name] = {
            "exit_code": result.returncode,
            "output": result.stdout,
            "error": result.stderr,
            "compliance": compliance,
            "report_path": tool_info["report_path"]
        }
        
        print(f"  Status: {'PASS' if result.returncode == 0 else 'FAIL'}")
        print(f"  Compliance: {compliance}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report_path = generate_comprehensive_report(results, report_dir)
    print(f"Comprehensive report saved to {report_path}")
    
    # Determine overall status
    overall_status = all(result["exit_code"] == 0 for result in results.values())
    
    print("\nOverall Validation Status:", "PASS" if overall_status else "FAIL")
    
    return 0 if overall_status else 1


if __name__ == "__main__":
    sys.exit(main())
