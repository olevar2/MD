#!/usr/bin/env python
"""
Deprecation Report Generator

This script generates reports on usage of deprecated modules.
It helps track migration progress and identify areas that need attention.

Usage:
    python deprecation_report.py [--format {text,json,html,csv}] [--output FILE]

Options:
    --format FORMAT  Output format (text, json, html, csv) [default: text]
    --output FILE    Output file [default: stdout]
"""

import sys
import os
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional
import csv
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import deprecation monitor
from analysis_engine.core.deprecation_monitor import get_usage_report


def format_text(report: Dict) -> str:
    """Format report as text."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"DEPRECATION USAGE REPORT - Generated at {report['generated_at']}")
    lines.append("=" * 80)
    lines.append(f"Total usages: {report['total_usages']}")
    lines.append("")
    
    for module_name, module_data in report.get("modules", {}).items():
        lines.append(f"MODULE: {module_name}")
        lines.append("-" * 80)
        lines.append(f"Total usages: {module_data['total_usages']}")
        lines.append(f"Unique locations: {module_data['unique_locations']}")
        lines.append("")
        
        # Sort usages by count (descending)
        usages = sorted(module_data["usages"], key=lambda u: u["count"], reverse=True)
        
        for i, usage in enumerate(usages, 1):
            lines.append(f"{i}. File: {usage['caller_file']}")
            lines.append(f"   Line: {usage['caller_line']}")
            lines.append(f"   Function: {usage['caller_function']}")
            lines.append(f"   Count: {usage['count']}")
            lines.append(f"   Last seen: {usage['last_seen']}")
            lines.append("")
    
    return "\n".join(lines)


def format_html(report: Dict) -> str:
    """Format report as HTML."""
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html>")
    html.append("<head>")
    html.append("  <title>Deprecation Usage Report</title>")
    html.append("  <style>")
    html.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
    html.append("    h1 { color: #333; }")
    html.append("    h2 { color: #666; margin-top: 30px; }")
    html.append("    table { border-collapse: collapse; width: 100%; margin-top: 10px; }")
    html.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html.append("    th { background-color: #f2f2f2; }")
    html.append("    tr:nth-child(even) { background-color: #f9f9f9; }")
    html.append("    .summary { margin: 20px 0; padding: 10px; background-color: #eef; }")
    html.append("    .high-usage { color: red; font-weight: bold; }")
    html.append("    .medium-usage { color: orange; }")
    html.append("    .low-usage { color: green; }")
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")
    html.append(f"  <h1>Deprecation Usage Report</h1>")
    html.append(f"  <p>Generated at: {report['generated_at']}</p>")
    
    html.append("  <div class='summary'>")
    html.append(f"    <h2>Summary</h2>")
    html.append(f"    <p>Total usages: {report['total_usages']}</p>")
    
    # Add module summary
    html.append("    <table>")
    html.append("      <tr><th>Module</th><th>Total Usages</th><th>Unique Locations</th></tr>")
    for module_name, module_data in report.get("modules", {}).items():
        html.append(f"      <tr>")
        html.append(f"        <td>{module_name}</td>")
        html.append(f"        <td>{module_data['total_usages']}</td>")
        html.append(f"        <td>{module_data['unique_locations']}</td>")
        html.append(f"      </tr>")
    html.append("    </table>")
    html.append("  </div>")
    
    # Add detailed usage information
    for module_name, module_data in report.get("modules", {}).items():
        html.append(f"  <h2>Module: {module_name}</h2>")
        
        # Sort usages by count (descending)
        usages = sorted(module_data["usages"], key=lambda u: u["count"], reverse=True)
        
        html.append("  <table>")
        html.append("    <tr><th>File</th><th>Line</th><th>Function</th><th>Count</th><th>Last Seen</th></tr>")
        
        for usage in usages:
            # Determine usage level for styling
            usage_class = "low-usage"
            if usage["count"] > 100:
                usage_class = "high-usage"
            elif usage["count"] > 20:
                usage_class = "medium-usage"
                
            html.append(f"    <tr>")
            html.append(f"      <td>{usage['caller_file']}</td>")
            html.append(f"      <td>{usage['caller_line']}</td>")
            html.append(f"      <td>{usage['caller_function']}</td>")
            html.append(f"      <td class='{usage_class}'>{usage['count']}</td>")
            html.append(f"      <td>{usage['last_seen']}</td>")
            html.append(f"    </tr>")
        
        html.append("  </table>")
    
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)


def format_csv(report: Dict) -> str:
    """Format report as CSV."""
    output = []
    
    # Create CSV writer
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["Module", "File", "Line", "Function", "Count", "Last Seen"])
    
    # Write data
    for module_name, module_data in report.get("modules", {}).items():
        for usage in module_data["usages"]:
            writer.writerow([
                module_name,
                usage["caller_file"],
                usage["caller_line"],
                usage["caller_function"],
                usage["count"],
                usage["last_seen"]
            ])
    
    return "".join(output)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate deprecation usage report")
    parser.add_argument("--format", choices=["text", "json", "html", "csv"], default="text",
                        help="Output format (text, json, html, csv)")
    parser.add_argument("--output", help="Output file (default: stdout)")
    args = parser.parse_args()
    
    # Get report data
    report = get_usage_report()
    
    # Format report
    if args.format == "text":
        output = format_text(report)
    elif args.format == "json":
        output = json.dumps(report, indent=2)
    elif args.format == "html":
        output = format_html(report)
    elif args.format == "csv":
        output = format_csv(report)
    else:
        output = format_text(report)
    
    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
