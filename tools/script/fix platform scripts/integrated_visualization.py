#!/usr/bin/env python3
"""
Forex Trading Platform Integrated Visualization

This script creates integrated visualizations of the forex trading platform architecture
by combining the results from various analysis tools. It generates HTML reports with
interactive visualizations to help understand the project structure.

Usage:
python integrated_visualization.py [--project-root <project_root>] [--output-dir <output_dir>]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import base64
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output\integrated"

# HTML template for the report
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forex Trading Platform Architecture Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #0066cc;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .visualization {
            margin-top: 20px;
            text-align: center;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }
        .stats {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            flex: 1;
            min-width: 200px;
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin-top: 0;
            color: #0066cc;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #0066cc;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #0066cc;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background-color: #eee;
            cursor: pointer;
            border: 1px solid #ddd;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }
        .tab.active {
            background-color: #0066cc;
            color: white;
        }
        .tab-content {
            display: none;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 0 5px 5px 5px;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Forex Trading Platform Architecture Analysis</h1>
        <p>Generated on: {generation_date}</p>
        
        <div class="section">
            <h2>Project Overview</h2>
            <div class="stats">
                <div class="stat-card">
                    <h3>Services</h3>
                    <div class="stat-value">{service_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Files</h3>
                    <div class="stat-value">{file_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Dependencies</h3>
                    <div class="stat-value">{dependency_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Circular Dependencies</h3>
                    <div class="stat-value">{circular_count}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Service Dependencies</h2>
            <div class="visualization">
                <img src="data:image/png;base64,{service_dependencies_image}" alt="Service Dependencies">
            </div>
        </div>
        
        <div class="section">
            <h2>Circular Dependencies</h2>
            <table>
                <tr>
                    <th>Service 1</th>
                    <th>Service 2</th>
                    <th>Suggested Fix</th>
                </tr>
                {circular_dependencies_rows}
            </table>
        </div>
        
        <div class="section">
            <h2>Service Analysis</h2>
            <div class="tabs">
                {service_tabs}
            </div>
            {service_tab_contents}
        </div>
        
        <div class="section">
            <h2>Duplicate Code</h2>
            <p>Found {duplicate_groups_count} groups of duplicate code across the codebase.</p>
            <table>
                <tr>
                    <th>Group</th>
                    <th>Files</th>
                    <th>Similarity</th>
                    <th>Lines</th>
                </tr>
                {duplicate_code_rows}
            </table>
        </div>
        
        <div class="footer">
            <p>Generated using Forex Trading Platform Architecture Analysis Tools</p>
        </div>
    </div>
    
    <script>
        // Tab functionality
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs and tab contents
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                tab.classList.add('active');
                document.getElementById(tab.getAttribute('data-tab')).classList.add('active');
            });
        });
        
        // Activate first tab by default
        const firstTab = document.querySelector('.tab');
        if (firstTab) {
            firstTab.click();
        }
    </script>
</body>
</html>
"""

def load_dependency_report(report_path: str) -> Dict[str, Any]:
    """
    Load the dependency report.
    
    Args:
        report_path: Path to the dependency report
        
    Returns:
        Dependency report data
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading dependency report: {e}")
        return {}

def load_duplicate_code_report(report_path: str) -> Dict[str, Any]:
    """
    Load the duplicate code report.
    
    Args:
        report_path: Path to the duplicate code report
        
    Returns:
        Duplicate code report data
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading duplicate code report: {e}")
        return {}

def load_circular_dependencies_report(report_path: str) -> Dict[str, Any]:
    """
    Load the circular dependencies report.
    
    Args:
        report_path: Path to the circular dependencies report
        
    Returns:
        Circular dependencies report data
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading circular dependencies report: {e}")
        return {}

def image_to_base64(image_path: str) -> str:
    """
    Convert an image to base64.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Base64-encoded image
    """
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return ""

def generate_service_tabs(services: List[str]) -> str:
    """
    Generate HTML for service tabs.
    
    Args:
        services: List of service names
        
    Returns:
        HTML for service tabs
    """
    tabs = []
    for i, service in enumerate(services):
        active = ' active' if i == 0 else ''
        tabs.append(f'<div class="tab{active}" data-tab="service-{i}">{service}</div>')
    
    return '\n'.join(tabs)

def generate_service_tab_contents(services: List[str], project_root: str, output_dir: str) -> str:
    """
    Generate HTML for service tab contents.
    
    Args:
        services: List of service names
        project_root: Root directory of the project
        output_dir: Directory containing output files
        
    Returns:
        HTML for service tab contents
    """
    contents = []
    
    for i, service in enumerate(services):
        active = ' active' if i == 0 else ''
        
        # Check if service has a module dependency visualization
        module_image_path = os.path.join(output_dir, 'pydeps', service, f"{service}_modules.png")
        module_image_html = ""
        if os.path.exists(module_image_path):
            module_image_base64 = image_to_base64(module_image_path)
            module_image_html = f"""
            <h3>Module Dependencies</h3>
            <div class="visualization">
                <img src="data:image/png;base64,{module_image_base64}" alt="{service} Module Dependencies">
            </div>
            """
        
        # Check if service has a function dependency visualization
        function_image_path = os.path.join(output_dir, 'pyan', service, f"{service}_functions.png")
        function_image_html = ""
        if os.path.exists(function_image_path):
            function_image_base64 = image_to_base64(function_image_path)
            function_image_html = f"""
            <h3>Function Dependencies</h3>
            <div class="visualization">
                <img src="data:image/png;base64,{function_image_base64}" alt="{service} Function Dependencies">
            </div>
            """
        
        # Generate service stats
        service_path = os.path.join(project_root, service)
        file_count = 0
        line_count = 0
        
        if os.path.exists(service_path):
            for root, _, files in os.walk(service_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        file_count += 1
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                line_count += len(f.readlines())
                        except Exception:
                            pass
        
        contents.append(f"""
        <div id="service-{i}" class="tab-content{active}">
            <h3>{service}</h3>
            <div class="stats">
                <div class="stat-card">
                    <h3>Files</h3>
                    <div class="stat-value">{file_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Lines of Code</h3>
                    <div class="stat-value">{line_count}</div>
                </div>
            </div>
            {module_image_html}
            {function_image_html}
        </div>
        """)
    
    return '\n'.join(contents)

def generate_circular_dependencies_rows(circular_dependencies: List[Dict[str, Any]]) -> str:
    """
    Generate HTML for circular dependencies rows.
    
    Args:
        circular_dependencies: List of circular dependencies
        
    Returns:
        HTML for circular dependencies rows
    """
    rows = []
    
    for i, dep in enumerate(circular_dependencies):
        service1 = dep.get('service1', '')
        service2 = dep.get('service2', '')
        suggestions = dep.get('suggestions', [])
        
        suggestion_html = '<ul>'
        for suggestion in suggestions[:3]:  # Limit to 3 suggestions
            suggestion_html += f'<li>{suggestion}</li>'
        suggestion_html += '</ul>'
        
        rows.append(f"""
        <tr>
            <td>{service1}</td>
            <td>{service2}</td>
            <td>{suggestion_html}</td>
        </tr>
        """)
    
    return '\n'.join(rows)

def generate_duplicate_code_rows(duplicates: List[Dict[str, Any]]) -> str:
    """
    Generate HTML for duplicate code rows.
    
    Args:
        duplicates: List of duplicate code groups
        
    Returns:
        HTML for duplicate code rows
    """
    rows = []
    
    for i, group in enumerate(duplicates[:10]):  # Limit to 10 groups
        blocks = group.get('blocks', [])
        similarity = group.get('similarity', 0.0)
        
        files = []
        total_lines = 0
        
        for block in blocks:
            file_path = block.get('file', '')
            start_line = block.get('start_line', 0)
            end_line = block.get('end_line', 0)
            
            file_name = os.path.basename(file_path)
            files.append(f"{file_name} (lines {start_line}-{end_line})")
            total_lines += end_line - start_line + 1
        
        files_html = '<ul>'
        for file in files:
            files_html += f'<li>{file}</li>'
        files_html += '</ul>'
        
        rows.append(f"""
        <tr>
            <td>Group {i+1}</td>
            <td>{files_html}</td>
            <td>{similarity:.2f}</td>
            <td>{total_lines}</td>
        </tr>
        """)
    
    return '\n'.join(rows)

def generate_report(project_root: str, output_dir: str) -> None:
    """
    Generate an integrated visualization report.
    
    Args:
        project_root: Root directory of the project
        output_dir: Directory to save output files
    """
    logger.info("Generating integrated visualization report...")
    
    # Load dependency report
    dependency_report_path = os.path.join(output_dir, 'dependency-report.json')
    dependency_report = load_dependency_report(dependency_report_path)
    
    # Load duplicate code report
    duplicate_code_report_path = os.path.join(output_dir, 'duplicate-code-report.json')
    duplicate_code_report = load_duplicate_code_report(duplicate_code_report_path)
    
    # Load circular dependencies report
    circular_dependencies_report_path = os.path.join(output_dir, 'circular_dependencies_report.json')
    circular_dependencies_report = load_circular_dependencies_report(circular_dependencies_report_path)
    
    # Get service dependencies image
    service_dependencies_image_path = os.path.join(output_dir, 'service_dependencies.png')
    service_dependencies_image = image_to_base64(service_dependencies_image_path)
    
    # Get services
    services = list(dependency_report.get('services', {}).keys())
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        service_count=len(services),
        file_count=sum(1 for _ in os.walk(project_root) for _ in filter(lambda x: x.endswith('.py'), _[2])),
        dependency_count=sum(len(deps) for deps in dependency_report.get('service_dependencies', {}).values()),
        circular_count=len(dependency_report.get('circular_dependencies', [])),
        service_dependencies_image=service_dependencies_image,
        circular_dependencies_rows=generate_circular_dependencies_rows(
            circular_dependencies_report.get('circular_dependencies', [])
        ),
        service_tabs=generate_service_tabs(services),
        service_tab_contents=generate_service_tab_contents(services, project_root, output_dir),
        duplicate_groups_count=duplicate_code_report.get('duplicate_groups', 0),
        duplicate_code_rows=generate_duplicate_code_rows(
            duplicate_code_report.get('duplicates', [])
        )
    )
    
    # Save HTML report
    report_path = os.path.join(output_dir, 'integrated_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Integrated visualization report saved to {report_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate integrated visualizations")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate report
    generate_report(args.project_root, args.output_dir)

if __name__ == "__main__":
    main()
