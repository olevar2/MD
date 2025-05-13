#!/usr/bin/env python3
"""
Visualize Service Dependencies

This script creates a visualization of service dependencies using the dependency report.
"""

import json
import os
import sys
from typing import Dict, List, Any
import argparse

def load_report(report_path: str) -> Dict[str, Any]:
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
        print(f"Error loading report: {e}")
        sys.exit(1)

def generate_dot_graph(data: Dict[str, Any]) -> str:
    """
    Generate a DOT graph representation of service dependencies.
    
    Args:
        data: Dependency report data
        
    Returns:
        DOT graph representation
    """
    service_dependencies = data.get('service_dependencies', {})
    circular_dependencies = data.get('circular_dependencies', [])
    
    # Create a set of circular dependency pairs for easy lookup
    circular_pairs = set()
    for service1, service2 in circular_dependencies:
        circular_pairs.add((service1, service2))
        circular_pairs.add((service2, service1))
    
    # Start DOT graph
    dot = [
        'digraph "Service Dependencies" {',
        '  rankdir=LR;',
        '  node [shape=box, style=filled, fillcolor=lightblue];',
        '  edge [color=black];'
    ]
    
    # Add nodes (services)
    for service in data.get('services', {}):
        # Skip duplicate services (different naming conventions)
        if service.replace('-', '_') in data.get('services', {}) and service.replace('-', '_') != service:
            continue
        
        # Add node
        dot.append(f'  "{service}" [label="{service}"];')
    
    # Add edges (dependencies)
    for service, dependencies in service_dependencies.items():
        # Skip duplicate services (different naming conventions)
        if service.replace('-', '_') in service_dependencies and service.replace('-', '_') != service:
            continue
        
        for dep in dependencies:
            # Skip duplicate services (different naming conventions)
            if dep.replace('-', '_') in service_dependencies and dep.replace('-', '_') != dep:
                continue
            
            # Skip self-dependencies
            if service == dep or service.replace('-', '_') == dep or service == dep.replace('-', '_'):
                continue
            
            # Check if this is a circular dependency
            if (service, dep) in circular_pairs:
                dot.append(f'  "{service}" -> "{dep}" [color=red, penwidth=2.0];')
            else:
                dot.append(f'  "{service}" -> "{dep}";')
    
    # End DOT graph
    dot.append('}')
    
    return '\n'.join(dot)

def generate_mermaid_graph(data: Dict[str, Any]) -> str:
    """
    Generate a Mermaid graph representation of service dependencies.
    
    Args:
        data: Dependency report data
        
    Returns:
        Mermaid graph representation
    """
    service_dependencies = data.get('service_dependencies', {})
    circular_dependencies = data.get('circular_dependencies', [])
    
    # Create a set of circular dependency pairs for easy lookup
    circular_pairs = set()
    for service1, service2 in circular_dependencies:
        circular_pairs.add((service1, service2))
        circular_pairs.add((service2, service1))
    
    # Start Mermaid graph
    mermaid = [
        '```mermaid',
        'graph LR;'
    ]
    
    # Add nodes (services)
    for service in data.get('services', {}):
        # Skip duplicate services (different naming conventions)
        if service.replace('-', '_') in data.get('services', {}) and service.replace('-', '_') != service:
            continue
        
        # Add node
        mermaid.append(f'  {service}["{service}"];')
    
    # Add edges (dependencies)
    for service, dependencies in service_dependencies.items():
        # Skip duplicate services (different naming conventions)
        if service.replace('-', '_') in service_dependencies and service.replace('-', '_') != service:
            continue
        
        for dep in dependencies:
            # Skip duplicate services (different naming conventions)
            if dep.replace('-', '_') in service_dependencies and dep.replace('-', '_') != dep:
                continue
            
            # Skip self-dependencies
            if service == dep or service.replace('-', '_') == dep or service == dep.replace('-', '_'):
                continue
            
            # Check if this is a circular dependency
            if (service, dep) in circular_pairs:
                mermaid.append(f'  {service} -->|depends on| {dep};')
                mermaid.append(f'  {dep} -->|depends on| {service};')
            else:
                mermaid.append(f'  {service} -->|depends on| {dep};')
    
    # End Mermaid graph
    mermaid.append('```')
    
    return '\n'.join(mermaid)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize service dependencies")
    parser.add_argument(
        "--report-path",
        default="tools/output/dependency-report.json",
        help="Path to the dependency report"
    )
    parser.add_argument(
        "--output-dir",
        default="tools/output",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--format",
        choices=["dot", "mermaid", "both"],
        default="both",
        help="Output format"
    )
    args = parser.parse_args()
    
    # Load report
    data = load_report(args.report_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate and save DOT graph
    if args.format in ["dot", "both"]:
        dot_graph = generate_dot_graph(data)
        dot_path = os.path.join(args.output_dir, "service_dependencies.dot")
        with open(dot_path, 'w', encoding='utf-8') as f:
            f.write(dot_graph)
        print(f"DOT graph saved to {dot_path}")
        
        # Try to generate PNG if Graphviz is installed
        try:
            import subprocess
            png_path = os.path.join(args.output_dir, "service_dependencies.png")
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True)
            print(f"PNG graph saved to {png_path}")
        except Exception as e:
            print(f"Could not generate PNG: {e}")
            print("To generate a PNG, install Graphviz and run:")
            print(f"  dot -Tpng {dot_path} -o {os.path.join(args.output_dir, 'service_dependencies.png')}")
    
    # Generate and save Mermaid graph
    if args.format in ["mermaid", "both"]:
        mermaid_graph = generate_mermaid_graph(data)
        mermaid_path = os.path.join(args.output_dir, "service_dependencies.mermaid.md")
        with open(mermaid_path, 'w', encoding='utf-8') as f:
            f.write(mermaid_graph)
        print(f"Mermaid graph saved to {mermaid_path}")

if __name__ == "__main__":
    main()
