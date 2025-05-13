#!/usr/bin/env python3
"""
Visualize Dependencies in the Forex Trading Platform

This script visualizes the dependencies between services in the forex trading platform.
It generates dependency graphs in various formats (DOT, PNG, Mermaid).

Usage:
    python visualize_dependencies.py [--input-file INPUT_FILE] [--output-dir OUTPUT_DIR]

Options:
    --input-file INPUT_FILE    Input file with dependency data (default: dependency-report.json)
    --output-dir OUTPUT_DIR    Output directory for visualization files (default: tools/output)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

# Try to import graphviz for visualization
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz package not found. PNG visualization will not be available.")
    print("Install with: pip install graphviz")

# Root directory of the forex trading platform
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_dependency_report(file_path: str) -> Dict[str, Any]:
    """Load dependency report from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dependency report: {e}")
        sys.exit(1)

def generate_dot_graph(dependencies: Dict[str, List[str]], circular_deps: List[List[str]]) -> str:
    """Generate a DOT graph representation of the dependencies."""
    dot = ['digraph G {']
    dot.append('  rankdir=LR;')
    dot.append('  node [shape=box, style=filled, fillcolor=lightblue];')
    
    # Add nodes
    for service in dependencies:
        # Highlight services involved in circular dependencies
        if any(service in cycle for cycle in circular_deps):
            dot.append(f'  "{service}" [fillcolor=lightcoral];')
        else:
            dot.append(f'  "{service}";')
    
    # Add edges
    for service, deps in dependencies.items():
        for dep in deps:
            # Highlight edges involved in circular dependencies
            if any(service in cycle and dep in cycle for cycle in circular_deps):
                dot.append(f'  "{service}" -> "{dep}" [color=red, penwidth=2.0];')
            else:
                dot.append(f'  "{service}" -> "{dep}";')
    
    dot.append('}')
    return '\n'.join(dot)

def generate_mermaid_graph(dependencies: Dict[str, List[str]], circular_deps: List[List[str]]) -> str:
    """Generate a Mermaid graph representation of the dependencies."""
    mermaid = ['```mermaid', 'graph LR']
    
    # Define node styles
    mermaid.append('classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px')
    mermaid.append('classDef circular fill:#ffcccc,stroke:#ff0000,stroke-width:2px')
    
    # Add nodes and edges
    for service, deps in dependencies.items():
        for dep in deps:
            # Check if this edge is part of a circular dependency
            is_circular = any(service in cycle and dep in cycle for cycle in circular_deps)
            edge_style = ' --> ' if not is_circular else ' ==> '
            
            # Replace hyphens with underscores for Mermaid compatibility
            service_id = service.replace('-', '_')
            dep_id = dep.replace('-', '_')
            
            mermaid.append(f'    {service_id}["{service}"]{edge_style}{dep_id}["{dep}"]')
    
    # Apply styles to nodes involved in circular dependencies
    circular_nodes = set()
    for cycle in circular_deps:
        circular_nodes.update(cycle)
    
    for node in circular_nodes:
        node_id = node.replace('-', '_')
        mermaid.append(f'class {node_id} circular')
    
    mermaid.append('```')
    return '\n'.join(mermaid)

def generate_png_graph(dot_graph: str, output_path: str) -> bool:
    """Generate a PNG visualization from a DOT graph."""
    if not GRAPHVIZ_AVAILABLE:
        return False
    
    try:
        graph = graphviz.Source(dot_graph)
        graph.render(output_path, format='png', cleanup=True)
        return True
    except Exception as e:
        print(f"Error generating PNG graph: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Visualize dependencies in the forex trading platform')
    parser.add_argument('--input-file', default='dependency-report.json', help='Input file with dependency data')
    parser.add_argument('--output-dir', default='tools/output', help='Output directory for visualization files')
    args = parser.parse_args()
    
    # Load dependency report
    input_path = os.path.join(ROOT_DIR, 'tools', 'output', args.input_file)
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        sys.exit(1)
    
    report = load_dependency_report(input_path)
    dependencies = report.get('dependencies', {})
    circular_deps = report.get('circular_dependencies', [])
    
    # Create output directory
    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate DOT graph
    print("Generating DOT graph...")
    dot_graph = generate_dot_graph(dependencies, circular_deps)
    dot_path = os.path.join(output_dir, 'service_dependencies.dot')
    with open(dot_path, 'w', encoding='utf-8') as f:
        f.write(dot_graph)
    print(f"DOT graph saved to {dot_path}")
    
    # Generate Mermaid graph
    print("Generating Mermaid graph...")
    mermaid_graph = generate_mermaid_graph(dependencies, circular_deps)
    mermaid_path = os.path.join(output_dir, 'service_dependencies.mermaid.md')
    with open(mermaid_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_graph)
    print(f"Mermaid graph saved to {mermaid_path}")
    
    # Generate PNG graph
    if GRAPHVIZ_AVAILABLE:
        print("Generating PNG graph...")
        png_path = os.path.join(output_dir, 'service_dependencies')
        if generate_png_graph(dot_graph, png_path):
            print(f"PNG graph saved to {png_path}.png")
    
    print("\nVisualization complete!")
    print(f"All visualization files saved to {output_dir}")

if __name__ == "__main__":
    main()
