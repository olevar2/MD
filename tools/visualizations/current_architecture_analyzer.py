#!/usr/bin/env python3
"""
Current Forex Trading Platform Architecture Analyzer

This script analyzes the current architecture of the Forex Trading Platform by:
1. Scanning the directory structure to identify services
2. Analyzing dependencies between services
3. Identifying API endpoints and their relationships
4. Generating a visual representation of the current architecture
5. Outputting a comprehensive report with key metrics and insights

The script combines functionality from existing tools:
- dependency_analyzer.py for service dependencies
- code_statistics.py for service size and complexity metrics
- api_endpoint_detector.py for API endpoints
"""

import os
import sys
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('current_architecture_analyzer')

# Configuration
REPORT_DIR = Path("D:/MD/forex_trading_platform/tools/reports")
IGNORE_DIRS = [".git", ".github", ".venv", "__pycache__", "node_modules", "corrupted_backups", "tools", "docs"]
IGNORE_FILES = [".gitignore", ".DS_Store"]

# Define layer colors for visualization
LAYER_COLORS = {
    'foundation': '#8A2BE2',  # BlueViolet
    'data': '#4169E1',        # RoyalBlue
    'analysis': '#FF6347',    # Tomato
    'execution': '#FF8C00',   # DarkOrange
    'presentation': '#2E8B57', # SeaGreen
    'cross-cutting': '#20B2AA', # LightSeaGreen
    'event-bus': '#FFD700',   # Gold
    'unknown': '#808080',     # Gray
}

# Import patterns for dependency analysis
PYTHON_IMPORT_PATTERNS = [
    r'^\s*import\s+([a-zA-Z0-9_\.]+)',
    r'^\s*from\s+([a-zA-Z0-9_\.]+)\s+import',
]

JS_IMPORT_PATTERNS = [
    r'^\s*import\s+.*\s+from\s+[\'"]([a-zA-Z0-9_\.\-\/]+)[\'"]',
    r'^\s*const\s+.*\s+=\s+require\([\'"]([a-zA-Z0-9_\.\-\/]+)[\'"]\)',
]

# Service classification patterns
SERVICE_PATTERNS = {
    'foundation': [r'common-lib', r'common-js-lib', r'core-foundations', r'security-service'],
    'data': [r'data-pipeline', r'feature-store', r'market-data', r'database'],
    'analysis': [r'analysis-engine', r'ml-', r'technical-analysis', r'pattern-recognition', r'market-regime'],
    'execution': [r'strategy-execution', r'risk-management', r'portfolio-management', r'trading-gateway', r'order-management'],
    'presentation': [r'web-', r'ui-', r'frontend-', r'dashboard', r'client-app'],
    'cross-cutting': [r'monitoring', r'logging', r'notification', r'config'],
    'event-bus': [r'event-', r'message-', r'queue-'],
}


class CurrentArchitectureAnalyzer:
    """Class for analyzing the current architecture of the forex trading platform."""

    def __init__(self, root_dir: str, output_dir: str):
        """
        Initialize the analyzer.

        Args:
            root_dir: Root directory of the project
            output_dir: Directory to save the output
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.graph = nx.DiGraph()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results
        self.services = set()
        self.service_info = {}
        self.dependencies = defaultdict(set)
        self.import_details = defaultdict(list)
        self.api_endpoints = defaultdict(list)
        self.service_modules = defaultdict(set)
        self.service_layers = {}
        self.service_stats = {}
        self.circular_dependencies = []

    def should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        parts = Path(path).parts
        for part in parts:
            if part in IGNORE_DIRS:
                return True
        return False

    def get_service_name(self, file_path: str) -> str:
        """Extract service name from file path."""
        rel_path = os.path.relpath(file_path, self.root_dir)
        parts = Path(rel_path).parts
        
        if not parts:
            return "unknown"
            
        # First part is usually the service name
        service_name = parts[0]
        
        # Handle special cases
        if service_name.endswith(".py") or service_name.endswith(".js") or service_name.endswith(".md"):
            return "unknown"
            
        # Clean up service name
        service_name = service_name.replace("_", "-").lower()
        
        return service_name

    def classify_service_layer(self, service_name: str) -> str:
        """Classify a service into a layer based on its name."""
        for layer, patterns in SERVICE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, service_name, re.IGNORECASE):
                    return layer
        return "unknown"

    def extract_imports(self, file_path: str) -> List[str]:
        """Extract imports from a file."""
        imports = []
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Use appropriate patterns based on file extension
                if ext in [".py"]:
                    patterns = PYTHON_IMPORT_PATTERNS
                elif ext in [".js", ".jsx", ".ts", ".tsx"]:
                    patterns = JS_IMPORT_PATTERNS
                else:
                    return imports  # Unsupported file type
                
                # Extract imports using regex patterns
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    for match in matches:
                        # Clean up the import name
                        import_name = match.strip()
                        if import_name:
                            imports.append(import_name)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        return imports

    def map_import_to_service(self, import_name: str) -> Optional[str]:
        """Map an import name to a service."""
        for service, modules in self.service_modules.items():
            for module in modules:
                if import_name.startswith(module):
                    return service
        return None

    def count_lines(self, file_path: str) -> int:
        """Count the number of lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"Error counting lines in {file_path}: {e}")
            return 0

    def detect_api_endpoints(self, file_path: str, service_name: str) -> None:
        """Detect API endpoints in a file."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Only process Python and JavaScript files
        if ext not in [".py", ".js", ".ts"]:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # FastAPI endpoint patterns
                if ext == ".py":
                    # Look for FastAPI route decorators
                    route_patterns = [
                        r'@(?:router|app)\.(?:get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
                        r'@api_router\.(?:get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
                    ]
                    
                    for pattern in route_patterns:
                        matches = re.findall(pattern, content, re.MULTILINE)
                        for endpoint in matches:
                            self.api_endpoints[service_name].append({
                                "endpoint": endpoint,
                                "file": os.path.relpath(file_path, self.root_dir),
                                "type": "FastAPI"
                            })
                
                # Express/NestJS endpoint patterns
                elif ext in [".js", ".ts"]:
                    # Look for Express route handlers
                    route_patterns = [
                        r'(?:router|app)\.(?:get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
                        r'@(?:Get|Post|Put|Delete|Patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
                    ]
                    
                    for pattern in route_patterns:
                        matches = re.findall(pattern, content, re.MULTILINE)
                        for endpoint in matches:
                            self.api_endpoints[service_name].append({
                                "endpoint": endpoint,
                                "file": os.path.relpath(file_path, self.root_dir),
                                "type": "Express/NestJS"
                            })
        except Exception as e:
            logger.error(f"Error detecting API endpoints in {file_path}: {e}")

    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the dependency graph."""
        try:
            # Create a directed graph from dependencies
            G = nx.DiGraph()
            for source, targets in self.dependencies.items():
                for target in targets:
                    G.add_edge(source, target)
            
            # Find simple cycles
            cycles = list(nx.simple_cycles(G))
            return sorted(cycles, key=len)
        except Exception as e:
            logger.error(f"Error finding circular dependencies: {e}")
            return []

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the current architecture of the forex trading platform.
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing architecture at {self.root_dir}...")
        
        # First pass: identify all services and their modules
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Skip ignored directories
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            
            for filename in filenames:
                if filename in IGNORE_FILES:
                    continue
                    
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, self.root_dir)
                
                if self.should_ignore(rel_path):
                    continue
                
                service_name = self.get_service_name(file_path)
                if service_name != "unknown":
                    self.services.add(service_name)
                    
                    # Extract potential module names from the path
                    parts = Path(rel_path).parts
                    if len(parts) > 1:
                        # Add the service name as a module
                        self.service_modules[service_name].add(service_name)
                        
                        # Add service_name/second_part as a module
                        if len(parts) > 1:
                            module = f"{parts[0]}.{parts[1]}"
                            self.service_modules[service_name].add(module.replace("/", ".").replace("\\", "."))
                            
                            # For Python packages, add the package name
                            if parts[1].endswith("_service") or parts[1].endswith("_engine"):
                                self.service_modules[service_name].add(parts[1])
        
        # Classify services into layers
        for service in self.services:
            self.service_layers[service] = self.classify_service_layer(service)
            self.service_info[service] = {
                "name": service,
                "layer": self.service_layers[service],
                "file_count": 0,
                "line_count": 0,
                "api_endpoints": 0,
            }
        
        # Second pass: analyze imports, count files/lines, and detect API endpoints
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Skip ignored directories
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            
            for filename in filenames:
                if filename in IGNORE_FILES:
                    continue
                    
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, self.root_dir)
                
                if self.should_ignore(rel_path):
                    continue
                
                # Only process Python, JavaScript, and TypeScript files
                _, ext = os.path.splitext(filename)
                if ext.lower() not in [".py", ".js", ".jsx", ".ts", ".tsx"]:
                    continue
                
                service_name = self.get_service_name(file_path)
                if service_name == "unknown":
                    continue
                
                # Count files and lines
                self.service_info[service_name]["file_count"] += 1
                line_count = self.count_lines(file_path)
                self.service_info[service_name]["line_count"] += line_count
                
                # Detect API endpoints
                self.detect_api_endpoints(file_path, service_name)
                
                # Extract imports
                imports = self.extract_imports(file_path)
                for import_name in imports:
                    target_service = self.map_import_to_service(import_name)
                    if target_service and target_service != service_name:
                        # Add dependency
                        self.dependencies[service_name].add(target_service)
                        
                        # Add import detail
                        key = (service_name, target_service)
                        self.import_details[key].append({
                            "file": rel_path,
                            "import": import_name
                        })
        
        # Update API endpoint counts
        for service, endpoints in self.api_endpoints.items():
            self.service_info[service]["api_endpoints"] = len(endpoints)
        
        # Find circular dependencies
        self.circular_dependencies = self.find_circular_dependencies()
        
        # Prepare the result
        result = {
            "services": list(self.services),
            "service_info": self.service_info,
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "import_details": {f"{k[0]}->{k[1]}": v for k, v in self.import_details.items()},
            "api_endpoints": self.api_endpoints,
            "service_modules": {k: list(v) for k, v in self.service_modules.items()},
            "service_layers": self.service_layers,
            "circular_dependencies": self.circular_dependencies,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Analysis complete. Found {len(self.services)} services with {len(self.dependencies)} dependency relationships.")
        return result

    def build_graph(self) -> nx.DiGraph:
        """
        Build the graph of the current architecture.

        Returns:
            NetworkX directed graph of the current architecture
        """
        # Add nodes for all services
        for service, info in self.service_info.items():
            self.graph.add_node(service, 
                               layer=info["layer"], 
                               file_count=info["file_count"],
                               line_count=info["line_count"],
                               api_endpoints=info["api_endpoints"])

        # Add edges for dependencies
        for service, deps in self.dependencies.items():
            for dep in deps:
                self.graph.add_edge(service, dep)

        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph

    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Get positions for nodes in the graph visualization.

        Returns:
            Dictionary mapping node names to (x, y) positions
        """
        # Group nodes by layer
        layers = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            layer = data.get('layer', 'unknown')
            layers[layer].append(node)

        # Calculate positions
        positions = {}
        layer_heights = {
            'foundation': 0,
            'data': 1,
            'analysis': 2,
            'execution': 3,
            'presentation': 4,
            'cross-cutting': 2.5,
            'event-bus': 2,
            'unknown': 5
        }

        for layer, nodes in layers.items():
            y = layer_heights.get(layer, 0)
            width = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                x = i - width / 2
                positions[node] = (x, y)

        return positions

    def visualize_architecture(self, output_file: str = 'current_architecture.png') -> None:
        """
        Visualize the current architecture.

        Args:
            output_file: Path to save the visualization
        """
        # Build the graph
        self.build_graph()

        # Get node positions
        pos = self.get_node_positions()

        # Create figure
        plt.figure(figsize=(16, 12))

        # Draw nodes by layer with different colors
        for layer, color in LAYER_COLORS.items():
            nodes = [node for node, data in self.graph.nodes(data=True) if data.get('layer') == layer]
            if nodes:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, node_color=color, 
                                      node_size=[self.service_info[n]["line_count"] / 100 + 500 for n in nodes], 
                                      alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15,
                              connectionstyle='arc3,rad=0.1')

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_family='sans-serif')

        # Add legend
        legend_patches = []
        for layer, color in LAYER_COLORS.items():
            if any(data.get('layer') == layer for _, data in self.graph.nodes(data=True)):
                legend_patches.append(mpatches.Patch(color=color, label=layer.capitalize()))
        plt.legend(handles=legend_patches, loc='upper right')

        # Add title and adjust layout
        plt.title("Current Forex Trading Platform Architecture", fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        # Save the visualization
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved architecture visualization to {output_path}")

    def generate_layered_architecture(self, output_file: str = 'current_layered_architecture.png') -> None:
        """
        Generate a layered architecture visualization.

        Args:
            output_file: Path to save the visualization
        """
        # Create figure
        plt.figure(figsize=(14, 10))

        # Define layer heights and positions
        layer_heights = {
            'presentation': 0.8,
            'execution': 1.2,
            'analysis': 1.2,
            'data': 0.8,
            'foundation': 0.8,
            'cross-cutting': 0.6,
            'event-bus': 0.6,
            'unknown': 0.6
        }

        layer_positions = {
            'presentation': 4.0,
            'execution': 3.0,
            'analysis': 1.8,
            'data': 0.8,
            'foundation': 0.0,
            'cross-cutting': -0.8,
            'event-bus': -1.6,
            'unknown': 5.0
        }

        # Draw layers
        for layer, height in layer_heights.items():
            y = layer_positions[layer]
            rect = plt.Rectangle((-7, y), 14, height, facecolor=LAYER_COLORS[layer], alpha=0.3, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(-6.8, y + height/2, layer.capitalize(), fontsize=12, verticalalignment='center')

            # Add services to the layer
            services = [s for s, l in self.service_layers.items() if l == layer]
            for i, service in enumerate(sorted(services)):
                x = -6 + (i % 3) * 4
                y_offset = 0.2 + (i // 3) * 0.2
                plt.text(x, y + y_offset, service, fontsize=9)

        # Set plot limits and turn off axis
        plt.xlim(-7, 7)
        plt.ylim(-2.5, 6)
        plt.axis('off')

        # Add title
        plt.title("Current Forex Trading Platform - Layered Architecture", fontsize=16)

        # Save the visualization
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved layered architecture visualization to {output_path}")

    def generate_markdown_report(self, output_file: str = 'current_architecture_report.md') -> None:
        """
        Generate a markdown report of the current architecture.

        Args:
            output_file: Path to save the report
        """
        # Prepare the report
        report = []
        report.append("# Current Forex Trading Platform Architecture Report")
        report.append(f"\n*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Overview section
        report.append("## Overview")
        report.append(f"\nThe Forex Trading Platform consists of {len(self.services)} services organized into {len(set(self.service_layers.values()))} layers.\n")
        
        # Services by layer
        report.append("### Services by Layer")
        for layer in sorted(set(self.service_layers.values())):
            services = [s for s, l in self.service_layers.items() if l == layer]
            report.append(f"\n#### {layer.capitalize()}")
            for service in sorted(services):
                info = self.service_info[service]
                report.append(f"- **{service}**: {info['file_count']} files, {info['line_count']} lines, {info['api_endpoints']} API endpoints")
        
        # Dependencies section
        report.append("\n## Dependencies")
        report.append("\n### Service Dependencies")
        for service, deps in sorted(self.dependencies.items()):
            if deps:
                report.append(f"\n#### {service}")
                for dep in sorted(deps):
                    report.append(f"- {dep}")
        
        # Circular dependencies section
        if self.circular_dependencies:
            report.append("\n### Circular Dependencies")
            for i, cycle in enumerate(self.circular_dependencies):
                report.append(f"\n{i+1}. {' -> '.join(cycle)} -> {cycle[0]}")
        
        # API endpoints section
        report.append("\n## API Endpoints")
        for service, endpoints in sorted(self.api_endpoints.items()):
            if endpoints:
                report.append(f"\n### {service}")
                report.append("\n| Endpoint | File | Type |")
                report.append("|----------|------|------|")
                for endpoint in sorted(endpoints, key=lambda x: x['endpoint']):
                    report.append(f"| {endpoint['endpoint']} | {endpoint['file']} | {endpoint['type']} |")
        
        # Save the report
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        logger.info(f"Saved architecture report to {output_path}")

    def generate_visualizations(self) -> None:
        """Generate all architecture visualizations and reports."""
        self.visualize_architecture()
        self.generate_layered_architecture()
        self.generate_markdown_report()


def main():
    """Main function to run the current architecture analyzer."""
    parser = argparse.ArgumentParser(description='Analyze and visualize the current architecture of the Forex Trading Platform')
    parser.add_argument('--root', type=str, default='D:/MD/forex_trading_platform', help='Root directory of the project')
    parser.add_argument('--output-dir', type=str, default=str(REPORT_DIR), help='Directory to save output files')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize and run the analyzer
    analyzer = CurrentArchitectureAnalyzer(args.root, args.output_dir)
    analyzer.analyze()
    analyzer.generate_visualizations()

    print(f"Analysis complete. Reports and visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
