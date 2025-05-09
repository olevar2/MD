#!/usr/bin/env python3
"""
Improved Forex Trading Platform Architecture Visualizer

This script analyzes the current architecture of the Forex Trading Platform and
generates a visualization of an improved architecture based on the current state,
showing recommended service organization and dependencies.
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
from typing import Dict, List, Tuple, Set, Any, Optional
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('improved_architecture_visualizer')

# Configuration
IGNORE_DIRS = [".git", ".github", ".venv", "__pycache__", "node_modules", "corrupted_backups", "tools", "docs"]
IGNORE_FILES = [".gitignore", ".DS_Store"]

# Define layer colors
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

class ImprovedArchitectureVisualizer:
    """Class for visualizing the improved architecture of the forex trading platform."""

    def __init__(self, root_dir: str, output_dir: str):
        """
        Initialize the visualizer.

        Args:
            root_dir: Root directory of the project
            output_dir: Directory to save the visualization
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
        self.service_modules = defaultdict(set)
        self.service_layers = {}

        # Analyze the current architecture
        self.analyze_current_architecture()

        # Define the improved architecture based on current services
        self.define_improved_architecture()

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

    def analyze_current_architecture(self) -> None:
        """Analyze the current architecture of the forex trading platform."""
        logger.info(f"Analyzing current architecture at {self.root_dir}...")

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
            }

        # Second pass: analyze imports to detect dependencies
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

                # Extract imports
                imports = self.extract_imports(file_path)
                for import_name in imports:
                    target_service = self.map_import_to_service(import_name)
                    if target_service and target_service != service_name:
                        # Add dependency
                        self.dependencies[service_name].add(target_service)

        logger.info(f"Current architecture analysis complete. Found {len(self.services)} services with {len(self.dependencies)} dependency relationships.")

    def define_improved_architecture(self) -> None:
        """Define the improved architecture based on the current services."""
        # Create a mapping of current services to improved services
        self.improved_services = {}
        self.improved_layers = {}
        self.improved_dependencies = defaultdict(set)

        # Map current services to improved services
        for service in self.services:
            # Handle special cases for service renaming or splitting
            if service == "analysis-engine-service":
                # Split analysis-engine-service into multiple specialized services
                self.improved_services[service] = [
                    "technical-analysis-service",
                    "pattern-recognition-service",
                    "market-regime-service"
                ]
            elif service == "data-service":
                # Rename data-service to data-pipeline-service
                self.improved_services[service] = ["data-pipeline-service"]
            else:
                # Keep the service as is
                self.improved_services[service] = [service]

        # Add new services that don't exist in the current architecture
        new_services = {
            "market-data-service": "data",
            "order-management-service": "execution",
            "api-gateway-service": "presentation",
            "notification-service": "presentation",
            "reporting-service": "presentation",
            "configuration-service": "cross-cutting",
            "service-registry": "cross-cutting",
            "circuit-breaker-service": "cross-cutting",
            "event-bus": "event-bus",
            "security-service": "foundation"
        }

        # Organize services into layers
        self.layers = defaultdict(list)

        # Add existing services to layers
        for service, layer in self.service_layers.items():
            for improved_service in self.improved_services.get(service, [service]):
                if improved_service not in self.layers[layer]:
                    self.layers[layer].append(improved_service)
                self.improved_layers[improved_service] = layer

        # Add new services to layers
        for service, layer in new_services.items():
            if service not in self.layers[layer]:
                self.layers[layer].append(service)
            self.improved_layers[service] = layer

        # Define dependencies for the improved architecture
        # First, map existing dependencies
        for service, deps in self.dependencies.items():
            for improved_service in self.improved_services.get(service, [service]):
                for dep in deps:
                    for improved_dep in self.improved_services.get(dep, [dep]):
                        self.improved_dependencies[improved_service].add(improved_dep)

        # Add dependencies for new services and enhance existing ones
        additional_dependencies = {
            # Foundation layer dependencies
            'common-lib': ['core-foundations'],
            'common-js-lib': ['core-foundations'],
            'security-service': ['core-foundations'],

            # Data layer dependencies
            'data-pipeline-service': ['event-bus'],
            'feature-store-service': ['data-pipeline-service', 'event-bus'],
            'market-data-service': ['core-foundations', 'common-lib', 'event-bus'],

            # Analysis layer dependencies
            'technical-analysis-service': ['feature-store-service', 'event-bus'],
            'pattern-recognition-service': ['feature-store-service', 'event-bus'],
            'market-regime-service': ['feature-store-service', 'event-bus'],
            'ml-workbench-service': ['feature-store-service', 'event-bus'],
            'ml-integration-service': ['feature-store-service', 'ml-workbench-service', 'event-bus'],

            # Execution layer dependencies
            'strategy-execution-engine': ['technical-analysis-service', 'pattern-recognition-service', 'market-regime-service', 'ml-integration-service', 'event-bus'],
            'risk-management-service': ['event-bus'],
            'portfolio-management-service': ['event-bus'],
            'trading-gateway-service': ['risk-management-service', 'event-bus'],
            'order-management-service': ['trading-gateway-service', 'risk-management-service', 'event-bus'],

            # Presentation layer dependencies
            'api-gateway-service': ['security-service', 'event-bus'],
            'ui-service': ['api-gateway-service'],
            'notification-service': ['event-bus'],
            'reporting-service': ['portfolio-management-service', 'event-bus'],

            # Cross-cutting concerns dependencies
            'monitoring-alerting-service': ['event-bus'],
            'configuration-service': ['core-foundations'],
            'service-registry': ['core-foundations'],
            'circuit-breaker-service': ['service-registry'],

            # Event bus has no dependencies
            'event-bus': [],
        }

        # Merge additional dependencies with existing ones
        for service, deps in additional_dependencies.items():
            for dep in deps:
                self.improved_dependencies[service].add(dep)

        # Convert dependencies to dictionary format
        self.dependencies = {k: list(v) for k, v in self.improved_dependencies.items()}

        logger.info(f"Defined improved architecture with {sum(len(services) for services in self.layers.values())} services organized into {len(self.layers)} layers.")

    def build_graph(self) -> nx.DiGraph:
        """
        Build the graph of the improved architecture.

        Returns:
            NetworkX directed graph of the improved architecture
        """
        # Add nodes for all services
        for layer, services in self.layers.items():
            for service in services:
                self.graph.add_node(service, layer=layer)

        # Add edges for dependencies
        for service, deps in self.dependencies.items():
            for dep in deps:
                if service in self.graph and dep in self.graph:
                    self.graph.add_edge(service, dep)

        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph

    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Get positions for nodes in the graph visualization.

        Returns:
            Dictionary mapping node names to (x, y) positions
        """
        # Define layer positions (y-coordinates)
        layer_positions = {
            'foundation': 0,
            'data': 1,
            'analysis': 2,
            'execution': 3,
            'presentation': 4,
            'cross-cutting': 2,  # Same level as analysis
            'event-bus': 2,      # Central position
            'unknown': 5,        # Place unknown services at the top
        }

        # Calculate positions for each node
        positions = {}
        for layer, services in self.layers.items():
            y = layer_positions[layer]

            # Special handling for event-bus
            if layer == 'event-bus':
                positions['event-bus'] = (0, y)  # Central position
                continue

            # Special handling for cross-cutting concerns
            if layer == 'cross-cutting':
                x_offset = -6  # Position on the left side
                for i, service in enumerate(services):
                    positions[service] = (x_offset, y + i * 0.5)
                continue

            # Position services in each layer
            num_services = len(services)
            for i, service in enumerate(services):
                # Calculate x position to spread services evenly
                x = (i - (num_services - 1) / 2) * 2
                positions[service] = (x, y)

        return positions

    def visualize_architecture(self, output_file: str = 'improved_architecture.png') -> None:
        """
        Visualize the improved architecture.

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
            nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, node_color=color, node_size=1000, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15,
                              connectionstyle='arc3,rad=0.1')

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_family='sans-serif')

        # Add legend
        legend_patches = []
        for layer, color in LAYER_COLORS.items():
            legend_patches.append(mpatches.Patch(color=color, label=layer.capitalize()))
        plt.legend(handles=legend_patches, loc='upper right')

        # Add title and adjust layout
        plt.title("Improved Forex Trading Platform Architecture", fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        # Save the visualization
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved architecture visualization to {output_path}")

    def generate_layered_architecture(self, output_file: str = 'layered_architecture.png') -> None:
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
            'unknown': 0.6,
        }

        layer_positions = {
            'presentation': 4.0,
            'execution': 3.0,
            'analysis': 1.8,
            'data': 0.8,
            'foundation': 0.0,
            'cross-cutting': -0.8,
            'event-bus': -1.6,
            'unknown': 5.0,
        }

        # Draw main layers
        for layer, height in layer_heights.items():
            if layer == 'cross-cutting' or layer == 'event-bus':
                continue

            y = layer_positions[layer]
            rect = plt.Rectangle((-7, y), 14, height, facecolor=LAYER_COLORS.get(layer, '#808080'), alpha=0.3, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(-6.8, y + height/2, layer.capitalize(), fontsize=12, verticalalignment='center')

        # Draw cross-cutting concerns
        if 'cross-cutting' in self.layers and self.layers['cross-cutting']:
            rect = plt.Rectangle((-9, 0), 1.8, 5, facecolor=LAYER_COLORS['cross-cutting'], alpha=0.3, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(-8.8, 2.5, 'Cross-Cutting\nConcerns', fontsize=10, verticalalignment='center', rotation=90)

        # Draw event bus
        if 'event-bus' in self.layers and self.layers['event-bus']:
            rect = plt.Rectangle((-7, 1.5), 14, 0.3, facecolor=LAYER_COLORS['event-bus'], alpha=0.7, edgecolor='black')
            plt.gca().add_patch(rect)
            plt.text(0, 1.65, 'Event Bus', fontsize=10, horizontalalignment='center')

        # Add services to each layer
        for layer, services in self.layers.items():
            if layer == 'event-bus':
                continue

            if layer == 'cross-cutting':
                x = -8.5
                for i, service in enumerate(services):
                    y = 0.5 + i * 1.0
                    plt.text(x, y, service.replace('-service', ''), fontsize=8, horizontalalignment='center')
                continue

            if layer in layer_positions and layer in layer_heights:
                y = layer_positions[layer] + layer_heights[layer] / 2
                num_services = len(services)
                if num_services > 0:
                    for i, service in enumerate(services):
                        x = -6 + (i + 0.5) * (12 / num_services)
                        plt.text(x, y, service.replace('-service', ''), fontsize=8, horizontalalignment='center')

        # Set plot limits and turn off axis
        plt.xlim(-10, 8)
        plt.ylim(-0.5, 5.5)
        plt.axis('off')

        # Add title
        plt.title("Improved Forex Trading Platform - Layered Architecture", fontsize=16)

        # Save the visualization
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved layered architecture visualization to {output_path}")

    def generate_visualizations(self) -> None:
        """Generate all architecture visualizations."""
        self.visualize_architecture()
        self.generate_layered_architecture()


def main():
    """Main function to run the improved architecture visualizer."""
    reports_dir = "D:/MD/forex_trading_platform/tools/reports"
    root_dir = "D:/MD/forex_trading_platform"
    parser = argparse.ArgumentParser(description='Generate visualizations of the improved Forex Trading Platform architecture')
    parser.add_argument('--root', type=str, default=root_dir, help='Root directory of the project')
    parser.add_argument('--output-dir', type=str, default=reports_dir, help='Directory to save output files')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize and run the visualizer
    visualizer = ImprovedArchitectureVisualizer(args.root, args.output_dir)
    visualizer.generate_visualizations()

    print(f"Visualizations complete. Images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
