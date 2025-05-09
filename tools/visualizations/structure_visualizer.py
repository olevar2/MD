#!/usr/bin/env python3
"""
Forex Trading Platform Structure Visualizer

This script generates a visual representation of the project structure,
creating an interactive HTML visualization that shows the hierarchy,
components, and relationships between different parts of the system.
"""

import os
import json
import re
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('structure_visualizer')

# HTML template for the visualization
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forex Trading Platform Structure</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 5px;
        }}
        h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        h3 {{
            color: #2980b9;
            margin-top: 25px;
        }}
        .overview {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .service-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .service-card {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            transition: transform 0.3s ease;
        }}
        .service-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .service-name {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .service-details {{
            font-size: 0.9em;
        }}
        .service-details p {{
            margin: 5px 0;
        }}
        .language-bar {{
            height: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .language-segment {{
            height: 100%;
            float: left;
        }}
        .dependency-graph {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-top: 30px;
            text-align: center;
        }}
        .dependency-graph img {{
            max-width: 100%;
            height: auto;
            margin-top: 20px;
        }}
        .metrics {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .collapsible {{
            background-color: #f8f9fa;
            color: #444;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .active, .collapsible:hover {{
            background-color: #e9ecef;
        }}
        .content {{
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            background-color: white;
            border-radius: 0 0 5px 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e1e1e1;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .search-container {{
            margin-bottom: 20px;
        }}
        #searchInput {{
            width: 100%;
            padding: 12px 20px;
            margin: 8px 0;
            box-sizing: border-box;
            border: 2px solid #ccc;
            border-radius: 4px;
            font-size: 16px;
        }}
        .language-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }}
        .language-item {{
            display: flex;
            align-items: center;
            margin-right: 15px;
        }}
        .language-color {{
            width: 15px;
            height: 15px;
            border-radius: 3px;
            margin-right: 5px;
        }}
        .tab {{
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 5px 5px 0 0;
        }}
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 17px;
        }}
        .tab button:hover {{
            background-color: #ddd;
        }}
        .tab button.active {{
            background-color: #3498db;
            color: white;
        }}
        .tabcontent {{
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 5px 5px;
            background-color: white;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Forex Trading Platform Structure</h1>
        <p>Interactive visualization of the platform architecture and components</p>
    </header>

    <div class="container">
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="Search for services, components, or features...">
        </div>

        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'Overview')">Overview</button>
            <button class="tablinks" onclick="openTab(event, 'Services')">Services</button>
            <button class="tablinks" onclick="openTab(event, 'Dependencies')">Dependencies</button>
            <button class="tablinks" onclick="openTab(event, 'Metrics')">Metrics</button>
            <button class="tablinks" onclick="openTab(event, 'Recommendations')">Recommendations</button>
        </div>

        <div id="Overview" class="tabcontent" style="display: block;">
            <div class="overview">
                <h2>Platform Overview</h2>
                <p>The Forex Trading Platform is a comprehensive system for analyzing the Forex market, making trading decisions, monitoring trades, and self-learning. The platform consists of multiple specialized services that work together to provide a complete trading solution.</p>

                <h3>Key Components</h3>
                <ul>
                    {overview_components}
                </ul>

                <h3>Technology Stack</h3>
                <div class="language-legend">
                    {language_legend}
                </div>

                <h3>System Architecture</h3>
                <p>The platform follows a microservices architecture, with specialized services for different aspects of the trading process. These services communicate through standardized APIs and message queues.</p>
            </div>
        </div>

        <div id="Services" class="tabcontent">
            <h2>Services</h2>
            <p>The platform consists of the following services, each responsible for specific functionality:</p>

            <div class="service-grid">
                {service_cards}
            </div>
        </div>

        <div id="Dependencies" class="tabcontent">
            <h2>Service Dependencies</h2>
            <p>This visualization shows the dependencies between different services in the platform:</p>

            <div class="dependency-graph">
                <img src="dependency_graph.png" alt="Service Dependency Graph">
            </div>

            <h3>Dependency Details</h3>
            <table>
                <thead>
                    <tr>
                        <th>Service</th>
                        <th>Depends On</th>
                        <th>Used By</th>
                    </tr>
                </thead>
                <tbody>
                    {dependency_rows}
                </tbody>
            </table>
        </div>

        <div id="Metrics" class="tabcontent">
            <h2>System Metrics</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Services</div>
                    <div class="metric-value">{total_services}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Files</div>
                    <div class="metric-value">{total_files}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Languages Used</div>
                    <div class="metric-value">{total_languages}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">API Endpoints</div>
                    <div class="metric-value">{total_endpoints}</div>
                </div>
            </div>

            <h3>Language Distribution</h3>
            <table>
                <thead>
                    <tr>
                        <th>Language</th>
                        <th>Files</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {language_rows}
                </tbody>
            </table>
        </div>

        <div id="Recommendations" class="tabcontent">
            <h2>Architecture Recommendations</h2>

            <h3>General Recommendations</h3>
            <ul>
                <li><strong>Standardize Error Handling:</strong> Ensure all services use a consistent approach to error handling.</li>
                <li><strong>API Documentation:</strong> Maintain comprehensive API documentation for all services.</li>
                <li><strong>Dependency Management:</strong> Regularly review and minimize dependencies between services.</li>
                <li><strong>Monitoring:</strong> Implement consistent monitoring across all services.</li>
                <li><strong>Testing:</strong> Ensure comprehensive test coverage for all services.</li>
            </ul>

            <h3>Service-Specific Recommendations</h3>
            {service_recommendations}
        </div>
    </div>

    <script>
        // Collapsible sections
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            });
        }

        // Search functionality
        document.getElementById('searchInput').addEventListener('keyup', function() {
            var input = this.value.toLowerCase();
            var serviceCards = document.getElementsByClassName('service-card');

            for (var i = 0; i < serviceCards.length; i++) {
                var card = serviceCards[i];
                var text = card.textContent.toLowerCase();

                if (text.includes(input)) {
                    card.style.display = "";
                } else {
                    card.style.display = "none";
                }
            }
        });

        // Tab functionality
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;

            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }

            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }

            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
    </script>
</body>
</html>
"""

# Language colors for visualization
LANGUAGE_COLORS = {
    'Python': '#3572A5',
    'JavaScript': '#F7DF1E',
    'TypeScript': '#2B7489',
    'JavaScript (React)': '#61DAFB',
    'TypeScript (React)': '#61DAFB',
    'HTML': '#E34C26',
    'CSS': '#563D7C',
    'SCSS': '#C6538C',
    'JSON': '#292929',
    'Markdown': '#083FA1',
    'YAML': '#CB171E',
    'Shell': '#89E051',
    'SQL': '#E38C00',
    'Java': '#B07219',
    'C#': '#178600',
    'Go': '#00ADD8',
    'Rust': '#DEA584',
    'Ruby': '#701516',
    'PHP': '#4F5D95',
    'C': '#555555',
    'C++': '#F34B7D',
}

class StructureVisualizer:
    """Class for visualizing the structure of the forex trading platform."""

    def __init__(self, architecture_data_path: str, output_dir: str):
        """
        Initialize the visualizer with architecture data.

        Args:
            architecture_data_path: Path to the architecture data JSON file
            output_dir: Directory to save the visualization
        """
        self.output_dir = output_dir

        # Load architecture data
        with open(architecture_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def generate_service_cards(self) -> str:
        """
        Generate HTML for service cards.

        Returns:
            HTML string for service cards
        """
        service_cards = []

        for service in sorted(self.data['services']):
            # Get language statistics for this service
            language_stats = self.data['file_counts'].get(service, {})
            total_files = sum(language_stats.values())

            # Generate language bar segments
            language_bar = ""
            if total_files > 0:
                for language, count in language_stats.items():
                    percentage = (count / total_files) * 100
                    color = LANGUAGE_COLORS.get(language, '#CCCCCC')
                    language_bar += f'<div class="language-segment" style="width: {percentage}%; background-color: {color};" title="{language}: {count} files"></div>'

            # Count dependencies and dependents
            dependencies = len(self.data['dependencies'].get(service, []))
            dependents = sum(1 for s in self.data['services'] if service in self.data['dependencies'].get(s, []))

            # Count API endpoints
            endpoints = len(self.data['api_endpoints'].get(service, []))

            # Generate card HTML
            card = f"""
            <div class="service-card">
                <div class="service-name">{service}</div>
                <div class="service-details">
                    <p><strong>Files:</strong> {total_files}</p>
                    <p><strong>Dependencies:</strong> {dependencies}</p>
                    <p><strong>Dependents:</strong> {dependents}</p>
                    <p><strong>API Endpoints:</strong> {endpoints}</p>
                    <div class="language-bar">
                        {language_bar}
                    </div>
                </div>
                <button class="collapsible">Details</button>
                <div class="content">
                    <p><strong>Languages:</strong></p>
                    <ul>
            """

            # Add language details
            for language, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_files) * 100 if total_files > 0 else 0
                card += f'<li>{language}: {count} files ({percentage:.1f}%)</li>'

            card += """
                    </ul>
                </div>
            </div>
            """

            service_cards.append(card)

        return '\n'.join(service_cards)

    def generate_dependency_rows(self) -> str:
        """
        Generate HTML for dependency table rows.

        Returns:
            HTML string for dependency table rows
        """
        dependency_rows = []

        for service in sorted(self.data['services']):
            # Get dependencies
            dependencies = self.data['dependencies'].get(service, [])
            dependencies_str = ', '.join(sorted(dependencies)) if dependencies else 'None'

            # Get dependents
            dependents = [s for s in self.data['services'] if service in self.data['dependencies'].get(s, [])]
            dependents_str = ', '.join(sorted(dependents)) if dependents else 'None'

            # Generate row HTML
            row = f"""
            <tr>
                <td>{service}</td>
                <td>{dependencies_str}</td>
                <td>{dependents_str}</td>
            </tr>
            """

            dependency_rows.append(row)

        return '\n'.join(dependency_rows)

    def generate_language_rows(self) -> str:
        """
        Generate HTML for language table rows.

        Returns:
            HTML string for language table rows
        """
        language_rows = []
        total_files = sum(self.data['language_stats'].values())

        for language, count in sorted(self.data['language_stats'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            color = LANGUAGE_COLORS.get(language, '#CCCCCC')

            # Generate row HTML
            row = f"""
            <tr>
                <td><span style="display: inline-block; width: 15px; height: 15px; background-color: {color}; margin-right: 5px; border-radius: 3px;"></span> {language}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """

            language_rows.append(row)

        return '\n'.join(language_rows)

    def generate_language_legend(self) -> str:
        """
        Generate HTML for language legend.

        Returns:
            HTML string for language legend
        """
        legend_items = []

        for language, color in LANGUAGE_COLORS.items():
            if language in self.data['language_stats']:
                item = f"""
                <div class="language-item">
                    <div class="language-color" style="background-color: {color};"></div>
                    <div>{language}</div>
                </div>
                """
                legend_items.append(item)

        return '\n'.join(legend_items)

    def generate_overview_components(self) -> str:
        """
        Generate HTML for overview components.

        Returns:
            HTML string for overview components
        """
        components = []

        # Add main services with brief descriptions
        service_descriptions = {
            'analysis-engine-service': 'Performs technical analysis and indicator calculations',
            'trading-gateway-service': 'Interfaces with trading platforms for order execution',
            'data-pipeline-service': 'Handles data ingestion, processing, and storage',
            'feature-store-service': 'Manages technical indicators and features',
            'ml-integration-service': 'Integrates machine learning models with trading systems',
            'ml-workbench-service': 'Provides environment for ML model development',
            'portfolio-management-service': 'Tracks and manages trading portfolios',
            'risk-management-service': 'Assesses and manages trading risks',
            'strategy-execution-engine': 'Executes trading strategies',
            'monitoring-alerting-service': 'Monitors system performance and generates alerts',
            'common-lib': 'Shared Python libraries and utilities',
            'common-js-lib': 'Shared JavaScript libraries and utilities',
            'core-foundations': 'Core libraries and utilities',
            'ui-service': 'User interface for the platform',
        }

        for service in sorted(self.data['services']):
            description = service_descriptions.get(service, 'Service component')
            components.append(f'<li><strong>{service}</strong>: {description}</li>')

        return '\n'.join(components)

    def generate_service_recommendations(self) -> str:
        """
        Generate HTML for service recommendations.

        Returns:
            HTML string for service recommendations
        """
        recommendations = []

        # Check for services with many dependencies
        for service, deps in self.data['dependencies'].items():
            if len(deps) > 5:
                rec = f"""
                <div class="service-card">
                    <div class="service-name">{service}</div>
                    <div class="service-details">
                        <p><strong>Issue:</strong> High number of dependencies ({len(deps)})</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Consider refactoring to reduce dependencies</li>
                            <li>Evaluate if some functionality should be moved to a shared library</li>
                            <li>Review if all dependencies are necessary</li>
                        </ul>
                    </div>
                </div>
                """
                recommendations.append(rec)

        # Check for services with no error handling
        for service in self.data['services']:
            if service not in self.data['error_handlers'] or not self.data['error_handlers'][service]:
                rec = f"""
                <div class="service-card">
                    <div class="service-name">{service}</div>
                    <div class="service-details">
                        <p><strong>Issue:</strong> No error handlers detected</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Implement standardized error handling</li>
                            <li>Use common-lib exceptions</li>
                            <li>Add comprehensive error logging</li>
                        </ul>
                    </div>
                </div>
                """
                recommendations.append(rec)

        # If no specific recommendations, provide a general note
        if not recommendations:
            return "<p>No specific service recommendations at this time.</p>"

        return '<div class="service-grid">\n' + '\n'.join(recommendations) + '\n</div>'

    def generate_visualization(self) -> None:
        """Generate the HTML visualization."""
        # Copy dependency graph image to output directory
        if os.path.exists('dependency_graph.png'):
            src_path = 'dependency_graph.png'
            dst_path = os.path.join(self.output_dir, 'dependency_graph.png')
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copy(src_path, dst_path)

        # Calculate metrics
        total_services = len(self.data['services'])
        total_files = sum(self.data['language_stats'].values())
        total_languages = len(self.data['language_stats'])
        total_endpoints = sum(len(endpoints) for endpoints in self.data['api_endpoints'].values())

        # Generate HTML components
        service_cards = self.generate_service_cards()
        dependency_rows = self.generate_dependency_rows()
        language_rows = self.generate_language_rows()
        language_legend = self.generate_language_legend()
        overview_components = self.generate_overview_components()
        service_recommendations = self.generate_service_recommendations()

        # Fill in the HTML template
        # Replace placeholders manually to avoid issues with curly braces in the HTML
        html_content = HTML_TEMPLATE
        html_content = html_content.replace("{service_cards}", service_cards)
        html_content = html_content.replace("{dependency_rows}", dependency_rows)
        html_content = html_content.replace("{language_rows}", language_rows)
        html_content = html_content.replace("{language_legend}", language_legend)
        html_content = html_content.replace("{overview_components}", overview_components)
        html_content = html_content.replace("{service_recommendations}", service_recommendations)
        html_content = html_content.replace("{total_services}", str(total_services))
        html_content = html_content.replace("{total_files}", str(total_files))
        html_content = html_content.replace("{total_languages}", str(total_languages))
        html_content = html_content.replace("{total_endpoints}", str(total_endpoints))

        # Write the HTML file
        output_path = os.path.join(self.output_dir, 'structure_visualization.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Generated structure visualization at {output_path}")


def generate_architecture_data(root_dir):
    """
    Generate architecture data by analyzing the codebase.

    Args:
        root_dir: Root directory of the project

    Returns:
        Dictionary containing architecture data
    """
    import re
    from pathlib import Path
    from collections import defaultdict

    # Configuration
    IGNORE_DIRS = [".git", ".github", ".venv", "__pycache__", "node_modules", "corrupted_backups", "tools", "docs"]
    IGNORE_FILES = [".gitignore", ".DS_Store"]

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

    # Initialize data structures
    services = set()
    service_modules = defaultdict(set)
    service_layers = {}
    dependencies = defaultdict(set)
    api_endpoints = defaultdict(list)
    file_counts = defaultdict(lambda: defaultdict(int))
    language_stats = defaultdict(int)
    error_handlers = {}

    # Helper functions
    def should_ignore(path):
        parts = Path(path).parts
        for part in parts:
            if part in IGNORE_DIRS:
                return True
        return False

    def get_service_name(file_path):
        rel_path = os.path.relpath(file_path, root_dir)
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

    def classify_service_layer(service_name):
        for layer, patterns in SERVICE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, service_name, re.IGNORECASE):
                    return layer
        return "unknown"

    def get_language(file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.py':
            return 'Python'
        elif ext == '.js':
            return 'JavaScript'
        elif ext == '.jsx':
            return 'JavaScript (React)'
        elif ext == '.ts':
            return 'TypeScript'
        elif ext == '.tsx':
            return 'TypeScript (React)'
        elif ext == '.html':
            return 'HTML'
        elif ext == '.css':
            return 'CSS'
        elif ext == '.scss':
            return 'SCSS'
        elif ext == '.json':
            return 'JSON'
        elif ext == '.md':
            return 'Markdown'
        elif ext == '.yml' or ext == '.yaml':
            return 'YAML'
        elif ext == '.sh':
            return 'Shell'
        elif ext == '.sql':
            return 'SQL'
        elif ext == '.java':
            return 'Java'
        elif ext == '.cs':
            return 'C#'
        elif ext == '.go':
            return 'Go'
        elif ext == '.rs':
            return 'Rust'
        elif ext == '.rb':
            return 'Ruby'
        elif ext == '.php':
            return 'PHP'
        elif ext == '.c':
            return 'C'
        elif ext == '.cpp' or ext == '.cc' or ext == '.cxx':
            return 'C++'
        else:
            return 'Other'

    def extract_imports(file_path):
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
            print(f"Error processing {file_path}: {e}")

        return imports

    def detect_api_endpoints(file_path, service_name):
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
                            api_endpoints[service_name].append({
                                "endpoint": endpoint,
                                "file": os.path.relpath(file_path, root_dir),
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
                            api_endpoints[service_name].append({
                                "endpoint": endpoint,
                                "file": os.path.relpath(file_path, root_dir),
                                "type": "Express/NestJS"
                            })
        except Exception as e:
            print(f"Error detecting API endpoints in {file_path}: {e}")

    def detect_error_handlers(file_path, service_name):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Only process Python and JavaScript files
        if ext not in [".py", ".js", ".ts"]:
            return False

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                # Look for error handling patterns
                if ext == ".py":
                    # Python error handling patterns
                    if re.search(r'except\s+\w+Error', content) or re.search(r'@app\.exception_handler', content):
                        return True
                elif ext in [".js", ".ts"]:
                    # JavaScript error handling patterns
                    if re.search(r'\.catch\(', content) or re.search(r'try\s*{.*}\s*catch', content, re.DOTALL):
                        return True
        except Exception:
            pass

        return False

    def map_import_to_service(import_name):
        for service, modules in service_modules.items():
            for module in modules:
                if import_name.startswith(module):
                    return service
        return None

    # First pass: identify all services and their modules
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip ignored directories
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        for filename in filenames:
            if filename in IGNORE_FILES:
                continue

            file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(file_path, root_dir)

            if should_ignore(rel_path):
                continue

            service_name = get_service_name(file_path)
            if service_name != "unknown":
                services.add(service_name)

                # Extract potential module names from the path
                parts = Path(rel_path).parts
                if len(parts) > 1:
                    # Add the service name as a module
                    service_modules[service_name].add(service_name)

                    # Add service_name/second_part as a module
                    if len(parts) > 1:
                        module = f"{parts[0]}.{parts[1]}"
                        service_modules[service_name].add(module.replace("/", ".").replace("\\", "."))

                        # For Python packages, add the package name
                        if parts[1].endswith("_service") or parts[1].endswith("_engine"):
                            service_modules[service_name].add(parts[1])

    # Classify services into layers
    for service in services:
        service_layers[service] = classify_service_layer(service)
        error_handlers[service] = False  # Initialize error handlers

    # Second pass: analyze files, count by language, detect API endpoints and error handlers
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip ignored directories
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        for filename in filenames:
            if filename in IGNORE_FILES:
                continue

            file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(file_path, root_dir)

            if should_ignore(rel_path):
                continue

            service_name = get_service_name(file_path)
            if service_name == "unknown":
                continue

            # Count files by language
            language = get_language(file_path)
            file_counts[service_name][language] += 1
            language_stats[language] += 1

            # Detect API endpoints
            detect_api_endpoints(file_path, service_name)

            # Detect error handlers
            if detect_error_handlers(file_path, service_name):
                error_handlers[service_name] = True

            # Extract imports for dependencies
            _, ext = os.path.splitext(filename)
            if ext.lower() in [".py", ".js", ".jsx", ".ts", ".tsx"]:
                imports = extract_imports(file_path)
                for import_name in imports:
                    target_service = map_import_to_service(import_name)
                    if target_service and target_service != service_name:
                        dependencies[service_name].add(target_service)

    # Prepare the result
    result = {
        "services": list(services),
        "file_counts": file_counts,
        "dependencies": {k: list(v) for k, v in dependencies.items()},
        "api_endpoints": api_endpoints,
        "language_stats": language_stats,
        "error_handlers": error_handlers
    }

    return result

def main():
    """Main function to run the structure visualizer."""
    reports_dir = "D:/MD/forex_trading_platform/tools/reports"
    root_dir = "D:/MD/forex_trading_platform"
    parser = argparse.ArgumentParser(description='Generate a visual representation of the Forex Trading Platform structure')
    parser.add_argument('--root', type=str, default=root_dir, help='Root directory of the project')
    parser.add_argument('--output-dir', type=str, default=reports_dir, help='Directory to save the visualization')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate architecture data from the codebase
    print("Analyzing codebase to generate architecture data...")
    architecture_data = generate_architecture_data(args.root)

    # Save the architecture data to a JSON file in the output directory
    architecture_data_path = os.path.join(args.output_dir, 'architecture_data.json')
    with open(architecture_data_path, 'w', encoding='utf-8') as f:
        json.dump(architecture_data, f, indent=2)
    print(f"Architecture data saved to {architecture_data_path}")

    # Initialize and run the visualizer
    visualizer = StructureVisualizer(architecture_data_path, args.output_dir)
    visualizer.generate_visualization()

    print(f"Visualization complete. HTML file saved to {os.path.join(args.output_dir, 'structure_visualization.html')}")


if __name__ == '__main__':
    main()
