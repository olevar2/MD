#!/usr/bin/env python3
"""
Generate Comprehensive Architecture Report for the Forex Trading Platform

This script generates a comprehensive architecture report for the forex trading platform
by combining the results from various analysis scripts.

Usage:
    python generate_architecture_report.py [--output-file OUTPUT_FILE]

Options:
    --output-file OUTPUT_FILE    Output file for the architecture report (default: architecture-report.html)
"""

import os
import sys
import json
import argparse
import base64
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Root directory of the forex trading platform
ROOT_DIR = "D:/MD/forex_trading_platform"

# HTML template for the report
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forex Trading Platform Architecture Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3, h4 {{
            color: #0066cc;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .visualization {{
            margin-top: 20px;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}
        .stats {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            flex: 1;
            min-width: 200px;
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin-top: 0;
            color: #0066cc;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #0066cc;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #0066cc;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }}
        .tabs {{
            display: flex;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            background-color: #eee;
            cursor: pointer;
            border: 1px solid #ddd;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }}
        .tab.active {{
            background-color: #0066cc;
            color: white;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 0 5px 5px 5px;
        }}
        .tab-content.active {{
            display: block;
        }}
        .chart-container {{
            height: 400px;
            margin-top: 20px;
        }}
        .service-card {{
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #fff;
        }}
        .service-card h3 {{
            margin-top: 0;
            color: #0066cc;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .service-details {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .service-detail {{
            flex: 1;
            min-width: 200px;
        }}
        .service-detail h4 {{
            margin-top: 0;
            color: #0066cc;
        }}
        .tag {{
            display: inline-block;
            background-color: #e1f0ff;
            color: #0066cc;
            padding: 3px 8px;
            border-radius: 3px;
            margin-right: 5px;
            margin-bottom: 5px;
            font-size: 12px;
        }}
        .dependency-matrix {{
            overflow-x: auto;
        }}
        .dependency-matrix table {{
            border-collapse: collapse;
        }}
        .dependency-matrix th, .dependency-matrix td {{
            text-align: center;
            min-width: 40px;
        }}
        .dependency-matrix th {{
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            height: 150px;
        }}
        .dependency-matrix td.has-dependency {{
            background-color: #0066cc;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Forex Trading Platform Architecture Report</h1>
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
                    <h3>Classes</h3>
                    <div class="stat-value">{class_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Functions</h3>
                    <div class="stat-value">{function_count}</div>
                </div>
            </div>

            <h3>Architecture Diagram</h3>
            <div class="visualization">
                <img src="data:image/png;base64,{architecture_diagram}" alt="Architecture Diagram">
            </div>
        </div>

        <div class="section">
            <h2>Service Dependencies</h2>
            <p>This section shows the dependencies between services in the forex trading platform.</p>

            <h3>Dependency Matrix</h3>
            <div class="dependency-matrix">
                <table>
                    <tr>
                        <th></th>
                        {dependency_matrix_headers}
                    </tr>
                    {dependency_matrix_rows}
                </table>

                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #0066cc;"></div>
                        <span>Has Dependency</span>
                    </div>
                </div>
            </div>

            <h3>Most Dependent Services</h3>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Dependencies</th>
                </tr>
                {most_dependent_services_rows}
            </table>

            <h3>Most Depended-on Services</h3>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Dependents</th>
                </tr>
                {most_depended_on_services_rows}
            </table>
        </div>

        <div class="section">
            <h2>Service Structure</h2>
            <p>This section shows the structure of each service in the forex trading platform.</p>

            <div class="tabs">
                {service_tabs}
            </div>

            {service_tab_contents}
        </div>

        <div class="section">
            <h2>API Endpoints</h2>
            <p>This section shows the API endpoints exposed by each service in the forex trading platform.</p>

            <h3>Endpoint Types</h3>
            <div class="stats">
                <div class="stat-card">
                    <h3>REST</h3>
                    <div class="stat-value">{rest_endpoint_count}</div>
                </div>
                <div class="stat-card">
                    <h3>gRPC</h3>
                    <div class="stat-value">{grpc_service_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Message Queues</h3>
                    <div class="stat-value">{message_queue_count}</div>
                </div>
                <div class="stat-card">
                    <h3>WebSocket</h3>
                    <div class="stat-value">{websocket_endpoint_count}</div>
                </div>
            </div>

            <h3>Services with Most Endpoints</h3>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Endpoints</th>
                </tr>
                {services_with_most_endpoints_rows}
            </table>
        </div>

        <div class="section">
            <h2>Database Models</h2>
            <p>This section shows the database models used in the forex trading platform.</p>

            <h3>Model Statistics</h3>
            <div class="stats">
                <div class="stat-card">
                    <h3>Models</h3>
                    <div class="stat-value">{model_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Relationships</h3>
                    <div class="stat-value">{relationship_count}</div>
                </div>
            </div>

            <h3>Services with Most Models</h3>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Models</th>
                </tr>
                {services_with_most_models_rows}
            </table>

            <h3>Most Used Data Access Patterns</h3>
            <table>
                <tr>
                    <th>Pattern</th>
                    <th>Occurrences</th>
                </tr>
                {most_used_data_access_patterns_rows}
            </table>
        </div>

        <div class="section">
            <h2>Architectural Patterns</h2>
            <p>This section shows the architectural patterns used in the forex trading platform.</p>

            <h3>Most Common Patterns</h3>
            <table>
                <tr>
                    <th>Pattern</th>
                    <th>Occurrences</th>
                </tr>
                {most_common_patterns_rows}
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

def load_report(file_path: str) -> Dict[str, Any]:
    """Load a report from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report from {file_path}: {e}")
        return {}

def image_to_base64(image_path: str) -> str:
    """Convert an image to base64."""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def generate_dependency_matrix(dependencies: Dict[str, List[str]], services: List[str]) -> Tuple[str, str]:
    """Generate HTML for the dependency matrix."""
    # Generate headers
    headers = ""
    for service in services:
        headers += f'<th>{service}</th>'

    # Generate rows
    rows = ""
    for source in services:
        row = f'<tr><th>{source}</th>'

        for target in services:
            if target in dependencies.get(source, []):
                row += '<td class="has-dependency"></td>'
            else:
                row += '<td></td>'

        row += '</tr>'
        rows += row

    return headers, rows

def generate_service_tabs(services: List[str]) -> str:
    """Generate HTML for service tabs."""
    tabs = []
    for i, service in enumerate(services):
        active = ' active' if i == 0 else ''
        tabs.append(f'<div class="tab{active}" data-tab="service-{i}">{service}</div>')

    return '\n'.join(tabs)

def generate_service_tab_contents(services: List[str], service_structure: Dict[str, Any], api_endpoints: Dict[str, Any], database_schema: Dict[str, Any]) -> str:
    """Generate HTML for service tab contents."""
    contents = []

    for i, service in enumerate(services):
        active = ' active' if i == 0 else ''

        # Get service structure
        structure = service_structure.get('service_structure', {}).get(service, {})

        # Get service endpoints
        endpoints = api_endpoints.get('service_endpoints', {}).get(service, [])

        # Get service models
        models = []
        for model in database_schema.get('models', []):
            if model.get('file_path', '').startswith(os.path.join(ROOT_DIR, service)):
                models.append(model)

        # Generate HTML
        content = f"""
        <div id="service-{i}" class="tab-content{active}">
            <div class="service-card">
                <h3>{service}</h3>

                <div class="service-details">
                    <div class="service-detail">
                        <h4>Statistics</h4>
                        <p>Files: {structure.get('file_count', 0)}</p>
                        <p>Classes: {structure.get('class_count', 0)}</p>
                        <p>Functions: {structure.get('function_count', 0)}</p>
                        <p>Modules: {structure.get('module_count', 0)}</p>
                        <p>Directories: {structure.get('directory_count', 0)}</p>
                    </div>

                    <div class="service-detail">
                        <h4>Patterns</h4>
                        <div>
        """

        # Add patterns
        for pattern in structure.get('most_common_patterns', [])[:10]:
            content += f'<span class="tag">{pattern["pattern"]} ({pattern["count"]})</span>'

        content += """
                        </div>
                    </div>
                </div>
            </div>

            <div class="service-card">
                <h3>API Endpoints</h3>

                <table>
                    <tr>
                        <th>Type</th>
                        <th>Endpoint</th>
                        <th>Details</th>
                    </tr>
        """

        # Add endpoints
        for endpoint in endpoints[:20]:  # Limit to 20 endpoints
            endpoint_type = endpoint.get('type', '')

            if endpoint_type == 'REST':
                content += f"""
                <tr>
                    <td>{endpoint_type}</td>
                    <td>{endpoint.get('route', '')}</td>
                    <td>{endpoint.get('method', '')}</td>
                </tr>
                """
            elif endpoint_type == 'gRPC':
                content += f"""
                <tr>
                    <td>{endpoint_type}</td>
                    <td>{endpoint.get('service', '')}</td>
                    <td>{endpoint.get('method', '')}</td>
                </tr>
                """
            elif endpoint_type == 'Kafka':
                content += f"""
                <tr>
                    <td>{endpoint_type}</td>
                    <td>{endpoint.get('topic', '')}</td>
                    <td></td>
                </tr>
                """
            elif endpoint_type == 'RabbitMQ':
                content += f"""
                <tr>
                    <td>{endpoint_type}</td>
                    <td>{endpoint.get('queue', '')}</td>
                    <td></td>
                </tr>
                """
            elif endpoint_type == 'WebSocket':
                content += f"""
                <tr>
                    <td>{endpoint_type}</td>
                    <td>{endpoint.get('route', '')}</td>
                    <td></td>
                </tr>
                """
            else:
                content += f"""
                <tr>
                    <td>{endpoint_type}</td>
                    <td></td>
                    <td></td>
                </tr>
                """

        content += """
                </table>
            </div>

            <div class="service-card">
                <h3>Database Models</h3>

                <table>
                    <tr>
                        <th>Model</th>
                        <th>Fields</th>
                        <th>Relationships</th>
                    </tr>
        """

        # Add models
        for model in models[:20]:  # Limit to 20 models
            content += f"""
            <tr>
                <td>{model.get('name', '')}</td>
                <td>{', '.join(model.get('fields', []))}</td>
                <td>{', '.join(model.get('relationships', []))}</td>
            </tr>
            """

        content += """
                </table>
            </div>
        </div>
        """

        contents.append(content)

    return '\n'.join(contents)

def generate_table_rows(items: List[Dict[str, Any]], key1: str, key2: str, limit: int = 10) -> str:
    """Generate HTML table rows."""
    rows = []

    for item in items[:limit]:
        rows.append(f"""
        <tr>
            <td>{item.get(key1, '')}</td>
            <td>{item.get(key2, 0)}</td>
        </tr>
        """)

    return '\n'.join(rows)

def generate_architecture_report() -> None:
    """Generate a comprehensive architecture report."""
    # Load reports
    dependency_report_path = os.path.join(ROOT_DIR, 'tools', 'output', 'dependency-report.json')
    service_structure_report_path = os.path.join(ROOT_DIR, 'tools', 'output', 'service-structure-report.json')
    api_endpoints_report_path = os.path.join(ROOT_DIR, 'tools', 'output', 'api-endpoints-report.json')
    database_schema_report_path = os.path.join(ROOT_DIR, 'tools', 'output', 'database-schema-report.json')

    dependency_report = load_report(dependency_report_path)
    service_structure_report = load_report(service_structure_report_path)
    api_endpoints_report = load_report(api_endpoints_report_path)
    database_schema_report = load_report(database_schema_report_path)

    # Load architecture diagram
    architecture_diagram_path = os.path.join(ROOT_DIR, 'tools', 'output', 'architecture_diagrams', 'architecture-diagram.png')
    architecture_diagram = image_to_base64(architecture_diagram_path)

    # Get services
    services = list(dependency_report.get('dependencies', {}).keys())

    # Generate dependency matrix
    dependency_matrix_headers, dependency_matrix_rows = generate_dependency_matrix(
        dependency_report.get('dependencies', {}),
        services
    )

    # Generate service tabs and tab contents
    service_tabs = generate_service_tabs(services)
    service_tab_contents = generate_service_tab_contents(
        services,
        service_structure_report,
        api_endpoints_report.get('api_endpoints', {}),
        database_schema_report.get('database_schema', {})
    )

    # Generate HTML
    html = HTML_TEMPLATE.format(
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        service_count=len(services),
        file_count=service_structure_report.get('summary', {}).get('total_files', 0),
        class_count=service_structure_report.get('summary', {}).get('total_classes', 0),
        function_count=service_structure_report.get('summary', {}).get('total_functions', 0),
        architecture_diagram=architecture_diagram,
        dependency_matrix_headers=dependency_matrix_headers,
        dependency_matrix_rows=dependency_matrix_rows,
        most_dependent_services_rows=generate_table_rows(
            dependency_report.get('services_with_most_dependencies', []),
            'service',
            'count'
        ),
        most_depended_on_services_rows=generate_table_rows(
            dependency_report.get('most_depended_on_services', []),
            'service',
            'count'
        ),
        service_tabs=service_tabs,
        service_tab_contents=service_tab_contents,
        rest_endpoint_count=api_endpoints_report.get('summary', {}).get('total_rest_endpoints', 0),
        grpc_service_count=api_endpoints_report.get('summary', {}).get('total_grpc_services', 0),
        message_queue_count=api_endpoints_report.get('summary', {}).get('total_message_queues', 0),
        websocket_endpoint_count=api_endpoints_report.get('summary', {}).get('total_websocket_endpoints', 0),
        services_with_most_endpoints_rows=generate_table_rows(
            [{'service': service, 'count': count} for service, count in api_endpoints_report.get('summary', {}).get('endpoints_per_service', {}).items()],
            'service',
            'count'
        ),
        model_count=database_schema_report.get('summary', {}).get('total_models', 0),
        relationship_count=database_schema_report.get('summary', {}).get('total_relationships', 0),
        services_with_most_models_rows=generate_table_rows(
            [{'service': service, 'count': count} for service, count in database_schema_report.get('summary', {}).get('models_per_service', {}).items()],
            'service',
            'count'
        ),
        most_used_data_access_patterns_rows=generate_table_rows(
            database_schema_report.get('summary', {}).get('most_used_data_access_patterns', []),
            'pattern',
            'count'
        ),
        most_common_patterns_rows=generate_table_rows(
            service_structure_report.get('summary', {}).get('most_common_patterns', []),
            'pattern',
            'count'
        )
    )

    # Save HTML report
    output_path = os.path.join(ROOT_DIR, 'tools', 'output', 'architecture-report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Architecture report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate architecture report for the forex trading platform')
    parser.add_argument('--output-file', default='architecture-report.html', help='Output file for the architecture report')
    args = parser.parse_args()

    print("Generating architecture report...")
    generate_architecture_report()

if __name__ == "__main__":
    main()