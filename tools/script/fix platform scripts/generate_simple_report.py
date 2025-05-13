#!/usr/bin/env python3
"""
Generate Simple Architecture Report for the Forex Trading Platform

This script generates a simple architecture report for the forex trading platform
by combining the results from various analysis scripts.

Usage:
    python generate_simple_report.py [--output-file OUTPUT_FILE]

Options:
    --output-file OUTPUT_FILE    Output file for the architecture report (default: architecture-report.md)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Root directory of the forex trading platform
ROOT_DIR = "D:/MD/forex_trading_platform"

def load_report(file_path: str) -> Dict[str, Any]:
    """Load a report from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report from {file_path}: {e}")
        return {}

def generate_simple_report() -> None:
    """Generate a simple architecture report."""
    # Load reports
    dependency_report_path = os.path.join(ROOT_DIR, 'tools', 'output', 'dependency-report.json')
    service_structure_report_path = os.path.join(ROOT_DIR, 'tools', 'output', 'service-structure-report.json')
    api_endpoints_report_path = os.path.join(ROOT_DIR, 'tools', 'output', 'api-endpoints-report.json')
    database_schema_report_path = os.path.join(ROOT_DIR, 'tools', 'output', 'database-schema-report.json')
    
    dependency_report = load_report(dependency_report_path)
    service_structure_report = load_report(service_structure_report_path)
    api_endpoints_report = load_report(api_endpoints_report_path)
    database_schema_report = load_report(database_schema_report_path)
    
    # Generate markdown report
    report = []
    
    # Add header
    report.append("# Forex Trading Platform Architecture Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Add project overview
    report.append("## Project Overview")
    report.append("")
    
    services = list(dependency_report.get('dependencies', {}).keys())
    file_count = service_structure_report.get('summary', {}).get('total_files', 0)
    class_count = service_structure_report.get('summary', {}).get('total_classes', 0)
    function_count = service_structure_report.get('summary', {}).get('total_functions', 0)
    
    report.append(f"- **Services:** {len(services)}")
    report.append(f"- **Files:** {file_count}")
    report.append(f"- **Classes:** {class_count}")
    report.append(f"- **Functions:** {function_count}")
    report.append("")
    
    # Add service dependencies
    report.append("## Service Dependencies")
    report.append("")
    
    report.append("### Most Dependent Services")
    report.append("")
    report.append("| Service | Dependencies |")
    report.append("| --- | --- |")
    
    for item in dependency_report.get('services_with_most_dependencies', []):
        report.append(f"| {item.get('service', '')} | {item.get('count', 0)} |")
    
    report.append("")
    
    report.append("### Most Depended-on Services")
    report.append("")
    report.append("| Service | Dependents |")
    report.append("| --- | --- |")
    
    for item in dependency_report.get('most_depended_on_services', []):
        report.append(f"| {item.get('service', '')} | {item.get('count', 0)} |")
    
    report.append("")
    
    # Add service structure
    report.append("## Service Structure")
    report.append("")
    
    report.append("### Services with Most Files")
    report.append("")
    report.append("| Service | Files |")
    report.append("| --- | --- |")
    
    for item in service_structure_report.get('summary', {}).get('services_by_files', [])[:10]:
        report.append(f"| {item.get('service', '')} | {item.get('count', 0)} |")
    
    report.append("")
    
    report.append("### Most Common Patterns")
    report.append("")
    report.append("| Pattern | Occurrences |")
    report.append("| --- | --- |")
    
    for item in service_structure_report.get('summary', {}).get('most_common_patterns', [])[:10]:
        report.append(f"| {item.get('pattern', '')} | {item.get('count', 0)} |")
    
    report.append("")
    
    # Add API endpoints
    report.append("## API Endpoints")
    report.append("")
    
    rest_endpoint_count = api_endpoints_report.get('summary', {}).get('total_rest_endpoints', 0)
    grpc_service_count = api_endpoints_report.get('summary', {}).get('total_grpc_services', 0)
    message_queue_count = api_endpoints_report.get('summary', {}).get('total_message_queues', 0)
    websocket_endpoint_count = api_endpoints_report.get('summary', {}).get('total_websocket_endpoints', 0)
    
    report.append(f"- **REST Endpoints:** {rest_endpoint_count}")
    report.append(f"- **gRPC Services:** {grpc_service_count}")
    report.append(f"- **Message Queues:** {message_queue_count}")
    report.append(f"- **WebSocket Endpoints:** {websocket_endpoint_count}")
    report.append("")
    
    report.append("### Services with Most Endpoints")
    report.append("")
    report.append("| Service | Endpoints |")
    report.append("| --- | --- |")
    
    endpoints_per_service = api_endpoints_report.get('summary', {}).get('endpoints_per_service', {})
    for service, count in sorted(endpoints_per_service.items(), key=lambda x: x[1], reverse=True)[:10]:
        report.append(f"| {service} | {count} |")
    
    report.append("")
    
    # Add database models
    report.append("## Database Models")
    report.append("")
    
    model_count = database_schema_report.get('summary', {}).get('total_models', 0)
    relationship_count = database_schema_report.get('summary', {}).get('total_relationships', 0)
    
    report.append(f"- **Models:** {model_count}")
    report.append(f"- **Relationships:** {relationship_count}")
    report.append("")
    
    report.append("### Services with Most Models")
    report.append("")
    report.append("| Service | Models |")
    report.append("| --- | --- |")
    
    models_per_service = database_schema_report.get('summary', {}).get('models_per_service', {})
    for service, count in sorted(models_per_service.items(), key=lambda x: x[1], reverse=True)[:10]:
        report.append(f"| {service} | {count} |")
    
    report.append("")
    
    report.append("### Most Used Data Access Patterns")
    report.append("")
    report.append("| Pattern | Occurrences |")
    report.append("| --- | --- |")
    
    for item in database_schema_report.get('summary', {}).get('most_used_data_access_patterns', [])[:10]:
        report.append(f"| {item.get('pattern', '')} | {item.get('count', 0)} |")
    
    report.append("")
    
    # Add service details
    report.append("## Service Details")
    report.append("")
    
    for service in services:
        report.append(f"### {service}")
        report.append("")
        
        # Get service structure
        structure = service_structure_report.get('service_structure', {}).get(service, {})
        
        report.append("#### Statistics")
        report.append("")
        report.append(f"- **Files:** {structure.get('file_count', 0)}")
        report.append(f"- **Classes:** {structure.get('class_count', 0)}")
        report.append(f"- **Functions:** {structure.get('function_count', 0)}")
        report.append(f"- **Modules:** {structure.get('module_count', 0)}")
        report.append(f"- **Directories:** {structure.get('directory_count', 0)}")
        report.append("")
        
        report.append("#### Dependencies")
        report.append("")
        
        dependencies = dependency_report.get('dependencies', {}).get(service, [])
        if dependencies:
            report.append("This service depends on:")
            report.append("")
            for dep in dependencies:
                report.append(f"- {dep}")
        else:
            report.append("This service has no dependencies.")
        
        report.append("")
        
        # Get service endpoints
        endpoints = api_endpoints_report.get('api_endpoints', {}).get('service_endpoints', {}).get(service, [])
        
        if endpoints:
            report.append("#### API Endpoints")
            report.append("")
            report.append("| Type | Endpoint | Details |")
            report.append("| --- | --- | --- |")
            
            for endpoint in endpoints[:20]:  # Limit to 20 endpoints
                endpoint_type = endpoint.get('type', '')
                
                if endpoint_type == 'REST':
                    report.append(f"| {endpoint_type} | {endpoint.get('route', '')} | {endpoint.get('method', '')} |")
                elif endpoint_type == 'gRPC':
                    report.append(f"| {endpoint_type} | {endpoint.get('service', '')} | {endpoint.get('method', '')} |")
                elif endpoint_type == 'Kafka':
                    report.append(f"| {endpoint_type} | {endpoint.get('topic', '')} | |")
                elif endpoint_type == 'RabbitMQ':
                    report.append(f"| {endpoint_type} | {endpoint.get('queue', '')} | |")
                elif endpoint_type == 'WebSocket':
                    report.append(f"| {endpoint_type} | {endpoint.get('route', '')} | |")
                else:
                    report.append(f"| {endpoint_type} | | |")
            
            report.append("")
        
        # Get most common patterns
        patterns = structure.get('most_common_patterns', [])
        
        if patterns:
            report.append("#### Most Common Patterns")
            report.append("")
            report.append("| Pattern | Occurrences |")
            report.append("| --- | --- |")
            
            for pattern in patterns[:10]:  # Limit to 10 patterns
                report.append(f"| {pattern.get('pattern', '')} | {pattern.get('count', 0)} |")
            
            report.append("")
    
    # Save report
    output_path = os.path.join(ROOT_DIR, 'tools', 'output', 'architecture-report.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Architecture report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate simple architecture report for the forex trading platform')
    parser.add_argument('--output-file', default='architecture-report.md', help='Output file for the architecture report')
    args = parser.parse_args()
    
    print("Generating architecture report...")
    generate_simple_report()

if __name__ == "__main__":
    main()