#!/usr/bin/env python3
"""
Analyze Service Structure in the Forex Trading Platform

This script analyzes the structure of each service in the forex trading platform,
identifying key components, patterns, and architectural characteristics.

Usage:
    python analyze_service_structure.py [--output-file OUTPUT_FILE]

Options:
    --output-file OUTPUT_FILE    Output file for the service structure report (default: service-structure-report.json)
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any

# Root directory of the forex trading platform
ROOT_DIR = "D:/MD/forex_trading_platform"

# Service directories to analyze
SERVICE_DIRS = [
    'analysis-engine-service',
    'api-gateway',
    'data-management-service',
    'data-pipeline-service',
    'feature-store-service',
    'ml-integration-service',
    'ml-workbench-service',
    'model-registry-service',
    'monitoring-alerting-service',
    'portfolio-management-service',
    'risk-management-service',
    'strategy-execution-engine',
    'trading-gateway-service',
    'ui-service',
    'common-lib',
    'common-js-lib'
]

# Common architectural patterns to identify
PATTERNS = {
    'repository': r'(?:class|def)\s+\w*Repository',
    'service': r'(?:class|def)\s+\w*Service',
    'controller': r'(?:class|def)\s+\w*Controller',
    'factory': r'(?:class|def)\s+\w*Factory',
    'adapter': r'(?:class|def)\s+\w*Adapter',
    'strategy': r'(?:class|def)\s+\w*Strategy',
    'observer': r'(?:class|def)\s+\w*Observer',
    'decorator': r'(?:class|def)\s+\w*Decorator',
    'singleton': r'(?:_instance|__instance)\s*=\s*None',
    'dto': r'(?:class|def)\s+\w*DTO',
    'model': r'(?:class|def)\s+\w*Model',
    'entity': r'(?:class|def)\s+\w*Entity',
    'validator': r'(?:class|def)\s+\w*Validator',
    'middleware': r'(?:class|def)\s+\w*Middleware',
    'handler': r'(?:class|def)\s+\w*Handler',
    'client': r'(?:class|def)\s+\w*Client',
    'provider': r'(?:class|def)\s+\w*Provider',
    'manager': r'(?:class|def)\s+\w*Manager',
    'builder': r'(?:class|def)\s+\w*Builder',
    'facade': r'(?:class|def)\s+\w*Facade',
    'proxy': r'(?:class|def)\s+\w*Proxy',
    'command': r'(?:class|def)\s+\w*Command',
    'event': r'(?:class|def)\s+\w*Event',
    'listener': r'(?:class|def)\s+\w*Listener',
    'publisher': r'(?:class|def)\s+\w*Publisher',
    'subscriber': r'(?:class|def)\s+\w*Subscriber',
    'worker': r'(?:class|def)\s+\w*Worker',
    'processor': r'(?:class|def)\s+\w*Processor',
    'transformer': r'(?:class|def)\s+\w*Transformer',
    'filter': r'(?:class|def)\s+\w*Filter',
    'router': r'(?:class|def)\s+\w*Router',
    'gateway': r'(?:class|def)\s+\w*Gateway',
    'connector': r'(?:class|def)\s+\w*Connector',
    'executor': r'(?:class|def)\s+\w*Executor',
    'scheduler': r'(?:class|def)\s+\w*Scheduler',
    'monitor': r'(?:class|def)\s+\w*Monitor',
    'logger': r'(?:class|def)\s+\w*Logger',
    'cache': r'(?:class|def)\s+\w*Cache',
    'store': r'(?:class|def)\s+\w*Store',
    'registry': r'(?:class|def)\s+\w*Registry',
    'container': r'(?:class|def)\s+\w*Container',
    'context': r'(?:class|def)\s+\w*Context',
    'config': r'(?:class|def)\s+\w*Config',
    'settings': r'(?:class|def)\s+\w*Settings',
    'util': r'(?:class|def)\s+\w*Util',
    'helper': r'(?:class|def)\s+\w*Helper',
    'exception': r'(?:class|def)\s+\w*Exception',
    'error': r'(?:class|def)\s+\w*Error',
    'test': r'(?:class|def)\s+\w*Test',
    'mock': r'(?:class|def)\s+\w*Mock',
    'stub': r'(?:class|def)\s+\w*Stub',
    'fixture': r'(?:class|def)\s+\w*Fixture',
    'factory_method': r'def\s+create_\w+',
    'dependency_injection': r'def\s+__init__\([^)]*\w+_service',
    'async_pattern': r'async\s+def',
    'event_driven': r'(?:publish|subscribe|emit|on_event)',
    'circuit_breaker': r'circuit_breaker',
    'retry': r'retry',
    'bulkhead': r'bulkhead',
    'timeout': r'timeout',
    'fallback': r'fallback',
    'rate_limiter': r'rate_limit',
    'cache_aside': r'cache\.get|cache\.set',
    'saga': r'saga',
    'cqrs': r'command|query',
    'api_gateway': r'api_gateway',
    'service_registry': r'service_registry',
    'config_server': r'config_server',
    'load_balancer': r'load_balance',
    'circuit_breaker': r'circuit_break',
    'api_composition': r'compose_api',
    'backend_for_frontend': r'bff',
    'strangler': r'strangler',
    'sidecar': r'sidecar',
    'ambassador': r'ambassador',
    'anti_corruption_layer': r'anti_corruption',
    'gateway_routing': r'gateway_routing',
    'gateway_aggregation': r'gateway_aggregation',
    'gateway_offloading': r'gateway_offloading',
    'sharding': r'shard',
    'static_content_hosting': r'static_content',
    'health_endpoint_monitoring': r'health_check',
    'log_aggregation': r'log_aggregation',
    'performance_monitoring': r'performance_monitor',
    'distributed_tracing': r'tracing',
    'external_configuration_store': r'external_config',
    'runtime_reconfiguration': r'runtime_config',
    'federated_identity': r'federated_identity',
    'gatekeeper': r'gatekeeper',
    'valet_key': r'valet_key',
    'throttling': r'throttle',
    'cache_aside': r'cache_aside',
    'materialized_view': r'materialized_view',
    'static_content_hosting': r'static_content',
    'claim_check': r'claim_check',
    'competing_consumers': r'competing_consumers',
    'pipes_and_filters': r'pipes_and_filters',
    'priority_queue': r'priority_queue',
    'queue_based_load_leveling': r'load_leveling',
    'scheduler_agent_supervisor': r'scheduler_agent_supervisor',
    'sequential_convoy': r'sequential_convoy',
    'asynchronous_request_reply': r'async_request_reply',
    'compensating_transaction': r'compensating_transaction',
    'leader_election': r'leader_election',
    'publisher_subscriber': r'pub_sub',
    'index_table': r'index_table',
    'materialized_view': r'materialized_view',
    'outbox': r'outbox',
    'saga': r'saga',
    'domain_event': r'domain_event',
    'event_sourcing': r'event_sourcing',
    'command_query_responsibility_segregation': r'cqrs',
    'specification': r'specification',
    'repository': r'repository',
    'unit_of_work': r'unit_of_work',
    'aggregate': r'aggregate',
    'entity': r'entity',
    'value_object': r'value_object',
    'domain_service': r'domain_service',
    'factory': r'factory',
    'bounded_context': r'bounded_context',
    'anti_corruption_layer': r'anti_corruption_layer',
    'shared_kernel': r'shared_kernel',
    'customer_supplier': r'customer_supplier',
    'conformist': r'conformist',
    'open_host_service': r'open_host_service',
    'published_language': r'published_language',
    'separate_ways': r'separate_ways',
    'big_ball_of_mud': r'big_ball_of_mud',
    'domain_model': r'domain_model',
    'transaction_script': r'transaction_script',
    'table_module': r'table_module',
    'service_layer': r'service_layer',
    'active_record': r'active_record',
    'data_mapper': r'data_mapper',
    'table_data_gateway': r'table_data_gateway',
    'row_data_gateway': r'row_data_gateway',
    'lazy_load': r'lazy_load',
    'identity_map': r'identity_map',
    'unit_of_work': r'unit_of_work',
    'plugin': r'plugin',
    'service_stub': r'service_stub',
    'remote_facade': r'remote_facade',
    'data_transfer_object': r'data_transfer_object',
    'assembler': r'assembler',
    'money': r'money',
    'special_case': r'special_case',
    'plugin': r'plugin',
    'service_locator': r'service_locator',
    'record_set': r'record_set',
    'layer_supertype': r'layer_supertype',
    'separated_interface': r'separated_interface',
    'gateway': r'gateway',
    'mapper': r'mapper',
    'metadata_mapping': r'metadata_mapping',
    'query_object': r'query_object',
    'repository': r'repository',
    'value_object': r'value_object',
    'money': r'money',
    'null_object': r'null_object',
    'lazy_initialization': r'lazy_initialization',
    'virtual_proxy': r'virtual_proxy',
    'decorator': r'decorator',
    'flyweight': r'flyweight',
    'foreign_key_mapping': r'foreign_key_mapping',
    'embedded_value': r'embedded_value',
    'serialized_lob': r'serialized_lob',
    'single_table_inheritance': r'single_table_inheritance',
    'class_table_inheritance': r'class_table_inheritance',
    'concrete_table_inheritance': r'concrete_table_inheritance',
    'inheritance_mappers': r'inheritance_mappers',
    'dependent_mapping': r'dependent_mapping',
    'aggregate_mapping': r'aggregate_mapping',
    'identity_field': r'identity_field',
    'sequence_generator': r'sequence_generator',
    'value_object': r'value_object',
    'money': r'money',
    'special_case': r'special_case',
    'plugin': r'plugin',
    'service_locator': r'service_locator',
    'record_set': r'record_set',
    'layer_supertype': r'layer_supertype',
    'separated_interface': r'separated_interface',
    'gateway': r'gateway',
    'mapper': r'mapper',
    'metadata_mapping': r'metadata_mapping',
    'query_object': r'query_object',
    'repository': r'repository',
    'value_object': r'value_object',
    'money': r'money',
    'null_object': r'null_object',
    'lazy_initialization': r'lazy_initialization',
    'virtual_proxy': r'virtual_proxy',
    'decorator': r'decorator',
    'flyweight': r'flyweight',
    'foreign_key_mapping': r'foreign_key_mapping',
    'embedded_value': r'embedded_value',
    'serialized_lob': r'serialized_lob',
    'single_table_inheritance': r'single_table_inheritance',
    'class_table_inheritance': r'class_table_inheritance',
    'concrete_table_inheritance': r'concrete_table_inheritance',
    'inheritance_mappers': r'inheritance_mappers',
    'dependent_mapping': r'dependent_mapping',
    'aggregate_mapping': r'aggregate_mapping',
    'identity_field': r'identity_field',
    'sequence_generator': r'sequence_generator'
}

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in a directory recursively."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_classes_and_functions(file_path: str) -> Tuple[List[str], List[str]]:
    """Extract class and function definitions from a Python file."""
    classes = []
    functions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract class definitions
            class_pattern = r'class\s+(\w+)'
            classes = re.findall(class_pattern, content)
            
            # Extract function definitions
            function_pattern = r'def\s+(\w+)'
            functions = re.findall(function_pattern, content)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return classes, functions

def identify_patterns(file_path: str) -> Dict[str, int]:
    """Identify architectural patterns in a Python file."""
    pattern_counts = defaultdict(int)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for pattern_name, pattern_regex in PATTERNS.items():
                matches = re.findall(pattern_regex, content, re.IGNORECASE)
                pattern_counts[pattern_name] = len(matches)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return pattern_counts

def analyze_service_structure() -> Dict[str, Any]:
    """Analyze the structure of each service."""
    service_structure = {}
    
    for service in SERVICE_DIRS:
        service_dir = os.path.join(ROOT_DIR, service)
        if not os.path.isdir(service_dir):
            print(f"Warning: Service directory {service_dir} not found")
            continue
        
        python_files = find_python_files(service_dir)
        
        # Initialize service structure
        service_structure[service] = {
            'file_count': len(python_files),
            'classes': [],
            'functions': [],
            'patterns': defaultdict(int),
            'modules': [],
            'directory_structure': []
        }
        
        # Analyze files
        for file in python_files:
            # Extract classes and functions
            classes, functions = extract_classes_and_functions(file)
            service_structure[service]['classes'].extend(classes)
            service_structure[service]['functions'].extend(functions)
            
            # Identify patterns
            patterns = identify_patterns(file)
            for pattern, count in patterns.items():
                service_structure[service]['patterns'][pattern] += count
            
            # Extract module name
            relative_path = os.path.relpath(file, service_dir)
            module_name = os.path.splitext(relative_path)[0].replace('\\', '.').replace('/', '.')
            service_structure[service]['modules'].append(module_name)
        
        # Extract directory structure
        for root, dirs, files in os.walk(service_dir):
            relative_path = os.path.relpath(root, service_dir)
            if relative_path == '.':
                relative_path = ''
            
            # Skip __pycache__ and other hidden directories
            if '__pycache__' in relative_path or '/.git' in relative_path:
                continue
            
            # Count Python files in this directory
            py_files = [f for f in files if f.endswith('.py')]
            
            if py_files:  # Only include directories with Python files
                service_structure[service]['directory_structure'].append({
                    'path': relative_path,
                    'file_count': len(py_files)
                })
        
        # Convert defaultdict to dict for JSON serialization
        service_structure[service]['patterns'] = dict(service_structure[service]['patterns'])
        
        # Sort classes and functions by name
        service_structure[service]['classes'] = sorted(service_structure[service]['classes'])
        service_structure[service]['functions'] = sorted(service_structure[service]['functions'])
        
        # Sort modules by name
        service_structure[service]['modules'] = sorted(service_structure[service]['modules'])
        
        # Sort directory structure by path
        service_structure[service]['directory_structure'] = sorted(
            service_structure[service]['directory_structure'],
            key=lambda x: x['path']
        )
        
        # Add summary statistics
        service_structure[service]['class_count'] = len(service_structure[service]['classes'])
        service_structure[service]['function_count'] = len(service_structure[service]['functions'])
        service_structure[service]['module_count'] = len(service_structure[service]['modules'])
        service_structure[service]['directory_count'] = len(service_structure[service]['directory_structure'])
        
        # Identify most common patterns
        most_common_patterns = sorted(
            service_structure[service]['patterns'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        service_structure[service]['most_common_patterns'] = [
            {'pattern': pattern, 'count': count}
            for pattern, count in most_common_patterns if count > 0
        ]
    
    return service_structure

def generate_service_structure_report(service_structure: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive service structure report."""
    # Calculate overall statistics
    total_files = sum(service['file_count'] for service in service_structure.values())
    total_classes = sum(service['class_count'] for service in service_structure.values())
    total_functions = sum(service['function_count'] for service in service_structure.values())
    total_modules = sum(service['module_count'] for service in service_structure.values())
    total_directories = sum(service['directory_count'] for service in service_structure.values())
    
    # Aggregate patterns across all services
    all_patterns = defaultdict(int)
    for service in service_structure.values():
        for pattern, count in service['patterns'].items():
            all_patterns[pattern] += count
    
    # Identify most common patterns across all services
    most_common_patterns = sorted(
        all_patterns.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]  # Top 20
    
    # Identify services with most files, classes, functions
    services_by_files = sorted(
        [(service, data['file_count']) for service, data in service_structure.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    services_by_classes = sorted(
        [(service, data['class_count']) for service, data in service_structure.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    services_by_functions = sorted(
        [(service, data['function_count']) for service, data in service_structure.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Generate report
    report = {
        'service_structure': service_structure,
        'summary': {
            'total_services': len(service_structure),
            'total_files': total_files,
            'total_classes': total_classes,
            'total_functions': total_functions,
            'total_modules': total_modules,
            'total_directories': total_directories,
            'most_common_patterns': [
                {'pattern': pattern, 'count': count}
                for pattern, count in most_common_patterns if count > 0
            ],
            'services_by_files': [
                {'service': service, 'count': count}
                for service, count in services_by_files
            ],
            'services_by_classes': [
                {'service': service, 'count': count}
                for service, count in services_by_classes
            ],
            'services_by_functions': [
                {'service': service, 'count': count}
                for service, count in services_by_functions
            ]
        }
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Analyze service structure in the forex trading platform')
    parser.add_argument('--output-file', default='service-structure-report.json', help='Output file for the service structure report')
    args = parser.parse_args()
    
    print("Analyzing service structure...")
    service_structure = analyze_service_structure()
    
    print("Generating service structure report...")
    report = generate_service_structure_report(service_structure)
    
    # Save report to file
    output_path = os.path.join(ROOT_DIR, 'tools', 'output', args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Service structure report saved to {output_path}")
    
    # Print summary
    print("\nService Structure Analysis Summary:")
    print(f"Total services: {report['summary']['total_services']}")
    print(f"Total files: {report['summary']['total_files']}")
    print(f"Total classes: {report['summary']['total_classes']}")
    print(f"Total functions: {report['summary']['total_functions']}")
    
    print("\nServices with Most Files:")
    for item in report['summary']['services_by_files'][:5]:  # Top 5
        print(f"{item['service']}: {item['count']} files")
    
    print("\nMost Common Patterns:")
    for item in report['summary']['most_common_patterns'][:10]:  # Top 10
        print(f"{item['pattern']}: {item['count']} occurrences")

if __name__ == "__main__":
    main()