#!/usr/bin/env python3
"""
Analyze Database Schema and Data Flow in the Forex Trading Platform

This script analyzes the database schema and data flow in the forex trading platform,
identifying database models, relationships, and data access patterns.

Usage:
    python analyze_database_schema.py [--output-file OUTPUT_FILE]

Options:
    --output-file OUTPUT_FILE    Output file for the database schema report (default: database-schema-report.json)
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict
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

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in a directory recursively."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_model_classes(file_path: str) -> List[Dict[str, Any]]:
    """Extract model classes from a Python file."""
    models = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for SQLAlchemy model definitions
            class_pattern = r'class\s+(\w+)\s*\([^)]*(?:Base|db\.Model|Model)[^)]*\):'
            class_matches = re.finditer(class_pattern, content, re.MULTILINE)
            
            for match in class_matches:
                model_name = match.group(1)
                model_start = match.start()
                
                # Find the end of the class definition
                # This is a simplistic approach and might not work for complex class definitions
                next_class_match = re.search(r'class\s+\w+', content[model_start + 1:])
                if next_class_match:
                    model_end = model_start + 1 + next_class_match.start()
                else:
                    model_end = len(content)
                
                model_content = content[model_start:model_end]
                
                # Extract fields
                field_pattern = r'(\w+)\s*=\s*(?:db\.)?Column\('
                fields = re.findall(field_pattern, model_content)
                
                # Extract relationships
                relationship_pattern = r'(\w+)\s*=\s*(?:db\.)?relationship\('
                relationships = re.findall(relationship_pattern, model_content)
                
                # Extract foreign keys
                foreign_key_pattern = r'ForeignKey\([\'"]([^\'"]+)[\'"]\)'
                foreign_keys = re.findall(foreign_key_pattern, model_content)
                
                models.append({
                    'name': model_name,
                    'fields': fields,
                    'relationships': relationships,
                    'foreign_keys': foreign_keys,
                    'file_path': file_path
                })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return models

def extract_data_access_patterns(file_path: str) -> Dict[str, int]:
    """Extract data access patterns from a Python file."""
    patterns = {
        'select': 0,
        'insert': 0,
        'update': 0,
        'delete': 0,
        'join': 0,
        'group_by': 0,
        'order_by': 0,
        'limit': 0,
        'offset': 0,
        'filter': 0,
        'query': 0,
        'execute': 0,
        'commit': 0,
        'rollback': 0,
        'transaction': 0,
        'session': 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()  # Convert to lowercase for case-insensitive matching
            
            # Count occurrences of each pattern
            patterns['select'] = content.count('select(') + content.count('select ')
            patterns['insert'] = content.count('insert(') + content.count('insert ')
            patterns['update'] = content.count('update(') + content.count('update ')
            patterns['delete'] = content.count('delete(') + content.count('delete ')
            patterns['join'] = content.count('join(') + content.count('join ')
            patterns['group_by'] = content.count('group_by(') + content.count('group by')
            patterns['order_by'] = content.count('order_by(') + content.count('order by')
            patterns['limit'] = content.count('limit(') + content.count('limit ')
            patterns['offset'] = content.count('offset(') + content.count('offset ')
            patterns['filter'] = content.count('filter(') + content.count('filter ')
            patterns['query'] = content.count('query.') + content.count('query(')
            patterns['execute'] = content.count('execute(') + content.count('execute ')
            patterns['commit'] = content.count('commit(') + content.count('commit ')
            patterns['rollback'] = content.count('rollback(') + content.count('rollback ')
            patterns['transaction'] = content.count('transaction') + content.count('begin(')
            patterns['session'] = content.count('session.') + content.count('session(')
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return patterns

def analyze_database_schema() -> Dict[str, Any]:
    """Analyze the database schema and data flow."""
    database_schema = {
        'models': [],
        'data_access_patterns': defaultdict(int),
        'service_models': defaultdict(list),
        'model_relationships': []
    }
    
    for service in SERVICE_DIRS:
        service_dir = os.path.join(ROOT_DIR, service)
        if not os.path.isdir(service_dir):
            print(f"Warning: Service directory {service_dir} not found")
            continue
        
        python_files = find_python_files(service_dir)
        
        for file in python_files:
            # Extract model classes
            models = extract_model_classes(file)
            database_schema['models'].extend(models)
            
            # Add models to service_models
            for model in models:
                database_schema['service_models'][service].append(model['name'])
            
            # Extract data access patterns
            patterns = extract_data_access_patterns(file)
            for pattern, count in patterns.items():
                database_schema['data_access_patterns'][pattern] += count
    
    # Identify model relationships
    model_name_to_model = {model['name']: model for model in database_schema['models']}
    
    for model in database_schema['models']:
        for relationship in model['relationships']:
            # Try to find the target model
            for target_model in database_schema['models']:
                if relationship.lower() in [field.lower() for field in target_model['fields']]:
                    database_schema['model_relationships'].append({
                        'source': model['name'],
                        'target': target_model['name'],
                        'type': 'relationship'
                    })
        
        for foreign_key in model['foreign_keys']:
            # Extract table name from foreign key
            table_name = foreign_key.split('.')[0] if '.' in foreign_key else foreign_key
            
            # Try to find the target model
            for target_model in database_schema['models']:
                if table_name.lower() == target_model['name'].lower():
                    database_schema['model_relationships'].append({
                        'source': model['name'],
                        'target': target_model['name'],
                        'type': 'foreign_key'
                    })
    
    # Convert defaultdict to dict for JSON serialization
    database_schema['data_access_patterns'] = dict(database_schema['data_access_patterns'])
    database_schema['service_models'] = dict(database_schema['service_models'])
    
    return database_schema

def generate_database_schema_report(database_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive database schema report."""
    # Calculate statistics
    total_models = len(database_schema['models'])
    total_relationships = len(database_schema['model_relationships'])
    
    # Count models per service
    models_per_service = {
        service: len(models)
        for service, models in database_schema['service_models'].items()
    }
    
    # Identify most used data access patterns
    most_used_patterns = sorted(
        database_schema['data_access_patterns'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Identify most connected models
    model_connections = defaultdict(int)
    for relationship in database_schema['model_relationships']:
        model_connections[relationship['source']] += 1
        model_connections[relationship['target']] += 1
    
    most_connected_models = sorted(
        model_connections.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]  # Top 10
    
    # Generate report
    report = {
        'database_schema': database_schema,
        'summary': {
            'total_models': total_models,
            'total_relationships': total_relationships,
            'models_per_service': models_per_service,
            'most_used_data_access_patterns': [
                {'pattern': pattern, 'count': count}
                for pattern, count in most_used_patterns
            ],
            'most_connected_models': [
                {'model': model, 'connections': connections}
                for model, connections in most_connected_models
            ]
        }
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Analyze database schema in the forex trading platform')
    parser.add_argument('--output-file', default='database-schema-report.json', help='Output file for the database schema report')
    args = parser.parse_args()
    
    print("Analyzing database schema...")
    database_schema = analyze_database_schema()
    
    print("Generating database schema report...")
    report = generate_database_schema_report(database_schema)
    
    # Save report to file
    output_path = os.path.join(ROOT_DIR, 'tools', 'output', args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Database schema report saved to {output_path}")
    
    # Print summary
    print("\nDatabase Schema Analysis Summary:")
    print(f"Total models: {report['summary']['total_models']}")
    print(f"Total relationships: {report['summary']['total_relationships']}")
    
    print("\nModels per Service:")
    for service, count in sorted(report['summary']['models_per_service'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{service}: {count} models")
    
    print("\nMost Used Data Access Patterns:")
    for item in report['summary']['most_used_data_access_patterns'][:5]:
        print(f"{item['pattern']}: {item['count']} occurrences")
    
    print("\nMost Connected Models:")
    for item in report['summary']['most_connected_models'][:5]:
        print(f"{item['model']}: {item['connections']} connections")

if __name__ == "__main__":
    main()