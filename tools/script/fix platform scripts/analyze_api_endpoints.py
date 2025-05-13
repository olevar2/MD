#!/usr/bin/env python3
"""
Analyze API Endpoints and Communication Patterns in the Forex Trading Platform

This script analyzes the API endpoints and communication patterns in the forex trading platform,
identifying REST endpoints, gRPC services, message queues, and other communication mechanisms.

Usage:
    python analyze_api_endpoints.py [--output-file OUTPUT_FILE]

Options:
    --output-file OUTPUT_FILE    Output file for the API endpoints report (default: api-endpoints-report.json)
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

def find_files(directory: str, extensions: List[str]) -> List[str]:
    """Find all files with specified extensions in a directory recursively."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files

def extract_rest_endpoints(file_path: str) -> List[Dict[str, Any]]:
    """Extract REST API endpoints from a Python file."""
    endpoints = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for Flask/FastAPI route decorators
            flask_pattern = r'@(?:app|blueprint|router|api)\.(?:route|get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]'
            flask_matches = re.finditer(flask_pattern, content, re.MULTILINE)
            
            for match in flask_matches:
                route = match.group(1)
                
                # Determine HTTP method
                method_match = re.search(r'@(?:app|blueprint|router|api)\.(get|post|put|delete|patch)', match.group(0), re.IGNORECASE)
                method = method_match.group(1).upper() if method_match else 'GET'
                
                # Find the function name
                func_match = re.search(r'def\s+(\w+)\s*\(', content[match.end():], re.MULTILINE)
                func_name = func_match.group(1) if func_match else 'unknown'
                
                endpoints.append({
                    'route': route,
                    'method': method,
                    'function': func_name,
                    'type': 'REST',
                    'framework': 'Flask/FastAPI',
                    'file_path': file_path
                })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return endpoints

def extract_grpc_services(file_path: str) -> List[Dict[str, Any]]:
    """Extract gRPC services from a Python file."""
    services = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for gRPC service definitions
            service_pattern = r'class\s+(\w+)(?:Servicer|Handler|Server|Service)\s*\([^)]*\):'
            service_matches = re.finditer(service_pattern, content, re.MULTILINE)
            
            for match in service_matches:
                service_name = match.group(1)
                
                # Find methods in the service
                methods = []
                service_start = match.end()
                
                # Find the end of the class definition
                next_class_match = re.search(r'class\s+\w+', content[service_start:])
                if next_class_match:
                    service_end = service_start + next_class_match.start()
                else:
                    service_end = len(content)
                
                service_content = content[service_start:service_end]
                
                # Extract method definitions
                method_pattern = r'def\s+(\w+)\s*\([^)]*\):'
                method_matches = re.finditer(method_pattern, service_content, re.MULTILINE)
                
                for method_match in method_matches:
                    method_name = method_match.group(1)
                    if not method_name.startswith('__'):  # Skip special methods
                        methods.append(method_name)
                
                services.append({
                    'service': service_name,
                    'methods': methods,
                    'type': 'gRPC',
                    'file_path': file_path
                })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return services

def extract_message_queues(file_path: str) -> List[Dict[str, Any]]:
    """Extract message queue patterns from a Python file."""
    queues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for Kafka patterns
            kafka_producer_pattern = r'KafkaProducer\s*\('
            kafka_consumer_pattern = r'KafkaConsumer\s*\('
            kafka_topic_pattern = r'(?:topic|topics)\s*=\s*[\'"]([^\'"]+)[\'"]'
            
            # Look for RabbitMQ patterns
            rabbitmq_pattern = r'(?:pika\.ConnectionParameters|BlockingConnection|channel\.queue_declare)'
            rabbitmq_queue_pattern = r'queue_declare\s*\(\s*[\'"]([^\'"]+)[\'"]'
            
            # Look for Redis patterns
            redis_pattern = r'(?:redis\.Redis|StrictRedis)'
            redis_pubsub_pattern = r'(?:publish|subscribe|pubsub)'
            
            # Check for Kafka
            if re.search(kafka_producer_pattern, content) or re.search(kafka_consumer_pattern, content):
                # Extract topics
                topics = re.findall(kafka_topic_pattern, content)
                
                queues.append({
                    'type': 'Kafka',
                    'topics': topics,
                    'file_path': file_path
                })
            
            # Check for RabbitMQ
            if re.search(rabbitmq_pattern, content):
                # Extract queues
                rabbit_queues = re.findall(rabbitmq_queue_pattern, content)
                
                queues.append({
                    'type': 'RabbitMQ',
                    'queues': rabbit_queues,
                    'file_path': file_path
                })
            
            # Check for Redis
            if re.search(redis_pattern, content) and re.search(redis_pubsub_pattern, content):
                queues.append({
                    'type': 'Redis PubSub',
                    'file_path': file_path
                })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return queues

def extract_websocket_endpoints(file_path: str) -> List[Dict[str, Any]]:
    """Extract WebSocket endpoints from a Python file."""
    endpoints = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for WebSocket patterns
            websocket_pattern = r'@(?:sockets|websocket|socket)\.(?:route|on)\s*\(\s*[\'"]([^\'"]+)[\'"]'
            websocket_matches = re.finditer(websocket_pattern, content, re.MULTILINE)
            
            for match in websocket_matches:
                route = match.group(1)
                
                # Find the function name
                func_match = re.search(r'def\s+(\w+)\s*\(', content[match.end():], re.MULTILINE)
                func_name = func_match.group(1) if func_match else 'unknown'
                
                endpoints.append({
                    'route': route,
                    'function': func_name,
                    'type': 'WebSocket',
                    'file_path': file_path
                })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return endpoints

def analyze_api_endpoints() -> Dict[str, Any]:
    """Analyze API endpoints and communication patterns."""
    api_endpoints = {
        'rest_endpoints': [],
        'grpc_services': [],
        'message_queues': [],
        'websocket_endpoints': [],
        'service_endpoints': defaultdict(list),
        'service_communication': []
    }
    
    for service in SERVICE_DIRS:
        service_dir = os.path.join(ROOT_DIR, service)
        if not os.path.isdir(service_dir):
            print(f"Warning: Service directory {service_dir} not found")
            continue
        
        # Find Python files
        python_files = find_files(service_dir, ['.py'])
        
        for file in python_files:
            # Extract REST endpoints
            rest_endpoints = extract_rest_endpoints(file)
            api_endpoints['rest_endpoints'].extend(rest_endpoints)
            
            # Add endpoints to service_endpoints
            for endpoint in rest_endpoints:
                api_endpoints['service_endpoints'][service].append({
                    'type': 'REST',
                    'route': endpoint['route'],
                    'method': endpoint['method']
                })
            
            # Extract gRPC services
            grpc_services = extract_grpc_services(file)
            api_endpoints['grpc_services'].extend(grpc_services)
            
            # Add services to service_endpoints
            for service_def in grpc_services:
                for method in service_def['methods']:
                    api_endpoints['service_endpoints'][service].append({
                        'type': 'gRPC',
                        'service': service_def['service'],
                        'method': method
                    })
            
            # Extract message queues
            message_queues = extract_message_queues(file)
            api_endpoints['message_queues'].extend(message_queues)
            
            # Add queues to service_endpoints
            for queue in message_queues:
                if queue['type'] == 'Kafka' and 'topics' in queue:
                    for topic in queue['topics']:
                        api_endpoints['service_endpoints'][service].append({
                            'type': 'Kafka',
                            'topic': topic
                        })
                elif queue['type'] == 'RabbitMQ' and 'queues' in queue:
                    for queue_name in queue['queues']:
                        api_endpoints['service_endpoints'][service].append({
                            'type': 'RabbitMQ',
                            'queue': queue_name
                        })
                elif queue['type'] == 'Redis PubSub':
                    api_endpoints['service_endpoints'][service].append({
                        'type': 'Redis PubSub'
                    })
            
            # Extract WebSocket endpoints
            websocket_endpoints = extract_websocket_endpoints(file)
            api_endpoints['websocket_endpoints'].extend(websocket_endpoints)
            
            # Add endpoints to service_endpoints
            for endpoint in websocket_endpoints:
                api_endpoints['service_endpoints'][service].append({
                    'type': 'WebSocket',
                    'route': endpoint['route']
                })
    
    # Identify service communication patterns
    for service, endpoints in api_endpoints['service_endpoints'].items():
        # Look for client code that communicates with other services
        service_dir = os.path.join(ROOT_DIR, service)
        python_files = find_files(service_dir, ['.py'])
        
        for file in python_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Look for HTTP client patterns
                    http_pattern = r'(?:requests|httpx|aiohttp)\.(?:get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]'
                    http_matches = re.finditer(http_pattern, content, re.MULTILINE)
                    
                    for match in http_matches:
                        url = match.group(1)
                        
                        # Try to identify the target service
                        target_service = None
                        for s, eps in api_endpoints['service_endpoints'].items():
                            if s != service:  # Skip self-communication
                                for ep in eps:
                                    if ep['type'] == 'REST' and 'route' in ep:
                                        if ep['route'] in url or url.endswith(ep['route']):
                                            target_service = s
                                            break
                        
                        if target_service:
                            api_endpoints['service_communication'].append({
                                'source': service,
                                'target': target_service,
                                'type': 'HTTP',
                                'url': url
                            })
                    
                    # Look for gRPC client patterns
                    grpc_pattern = r'(?:grpc\.(?:insecure_channel|secure_channel)|stub\.\w+)'
                    if re.search(grpc_pattern, content):
                        # Try to identify the target service
                        for s, eps in api_endpoints['service_endpoints'].items():
                            if s != service:  # Skip self-communication
                                for ep in eps:
                                    if ep['type'] == 'gRPC' and 'service' in ep:
                                        if ep['service'] in content:
                                            api_endpoints['service_communication'].append({
                                                'source': service,
                                                'target': s,
                                                'type': 'gRPC',
                                                'service': ep['service']
                                            })
                    
                    # Look for Kafka producer patterns
                    kafka_pattern = r'KafkaProducer\s*\('
                    kafka_topic_pattern = r'(?:topic|topics)\s*=\s*[\'"]([^\'"]+)[\'"]'
                    
                    if re.search(kafka_pattern, content):
                        topics = re.findall(kafka_topic_pattern, content)
                        
                        for topic in topics:
                            # Try to identify the target service
                            for s, eps in api_endpoints['service_endpoints'].items():
                                if s != service:  # Skip self-communication
                                    for ep in eps:
                                        if ep['type'] == 'Kafka' and 'topic' in ep and ep['topic'] == topic:
                                            api_endpoints['service_communication'].append({
                                                'source': service,
                                                'target': s,
                                                'type': 'Kafka',
                                                'topic': topic
                                            })
                    
                    # Look for RabbitMQ producer patterns
                    rabbitmq_pattern = r'channel\.basic_publish\s*\(\s*[\'"]([^\'"]+)[\'"]'
                    rabbitmq_matches = re.finditer(rabbitmq_pattern, content, re.MULTILINE)
                    
                    for match in rabbitmq_matches:
                        exchange = match.group(1)
                        
                        # Try to identify the target service
                        for s, eps in api_endpoints['service_endpoints'].items():
                            if s != service:  # Skip self-communication
                                for ep in eps:
                                    if ep['type'] == 'RabbitMQ' and 'queue' in ep and ep['queue'] == exchange:
                                        api_endpoints['service_communication'].append({
                                            'source': service,
                                            'target': s,
                                            'type': 'RabbitMQ',
                                            'exchange': exchange
                                        })
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Convert defaultdict to dict for JSON serialization
    api_endpoints['service_endpoints'] = dict(api_endpoints['service_endpoints'])
    
    return api_endpoints

def generate_api_endpoints_report(api_endpoints: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive API endpoints report."""
    # Calculate statistics
    total_rest_endpoints = len(api_endpoints['rest_endpoints'])
    total_grpc_services = len(api_endpoints['grpc_services'])
    total_message_queues = len(api_endpoints['message_queues'])
    total_websocket_endpoints = len(api_endpoints['websocket_endpoints'])
    total_service_communications = len(api_endpoints['service_communication'])
    
    # Count endpoints per service
    endpoints_per_service = {
        service: len(endpoints)
        for service, endpoints in api_endpoints['service_endpoints'].items()
    }
    
    # Count communication types
    communication_types = defaultdict(int)
    for comm in api_endpoints['service_communication']:
        communication_types[comm['type']] += 1
    
    # Identify most communicative services
    service_communications = defaultdict(int)
    for comm in api_endpoints['service_communication']:
        service_communications[comm['source']] += 1
        service_communications[comm['target']] += 1
    
    most_communicative_services = sorted(
        service_communications.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]  # Top 10
    
    # Generate report
    report = {
        'api_endpoints': api_endpoints,
        'summary': {
            'total_rest_endpoints': total_rest_endpoints,
            'total_grpc_services': total_grpc_services,
            'total_message_queues': total_message_queues,
            'total_websocket_endpoints': total_websocket_endpoints,
            'total_service_communications': total_service_communications,
            'endpoints_per_service': endpoints_per_service,
            'communication_types': dict(communication_types),
            'most_communicative_services': [
                {'service': service, 'communications': communications}
                for service, communications in most_communicative_services
            ]
        }
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Analyze API endpoints in the forex trading platform')
    parser.add_argument('--output-file', default='api-endpoints-report.json', help='Output file for the API endpoints report')
    args = parser.parse_args()
    
    print("Analyzing API endpoints...")
    api_endpoints = analyze_api_endpoints()
    
    print("Generating API endpoints report...")
    report = generate_api_endpoints_report(api_endpoints)
    
    # Save report to file
    output_path = os.path.join(ROOT_DIR, 'tools', 'output', args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"API endpoints report saved to {output_path}")
    
    # Print summary
    print("\nAPI Endpoints Analysis Summary:")
    print(f"Total REST endpoints: {report['summary']['total_rest_endpoints']}")
    print(f"Total gRPC services: {report['summary']['total_grpc_services']}")
    print(f"Total message queues: {report['summary']['total_message_queues']}")
    print(f"Total WebSocket endpoints: {report['summary']['total_websocket_endpoints']}")
    print(f"Total service communications: {report['summary']['total_service_communications']}")
    
    print("\nEndpoints per Service:")
    for service, count in sorted(report['summary']['endpoints_per_service'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{service}: {count} endpoints")
    
    print("\nCommunication Types:")
    for comm_type, count in sorted(report['summary']['communication_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"{comm_type}: {count} communications")
    
    print("\nMost Communicative Services:")
    for item in report['summary']['most_communicative_services'][:5]:
        print(f"{item['service']}: {item['communications']} communications")

if __name__ == "__main__":
    main()