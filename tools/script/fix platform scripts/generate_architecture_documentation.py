#!/usr/bin/env python3
"""
Forex Trading Platform Architecture Documentation Generator

This script generates comprehensive documentation for the forex trading platform architecture.
It creates Markdown files documenting the platform structure, services, interfaces, and interactions.

Usage:
python generate_architecture_documentation.py [--project-root <project_root>] [--output-dir <output_dir>]
"""

import os
import sys
import re
import ast
import argparse
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\docs\architecture"

class ArchitectureDocumentationGenerator:
    """Generates documentation for the forex trading platform architecture."""
    
    def __init__(self, project_root: str, output_dir: str):
        """
        Initialize the generator.
        
        Args:
            project_root: Root directory of the project
            output_dir: Output directory for documentation
        """
        self.project_root = project_root
        self.output_dir = output_dir
        self.services = []
        self.interfaces = []
        self.dependencies = {}
        self.service_descriptions = {}
    
    def identify_services(self) -> None:
        """Identify services in the project based on directory structure."""
        logger.info("Identifying services...")
        
        # Look for service directories
        for item in os.listdir(self.project_root):
            item_path = os.path.join(self.project_root, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it's likely a service
                if (
                    item.endswith('-service') or 
                    item.endswith('_service') or 
                    item.endswith('-api') or 
                    item.endswith('-engine') or
                    'service' in item.lower() or
                    'api' in item.lower()
                ):
                    self.services.append(item)
        
        logger.info(f"Identified {len(self.services)} services")
    
    def load_dependencies(self) -> None:
        """Load service dependencies from the dependency report."""
        logger.info("Loading service dependencies...")
        
        dependency_report_path = os.path.join(self.project_root, 'tools', 'output', 'dependency-report.json')
        
        if not os.path.exists(dependency_report_path):
            logger.warning("Dependency report not found, running dependency analysis...")
            
            # Run dependency analysis
            import subprocess
            subprocess.run([
                'python',
                os.path.join(self.project_root, 'tools', 'analyze_dependencies.py'),
                '--project-root', self.project_root,
                '--output-dir', os.path.join(self.project_root, 'tools', 'output'),
                '--report-file', 'dependency-report.json'
            ])
        
        # Load dependency report
        try:
            with open(dependency_report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
                self.dependencies = report.get('service_dependencies', {})
            
            logger.info(f"Loaded dependencies for {len(self.dependencies)} services")
        except Exception as e:
            logger.error(f"Error loading dependency report: {e}")
            self.dependencies = {}
    
    def identify_interfaces(self) -> None:
        """Identify interfaces in the common-lib."""
        logger.info("Identifying interfaces...")
        
        common_lib_interfaces_path = os.path.join(self.project_root, 'common-lib', 'interfaces')
        
        if not os.path.exists(common_lib_interfaces_path):
            logger.warning("Common-lib interfaces directory not found")
            return
        
        # Look for interface files
        for item in os.listdir(common_lib_interfaces_path):
            if item.endswith('_interface.py'):
                interface_path = os.path.join(common_lib_interfaces_path, item)
                
                # Extract interface name and description
                try:
                    with open(interface_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse the file
                    tree = ast.parse(content)
                    
                    # Look for interface class
                    interface_name = None
                    interface_description = None
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and 'Interface' in node.name:
                            interface_name = node.name
                            
                            # Extract docstring
                            if ast.get_docstring(node):
                                interface_description = ast.get_docstring(node)
                            
                            break
                    
                    if interface_name:
                        self.interfaces.append({
                            'name': interface_name,
                            'file': item,
                            'description': interface_description or f"Interface for {item.replace('_interface.py', '').replace('_', ' ')}",
                            'methods': self.extract_interface_methods(tree, interface_name)
                        })
                except Exception as e:
                    logger.error(f"Error parsing interface {item}: {e}")
        
        logger.info(f"Identified {len(self.interfaces)} interfaces")
    
    def extract_interface_methods(self, tree: ast.AST, interface_name: str) -> List[Dict[str, Any]]:
        """
        Extract methods from an interface.
        
        Args:
            tree: AST tree
            interface_name: Name of the interface
            
        Returns:
            List of methods
        """
        methods = []
        
        # Find the interface class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == interface_name:
                # Extract methods
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name != '__init__':
                        # Extract method parameters
                        params = []
                        for arg in child.args.args:
                            if arg.arg != 'self':
                                param_type = 'Any'
                                if arg.annotation:
                                    if isinstance(arg.annotation, ast.Name):
                                        param_type = arg.annotation.id
                                    elif isinstance(arg.annotation, ast.Subscript):
                                        if isinstance(arg.annotation.value, ast.Name):
                                            param_type = arg.annotation.value.id
                                
                                params.append({
                                    'name': arg.arg,
                                    'type': param_type
                                })
                        
                        # Extract return type
                        return_type = 'None'
                        if child.returns:
                            if isinstance(child.returns, ast.Name):
                                return_type = child.returns.id
                            elif isinstance(child.returns, ast.Subscript):
                                if isinstance(child.returns.value, ast.Name):
                                    return_type = child.returns.value.id
                        
                        # Extract docstring
                        method_description = None
                        if ast.get_docstring(child):
                            method_description = ast.get_docstring(child)
                        
                        methods.append({
                            'name': child.name,
                            'params': params,
                            'return_type': return_type,
                            'description': method_description or f"Method {child.name}"
                        })
        
        return methods
    
    def extract_service_descriptions(self) -> None:
        """Extract descriptions for each service."""
        logger.info("Extracting service descriptions...")
        
        for service in self.services:
            service_path = os.path.join(self.project_root, service)
            
            # Look for README.md
            readme_path = os.path.join(service_path, 'README.md')
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract first paragraph as description
                    match = re.search(r'^#\s+(.+?)\n\n(.+?)(\n\n|$)', content, re.DOTALL)
                    if match:
                        self.service_descriptions[service] = {
                            'name': match.group(1),
                            'description': match.group(2)
                        }
                        continue
                except Exception as e:
                    logger.error(f"Error reading README for {service}: {e}")
            
            # Look for main.py or app.py
            for file_name in ['main.py', 'app.py', '__init__.py']:
                file_path = os.path.join(service_path, file_name)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract module docstring
                        tree = ast.parse(content)
                        docstring = ast.get_docstring(tree)
                        
                        if docstring:
                            # Extract first paragraph as description
                            description = docstring.split('\n\n')[0]
                            self.service_descriptions[service] = {
                                'name': service.replace('-', ' ').title(),
                                'description': description
                            }
                            break
                    except Exception as e:
                        logger.error(f"Error parsing {file_name} for {service}: {e}")
            
            # If no description found, use a default one
            if service not in self.service_descriptions:
                self.service_descriptions[service] = {
                    'name': service.replace('-', ' ').title(),
                    'description': f"Service for {service.replace('-service', '').replace('-', ' ')} functionality."
                }
    
    def generate_overview_documentation(self) -> None:
        """Generate overview documentation."""
        logger.info("Generating overview documentation...")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate overview.md
        overview_path = os.path.join(self.output_dir, 'overview.md')
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Forex Trading Platform Architecture Overview

*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Introduction

The Forex Trading Platform is a comprehensive system for forex trading, analysis, and portfolio management. It follows a microservices architecture with clear separation of concerns and standardized interfaces for service interactions.

## Architecture Principles

1. **Microservices Architecture**: The platform is composed of independent services, each with a specific responsibility.
2. **Interface-Based Design**: Services interact through well-defined interfaces, reducing direct coupling.
3. **Standardized Error Handling**: Consistent error handling across all services with correlation IDs for request tracing.
4. **Resilience Patterns**: Circuit breaker and retry with exponential backoff for all service interactions.
5. **Dependency Injection**: Adapter factories for dependency injection in all services.

## Services

The platform consists of {len(self.services)} services:

""")
            
            # List services
            for service in sorted(self.services):
                description = self.service_descriptions.get(service, {}).get('description', '')
                f.write(f"- **{service}**: {description}\n")
            
            # Add dependency diagram
            f.write("""
## Service Dependencies

The following diagram shows the dependencies between services:

```mermaid
graph TD
""")
            
            # Add dependencies
            for service, deps in self.dependencies.items():
                for dep in deps:
                    f.write(f"    {service} --> {dep}\n")
            
            f.write("```\n")
            
            # Add interfaces section
            f.write("""
## Interfaces

The platform uses the following interfaces for service interactions:

""")
            
            # List interfaces
            for interface in sorted(self.interfaces, key=lambda x: x['name']):
                f.write(f"- **{interface['name']}**: {interface['description']}\n")
            
            # Add directory structure section
            f.write("""
## Standard Directory Structure

Each service follows a standardized directory structure:

- **api**: API routes and controllers
- **config**: Configuration files
- **core**: Core business logic
- **models**: Data models and schemas
- **repositories**: Data access layer
- **services**: Service implementations
- **utils**: Utility functions
- **adapters**: Adapters for external services
- **interfaces**: Interface definitions
- **tests**: Unit and integration tests
""")
        
        logger.info(f"Generated overview documentation: {overview_path}")
    
    def generate_service_documentation(self) -> None:
        """Generate documentation for each service."""
        logger.info("Generating service documentation...")
        
        # Create services directory if it doesn't exist
        services_dir = os.path.join(self.output_dir, 'services')
        os.makedirs(services_dir, exist_ok=True)
        
        # Generate documentation for each service
        for service in self.services:
            service_doc_path = os.path.join(services_dir, f"{service}.md")
            
            with open(service_doc_path, 'w', encoding='utf-8') as f:
                # Service name and description
                service_name = self.service_descriptions.get(service, {}).get('name', service.replace('-', ' ').title())
                service_description = self.service_descriptions.get(service, {}).get('description', '')
                
                f.write(f"""# {service_name}

*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Description

{service_description}

## Dependencies

""")
                
                # List dependencies
                dependencies = self.dependencies.get(service, [])
                if dependencies:
                    f.write("This service depends on the following services:\n\n")
                    for dep in sorted(dependencies):
                        dep_description = self.service_descriptions.get(dep, {}).get('description', '')
                        f.write(f"- **{dep}**: {dep_description}\n")
                else:
                    f.write("This service has no dependencies on other services.\n")
                
                # List dependents
                dependents = []
                for s, deps in self.dependencies.items():
                    if service in deps:
                        dependents.append(s)
                
                f.write("\n## Dependents\n\n")
                if dependents:
                    f.write("The following services depend on this service:\n\n")
                    for dep in sorted(dependents):
                        dep_description = self.service_descriptions.get(dep, {}).get('description', '')
                        f.write(f"- **{dep}**: {dep_description}\n")
                else:
                    f.write("No other services depend on this service.\n")
                
                # List interfaces
                f.write("\n## Interfaces\n\n")
                
                # Find interfaces for this service
                service_interfaces = []
                for interface in self.interfaces:
                    if service.replace('-', '_') in interface['file']:
                        service_interfaces.append(interface)
                
                if service_interfaces:
                    f.write("This service provides the following interfaces:\n\n")
                    for interface in service_interfaces:
                        f.write(f"### {interface['name']}\n\n")
                        f.write(f"{interface['description']}\n\n")
                        
                        # List methods
                        f.write("#### Methods\n\n")
                        for method in interface['methods']:
                            params = ', '.join([f"{p['name']}: {p['type']}" for p in method['params']])
                            f.write(f"- **{method['name']}({params}) -> {method['return_type']}**: {method['description']}\n")
                else:
                    f.write("This service does not provide any interfaces.\n")
                
                # Add directory structure
                f.write("""
## Directory Structure

The service follows the standardized directory structure:

- **api**: API routes and controllers
- **config**: Configuration files
- **core**: Core business logic
- **models**: Data models and schemas
- **repositories**: Data access layer
- **services**: Service implementations
- **utils**: Utility functions
- **adapters**: Adapters for external services
- **interfaces**: Interface definitions
- **tests**: Unit and integration tests
""")
            
            logger.info(f"Generated documentation for {service}: {service_doc_path}")
    
    def generate_interface_documentation(self) -> None:
        """Generate documentation for interfaces."""
        logger.info("Generating interface documentation...")
        
        # Create interfaces directory if it doesn't exist
        interfaces_dir = os.path.join(self.output_dir, 'interfaces')
        os.makedirs(interfaces_dir, exist_ok=True)
        
        # Generate documentation for each interface
        for interface in self.interfaces:
            interface_doc_path = os.path.join(interfaces_dir, f"{interface['name']}.md")
            
            with open(interface_doc_path, 'w', encoding='utf-8') as f:
                f.write(f"""# {interface['name']}

*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Description

{interface['description']}

## File

`{interface['file']}`

## Methods

""")
                
                # List methods
                for method in interface['methods']:
                    params = ', '.join([f"{p['name']}: {p['type']}" for p in method['params']])
                    f.write(f"### {method['name']}({params}) -> {method['return_type']}\n\n")
                    f.write(f"{method['description']}\n\n")
                    
                    # List parameters
                    if method['params']:
                        f.write("#### Parameters\n\n")
                        for param in method['params']:
                            f.write(f"- **{param['name']}** ({param['type']})\n")
                        
                        f.write("\n")
                    
                    # Return type
                    f.write(f"#### Returns\n\n- {method['return_type']}\n\n")
            
            logger.info(f"Generated documentation for {interface['name']}: {interface_doc_path}")
    
    def generate_documentation(self) -> None:
        """Generate comprehensive documentation for the forex trading platform architecture."""
        logger.info("Starting documentation generation...")
        
        # Identify services
        self.identify_services()
        
        # Load dependencies
        self.load_dependencies()
        
        # Identify interfaces
        self.identify_interfaces()
        
        # Extract service descriptions
        self.extract_service_descriptions()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate overview documentation
        self.generate_overview_documentation()
        
        # Generate service documentation
        self.generate_service_documentation()
        
        # Generate interface documentation
        self.generate_interface_documentation()
        
        # Copy platform fixing log
        platform_fixing_log_path = os.path.join(self.project_root, 'platform_fixing_log.md')
        if os.path.exists(platform_fixing_log_path):
            shutil.copy2(platform_fixing_log_path, os.path.join(self.output_dir, 'platform_fixing_log.md'))
        
        logger.info("Documentation generation complete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate architecture documentation")
    parser.add_argument(
        "--project-root", 
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for documentation"
    )
    args = parser.parse_args()
    
    # Generate documentation
    generator = ArchitectureDocumentationGenerator(args.project_root, args.output_dir)
    generator.generate_documentation()
    
    logger.info(f"Documentation generated in {args.output_dir}")

if __name__ == "__main__":
    main()
