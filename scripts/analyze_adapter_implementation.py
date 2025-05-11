#!/usr/bin/env python3
"""
Adapter Implementation Analyzer

This script analyzes the current implementation of the interface-based adapter pattern
in the Forex Trading Platform. It identifies interfaces, adapters, and direct dependencies
between services, and generates a report of the findings.
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("adapter_analyzer")

# Constants
IGNORED_DIRS = {'.git', '.github', '.venv', '.pytest_cache', '__pycache__', 'node_modules'}
SERVICE_DIRS = {
    'analysis-engine-service', 
    'analysis_engine',
    'api-gateway', 
    'common-lib', 
    'common-js-lib',
    'core-foundations',
    'data-management-service',
    'data-pipeline-service',
    'feature-store-service',
    'feature_store_service',
    'ml-integration-service',
    'ml-workbench-service',
    'model-registry-service',
    'monitoring-alerting-service',
    'portfolio-management-service',
    'risk-management-service',
    'strategy-execution-engine',
    'trading-gateway-service',
    'ui-service'
}

class AdapterAnalyzer:
    """Analyzes the implementation of the interface-based adapter pattern."""

    def __init__(self, root_dir: str):
        """Initialize the analyzer with the root directory of the project."""
        self.root_dir = Path(root_dir)
        self.interfaces: Dict[str, List[str]] = {}  # service -> interfaces
        self.adapters: Dict[str, List[str]] = {}    # service -> adapters
        self.direct_deps: Dict[str, Set[str]] = {}  # service -> direct dependencies
        self.service_clients: Dict[str, List[str]] = {}  # service -> service clients
        self.adapter_factories: Dict[str, List[str]] = {}  # service -> adapter factories
        self.interface_implementations: Dict[str, List[str]] = {}  # interface -> implementations

    def analyze(self) -> Dict[str, Any]:
        """Analyze the project and return the results."""
        logger.info("Starting adapter implementation analysis...")
        
        # Find all interfaces
        self._find_interfaces()
        
        # Find all adapters
        self._find_adapters()
        
        # Find all direct dependencies
        self._find_direct_dependencies()
        
        # Find all service clients
        self._find_service_clients()
        
        # Find all adapter factories
        self._find_adapter_factories()
        
        # Find all interface implementations
        self._find_interface_implementations()
        
        # Generate the report
        report = self._generate_report()
        
        logger.info("Adapter implementation analysis complete.")
        return report

    def _find_interfaces(self) -> None:
        """Find all interfaces in the project."""
        logger.info("Finding interfaces...")
        
        # Look for interface definitions in common-lib
        common_lib_interfaces_dir = self.root_dir / 'common-lib' / 'common_lib' / 'interfaces'
        if common_lib_interfaces_dir.exists():
            for file_path in common_lib_interfaces_dir.glob('**/*.py'):
                if file_path.name == '__init__.py':
                    continue
                
                service = 'common-lib'
                if service not in self.interfaces:
                    self.interfaces[service] = []
                
                # Parse the file to find interface definitions
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all class definitions that start with 'I' and have 'ABC' as a base class
                interface_pattern = r'class\s+(I[A-Za-z0-9_]+)\s*\([^)]*ABC[^)]*\):'
                interfaces = re.findall(interface_pattern, content)
                
                for interface in interfaces:
                    self.interfaces[service].append(interface)
        
        logger.info(f"Found {sum(len(interfaces) for interfaces in self.interfaces.values())} interfaces.")

    def _find_adapters(self) -> None:
        """Find all adapters in the project."""
        logger.info("Finding adapters...")
        
        # Look for adapter implementations in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            # Look for adapter directories
            adapter_dirs = []
            for root, dirs, _ in os.walk(service_path):
                for dir_name in dirs:
                    if dir_name == 'adapters':
                        adapter_dirs.append(Path(root) / dir_name)
            
            # Look for adapter files in adapter directories
            for adapter_dir in adapter_dirs:
                for file_path in adapter_dir.glob('**/*.py'):
                    if file_path.name == '__init__.py':
                        continue
                    
                    if service_dir not in self.adapters:
                        self.adapters[service_dir] = []
                    
                    # Parse the file to find adapter implementations
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find all class definitions that end with 'Adapter'
                    adapter_pattern = r'class\s+([A-Za-z0-9_]+Adapter)\s*\('
                    adapters = re.findall(adapter_pattern, content)
                    
                    for adapter in adapters:
                        self.adapters[service_dir].append(adapter)
        
        logger.info(f"Found {sum(len(adapters) for adapters in self.adapters.values())} adapters.")

    def _find_direct_dependencies(self) -> None:
        """Find all direct dependencies between services."""
        logger.info("Finding direct dependencies...")
        
        # Look for import statements in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            if service_dir not in self.direct_deps:
                self.direct_deps[service_dir] = set()
            
            # Look for Python files
            for file_path in service_path.glob('**/*.py'):
                # Parse the file to find import statements
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all import statements
                import_pattern = r'(?:from|import)\s+([A-Za-z0-9_.]+)'
                imports = re.findall(import_pattern, content)
                
                # Check if any import is from another service
                for import_stmt in imports:
                    for other_service in SERVICE_DIRS:
                        if other_service == service_dir:
                            continue
                        
                        # Convert kebab-case to snake_case for import comparison
                        other_service_import = other_service.replace('-', '_')
                        
                        # Check if the import is from the other service
                        if import_stmt == other_service_import or import_stmt.startswith(f"{other_service_import}."):
                            self.direct_deps[service_dir].add(other_service)
        
        logger.info(f"Found {sum(len(deps) for deps in self.direct_deps.values())} direct dependencies.")

    def _find_service_clients(self) -> None:
        """Find all service clients in the project."""
        logger.info("Finding service clients...")
        
        # Look for service client implementations in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            # Look for client directories
            client_dirs = []
            for root, dirs, _ in os.walk(service_path):
                for dir_name in dirs:
                    if dir_name in ['clients', 'client']:
                        client_dirs.append(Path(root) / dir_name)
            
            # Look for client files in client directories
            for client_dir in client_dirs:
                for file_path in client_dir.glob('**/*.py'):
                    if file_path.name == '__init__.py':
                        continue
                    
                    if service_dir not in self.service_clients:
                        self.service_clients[service_dir] = []
                    
                    # Parse the file to find client implementations
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find all class definitions that end with 'Client'
                    client_pattern = r'class\s+([A-Za-z0-9_]+Client)\s*\('
                    clients = re.findall(client_pattern, content)
                    
                    for client in clients:
                        self.service_clients[service_dir].append(client)
        
        logger.info(f"Found {sum(len(clients) for clients in self.service_clients.values())} service clients.")

    def _find_adapter_factories(self) -> None:
        """Find all adapter factories in the project."""
        logger.info("Finding adapter factories...")
        
        # Look for adapter factory implementations in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            # Look for factory files
            factory_files = []
            for root, _, files in os.walk(service_path):
                for file_name in files:
                    if 'factory' in file_name.lower() and file_name.endswith('.py'):
                        factory_files.append(Path(root) / file_name)
            
            # Look for factory implementations in factory files
            for file_path in factory_files:
                if service_dir not in self.adapter_factories:
                    self.adapter_factories[service_dir] = []
                
                # Parse the file to find factory implementations
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all class definitions that contain 'Factory'
                factory_pattern = r'class\s+([A-Za-z0-9_]*Factory[A-Za-z0-9_]*)\s*\('
                factories = re.findall(factory_pattern, content)
                
                for factory in factories:
                    self.adapter_factories[service_dir].append(factory)
        
        logger.info(f"Found {sum(len(factories) for factories in self.adapter_factories.values())} adapter factories.")

    def _find_interface_implementations(self) -> None:
        """Find all implementations of interfaces."""
        logger.info("Finding interface implementations...")
        
        # Get all interfaces
        all_interfaces = []
        for interfaces in self.interfaces.values():
            all_interfaces.extend(interfaces)
        
        # Look for implementations of interfaces in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            # Look for Python files
            for file_path in service_path.glob('**/*.py'):
                # Parse the file to find class definitions
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for implementations of each interface
                for interface in all_interfaces:
                    # Find all class definitions that implement the interface
                    impl_pattern = rf'class\s+([A-Za-z0-9_]+)\s*\([^)]*{interface}[^)]*\):'
                    implementations = re.findall(impl_pattern, content)
                    
                    for impl in implementations:
                        if interface not in self.interface_implementations:
                            self.interface_implementations[interface] = []
                        
                        self.interface_implementations[interface].append(f"{service_dir}.{impl}")
        
        logger.info(f"Found {sum(len(impls) for impls in self.interface_implementations.values())} interface implementations.")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate a report of the findings."""
        logger.info("Generating report...")
        
        report = {
            "interfaces": self.interfaces,
            "adapters": self.adapters,
            "direct_dependencies": {service: list(deps) for service, deps in self.direct_deps.items()},
            "service_clients": self.service_clients,
            "adapter_factories": self.adapter_factories,
            "interface_implementations": self.interface_implementations,
            "summary": {
                "total_interfaces": sum(len(interfaces) for interfaces in self.interfaces.values()),
                "total_adapters": sum(len(adapters) for adapters in self.adapters.values()),
                "total_direct_dependencies": sum(len(deps) for deps in self.direct_deps.values()),
                "total_service_clients": sum(len(clients) for clients in self.service_clients.values()),
                "total_adapter_factories": sum(len(factories) for factories in self.adapter_factories.values()),
                "total_interface_implementations": sum(len(impls) for impls in self.interface_implementations.values()),
            }
        }
        
        # Add analysis of interface coverage
        report["analysis"] = {
            "interface_coverage": {},
            "adapter_coverage": {},
            "direct_dependency_issues": [],
            "missing_adapters": [],
            "missing_interface_implementations": [],
        }
        
        # Analyze interface coverage
        for service, interfaces in self.interfaces.items():
            report["analysis"]["interface_coverage"][service] = {
                "total_interfaces": len(interfaces),
                "implemented_interfaces": 0,
                "implementation_percentage": 0.0,
            }
            
            for interface in interfaces:
                if interface in self.interface_implementations:
                    report["analysis"]["interface_coverage"][service]["implemented_interfaces"] += 1
            
            if report["analysis"]["interface_coverage"][service]["total_interfaces"] > 0:
                report["analysis"]["interface_coverage"][service]["implementation_percentage"] = (
                    report["analysis"]["interface_coverage"][service]["implemented_interfaces"] /
                    report["analysis"]["interface_coverage"][service]["total_interfaces"] * 100.0
                )
        
        # Analyze adapter coverage
        for service, direct_deps in self.direct_deps.items():
            report["analysis"]["adapter_coverage"][service] = {
                "total_dependencies": len(direct_deps),
                "adapter_covered_dependencies": 0,
                "adapter_coverage_percentage": 0.0,
            }
            
            for dep in direct_deps:
                # Check if there's an adapter for this dependency
                has_adapter = False
                if service in self.adapters:
                    for adapter in self.adapters[service]:
                        if dep.replace('-', '_') in adapter.lower() or dep.replace('_', '-') in adapter.lower():
                            has_adapter = True
                            break
                
                if has_adapter:
                    report["analysis"]["adapter_coverage"][service]["adapter_covered_dependencies"] += 1
            
            if report["analysis"]["adapter_coverage"][service]["total_dependencies"] > 0:
                report["analysis"]["adapter_coverage"][service]["adapter_coverage_percentage"] = (
                    report["analysis"]["adapter_coverage"][service]["adapter_covered_dependencies"] /
                    report["analysis"]["adapter_coverage"][service]["total_dependencies"] * 100.0
                )
        
        # Identify direct dependency issues
        for service, direct_deps in self.direct_deps.items():
            for dep in direct_deps:
                # Skip common-lib and core-foundations
                if dep in ['common-lib', 'core-foundations']:
                    continue
                
                # Check if there's an adapter for this dependency
                has_adapter = False
                if service in self.adapters:
                    for adapter in self.adapters[service]:
                        if dep.replace('-', '_') in adapter.lower() or dep.replace('_', '-') in adapter.lower():
                            has_adapter = True
                            break
                
                if not has_adapter:
                    report["analysis"]["direct_dependency_issues"].append({
                        "service": service,
                        "dependency": dep,
                        "issue": "Direct dependency without adapter"
                    })
        
        # Identify missing adapters
        for interface, implementations in self.interface_implementations.items():
            # Check if there's an adapter for each implementation
            for impl in implementations:
                service = impl.split('.')[0]
                
                # Check if there's an adapter for this implementation
                has_adapter = False
                if service in self.adapters:
                    for adapter in self.adapters[service]:
                        if interface in adapter:
                            has_adapter = True
                            break
                
                if not has_adapter:
                    report["analysis"]["missing_adapters"].append({
                        "service": service,
                        "interface": interface,
                        "implementation": impl,
                        "issue": "Missing adapter for interface implementation"
                    })
        
        # Identify missing interface implementations
        for service, interfaces in self.interfaces.items():
            for interface in interfaces:
                if interface not in self.interface_implementations:
                    report["analysis"]["missing_interface_implementations"].append({
                        "service": service,
                        "interface": interface,
                        "issue": "Interface has no implementations"
                    })
        
        logger.info("Report generation complete.")
        return report

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze adapter implementation in the Forex Trading Platform.")
    parser.add_argument("--root", default=".", help="Root directory of the project")
    parser.add_argument("--output", default="adapter_analysis.json", help="Output file for the analysis report")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format")
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = AdapterAnalyzer(args.root)
    
    # Run the analysis
    report = analyzer.analyze()
    
    # Save the report
    if args.format == "json":
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.output}")
    elif args.format == "markdown":
        markdown_output = args.output.replace(".json", ".md")
        with open(markdown_output, 'w', encoding='utf-8') as f:
            f.write("# Adapter Implementation Analysis Report\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total Interfaces: {report['summary']['total_interfaces']}\n")
            f.write(f"- Total Adapters: {report['summary']['total_adapters']}\n")
            f.write(f"- Total Direct Dependencies: {report['summary']['total_direct_dependencies']}\n")
            f.write(f"- Total Service Clients: {report['summary']['total_service_clients']}\n")
            f.write(f"- Total Adapter Factories: {report['summary']['total_adapter_factories']}\n")
            f.write(f"- Total Interface Implementations: {report['summary']['total_interface_implementations']}\n\n")
            
            f.write("## Interface Coverage\n\n")
            f.write("| Service | Total Interfaces | Implemented Interfaces | Implementation Percentage |\n")
            f.write("|---------|-----------------|------------------------|---------------------------|\n")
            for service, coverage in report["analysis"]["interface_coverage"].items():
                f.write(f"| {service} | {coverage['total_interfaces']} | {coverage['implemented_interfaces']} | {coverage['implementation_percentage']:.2f}% |\n")
            f.write("\n")
            
            f.write("## Adapter Coverage\n\n")
            f.write("| Service | Total Dependencies | Adapter Covered Dependencies | Adapter Coverage Percentage |\n")
            f.write("|---------|-------------------|------------------------------|-----------------------------|\n")
            for service, coverage in report["analysis"]["adapter_coverage"].items():
                f.write(f"| {service} | {coverage['total_dependencies']} | {coverage['adapter_covered_dependencies']} | {coverage['adapter_coverage_percentage']:.2f}% |\n")
            f.write("\n")
            
            f.write("## Direct Dependency Issues\n\n")
            if report["analysis"]["direct_dependency_issues"]:
                f.write("| Service | Dependency | Issue |\n")
                f.write("|---------|------------|-------|\n")
                for issue in report["analysis"]["direct_dependency_issues"]:
                    f.write(f"| {issue['service']} | {issue['dependency']} | {issue['issue']} |\n")
            else:
                f.write("No direct dependency issues found.\n")
            f.write("\n")
            
            f.write("## Missing Adapters\n\n")
            if report["analysis"]["missing_adapters"]:
                f.write("| Service | Interface | Implementation | Issue |\n")
                f.write("|---------|-----------|----------------|-------|\n")
                for issue in report["analysis"]["missing_adapters"]:
                    f.write(f"| {issue['service']} | {issue['interface']} | {issue['implementation']} | {issue['issue']} |\n")
            else:
                f.write("No missing adapters found.\n")
            f.write("\n")
            
            f.write("## Missing Interface Implementations\n\n")
            if report["analysis"]["missing_interface_implementations"]:
                f.write("| Service | Interface | Issue |\n")
                f.write("|---------|-----------|-------|\n")
                for issue in report["analysis"]["missing_interface_implementations"]:
                    f.write(f"| {issue['service']} | {issue['interface']} | {issue['issue']} |\n")
            else:
                f.write("No missing interface implementations found.\n")
        
        logger.info(f"Report saved to {markdown_output}")

if __name__ == "__main__":
    main()
