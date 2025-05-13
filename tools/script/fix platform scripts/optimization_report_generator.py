#!/usr/bin/env python3
"""
Forex Trading Platform Optimization Report Generator

This script generates a comprehensive optimization report for the forex trading platform
by analyzing the results from various analysis tools. It provides actionable recommendations
for improving the project structure and resolving issues.

Usage:
python optimization_report_generator.py [--project-root <project_root>] [--output-dir <output_dir>]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"

# Markdown template for the report
MARKDOWN_TEMPLATE = """
# Forex Trading Platform Optimization Report

**Generated on:** {generation_date}

## Executive Summary

This report provides a comprehensive analysis of the Forex Trading Platform architecture and identifies opportunities for optimization. The analysis was performed using a combination of specialized tools to examine dependencies, code duplication, and structural issues.

### Key Findings

- **Services:** {service_count} services identified
- **Files:** {file_count} Python files analyzed
- **Dependencies:** {dependency_count} service dependencies detected
- **Circular Dependencies:** {circular_count} circular dependencies found
- **Duplicate Code:** {duplicate_groups_count} groups of duplicate code identified

## Dependency Analysis

### Service Dependencies

The platform consists of {service_count} services with {dependency_count} dependencies between them. The dependency graph reveals the following insights:

{dependency_insights}

### Circular Dependencies

{circular_dependencies_count} circular dependencies were identified between services. These circular dependencies can lead to tight coupling, making the codebase harder to maintain and evolve.

{circular_dependencies_section}

## Code Duplication Analysis

The analysis identified {duplicate_groups_count} groups of duplicate code across the codebase. Code duplication can lead to maintenance challenges, as changes need to be applied in multiple places.

{duplicate_code_section}

## Service Analysis

{service_analysis_section}

## Optimization Recommendations

Based on the analysis, the following recommendations are provided to optimize the platform architecture:

### 1. Resolve Circular Dependencies

{circular_dependencies_recommendations}

### 2. Consolidate Duplicate Code

{duplicate_code_recommendations}

### 3. Improve Service Structure

{service_structure_recommendations}

### 4. Enhance Modularity

{modularity_recommendations}

### 5. Standardize Interfaces

{interface_recommendations}

## Implementation Plan

To implement the recommended optimizations, the following phased approach is suggested:

### Phase 1: Immediate Improvements

1. Resolve critical circular dependencies
2. Consolidate high-similarity duplicate code
3. Standardize service interfaces

### Phase 2: Structural Enhancements

1. Refactor service boundaries
2. Implement common libraries for shared functionality
3. Enhance modularity through clear separation of concerns

### Phase 3: Long-term Architecture Evolution

1. Migrate to a more event-driven architecture
2. Implement comprehensive testing for refactored components
3. Document the improved architecture

## Conclusion

The Forex Trading Platform has a complex architecture with several opportunities for optimization. By addressing the identified issues, the platform can become more maintainable, scalable, and resilient.

"""

def load_dependency_report(report_path: str) -> Dict[str, Any]:
    """
    Load the dependency report.
    
    Args:
        report_path: Path to the dependency report
        
    Returns:
        Dependency report data
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading dependency report: {e}")
        return {}

def load_duplicate_code_report(report_path: str) -> Dict[str, Any]:
    """
    Load the duplicate code report.
    
    Args:
        report_path: Path to the duplicate code report
        
    Returns:
        Duplicate code report data
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading duplicate code report: {e}")
        return {}

def load_circular_dependencies_report(report_path: str) -> Dict[str, Any]:
    """
    Load the circular dependencies report.
    
    Args:
        report_path: Path to the circular dependencies report
        
    Returns:
        Circular dependencies report data
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading circular dependencies report: {e}")
        return {}

def generate_dependency_insights(dependency_report: Dict[str, Any]) -> str:
    """
    Generate insights from the dependency report.
    
    Args:
        dependency_report: Dependency report data
        
    Returns:
        Markdown text with dependency insights
    """
    services = dependency_report.get('services', {})
    service_dependencies = dependency_report.get('service_dependencies', {})
    
    # Find services with most dependencies
    service_dep_count = {s: len(deps) for s, deps in service_dependencies.items()}
    most_dependent = sorted(service_dep_count.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Find services with most dependents
    service_dependents = {}
    for service, deps in service_dependencies.items():
        for dep in deps:
            if dep not in service_dependents:
                service_dependents[dep] = []
            service_dependents[dep].append(service)
    
    most_depended_on = sorted([(s, len(deps)) for s, deps in service_dependents.items()], key=lambda x: x[1], reverse=True)[:5]
    
    insights = []
    
    # Add insights about most dependent services
    insights.append("#### Services with Most Dependencies")
    insights.append("")
    insights.append("These services depend on many other services and may benefit from refactoring to reduce coupling:")
    insights.append("")
    for service, count in most_dependent:
        insights.append(f"- **{service}**: Depends on {count} other services")
    
    insights.append("")
    
    # Add insights about most depended-on services
    insights.append("#### Most Depended-On Services")
    insights.append("")
    insights.append("These services are depended on by many other services and may represent core functionality:")
    insights.append("")
    for service, count in most_depended_on:
        insights.append(f"- **{service}**: Depended on by {count} other services")
    
    return "\n".join(insights)

def generate_circular_dependencies_section(circular_dependencies_report: Dict[str, Any]) -> str:
    """
    Generate the circular dependencies section.
    
    Args:
        circular_dependencies_report: Circular dependencies report data
        
    Returns:
        Markdown text for the circular dependencies section
    """
    circular_dependencies = circular_dependencies_report.get('circular_dependencies', [])
    
    if not circular_dependencies:
        return "No circular dependencies were found."
    
    section = ["The following circular dependencies were identified:"]
    section.append("")
    
    for i, dep in enumerate(circular_dependencies[:10]):  # Limit to 10 for readability
        service1 = dep.get('service1', '')
        service2 = dep.get('service2', '')
        
        section.append(f"### {service1} <-> {service2}")
        section.append("")
        
        # Add examples of imports
        service1_to_service2 = dep.get('service1_to_service2', [])
        if service1_to_service2:
            section.append(f"**{service1}** imports from **{service2}**:")
            section.append("")
            for i, import_info in enumerate(service1_to_service2[:3]):  # Limit to 3 examples
                module = import_info.get('module', '')
                import_name = import_info.get('import', '')
                section.append(f"- `{os.path.basename(module)}` imports `{import_name}`")
            section.append("")
        
        service2_to_service1 = dep.get('service2_to_service1', [])
        if service2_to_service1:
            section.append(f"**{service2}** imports from **{service1}**:")
            section.append("")
            for i, import_info in enumerate(service2_to_service1[:3]):  # Limit to 3 examples
                module = import_info.get('module', '')
                import_name = import_info.get('import', '')
                section.append(f"- `{os.path.basename(module)}` imports `{import_name}`")
            section.append("")
        
        # Add suggestions
        suggestions = dep.get('suggestions', [])
        if suggestions:
            section.append("**Suggested fixes:**")
            section.append("")
            for suggestion in suggestions:
                section.append(f"- {suggestion}")
            section.append("")
    
    if len(circular_dependencies) > 10:
        section.append(f"*...and {len(circular_dependencies) - 10} more circular dependencies.*")
    
    return "\n".join(section)

def generate_duplicate_code_section(duplicate_code_report: Dict[str, Any]) -> str:
    """
    Generate the duplicate code section.
    
    Args:
        duplicate_code_report: Duplicate code report data
        
    Returns:
        Markdown text for the duplicate code section
    """
    duplicates = duplicate_code_report.get('duplicates', [])
    
    if not duplicates:
        return "No duplicate code was found."
    
    section = ["The analysis identified the following groups of duplicate code:"]
    section.append("")
    
    for i, group in enumerate(duplicates[:5]):  # Limit to 5 for readability
        blocks = group.get('blocks', [])
        similarity = group.get('similarity', 0.0)
        
        section.append(f"### Duplicate Group {i+1} (Similarity: {similarity:.2f})")
        section.append("")
        
        for j, block in enumerate(blocks):
            file_path = block.get('file', '')
            block_type = block.get('type', '')
            name = block.get('name', '')
            start_line = block.get('start_line', 0)
            end_line = block.get('end_line', 0)
            
            section.append(f"**Block {j+1}:** `{os.path.basename(file_path)}` (lines {start_line}-{end_line})")
            section.append(f"- Type: {block_type}")
            section.append(f"- Name: {name}")
            section.append("")
        
        # Add recommendation
        section.append("**Recommendation:** Consider extracting this functionality into a common utility or library.")
        section.append("")
    
    if len(duplicates) > 5:
        section.append(f"*...and {len(duplicates) - 5} more duplicate groups.*")
    
    return "\n".join(section)

def generate_service_analysis_section(dependency_report: Dict[str, Any], project_root: str) -> str:
    """
    Generate the service analysis section.
    
    Args:
        dependency_report: Dependency report data
        project_root: Root directory of the project
        
    Returns:
        Markdown text for the service analysis section
    """
    services = dependency_report.get('services', {})
    
    if not services:
        return "No services were found."
    
    section = ["The following services were analyzed:"]
    section.append("")
    
    for service_name, service_info in list(services.items())[:10]:  # Limit to 10 for readability
        service_path = service_info.get('path', '')
        
        # Count files and lines
        file_count = 0
        line_count = 0
        
        if os.path.exists(service_path):
            for root, _, files in os.walk(service_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        file_count += 1
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                line_count += len(f.readlines())
                        except Exception:
                            pass
        
        # Get dependencies and dependents
        dependencies = service_info.get('dependencies', [])
        dependents = service_info.get('dependents', [])
        
        section.append(f"### {service_name}")
        section.append("")
        section.append(f"- **Files:** {file_count}")
        section.append(f"- **Lines of Code:** {line_count}")
        section.append(f"- **Dependencies:** {len(dependencies)}")
        section.append(f"- **Dependents:** {len(dependents)}")
        section.append("")
        
        if dependencies:
            section.append("**Dependencies:**")
            section.append("")
            for dep in dependencies[:5]:  # Limit to 5 for readability
                section.append(f"- {dep}")
            if len(dependencies) > 5:
                section.append(f"- *...and {len(dependencies) - 5} more*")
            section.append("")
        
        if dependents:
            section.append("**Dependents:**")
            section.append("")
            for dep in dependents[:5]:  # Limit to 5 for readability
                section.append(f"- {dep}")
            if len(dependents) > 5:
                section.append(f"- *...and {len(dependents) - 5} more*")
            section.append("")
    
    if len(services) > 10:
        section.append(f"*...and {len(services) - 10} more services.*")
    
    return "\n".join(section)

def generate_recommendations(dependency_report: Dict[str, Any], duplicate_code_report: Dict[str, Any], circular_dependencies_report: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate recommendations based on the analysis.
    
    Args:
        dependency_report: Dependency report data
        duplicate_code_report: Duplicate code report data
        circular_dependencies_report: Circular dependencies report data
        
    Returns:
        Dictionary of recommendation sections
    """
    recommendations = {}
    
    # Circular dependencies recommendations
    circular_dependencies = circular_dependencies_report.get('circular_dependencies', [])
    if circular_dependencies:
        circular_recs = []
        circular_recs.append("To resolve the identified circular dependencies, consider the following approaches:")
        circular_recs.append("")
        
        # Group by suggestion type
        suggestion_groups = {}
        for dep in circular_dependencies:
            for suggestion in dep.get('suggestions', []):
                if suggestion not in suggestion_groups:
                    suggestion_groups[suggestion] = []
                suggestion_groups[suggestion].append((dep.get('service1', ''), dep.get('service2', '')))
        
        for suggestion, services in suggestion_groups.items():
            circular_recs.append(f"- **{suggestion}**")
            circular_recs.append("  - Applicable to:")
            for service1, service2 in services[:3]:  # Limit to 3 examples
                circular_recs.append(f"    - {service1} <-> {service2}")
            if len(services) > 3:
                circular_recs.append(f"    - *...and {len(services) - 3} more*")
            circular_recs.append("")
        
        recommendations['circular_dependencies_recommendations'] = "\n".join(circular_recs)
    else:
        recommendations['circular_dependencies_recommendations'] = "No circular dependencies were found."
    
    # Duplicate code recommendations
    duplicates = duplicate_code_report.get('duplicates', [])
    if duplicates:
        duplicate_recs = []
        duplicate_recs.append("To address the duplicate code issues, consider the following approaches:")
        duplicate_recs.append("")
        
        # Group by similarity
        high_similarity = [g for g in duplicates if g.get('similarity', 0.0) > 0.9]
        medium_similarity = [g for g in duplicates if 0.8 <= g.get('similarity', 0.0) <= 0.9]
        low_similarity = [g for g in duplicates if g.get('similarity', 0.0) < 0.8]
        
        if high_similarity:
            duplicate_recs.append("- **Extract Common Utilities for High-Similarity Code (>90%)**")
            duplicate_recs.append("  - Create utility functions or classes in a common library")
            duplicate_recs.append("  - Replace duplicate implementations with calls to the common utilities")
            duplicate_recs.append("  - Example candidates:")
            
            for i, group in enumerate(high_similarity[:3]):  # Limit to 3 examples
                blocks = group.get('blocks', [])
                files = [os.path.basename(block.get('file', '')) for block in blocks]
                duplicate_recs.append(f"    - Group {i+1}: {', '.join(files)}")
            
            if len(high_similarity) > 3:
                duplicate_recs.append(f"    - *...and {len(high_similarity) - 3} more groups*")
            
            duplicate_recs.append("")
        
        if medium_similarity:
            duplicate_recs.append("- **Create Parameterized Components for Medium-Similarity Code (80-90%)**")
            duplicate_recs.append("  - Identify the varying parts of the code")
            duplicate_recs.append("  - Create parameterized functions or classes")
            duplicate_recs.append("  - Replace duplicates with calls to the parameterized components")
            duplicate_recs.append("")
        
        if low_similarity:
            duplicate_recs.append("- **Review and Refactor Low-Similarity Code (<80%)**")
            duplicate_recs.append("  - Manually review the code to identify common patterns")
            duplicate_recs.append("  - Consider applying design patterns to reduce duplication")
            duplicate_recs.append("  - Focus on the most critical areas first")
            duplicate_recs.append("")
        
        recommendations['duplicate_code_recommendations'] = "\n".join(duplicate_recs)
    else:
        recommendations['duplicate_code_recommendations'] = "No duplicate code was found."
    
    # Service structure recommendations
    services = dependency_report.get('services', {})
    if services:
        service_recs = []
        service_recs.append("To improve the structure of services, consider the following approaches:")
        service_recs.append("")
        
        # Find services with most dependencies
        service_dep_count = {s: len(deps) for s, deps in dependency_report.get('service_dependencies', {}).items()}
        most_dependent = sorted(service_dep_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if most_dependent:
            service_recs.append("- **Refactor Highly Coupled Services**")
            service_recs.append("  - Focus on services with many dependencies:")
            for service, count in most_dependent:
                service_recs.append(f"    - {service} ({count} dependencies)")
            service_recs.append("  - Consider breaking these services into smaller, more focused components")
            service_recs.append("  - Use dependency injection to reduce direct coupling")
            service_recs.append("")
        
        service_recs.append("- **Standardize Service Structure**")
        service_recs.append("  - Ensure all services follow a consistent directory structure")
        service_recs.append("  - Implement clear separation of concerns within each service")
        service_recs.append("  - Use consistent naming conventions across all services")
        service_recs.append("")
        
        recommendations['service_structure_recommendations'] = "\n".join(service_recs)
    else:
        recommendations['service_structure_recommendations'] = "No services were found."
    
    # Modularity recommendations
    modularity_recs = []
    modularity_recs.append("To enhance the modularity of the platform, consider the following approaches:")
    modularity_recs.append("")
    modularity_recs.append("- **Define Clear Service Boundaries**")
    modularity_recs.append("  - Ensure each service has a well-defined responsibility")
    modularity_recs.append("  - Minimize overlap between services")
    modularity_recs.append("  - Document service responsibilities and interfaces")
    modularity_recs.append("")
    modularity_recs.append("- **Implement Domain-Driven Design Principles**")
    modularity_recs.append("  - Identify bounded contexts within the platform")
    modularity_recs.append("  - Align service boundaries with bounded contexts")
    modularity_recs.append("  - Use ubiquitous language within each context")
    modularity_recs.append("")
    modularity_recs.append("- **Adopt Microservice Best Practices**")
    modularity_recs.append("  - Ensure services are independently deployable")
    modularity_recs.append("  - Implement proper service discovery and communication")
    modularity_recs.append("  - Use event-driven communication where appropriate")
    modularity_recs.append("")
    
    recommendations['modularity_recommendations'] = "\n".join(modularity_recs)
    
    # Interface recommendations
    interface_recs = []
    interface_recs.append("To standardize interfaces across the platform, consider the following approaches:")
    interface_recs.append("")
    interface_recs.append("- **Define Service Contracts**")
    interface_recs.append("  - Create clear interface definitions for each service")
    interface_recs.append("  - Document input/output formats and error handling")
    interface_recs.append("  - Version interfaces appropriately")
    interface_recs.append("")
    interface_recs.append("- **Implement Interface-Based Design**")
    interface_recs.append("  - Use abstract base classes or interfaces to define contracts")
    interface_recs.append("  - Implement the adapter pattern for external dependencies")
    interface_recs.append("  - Use dependency injection to provide implementations")
    interface_recs.append("")
    interface_recs.append("- **Standardize Error Handling**")
    interface_recs.append("  - Define a consistent error model across all services")
    interface_recs.append("  - Use custom exceptions with clear semantics")
    interface_recs.append("  - Implement proper error propagation and handling")
    interface_recs.append("")
    
    recommendations['interface_recommendations'] = "\n".join(interface_recs)
    
    return recommendations

def generate_report(project_root: str, output_dir: str) -> None:
    """
    Generate an optimization report.
    
    Args:
        project_root: Root directory of the project
        output_dir: Directory to save output files
    """
    logger.info("Generating optimization report...")
    
    # Load dependency report
    dependency_report_path = os.path.join(output_dir, 'dependency-report.json')
    dependency_report = load_dependency_report(dependency_report_path)
    
    # Load duplicate code report
    duplicate_code_report_path = os.path.join(output_dir, 'duplicate-code-report.json')
    duplicate_code_report = load_duplicate_code_report(duplicate_code_report_path)
    
    # Load circular dependencies report
    circular_dependencies_report_path = os.path.join(output_dir, 'circular_dependencies_report.json')
    circular_dependencies_report = load_circular_dependencies_report(circular_dependencies_report_path)
    
    # Count files
    file_count = 0
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                file_count += 1
    
    # Generate recommendations
    recommendations = generate_recommendations(
        dependency_report,
        duplicate_code_report,
        circular_dependencies_report
    )
    
    # Generate report
    report = MARKDOWN_TEMPLATE.format(
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        service_count=len(dependency_report.get('services', {})),
        file_count=file_count,
        dependency_count=sum(len(deps) for deps in dependency_report.get('service_dependencies', {}).values()),
        circular_count=len(dependency_report.get('circular_dependencies', [])),
        duplicate_groups_count=duplicate_code_report.get('duplicate_groups', 0),
        dependency_insights=generate_dependency_insights(dependency_report),
        circular_dependencies_count=len(circular_dependencies_report.get('circular_dependencies', [])),
        circular_dependencies_section=generate_circular_dependencies_section(circular_dependencies_report),
        duplicate_code_section=generate_duplicate_code_section(duplicate_code_report),
        service_analysis_section=generate_service_analysis_section(dependency_report, project_root),
        **recommendations
    )
    
    # Save report
    report_path = os.path.join(output_dir, 'optimization_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Optimization report saved to {report_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate optimization report")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate report
    generate_report(args.project_root, args.output_dir)

if __name__ == "__main__":
    main()
