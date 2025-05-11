#!/usr/bin/env python3
"""
Error Handling and Resilience Analyzer

This script analyzes the current implementation of error handling and resilience patterns
in the Forex Trading Platform. It identifies error types, error handling patterns,
resilience patterns, and generates a report of the findings.
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
logger = logging.getLogger("error_resilience_analyzer")

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

class ErrorResilienceAnalyzer:
    """Analyzes the implementation of error handling and resilience patterns."""

    def __init__(self, root_dir: str):
        """Initialize the analyzer with the root directory of the project."""
        self.root_dir = Path(root_dir)
        self.error_types: Dict[str, List[str]] = {}  # service -> error types
        self.error_handlers: Dict[str, List[str]] = {}  # service -> error handlers
        self.resilience_patterns: Dict[str, Dict[str, List[str]]] = {}  # service -> pattern type -> patterns
        self.error_handling_patterns: Dict[str, List[Dict[str, Any]]] = {}  # service -> error handling patterns
        self.resilience_usage: Dict[str, List[Dict[str, Any]]] = {}  # service -> resilience usage

    def analyze(self) -> Dict[str, Any]:
        """Analyze the project and return the results."""
        logger.info("Starting error handling and resilience analysis...")
        
        # Find all error types
        self._find_error_types()
        
        # Find all error handlers
        self._find_error_handlers()
        
        # Find all resilience patterns
        self._find_resilience_patterns()
        
        # Find all error handling patterns
        self._find_error_handling_patterns()
        
        # Find all resilience usage
        self._find_resilience_usage()
        
        # Generate the report
        report = self._generate_report()
        
        logger.info("Error handling and resilience analysis complete.")
        return report

    def _find_error_types(self) -> None:
        """Find all error types in the project."""
        logger.info("Finding error types...")
        
        # Look for error type definitions in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            # Look for error type files
            error_files = []
            for root, _, files in os.walk(service_path):
                for file_name in files:
                    if ('error' in file_name.lower() or 'exception' in file_name.lower()) and file_name.endswith('.py'):
                        error_files.append(Path(root) / file_name)
            
            # Look for error type definitions in error files
            for file_path in error_files:
                if service_dir not in self.error_types:
                    self.error_types[service_dir] = []
                
                # Parse the file to find error type definitions
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all class definitions that inherit from Exception
                error_pattern = r'class\s+([A-Za-z0-9_]+(?:Error|Exception))\s*\([^)]*(?:Exception|Error)[^)]*\):'
                errors = re.findall(error_pattern, content)
                
                for error in errors:
                    self.error_types[service_dir].append(error)
        
        logger.info(f"Found {sum(len(errors) for errors in self.error_types.values())} error types.")

    def _find_error_handlers(self) -> None:
        """Find all error handlers in the project."""
        logger.info("Finding error handlers...")
        
        # Look for error handler implementations in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            # Look for error handler files
            handler_files = []
            for root, _, files in os.walk(service_path):
                for file_name in files:
                    if ('error' in file_name.lower() or 'exception' in file_name.lower() or 'handler' in file_name.lower()) and file_name.endswith('.py'):
                        handler_files.append(Path(root) / file_name)
            
            # Look for error handler definitions in handler files
            for file_path in handler_files:
                if service_dir not in self.error_handlers:
                    self.error_handlers[service_dir] = []
                
                # Parse the file to find error handler definitions
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all function definitions that handle errors
                handler_pattern = r'def\s+([A-Za-z0-9_]+(?:_error_handler|_exception_handler|handle_[A-Za-z0-9_]+_error|handle_[A-Za-z0-9_]+_exception))\s*\('
                handlers = re.findall(handler_pattern, content)
                
                for handler in handlers:
                    self.error_handlers[service_dir].append(handler)
        
        logger.info(f"Found {sum(len(handlers) for handlers in self.error_handlers.values())} error handlers.")

    def _find_resilience_patterns(self) -> None:
        """Find all resilience patterns in the project."""
        logger.info("Finding resilience patterns...")
        
        # Define resilience pattern types
        pattern_types = {
            'circuit_breaker': ['circuit_breaker', 'circuitbreaker'],
            'retry': ['retry', 'retries', 'retry_policy'],
            'timeout': ['timeout', 'timeouts'],
            'bulkhead': ['bulkhead'],
            'fallback': ['fallback']
        }
        
        # Look for resilience pattern implementations in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            # Look for resilience pattern files
            pattern_files = []
            for root, _, files in os.walk(service_path):
                for file_name in files:
                    if any(pattern in file_name.lower() for patterns in pattern_types.values() for pattern in patterns) and file_name.endswith('.py'):
                        pattern_files.append(Path(root) / file_name)
            
            # Look for resilience pattern definitions in pattern files
            for file_path in pattern_files:
                if service_dir not in self.resilience_patterns:
                    self.resilience_patterns[service_dir] = {pattern_type: [] for pattern_type in pattern_types}
                
                # Parse the file to find resilience pattern definitions
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all class and function definitions for each pattern type
                for pattern_type, patterns in pattern_types.items():
                    # Find class definitions
                    class_pattern = r'class\s+([A-Za-z0-9_]+(?:' + '|'.join(patterns) + r')[A-Za-z0-9_]*)\s*\('
                    classes = re.findall(class_pattern, content, re.IGNORECASE)
                    
                    # Find function definitions
                    func_pattern = r'def\s+([A-Za-z0-9_]+(?:' + '|'.join(patterns) + r')[A-Za-z0-9_]*)\s*\('
                    funcs = re.findall(func_pattern, content, re.IGNORECASE)
                    
                    # Add all patterns
                    self.resilience_patterns[service_dir][pattern_type].extend(classes)
                    self.resilience_patterns[service_dir][pattern_type].extend(funcs)
        
        logger.info(f"Found {sum(len(patterns) for service in self.resilience_patterns.values() for patterns in service.values())} resilience patterns.")

    def _find_error_handling_patterns(self) -> None:
        """Find all error handling patterns in the project."""
        logger.info("Finding error handling patterns...")
        
        # Look for error handling patterns in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            if service_dir not in self.error_handling_patterns:
                self.error_handling_patterns[service_dir] = []
            
            # Look for Python files
            for file_path in service_path.glob('**/*.py'):
                # Parse the file to find error handling patterns
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all try-except blocks
                try_except_pattern = r'try:.*?except\s+([A-Za-z0-9_\.]+(?:Error|Exception))?(?:\s+as\s+([A-Za-z0-9_]+))?:'
                try_except_blocks = re.findall(try_except_pattern, content, re.DOTALL)
                
                for exception_type, exception_var in try_except_blocks:
                    self.error_handling_patterns[service_dir].append({
                        'file': str(file_path.relative_to(self.root_dir)),
                        'pattern': 'try-except',
                        'exception_type': exception_type.strip() if exception_type else 'Exception',
                        'exception_var': exception_var.strip() if exception_var else None
                    })
                
                # Find all error handler registrations
                handler_pattern = r'app\.add_exception_handler\s*\(\s*([A-Za-z0-9_\.]+(?:Error|Exception))\s*,\s*([A-Za-z0-9_\.]+)\s*\)'
                handlers = re.findall(handler_pattern, content)
                
                for exception_type, handler_func in handlers:
                    self.error_handling_patterns[service_dir].append({
                        'file': str(file_path.relative_to(self.root_dir)),
                        'pattern': 'exception-handler',
                        'exception_type': exception_type.strip(),
                        'handler_func': handler_func.strip()
                    })
        
        logger.info(f"Found {sum(len(patterns) for patterns in self.error_handling_patterns.values())} error handling patterns.")

    def _find_resilience_usage(self) -> None:
        """Find all resilience pattern usage in the project."""
        logger.info("Finding resilience pattern usage...")
        
        # Define resilience pattern usage patterns
        usage_patterns = {
            'circuit_breaker': [
                r'CircuitBreaker\s*\(',
                r'circuit_breaker\s*\(',
                r'with\s+circuit_breaker\s*\(',
                r'@circuit_breaker'
            ],
            'retry': [
                r'RetryPolicy\s*\(',
                r'retry_with_policy\s*\(',
                r'with\s+retry\s*\(',
                r'@retry',
                r'@retry_with_policy'
            ],
            'timeout': [
                r'with\s+timeout\s*\(',
                r'@timeout',
                r'withTimeout\s*\('
            ],
            'bulkhead': [
                r'Bulkhead\s*\(',
                r'bulkhead\s*\(',
                r'with\s+bulkhead\s*\(',
                r'@bulkhead'
            ],
            'fallback': [
                r'with\s+fallback\s*\(',
                r'@fallback',
                r'withFallback\s*\('
            ]
        }
        
        # Look for resilience pattern usage in all services
        for service_dir in SERVICE_DIRS:
            service_path = self.root_dir / service_dir
            if not service_path.exists():
                continue
            
            if service_dir not in self.resilience_usage:
                self.resilience_usage[service_dir] = []
            
            # Look for Python files
            for file_path in service_path.glob('**/*.py'):
                # Parse the file to find resilience pattern usage
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all resilience pattern usage
                for pattern_type, patterns in usage_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        
                        for _ in matches:
                            self.resilience_usage[service_dir].append({
                                'file': str(file_path.relative_to(self.root_dir)),
                                'pattern_type': pattern_type,
                                'pattern': pattern
                            })
        
        logger.info(f"Found {sum(len(usage) for usage in self.resilience_usage.values())} resilience pattern usages.")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate a report of the findings."""
        logger.info("Generating report...")
        
        report = {
            "error_types": self.error_types,
            "error_handlers": self.error_handlers,
            "resilience_patterns": self.resilience_patterns,
            "error_handling_patterns": self.error_handling_patterns,
            "resilience_usage": self.resilience_usage,
            "summary": {
                "total_error_types": sum(len(errors) for errors in self.error_types.values()),
                "total_error_handlers": sum(len(handlers) for handlers in self.error_handlers.values()),
                "total_resilience_patterns": sum(len(patterns) for service in self.resilience_patterns.values() for patterns in service.values()),
                "total_error_handling_patterns": sum(len(patterns) for patterns in self.error_handling_patterns.values()),
                "total_resilience_usage": sum(len(usage) for usage in self.resilience_usage.values()),
            }
        }
        
        # Add analysis of error handling coverage
        report["analysis"] = {
            "error_handling_coverage": {},
            "resilience_coverage": {},
            "error_handling_issues": [],
            "resilience_issues": [],
        }
        
        # Analyze error handling coverage
        for service_dir in SERVICE_DIRS:
            if service_dir not in self.error_handling_patterns:
                continue
            
            report["analysis"]["error_handling_coverage"][service_dir] = {
                "total_patterns": len(self.error_handling_patterns[service_dir]),
                "exception_types": {},
                "custom_error_usage": 0,
                "standard_error_usage": 0,
                "custom_error_percentage": 0.0,
            }
            
            # Count exception types
            for pattern in self.error_handling_patterns[service_dir]:
                exception_type = pattern.get('exception_type')
                if exception_type:
                    if exception_type not in report["analysis"]["error_handling_coverage"][service_dir]["exception_types"]:
                        report["analysis"]["error_handling_coverage"][service_dir]["exception_types"][exception_type] = 0
                    
                    report["analysis"]["error_handling_coverage"][service_dir]["exception_types"][exception_type] += 1
                    
                    # Check if it's a custom error
                    if any(error_type in exception_type for service in self.error_types.values() for error_type in service):
                        report["analysis"]["error_handling_coverage"][service_dir]["custom_error_usage"] += 1
                    else:
                        report["analysis"]["error_handling_coverage"][service_dir]["standard_error_usage"] += 1
            
            # Calculate custom error percentage
            total_errors = report["analysis"]["error_handling_coverage"][service_dir]["custom_error_usage"] + report["analysis"]["error_handling_coverage"][service_dir]["standard_error_usage"]
            if total_errors > 0:
                report["analysis"]["error_handling_coverage"][service_dir]["custom_error_percentage"] = (
                    report["analysis"]["error_handling_coverage"][service_dir]["custom_error_usage"] / total_errors * 100.0
                )
        
        # Analyze resilience coverage
        for service_dir in SERVICE_DIRS:
            if service_dir not in self.resilience_usage:
                continue
            
            report["analysis"]["resilience_coverage"][service_dir] = {
                "total_usage": len(self.resilience_usage[service_dir]),
                "pattern_types": {},
            }
            
            # Count pattern types
            for usage in self.resilience_usage[service_dir]:
                pattern_type = usage.get('pattern_type')
                if pattern_type:
                    if pattern_type not in report["analysis"]["resilience_coverage"][service_dir]["pattern_types"]:
                        report["analysis"]["resilience_coverage"][service_dir]["pattern_types"][pattern_type] = 0
                    
                    report["analysis"]["resilience_coverage"][service_dir]["pattern_types"][pattern_type] += 1
        
        # Identify error handling issues
        for service_dir in SERVICE_DIRS:
            # Skip common-lib and core-foundations
            if service_dir in ['common-lib', 'core-foundations']:
                continue
            
            # Check if the service has error handling patterns
            if service_dir not in self.error_handling_patterns or not self.error_handling_patterns[service_dir]:
                report["analysis"]["error_handling_issues"].append({
                    "service": service_dir,
                    "issue": "No error handling patterns found"
                })
                continue
            
            # Check if the service uses custom errors
            if service_dir in report["analysis"]["error_handling_coverage"] and report["analysis"]["error_handling_coverage"][service_dir]["custom_error_percentage"] < 50.0:
                report["analysis"]["error_handling_issues"].append({
                    "service": service_dir,
                    "issue": f"Low custom error usage ({report['analysis']['error_handling_coverage'][service_dir]['custom_error_percentage']:.2f}%)"
                })
        
        # Identify resilience issues
        for service_dir in SERVICE_DIRS:
            # Skip common-lib and core-foundations
            if service_dir in ['common-lib', 'core-foundations']:
                continue
            
            # Check if the service has resilience usage
            if service_dir not in self.resilience_usage or not self.resilience_usage[service_dir]:
                report["analysis"]["resilience_issues"].append({
                    "service": service_dir,
                    "issue": "No resilience patterns used"
                })
                continue
            
            # Check if the service uses all resilience patterns
            if service_dir in report["analysis"]["resilience_coverage"]:
                pattern_types = report["analysis"]["resilience_coverage"][service_dir]["pattern_types"]
                missing_patterns = set(['circuit_breaker', 'retry', 'timeout', 'bulkhead', 'fallback']) - set(pattern_types.keys())
                
                if missing_patterns:
                    report["analysis"]["resilience_issues"].append({
                        "service": service_dir,
                        "issue": f"Missing resilience patterns: {', '.join(missing_patterns)}"
                    })
        
        logger.info("Report generation complete.")
        return report

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze error handling and resilience patterns in the Forex Trading Platform.")
    parser.add_argument("--root", default=".", help="Root directory of the project")
    parser.add_argument("--output", default="error_resilience_analysis.json", help="Output file for the analysis report")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format")
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = ErrorResilienceAnalyzer(args.root)
    
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
            f.write("# Error Handling and Resilience Analysis Report\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total Error Types: {report['summary']['total_error_types']}\n")
            f.write(f"- Total Error Handlers: {report['summary']['total_error_handlers']}\n")
            f.write(f"- Total Resilience Patterns: {report['summary']['total_resilience_patterns']}\n")
            f.write(f"- Total Error Handling Patterns: {report['summary']['total_error_handling_patterns']}\n")
            f.write(f"- Total Resilience Usage: {report['summary']['total_resilience_usage']}\n\n")
            
            f.write("## Error Handling Coverage\n\n")
            f.write("| Service | Total Patterns | Custom Error Usage | Standard Error Usage | Custom Error Percentage |\n")
            f.write("|---------|---------------|-------------------|---------------------|-------------------------|\n")
            for service, coverage in report["analysis"]["error_handling_coverage"].items():
                f.write(f"| {service} | {coverage['total_patterns']} | {coverage['custom_error_usage']} | {coverage['standard_error_usage']} | {coverage['custom_error_percentage']:.2f}% |\n")
            f.write("\n")
            
            f.write("## Resilience Coverage\n\n")
            f.write("| Service | Total Usage | Circuit Breaker | Retry | Timeout | Bulkhead | Fallback |\n")
            f.write("|---------|-------------|----------------|-------|---------|----------|----------|\n")
            for service, coverage in report["analysis"]["resilience_coverage"].items():
                pattern_types = coverage["pattern_types"]
                f.write(f"| {service} | {coverage['total_usage']} | {pattern_types.get('circuit_breaker', 0)} | {pattern_types.get('retry', 0)} | {pattern_types.get('timeout', 0)} | {pattern_types.get('bulkhead', 0)} | {pattern_types.get('fallback', 0)} |\n")
            f.write("\n")
            
            f.write("## Error Handling Issues\n\n")
            if report["analysis"]["error_handling_issues"]:
                f.write("| Service | Issue |\n")
                f.write("|---------|-------|\n")
                for issue in report["analysis"]["error_handling_issues"]:
                    f.write(f"| {issue['service']} | {issue['issue']} |\n")
            else:
                f.write("No error handling issues found.\n")
            f.write("\n")
            
            f.write("## Resilience Issues\n\n")
            if report["analysis"]["resilience_issues"]:
                f.write("| Service | Issue |\n")
                f.write("|---------|-------|\n")
                for issue in report["analysis"]["resilience_issues"]:
                    f.write(f"| {issue['service']} | {issue['issue']} |\n")
            else:
                f.write("No resilience issues found.\n")
        
        logger.info(f"Report saved to {markdown_output}")

if __name__ == "__main__":
    main()
