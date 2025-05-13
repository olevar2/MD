"""
Script to identify services with inconsistent monitoring and observability.

This script scans the codebase for services with inconsistent monitoring and observability
implementation and identifies files that need to be updated to use the standardized
monitoring and observability system.
"""

import os
import re
import json
from typing import Dict, List, Set, Tuple, Optional, Any

# Service directories to scan
SERVICE_DIRS = [
    "analysis-engine-service",
    "data-pipeline-service",
    "feature-store-service",
    "ml-integration-service",
    "ml-workbench-service",
    "monitoring-alerting-service",
    "portfolio-management-service",
    "strategy-execution-engine",
    "trading-gateway-service",
    "ui-service"
]

# Directories to skip
SKIP_DIRS = ['.git', '.venv', 'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist']

# Monitoring and observability patterns
MONITORING_PATTERNS = {
    "health_check": [
        r"@app\.get\(['\"]\/health['\"]\)",
        r"@app\.get\(['\"]\/ready['\"]\)",
        r"health_manager\.register_health_check\(",
        r"HealthManager\(",
        r"HealthCheck\(",
        r"add_health_check_to_app\("
    ],
    "metrics": [
        r"@app\.get\(['\"]\/metrics['\"]\)",
        r"metrics_manager\.create_counter\(",
        r"metrics_manager\.create_gauge\(",
        r"metrics_manager\.create_histogram\(",
        r"metrics_manager\.create_summary\(",
        r"MetricsManager\(",
        r"track_time\(",
        r"prometheus_client\.generate_latest\("
    ],
    "tracing": [
        r"TracingManager\(",
        r"trace_function\(",
        r"tracing_manager\.start_span\(",
        r"tracing_manager\.extract_context\(",
        r"OpenTelemetry",
        r"OTLPSpanExporter\(",
        r"TracerProvider\("
    ],
    "logging": [
        r"configure_logging\(",
        r"get_logger\(",
        r"log_with_context\(",
        r"LoggingManager\(",
        r"JsonFormatter\(",
        r"log_execution\(",
        r"CorrelationIdFilter\(",
        r"StructuredLogFormatter\("
    ]
}

# Standardized monitoring and observability patterns
STANDARDIZED_MONITORING_PATTERNS = {
    "health_check": [
        r"from\s+common_lib\.monitoring\s+import\s+(?:[^,]+,\s*)*(?:HealthManager|HealthCheck|HealthStatus)",
        r"health_manager\s*=\s*HealthManager\(",
        r"health_manager\.register_health_check\("
    ],
    "metrics": [
        r"from\s+common_lib\.monitoring\s+import\s+(?:[^,]+,\s*)*(?:MetricsManager|MetricType|track_time)",
        r"metrics_manager\s*=\s*MetricsManager\(",
        r"metrics_manager\.create_counter\(",
        r"@track_time\("
    ],
    "tracing": [
        r"from\s+common_lib\.monitoring\s+import\s+(?:[^,]+,\s*)*(?:TracingManager|trace_function)",
        r"tracing_manager\s*=\s*TracingManager\(",
        r"@trace_function\("
    ],
    "logging": [
        r"from\s+common_lib\.monitoring\s+import\s+(?:[^,]+,\s*)*(?:configure_logging|get_logger|log_with_context)",
        r"configure_logging\(",
        r"logger\s*=\s*get_logger\("
    ]
}


class MonitoringImplementationChecker:
    """
    Class to check monitoring and observability implementation in services.
    """
    
    def __init__(self, root_dir: str = '.'):
        """
        Initialize the checker.
        
        Args:
            root_dir: Root directory to scan
        """
        self.root_dir = root_dir
        self.results = {}
    
    def scan_codebase(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan the codebase for monitoring and observability implementation.
        
        Returns:
            Dictionary mapping service names to dictionaries with monitoring and observability implementation
        """
        for service_dir in SERVICE_DIRS:
            service_path = os.path.join(self.root_dir, service_dir)
            if not os.path.exists(service_path):
                continue
            
            self.results[service_dir] = {
                "health_check": {
                    "implemented": False,
                    "standardized": False,
                    "files": []
                },
                "metrics": {
                    "implemented": False,
                    "standardized": False,
                    "files": []
                },
                "tracing": {
                    "implemented": False,
                    "standardized": False,
                    "files": []
                },
                "logging": {
                    "implemented": False,
                    "standardized": False,
                    "files": []
                }
            }
            
            for dirpath, dirnames, filenames in os.walk(service_path):
                # Skip directories
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                
                for filename in filenames:
                    if not filename.endswith('.py'):
                        continue
                    
                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, self.root_dir)
                    
                    try:
                        self._analyze_file(file_path, rel_path, service_dir)
                    except Exception as e:
                        print(f"Error analyzing {rel_path}: {str(e)}")
        
        return self.results
    
    def _analyze_file(self, file_path: str, rel_path: str, service_dir: str) -> None:
        """
        Analyze a file for monitoring and observability patterns.
        
        Args:
            file_path: Path to the file
            rel_path: Relative path to the file
            service_dir: Service directory
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for monitoring and observability patterns
        for component, patterns in MONITORING_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    self.results[service_dir][component]["implemented"] = True
                    if rel_path not in self.results[service_dir][component]["files"]:
                        self.results[service_dir][component]["files"].append(rel_path)
        
        # Check for standardized monitoring and observability patterns
        for component, patterns in STANDARDIZED_MONITORING_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    self.results[service_dir][component]["standardized"] = True


def main():
    """Main function."""
    checker = MonitoringImplementationChecker()
    results = checker.scan_codebase()
    
    # Print results
    print("Monitoring and Observability Implementation Analysis")
    print("==================================================")
    
    for service_dir, components in results.items():
        print(f"\n{service_dir}:")
        
        for component, info in components.items():
            status = "Not implemented"
            if info["implemented"]:
                status = "Standardized" if info["standardized"] else "Custom implementation"
            
            print(f"  {component.capitalize()}: {status}")
            
            if info["implemented"] and info["files"]:
                print(f"    Files:")
                for file in info["files"][:3]:  # Show only the first 3 files
                    print(f"      {file}")
                if len(info["files"]) > 3:
                    print(f"      ... and {len(info['files']) - 3} more")
    
    # Save results to file
    with open('monitoring_implementation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to monitoring_implementation_analysis.json")
    
    # Print summary
    print("\nSummary:")
    
    components = ["health_check", "metrics", "tracing", "logging"]
    
    for component in components:
        implemented_count = sum(1 for service in results.values() if service[component]["implemented"])
        standardized_count = sum(1 for service in results.values() if service[component]["standardized"])
        
        print(f"  {component.capitalize()}: {implemented_count}/{len(results)} services implemented, {standardized_count}/{implemented_count} standardized")


if __name__ == "__main__":
    main()
