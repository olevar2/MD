#!/usr/bin/env python3
"""
Forex Trading Platform Scalability Assessment Analyzer

This script analyzes the scalability aspects of the forex trading platform:
1. Load testing results (if available in logs or reports)
2. Horizontal scaling capabilities (containerization, statelessness)
3. Resource scaling thresholds (auto-scaling configurations)
4. Performance under load (caching, connection pooling, etc.)

Output is a comprehensive JSON file that maps the scalability status of the platform.
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import concurrent.futures
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    ".git", ".github", ".pytest_cache", "__pycache__",
    "node_modules", ".venv", "venv", "env", ".vscode",
    "corrupted_backups", "tools/output", "tests", "testing"
}

# File extensions to analyze
PYTHON_EXTENSIONS = {".py"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
CONFIG_EXTENSIONS = {".json", ".yml", ".yaml", ".toml", ".ini", ".env", ".tf", ".hcl"}
LOG_EXTENSIONS = {".log"}
DOCKER_FILES = {"Dockerfile", "docker-compose.yml", "docker-compose.yaml"}
K8S_FILES = {"deployment.yaml", "service.yaml", "ingress.yaml", "configmap.yaml", "statefulset.yaml"}
ALL_EXTENSIONS = PYTHON_EXTENSIONS | JS_EXTENSIONS | CONFIG_EXTENSIONS | LOG_EXTENSIONS

# Patterns for detecting scalability mechanisms
LOAD_TESTING_PATTERNS = {
    'load_test_tools': [
        r'(locust|jmeter|gatling|k6|artillery|siege|wrk|ab|loadrunner)',
        r'(load_test|stress_test|performance_test|benchmark)',
        r'(concurrent_users|requests_per_second|response_time)'
    ],
    'load_test_results': [
        r'(throughput|latency|response_time)\s*[:=]\s*\d+',
        r'(success_rate|error_rate|failure_rate)\s*[:=]\s*\d+',
        r'(p50|p90|p95|p99)\s*[:=]\s*\d+',
        r'(min|max|avg|mean|median)\s*[:=]\s*\d+'
    ]
}

HORIZONTAL_SCALING_PATTERNS = {
    'containerization': [
        r'(docker|container|image|dockerfile)',
        r'(kubernetes|k8s|pod|deployment|service|ingress)',
        r'(FROM\s+\w+|COPY\s+|ADD\s+|RUN\s+|CMD\s+|ENTRYPOINT\s+)',
        r'(docker-compose|docker\s+build|docker\s+run)'
    ],
    'statelessness': [
        r'(stateless|idempotent|immutable)',
        r'(redis|memcached|cache)',
        r'(session\s+store|token\s+store)',
        r'(distributed\s+session|shared\s+nothing)'
    ],
    'service_discovery': [
        r'(service\s+discovery|service\s+registry)',
        r'(consul|etcd|zookeeper|eureka)',
        r'(dns\s+resolution|load\s+balancer)',
        r'(service\s+mesh|istio|linkerd|envoy)'
    ]
}

RESOURCE_SCALING_PATTERNS = {
    'auto_scaling': [
        r'(auto\s*scaling|autoscaler|scale\s+up|scale\s+down)',
        r'(horizontal\s+pod\s+autoscaler|vertical\s+pod\s+autoscaler)',
        r'(min\s+replicas|max\s+replicas|desired\s+count)',
        r'(scale\s+policy|scaling\s+trigger|scaling\s+threshold)'
    ],
    'resource_limits': [
        r'(cpu\s+limit|memory\s+limit|resource\s+limit)',
        r'(requests\s+cpu|requests\s+memory)',
        r'(limits\s+cpu|limits\s+memory)',
        r'(quota|resource\s+quota|limit\s+range)'
    ],
    'infrastructure_as_code': [
        r'(terraform|cloudformation|pulumi|ansible)',
        r'(infrastructure\s+as\s+code|iac)',
        r'(provision|deploy|stack)',
        r'(aws|azure|gcp|cloud)'
    ]
}

PERFORMANCE_PATTERNS = {
    'caching': [
        r'(cache|caching|cached)',
        r'(redis|memcached|elasticache)',
        r'(ttl|expiration|invalidation)',
        r'(lru|mru|fifo|lifo)'
    ],
    'connection_pooling': [
        r'(connection\s+pool|pooling)',
        r'(database\s+pool|db\s+pool)',
        r'(max\s+connections|min\s+connections|idle\s+connections)',
        r'(connection\s+timeout|connection\s+reuse)'
    ],
    'async_processing': [
        r'(async|await|promise|future|coroutine)',
        r'(queue|message\s+broker|pubsub)',
        r'(kafka|rabbitmq|sqs|sns|eventbridge)',
        r'(worker|consumer|producer|subscriber)'
    ],
    'database_optimization': [
        r'(index|partition|shard)',
        r'(read\s+replica|write\s+master|follower)',
        r'(query\s+optimization|execution\s+plan)',
        r'(denormalization|materialized\s+view)'
    ]
}

class ScalabilityAssessmentAnalyzer:
    """Analyzes the scalability aspects of the forex trading platform."""

    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.files = []
        self.docker_files = []
        self.k8s_files = []
        self.config_files = []
        self.load_testing = defaultdict(list)
        self.horizontal_scaling = defaultdict(list)
        self.resource_scaling = defaultdict(list)
        self.performance = defaultdict(list)

    def find_files(self) -> None:
        """Find all relevant files in the project."""
        logger.info(f"Finding files in {self.project_root}...")

        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_root)

                # Skip files in excluded directories
                if any(part in EXCLUDE_DIRS for part in Path(rel_path).parts):
                    continue

                # Categorize files
                if file in DOCKER_FILES:
                    self.docker_files.append(file_path)
                    self.files.append(file_path)
                elif file in K8S_FILES:
                    self.k8s_files.append(file_path)
                    self.files.append(file_path)
                else:
                    # Only include files with relevant extensions
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ALL_EXTENSIONS:
                        self.files.append(file_path)
                        if ext in CONFIG_EXTENSIONS:
                            self.config_files.append(file_path)

        logger.info(f"Found {len(self.files)} files to analyze")
        logger.info(f"Docker files: {len(self.docker_files)}")
        logger.info(f"Kubernetes files: {len(self.k8s_files)}")
        logger.info(f"Configuration files: {len(self.config_files)}")

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file for scalability mechanisms.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with analysis results
        """
        result = {
            'file_path': file_path,
            'load_testing': defaultdict(list),
            'horizontal_scaling': defaultdict(list),
            'resource_scaling': defaultdict(list),
            'performance': defaultdict(list)
        }

        # Skip large files (>1MB)
        if os.path.getsize(file_path) > 1024 * 1024:
            return result

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip files that are too large after reading
            if len(content) > 1024 * 1024:
                return result

            # Compile all patterns into a single regex for each category
            # to reduce the number of passes through the file

            # Process load testing patterns
            for category, patterns in LOAD_TESTING_PATTERNS.items():
                combined_pattern = '|'.join(patterns)
                for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                    result['load_testing'][category].append({
                        'match': match.group(0),
                        'file': os.path.basename(file_path)
                    })

            # Process horizontal scaling patterns
            for category, patterns in HORIZONTAL_SCALING_PATTERNS.items():
                combined_pattern = '|'.join(patterns)
                for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                    result['horizontal_scaling'][category].append({
                        'match': match.group(0),
                        'file': os.path.basename(file_path)
                    })

            # Process resource scaling patterns
            for category, patterns in RESOURCE_SCALING_PATTERNS.items():
                combined_pattern = '|'.join(patterns)
                for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                    result['resource_scaling'][category].append({
                        'match': match.group(0),
                        'file': os.path.basename(file_path)
                    })

            # Process performance patterns
            for category, patterns in PERFORMANCE_PATTERNS.items():
                combined_pattern = '|'.join(patterns)
                for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                    result['performance'][category].append({
                        'match': match.group(0),
                        'file': os.path.basename(file_path)
                    })

        except Exception as e:
            # Just skip problematic files
            pass

        return result

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the scalability aspects.

        Returns:
            Analysis results
        """
        logger.info("Starting scalability assessment analysis...")

        # Find all files
        self.find_files()

        # Take a sample of files if there are too many
        max_files = 500
        if len(self.files) > max_files:
            import random
            random.shuffle(self.files)
            self.files = self.files[:max_files]
            logger.info(f"Analyzing a sample of {max_files} files")

        # Analyze files using multiprocessing for better performance
        logger.info("Analyzing files...")

        # Determine the number of processes to use (leave some cores free)
        num_processes = max(1, multiprocessing.cpu_count() - 1)

        # Split files into chunks for each process
        chunk_size = max(1, len(self.files) // num_processes)
        file_chunks = [self.files[i:i + chunk_size] for i in range(0, len(self.files), chunk_size)]

        # Process files in parallel using ThreadPoolExecutor instead of ProcessPoolExecutor
        # to avoid serialization issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
            # Process files in parallel
            all_results = []

            # Process each file
            for file in self.files:
                try:
                    # Skip large files (>1MB)
                    if os.path.getsize(file) > 1024 * 1024:
                        continue

                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Skip files that are too large after reading
                    if len(content) > 1024 * 1024:
                        continue

                    file_result = {
                        'file': os.path.basename(file),
                        'load_testing': {},
                        'horizontal_scaling': {},
                        'resource_scaling': {},
                        'performance': {}
                    }

                    # Process load testing patterns
                    for category, patterns in LOAD_TESTING_PATTERNS.items():
                        matches = []
                        combined_pattern = '|'.join(patterns)
                        for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                            matches.append(match.group(0))
                        if matches:
                            file_result['load_testing'][category] = matches

                    # Process horizontal scaling patterns
                    for category, patterns in HORIZONTAL_SCALING_PATTERNS.items():
                        matches = []
                        combined_pattern = '|'.join(patterns)
                        for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                            matches.append(match.group(0))
                        if matches:
                            file_result['horizontal_scaling'][category] = matches

                    # Process resource scaling patterns
                    for category, patterns in RESOURCE_SCALING_PATTERNS.items():
                        matches = []
                        combined_pattern = '|'.join(patterns)
                        for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                            matches.append(match.group(0))
                        if matches:
                            file_result['resource_scaling'][category] = matches

                    # Process performance patterns
                    for category, patterns in PERFORMANCE_PATTERNS.items():
                        matches = []
                        combined_pattern = '|'.join(patterns)
                        for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                            matches.append(match.group(0))
                        if matches:
                            file_result['performance'][category] = matches

                    all_results.append(file_result)
                except Exception:
                    # Skip problematic files
                    pass

        # Merge results from all processes
        for result in all_results:
            # Merge load_testing findings
            for category, matches in result.get('load_testing', {}).items():
                for match in matches:
                    self.load_testing[category].append({
                        'match': match,
                        'file': result['file']
                    })

            # Merge horizontal_scaling findings
            for category, matches in result.get('horizontal_scaling', {}).items():
                for match in matches:
                    self.horizontal_scaling[category].append({
                        'match': match,
                        'file': result['file']
                    })

            # Merge resource_scaling findings
            for category, matches in result.get('resource_scaling', {}).items():
                for match in matches:
                    self.resource_scaling[category].append({
                        'match': match,
                        'file': result['file']
                    })

            # Merge performance findings
            for category, matches in result.get('performance', {}).items():
                for match in matches:
                    self.performance[category].append({
                        'match': match,
                        'file': result['file']
                    })

        # Calculate statistics
        load_testing_stats = {category: len(findings) for category, findings in self.load_testing.items()}
        horizontal_scaling_stats = {category: len(findings) for category, findings in self.horizontal_scaling.items()}
        resource_scaling_stats = {category: len(findings) for category, findings in self.resource_scaling.items()}
        performance_stats = {category: len(findings) for category, findings in self.performance.items()}

        # Assess scalability readiness
        scalability_readiness = {
            'containerization': 'high' if len(self.docker_files) > 0 else 'low',
            'orchestration': 'high' if len(self.k8s_files) > 0 else 'low',
            'auto_scaling': 'high' if len(self.resource_scaling['auto_scaling']) > 0 else 'low',
            'caching': 'high' if len(self.performance['caching']) > 0 else 'low',
            'async_processing': 'high' if len(self.performance['async_processing']) > 0 else 'low',
            'connection_pooling': 'high' if len(self.performance['connection_pooling']) > 0 else 'low',
            'database_optimization': 'high' if len(self.performance['database_optimization']) > 0 else 'low'
        }

        # Generate summary
        summary = {
            'load_testing': dict(self.load_testing),
            'horizontal_scaling': dict(self.horizontal_scaling),
            'resource_scaling': dict(self.resource_scaling),
            'performance': dict(self.performance),
            'stats': {
                'load_testing': load_testing_stats,
                'horizontal_scaling': horizontal_scaling_stats,
                'resource_scaling': resource_scaling_stats,
                'performance': performance_stats,
                'docker_files': len(self.docker_files),
                'k8s_files': len(self.k8s_files),
                'config_files': len(self.config_files),
                'total_files_analyzed': len(self.files)
            },
            'scalability_readiness': scalability_readiness
        }

        logger.info("Scalability assessment analysis complete")
        return summary

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze forex trading platform scalability")
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

    # Analyze scalability
    analyzer = ScalabilityAssessmentAnalyzer(Path(args.project_root))
    results = analyzer.analyze()

    # Save results
    output_path = os.path.join(args.output_dir, "scalability_assessment_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Scalability assessment analysis saved to {output_path}")

if __name__ == "__main__":
    main()
