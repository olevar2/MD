#!/usr/bin/env python3
"""
Test Coverage Analyzer

This script analyzes test coverage of the forex trading platform:
1. Unit test coverage percentage
2. Integration test coverage
3. End-to-end test coverage
4. Test success/failure rates

Output is a JSON file with comprehensive test coverage metrics.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"

class TestCoverageAnalyzer:
    """Analyzes test coverage of the forex trading platform."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.unit_tests = {}
        self.integration_tests = {}
        self.e2e_tests = {}
        self.test_results = {}
        
    def find_test_files(self) -> Dict[str, List[str]]:
        """
        Find all test files in the project.
        
        Returns:
            Dictionary with test files categorized by type
        """
        logger.info(f"Finding test files in {self.project_root}...")
        
        test_files = {
            'unit': [],
            'integration': [],
            'e2e': []
        }
        
        # Walk through the project directory
        for root, _, files in os.walk(self.project_root):
            for file in files:
                # Skip non-Python files
                if not file.endswith('.py'):
                    continue
                
                # Skip files that don't look like tests
                if not (file.startswith('test_') or file.endswith('_test.py') or 'test' in file.lower()):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_root)
                
                # Categorize the test file
                if 'unit' in file.lower() or 'unit' in rel_path.lower():
                    test_files['unit'].append(file_path)
                elif 'integration' in file.lower() or 'integration' in rel_path.lower():
                    test_files['integration'].append(file_path)
                elif 'e2e' in file.lower() or 'end_to_end' in rel_path.lower() or 'functional' in file.lower():
                    test_files['e2e'].append(file_path)
                else:
                    # Default to unit test if not clearly categorized
                    test_files['unit'].append(file_path)
        
        logger.info(f"Found {len(test_files['unit'])} unit tests, {len(test_files['integration'])} integration tests, and {len(test_files['e2e'])} end-to-end tests")
        return test_files
    
    def analyze_coverage_reports(self) -> Dict[str, Any]:
        """
        Analyze coverage reports if they exist.
        
        Returns:
            Dictionary with coverage metrics
        """
        logger.info("Analyzing coverage reports...")
        
        coverage_results = {
            'unit': None,
            'integration': None,
            'e2e': None,
            'overall': None
        }
        
        # Look for coverage XML reports
        coverage_files = []
        for pattern in ['coverage.xml', '*coverage*.xml', 'cov.xml', 'coverage-*.xml']:
            coverage_files.extend(glob.glob(os.path.join(self.project_root, '**', pattern), recursive=True))
        
        for coverage_file in coverage_files:
            try:
                # Parse the coverage XML
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                
                # Extract coverage metrics
                if root.tag == 'coverage':
                    line_rate = float(root.get('line-rate', 0))
                    branch_rate = float(root.get('branch-rate', 0))
                    
                    # Determine coverage type based on filename or content
                    coverage_type = 'overall'
                    if 'unit' in coverage_file.lower():
                        coverage_type = 'unit'
                    elif 'integration' in coverage_file.lower():
                        coverage_type = 'integration'
                    elif 'e2e' in coverage_file.lower() or 'end_to_end' in coverage_file.lower():
                        coverage_type = 'e2e'
                    
                    # Extract package/module coverage
                    packages = {}
                    for package in root.findall('.//package'):
                        package_name = package.get('name', 'unknown')
                        package_line_rate = float(package.get('line-rate', 0))
                        package_branch_rate = float(package.get('branch-rate', 0))
                        
                        packages[package_name] = {
                            'line_rate': package_line_rate,
                            'branch_rate': package_branch_rate,
                            'line_percent': round(package_line_rate * 100, 2),
                            'branch_percent': round(package_branch_rate * 100, 2)
                        }
                    
                    coverage_results[coverage_type] = {
                        'line_rate': line_rate,
                        'branch_rate': branch_rate,
                        'line_percent': round(line_rate * 100, 2),
                        'branch_percent': round(branch_rate * 100, 2),
                        'packages': packages,
                        'source': os.path.basename(coverage_file)
                    }
            except Exception as e:
                logger.error(f"Error parsing coverage file {coverage_file}: {e}")
        
        return coverage_results
    
    def analyze_test_results(self) -> Dict[str, Any]:
        """
        Analyze test results if they exist.
        
        Returns:
            Dictionary with test result metrics
        """
        logger.info("Analyzing test results...")
        
        test_results = {
            'unit': None,
            'integration': None,
            'e2e': None,
            'overall': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'error': 0,
                'success_rate': 0
            }
        }
        
        # Look for JUnit XML reports
        result_files = []
        for pattern in ['junit.xml', '*junit*.xml', 'test-results.xml', 'test-results-*.xml', 'pytest-*.xml']:
            result_files.extend(glob.glob(os.path.join(self.project_root, '**', pattern), recursive=True))
        
        for result_file in result_files:
            try:
                # Parse the JUnit XML
                tree = ET.parse(result_file)
                root = tree.getroot()
                
                # Extract test results
                if root.tag in ['testsuites', 'testsuite']:
                    # Determine test type based on filename or content
                    test_type = 'overall'
                    if 'unit' in result_file.lower():
                        test_type = 'unit'
                    elif 'integration' in result_file.lower():
                        test_type = 'integration'
                    elif 'e2e' in result_file.lower() or 'end_to_end' in result_file.lower():
                        test_type = 'e2e'
                    
                    # Initialize if not already
                    if test_results[test_type] is None:
                        test_results[test_type] = {
                            'total': 0,
                            'passed': 0,
                            'failed': 0,
                            'skipped': 0,
                            'error': 0,
                            'success_rate': 0,
                            'test_suites': {}
                        }
                    
                    # Process test suites
                    suites = [root] if root.tag == 'testsuite' else root.findall('./testsuite')
                    for suite in suites:
                        suite_name = suite.get('name', 'unknown')
                        suite_tests = int(suite.get('tests', 0))
                        suite_failures = int(suite.get('failures', 0))
                        suite_errors = int(suite.get('errors', 0))
                        suite_skipped = int(suite.get('skipped', 0))
                        suite_passed = suite_tests - suite_failures - suite_errors - suite_skipped
                        
                        # Add to test type totals
                        test_results[test_type]['total'] += suite_tests
                        test_results[test_type]['passed'] += suite_passed
                        test_results[test_type]['failed'] += suite_failures
                        test_results[test_type]['error'] += suite_errors
                        test_results[test_type]['skipped'] += suite_skipped
                        
                        # Add to overall totals
                        test_results['overall']['total'] += suite_tests
                        test_results['overall']['passed'] += suite_passed
                        test_results['overall']['failed'] += suite_failures
                        test_results['overall']['error'] += suite_errors
                        test_results['overall']['skipped'] += suite_skipped
                        
                        # Store suite details
                        test_results[test_type]['test_suites'][suite_name] = {
                            'total': suite_tests,
                            'passed': suite_passed,
                            'failed': suite_failures,
                            'error': suite_errors,
                            'skipped': suite_skipped,
                            'success_rate': round(suite_passed / suite_tests * 100, 2) if suite_tests > 0 else 0
                        }
                    
                    # Calculate success rates
                    if test_results[test_type]['total'] > 0:
                        test_results[test_type]['success_rate'] = round(
                            test_results[test_type]['passed'] / test_results[test_type]['total'] * 100, 2
                        )
            except Exception as e:
                logger.error(f"Error parsing test result file {result_file}: {e}")
        
        # Calculate overall success rate
        if test_results['overall']['total'] > 0:
            test_results['overall']['success_rate'] = round(
                test_results['overall']['passed'] / test_results['overall']['total'] * 100, 2
            )
        
        return test_results
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze test coverage.
        
        Returns:
            Dictionary with test coverage metrics
        """
        # Find test files
        test_files = self.find_test_files()
        
        # Analyze coverage reports
        coverage_results = self.analyze_coverage_reports()
        
        # Analyze test results
        test_results = self.analyze_test_results()
        
        # Generate summary
        summary = {
            'unit_tests': {
                'count': len(test_files['unit']),
                'coverage': coverage_results['unit']['line_percent'] if coverage_results['unit'] else None,
                'success_rate': test_results['unit']['success_rate'] if test_results['unit'] else None
            },
            'integration_tests': {
                'count': len(test_files['integration']),
                'coverage': coverage_results['integration']['line_percent'] if coverage_results['integration'] else None,
                'success_rate': test_results['integration']['success_rate'] if test_results['integration'] else None
            },
            'e2e_tests': {
                'count': len(test_files['e2e']),
                'coverage': coverage_results['e2e']['line_percent'] if coverage_results['e2e'] else None,
                'success_rate': test_results['e2e']['success_rate'] if test_results['e2e'] else None
            },
            'overall': {
                'total_tests': len(test_files['unit']) + len(test_files['integration']) + len(test_files['e2e']),
                'coverage': coverage_results['overall']['line_percent'] if coverage_results['overall'] else None,
                'success_rate': test_results['overall']['success_rate']
            }
        }
        
        # Combine results
        return {
            'test_files': {
                'unit': [os.path.relpath(path, self.project_root) for path in test_files['unit']],
                'integration': [os.path.relpath(path, self.project_root) for path in test_files['integration']],
                'e2e': [os.path.relpath(path, self.project_root) for path in test_files['e2e']]
            },
            'coverage': coverage_results,
            'test_results': test_results,
            'summary': summary
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze test coverage")
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
    
    # Analyze test coverage
    analyzer = TestCoverageAnalyzer(Path(args.project_root))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, "test_coverage.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test coverage metrics saved to {output_path}")

if __name__ == "__main__":
    main()
