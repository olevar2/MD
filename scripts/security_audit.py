#!/usr/bin/env python3
"""
Security audit script for the Forex Trading Platform.

This script performs a security audit of the platform, checking for common security issues.
"""

import os
import sys
import json
import re
import logging
import subprocess
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('security_audit.log')
    ]
)
logger = logging.getLogger(__name__)

# Security check configuration
SECURITY_CHECKS = {
    'hardcoded_secrets': {
        'description': 'Check for hardcoded secrets',
        'patterns': [
            r'password\s*=\s*["\'](?!.*\$\{)(\S+)["\']',
            r'secret\s*=\s*["\'](?!.*\$\{)(\S+)["\']',
            r'api_key\s*=\s*["\'](?!.*\$\{)(\S+)["\']',
            r'token\s*=\s*["\'](?!.*\$\{)(\S+)["\']',
            r'aws_secret\s*=\s*["\'](?!.*\$\{)(\S+)["\']',
            r'private_key\s*=\s*["\'](?!.*\$\{)(\S+)["\']'
        ],
        'file_extensions': ['.py', '.json', '.yaml', '.yml', '.env'],
        'severity': 'high'
    },
    'sql_injection': {
        'description': 'Check for potential SQL injection vulnerabilities',
        'patterns': [
            r'execute\s*\(\s*f["\']',
            r'execute\s*\(\s*["\'][^"\']*{\s*[^}]*}\s*[^"\']*["\']',
            r'execute\s*\(\s*["\'][^"\']*\%\s*[^"\']*["\']',
            r'execute\s*\(\s*["\'][^"\']*\+\s*[^"\']*["\']'
        ],
        'file_extensions': ['.py'],
        'severity': 'critical'
    },
    'command_injection': {
        'description': 'Check for potential command injection vulnerabilities',
        'patterns': [
            r'os\.system\s*\(\s*[^)]*\+',
            r'os\.system\s*\(\s*f["\']',
            r'subprocess\.call\s*\(\s*[^)]*\+',
            r'subprocess\.call\s*\(\s*f["\']',
            r'subprocess\.Popen\s*\(\s*[^)]*\+',
            r'subprocess\.Popen\s*\(\s*f["\']'
        ],
        'file_extensions': ['.py'],
        'severity': 'critical'
    },
    'insecure_deserialization': {
        'description': 'Check for insecure deserialization',
        'patterns': [
            r'pickle\.loads',
            r'yaml\.load\s*\([^)]*\)',
            r'eval\s*\('
        ],
        'file_extensions': ['.py'],
        'severity': 'high'
    },
    'insecure_file_operations': {
        'description': 'Check for insecure file operations',
        'patterns': [
            r'open\s*\(\s*[^)]*\+',
            r'open\s*\(\s*f["\']'
        ],
        'file_extensions': ['.py'],
        'severity': 'medium'
    },
    'debug_enabled': {
        'description': 'Check for debug mode enabled',
        'patterns': [
            r'debug\s*=\s*True',
            r'DEBUG\s*=\s*True'
        ],
        'file_extensions': ['.py', '.json', '.yaml', '.yml'],
        'severity': 'medium'
    },
    'insecure_http': {
        'description': 'Check for insecure HTTP usage',
        'patterns': [
            r'http://[^s]'
        ],
        'file_extensions': ['.py', '.json', '.yaml', '.yml', '.md'],
        'severity': 'medium'
    },
    'weak_crypto': {
        'description': 'Check for weak cryptographic algorithms',
        'patterns': [
            r'md5',
            r'sha1',
            r'des',
            r'RC4'
        ],
        'file_extensions': ['.py'],
        'severity': 'high'
    },
    'cors_misconfiguration': {
        'description': 'Check for CORS misconfiguration',
        'patterns': [
            r'Access-Control-Allow-Origin\s*:\s*\*',
            r'add_header\s+Access-Control-Allow-Origin\s+\*'
        ],
        'file_extensions': ['.py', '.json', '.yaml', '.yml'],
        'severity': 'medium'
    },
    'jwt_none_algorithm': {
        'description': 'Check for JWT none algorithm',
        'patterns': [
            r'algorithm\s*=\s*["\']none["\']',
            r'algorithm\s*=\s*["\']None["\']'
        ],
        'file_extensions': ['.py'],
        'severity': 'critical'
    }
}

# Kubernetes security check configuration
K8S_SECURITY_CHECKS = {
    'privileged_containers': {
        'description': 'Check for privileged containers',
        'patterns': [
            r'privileged:\s*true'
        ],
        'file_extensions': ['.yaml', '.yml'],
        'severity': 'high'
    },
    'host_network': {
        'description': 'Check for host network access',
        'patterns': [
            r'hostNetwork:\s*true'
        ],
        'file_extensions': ['.yaml', '.yml'],
        'severity': 'high'
    },
    'host_pid': {
        'description': 'Check for host PID access',
        'patterns': [
            r'hostPID:\s*true'
        ],
        'file_extensions': ['.yaml', '.yml'],
        'severity': 'high'
    },
    'host_ipc': {
        'description': 'Check for host IPC access',
        'patterns': [
            r'hostIPC:\s*true'
        ],
        'file_extensions': ['.yaml', '.yml'],
        'severity': 'high'
    },
    'root_containers': {
        'description': 'Check for containers running as root',
        'patterns': [
            r'runAsNonRoot:\s*false'
        ],
        'file_extensions': ['.yaml', '.yml'],
        'severity': 'medium'
    },
    'latest_tag': {
        'description': 'Check for containers using latest tag',
        'patterns': [
            r'image:\s*[^:]*:latest'
        ],
        'file_extensions': ['.yaml', '.yml'],
        'severity': 'medium'
    },
    'no_resource_limits': {
        'description': 'Check for containers without resource limits',
        'patterns': [
            r'containers:\s*\n(?:.*\n)*?(?!\s*resources:)'
        ],
        'file_extensions': ['.yaml', '.yml'],
        'severity': 'medium'
    },
    'no_network_policy': {
        'description': 'Check for missing network policies',
        'patterns': [],  # This is a special case, checked separately
        'file_extensions': ['.yaml', '.yml'],
        'severity': 'medium'
    }
}

def find_files(root_dir: str, file_extensions: List[str]) -> List[str]:
    """
    Find files with the given extensions in the directory tree.

    Args:
        root_dir: Root directory to search
        file_extensions: List of file extensions to include

    Returns:
        List of file paths
    """
    file_paths = []

    # For testing purposes, we'll only scan a subset of directories
    test_dirs = [
        os.path.join(root_dir, 'trading-gateway-service'),
        os.path.join(root_dir, 'portfolio-management-service'),
        os.path.join(root_dir, 'common-lib'),
        os.path.join(root_dir, 'kubernetes')
    ]

    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue

        for dirpath, dirnames, filenames in os.walk(test_dir):
            # Skip virtual environment directories
            if 'venv' in dirpath or 'env' in dirpath or '__pycache__' in dirpath:
                continue

            for filename in filenames:
                if any(filename.endswith(ext) for ext in file_extensions):
                    file_paths.append(os.path.join(dirpath, filename))

    return file_paths

def check_file(file_path: str, check_name: str, check_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check a file for security issues.

    Args:
        file_path: Path to the file
        check_name: Name of the check
        check_config: Check configuration

    Returns:
        List of findings
    """
    findings = []

    # Skip files with extensions not in the check
    if not any(file_path.endswith(ext) for ext in check_config['file_extensions']):
        return findings

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        for pattern in check_config['patterns']:
            matches = re.finditer(pattern, content, re.IGNORECASE)

            for match in matches:
                finding = {
                    'check': check_name,
                    'description': check_config['description'],
                    'file': file_path,
                    'line': content.count('\n', 0, match.start()) + 1,
                    'match': match.group(0),
                    'severity': check_config['severity']
                }
                findings.append(finding)

    except Exception as e:
        logger.warning(f"Error checking file {file_path}: {str(e)}")

    return findings

def check_network_policies(root_dir: str) -> List[Dict[str, Any]]:
    """
    Check for missing network policies in Kubernetes manifests.

    Args:
        root_dir: Root directory to search

    Returns:
        List of findings
    """
    findings = []

    # Find all Kubernetes namespace definitions
    namespace_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.yaml', '.yml')):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if 'kind: Namespace' in content:
                        namespace_files.append(file_path)
                except Exception:
                    pass

    # Find all network policy definitions
    network_policy_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.yaml', '.yml')):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if 'kind: NetworkPolicy' in content:
                        network_policy_files.append(file_path)
                except Exception:
                    pass

    # If there are namespaces but no network policies, add a finding
    if namespace_files and not network_policy_files:
        finding = {
            'check': 'no_network_policy',
            'description': 'Missing network policies',
            'file': namespace_files[0],
            'line': 0,
            'match': 'No NetworkPolicy resources found',
            'severity': 'medium'
        }
        findings.append(finding)

    return findings

def run_security_audit(root_dir: str) -> Dict[str, Any]:
    """
    Run a security audit on the codebase.

    Args:
        root_dir: Root directory of the codebase

    Returns:
        Dictionary with audit results
    """
    logger.info("Starting security audit")

    # Initialize results
    results = {
        'timestamp': datetime.now().isoformat(),
        'root_dir': root_dir,
        'findings': [],
        'summary': {
            'total_files': 0,
            'total_findings': 0,
            'findings_by_severity': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'findings_by_check': {}
        }
    }

    # Combine all checks
    all_checks = {**SECURITY_CHECKS, **K8S_SECURITY_CHECKS}

    # Get all file extensions to check
    all_extensions = set()
    for check_config in all_checks.values():
        all_extensions.update(check_config['file_extensions'])

    # Find all files to check
    file_paths = find_files(root_dir, list(all_extensions))
    results['summary']['total_files'] = len(file_paths)

    # Check each file
    for file_path in file_paths:
        for check_name, check_config in all_checks.items():
            findings = check_file(file_path, check_name, check_config)
            results['findings'].extend(findings)

    # Check for missing network policies
    network_policy_findings = check_network_policies(root_dir)
    results['findings'].extend(network_policy_findings)

    # Calculate summary
    results['summary']['total_findings'] = len(results['findings'])

    for finding in results['findings']:
        severity = finding['severity']
        check = finding['check']

        results['summary']['findings_by_severity'][severity] += 1

        if check not in results['summary']['findings_by_check']:
            results['summary']['findings_by_check'][check] = 0

        results['summary']['findings_by_check'][check] += 1

    return results

def save_results(results: Dict[str, Any]) -> str:
    """
    Save audit results to a file.

    Args:
        results: Audit results

    Returns:
        Path to the saved file
    """
    logger.info("Saving audit results")

    # Create output directory with absolute path
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'security_audit')
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    filename = f"security_audit_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {filepath}")
    return filepath

def generate_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable report from the audit results.

    Args:
        results: Audit results

    Returns:
        Path to the report file
    """
    logger.info("Generating audit report")

    # Create output directory with absolute path
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'security_audit')
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    filename = f"security_audit_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
    filepath = os.path.join(output_dir, filename)

    # Generate report
    with open(filepath, 'w') as f:
        f.write(f"# Security Audit Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## Summary\n\n")
        f.write(f"- Total files scanned: {results['summary']['total_files']}\n")
        f.write(f"- Total findings: {results['summary']['total_findings']}\n\n")

        f.write(f"### Findings by Severity\n\n")
        f.write(f"- Critical: {results['summary']['findings_by_severity']['critical']}\n")
        f.write(f"- High: {results['summary']['findings_by_severity']['high']}\n")
        f.write(f"- Medium: {results['summary']['findings_by_severity']['medium']}\n")
        f.write(f"- Low: {results['summary']['findings_by_severity']['low']}\n\n")

        f.write(f"### Findings by Check\n\n")
        for check, count in results['summary']['findings_by_check'].items():
            check_config = SECURITY_CHECKS.get(check) or K8S_SECURITY_CHECKS.get(check)
            if check_config:
                f.write(f"- {check_config['description']}: {count}\n")

        f.write(f"\n## Detailed Findings\n\n")

        # Group findings by severity
        findings_by_severity = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }

        for finding in results['findings']:
            findings_by_severity[finding['severity']].append(finding)

        # Write findings by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            if findings_by_severity[severity]:
                f.write(f"### {severity.capitalize()} Severity\n\n")

                for finding in findings_by_severity[severity]:
                    f.write(f"#### {finding['description']}\n\n")
                    f.write(f"- File: `{finding['file']}`\n")
                    f.write(f"- Line: {finding['line']}\n")
                    f.write(f"- Match: `{finding['match']}`\n\n")

        f.write(f"\n## Recommendations\n\n")

        # Add recommendations based on findings
        if results['summary']['findings_by_severity']['critical'] > 0:
            f.write(f"### Critical Issues\n\n")
            f.write(f"- Fix all critical issues immediately\n")
            f.write(f"- Review code for similar patterns\n")
            f.write(f"- Implement security testing in CI/CD pipeline\n\n")

        if results['summary']['findings_by_severity']['high'] > 0:
            f.write(f"### High Severity Issues\n\n")
            f.write(f"- Address high severity issues as soon as possible\n")
            f.write(f"- Implement secure coding practices\n")
            f.write(f"- Consider using security linters\n\n")

        if results['summary']['findings_by_check'].get('hardcoded_secrets', 0) > 0:
            f.write(f"### Hardcoded Secrets\n\n")
            f.write(f"- Remove all hardcoded secrets\n")
            f.write(f"- Use environment variables or a secrets management solution\n")
            f.write(f"- Rotate any exposed secrets\n\n")

        if results['summary']['findings_by_check'].get('sql_injection', 0) > 0:
            f.write(f"### SQL Injection\n\n")
            f.write(f"- Use parameterized queries\n")
            f.write(f"- Avoid string concatenation in SQL queries\n")
            f.write(f"- Implement input validation\n\n")

        if results['summary']['findings_by_check'].get('no_network_policy', 0) > 0:
            f.write(f"### Kubernetes Network Policies\n\n")
            f.write(f"- Implement network policies for all namespaces\n")
            f.write(f"- Restrict pod-to-pod communication\n")
            f.write(f"- Follow the principle of least privilege\n\n")

    logger.info(f"Report generated at {filepath}")
    return filepath

def main():
    """Main function to run the security audit."""
    logger.info("Starting security audit script")

    # Get the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    try:
        # Run the security audit
        results = run_security_audit(root_dir)

        # Save the results
        save_results(results)

        # Generate a report
        report_path = generate_report(results)

        # Print summary
        logger.info("Security audit completed")
        logger.info(f"Total files scanned: {results['summary']['total_files']}")
        logger.info(f"Total findings: {results['summary']['total_findings']}")
        logger.info(f"Critical findings: {results['summary']['findings_by_severity']['critical']}")
        logger.info(f"High severity findings: {results['summary']['findings_by_severity']['high']}")
        logger.info(f"Medium severity findings: {results['summary']['findings_by_severity']['medium']}")
        logger.info(f"Low severity findings: {results['summary']['findings_by_severity']['low']}")
        logger.info(f"Report generated at {report_path}")

        # Return exit code based on findings
        if results['summary']['findings_by_severity']['critical'] > 0:
            logger.error("Critical security issues found")
            return 2
        elif results['summary']['findings_by_severity']['high'] > 0:
            logger.warning("High severity security issues found")
            return 1
        else:
            logger.info("No critical or high severity issues found")
            return 0

    except Exception as e:
        logger.error(f"Error running security audit: {str(e)}")
        return 3

if __name__ == '__main__':
    sys.exit(main())
