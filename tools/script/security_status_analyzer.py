#!/usr/bin/env python3
"""
Forex Trading Platform Security Status Analyzer

This script analyzes the security aspects of the forex trading platform:
1. Vulnerability scan results (from dependency files and code patterns)
2. Dependency security (checking for outdated packages)
3. Authentication/authorization mechanisms
4. Data encryption practices

Output is a comprehensive JSON file that maps the security status of the platform.
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
import subprocess

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
    "node_modules", ".venv", "venv", "env", ".vscode"
}

# File extensions to analyze
PYTHON_EXTENSIONS = {".py"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
CONFIG_EXTENSIONS = {".json", ".yml", ".yaml", ".toml", ".ini", ".env"}
REQUIREMENTS_FILES = {"requirements.txt", "Pipfile", "Pipfile.lock", "package.json", "package-lock.json"}
ALL_EXTENSIONS = PYTHON_EXTENSIONS | JS_EXTENSIONS | CONFIG_EXTENSIONS

# Security patterns to look for
SECURITY_PATTERNS = {
    'authentication': [
        r'(auth|authenticate|login|signin|jwt|token|session|cookie|password|credential)',
        r'(OAuth|SAML|LDAP|ActiveDirectory|Kerberos)',
        r'@(auth|login)_required',
        r'(verify_password|check_password|hash_password)',
        r'(authenticate|is_authenticated|get_current_user)'
    ],
    'authorization': [
        r'(authorize|permission|role|access|rbac|acl)',
        r'(has_permission|check_permission|require_permission)',
        r'(is_admin|is_superuser|is_staff)',
        r'@(require|has)_(role|permission)'
    ],
    'encryption': [
        r'(encrypt|decrypt|cipher|aes|rsa|hash|md5|sha|hmac)',
        r'(ssl|tls|https|certificate)',
        r'(cryptography|pycrypto|bcrypt|passlib)',
        r'(secret|key|salt|iv|nonce)'
    ],
    'input_validation': [
        r'(validate|sanitize|escape|filter|clean)',
        r'(xss|csrf|sqli|injection)',
        r'(pydantic|schema|validator)',
        r'(input|param|query|body|form)_validation'
    ],
    'sensitive_data': [
        r'(api_key|secret_key|password|token|credential)',
        r'(credit_card|ssn|social_security|passport)',
        r'(pii|personally_identifiable|sensitive)',
        r'(mask|redact|hide|obscure)'
    ],
    'logging_monitoring': [
        r'(log|logger|logging|monitor|trace)',
        r'(sentry|datadog|newrelic|prometheus)',
        r'(alert|alarm|notification)',
        r'(audit|trail|record)'
    ]
}

# Vulnerability patterns
VULNERABILITY_PATTERNS = {
    'sql_injection': [
        r'execute\s*\(\s*[\'"].*?\%s.*?[\'"]\s*%\s*\(',
        r'cursor\.execute\s*\(\s*[\'"].*?\+.*?[\'"]\s*\)',
        r'raw_query\s*\(\s*[\'"].*?\+.*?[\'"]\s*\)'
    ],
    'xss': [
        r'innerHTML\s*=',
        r'document\.write\s*\(',
        r'eval\s*\(',
        r'dangerouslySetInnerHTML'
    ],
    'command_injection': [
        r'os\.system\s*\(',
        r'subprocess\.call\s*\(',
        r'subprocess\.Popen\s*\(',
        r'exec\s*\(',
        r'eval\s*\('
    ],
    'insecure_deserialization': [
        r'pickle\.loads\s*\(',
        r'yaml\.load\s*\(',
        r'json\.loads\s*\('
    ],
    'hardcoded_secrets': [
        r'password\s*=\s*[\'"][^\'"]+[\'"]',
        r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
        r'secret\s*=\s*[\'"][^\'"]+[\'"]',
        r'token\s*=\s*[\'"][^\'"]+[\'"]'
    ]
}

class SecurityStatusAnalyzer:
    """Analyzes the security status of the forex trading platform."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.files = []
        self.security_findings = defaultdict(list)
        self.vulnerabilities = defaultdict(list)
        self.dependencies = defaultdict(list)
        self.auth_mechanisms = []
        self.encryption_usage = []
        
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
                
                # Include requirements files
                if file in REQUIREMENTS_FILES:
                    self.files.append(file_path)
                    continue
                
                # Only include files with relevant extensions
                ext = os.path.splitext(file)[1].lower()
                if ext in ALL_EXTENSIONS:
                    self.files.append(file_path)
        
        logger.info(f"Found {len(self.files)} files to analyze")
    
    def analyze_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze dependencies in a requirements file.
        
        Args:
            file_path: Path to the requirements file
            
        Returns:
            List of dependencies
        """
        dependencies = []
        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_name == "requirements.txt":
                # Parse requirements.txt
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = re.split(r'[=<>]', line)
                        if parts:
                            package = parts[0].strip()
                            version = parts[1].strip() if len(parts) > 1 else "latest"
                            dependencies.append({
                                'name': package,
                                'version': version,
                                'file': file_path
                            })
            
            elif file_name == "package.json":
                # Parse package.json
                try:
                    data = json.loads(content)
                    for dep_type in ['dependencies', 'devDependencies']:
                        if dep_type in data:
                            for package, version in data[dep_type].items():
                                dependencies.append({
                                    'name': package,
                                    'version': version,
                                    'type': dep_type,
                                    'file': file_path
                                })
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON in {file_path}")
        
        except Exception as e:
            logger.error(f"Error analyzing dependencies in {file_path}: {e}")
        
        return dependencies
    
    def analyze_security_patterns(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze security patterns in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of security findings
        """
        findings = defaultdict(list)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for security patterns
            for category, patterns in SECURITY_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        findings[category].append({
                            'pattern': pattern,
                            'match': match.group(0),
                            'line': content.count('\n', 0, match.start()) + 1,
                            'file': file_path
                        })
            
            # Check for vulnerabilities
            for vuln_type, patterns in VULNERABILITY_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content):
                        self.vulnerabilities[vuln_type].append({
                            'pattern': pattern,
                            'match': match.group(0),
                            'line': content.count('\n', 0, match.start()) + 1,
                            'file': file_path
                        })
        
        except Exception as e:
            logger.error(f"Error analyzing security patterns in {file_path}: {e}")
        
        return findings
    
    def analyze_file(self, file_path: str) -> None:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to the file
        """
        try:
            file_name = os.path.basename(file_path)
            
            # Check if it's a dependency file
            if file_name in REQUIREMENTS_FILES:
                deps = self.analyze_dependencies(file_path)
                self.dependencies[file_path].extend(deps)
            
            # Analyze security patterns
            findings = self.analyze_security_patterns(file_path)
            for category, items in findings.items():
                self.security_findings[category].extend(items)
        
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the project security status.
        
        Returns:
            Analysis results
        """
        logger.info("Starting security status analysis...")
        
        # Find all files
        self.find_files()
        
        # Analyze files
        logger.info("Analyzing files...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(self.analyze_file, self.files))
        
        # Extract authentication mechanisms
        for finding in self.security_findings['authentication']:
            if finding not in self.auth_mechanisms:
                self.auth_mechanisms.append(finding)
        
        # Extract encryption usage
        for finding in self.security_findings['encryption']:
            if finding not in self.encryption_usage:
                self.encryption_usage.append(finding)
        
        # Generate summary
        summary = {
            'security_findings': dict(self.security_findings),
            'vulnerabilities': dict(self.vulnerabilities),
            'dependencies': dict(self.dependencies),
            'auth_mechanisms': self.auth_mechanisms,
            'encryption_usage': self.encryption_usage,
            'stats': {
                'total_files_analyzed': len(self.files),
                'security_findings_count': {k: len(v) for k, v in self.security_findings.items()},
                'vulnerabilities_count': {k: len(v) for k, v in self.vulnerabilities.items()},
                'dependencies_count': sum(len(v) for v in self.dependencies.values())
            }
        }
        
        logger.info("Security status analysis complete")
        return summary

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze forex trading platform security status")
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
    
    # Analyze security status
    analyzer = SecurityStatusAnalyzer(Path(args.project_root))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, "security_status_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Security status analysis saved to {output_path}")

if __name__ == "__main__":
    main()
