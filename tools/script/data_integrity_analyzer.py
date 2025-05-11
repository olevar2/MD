#!/usr/bin/env python3
"""
Forex Trading Platform Data Integrity Analyzer

This script analyzes the data integrity mechanisms in the forex trading platform:
1. Database consistency checks (foreign key constraints, unique constraints)
2. Data validation mechanisms (input validation, schema validation)
3. Backup procedures (backup scripts, configuration)
4. Data reconciliation processes (cross-system validation, audit trails)

Output is a comprehensive JSON file that maps the data integrity status of the platform.
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
import ast

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
SQL_EXTENSIONS = {".sql"}
CONFIG_EXTENSIONS = {".json", ".yml", ".yaml", ".toml", ".ini", ".env"}
ALL_EXTENSIONS = PYTHON_EXTENSIONS | JS_EXTENSIONS | SQL_EXTENSIONS | CONFIG_EXTENSIONS

# Patterns for detecting data integrity mechanisms
DB_CONSISTENCY_PATTERNS = {
    'foreign_key': [
        r'ForeignKey\s*\(',
        r'FOREIGN\s+KEY',
        r'REFERENCES',
        r'relationship\s*\(',
        r'backref\s*\(',
        r'CASCADE',
        r'ON\s+DELETE',
        r'ON\s+UPDATE'
    ],
    'unique_constraint': [
        r'UNIQUE',
        r'UniqueConstraint',
        r'unique=True',
        r'PRIMARY\s+KEY',
        r'Index\s*\('
    ],
    'check_constraint': [
        r'CHECK\s*\(',
        r'CheckConstraint',
        r'validate\s*\(',
        r'validates\s*\('
    ],
    'transaction': [
        r'BEGIN\s+TRANSACTION',
        r'COMMIT',
        r'ROLLBACK',
        r'session\.commit\(\)',
        r'session\.rollback\(\)',
        r'transaction\.atomic',
        r'with\s+transaction'
    ]
}

DATA_VALIDATION_PATTERNS = {
    'input_validation': [
        r'validate\s*\(',
        r'validator\s*\(',
        r'schema\s*\(',
        r'Schema\s*\(',
        r'pydantic',
        r'BaseModel',
        r'Field\s*\(',
        r'@validator',
        r'clean\s*\(',
        r'sanitize\s*\('
    ],
    'type_checking': [
        r'isinstance\s*\(',
        r'type\s*\(',
        r'TypeVar',
        r'Union\s*\[',
        r'Optional\s*\[',
        r'List\s*\[',
        r'Dict\s*\[',
        r'Tuple\s*\[',
        r'Any',
        r'typing\.'
    ],
    'error_handling': [
        r'try\s*:',
        r'except\s+',
        r'raise\s+',
        r'ValidationError',
        r'DataError',
        r'IntegrityError',
        r'assert\s+'
    ]
}

BACKUP_PATTERNS = {
    'backup_procedure': [
        r'backup',
        r'dump',
        r'export',
        r'archive',
        r'snapshot',
        r'pg_dump',
        r'mysqldump',
        r'mongodump'
    ],
    'schedule': [
        r'cron',
        r'schedule',
        r'periodic',
        r'interval',
        r'daily',
        r'weekly',
        r'monthly'
    ],
    'storage': [
        r's3',
        r'blob',
        r'storage',
        r'bucket',
        r'cloud',
        r'backup_dir',
        r'archive_path'
    ]
}

RECONCILIATION_PATTERNS = {
    'data_comparison': [
        r'reconcile',
        r'compare',
        r'diff',
        r'match',
        r'verify',
        r'validate',
        r'consistency_check'
    ],
    'audit': [
        r'audit',
        r'log',
        r'trace',
        r'history',
        r'journal',
        r'record',
        r'track'
    ],
    'reporting': [
        r'report',
        r'alert',
        r'notify',
        r'summary',
        r'dashboard',
        r'monitor'
    ]
}

class DataIntegrityAnalyzer:
    """Analyzes the data integrity mechanisms of the forex trading platform."""

    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.files = []
        self.db_consistency = defaultdict(list)
        self.data_validation = defaultdict(list)
        self.backup_procedures = defaultdict(list)
        self.reconciliation_processes = defaultdict(list)

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

                # Only include files with relevant extensions
                ext = os.path.splitext(file)[1].lower()
                if ext in ALL_EXTENSIONS:
                    self.files.append(file_path)

        logger.info(f"Found {len(self.files)} files to analyze")

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file for data integrity mechanisms.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with analysis results
        """
        result = {
            'file_path': file_path,
            'db_consistency': defaultdict(list),
            'data_validation': defaultdict(list),
            'backup_procedures': defaultdict(list),
            'reconciliation_processes': defaultdict(list)
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

            # Process database consistency patterns
            for category, patterns in DB_CONSISTENCY_PATTERNS.items():
                combined_pattern = '|'.join(patterns)
                for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                    result['db_consistency'][category].append({
                        'match': match.group(0),
                        'file': os.path.basename(file_path)
                    })

            # Process data validation patterns
            for category, patterns in DATA_VALIDATION_PATTERNS.items():
                combined_pattern = '|'.join(patterns)
                for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                    result['data_validation'][category].append({
                        'match': match.group(0),
                        'file': os.path.basename(file_path)
                    })

            # Process backup patterns
            for category, patterns in BACKUP_PATTERNS.items():
                combined_pattern = '|'.join(patterns)
                for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                    result['backup_procedures'][category].append({
                        'match': match.group(0),
                        'file': os.path.basename(file_path)
                    })

            # Process reconciliation patterns
            for category, patterns in RECONCILIATION_PATTERNS.items():
                combined_pattern = '|'.join(patterns)
                for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                    result['reconciliation_processes'][category].append({
                        'match': match.group(0),
                        'file': os.path.basename(file_path)
                    })

        except Exception as e:
            # Just skip problematic files
            pass

        return result

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the data integrity mechanisms.

        Returns:
            Analysis results
        """
        logger.info("Starting data integrity analysis...")

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
                        'db_consistency': {},
                        'data_validation': {},
                        'backup_procedures': {},
                        'reconciliation_processes': {}
                    }

                    # Process database consistency patterns
                    for category, patterns in DB_CONSISTENCY_PATTERNS.items():
                        matches = []
                        combined_pattern = '|'.join(patterns)
                        for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                            matches.append(match.group(0))
                        if matches:
                            file_result['db_consistency'][category] = matches

                    # Process data validation patterns
                    for category, patterns in DATA_VALIDATION_PATTERNS.items():
                        matches = []
                        combined_pattern = '|'.join(patterns)
                        for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                            matches.append(match.group(0))
                        if matches:
                            file_result['data_validation'][category] = matches

                    # Process backup patterns
                    for category, patterns in BACKUP_PATTERNS.items():
                        matches = []
                        combined_pattern = '|'.join(patterns)
                        for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                            matches.append(match.group(0))
                        if matches:
                            file_result['backup_procedures'][category] = matches

                    # Process reconciliation patterns
                    for category, patterns in RECONCILIATION_PATTERNS.items():
                        matches = []
                        combined_pattern = '|'.join(patterns)
                        for match in re.finditer(combined_pattern, content, re.IGNORECASE):
                            matches.append(match.group(0))
                        if matches:
                            file_result['reconciliation_processes'][category] = matches

                    all_results.append(file_result)
                except Exception:
                    # Skip problematic files
                    pass

            # No need to collect results as we're processing files directly

        # Merge results from all processes
        for result in all_results:
            # Merge db_consistency findings
            for category, matches in result.get('db_consistency', {}).items():
                for match in matches:
                    self.db_consistency[category].append({
                        'match': match,
                        'file': result['file']
                    })

            # Merge data_validation findings
            for category, matches in result.get('data_validation', {}).items():
                for match in matches:
                    self.data_validation[category].append({
                        'match': match,
                        'file': result['file']
                    })

            # Merge backup_procedures findings
            for category, matches in result.get('backup_procedures', {}).items():
                for match in matches:
                    self.backup_procedures[category].append({
                        'match': match,
                        'file': result['file']
                    })

            # Merge reconciliation_processes findings
            for category, matches in result.get('reconciliation_processes', {}).items():
                for match in matches:
                    self.reconciliation_processes[category].append({
                        'match': match,
                        'file': result['file']
                    })

        # Calculate statistics
        db_consistency_stats = {category: len(findings) for category, findings in self.db_consistency.items()}
        data_validation_stats = {category: len(findings) for category, findings in self.data_validation.items()}
        backup_stats = {category: len(findings) for category, findings in self.backup_procedures.items()}
        reconciliation_stats = {category: len(findings) for category, findings in self.reconciliation_processes.items()}

        # Generate summary
        summary = {
            'db_consistency': dict(self.db_consistency),
            'data_validation': dict(self.data_validation),
            'backup_procedures': dict(self.backup_procedures),
            'reconciliation_processes': dict(self.reconciliation_processes),
            'stats': {
                'db_consistency': db_consistency_stats,
                'data_validation': data_validation_stats,
                'backup_procedures': backup_stats,
                'reconciliation_processes': reconciliation_stats,
                'total_files_analyzed': len(self.files)
            }
        }

        logger.info("Data integrity analysis complete")
        return summary

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze forex trading platform data integrity")
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

    # Analyze data integrity
    analyzer = DataIntegrityAnalyzer(Path(args.project_root))
    results = analyzer.analyze()

    # Save results
    output_path = os.path.join(args.output_dir, "data_integrity_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Data integrity analysis saved to {output_path}")

if __name__ == "__main__":
    main()
