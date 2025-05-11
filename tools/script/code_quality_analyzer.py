#!/usr/bin/env python3
"""
Code Quality Analyzer

This script analyzes code quality metrics of the forex trading platform:
1. Static analysis results (linting)
2. Cyclomatic complexity
3. Technical debt measurements
4. Duplicate code percentage

Output is a JSON file with comprehensive code quality metrics.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures
import ast
import tokenize
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"

# File extensions to analyze
PYTHON_EXTENSIONS = {".py"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
ALL_EXTENSIONS = PYTHON_EXTENSIONS | JS_EXTENSIONS

class CodeQualityAnalyzer:
    """Analyzes code quality metrics of the forex trading platform."""

    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.files = []
        self.linting_results = {}
        self.complexity_results = {}
        self.tech_debt_results = {}
        self.duplication_results = {}

    def find_files(self) -> None:
        """Find all relevant files in the project."""
        logger.info(f"Finding files in {self.project_root}...")

        # Directories to exclude
        exclude_dirs = {
            ".git", ".github", ".pytest_cache", "__pycache__",
            "node_modules", ".venv", "venv", "env", ".vscode",
            "corrupted_backups", "tools/output", "tests", "testing"
        }

        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_root)

                # Skip files in excluded directories
                if any(part in exclude_dirs for part in Path(rel_path).parts):
                    continue

                # Only include files with relevant extensions
                ext = os.path.splitext(file)[1].lower()
                if ext in ALL_EXTENSIONS:
                    self.files.append(file_path)

        logger.info(f"Found {len(self.files)} files to analyze")

    def analyze_python_linting(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze Python file for linting issues.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary with linting results
        """
        issues = []

        try:
            # Check for syntax errors
            with tokenize.open(file_path) as f:
                tokens = list(tokenize.generate_tokens(f.readline))
        except (SyntaxError, tokenize.TokenError) as e:
            issues.append({
                'line': getattr(e, 'lineno', 0),
                'column': getattr(e, 'offset', 0),
                'message': f"Syntax error: {str(e)}",
                'severity': 'error'
            })
            return {'issues': issues, 'count': len(issues)}

        try:
            # Simple linting checks
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()

            # Check line length
            for i, line in enumerate(lines, 1):
                if len(line) > 100:
                    issues.append({
                        'line': i,
                        'column': 101,
                        'message': f"Line too long ({len(line)} > 100 characters)",
                        'severity': 'warning'
                    })

            # Check for TODO comments
            for i, line in enumerate(lines, 1):
                if 'TODO' in line or 'FIXME' in line:
                    issues.append({
                        'line': i,
                        'column': line.find('TODO') if 'TODO' in line else line.find('FIXME'),
                        'message': "TODO or FIXME comment found",
                        'severity': 'info'
                    })

            # Check for trailing whitespace
            for i, line in enumerate(lines, 1):
                if line.rstrip() != line:
                    issues.append({
                        'line': i,
                        'column': len(line.rstrip()),
                        'message': "Trailing whitespace",
                        'severity': 'warning'
                    })
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            issues.append({
                'line': 0,
                'column': 0,
                'message': f"Error analyzing file: {str(e)}",
                'severity': 'error'
            })

        return {'issues': issues, 'count': len(issues)}

    def analyze_js_linting(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze JavaScript/TypeScript file for linting issues.

        Args:
            file_path: Path to the JS/TS file

        Returns:
            Dictionary with linting results
        """
        issues = []

        try:
            # Simple linting checks
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()

            # Check line length
            for i, line in enumerate(lines, 1):
                if len(line) > 100:
                    issues.append({
                        'line': i,
                        'column': 101,
                        'message': f"Line too long ({len(line)} > 100 characters)",
                        'severity': 'warning'
                    })

            # Check for TODO comments
            for i, line in enumerate(lines, 1):
                if 'TODO' in line or 'FIXME' in line:
                    issues.append({
                        'line': i,
                        'column': line.find('TODO') if 'TODO' in line else line.find('FIXME'),
                        'message': "TODO or FIXME comment found",
                        'severity': 'info'
                    })

            # Check for trailing whitespace
            for i, line in enumerate(lines, 1):
                if line.rstrip() != line:
                    issues.append({
                        'line': i,
                        'column': len(line.rstrip()),
                        'message': "Trailing whitespace",
                        'severity': 'warning'
                    })

            # Check for console.log statements
            for i, line in enumerate(lines, 1):
                if 'console.log' in line:
                    issues.append({
                        'line': i,
                        'column': line.find('console.log'),
                        'message': "console.log statement found",
                        'severity': 'warning'
                    })
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            issues.append({
                'line': 0,
                'column': 0,
                'message': f"Error analyzing file: {str(e)}",
                'severity': 'error'
            })

        return {'issues': issues, 'count': len(issues)}

    def calculate_cyclomatic_complexity(self, file_path: str) -> Dict[str, Any]:
        """
        Calculate cyclomatic complexity for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with complexity metrics
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in PYTHON_EXTENSIONS:
            return self.calculate_python_complexity(file_path)
        elif ext in JS_EXTENSIONS:
            return self.calculate_js_complexity(file_path)
        else:
            return {'functions': [], 'average': 0, 'max': 0}

    def calculate_python_complexity(self, file_path: str) -> Dict[str, Any]:
        """
        Calculate cyclomatic complexity for a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary with complexity metrics
        """
        functions = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip large files
            if len(content) > 100000:  # Skip files larger than 100KB
                return {
                    'functions': [],
                    'average': 0,
                    'max': 0
                }

            # Parse the Python file
            try:
                tree = ast.parse(content)
            except SyntaxError:
                # Skip files with syntax errors
                return {
                    'functions': [],
                    'average': 0,
                    'max': 0
                }

            # Simplified complexity calculation - just count the number of functions and classes
            # and estimate complexity based on file size and number of control structures
            function_count = 0
            class_count = 0
            if_count = 0
            loop_count = 0
            try_count = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'complexity': 1  # Default complexity
                    })
                elif isinstance(node, ast.ClassDef):
                    class_count += 1
                elif isinstance(node, ast.If):
                    if_count += 1
                elif isinstance(node, (ast.For, ast.While)):
                    loop_count += 1
                elif isinstance(node, ast.Try):
                    try_count += 1

            # Estimate average complexity
            if function_count > 0:
                # Estimate complexity based on control structures per function
                total_control = if_count + loop_count + try_count
                avg_complexity = 1 + (total_control / function_count)
                max_complexity = min(10, avg_complexity * 2)  # Estimate max as 2x average, capped at 10
            else:
                avg_complexity = 0
                max_complexity = 0
        except Exception as e:
            logger.error(f"Error calculating complexity for {file_path}: {e}")
            return {
                'functions': [],
                'average': 0,
                'max': 0
            }

        return {
            'functions': functions[:10],  # Limit to 10 functions for performance
            'average': round(avg_complexity, 2),
            'max': round(max_complexity, 2)
        }

    def calculate_js_complexity(self, file_path: str) -> Dict[str, Any]:
        """
        Calculate cyclomatic complexity for a JavaScript/TypeScript file.

        Args:
            file_path: Path to the JS/TS file

        Returns:
            Dictionary with complexity metrics
        """
        functions = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip large files
            if len(content) > 100000:  # Skip files larger than 100KB
                return {
                    'functions': [],
                    'average': 0,
                    'max': 0
                }

            # Simplified approach: just count functions and control structures
            function_count = len(re.findall(r'function\s+([A-Za-z0-9_$]+)', content))
            function_count += len(re.findall(r'(const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*function', content))
            function_count += len(re.findall(r'(const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*\([^)]*\)\s*=>', content))

            # Count control structures
            if_count = len(re.findall(r'\bif\b', content))
            else_if_count = len(re.findall(r'\belse\s+if\b', content))
            for_count = len(re.findall(r'\bfor\b', content))
            while_count = len(re.findall(r'\bwhile\b', content))
            do_count = len(re.findall(r'\bdo\b', content))
            catch_count = len(re.findall(r'\bcatch\b', content))
            switch_count = len(re.findall(r'\bswitch\b', content))
            case_count = len(re.findall(r'\bcase\b', content))

            # Add a few sample functions for the report
            function_matches = re.finditer(r'function\s+([A-Za-z0-9_$]+)', content)
            for i, match in enumerate(function_matches):
                if i >= 10:  # Limit to 10 functions
                    break
                name = match.group(1)
                lineno = content[:match.start()].count('\n') + 1
                functions.append({
                    'name': name,
                    'line': lineno,
                    'complexity': 1  # Default complexity
                })

            # Estimate complexity
            if function_count > 0:
                total_control = if_count + else_if_count + for_count + while_count + do_count + catch_count + switch_count + case_count
                avg_complexity = 1 + (total_control / function_count)
                max_complexity = min(10, avg_complexity * 2)  # Estimate max as 2x average, capped at 10
            else:
                avg_complexity = 0
                max_complexity = 0
        except Exception as e:
            logger.error(f"Error calculating complexity for {file_path}: {e}")
            return {
                'functions': [],
                'average': 0,
                'max': 0
            }

        return {
            'functions': functions[:10],  # Limit to 10 functions for performance
            'average': round(avg_complexity, 2),
            'max': round(max_complexity, 2)
        }

    def estimate_technical_debt(self, file_path: str) -> Dict[str, Any]:
        """
        Estimate technical debt for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with technical debt metrics
        """
        debt_indicators = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip large files
            if len(content) > 100000:  # Skip files larger than 100KB
                return {
                    'indicators': [],
                    'score': 0,
                    'ratio': 0,
                    'level': 'unknown'
                }

            lines = content.splitlines()

            # Simplified approach: just count indicators without storing all details
            todo_count = 0
            fixme_count = 0
            magic_number_count = 0
            commented_code_count = 0
            hardcoded_url_count = 0

            # Sample a subset of lines for detailed indicators
            max_indicators = 10
            sample_size = min(len(lines), 1000)  # Sample at most 1000 lines

            # Use regular expressions to find patterns in the entire content
            todo_count = len(re.findall(r'TODO', content))
            fixme_count = len(re.findall(r'FIXME', content))
            magic_number_count = len(re.findall(r'[^A-Za-z0-9_"\'][-+]?[0-9]+[^A-Za-z0-9_]', content))
            commented_code_count = len(re.findall(r'\s*#\s*[a-zA-Z0-9_]+\s*\(', content)) + len(re.findall(r'\s*//\s*[a-zA-Z0-9_]+\s*\(', content))
            hardcoded_url_count = len(re.findall(r'["\'](http|https|ftp|file)://', content))

            # Sample a few lines for detailed indicators
            import random
            if len(lines) > 0:
                sample_indices = random.sample(range(len(lines)), min(sample_size, len(lines)))
                for i in sorted(sample_indices):
                    line = lines[i]
                    line_num = i + 1

                    # Add a few examples of each type of debt
                    if 'TODO' in line and len(debt_indicators) < max_indicators:
                        debt_indicators.append({
                            'line': line_num,
                            'type': 'TODO',
                            'description': line.strip()
                        })
                    elif 'FIXME' in line and len(debt_indicators) < max_indicators:
                        debt_indicators.append({
                            'line': line_num,
                            'type': 'FIXME',
                            'description': line.strip()
                        })

            # Calculate debt score
            debt_score = todo_count + fixme_count + magic_number_count + commented_code_count + hardcoded_url_count

            # Adjust score based on file size
            file_size = len(lines)
            if file_size > 0:
                debt_ratio = debt_score / file_size
            else:
                debt_ratio = 0

            # Categorize debt level
            if debt_ratio > 0.1:
                debt_level = 'high'
            elif debt_ratio > 0.05:
                debt_level = 'medium'
            else:
                debt_level = 'low'
        except Exception as e:
            logger.error(f"Error estimating technical debt for {file_path}: {e}")
            debt_indicators = []
            debt_score = 0
            debt_ratio = 0
            debt_level = 'unknown'

        return {
            'indicators': debt_indicators[:max_indicators],  # Limit to max_indicators
            'score': debt_score,
            'ratio': round(debt_ratio, 4),
            'level': debt_level
        }

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file.

        Args:
            file_path: Path to the file

        Returns:
            Analysis results for the file
        """
        ext = os.path.splitext(file_path)[1].lower()
        rel_path = os.path.relpath(file_path, self.project_root)

        result = {
            'path': rel_path,
            'extension': ext,
            'linting': None,
            'complexity': None,
            'tech_debt': None
        }

        # Analyze linting
        if ext in PYTHON_EXTENSIONS:
            result['linting'] = self.analyze_python_linting(file_path)
        elif ext in JS_EXTENSIONS:
            result['linting'] = self.analyze_js_linting(file_path)

        # Calculate complexity
        result['complexity'] = self.calculate_cyclomatic_complexity(file_path)

        # Estimate technical debt
        result['tech_debt'] = self.estimate_technical_debt(file_path)

        return result

    def find_duplicates_simplified(self) -> Dict[str, Any]:
        """
        Find duplicate code across files using a simplified approach.

        Returns:
            Dictionary with duplication metrics
        """
        logger.info("Finding duplicate code (simplified approach)...")

        # Take a sample of files for duplication analysis
        max_files_for_duplication = 50
        if len(self.files) > max_files_for_duplication:
            import random
            duplication_files = random.sample(self.files, max_files_for_duplication)
        else:
            duplication_files = self.files

        logger.info(f"Checking {len(duplication_files)} files for duplication")

        # Group files by extension
        files_by_ext = {}
        for file_path in duplication_files:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in files_by_ext:
                files_by_ext[ext] = []
            files_by_ext[ext].append(file_path)

        duplicates = []
        total_lines = 0
        duplicate_lines = 0

        # Process each extension group separately
        for ext, files in files_by_ext.items():
            # Read file contents
            file_contents = {}
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.splitlines()
                        file_contents[file_path] = lines
                        total_lines += len(lines)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

            # Use a hash-based approach for finding duplicates
            # Hash each block of lines and check for collisions
            min_block_size = 6
            block_hashes = {}

            for file_path, lines in file_contents.items():
                if len(lines) < min_block_size:
                    continue

                for i in range(len(lines) - min_block_size + 1):
                    block = '\n'.join(lines[i:i+min_block_size])
                    if not block.strip():
                        continue  # Skip empty blocks

                    block_hash = hash(block)

                    if block_hash in block_hashes:
                        # Found a duplicate block
                        prev_file, prev_line = block_hashes[block_hash]

                        # Skip if it's the same file and overlapping
                        if prev_file == file_path and abs(prev_line - i) < min_block_size:
                            continue

                        duplicates.append({
                            'file1': os.path.relpath(prev_file, self.project_root),
                            'start_line1': prev_line + 1,
                            'end_line1': prev_line + min_block_size,
                            'file2': os.path.relpath(file_path, self.project_root),
                            'start_line2': i + 1,
                            'end_line2': i + min_block_size,
                            'size': min_block_size
                        })
                        duplicate_lines += min_block_size
                    else:
                        block_hashes[block_hash] = (file_path, i)

        # Calculate duplication percentage
        if total_lines > 0:
            duplication_percentage = (duplicate_lines / total_lines) * 100
        else:
            duplication_percentage = 0

        return {
            'duplicates': duplicates[:100],  # Limit to 100 duplicates for performance
            'total_lines': total_lines,
            'duplicate_lines': duplicate_lines,
            'duplication_percentage': round(duplication_percentage, 2)
        }

    def find_duplicates(self) -> Dict[str, Any]:
        """
        Original find_duplicates method - kept for compatibility but not used.
        """
        # Call the simplified version instead
        return self.find_duplicates_simplified()

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze code quality.

        Returns:
            Dictionary with code quality metrics
        """
        # Find files
        self.find_files()

        # Limit the number of files to analyze for performance
        max_files = 300  # Analyze at most 300 files
        if len(self.files) > max_files:
            logger.info(f"Limiting analysis to {max_files} files out of {len(self.files)} for performance")
            import random
            random.shuffle(self.files)
            self.files = self.files[:max_files]

        # Skip large files (>1MB)
        self.files = [f for f in self.files if os.path.getsize(f) < 1024 * 1024]
        logger.info(f"Analyzing {len(self.files)} files after filtering large files")

        # Analyze files
        logger.info("Analyzing files...")
        file_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_file = {executor.submit(self.analyze_file, file): file for file in self.files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    rel_path = os.path.relpath(file, self.project_root)
                    file_results[rel_path] = result
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")

        # Find duplicates (simplified approach)
        logger.info("Finding duplicates (simplified approach)...")
        duplication_results = self.find_duplicates_simplified()

        # Calculate summary metrics
        total_issues = sum(result['linting']['count'] for result in file_results.values() if result['linting'])
        avg_complexity = statistics.mean([result['complexity']['average'] for result in file_results.values() if result['complexity']['average'] > 0]) if file_results else 0
        max_complexity = max([result['complexity']['max'] for result in file_results.values()]) if file_results else 0

        tech_debt_scores = [result['tech_debt']['score'] for result in file_results.values() if result['tech_debt']]
        total_debt_score = sum(tech_debt_scores)
        avg_debt_score = statistics.mean(tech_debt_scores) if tech_debt_scores else 0

        # Count files by debt level
        debt_levels = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
        for result in file_results.values():
            if result['tech_debt']:
                debt_levels[result['tech_debt']['level']] += 1

        summary = {
            'total_files': len(file_results),
            'total_issues': total_issues,
            'issues_per_file': round(total_issues / len(file_results), 2) if file_results else 0,
            'avg_complexity': round(avg_complexity, 2),
            'max_complexity': max_complexity,
            'total_debt_score': total_debt_score,
            'avg_debt_score': round(avg_debt_score, 2),
            'debt_levels': debt_levels,
            'duplication_percentage': duplication_results['duplication_percentage']
        }

        return {
            'files': file_results,
            'duplication': duplication_results,
            'summary': summary
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze code quality metrics")
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

    # Analyze code quality
    analyzer = CodeQualityAnalyzer(Path(args.project_root))
    results = analyzer.analyze()

    # Save results
    output_path = os.path.join(args.output_dir, "code_quality.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Code quality metrics saved to {output_path}")

if __name__ == "__main__":
    import statistics
    from collections import defaultdict
    main()
