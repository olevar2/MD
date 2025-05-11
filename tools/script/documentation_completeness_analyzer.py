#!/usr/bin/env python3
"""
Forex Trading Platform Documentation Completeness Analyzer

This script analyzes the completeness of documentation in the forex trading platform:
1. API documentation (docstrings, OpenAPI/Swagger files)
2. Architecture documentation (README files, architecture diagrams)
3. Operational runbooks (deployment, monitoring, troubleshooting)
4. Developer onboarding materials (setup guides, contribution guidelines)

Output is a comprehensive JSON file that maps the documentation status of the platform.
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
    "node_modules", ".venv", "venv", "env", ".vscode"
}

# File extensions and names to analyze
PYTHON_EXTENSIONS = {".py"}
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".html"}
API_DOC_FILES = {"swagger.json", "openapi.json", "swagger.yaml", "openapi.yaml", "api-docs.json"}
ARCHITECTURE_DOC_FILES = {"architecture.md", "design.md", "overview.md", "structure.md"}
RUNBOOK_FILES = {"runbook.md", "deployment.md", "monitoring.md", "troubleshooting.md", "operations.md"}
ONBOARDING_FILES = {"setup.md", "getting-started.md", "contributing.md", "development.md"}
README_FILES = {"README.md", "README.rst", "README.txt"}

ALL_EXTENSIONS = PYTHON_EXTENSIONS | JS_EXTENSIONS | DOC_EXTENSIONS

class DocCompleteness:
    """Enum-like class for documentation completeness levels."""
    MISSING = "missing"
    MINIMAL = "minimal"
    PARTIAL = "partial"
    COMPLETE = "complete"

class DocumentationCompletenessAnalyzer:
    """Analyzes the documentation completeness of the forex trading platform."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.files = []
        self.python_files = []
        self.js_files = []
        self.doc_files = []
        self.api_doc_files = []
        self.architecture_doc_files = []
        self.runbook_files = []
        self.onboarding_files = []
        self.readme_files = []
        
        self.docstring_stats = {
            'module_docstrings': 0,
            'class_docstrings': 0,
            'function_docstrings': 0,
            'method_docstrings': 0,
            'total_modules': 0,
            'total_classes': 0,
            'total_functions': 0,
            'total_methods': 0
        }
        
        self.doc_completeness = {
            'api_documentation': DocCompleteness.MISSING,
            'architecture_documentation': DocCompleteness.MISSING,
            'operational_runbooks': DocCompleteness.MISSING,
            'developer_onboarding': DocCompleteness.MISSING
        }
        
        self.doc_content = {
            'api_documentation': [],
            'architecture_documentation': [],
            'operational_runbooks': [],
            'developer_onboarding': []
        }
    
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
                ext = os.path.splitext(file)[1].lower()
                
                if ext in PYTHON_EXTENSIONS:
                    self.python_files.append(file_path)
                    self.files.append(file_path)
                
                elif ext in JS_EXTENSIONS:
                    self.js_files.append(file_path)
                    self.files.append(file_path)
                
                elif ext in DOC_EXTENSIONS:
                    self.doc_files.append(file_path)
                    self.files.append(file_path)
                    
                    # Categorize documentation files
                    if file in API_DOC_FILES:
                        self.api_doc_files.append(file_path)
                    
                    elif file in ARCHITECTURE_DOC_FILES:
                        self.architecture_doc_files.append(file_path)
                    
                    elif file in RUNBOOK_FILES:
                        self.runbook_files.append(file_path)
                    
                    elif file in ONBOARDING_FILES:
                        self.onboarding_files.append(file_path)
                    
                    elif file in README_FILES:
                        self.readme_files.append(file_path)
        
        logger.info(f"Found {len(self.files)} files to analyze")
        logger.info(f"Python files: {len(self.python_files)}")
        logger.info(f"JS files: {len(self.js_files)}")
        logger.info(f"Documentation files: {len(self.doc_files)}")
    
    def analyze_python_docstrings(self, file_path: str) -> Dict[str, int]:
        """
        Analyze Python docstrings in a file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with docstring statistics
        """
        stats = {
            'module_docstring': 0,
            'class_docstrings': 0,
            'function_docstrings': 0,
            'method_docstrings': 0,
            'total_classes': 0,
            'total_functions': 0,
            'total_methods': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                
                # Check for module docstring
                if ast.get_docstring(tree):
                    stats['module_docstring'] = 1
                
                # Check classes and functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        stats['total_classes'] += 1
                        if ast.get_docstring(node):
                            stats['class_docstrings'] += 1
                        
                        # Check methods
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef):
                                stats['total_methods'] += 1
                                if ast.get_docstring(child):
                                    stats['method_docstrings'] += 1
                    
                    elif isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
                        stats['total_functions'] += 1
                        if ast.get_docstring(node):
                            stats['function_docstrings'] += 1
            
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, skipping docstring analysis")
        
        except Exception as e:
            logger.error(f"Error analyzing docstrings in {file_path}: {e}")
        
        return stats
    
    def analyze_js_docstrings(self, file_path: str) -> Dict[str, int]:
        """
        Analyze JavaScript/TypeScript docstrings (JSDoc) in a file.
        
        Args:
            file_path: Path to the JS/TS file
            
        Returns:
            Dictionary with docstring statistics
        """
        stats = {
            'module_docstring': 0,
            'class_docstrings': 0,
            'function_docstrings': 0,
            'method_docstrings': 0,
            'total_classes': 0,
            'total_functions': 0,
            'total_methods': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count JSDoc comments
            jsdoc_pattern = r'/\*\*[\s\S]*?\*/'
            jsdoc_comments = re.findall(jsdoc_pattern, content)
            
            # Count classes
            class_pattern = r'class\s+([A-Za-z0-9_]+)'
            classes = re.findall(class_pattern, content)
            stats['total_classes'] = len(classes)
            
            # Count functions
            function_pattern = r'function\s+([A-Za-z0-9_]+)'
            functions = re.findall(function_pattern, content)
            stats['total_functions'] = len(functions)
            
            # Count methods
            method_pattern = r'([A-Za-z0-9_]+)\s*\([^)]*\)\s*{'
            methods = re.findall(method_pattern, content)
            stats['total_methods'] = len(methods) - stats['total_functions']
            
            # Estimate docstring coverage (simplified)
            if jsdoc_comments:
                # Assume first comment is module docstring if it appears at the top
                if content.lstrip().startswith('/**'):
                    stats['module_docstring'] = 1
                    jsdoc_comments = jsdoc_comments[1:]
                
                # Distribute remaining comments among classes, functions, and methods
                # This is a simplification, as we're not parsing the JS/TS AST
                remaining = len(jsdoc_comments)
                stats['class_docstrings'] = min(remaining, stats['total_classes'])
                remaining -= stats['class_docstrings']
                
                stats['function_docstrings'] = min(remaining, stats['total_functions'])
                remaining -= stats['function_docstrings']
                
                stats['method_docstrings'] = min(remaining, stats['total_methods'])
        
        except Exception as e:
            logger.error(f"Error analyzing JSDoc in {file_path}: {e}")
        
        return stats
    
    def analyze_doc_content(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze documentation content in a file.
        
        Args:
            file_path: Path to the documentation file
            
        Returns:
            Dictionary with documentation content analysis
        """
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'size_bytes': os.path.getsize(file_path),
            'line_count': 0,
            'heading_count': 0,
            'code_block_count': 0,
            'image_count': 0,
            'link_count': 0,
            'table_count': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count lines
            result['line_count'] = len(content.splitlines())
            
            # Count Markdown elements
            if file_path.endswith(('.md', '.markdown')):
                # Count headings
                heading_pattern = r'^#{1,6}\s+.+$'
                result['heading_count'] = len(re.findall(heading_pattern, content, re.MULTILINE))
                
                # Count code blocks
                code_block_pattern = r'```[\s\S]*?```'
                result['code_block_count'] = len(re.findall(code_block_pattern, content))
                
                # Count images
                image_pattern = r'!\[.*?\]\(.*?\)'
                result['image_count'] = len(re.findall(image_pattern, content))
                
                # Count links
                link_pattern = r'\[.*?\]\(.*?\)'
                result['link_count'] = len(re.findall(link_pattern, content))
                
                # Count tables
                table_pattern = r'\|.*\|.*\|[\s\S]*?\|.*\|'
                result['table_count'] = len(re.findall(table_pattern, content))
        
        except Exception as e:
            logger.error(f"Error analyzing documentation content in {file_path}: {e}")
        
        return result
    
    def assess_completeness(self) -> None:
        """Assess the completeness of documentation."""
        # API documentation
        if self.api_doc_files:
            self.doc_completeness['api_documentation'] = DocCompleteness.COMPLETE
        elif self.docstring_stats['function_docstrings'] / max(1, self.docstring_stats['total_functions']) > 0.7:
            self.doc_completeness['api_documentation'] = DocCompleteness.PARTIAL
        elif self.docstring_stats['function_docstrings'] / max(1, self.docstring_stats['total_functions']) > 0.3:
            self.doc_completeness['api_documentation'] = DocCompleteness.MINIMAL
        
        # Architecture documentation
        if self.architecture_doc_files:
            self.doc_completeness['architecture_documentation'] = DocCompleteness.COMPLETE
        elif len(self.readme_files) > 5:  # Multiple README files suggest some architecture docs
            self.doc_completeness['architecture_documentation'] = DocCompleteness.PARTIAL
        elif self.readme_files:
            self.doc_completeness['architecture_documentation'] = DocCompleteness.MINIMAL
        
        # Operational runbooks
        if len(self.runbook_files) >= 3:  # Multiple runbooks
            self.doc_completeness['operational_runbooks'] = DocCompleteness.COMPLETE
        elif self.runbook_files:
            self.doc_completeness['operational_runbooks'] = DocCompleteness.PARTIAL
        elif any("deploy" in f.lower() or "operation" in f.lower() for f in self.doc_files):
            self.doc_completeness['operational_runbooks'] = DocCompleteness.MINIMAL
        
        # Developer onboarding
        if len(self.onboarding_files) >= 2:
            self.doc_completeness['developer_onboarding'] = DocCompleteness.COMPLETE
        elif self.onboarding_files:
            self.doc_completeness['developer_onboarding'] = DocCompleteness.PARTIAL
        elif any("setup" in f.lower() or "install" in f.lower() for f in self.doc_files):
            self.doc_completeness['developer_onboarding'] = DocCompleteness.MINIMAL
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the documentation completeness.
        
        Returns:
            Analysis results
        """
        logger.info("Starting documentation completeness analysis...")
        
        # Find all files
        self.find_files()
        
        # Analyze Python docstrings
        logger.info("Analyzing Python docstrings...")
        for file_path in self.python_files:
            stats = self.analyze_python_docstrings(file_path)
            
            # Update global stats
            self.docstring_stats['module_docstrings'] += stats['module_docstring']
            self.docstring_stats['class_docstrings'] += stats['class_docstrings']
            self.docstring_stats['function_docstrings'] += stats['function_docstrings']
            self.docstring_stats['method_docstrings'] += stats['method_docstrings']
            self.docstring_stats['total_modules'] += 1
            self.docstring_stats['total_classes'] += stats['total_classes']
            self.docstring_stats['total_functions'] += stats['total_functions']
            self.docstring_stats['total_methods'] += stats['total_methods']
        
        # Analyze JS docstrings
        logger.info("Analyzing JS/TS docstrings...")
        for file_path in self.js_files:
            stats = self.analyze_js_docstrings(file_path)
            
            # Update global stats
            self.docstring_stats['module_docstrings'] += stats['module_docstring']
            self.docstring_stats['class_docstrings'] += stats['class_docstrings']
            self.docstring_stats['function_docstrings'] += stats['function_docstrings']
            self.docstring_stats['method_docstrings'] += stats['method_docstrings']
            self.docstring_stats['total_modules'] += 1
            self.docstring_stats['total_classes'] += stats['total_classes']
            self.docstring_stats['total_functions'] += stats['total_functions']
            self.docstring_stats['total_methods'] += stats['total_methods']
        
        # Analyze documentation content
        logger.info("Analyzing documentation content...")
        
        # Analyze API documentation
        for file_path in self.api_doc_files:
            self.doc_content['api_documentation'].append(self.analyze_doc_content(file_path))
        
        # Analyze architecture documentation
        for file_path in self.architecture_doc_files:
            self.doc_content['architecture_documentation'].append(self.analyze_doc_content(file_path))
        
        # Analyze operational runbooks
        for file_path in self.runbook_files:
            self.doc_content['operational_runbooks'].append(self.analyze_doc_content(file_path))
        
        # Analyze developer onboarding
        for file_path in self.onboarding_files:
            self.doc_content['developer_onboarding'].append(self.analyze_doc_content(file_path))
        
        # Assess completeness
        self.assess_completeness()
        
        # Calculate docstring coverage percentages
        docstring_coverage = {
            'module_coverage': self.docstring_stats['module_docstrings'] / max(1, self.docstring_stats['total_modules']),
            'class_coverage': self.docstring_stats['class_docstrings'] / max(1, self.docstring_stats['total_classes']),
            'function_coverage': self.docstring_stats['function_docstrings'] / max(1, self.docstring_stats['total_functions']),
            'method_coverage': self.docstring_stats['method_docstrings'] / max(1, self.docstring_stats['total_methods']),
            'overall_coverage': (
                self.docstring_stats['module_docstrings'] + 
                self.docstring_stats['class_docstrings'] + 
                self.docstring_stats['function_docstrings'] + 
                self.docstring_stats['method_docstrings']
            ) / max(1, (
                self.docstring_stats['total_modules'] + 
                self.docstring_stats['total_classes'] + 
                self.docstring_stats['total_functions'] + 
                self.docstring_stats['total_methods']
            ))
        }
        
        # Generate summary
        summary = {
            'docstring_stats': self.docstring_stats,
            'docstring_coverage': docstring_coverage,
            'doc_completeness': self.doc_completeness,
            'doc_content': self.doc_content,
            'file_counts': {
                'python_files': len(self.python_files),
                'js_files': len(self.js_files),
                'doc_files': len(self.doc_files),
                'api_doc_files': len(self.api_doc_files),
                'architecture_doc_files': len(self.architecture_doc_files),
                'runbook_files': len(self.runbook_files),
                'onboarding_files': len(self.onboarding_files),
                'readme_files': len(self.readme_files)
            }
        }
        
        logger.info("Documentation completeness analysis complete")
        return summary

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze forex trading platform documentation completeness")
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
    
    # Analyze documentation completeness
    analyzer = DocumentationCompletenessAnalyzer(Path(args.project_root))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, "documentation_completeness_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Documentation completeness analysis saved to {output_path}")

if __name__ == "__main__":
    main()
