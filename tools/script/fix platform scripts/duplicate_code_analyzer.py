#!/usr/bin/env python3
"""
Forex Trading Platform Duplicate Code Analyzer

This script analyzes the forex trading platform codebase to identify duplicate code.
It uses a combination of token-based and AST-based approaches to find similar code
patterns across different files and services.

Usage:
python duplicate_code_analyzer.py [--project-root <project_root>] [--output-dir <output_dir>] [--min-lines <min_lines>] [--min-similarity <min_similarity>]
"""

import os
import sys
import ast
import json
import difflib
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output\duplicates"

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    ".git", ".github", ".pytest_cache", "__pycache__", 
    "node_modules", ".venv", "venv", "env", ".vscode"
}

class DuplicateCodeAnalyzer:
    """Analyzes the codebase to identify duplicate code."""
    
    def __init__(self, project_root: str, min_lines: int = 5, min_similarity: float = 0.8):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
            min_lines: Minimum number of lines for a code block to be considered
            min_similarity: Minimum similarity score (0.0-1.0) for code blocks to be considered duplicates
        """
        self.project_root = project_root
        self.min_lines = min_lines
        self.min_similarity = min_similarity
        self.files = []
        self.code_blocks = {}
        self.duplicates = []
        
    def find_files(self) -> None:
        """Find all Python files in the project."""
        logger.info(f"Finding Python files in {self.project_root}...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    
                    # Skip files in excluded directories
                    if any(part in EXCLUDE_DIRS for part in Path(rel_path).parts):
                        continue
                    
                    self.files.append(file_path)
        
        logger.info(f"Found {len(self.files)} Python files to analyze")
    
    def extract_code_blocks(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract code blocks from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of code blocks
        """
        blocks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the Python file
            try:
                tree = ast.parse(content)
                
                # Extract functions and methods
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        # Get the source code for this node
                        start_line = node.lineno
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                        
                        # Skip if too short
                        if end_line - start_line + 1 < self.min_lines:
                            continue
                        
                        # Get the source code
                        source_lines = content.splitlines()[start_line-1:end_line]
                        source_code = '\n'.join(source_lines)
                        
                        # Create a code block
                        block = {
                            'file': file_path,
                            'type': node.__class__.__name__,
                            'name': node.name,
                            'start_line': start_line,
                            'end_line': end_line,
                            'code': source_code,
                            'hash': hashlib.md5(source_code.encode()).hexdigest()
                        }
                        
                        blocks.append(block)
            except SyntaxError:
                # Fall back to line-based extraction for files with syntax errors
                lines = content.splitlines()
                
                # Extract blocks of code based on indentation
                current_block = []
                current_indent = 0
                
                for i, line in enumerate(lines):
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    
                    # Calculate indentation
                    indent = len(line) - len(line.lstrip())
                    
                    if not current_block or indent >= current_indent:
                        current_block.append(line)
                        current_indent = indent
                    else:
                        # End of block
                        if len(current_block) >= self.min_lines:
                            source_code = '\n'.join(current_block)
                            block = {
                                'file': file_path,
                                'type': 'Unknown',
                                'name': f"Block_{i-len(current_block)}",
                                'start_line': i - len(current_block) + 1,
                                'end_line': i,
                                'code': source_code,
                                'hash': hashlib.md5(source_code.encode()).hexdigest()
                            }
                            blocks.append(block)
                        
                        # Start a new block
                        current_block = [line]
                        current_indent = indent
                
                # Add the last block
                if len(current_block) >= self.min_lines:
                    source_code = '\n'.join(current_block)
                    block = {
                        'file': file_path,
                        'type': 'Unknown',
                        'name': f"Block_{len(lines)-len(current_block)}",
                        'start_line': len(lines) - len(current_block) + 1,
                        'end_line': len(lines),
                        'code': source_code,
                        'hash': hashlib.md5(source_code.encode()).hexdigest()
                    }
                    blocks.append(block)
        except Exception as e:
            logger.error(f"Error extracting code blocks from {file_path}: {e}")
        
        return blocks
    
    def calculate_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate the similarity between two code blocks.
        
        Args:
            code1: First code block
            code2: Second code block
            
        Returns:
            Similarity score (0.0-1.0)
        """
        return difflib.SequenceMatcher(None, code1, code2).ratio()
    
    def find_duplicates(self) -> None:
        """Find duplicate code blocks."""
        logger.info("Finding duplicate code blocks...")
        
        # Group blocks by hash
        hash_groups = defaultdict(list)
        for block_id, block in self.code_blocks.items():
            hash_groups[block['hash']].append(block_id)
        
        # Find exact duplicates
        exact_duplicates = []
        for hash_value, block_ids in hash_groups.items():
            if len(block_ids) > 1:
                exact_duplicates.append(block_ids)
        
        logger.info(f"Found {len(exact_duplicates)} groups of exact duplicates")
        
        # Find similar blocks
        similar_duplicates = []
        
        # Create a list of blocks to compare
        blocks_to_compare = []
        for i, block1_id in enumerate(self.code_blocks):
            for block2_id in list(self.code_blocks.keys())[i+1:]:
                # Skip if already exact duplicates
                if any(block1_id in group and block2_id in group for group in exact_duplicates):
                    continue
                
                blocks_to_compare.append((block1_id, block2_id))
        
        logger.info(f"Comparing {len(blocks_to_compare)} pairs of code blocks...")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for block1_id, block2_id in blocks_to_compare:
                futures.append(executor.submit(
                    self.compare_blocks,
                    block1_id,
                    block2_id
                ))
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    similar_duplicates.append(result)
        
        logger.info(f"Found {len(similar_duplicates)} pairs of similar blocks")
        
        # Combine exact and similar duplicates
        self.duplicates = exact_duplicates + similar_duplicates
    
    def compare_blocks(self, block1_id: str, block2_id: str) -> Optional[List[str]]:
        """
        Compare two code blocks for similarity.
        
        Args:
            block1_id: ID of the first block
            block2_id: ID of the second block
            
        Returns:
            List of block IDs if similar, None otherwise
        """
        block1 = self.code_blocks[block1_id]
        block2 = self.code_blocks[block2_id]
        
        # Skip if from the same file
        if block1['file'] == block2['file']:
            return None
        
        # Calculate similarity
        similarity = self.calculate_similarity(block1['code'], block2['code'])
        
        # Check if similar enough
        if similarity >= self.min_similarity:
            return [block1_id, block2_id]
        
        return None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the project for duplicate code.
        
        Returns:
            Analysis results
        """
        logger.info("Starting duplicate code analysis...")
        
        # Find all files
        self.find_files()
        
        # Extract code blocks
        logger.info("Extracting code blocks...")
        all_blocks = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.extract_code_blocks, file): file for file in self.files}
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    blocks = future.result()
                    all_blocks.extend(blocks)
                except Exception as e:
                    logger.error(f"Error processing result for {file}: {e}")
        
        # Create a dictionary of code blocks
        for i, block in enumerate(all_blocks):
            block_id = f"block_{i}"
            self.code_blocks[block_id] = block
        
        logger.info(f"Extracted {len(self.code_blocks)} code blocks")
        
        # Find duplicates
        self.find_duplicates()
        
        # Generate report
        duplicate_groups = []
        
        for group in self.duplicates:
            blocks = [self.code_blocks[block_id] for block_id in group]
            
            # Calculate similarity for each pair
            similarities = []
            for i, block1 in enumerate(blocks):
                for block2 in blocks[i+1:]:
                    similarity = self.calculate_similarity(block1['code'], block2['code'])
                    similarities.append(similarity)
            
            # Calculate average similarity
            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
            
            duplicate_groups.append({
                'blocks': blocks,
                'similarity': avg_similarity
            })
        
        # Sort by similarity (descending)
        duplicate_groups.sort(key=lambda g: g['similarity'], reverse=True)
        
        # Generate summary
        summary = {
            'total_files': len(self.files),
            'total_blocks': len(self.code_blocks),
            'duplicate_groups': len(duplicate_groups),
            'duplicates': duplicate_groups
        }
        
        logger.info("Duplicate code analysis complete")
        return summary

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze duplicate code")
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
    parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum number of lines for a code block to be considered"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.8,
        help="Minimum similarity score (0.0-1.0) for code blocks to be considered duplicates"
    )
    parser.add_argument(
        "--report-file",
        default="duplicate-code-report.json",
        help="Name of the report file"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze duplicate code
    analyzer = DuplicateCodeAnalyzer(
        args.project_root,
        args.min_lines,
        args.min_similarity
    )
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, args.report_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Duplicate code analysis saved to {output_path}")
    
    # Print summary
    print("\nDuplicate Code Analysis Summary:")
    print(f"- Analyzed {results['total_files']} files")
    print(f"- Found {results['total_blocks']} code blocks")
    print(f"- Detected {results['duplicate_groups']} groups of duplicate code")

if __name__ == "__main__":
    main()
