#!/usr/bin/env python3
"""
Check for syntax errors in Python files in a specific directory.
"""

import os
import ast
import sys

def check_syntax_errors(directory, max_files=100):
    """
    Check for syntax errors in Python files in the given directory.
    
    Args:
        directory: Directory to check
        max_files: Maximum number of files to check
    
    Returns:
        List of tuples (file_path, error_message)
    """
    syntax_errors = []
    files_checked = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                except Exception as e:
                    syntax_errors.append((file_path, str(e)))
                
                files_checked += 1
                if files_checked >= max_files:
                    return syntax_errors
    
    return syntax_errors

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_syntax_errors_limited.py <directory> [max_files]")
        sys.exit(1)
    
    directory = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    syntax_errors = check_syntax_errors(directory, max_files)
    
    print(f"Checked up to {max_files} Python files in {directory}")
    print(f"Found {len(syntax_errors)} Python files with syntax errors")
    
    if syntax_errors:
        print("\nFiles with syntax errors:")
        for file_path, error in syntax_errors:
            print(f"{file_path}: {error}")
    else:
        print("No syntax errors found.")