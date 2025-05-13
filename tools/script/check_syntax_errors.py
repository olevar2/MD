#!/usr/bin/env python3
"""
Check for syntax errors in Python files.
"""

import os
import ast
import sys

def check_syntax_errors(directory):
    """
    Check for syntax errors in Python files in the given directory.
    
    Args:
        directory: Directory to check
    
    Returns:
        List of tuples (file_path, error_message)
    """
    syntax_errors = []
    
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
    
    return syntax_errors

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    syntax_errors = check_syntax_errors(directory)
    
    print(f"Found {len(syntax_errors)} Python files with syntax errors")
    
    if syntax_errors:
        print("\nFiles with syntax errors:")
        for file_path, error in syntax_errors:
            print(f"{file_path}: {error}")
    else:
        print("No syntax errors found.")