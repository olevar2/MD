#!/usr/bin/env python3
"""
Check for hardcoded credentials in the codebase.
"""

import os
import re
import sys

# Patterns to look for
CREDENTIAL_PATTERNS = [
    r'password\s*=\s*[\'"][^\'"]+[\'"]',
    r'passwd\s*=\s*[\'"][^\'"]+[\'"]',
    r'pwd\s*=\s*[\'"][^\'"]+[\'"]',
    r'secret\s*=\s*[\'"][^\'"]+[\'"]',
    r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
    r'apikey\s*=\s*[\'"][^\'"]+[\'"]',
    r'token\s*=\s*[\'"][^\'"]+[\'"]',
    r'access_key\s*=\s*[\'"][^\'"]+[\'"]',
    r'auth\s*=\s*[\'"][^\'"]+[\'"]',
    r'credentials\s*=\s*[\'"][^\'"]+[\'"]',
]

# Files to exclude
EXCLUDE_EXTENSIONS = [
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot',
    '.zip', '.tar', '.gz', '.rar', '.7z', '.exe', '.dll', '.so', '.dylib',
    '.pyc', '.pyo', '.pyd', '.db', '.sqlite', '.sqlite3',
]

# Directories to exclude
EXCLUDE_DIRS = [
    '.git', '.github', '.pytest_cache', '__pycache__', 
    'node_modules', '.venv', 'venv', 'env', '.vscode',
]

def check_file(file_path):
    """
    Check a file for hardcoded credentials.
    
    Args:
        file_path: Path to the file to check
    
    Returns:
        List of tuples (line_number, line, pattern)
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                for pattern in CREDENTIAL_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip if it's in a comment
                        if line.strip().startswith('#') or line.strip().startswith('//'):
                            continue
                        
                        # Skip if it's importing a module
                        if 'import' in line and 'from' in line:
                            continue
                        
                        # Skip if it's a variable name definition without assignment
                        if re.match(r'^\s*\w+\s*:\s*\w+\s*$', line.strip()):
                            continue
                        
                        # Skip if it's in a docstring
                        if '"""' in line or "'''" in line:
                            continue
                        
                        # Skip if it's a template or placeholder
                        if '${' in line or '{' in line and '}' in line:
                            continue
                        
                        # Skip if it's a reference to an environment variable
                        if 'os.environ' in line or 'getenv' in line:
                            continue
                        
                        # Skip if it's a reference to a configuration file
                        if 'config' in line.lower() and ('get' in line.lower() or 'load' in line.lower()):
                            continue
                        
                        results.append((i + 1, line.strip(), pattern))
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
    
    return results

def check_directory(directory, max_files=1000):
    """
    Check a directory for hardcoded credentials.
    
    Args:
        directory: Directory to check
        max_files: Maximum number of files to check
    
    Returns:
        Dictionary mapping file paths to lists of tuples (line_number, line, pattern)
    """
    results = {}
    files_checked = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        
        for file in files:
            # Skip excluded file types
            if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                continue
            
            file_path = os.path.join(root, file)
            file_results = check_file(file_path)
            
            if file_results:
                results[file_path] = file_results
            
            files_checked += 1
            if files_checked >= max_files:
                return results
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_hardcoded_credentials.py <directory> [max_files]")
        sys.exit(1)
    
    directory = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    results = check_directory(directory, max_files)
    
    print(f"Checked up to {max_files} files in {directory}")
    print(f"Found {len(results)} files with potential hardcoded credentials")
    
    if results:
        print("\nFiles with potential hardcoded credentials:")
        for file_path, file_results in results.items():
            print(f"\n{file_path}:")
            for line_number, line, pattern in file_results:
                print(f"  Line {line_number}: {line}")
    else:
        print("No hardcoded credentials found.")