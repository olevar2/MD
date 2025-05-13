#!/usr/bin/env python3
"""
Find insecure deserialization vulnerabilities in Python files.
"""

import os
import re

def find_insecure_deserialization(directories):
    """
    Find insecure deserialization vulnerabilities.
    
    Args:
        directories: List of directories to search
    
    Returns:
        List of tuples (file_path, line_number, vulnerability_type, code_snippet)
    """
    vulnerabilities = []
    
    # Patterns to look for
    patterns = [
        (r'pickle\.loads?\(.*\)', 'pickle.loads'),
        (r'pickle\.dumps?\(.*\)', 'pickle.dumps'),
        (r'cPickle\.loads?\(.*\)', 'cPickle.loads'),
        (r'cPickle\.dumps?\(.*\)', 'cPickle.dumps'),
        (r'yaml\.load\(.*\)', 'yaml.load without safe=True'),
        (r'yaml\.unsafe_load\(.*\)', 'yaml.unsafe_load'),
        (r'eval\(.*\)', 'eval'),
        (r'exec\(.*\)', 'exec'),
        (r'__import__\(.*\)', '__import__'),
        (r'subprocess\.call\(.*shell\s*=\s*True', 'subprocess.call with shell=True'),
        (r'subprocess\.Popen\(.*shell\s*=\s*True', 'subprocess.Popen with shell=True'),
        (r'os\.system\(.*\)', 'os.system'),
        (r'os\.popen\(.*\)', 'os.popen'),
        (r'marshal\.loads?\(.*\)', 'marshal.loads'),
        (r'shelve\.open\(.*\)', 'shelve.open'),
        (r'dill\.loads?\(.*\)', 'dill.loads'),
        (r'jsonpickle\.decode\(.*\)', 'jsonpickle.decode')
    ]
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                            for i, line in enumerate(lines):
                                for pattern, vuln_type in patterns:
                                    if re.search(pattern, line):
                                        # Skip if it's in a comment
                                        if line.strip().startswith('#'):
                                            continue
                                        
                                        # Skip if it's in a docstring
                                        if '"""' in line or "'''" in line:
                                            continue
                                        
                                        # Get context (3 lines before and after)
                                        start = max(0, i - 3)
                                        end = min(len(lines), i + 4)
                                        context = ''.join(lines[start:end])
                                        
                                        vulnerabilities.append((file_path, i + 1, vuln_type, context.strip()))
                                        
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    return vulnerabilities

if __name__ == "__main__":
    # Focus on specific directories
    directories = [
        "./analysis-engine-service",
        "./common-lib",
        "./data-pipeline-service",
        "./feature-store-service",
        "./ml-integration-service",
        "./ml-workbench-service",
        "./strategy-execution-engine"
    ]
    
    vulnerabilities = find_insecure_deserialization(directories)
    
    print(f"Found {len(vulnerabilities)} potential insecure deserialization vulnerabilities:")
    
    # Group by file
    files = {}
    for file_path, line_number, vuln_type, context in vulnerabilities:
        if file_path not in files:
            files[file_path] = []
        files[file_path].append((line_number, vuln_type, context))
    
    # Print results
    for file_path, vulns in files.items():
        print(f"\n{file_path}:")
        for line_number, vuln_type, context in vulns:
            print(f"  Line {line_number}: {vuln_type}")
            print(f"  Context:\n{context}")
            print("  " + "-" * 50)