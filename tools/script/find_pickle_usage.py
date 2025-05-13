#!/usr/bin/env python3
"""
Find Python files using pickle for serialization/deserialization.
"""

import os
import re

def find_pickle_usage(directory):
    """
    Find Python files using pickle.
    
    Args:
        directory: Directory to search
    
    Returns:
        List of file paths using pickle
    """
    pickle_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for pickle imports
                        if re.search(r'import\s+pickle', content) or re.search(r'from\s+pickle\s+import', content):
                            pickle_files.append(file_path)
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return pickle_files

if __name__ == "__main__":
    pickle_files = find_pickle_usage(".")
    
    print(f"Found {len(pickle_files)} Python files using pickle:")
    for file in pickle_files:
        print(f"- {file}")