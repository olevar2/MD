#!/usr/bin/env python3
"""
Find hardcoded credentials in Python files.
"""

import os
import re

def find_hardcoded_credentials():
    """
    Find hardcoded credentials in Python files.
    """
    # Patterns to look for
    pattern = re.compile(r'(password|passwd|pwd|api_key|apikey|secret|token|username|user|private_key|connection_string|conn_str)\s*=\s*["\']([^"\']+)["\']')
    
    # Exclusion patterns
    exclusion_patterns = [
        r'example',
        r'sample',
        r'test',
        r'dummy',
        r'fake',
        r'placeholder',
        r'template',
        r'default',
        r'mock',
        r'demo',
        r'tutorial',
        r'env',
        r'getenv',
        r'config',
        r'settings',
        r'param',
        r'variable',
    ]
    
    results = []
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        for match in pattern.finditer(content):
                            # Check if match contains exclusion patterns
                            excluded = False
                            for exclusion in exclusion_patterns:
                                if re.search(exclusion, match.group(0), re.IGNORECASE):
                                    excluded = True
                                    break
                                    
                            if not excluded:
                                line_num = content[:match.start()].count('\n') + 1
                                results.append((file_path, line_num, match.group(1), match.group(0)))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return results

if __name__ == "__main__":
    results = find_hardcoded_credentials()
    
    print(f"Found {len(results)} potential hardcoded credentials:")
    
    for file_path, line_num, cred_type, line in results[:50]:
        print(f"{file_path}:{line_num} - {cred_type}: {line}")