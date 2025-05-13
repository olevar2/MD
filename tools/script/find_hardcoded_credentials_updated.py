#!/usr/bin/env python3
"""
Find hardcoded credentials in Python files.
"""

import os
import re
from typing import List, Tuple

def find_hardcoded_credentials(directories: List[str]) -> List[Tuple[str, int, str, str]]:
    """
    Find hardcoded credentials in Python files.
    
    Args:
        directories: List of directories to search
    
    Returns:
        List of tuples (file_path, line_number, credential_type, line)
    """
    results = []
    
    # Patterns to look for
    patterns = [
        (r'password\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'password'),
        (r'passwd\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'password'),
        (r'pwd\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'password'),
        (r'api_key\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'API key'),
        (r'apikey\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'API key'),
        (r'api_secret\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'API secret'),
        (r'apisecret\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'API secret'),
        (r'secret_key\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'secret key'),
        (r'secretkey\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'secret key'),
        (r'access_key\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'access key'),
        (r'accesskey\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'access key'),
        (r'auth_token\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'auth token'),
        (r'authtoken\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'auth token'),
        (r'token\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'token'),
        (r'username\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'username'),
        (r'user\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'username'),
        (r'private_key\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'private key'),
        (r'privatekey\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'private key'),
        (r'connection_string\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'connection string'),
        (r'connectionstring\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'connection string'),
        (r'conn_str\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'connection string'),
        (r'connstr\s*=\s*["\'](?!.*\$\{)(?!.*\{\{)(?!.*<%=)(?!.*env)(?!.*getenv)(?!.*config)(?!.*settings)(?!.*param)(?!.*variable)(?!.*placeholder)(?!.*example)(?!.*dummy)(?!.*fake)(?!.*test)(?!.*sample)(?!.*template)(?!.*default)(?!.*\[\])(?!.*""\s*\+)(?!.*\'\'\s*\+)[^"\']+["\']', 'connection string'),
    ]
    
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
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
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
                                # Skip comments
                                if line.strip().startswith('#'):
                                    continue
                                    
                                # Check for hardcoded credentials
                                for pattern, credential_type in patterns:
                                    matches = re.search(pattern, line)
                                    if matches:
                                        # Check if line contains exclusion patterns
                                        excluded = False
                                        for exclusion in exclusion_patterns:
                                            if re.search(exclusion, line, re.IGNORECASE):
                                                excluded = True
                                                break
                                                
                                        if not excluded:
                                            results.append((file_path, i + 1, credential_type, line.strip()))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    return results

if __name__ == "__main__":
    # Focus on specific directories
    directories = [
        "analysis-engine-service",
        "common-lib",
        "data-pipeline-service",
        "feature-store-service",
        "ml-integration-service",
        "ml-workbench-service",
        "strategy-execution-engine",
        "api-gateway",
        "trading-gateway-service",
        "ui-service"
    ]
    
    results = find_hardcoded_credentials(directories)
    
    print(f"Found {len(results)} potential hardcoded credentials:")
    
    # Group by file
    files = {}
    for file_path, line_number, credential_type, line in results:
        if file_path not in files:
            files[file_path] = []
        files[file_path].append((line_number, credential_type, line))
    
    # Print results
    for file_path, credentials in files.items():
        print(f"\n{file_path}:")
        for line_number, credential_type, line in credentials:
            print(f"  Line {line_number}: Potential hardcoded {credential_type}")
            print(f"  {line}")
            print("  " + "-" * 50)