#!/usr/bin/env python3
"""
Find potential SQL injection vulnerabilities in Python files.
"""

import os
import re
from typing import List, Tuple

def find_sql_injection_vulnerabilities(directories: List[str]) -> List[Tuple[str, int, str, str]]:
    """
    Find potential SQL injection vulnerabilities in Python files.
    
    Args:
        directories: List of directories to search
    
    Returns:
        List of tuples (file_path, line_number, vulnerability_type, line)
    """
    results = []
    
    # Patterns to look for
    patterns = [
        (r'execute\(\s*f["\']', 'Potential SQL injection in execute with f-string'),
        (r'execute\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in execute with string formatting'),
        (r'execute\(\s*["\'][^"\']*%s', 'Potential SQL injection in execute with string formatting'),
        (r'execute\(\s*["\'][^"\']*\+', 'Potential SQL injection in execute with string concatenation'),
        (r'executemany\(\s*f["\']', 'Potential SQL injection in executemany with f-string'),
        (r'executemany\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in executemany with string formatting'),
        (r'executemany\(\s*["\'][^"\']*%s', 'Potential SQL injection in executemany with string formatting'),
        (r'executemany\(\s*["\'][^"\']*\+', 'Potential SQL injection in executemany with string concatenation'),
        (r'raw_connection\.cursor\(\)\.execute\(\s*f["\']', 'Potential SQL injection in raw_connection.cursor().execute with f-string'),
        (r'raw_connection\.cursor\(\)\.execute\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in raw_connection.cursor().execute with string formatting'),
        (r'raw_connection\.cursor\(\)\.execute\(\s*["\'][^"\']*%s', 'Potential SQL injection in raw_connection.cursor().execute with string formatting'),
        (r'raw_connection\.cursor\(\)\.execute\(\s*["\'][^"\']*\+', 'Potential SQL injection in raw_connection.cursor().execute with string concatenation'),
        (r'text\(\s*f["\']', 'Potential SQL injection in text with f-string'),
        (r'text\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in text with string formatting'),
        (r'text\(\s*["\'][^"\']*%s', 'Potential SQL injection in text with string formatting'),
        (r'text\(\s*["\'][^"\']*\+', 'Potential SQL injection in text with string concatenation'),
        (r'query\(\s*f["\']', 'Potential SQL injection in query with f-string'),
        (r'query\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in query with string formatting'),
        (r'query\(\s*["\'][^"\']*%s', 'Potential SQL injection in query with string formatting'),
        (r'query\(\s*["\'][^"\']*\+', 'Potential SQL injection in query with string concatenation'),
        (r'filter\(\s*f["\']', 'Potential SQL injection in filter with f-string'),
        (r'filter\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in filter with string formatting'),
        (r'filter\(\s*["\'][^"\']*%s', 'Potential SQL injection in filter with string formatting'),
        (r'filter\(\s*["\'][^"\']*\+', 'Potential SQL injection in filter with string concatenation'),
        (r'where\(\s*f["\']', 'Potential SQL injection in where with f-string'),
        (r'where\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in where with string formatting'),
        (r'where\(\s*["\'][^"\']*%s', 'Potential SQL injection in where with string formatting'),
        (r'where\(\s*["\'][^"\']*\+', 'Potential SQL injection in where with string concatenation'),
        (r'having\(\s*f["\']', 'Potential SQL injection in having with f-string'),
        (r'having\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in having with string formatting'),
        (r'having\(\s*["\'][^"\']*%s', 'Potential SQL injection in having with string formatting'),
        (r'having\(\s*["\'][^"\']*\+', 'Potential SQL injection in having with string concatenation'),
        (r'select\(\s*f["\']', 'Potential SQL injection in select with f-string'),
        (r'select\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in select with string formatting'),
        (r'select\(\s*["\'][^"\']*%s', 'Potential SQL injection in select with string formatting'),
        (r'select\(\s*["\'][^"\']*\+', 'Potential SQL injection in select with string concatenation'),
        (r'from_\(\s*f["\']', 'Potential SQL injection in from_ with f-string'),
        (r'from_\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in from_ with string formatting'),
        (r'from_\(\s*["\'][^"\']*%s', 'Potential SQL injection in from_ with string formatting'),
        (r'from_\(\s*["\'][^"\']*\+', 'Potential SQL injection in from_ with string concatenation'),
        (r'join\(\s*f["\']', 'Potential SQL injection in join with f-string'),
        (r'join\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in join with string formatting'),
        (r'join\(\s*["\'][^"\']*%s', 'Potential SQL injection in join with string formatting'),
        (r'join\(\s*["\'][^"\']*\+', 'Potential SQL injection in join with string concatenation'),
        (r'order_by\(\s*f["\']', 'Potential SQL injection in order_by with f-string'),
        (r'order_by\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in order_by with string formatting'),
        (r'order_by\(\s*["\'][^"\']*%s', 'Potential SQL injection in order_by with string formatting'),
        (r'order_by\(\s*["\'][^"\']*\+', 'Potential SQL injection in order_by with string concatenation'),
        (r'group_by\(\s*f["\']', 'Potential SQL injection in group_by with f-string'),
        (r'group_by\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in group_by with string formatting'),
        (r'group_by\(\s*["\'][^"\']*%s', 'Potential SQL injection in group_by with string formatting'),
        (r'group_by\(\s*["\'][^"\']*\+', 'Potential SQL injection in group_by with string concatenation'),
        (r'limit\(\s*f["\']', 'Potential SQL injection in limit with f-string'),
        (r'limit\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in limit with string formatting'),
        (r'limit\(\s*["\'][^"\']*%s', 'Potential SQL injection in limit with string formatting'),
        (r'limit\(\s*["\'][^"\']*\+', 'Potential SQL injection in limit with string concatenation'),
        (r'offset\(\s*f["\']', 'Potential SQL injection in offset with f-string'),
        (r'offset\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in offset with string formatting'),
        (r'offset\(\s*["\'][^"\']*%s', 'Potential SQL injection in offset with string formatting'),
        (r'offset\(\s*["\'][^"\']*\+', 'Potential SQL injection in offset with string concatenation'),
        (r'insert\(\s*f["\']', 'Potential SQL injection in insert with f-string'),
        (r'insert\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in insert with string formatting'),
        (r'insert\(\s*["\'][^"\']*%s', 'Potential SQL injection in insert with string formatting'),
        (r'insert\(\s*["\'][^"\']*\+', 'Potential SQL injection in insert with string concatenation'),
        (r'update\(\s*f["\']', 'Potential SQL injection in update with f-string'),
        (r'update\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in update with string formatting'),
        (r'update\(\s*["\'][^"\']*%s', 'Potential SQL injection in update with string formatting'),
        (r'update\(\s*["\'][^"\']*\+', 'Potential SQL injection in update with string concatenation'),
        (r'delete\(\s*f["\']', 'Potential SQL injection in delete with f-string'),
        (r'delete\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in delete with string formatting'),
        (r'delete\(\s*["\'][^"\']*%s', 'Potential SQL injection in delete with string formatting'),
        (r'delete\(\s*["\'][^"\']*\+', 'Potential SQL injection in delete with string concatenation'),
        (r'raw\(\s*f["\']', 'Potential SQL injection in raw with f-string'),
        (r'raw\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in raw with string formatting'),
        (r'raw\(\s*["\'][^"\']*%s', 'Potential SQL injection in raw with string formatting'),
        (r'raw\(\s*["\'][^"\']*\+', 'Potential SQL injection in raw with string concatenation'),
        (r'sql\(\s*f["\']', 'Potential SQL injection in sql with f-string'),
        (r'sql\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in sql with string formatting'),
        (r'sql\(\s*["\'][^"\']*%s', 'Potential SQL injection in sql with string formatting'),
        (r'sql\(\s*["\'][^"\']*\+', 'Potential SQL injection in sql with string concatenation'),
        (r'exec_sql\(\s*f["\']', 'Potential SQL injection in exec_sql with f-string'),
        (r'exec_sql\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in exec_sql with string formatting'),
        (r'exec_sql\(\s*["\'][^"\']*%s', 'Potential SQL injection in exec_sql with string formatting'),
        (r'exec_sql\(\s*["\'][^"\']*\+', 'Potential SQL injection in exec_sql with string concatenation'),
        (r'execute_sql\(\s*f["\']', 'Potential SQL injection in execute_sql with f-string'),
        (r'execute_sql\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in execute_sql with string formatting'),
        (r'execute_sql\(\s*["\'][^"\']*%s', 'Potential SQL injection in execute_sql with string formatting'),
        (r'execute_sql\(\s*["\'][^"\']*\+', 'Potential SQL injection in execute_sql with string concatenation'),
        (r'run_sql\(\s*f["\']', 'Potential SQL injection in run_sql with f-string'),
        (r'run_sql\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in run_sql with string formatting'),
        (r'run_sql\(\s*["\'][^"\']*%s', 'Potential SQL injection in run_sql with string formatting'),
        (r'run_sql\(\s*["\'][^"\']*\+', 'Potential SQL injection in run_sql with string concatenation'),
        (r'query_sql\(\s*f["\']', 'Potential SQL injection in query_sql with f-string'),
        (r'query_sql\(\s*["\'][^"\']*\{.*\}', 'Potential SQL injection in query_sql with string formatting'),
        (r'query_sql\(\s*["\'][^"\']*%s', 'Potential SQL injection in query_sql with string formatting'),
        (r'query_sql\(\s*["\'][^"\']*\+', 'Potential SQL injection in query_sql with string concatenation'),
        (r'SELECT.*FROM.*WHERE.*\+', 'Potential SQL injection in raw SQL query with string concatenation'),
        (r'SELECT.*FROM.*WHERE.*\{.*\}', 'Potential SQL injection in raw SQL query with string formatting'),
        (r'SELECT.*FROM.*WHERE.*%s', 'Potential SQL injection in raw SQL query with string formatting'),
        (r'INSERT.*INTO.*VALUES.*\+', 'Potential SQL injection in raw SQL query with string concatenation'),
        (r'INSERT.*INTO.*VALUES.*\{.*\}', 'Potential SQL injection in raw SQL query with string formatting'),
        (r'INSERT.*INTO.*VALUES.*%s', 'Potential SQL injection in raw SQL query with string formatting'),
        (r'UPDATE.*SET.*WHERE.*\+', 'Potential SQL injection in raw SQL query with string concatenation'),
        (r'UPDATE.*SET.*WHERE.*\{.*\}', 'Potential SQL injection in raw SQL query with string formatting'),
        (r'UPDATE.*SET.*WHERE.*%s', 'Potential SQL injection in raw SQL query with string formatting'),
        (r'DELETE.*FROM.*WHERE.*\+', 'Potential SQL injection in raw SQL query with string concatenation'),
        (r'DELETE.*FROM.*WHERE.*\{.*\}', 'Potential SQL injection in raw SQL query with string formatting'),
        (r'DELETE.*FROM.*WHERE.*%s', 'Potential SQL injection in raw SQL query with string formatting'),
    ]
    
    # Exclusion patterns
    exclusion_patterns = [
        r'paramstyle',
        r'params=',
        r'parameters=',
        r'args=',
        r'arguments=',
        r'bind=',
        r'binds=',
        r'bindparams=',
        r'bindvars=',
        r'values=',
        r'placeholders=',
        r'prepared=',
        r'sanitize',
        r'escape',
        r'quote',
        r'safe',
        r'validate',
        r'check',
        r'verify',
        r'clean',
        r'filter',
        r'sanitize_sql',
        r'escape_sql',
        r'quote_sql',
        r'safe_sql',
        r'validate_sql',
        r'check_sql',
        r'verify_sql',
        r'clean_sql',
        r'filter_sql',
        r'sanitize_query',
        r'escape_query',
        r'quote_query',
        r'safe_query',
        r'validate_query',
        r'check_query',
        r'verify_query',
        r'clean_query',
        r'filter_query',
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
                            content = f.read()
                            
                            for pattern, vulnerability_type in patterns:
                                for match in re.finditer(pattern, content):
                                    # Check if match contains exclusion patterns
                                    excluded = False
                                    for exclusion in exclusion_patterns:
                                        if re.search(exclusion, content[max(0, match.start() - 50):min(len(content), match.end() + 50)], re.IGNORECASE):
                                            excluded = True
                                            break
                                            
                                    if not excluded:
                                        line_num = content[:match.start()].count('\n') + 1
                                        line = content.splitlines()[line_num - 1]
                                        results.append((file_path, line_num, vulnerability_type, line.strip()))
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
        "ui-service",
        "data-management-service"
    ]
    
    results = find_sql_injection_vulnerabilities(directories)
    
    print(f"Found {len(results)} potential SQL injection vulnerabilities:")
    
    # Group by file
    files = {}
    for file_path, line_number, vulnerability_type, line in results:
        if file_path not in files:
            files[file_path] = []
        files[file_path].append((line_number, vulnerability_type, line))
    
    # Print results
    for file_path, vulnerabilities in files.items():
        print(f"\n{file_path}:")
        for line_number, vulnerability_type, line in vulnerabilities:
            print(f"  Line {line_number}: {vulnerability_type}")
            print(f"  {line}")
            print("  " + "-" * 50)