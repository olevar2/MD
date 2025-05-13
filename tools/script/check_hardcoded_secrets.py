#!/usr/bin/env python3
"""
Check for hardcoded secrets in the security analysis.
"""

import json
import sys

def check_hardcoded_secrets(file_path):
    """
    Check for hardcoded secrets in the security analysis.
    
    Args:
        file_path: Path to the security analysis JSON file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'vulnerabilities' in data and 'hardcoded_secrets' in data['vulnerabilities']:
        vulnerabilities = data['vulnerabilities']['hardcoded_secrets']
        
        print(f"Found {len(vulnerabilities)} hardcoded secrets")
        
        if vulnerabilities:
            print("\nHardcoded Secrets:")
            for i, vuln in enumerate(vulnerabilities):
                print(f"{i+1}. {vuln.get('file', 'Unknown')}: {vuln.get('description', 'No description')}")
                print(f"   Line: {vuln.get('line', 'Unknown')}")
                print(f"   Severity: {vuln.get('severity', 'Unknown')}")
                print()
    else:
        print("No hardcoded secrets found")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_hardcoded_secrets.py <security_analysis_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    check_hardcoded_secrets(file_path)