#!/usr/bin/env python3
"""
Check for command injection vulnerabilities in the security analysis.
"""

import json
import sys

def check_command_injection(file_path):
    """
    Check for command injection vulnerabilities in the security analysis.
    
    Args:
        file_path: Path to the security analysis JSON file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'vulnerabilities' in data and 'command_injection' in data['vulnerabilities']:
        vulnerabilities = data['vulnerabilities']['command_injection']
        
        print(f"Found {len(vulnerabilities)} command injection vulnerabilities")
        
        if vulnerabilities:
            print("\nCommand Injection Vulnerabilities:")
            for i, vuln in enumerate(vulnerabilities[:10]):  # Show first 10
                print(f"{i+1}. {vuln.get('file', 'Unknown')}: {vuln.get('description', 'No description')}")
                print(f"   Line: {vuln.get('line', 'Unknown')}")
                print(f"   Severity: {vuln.get('severity', 'Unknown')}")
                print()
            
            if len(vulnerabilities) > 10:
                print(f"... and {len(vulnerabilities) - 10} more vulnerabilities")
    else:
        print("No command injection vulnerabilities found")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_command_injection.py <security_analysis_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    check_command_injection(file_path)