#!/usr/bin/env python3
"""
Check for insecure deserialization vulnerabilities in the security analysis.
"""

import json
import sys

def check_insecure_deserialization(file_path):
    """
    Check for insecure deserialization vulnerabilities in the security analysis.
    
    Args:
        file_path: Path to the security analysis JSON file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'vulnerabilities' in data and 'insecure_deserialization' in data['vulnerabilities']:
        vulnerabilities = data['vulnerabilities']['insecure_deserialization']
        
        print(f"Found {len(vulnerabilities)} insecure deserialization vulnerabilities")
        
        if vulnerabilities:
            print("\nInsecure Deserialization Vulnerabilities:")
            for i, vuln in enumerate(vulnerabilities[:10]):  # Show first 10
                print(f"{i+1}. {vuln.get('file', 'Unknown')}: {vuln.get('description', 'No description')}")
                print(f"   Line: {vuln.get('line', 'Unknown')}")
                print(f"   Severity: {vuln.get('severity', 'Unknown')}")
                print()
            
            if len(vulnerabilities) > 10:
                print(f"... and {len(vulnerabilities) - 10} more vulnerabilities")
    else:
        print("No insecure deserialization vulnerabilities found")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_insecure_deserialization.py <security_analysis_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    check_insecure_deserialization(file_path)