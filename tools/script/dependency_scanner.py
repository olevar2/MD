#!/usr/bin/env python3
"""
Dependency Scanner

This script scans Python dependencies for known vulnerabilities.
"""

import os
import sys
import json
import subprocess
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path


def find_requirements_files(directory: str) -> List[str]:
    """
    Find all requirements.txt files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of paths to requirements.txt files
    """
    requirements_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file == "requirements.txt" or file.endswith("-requirements.txt"):
                requirements_files.append(os.path.join(root, file))
    
    return requirements_files


def parse_requirements_file(file_path: str) -> List[str]:
    """
    Parse a requirements.txt file.
    
    Args:
        file_path: Path to requirements.txt file
        
    Returns:
        List of package specifications
    """
    packages = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip options
            if line.startswith('-'):
                continue
            
            # Skip URLs
            if line.startswith('http://') or line.startswith('https://'):
                continue
            
            # Skip editable installs
            if line.startswith('-e'):
                continue
            
            # Add package
            packages.append(line)
    
    return packages


def check_vulnerabilities(packages: List[str]) -> List[Dict[str, Any]]:
    """
    Check packages for known vulnerabilities using safety.
    
    Args:
        packages: List of package specifications
        
    Returns:
        List of vulnerabilities
    """
    # Write packages to temporary file
    temp_file = "temp_requirements.txt"
    with open(temp_file, 'w') as f:
        for package in packages:
            f.write(f"{package}\n")
    
    try:
        # Run safety check
        result = subprocess.run(
            ["safety", "check", "--json", "-r", temp_file],
            capture_output=True,
            text=True
        )
        
        # Parse JSON output
        if result.stdout:
            try:
                vulnerabilities = json.loads(result.stdout)
                return vulnerabilities
            except json.JSONDecodeError:
                print(f"Error parsing safety output: {result.stdout}")
                return []
        
        return []
    
    except Exception as e:
        print(f"Error running safety check: {str(e)}")
        return []
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def check_outdated_packages(packages: List[str]) -> List[Dict[str, Any]]:
    """
    Check for outdated packages.
    
    Args:
        packages: List of package specifications
        
    Returns:
        List of outdated packages
    """
    # Write packages to temporary file
    temp_file = "temp_requirements.txt"
    with open(temp_file, 'w') as f:
        for package in packages:
            f.write(f"{package}\n")
    
    try:
        # Run pip list --outdated
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True
        )
        
        # Parse JSON output
        if result.stdout:
            try:
                outdated = json.loads(result.stdout)
                return outdated
            except json.JSONDecodeError:
                print(f"Error parsing pip output: {result.stdout}")
                return []
        
        return []
    
    except Exception as e:
        print(f"Error checking outdated packages: {str(e)}")
        return []
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    """
    Main function.
    """
    # Check if safety is installed
    try:
        subprocess.run(["safety", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: safety is not installed. Please install it with 'pip install safety'.")
        sys.exit(1)
    
    # Get directory to scan
    directory = "."
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    # Find requirements files
    requirements_files = find_requirements_files(directory)
    
    if not requirements_files:
        print(f"No requirements.txt files found in {directory}")
        sys.exit(0)
    
    print(f"Found {len(requirements_files)} requirements files:")
    for file in requirements_files:
        print(f"  - {file}")
    
    # Parse requirements files
    all_packages = []
    for file in requirements_files:
        packages = parse_requirements_file(file)
        all_packages.extend(packages)
    
    # Remove duplicates
    all_packages = list(set(all_packages))
    
    print(f"\nFound {len(all_packages)} unique packages")
    
    # Check vulnerabilities
    print("\nChecking for vulnerabilities...")
    vulnerabilities = check_vulnerabilities(all_packages)
    
    if vulnerabilities:
        print(f"\nFound {len(vulnerabilities)} vulnerabilities:")
        for vuln in vulnerabilities:
            package = vuln[0]
            affected_version = vuln[1]
            fixed_version = vuln[2]
            vulnerability_id = vuln[3]
            vulnerability_details = vuln[4]
            
            print(f"\n  Package: {package}")
            print(f"  Affected version: {affected_version}")
            print(f"  Fixed version: {fixed_version}")
            print(f"  Vulnerability ID: {vulnerability_id}")
            print(f"  Details: {vulnerability_details}")
    else:
        print("\nNo vulnerabilities found")
    
    # Check outdated packages
    print("\nChecking for outdated packages...")
    outdated = check_outdated_packages(all_packages)
    
    if outdated:
        print(f"\nFound {len(outdated)} outdated packages:")
        for pkg in outdated:
            print(f"  {pkg['name']} {pkg['version']} -> {pkg['latest_version']}")
    else:
        print("\nNo outdated packages found")


if __name__ == "__main__":
    main()