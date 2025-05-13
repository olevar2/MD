"""
Check augment mcp module.

This module provides functionality for...
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def create_test_file():
    """Create a test file to check if Augment can access it"""
    test_file_path = Path("augment_mcp_test.txt")
    
    try:
        with open(test_file_path, "w") as f:
            f.write("This is a test file for Augment MCP integration.\n")
            f.write(f"Created at: {os.path.abspath(test_file_path)}\n")
        
        print(f"Created test file at: {os.path.abspath(test_file_path)}")
        return os.path.abspath(test_file_path)
    except Exception as e:
        print(f"Error creating test file: {e}")
        return None

def check_vscode_extensions():
    """Check VS Code extensions related to Augment and MCP"""
    print("\nChecking VS Code extensions...")
    
    try:
        # Check VS Code Insiders extensions
        cmd = 'powershell -Command "Get-ChildItem -Path \'$env:USERPROFILE\\.vscode-insiders\\extensions\' | Where-Object { $_.Name -like \'*augment*\' -or $_.Name -like \'*mcp*\' } | Select-Object Name"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and result.stdout.strip():
            print("VS Code Insiders extensions:")
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith("Name"):
                    print(f"  - {line.strip()}")
        else:
            print("No relevant VS Code Insiders extensions found or error running command.")
        
        # Check regular VS Code extensions
        cmd = 'powershell -Command "Get-ChildItem -Path \'$env:USERPROFILE\\.vscode\\extensions\' | Where-Object { $_.Name -like \'*augment*\' -or $_.Name -like \'*mcp*\' } | Select-Object Name"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and result.stdout.strip():
            print("\nRegular VS Code extensions:")
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith("Name"):
                    print(f"  - {line.strip()}")
        else:
            print("No relevant regular VS Code extensions found or error running command.")
    except Exception as e:
        print(f"Error checking VS Code extensions: {e}")

def check_augment_logs():
    """Check for Augment logs that might contain MCP-related information"""
    print("\nChecking for Augment logs...")
    
    log_paths = [
        Path.home() / "AppData" / "Roaming" / "Code - Insiders" / "logs",
        Path.home() / "AppData" / "Roaming" / "Code" / "logs"
    ]
    
    for log_dir in log_paths:
        if log_dir.exists():
            print(f"Checking logs in: {log_dir}")
            
            # Look for Augment or MCP related log files
            log_files = list(log_dir.glob("*augment*.log")) + list(log_dir.glob("*mcp*.log"))
            
            if log_files:
                print(f"Found {len(log_files)} potentially relevant log files:")
                for log_file in log_files:
                    print(f"  - {log_file.name}")
                    
                    # Try to read the last few lines of the most recent log file
                    if log_file == sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]:
                        try:
                            with open(log_file, 'r', errors='replace') as f:
                                lines = f.readlines()
                                last_lines = lines[-10:] if len(lines) >= 10 else lines
                                
                                print("\nLast few lines from most recent log file:")
                                for line in last_lines:
                                    print(f"    {line.strip()}")
                        except Exception as e:
                            print(f"    Error reading log file: {e}")
            else:
                print("No relevant log files found.")
        else:
            print(f"Log directory not found: {log_dir}")

def main():
    """
    Main.
    
    """

    print("Augment MCP Integration Check")
    print("============================")
    
    # Create a test file
    test_file_path = create_test_file()
    
    # Check VS Code extensions
    check_vscode_extensions()
    
    # Check Augment logs
    check_augment_logs()
    
    print("\nCheck completed.")
    print("To fix Augment MCP integration issues:")
    print("1. Make sure both Augment and MCP extensions are installed and up to date")
    print("2. Ensure MCP servers are running (they appear to be based on previous checks)")
    print("3. Check that MCP integration is enabled in VS Code settings")
    print("4. Try restarting VS Code completely")
    print("5. If issues persist, try reinstalling both Augment and MCP extensions")
    
    if test_file_path:
        print(f"\nYou can use the test file at {test_file_path} to check if Augment can access files through MCP.")

if __name__ == "__main__":
    main()
