"""
Check mcp processes module.

This module provides functionality for...
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_running_processes():
    """Check for running processes related to MCP"""
    print("Checking for running MCP-related processes...")
    
    try:
        # Use PowerShell to get process information
        cmd = 'powershell -Command "Get-Process | Where-Object { $_.ProcessName -like \'*node*\' -or $_.ProcessName -like \'*npm*\' -or $_.ProcessName -like \'*npx*\' } | Select-Object ProcessName, Id, Path | ConvertTo-Json"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and result.stdout.strip():
            try:
                processes = json.loads(result.stdout)
                # Handle case where only one process is returned (not in an array)
                if not isinstance(processes, list):
                    processes = [processes]
                
                print(f"Found {len(processes)} potentially related processes:")
                for proc in processes:
                    print(f"  - {proc.get('ProcessName')} (PID: {proc.get('Id')})")
                    if proc.get('Path'):
                        print(f"    Path: {proc.get('Path')}")
            except json.JSONDecodeError:
                print("Error parsing process list. Raw output:")
                print(result.stdout)
        else:
            print("No relevant processes found or error running command.")
            if result.stderr:
                print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error checking processes: {e}")

def check_network_connections():
    """Check for network connections that might be related to MCP servers"""
    print("\nChecking for network connections that might be related to MCP servers...")
    
    try:
        # Use PowerShell to get network connection information
        cmd = 'powershell -Command "Get-NetTCPConnection | Where-Object { $_.State -eq \'Listen\' -and $_.LocalPort -ge 3000 -and $_.LocalPort -le 10000 } | Select-Object LocalAddress, LocalPort, State | Sort-Object LocalPort | ConvertTo-Json"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and result.stdout.strip():
            try:
                connections = json.loads(result.stdout)
                # Handle case where only one connection is returned (not in an array)
                if not isinstance(connections, list):
                    connections = [connections]
                
                print(f"Found {len(connections)} potentially relevant listening ports:")
                for conn in connections:
                    print(f"  - {conn.get('LocalAddress')}:{conn.get('LocalPort')} ({conn.get('State')})")
            except json.JSONDecodeError:
                print("Error parsing connection list. Raw output:")
                print(result.stdout)
        else:
            print("No relevant network connections found or error running command.")
            if result.stderr:
                print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error checking network connections: {e}")

def main():
    """
    Main.
    
    """

    print("MCP Process and Network Check")
    print("=============================")
    
    check_running_processes()
    check_network_connections()
    
    print("\nCheck completed. If you see Node.js processes and listening ports in the 3000-10000 range,")
    print("those might be related to MCP servers. The Augment extension should connect to these servers.")

if __name__ == "__main__":
    main()
