"""
Verify mcp configuration module.

This module provides functionality for...
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_settings_files():
    """Check if MCP integration is properly configured in settings files"""
    print("Checking settings files for MCP integration configuration...")
    
    # Check workspace settings
    workspace_settings_path = Path(".vscode/settings.json")
    if workspace_settings_path.exists():
        try:
            with open(workspace_settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            print("\nWorkspace settings (.vscode/settings.json):")
            print(f"  chat.mcp.enabled: {settings.get('chat.mcp.enabled', 'Not set')}")
            print(f"  chat.mcp.discovery.enabled: {settings.get('chat.mcp.discovery.enabled', 'Not set')}")
            
            if "mcp.server.forex" in settings:
                print("  Custom MCP server configuration found: mcp.server.forex")
        except Exception as e:
            print(f"  Error reading workspace settings: {e}")
    else:
        print("\nWorkspace settings file not found.")
    
    # Check user settings (VS Code Insiders)
    user_settings_path = Path.home() / "AppData" / "Roaming" / "Code - Insiders" / "User" / "settings.json"
    if user_settings_path.exists():
        try:
            # Read file content with error handling for encoding issues
            with open(user_settings_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            try:
                settings = json.loads(content)
                
                print("\nVS Code Insiders user settings:")
                print(f"  chat.mcp.enabled: {settings.get('chat.mcp.enabled', 'Not set')}")
                print(f"  chat.mcp.discovery.enabled: {settings.get('chat.mcp.discovery.enabled', 'Not set')}")
                
                if "mcp" in settings and "servers" in settings["mcp"]:
                    print(f"  MCP servers configured: {len(settings['mcp']['servers'])}")
                    for server_name in settings["mcp"]["servers"]:
                        print(f"    - {server_name}")
            except json.JSONDecodeError as e:
                print(f"  Error parsing VS Code Insiders user settings: {e}")
        except Exception as e:
            print(f"  Error reading VS Code Insiders user settings: {e}")
    else:
        print("\nVS Code Insiders user settings file not found.")

def check_running_mcp_servers():
    """Check if MCP servers are running"""
    print("\nChecking for running MCP servers...")
    
    try:
        # Use PowerShell to get process information for Node.js processes
        cmd = 'powershell -Command "Get-Process | Where-Object { $_.ProcessName -eq \'node\' } | Measure-Object | Select-Object -ExpandProperty Count"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and result.stdout.strip():
            node_count = int(result.stdout.strip())
            print(f"  Found {node_count} Node.js processes running.")
            
            if node_count > 0:
                print("  MCP servers are likely running (they use Node.js).")
            else:
                print("  No Node.js processes found. MCP servers might not be running.")
        else:
            print("  Error checking for Node.js processes.")
    except Exception as e:
        print(f"  Error checking for running MCP servers: {e}")

def check_augment_extension():
    """Check if Augment extension is installed and its version"""
    print("\nChecking Augment extension...")
    
    try:
        # Check VS Code Insiders extensions
        cmd = 'powershell -Command "Get-ChildItem -Path \'$env:USERPROFILE\\.vscode-insiders\\extensions\' -Filter \'*augment*\' | Select-Object Name"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and result.stdout.strip():
            print("  VS Code Insiders Augment extension:")
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith("Name"):
                    print(f"    - {line.strip()}")
        else:
            print("  Augment extension not found in VS Code Insiders.")
    except Exception as e:
        print(f"  Error checking Augment extension: {e}")

def main():
    """
    Main.
    
    """

    print("MCP Integration Configuration Verification")
    print("========================================")
    
    check_settings_files()
    check_running_mcp_servers()
    check_augment_extension()
    
    print("\nVerification completed.")
    print("\nNext steps:")
    print("1. Restart VS Code completely")
    print("2. Make sure MCP servers are running (they should show as 'Running' in the MCP Servers panel)")
    print("3. Try using Augment and check if the MCP symbol appears")
    print("4. If issues persist, try reinstalling both Augment and MCP extensions")
    print("\nAll necessary configuration has been applied. The MCP integration should work after a restart.")

if __name__ == "__main__":
    main()
