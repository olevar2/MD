"""
Test mcp connection module.

This module provides functionality for...
"""

import os
import sys
import json
import subprocess
import requests
from pathlib import Path
import time

def check_mcp_servers():
    """Check if MCP servers are running and accessible"""
    print("Checking MCP servers...")
    
    # Try to find running MCP servers by checking common ports
    common_ports = [9000, 9001, 9002, 9003, 9004, 9005]
    found_servers = []
    
    for port in common_ports:
        try:
            response = requests.get(f"http://localhost:{port}/status", timeout=1)
            if response.status_code == 200:
                found_servers.append(f"Found MCP server on port {port}: {response.json()}")
        except:
            pass
    
    if found_servers:
        print("Found running MCP servers:")
        for server in found_servers:
            print(f"  - {server}")
    else:
        print("No running MCP servers found on common ports.")
    
    # Check if we can find MCP server configurations
    user_home = Path.home()
    vscode_paths = [
        user_home / "AppData" / "Roaming" / "Code - Insiders" / "User" / "settings.json",
        user_home / "AppData" / "Roaming" / "Code" / "User" / "settings.json"
    ]
    
    for path in vscode_paths:
        if path.exists():
            print(f"\nFound VS Code settings at: {path}")
            try:
                with open(path, 'r') as f:
                    settings = json.load(f)
                
                if 'mcp' in settings:
                    print("Found MCP configuration in settings:")
                    servers = settings.get('mcp', {}).get('servers', {})
                    for server_name in servers:
                        print(f"  - {server_name}")
                
                if 'chat.mcp.enabled' in settings:
                    print(f"MCP chat integration enabled: {settings.get('chat.mcp.enabled')}")
                
                if 'chat.mcp.discovery.enabled' in settings:
                    print(f"MCP discovery enabled: {settings.get('chat.mcp.discovery.enabled')}")
            except Exception as e:
                print(f"Error reading settings: {e}")

def check_augment_extension():
    """Check if Augment extension is installed and its version"""
    print("\nChecking Augment extension...")
    
    user_home = Path.home()
    extension_paths = [
        user_home / ".vscode-insiders" / "extensions",
        user_home / ".vscode" / "extensions"
    ]
    
    for path in extension_paths:
        if path.exists():
            print(f"Checking extensions in: {path}")
            augment_extensions = list(path.glob("*augment*"))
            
            if augment_extensions:
                for ext in augment_extensions:
                    print(f"  Found Augment extension: {ext.name}")
            else:
                print("  No Augment extensions found.")

def main():
    """
    Main.
    
    """

    print("MCP Connection Test Script")
    print("=========================")
    
    check_mcp_servers()
    check_augment_extension()
    
    print("\nTest completed. If MCP servers are running but not accessible to Augment,")
    print("try restarting VS Code and ensuring MCP integration is enabled in settings.")

if __name__ == "__main__":
    main()
