"""
Update workspace settings module.

This module provides functionality for...
"""

import os
import sys
import json
from pathlib import Path

def update_workspace_settings():
    """Update the workspace settings to ensure MCP integration is properly configured"""
    workspace_settings_path = Path(".vscode/settings.json")
    
    if not workspace_settings_path.exists():
        print(f"Workspace settings file not found: {workspace_settings_path}")
        print("Creating new workspace settings file...")
        
        # Create .vscode directory if it doesn't exist
        os.makedirs(".vscode", exist_ok=True)
        
        # Create new settings file with MCP configuration
        settings = {
            "chat.mcp.enabled": True,
            "chat.mcp.discovery.enabled": True
        }
        
        with open(workspace_settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        
        print(f"Created new workspace settings file: {workspace_settings_path}")
        return True
    
    try:
        # Read the current settings
        with open(workspace_settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Update MCP-related settings
        settings["chat.mcp.enabled"] = True
        settings["chat.mcp.discovery.enabled"] = True
        
        # Write the updated settings back to the file
        with open(workspace_settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        
        print(f"Successfully updated workspace settings file: {workspace_settings_path}")
        print("Updated the following settings:")
        print("  - chat.mcp.enabled = true")
        print("  - chat.mcp.discovery.enabled = true")
        
        return True
    except Exception as e:
        print(f"Error updating workspace settings file: {e}")
        return False

def main():
    """
    Main.
    
    """

    print("Workspace Settings Update for MCP Integration")
    print("===========================================")
    
    update_workspace_settings()
    
    print("\nUpdate completed.")
    print("Please restart VS Code completely for the changes to take effect.")
    print("After restarting, check if the MCP symbol appears when using Augment.")

if __name__ == "__main__":
    main()
