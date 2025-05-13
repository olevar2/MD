"""
Update vscode settings module.

This module provides functionality for...
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import shutil

def backup_settings_file(settings_path):
    """Create a backup of the settings file"""
    backup_path = str(settings_path) + ".backup"
    try:
        shutil.copy2(settings_path, backup_path)
        print(f"Created backup of settings file at: {backup_path}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def update_settings_file(settings_path):
    """Update the VS Code settings file to enable MCP integration"""
    if not settings_path.exists():
        print(f"Settings file not found: {settings_path}")
        return False
    
    # Create a backup first
    if not backup_settings_file(settings_path):
        print("Skipping update due to backup failure.")
        return False
    
    try:
        # Read the current settings
        with open(settings_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
            # Try to parse the JSON
            try:
                settings = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error parsing settings file: {e}")
                print("Will attempt to fix common JSON issues and retry...")
                
                # Try to fix common JSON issues
                content = content.replace("'", '"')  # Replace single quotes with double quotes
                content = content.replace(",\n}", "\n}")  # Remove trailing commas
                content = content.replace(",\n]", "\n]")  # Remove trailing commas in arrays
                
                try:
                    settings = json.loads(content)
                    print("Successfully fixed JSON format issues.")
                except json.JSONDecodeError as e:
                    print(f"Still unable to parse settings file after fixes: {e}")
                    return False
        
        # Update MCP-related settings
        settings["chat.mcp.enabled"] = True
        settings["chat.mcp.discovery.enabled"] = True
        
        # Ensure the mcp section exists
        if "mcp" not in settings:
            print("Adding mcp configuration section...")
            settings["mcp"] = {"servers": {}}
        
        # Write the updated settings back to the file
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        
        print(f"Successfully updated settings file: {settings_path}")
        print("Updated the following settings:")
        print("  - chat.mcp.enabled = true")
        print("  - chat.mcp.discovery.enabled = true")
        print("  - Ensured mcp configuration section exists")
        
        return True
    except Exception as e:
        print(f"Error updating settings file: {e}")
        return False

def main():
    """
    Main.
    
    """

    print("VS Code Settings Update for MCP Integration")
    print("==========================================")
    
    # Paths to VS Code settings files
    settings_paths = [
        Path.home() / "AppData" / "Roaming" / "Code - Insiders" / "User" / "settings.json",
        Path.home() / "AppData" / "Roaming" / "Code" / "User" / "settings.json"
    ]
    
    # Update each settings file
    for settings_path in settings_paths:
        if settings_path.exists():
            print(f"\nUpdating settings file: {settings_path}")
            update_settings_file(settings_path)
        else:
            print(f"\nSettings file not found: {settings_path}")
    
    print("\nUpdate completed.")
    print("Please restart VS Code completely for the changes to take effect.")
    print("After restarting, check if the MCP symbol appears when using Augment.")

if __name__ == "__main__":
    main()
