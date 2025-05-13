#!/usr/bin/env python3
"""
Fix Feature Store Structure

This script resolves the circular dependency between feature-store-service and feature_store_service
by consolidating the directory structure and standardizing imports.
"""

import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple

# Configure paths
PROJECT_ROOT = Path("D:/MD/forex_trading_platform")
FEATURE_STORE_DIR = PROJECT_ROOT / "feature-store-service"
FEATURE_STORE_PACKAGE = FEATURE_STORE_DIR / "feature_store_service"
STANDALONE_FEATURE_STORE = PROJECT_ROOT / "feature_store_service"

def analyze_structure():
    """Analyze the current structure of feature store directories."""
    print("Analyzing feature store directory structure...", flush=True)
    
    # Check if both directories exist
    feature_store_dir_exists = FEATURE_STORE_DIR.exists()
    standalone_feature_store_exists = STANDALONE_FEATURE_STORE.exists()
    
    print(f"feature-store-service directory exists: {feature_store_dir_exists}", flush=True)
    print(f"feature_store_service directory exists: {standalone_feature_store_exists}", flush=True)
    
    # Check contents
    if feature_store_dir_exists:
        feature_store_files = list(FEATURE_STORE_DIR.glob("*"))
        print(f"feature-store-service contains {len(feature_store_files)} items:", flush=True)
        for item in feature_store_files[:10]:  # Show first 10
            print(f"  {item.name}", flush=True)
        if len(feature_store_files) > 10:
            print(f"  ... and {len(feature_store_files) - 10} more", flush=True)
    
    if standalone_feature_store_exists:
        standalone_files = list(STANDALONE_FEATURE_STORE.glob("*"))
        print(f"feature_store_service contains {len(standalone_files)} items:", flush=True)
        for item in standalone_files:
            print(f"  {item.name}", flush=True)
    
    # Check if feature_store_service exists as a package inside feature-store-service
    nested_package_exists = (FEATURE_STORE_DIR / "feature_store_service").exists()
    print(f"Nested package exists: {nested_package_exists}", flush=True)
    
    if nested_package_exists:
        nested_files = list((FEATURE_STORE_DIR / "feature_store_service").glob("*"))
        print(f"Nested package contains {len(nested_files)} items:", flush=True)
        for item in nested_files[:10]:  # Show first 10
            print(f"  {item.name}", flush=True)
        if len(nested_files) > 10:
            print(f"  ... and {len(nested_files) - 10} more", flush=True)
    
    return {
        "feature_store_dir_exists": feature_store_dir_exists,
        "standalone_feature_store_exists": standalone_feature_store_exists,
        "nested_package_exists": nested_package_exists
    }

def merge_directories():
    """Merge the standalone feature_store_service into the nested package."""
    if not STANDALONE_FEATURE_STORE.exists():
        print("Standalone feature_store_service directory doesn't exist. Nothing to merge.", flush=True)
        return
    
    if not (FEATURE_STORE_DIR / "feature_store_service").exists():
        print("Nested feature_store_service package doesn't exist. Cannot merge.", flush=True)
        return
    
    print("Merging standalone feature_store_service into nested package...", flush=True)
    
    # Copy all files from standalone to nested
    for item in STANDALONE_FEATURE_STORE.glob("*"):
        if item.is_file():
            print(f"Copying file {item.name} to nested package", flush=True)
            shutil.copy2(item, FEATURE_STORE_DIR / "feature_store_service")
        elif item.is_dir():
            print(f"Copying directory {item.name} to nested package", flush=True)
            if (FEATURE_STORE_DIR / "feature_store_service" / item.name).exists():
                # Merge directory contents
                for subitem in item.glob("**/*"):
                    if subitem.is_file():
                        rel_path = subitem.relative_to(item)
                        target_path = FEATURE_STORE_DIR / "feature_store_service" / item.name / rel_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(subitem, target_path)
            else:
                # Copy entire directory
                shutil.copytree(item, FEATURE_STORE_DIR / "feature_store_service" / item.name)
    
    print("Merge complete.", flush=True)

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)
    return python_files

def fix_imports():
    """Fix imports in Python files to use consistent naming."""
    print("Fixing imports in Python files...", flush=True)
    
    # Find all Python files in the project
    python_files = find_python_files(PROJECT_ROOT)
    print(f"Found {len(python_files)} Python files", flush=True)
    
    # Patterns to search for
    import_patterns = [
        r'from\s+(feature[-_]store[-_]service)\.([^\s]+)\s+import\s+([^\n]+)',
        r'import\s+(feature[-_]store[-_]service)\.([^\s]+)(?:\s+as\s+([^\s]+))?'
    ]
    
    # Fix imports in each file
    total_replacements = 0
    files_modified = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes = []
            
            # Fix imports
            for pattern in import_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    module_name = match.group(1)
                    
                    # Standardize to snake_case
                    if module_name != "feature_store_service":
                        replacement = match.group(0).replace(module_name, "feature_store_service")
                        content = content.replace(match.group(0), replacement)
                        changes.append(f"Changed '{match.group(0)}' to '{replacement}'")
            
            # Write changes back to file if content was modified
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                files_modified += 1
                total_replacements += len(changes)
                
                print(f"Fixed {len(changes)} imports in {file_path}", flush=True)
                for change in changes:
                    print(f"  - {change}", flush=True)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}", flush=True)
    
    print(f"\nSummary: Fixed {total_replacements} imports in {files_modified} files", flush=True)

def rename_standalone_directory():
    """Rename the standalone feature_store_service directory to avoid conflicts."""
    if not STANDALONE_FEATURE_STORE.exists():
        print("Standalone feature_store_service directory doesn't exist. Nothing to rename.", flush=True)
        return
    
    backup_dir = PROJECT_ROOT / "feature_store_service_backup"
    
    # Create backup directory if it doesn't exist
    if not backup_dir.exists():
        backup_dir.mkdir()
    
    print(f"Renaming standalone feature_store_service to {backup_dir}", flush=True)
    
    # Move all contents to backup directory
    for item in STANDALONE_FEATURE_STORE.glob("*"):
        if item.is_file():
            shutil.copy2(item, backup_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, backup_dir / item.name, dirs_exist_ok=True)
    
    # Remove the original directory
    shutil.rmtree(STANDALONE_FEATURE_STORE)
    
    print("Rename complete.", flush=True)

def main():
    """Main entry point."""
    print("Starting feature store structure fix...", flush=True)
    
    # Analyze current structure
    structure = analyze_structure()
    
    # Merge directories if both exist
    if structure["standalone_feature_store_exists"] and structure["nested_package_exists"]:
        merge_directories()
    
    # Fix imports
    fix_imports()
    
    # Rename standalone directory
    if structure["standalone_feature_store_exists"]:
        rename_standalone_directory()
    
    print("Feature store structure fix complete.", flush=True)
    
    # Create a log file with all changes
    log_path = PROJECT_ROOT / "tools" / "output" / "feature_store_structure_fix.log"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Feature Store Structure Fix\n")
        f.write("=========================\n\n")
        f.write(f"Timestamp: {os.path.getmtime(log_path)}\n\n")
        f.write("Actions performed:\n")
        
        if structure["standalone_feature_store_exists"] and structure["nested_package_exists"]:
            f.write("- Merged standalone feature_store_service into nested package\n")
        
        f.write("- Fixed imports in Python files\n")
        
        if structure["standalone_feature_store_exists"]:
            f.write("- Renamed standalone feature_store_service to feature_store_service_backup\n")
    
    print(f"Log file created at {log_path}", flush=True)

if __name__ == "__main__":
    main()
