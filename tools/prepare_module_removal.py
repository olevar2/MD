#!/usr/bin/env python
"""
Prepare Module Removal

This script helps prepare for the removal of deprecated modules by:
1. Checking for any remaining usages of deprecated modules
2. Creating a backup of the modules to be removed
3. Generating a removal plan with impact assessment

Usage:
    python prepare_module_removal.py [--check-only] [--backup] [--plan]

Options:
    --check-only    Only check for remaining usages without creating backups or plans
    --backup        Create backups of modules to be removed
    --plan          Generate a removal plan with impact assessment
"""

import os
import sys
import json
import argparse
import datetime
import shutil
import re
from typing import Dict, List, Any, Set, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the deprecation monitor
try:
    from analysis_engine.core.deprecation_monitor import get_usage_report
    CAN_GET_LIVE_REPORT = True
except ImportError:
    CAN_GET_LIVE_REPORT = False


# Define modules to be removed
MODULES_TO_REMOVE = [
    {
        "name": "analysis_engine.core.config",
        "file_path": "analysis_engine/core/config.py",
        "replacement": "analysis_engine.config",
        "migration_guide": "docs/configuration_migration_guide.md"
    },
    {
        "name": "config.config",
        "file_path": "config/config.py",
        "replacement": "analysis_engine.config",
        "migration_guide": "docs/configuration_migration_guide.md"
    },
    {
        "name": "analysis_engine.api.router",
        "file_path": "analysis_engine/api/router.py",
        "replacement": "analysis_engine.api.routes",
        "migration_guide": "docs/api_router_migration_guide.md"
    }
]


def load_report() -> Dict[str, Any]:
    """
    Load the deprecation report.
    
    Returns:
        Dict containing the report data
    """
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "logs",
        "deprecation_report.json"
    )
    
    if not os.path.exists(report_path):
        if CAN_GET_LIVE_REPORT:
            print(f"Report file not found at {report_path}, generating live report")
            return get_usage_report()
        else:
            print(f"Report file not found at {report_path} and cannot generate live report")
            return {
                "generated_at": datetime.datetime.now().isoformat(),
                "total_usages": 0,
                "modules": {}
            }
    
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}")
        return {
            "generated_at": datetime.datetime.now().isoformat(),
            "total_usages": 0,
            "modules": {}
        }


def check_remaining_usages() -> Dict[str, List[Dict[str, Any]]]:
    """
    Check for any remaining usages of deprecated modules.
    
    Returns:
        Dict mapping module names to lists of usage information
    """
    report = load_report()
    remaining_usages = {}
    
    for module in MODULES_TO_REMOVE:
        module_name = module["name"]
        if module_name in report.get("modules", {}):
            usages = report["modules"][module_name].get("usages", [])
            if usages:
                remaining_usages[module_name] = usages
    
    return remaining_usages


def find_import_statements(module_name: str) -> List[Tuple[str, int, str]]:
    """
    Find import statements for a specific module in the codebase.
    
    Args:
        module_name: Name of the module to find imports for
        
    Returns:
        List of tuples (file_path, line_number, line_content)
    """
    results = []
    root_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Patterns to search for
    patterns = [
        re.compile(rf"from\s+{re.escape(module_name)}\s+import"),
        re.compile(rf"import\s+{re.escape(module_name)}")
    ]
    
    # Walk through all Python files
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
                
            file_path = os.path.join(dirpath, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    for i, line in enumerate(lines):
                        for pattern in patterns:
                            if pattern.search(line):
                                results.append((file_path, i + 1, line.strip()))
                                break
            except UnicodeDecodeError:
                # Skip files that can't be decoded as UTF-8
                continue
    
    return results


def create_module_backups() -> None:
    """Create backups of modules to be removed."""
    backup_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "backups",
        f"deprecated_modules_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    os.makedirs(backup_dir, exist_ok=True)
    
    for module in MODULES_TO_REMOVE:
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            module["file_path"]
        )
        
        if os.path.exists(file_path):
            # Create directory structure in backup
            backup_file_path = os.path.join(backup_dir, module["file_path"])
            os.makedirs(os.path.dirname(backup_file_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, backup_file_path)
            print(f"Created backup of {module['file_path']} at {backup_file_path}")
        else:
            print(f"Warning: Module file {module['file_path']} not found")


def generate_removal_plan() -> None:
    """Generate a removal plan with impact assessment."""
    remaining_usages = check_remaining_usages()
    
    plan_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "docs",
        "module_removal_plan.md"
    )
    
    with open(plan_path, 'w') as f:
        f.write("# Deprecated Module Removal Plan\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document outlines the plan for removing deprecated modules from the codebase.\n\n")
        
        f.write("## Modules to Remove\n\n")
        for module in MODULES_TO_REMOVE:
            f.write(f"### {module['name']}\n\n")
            f.write(f"- **File Path:** {module['file_path']}\n")
            f.write(f"- **Replacement:** {module['replacement']}\n")
            f.write(f"- **Migration Guide:** {module['migration_guide']}\n\n")
            
            # Check for remaining usages
            if module['name'] in remaining_usages:
                usages = remaining_usages[module['name']]
                f.write(f"**WARNING: {len(usages)} remaining usages found!**\n\n")
                f.write("These usages must be migrated before removal:\n\n")
                
                for usage in usages:
                    f.write(f"- {usage.get('caller_file', '')}:{usage.get('caller_line', '')} in {usage.get('caller_function', '')}\n")
                
                f.write("\n")
            else:
                f.write("No remaining usages found. Safe to remove.\n\n")
            
            # Find import statements
            imports = find_import_statements(module['name'])
            if imports:
                f.write(f"**Found {len(imports)} import statements that need to be updated:**\n\n")
                
                for file_path, line_number, line_content in imports:
                    rel_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(__file__)))
                    f.write(f"- {rel_path}:{line_number} - `{line_content}`\n")
                
                f.write("\n")
        
        f.write("## Removal Steps\n\n")
        f.write("1. **Verify Migration Completion**\n")
        f.write("   - Run `python tools/deprecation_dashboard.py` to check for any remaining usages\n")
        f.write("   - Ensure all usages have been migrated\n\n")
        
        f.write("2. **Create Backups**\n")
        f.write("   - Run `python tools/prepare_module_removal.py --backup` to create backups\n\n")
        
        f.write("3. **Remove Modules**\n")
        for module in MODULES_TO_REMOVE:
            f.write(f"   - Remove `{module['file_path']}`\n")
        f.write("\n")
        
        f.write("4. **Verify System Functionality**\n")
        f.write("   - Run all tests to ensure system still functions correctly\n")
        f.write("   - Check for any runtime errors related to missing modules\n\n")
        
        f.write("5. **Update Documentation**\n")
        f.write("   - Remove references to deprecated modules from documentation\n")
        f.write("   - Update README.md to remove migration notices\n\n")
        
        f.write("## Rollback Plan\n\n")
        f.write("If issues are encountered after removal:\n\n")
        f.write("1. Restore modules from backups\n")
        f.write("2. Run tests to verify functionality\n")
        f.write("3. Create new migration tickets for any issues discovered\n")
    
    print(f"Generated removal plan at {plan_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare for module removal")
    parser.add_argument("--check-only", action="store_true", help="Only check for remaining usages")
    parser.add_argument("--backup", action="store_true", help="Create backups of modules to be removed")
    parser.add_argument("--plan", action="store_true", help="Generate a removal plan")
    args = parser.parse_args()
    
    # Check for remaining usages
    remaining_usages = check_remaining_usages()
    
    if remaining_usages:
        print("WARNING: Found remaining usages of deprecated modules:")
        for module, usages in remaining_usages.items():
            print(f"- {module}: {len(usages)} usages")
    else:
        print("No remaining usages found. Safe to proceed with removal.")
    
    # Create backups if requested
    if args.backup:
        create_module_backups()
    
    # Generate removal plan if requested
    if args.plan:
        generate_removal_plan()
    
    # If no options specified, show help
    if not (args.check_only or args.backup or args.plan):
        parser.print_help()


if __name__ == "__main__":
    main()
