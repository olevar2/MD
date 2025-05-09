#!/usr/bin/env python3
"""
Script to make all validation tools executable.

This script:
1. Finds all Python scripts in the tools/linting directory
2. Makes them executable (chmod +x)
3. Ensures they have the proper shebang line
"""

import os
import sys
import stat
from pathlib import Path


def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/linting
    return Path(__file__).parent.parent.parent


def make_executable(file_path: Path) -> None:
    """
    Make a file executable.
    
    Args:
        file_path: Path to the file to make executable
    """
    # Get current permissions
    current_permissions = os.stat(file_path).st_mode
    
    # Add executable bit for user, group, and others
    new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    
    # Set new permissions
    os.chmod(file_path, new_permissions)
    
    print(f"Made {file_path} executable")


def ensure_shebang(file_path: Path) -> None:
    """
    Ensure a file has the proper shebang line.
    
    Args:
        file_path: Path to the file to check
    """
    with open(file_path, "r") as f:
        content = f.read()
    
    # Check if the file has a shebang line
    if not content.startswith("#!/usr/bin/env python3"):
        # Add shebang line if missing
        with open(file_path, "w") as f:
            f.write("#!/usr/bin/env python3\n" + content)
        
        print(f"Added shebang line to {file_path}")


def main():
    """Main function."""
    repo_root = get_repo_root()
    tools_dir = repo_root / "tools" / "linting"
    
    # Find all Python scripts in the tools/linting directory
    python_scripts = list(tools_dir.glob("*.py"))
    
    print(f"Found {len(python_scripts)} Python scripts in {tools_dir}")
    
    # Make each script executable and ensure it has a shebang line
    for script in python_scripts:
        ensure_shebang(script)
        make_executable(script)
    
    print(f"Made {len(python_scripts)} scripts executable")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
