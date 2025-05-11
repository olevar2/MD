#!/usr/bin/env python3
"""
Targeted Naming Convention Fixer

This script fixes specific naming convention issues in the codebase while preserving
the correct naming conventions.
"""

import os
import re
import ast
import argparse
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("targeted-fixer")


class TargetedNamingFixer(ast.NodeVisitor):
    """AST visitor that fixes specific naming convention issues."""

    def __init__(self, filename: str):
        """
        Initialize the fixer.

        Args:
            filename: Name of the file being fixed
        """
        self.filename = filename
        self.changes = []
        self.in_class = False
        self.current_class = None
        self.renames = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition node."""
        # Check class name
        class_name = node.name

        # Check if it's a test class method
        is_test_class = class_name.startswith("Test")
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "TestCase":
                is_test_class = True
                break
            elif isinstance(base, ast.Attribute):
                try:
                    if isinstance(base.value, ast.Name) and base.value.id == "unittest" and base.attr == "TestCase":
                        is_test_class = True
                        break
                except AttributeError:
                    pass

        if is_test_class:
            # Don't rename test class methods like setUp and tearDown
            self.in_class = True
            self.current_class = class_name
            self.generic_visit(node)
            self.in_class = False
            self.current_class = None
            return

        # Visit class body
        self.in_class = True
        self.current_class = class_name
        self.generic_visit(node)
        self.in_class = False
        self.current_class = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition node."""
        # Check function name
        function_name = node.name

        # Check if it's a test class method
        if self.in_class and self.current_class and self.current_class.startswith("Test"):
            if function_name in ["setUp", "tearDown", "setUpClass", "tearDownClass"]:
                # Don't rename test class methods
                self.generic_visit(node)
                return

        # Check if it's a method
        if self.in_class:
            # Check if it's a magic method
            if function_name.startswith("__") and function_name.endswith("__"):
                # Don't fix magic methods
                pass
            # Check if it's a private method
            elif function_name.startswith("_"):
                # Keep the leading underscore
                fixed_function_name = "_" + self.to_snake_case(function_name[1:])
                if function_name != fixed_function_name:
                    self.changes.append(f"Renamed private method from '{function_name}' to '{fixed_function_name}'")
                    self.renames[function_name] = fixed_function_name
                    node.name = fixed_function_name
            # Regular method
            else:
                fixed_function_name = self.to_snake_case(function_name)
                if function_name != fixed_function_name:
                    self.changes.append(f"Renamed method from '{function_name}' to '{fixed_function_name}'")
                    self.renames[function_name] = fixed_function_name
                    node.name = fixed_function_name
        # Regular function
        else:
            fixed_function_name = self.to_snake_case(function_name)
            if function_name != fixed_function_name:
                self.changes.append(f"Renamed function from '{function_name}' to '{fixed_function_name}'")
                self.renames[function_name] = fixed_function_name
                node.name = fixed_function_name

        # Visit function body
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Visit a name node."""
        # Fix variable names
        var_name = node.id

        # Check if it's a renamed variable
        if var_name in self.renames:
            node.id = self.renames[var_name]

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment node."""
        # Fix variable names
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # Check if it's a constant
                if var_name.isupper() or (len(var_name) > 0 and var_name[0].isupper() and not var_name[0].islower()):
                    # Don't fix constants
                    pass
                # Regular variable
                elif not var_name.startswith("_") or (var_name.startswith("_") and not var_name.startswith("__")):
                    if var_name.startswith("_"):
                        # Keep the leading underscore
                        fixed_var_name = "_" + self.to_snake_case(var_name[1:])
                        if var_name != fixed_var_name:
                            self.changes.append(f"Renamed private variable from '{var_name}' to '{fixed_var_name}'")
                            self.renames[var_name] = fixed_var_name
                            target.id = fixed_var_name
                    else:
                        fixed_var_name = self.to_snake_case(var_name)
                        if var_name != fixed_var_name:
                            self.changes.append(f"Renamed variable from '{var_name}' to '{fixed_var_name}'")
                            self.renames[var_name] = fixed_var_name
                            target.id = fixed_var_name

        # Visit assignment value
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit an annotated assignment node."""
        # Fix variable names
        if isinstance(node.target, ast.Name):
            var_name = node.target.id

            # Check if it's a constant
            if var_name.isupper() or (len(var_name) > 0 and var_name[0].isupper() and not var_name[0].islower()):
                # Don't fix constants
                pass
            # Check if it's a type variable
            elif not self.in_class and len(var_name) == 1 and var_name.isupper():
                # Don't fix type variables
                pass
            # Regular variable
            elif not var_name.startswith("_") or (var_name.startswith("_") and not var_name.startswith("__")):
                if var_name.startswith("_"):
                    # Keep the leading underscore
                    fixed_var_name = "_" + self.to_snake_case(var_name[1:])
                    if var_name != fixed_var_name:
                        self.changes.append(f"Renamed private variable from '{var_name}' to '{fixed_var_name}'")
                        self.renames[var_name] = fixed_var_name
                        node.target.id = fixed_var_name
                else:
                    fixed_var_name = self.to_snake_case(var_name)
                    if var_name != fixed_var_name:
                        self.changes.append(f"Renamed variable from '{var_name}' to '{fixed_var_name}'")
                        self.renames[var_name] = fixed_var_name
                        node.target.id = fixed_var_name

        # Visit annotation and value
        self.generic_visit(node)

    def to_snake_case(self, name: str) -> str:
        """
        Convert a name to snake_case.

        Args:
            name: Name to convert

        Returns:
            Name in snake_case
        """
        # Replace non-alphanumeric characters with underscores
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)

        # Insert underscores before uppercase letters
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

        # Convert to lowercase
        name = name.lower()

        # Replace multiple underscores with a single underscore
        name = re.sub(r"_+", "_", name)

        # Remove leading and trailing underscores
        name = name.strip("_")

        return name


def fix_file(filename: str, dry_run: bool = False) -> List[str]:
    """
    Fix naming convention issues in a file.

    Args:
        filename: Path to the file to fix
        dry_run: Whether to perform a dry run (don't modify the file)

    Returns:
        List of changes made
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the file
        tree = ast.parse(content)

        # Fix naming issues
        fixer = TargetedNamingFixer(filename)
        fixer.visit(tree)

        # Generate fixed code
        fixed_content = ast.unparse(tree)

        # Write fixed code back to the file
        if not dry_run and fixer.changes:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(fixed_content)

        return fixer.changes
    except Exception as e:
        logger.error(f"Error fixing file {filename}: {str(e)}")
        return [f"Error fixing file: {str(e)}"]


def fix_directory(directory: str, exclude_dirs: List[str] = None, dry_run: bool = False) -> Dict[str, List[str]]:
    """
    Fix naming convention issues in all Python files in a directory.

    Args:
        directory: Path to the directory to fix
        exclude_dirs: List of directories to exclude
        dry_run: Whether to perform a dry run (don't modify the files)

    Returns:
        Dictionary mapping filenames to lists of changes
    """
    if exclude_dirs is None:
        exclude_dirs = []

    changes = {}

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

        for file in files:
            if file.endswith(".py"):
                filename = os.path.join(root, file)
                file_changes = fix_file(filename, dry_run)
                if file_changes:
                    changes[filename] = file_changes

    return changes


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix specific naming convention issues in Python code")
    parser.add_argument("path", help="Path to file or directory to fix")
    parser.add_argument("--exclude", "-e", nargs="+", help="Directories to exclude", default=[])
    parser.add_argument("--dry-run", "-d", action="store_true", help="Dry run (don't modify files)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    path = args.path
    exclude_dirs = args.exclude
    dry_run = args.dry_run

    if dry_run:
        logger.info("Performing dry run (files will not be modified)")

    if os.path.isfile(path):
        changes = fix_file(path, dry_run)
        if changes:
            print(f"Changes in {path}:")
            for change in changes:
                print(f"  - {change}")
        else:
            print(f"No changes needed in {path}")
    elif os.path.isdir(path):
        changes = fix_directory(path, exclude_dirs, dry_run)
        if changes:
            print(f"Changes in {len(changes)} files:")
            for filename, file_changes in changes.items():
                print(f"\n{filename}:")
                for change in file_changes:
                    print(f"  - {change}")
        else:
            print(f"No changes needed in {path}")
    else:
        print(f"Path {path} does not exist")


if __name__ == "__main__":
    main()
