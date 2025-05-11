#!/usr/bin/env python3
"""
Naming Convention Fixer

This script automatically fixes naming convention issues in the codebase.
"""

import os
import re
import ast
import argparse
import logging
import tokenize
import io
from typing import List, Dict, Any, Tuple, Set, Optional
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("naming-fixer")


def to_snake_case(name: str) -> str:
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


def to_pascal_case(name: str) -> str:
    """
    Convert a name to PascalCase.
    
    Args:
        name: Name to convert
        
    Returns:
        Name in PascalCase
    """
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    
    # Split by underscores
    words = name.split("_")
    
    # Capitalize each word
    words = [word.capitalize() for word in words if word]
    
    # Join words
    name = "".join(words)
    
    return name


def to_camel_case(name: str) -> str:
    """
    Convert a name to camelCase.
    
    Args:
        name: Name to convert
        
    Returns:
        Name in camelCase
    """
    # Convert to PascalCase first
    name = to_pascal_case(name)
    
    # Convert first character to lowercase
    if name:
        name = name[0].lower() + name[1:]
    
    return name


def to_constant_case(name: str) -> str:
    """
    Convert a name to CONSTANT_CASE.
    
    Args:
        name: Name to convert
        
    Returns:
        Name in CONSTANT_CASE
    """
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    
    # Insert underscores before uppercase letters
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    
    # Convert to uppercase
    name = name.upper()
    
    # Replace multiple underscores with a single underscore
    name = re.sub(r"_+", "_", name)
    
    # Remove leading and trailing underscores
    name = name.strip("_")
    
    return name


class NamingFixer(ast.NodeTransformer):
    """AST transformer that fixes naming convention issues."""
    
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
    
    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visit a module node."""
        # Fix module name
        module_name = os.path.basename(self.filename).replace(".py", "")
        fixed_module_name = to_snake_case(module_name)
        
        if module_name != fixed_module_name:
            self.changes.append(f"Renamed module from '{module_name}' to '{fixed_module_name}'")
            # We'll rename the file later
        
        # Visit all nodes in the module
        return self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Visit a class definition node."""
        # Fix class name
        class_name = node.name
        
        # Check if it's an interface
        is_interface = class_name.startswith("I") and len(class_name) > 1 and class_name[1].isupper()
        
        # Check if it's an abstract class
        is_abstract = class_name.startswith("Abstract")
        
        # Check if it's an exception
        is_exception = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ["Exception", "BaseException"]:
                is_exception = True
            elif isinstance(base, ast.Name) and base.id.endswith("Error"):
                is_exception = True
        
        # Apply appropriate fix
        if is_interface:
            fixed_class_name = "I" + to_pascal_case(class_name[1:])
        elif is_abstract:
            fixed_class_name = "Abstract" + to_pascal_case(class_name[len("Abstract"):])
        elif is_exception:
            fixed_class_name = to_pascal_case(class_name)
            if not fixed_class_name.endswith("Error"):
                fixed_class_name += "Error"
        else:
            fixed_class_name = to_pascal_case(class_name)
        
        if class_name != fixed_class_name:
            self.changes.append(f"Renamed class from '{class_name}' to '{fixed_class_name}'")
            self.renames[class_name] = fixed_class_name
            node.name = fixed_class_name
        
        # Visit class body
        self.in_class = True
        self.current_class = fixed_class_name
        node = self.generic_visit(node)
        self.in_class = False
        self.current_class = None
        
        return node
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Visit a function definition node."""
        # Fix function name
        function_name = node.name
        
        # Check if it's a method
        if self.in_class:
            # Check if it's a magic method
            if function_name.startswith("__") and function_name.endswith("__"):
                # Don't fix magic methods
                fixed_function_name = function_name
            # Check if it's a private method
            elif function_name.startswith("_"):
                # Keep the leading underscore
                fixed_function_name = "_" + to_snake_case(function_name[1:])
            # Regular method
            else:
                fixed_function_name = to_snake_case(function_name)
        # Regular function
        else:
            fixed_function_name = to_snake_case(function_name)
        
        if function_name != fixed_function_name:
            self.changes.append(f"Renamed {'method' if self.in_class else 'function'} from '{function_name}' to '{fixed_function_name}'")
            self.renames[function_name] = fixed_function_name
            node.name = fixed_function_name
        
        # Visit function body
        return self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Visit a name node."""
        # Fix variable names
        var_name = node.id
        
        # Check if it's a renamed variable
        if var_name in self.renames:
            node.id = self.renames[var_name]
        
        return node
    
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Visit an assignment node."""
        # Fix variable names
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                
                # Check if it's a constant
                if var_name.isupper() or (len(var_name) > 0 and var_name[0].isupper() and not var_name[0].islower()):
                    fixed_var_name = to_constant_case(var_name)
                # Regular variable
                elif not var_name.startswith("_") or (var_name.startswith("_") and not var_name.startswith("__")):
                    if var_name.startswith("_"):
                        # Keep the leading underscore
                        fixed_var_name = "_" + to_snake_case(var_name[1:])
                    else:
                        fixed_var_name = to_snake_case(var_name)
                else:
                    # Don't fix other variables
                    fixed_var_name = var_name
                
                if var_name != fixed_var_name:
                    self.changes.append(f"Renamed variable from '{var_name}' to '{fixed_var_name}'")
                    self.renames[var_name] = fixed_var_name
                    target.id = fixed_var_name
        
        # Visit assignment value
        return self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Visit an annotated assignment node."""
        # Fix variable names
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            
            # Check if it's a constant
            if var_name.isupper() or (len(var_name) > 0 and var_name[0].isupper() and not var_name[0].islower()):
                fixed_var_name = to_constant_case(var_name)
            # Check if it's a type variable
            elif not self.in_class and len(var_name) == 1 and var_name.isupper():
                # Don't fix type variables
                fixed_var_name = var_name
            # Regular variable
            elif not var_name.startswith("_") or (var_name.startswith("_") and not var_name.startswith("__")):
                if var_name.startswith("_"):
                    # Keep the leading underscore
                    fixed_var_name = "_" + to_snake_case(var_name[1:])
                else:
                    fixed_var_name = to_snake_case(var_name)
            else:
                # Don't fix other variables
                fixed_var_name = var_name
            
            if var_name != fixed_var_name:
                self.changes.append(f"Renamed variable from '{var_name}' to '{fixed_var_name}'")
                self.renames[var_name] = fixed_var_name
                node.target.id = fixed_var_name
        
        # Visit annotation and value
        return self.generic_visit(node)


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
        fixer = NamingFixer(filename)
        fixed_tree = fixer.visit(tree)
        
        # Generate fixed code
        fixed_content = ast.unparse(fixed_tree)
        
        # Write fixed code back to the file
        if not dry_run and fixer.changes:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            
            # Fix module name (rename file)
            module_name = os.path.basename(filename).replace(".py", "")
            fixed_module_name = to_snake_case(module_name)
            
            if module_name != fixed_module_name:
                directory = os.path.dirname(filename)
                new_filename = os.path.join(directory, fixed_module_name + ".py")
                
                # Check if the new filename already exists
                if os.path.exists(new_filename):
                    logger.warning(f"Cannot rename file {filename} to {new_filename}: file already exists")
                else:
                    os.rename(filename, new_filename)
                    logger.info(f"Renamed file from {filename} to {new_filename}")
        
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
    parser = argparse.ArgumentParser(description="Fix naming conventions in Python code")
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
