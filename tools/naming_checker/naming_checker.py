#!/usr/bin/env python3
"""
Naming Convention Checker

This script checks if the codebase follows the naming conventions defined in the
naming_conventions.md file.
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
logger = logging.getLogger("naming-checker")


# Regular expressions for naming conventions
MODULE_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
PACKAGE_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
CLASS_REGEX = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
INTERFACE_REGEX = re.compile(r"^I[A-Z][a-zA-Z0-9]*$")
ABSTRACT_CLASS_REGEX = re.compile(r"^Abstract[A-Z][a-zA-Z0-9]*$")
EXCEPTION_REGEX = re.compile(r"^[A-Z][a-zA-Z0-9]*Error$")
FUNCTION_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
PRIVATE_METHOD_REGEX = re.compile(r"^_[a-z][a-z0-9_]*$")
MAGIC_METHOD_REGEX = re.compile(r"^__[a-z][a-z0-9_]*__$")
VARIABLE_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
CONSTANT_REGEX = re.compile(r"^[A-Z][A-Z0-9_]*$")
TYPE_VAR_REGEX = re.compile(r"^[A-Z]$")


class NamingVisitor(ast.NodeVisitor):
    """AST visitor that checks naming conventions."""
    
    def __init__(self, filename: str):
        """
        Initialize the visitor.
        
        Args:
            filename: Name of the file being checked
        """
        self.filename = filename
        self.issues = []
        self.in_class = False
        self.current_class = None
    
    def visit_Module(self, node: ast.Module) -> None:
        """Visit a module node."""
        # Check module name
        module_name = os.path.basename(self.filename).replace(".py", "")
        if not MODULE_REGEX.match(module_name):
            self.issues.append(f"Module name '{module_name}' does not follow naming convention (lowercase with underscores)")
        
        # Visit all nodes in the module
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition node."""
        # Check class name
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
        
        # Apply appropriate regex
        if is_interface:
            if not INTERFACE_REGEX.match(class_name):
                self.issues.append(f"Interface name '{class_name}' does not follow naming convention (start with 'I' followed by PascalCase)")
        elif is_abstract:
            if not ABSTRACT_CLASS_REGEX.match(class_name):
                self.issues.append(f"Abstract class name '{class_name}' does not follow naming convention (start with 'Abstract' followed by PascalCase)")
        elif is_exception:
            if not EXCEPTION_REGEX.match(class_name):
                self.issues.append(f"Exception name '{class_name}' does not follow naming convention (PascalCase ending with 'Error')")
        else:
            if not CLASS_REGEX.match(class_name):
                self.issues.append(f"Class name '{class_name}' does not follow naming convention (PascalCase)")
        
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
        
        # Check if it's a method
        if self.in_class:
            # Check if it's a magic method
            if function_name.startswith("__") and function_name.endswith("__"):
                if not MAGIC_METHOD_REGEX.match(function_name):
                    self.issues.append(f"Magic method name '{function_name}' in class '{self.current_class}' does not follow naming convention (start and end with double underscores)")
            # Check if it's a private method
            elif function_name.startswith("_"):
                if not PRIVATE_METHOD_REGEX.match(function_name):
                    self.issues.append(f"Private method name '{function_name}' in class '{self.current_class}' does not follow naming convention (start with single underscore followed by lowercase with underscores)")
            # Regular method
            else:
                if not FUNCTION_REGEX.match(function_name):
                    self.issues.append(f"Method name '{function_name}' in class '{self.current_class}' does not follow naming convention (lowercase with underscores)")
        # Regular function
        else:
            if not FUNCTION_REGEX.match(function_name):
                self.issues.append(f"Function name '{function_name}' does not follow naming convention (lowercase with underscores)")
        
        # Visit function body
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment node."""
        # Check variable names
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                
                # Check if it's a constant
                if self.in_class and var_name.isupper():
                    if not CONSTANT_REGEX.match(var_name):
                        self.issues.append(f"Class constant '{var_name}' in class '{self.current_class}' does not follow naming convention (UPPERCASE with underscores)")
                elif not self.in_class and var_name.isupper():
                    if not CONSTANT_REGEX.match(var_name):
                        self.issues.append(f"Module constant '{var_name}' does not follow naming convention (UPPERCASE with underscores)")
                # Regular variable
                elif not var_name.startswith("_") or (var_name.startswith("_") and not var_name.startswith("__")):
                    if not VARIABLE_REGEX.match(var_name):
                        if self.in_class:
                            self.issues.append(f"Instance variable '{var_name}' in class '{self.current_class}' does not follow naming convention (lowercase with underscores)")
                        else:
                            self.issues.append(f"Variable '{var_name}' does not follow naming convention (lowercase with underscores)")
        
        # Visit assignment value
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit an annotated assignment node."""
        # Check variable names
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            
            # Check if it's a constant
            if self.in_class and var_name.isupper():
                if not CONSTANT_REGEX.match(var_name):
                    self.issues.append(f"Class constant '{var_name}' in class '{self.current_class}' does not follow naming convention (UPPERCASE with underscores)")
            elif not self.in_class and var_name.isupper():
                if not CONSTANT_REGEX.match(var_name):
                    self.issues.append(f"Module constant '{var_name}' does not follow naming convention (UPPERCASE with underscores)")
            # Check if it's a type variable
            elif not self.in_class and len(var_name) == 1 and var_name.isupper():
                if not TYPE_VAR_REGEX.match(var_name):
                    self.issues.append(f"Type variable '{var_name}' does not follow naming convention (single uppercase letter)")
            # Regular variable
            elif not var_name.startswith("_") or (var_name.startswith("_") and not var_name.startswith("__")):
                if not VARIABLE_REGEX.match(var_name):
                    if self.in_class:
                        self.issues.append(f"Instance variable '{var_name}' in class '{self.current_class}' does not follow naming convention (lowercase with underscores)")
                    else:
                        self.issues.append(f"Variable '{var_name}' does not follow naming convention (lowercase with underscores)")
        
        # Visit annotation and value
        self.generic_visit(node)


def check_file(filename: str) -> List[str]:
    """
    Check a file for naming convention issues.
    
    Args:
        filename: Path to the file to check
        
    Returns:
        List of issues found
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Visit the AST
        visitor = NamingVisitor(filename)
        visitor.visit(tree)
        
        return visitor.issues
    except Exception as e:
        logger.error(f"Error checking file {filename}: {str(e)}")
        return [f"Error checking file: {str(e)}"]


def check_directory(directory: str, exclude_dirs: List[str] = None) -> Dict[str, List[str]]:
    """
    Check all Python files in a directory for naming convention issues.
    
    Args:
        directory: Path to the directory to check
        exclude_dirs: List of directories to exclude
        
    Returns:
        Dictionary mapping filenames to lists of issues
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    issues = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
        
        for file in files:
            if file.endswith(".py"):
                filename = os.path.join(root, file)
                file_issues = check_file(filename)
                if file_issues:
                    issues[filename] = file_issues
    
    return issues


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check naming conventions in Python code")
    parser.add_argument("path", help="Path to file or directory to check")
    parser.add_argument("--exclude", "-e", nargs="+", help="Directories to exclude", default=[])
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    path = args.path
    exclude_dirs = args.exclude
    
    if os.path.isfile(path):
        issues = check_file(path)
        if issues:
            print(f"Issues in {path}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"No issues found in {path}")
    elif os.path.isdir(path):
        issues = check_directory(path, exclude_dirs)
        if issues:
            print(f"Issues found in {len(issues)} files:")
            for filename, file_issues in issues.items():
                print(f"\n{filename}:")
                for issue in file_issues:
                    print(f"  - {issue}")
        else:
            print(f"No issues found in {path}")
    else:
        print(f"Path {path} does not exist")


if __name__ == "__main__":
    main()
