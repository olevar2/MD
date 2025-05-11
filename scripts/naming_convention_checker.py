#!/usr/bin/env python3
"""
Naming Convention Checker

This script checks the codebase for naming convention violations and
generates a report of files and directories that need to be renamed.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class NamingConventionChecker:
    """
    Checks naming conventions in the codebase.
    """

    def __init__(self, root_dir: str, config_file: str = None):
        """
        Initialize the checker.

        Args:
            root_dir: Root directory of the codebase
            config_file: Path to configuration file (optional)
        """
        self.root_dir = Path(root_dir)
        self.config = self._load_config(config_file)
        self.violations = []
        self.stats = {
            "files_checked": 0,
            "directories_checked": 0,
            "violations_found": 0,
        }

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.

        Args:
            config_file: Path to configuration file

        Returns:
            Configuration dictionary
        """
        default_config = {
            "python_files": {
                "pattern": r"^[a-z][a-z0-9_]*\.py$",
                "description": "snake_case"
            },
            "python_directories": {
                "pattern": r"^[a-z][a-z0-9_]*$",
                "description": "snake_case"
            },
            "js_ts_files": {
                "pattern": r"^[a-z][a-z0-9\-]*\.(js|ts|jsx|tsx)$",
                "description": "kebab-case"
            },
            "js_ts_directories": {
                "pattern": r"^[a-z][a-z0-9\-]*$",
                "description": "kebab-case"
            },
            "config_files": {
                "pattern": r"^[a-z][a-z0-9\-]*\.(json|yaml|yml|toml|ini|env)$",
                "description": "kebab-case"
            },
            "doc_files": {
                "pattern": r"^[A-Z][a-zA-Z0-9_]*\.(md|rst|txt)$",
                "description": "Title_Case"
            },
            "ignore_dirs": [
                ".git",
                ".github",
                ".vscode",
                ".pytest_cache",
                "__pycache__",
                "node_modules",
                "venv",
                ".venv",
                "build",
                "dist",
                "migrations"
            ],
            "ignore_files": [
                "README.md",
                "LICENSE",
                "CHANGELOG.md",
                "CONTRIBUTING.md",
                "requirements.txt",
                "package.json",
                "package-lock.json",
                "tsconfig.json",
                "tslint.json",
                "eslintrc.js",
                "prettierrc.js",
                "babel.config.js",
                "jest.config.js",
                "webpack.config.js",
                "setup.py",
                "pyproject.toml",
                "poetry.lock",
                "Dockerfile",
                "docker-compose.yml",
                ".dockerignore",
                ".gitignore",
                ".env.example",
                ".env.test",
                ".env.development",
                ".env.production"
            ]
        }

        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge user config with default config
                for key, value in user_config.items():
                    if key in default_config and isinstance(value, dict) and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value

        return default_config

    def check_naming_conventions(self) -> List[Dict[str, str]]:
        """
        Check naming conventions in the codebase.

        Returns:
            List of violations
        """
        self.violations = []
        self._check_directory(self.root_dir)
        self.stats["violations_found"] = len(self.violations)
        return self.violations

    def _check_directory(self, directory: Path) -> None:
        """
        Check naming conventions in a directory.

        Args:
            directory: Directory to check
        """
        # Skip ignored directories
        if directory.name in self.config["ignore_dirs"]:
            return

        # Check directory name
        if directory != self.root_dir:
            self.stats["directories_checked"] += 1
            self._check_directory_name(directory)

        # Check files and subdirectories
        for item in directory.iterdir():
            if item.is_dir():
                self._check_directory(item)
            else:
                self.stats["files_checked"] += 1
                self._check_file_name(item)

    def _check_directory_name(self, directory: Path) -> None:
        """
        Check if a directory name follows naming conventions.

        Args:
            directory: Directory to check
        """
        # Skip root directory
        if directory == self.root_dir:
            return

        # Determine the expected pattern based on the directory path
        if any(js_dir in str(directory) for js_dir in ["ui-service", "common-js-lib"]):
            pattern = self.config["js_ts_directories"]["pattern"]
            description = self.config["js_ts_directories"]["description"]
        else:
            pattern = self.config["python_directories"]["pattern"]
            description = self.config["python_directories"]["description"]

        # Check if the directory name matches the pattern
        if not re.match(pattern, directory.name):
            self.violations.append({
                "path": str(directory),
                "type": "directory",
                "expected": description,
                "actual": directory.name
            })

    def _check_file_name(self, file: Path) -> None:
        """
        Check if a file name follows naming conventions.

        Args:
            file: File to check
        """
        # Skip ignored files
        if file.name in self.config["ignore_files"]:
            return

        # Determine the expected pattern based on the file extension
        if file.suffix == ".py":
            pattern = self.config["python_files"]["pattern"]
            description = self.config["python_files"]["description"]
        elif file.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            pattern = self.config["js_ts_files"]["pattern"]
            description = self.config["js_ts_files"]["description"]
        elif file.suffix in [".json", ".yaml", ".yml", ".toml", ".ini", ".env"]:
            pattern = self.config["config_files"]["pattern"]
            description = self.config["config_files"]["description"]
        elif file.suffix in [".md", ".rst", ".txt"]:
            pattern = self.config["doc_files"]["pattern"]
            description = self.config["doc_files"]["description"]
        else:
            # Skip files with unknown extensions
            return

        # Check if the file name matches the pattern
        if not re.match(pattern, file.name):
            self.violations.append({
                "path": str(file),
                "type": "file",
                "expected": description,
                "actual": file.name
            })

    def generate_report(self, output_file: str = None) -> None:
        """
        Generate a report of naming convention violations.

        Args:
            output_file: Path to output file (optional)
        """
        report = {
            "stats": self.stats,
            "violations": self.violations
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            print(json.dumps(report, indent=2))

    def generate_rename_script(self, output_file: str) -> None:
        """
        Generate a script to rename files and directories.

        Args:
            output_file: Path to output file
        """
        with open(output_file, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Script to rename files and directories\n\n")

            # Sort violations to rename files before directories
            sorted_violations = sorted(
                self.violations,
                key=lambda v: 0 if v["type"] == "file" else 1
            )

            for violation in sorted_violations:
                path = violation["path"]
                expected = violation["expected"]
                actual = violation["actual"]

                # Generate a suggested new name
                if expected == "snake_case":
                    new_name = self._to_snake_case(actual)
                elif expected == "kebab-case":
                    new_name = self._to_kebab_case(actual)
                elif expected == "Title_Case":
                    new_name = self._to_title_case(actual)
                else:
                    new_name = actual

                # Skip if the new name is the same as the old name
                if new_name == actual:
                    continue

                # Generate the rename command
                old_path = path
                new_path = path.replace(actual, new_name)
                f.write(f'mv "{old_path}" "{new_path}"\n')

    def _to_snake_case(self, name: str) -> str:
        """
        Convert a name to snake_case.

        Args:
            name: Name to convert

        Returns:
            snake_case name
        """
        # Remove extension if present
        base_name = name
        extension = ""
        if "." in name:
            base_name, extension = name.rsplit(".", 1)
            extension = "." + extension

        # Convert kebab-case to snake_case
        if "-" in base_name:
            base_name = base_name.replace("-", "_")

        # Convert PascalCase or camelCase to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', base_name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        return s2 + extension

    def _to_kebab_case(self, name: str) -> str:
        """
        Convert a name to kebab-case.

        Args:
            name: Name to convert

        Returns:
            kebab-case name
        """
        # Remove extension if present
        base_name = name
        extension = ""
        if "." in name:
            base_name, extension = name.rsplit(".", 1)
            extension = "." + extension

        # Convert snake_case to kebab-case
        if "_" in base_name:
            base_name = base_name.replace("_", "-")

        # Convert PascalCase or camelCase to kebab-case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', base_name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()

        return s2 + extension

    def _to_title_case(self, name: str) -> str:
        """
        Convert a name to Title_Case.

        Args:
            name: Name to convert

        Returns:
            Title_Case name
        """
        # Remove extension if present
        base_name = name
        extension = ""
        if "." in name:
            base_name, extension = name.rsplit(".", 1)
            extension = "." + extension

        # Convert snake_case or kebab-case to Title_Case
        if "_" in base_name:
            words = base_name.split("_")
        elif "-" in base_name:
            words = base_name.split("-")
        else:
            # Convert PascalCase or camelCase to words
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', base_name)
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
            words = s2.split("_")

        # Capitalize each word and join with underscores
        title_case = "_".join(word.capitalize() for word in words if word)

        return title_case + extension


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Check naming conventions in the codebase")
    parser.add_argument("--root-dir", default=".", help="Root directory of the codebase")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Path to output file for the report")
    parser.add_argument("--rename-script", help="Path to output file for the rename script")
    args = parser.parse_args()

    checker = NamingConventionChecker(args.root_dir, args.config)
    checker.check_naming_conventions()
    checker.generate_report(args.output)

    if args.rename_script:
        checker.generate_rename_script(args.rename_script)


if __name__ == "__main__":
    main()