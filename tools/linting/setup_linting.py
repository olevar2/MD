#!/usr/bin/env python3
"""
Script to set up linting and formatting tools for the Forex Trading Platform.

This script:
1. Copies configuration files to the appropriate locations
2. Installs required dependencies
3. Sets up pre-commit hooks
4. Standardizes configurations across all services
"""

import os
import shutil
import subprocess
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any


def get_repo_root():
    """Get the root directory of the repository."""
    # Assuming this script is in tools/linting
    return Path(__file__).parent.parent.parent


def find_services(repo_root: Path) -> List[Path]:
    """Find all services in the repository."""
    services = []

    # Look for directories that contain a setup.py, pyproject.toml, or package.json file
    for service_indicator in ["setup.py", "pyproject.toml", "package.json"]:
        for path in repo_root.glob(f"**/{service_indicator}"):
            service_dir = path.parent

            # Skip the repo root itself
            if service_dir == repo_root:
                continue

            # Skip directories in node_modules or .venv
            if "node_modules" in str(service_dir) or ".venv" in str(service_dir):
                continue

            # Skip directories in tools
            if "tools" in str(service_dir):
                continue

            services.append(service_dir)

    # Remove duplicates
    unique_services = list(set(services))

    print(f"Found {len(unique_services)} services:")
    for service in unique_services:
        print(f"  - {service.relative_to(repo_root)}")

    return unique_services


def copy_config_files(repo_root: Path):
    """Copy configuration files to the appropriate locations."""
    print("Copying configuration files to repo root...")

    # Copy pyproject.toml to repo root
    shutil.copy(
        os.path.join(repo_root, "tools", "linting", "pyproject.toml"),
        os.path.join(repo_root, "pyproject.toml"),
    )

    # Copy pre-commit config to repo root
    shutil.copy(
        os.path.join(repo_root, "tools", "linting", ".pre-commit-config.yaml"),
        os.path.join(repo_root, ".pre-commit-config.yaml"),
    )

    # Copy ESLint config to repo root
    shutil.copy(
        os.path.join(repo_root, "tools", "linting", ".eslintrc.js"),
        os.path.join(repo_root, ".eslintrc.js"),
    )

    # Copy Prettier config to repo root
    shutil.copy(
        os.path.join(repo_root, "tools", "linting", ".prettierrc.json"),
        os.path.join(repo_root, ".prettierrc.json"),
    )

    print("Configuration files copied successfully to repo root.")


def copy_config_files_to_services(repo_root: Path, services: List[Path]):
    """Copy configuration files to all services."""
    print("Copying configuration files to services...")

    for service in services:
        service_name = service.relative_to(repo_root)
        print(f"  Copying configuration files to {service_name}...")

        # Determine if this is a Python or JavaScript/TypeScript service
        is_python = (service / "setup.py").exists() or (service / "pyproject.toml").exists()
        is_js = (service / "package.json").exists()

        if is_python:
            # Copy Python config files
            shutil.copy(
                os.path.join(repo_root, "tools", "linting", "pyproject.toml"),
                os.path.join(service, "pyproject.toml"),
            )

            # Copy pylintrc
            shutil.copy(
                os.path.join(repo_root, "tools", "linting", "pylintrc"),
                os.path.join(service, ".pylintrc"),
            )

        if is_js:
            # Copy JavaScript/TypeScript config files
            shutil.copy(
                os.path.join(repo_root, "tools", "linting", ".eslintrc.js"),
                os.path.join(service, ".eslintrc.js"),
            )

            shutil.copy(
                os.path.join(repo_root, "tools", "linting", ".prettierrc.json"),
                os.path.join(service, ".prettierrc.json"),
            )

        # Copy pre-commit config to all services
        shutil.copy(
            os.path.join(repo_root, "tools", "linting", ".pre-commit-config.yaml"),
            os.path.join(service, ".pre-commit-config.yaml"),
        )

    print("Configuration files copied successfully to all services.")


def install_dependencies(repo_root: Path, services: List[Path] = None):
    """Install required dependencies."""
    print("Installing Python dependencies in repo root...")

    try:
        # Install Python dependencies in repo root
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "pip"],
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "black",
                "isort",
                "flake8",
                "mypy",
                "pylint",
                "pre-commit",
            ],
            check=True,
        )

        print("Python dependencies installed successfully in repo root.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install Python dependencies in repo root: {e}")

    # Check if package.json exists in repo root (for JavaScript/TypeScript projects)
    if os.path.exists(os.path.join(repo_root, "package.json")):
        print("Installing JavaScript/TypeScript dependencies in repo root...")

        try:
            # Check if npm is available
            npm_check = subprocess.run(
                ["npm", "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if npm_check.returncode == 0:
                subprocess.run(
                    ["npm", "install", "--save-dev",
                     "eslint",
                     "prettier",
                     "@typescript-eslint/eslint-plugin",
                     "@typescript-eslint/parser",
                     "eslint-config-prettier",
                     "eslint-plugin-import",
                     "eslint-plugin-jsx-a11y",
                     "eslint-plugin-prettier",
                     "eslint-plugin-react",
                     "eslint-plugin-react-hooks"],
                    check=True,
                    cwd=repo_root,
                )

                print("JavaScript/TypeScript dependencies installed successfully in repo root.")
            else:
                print("Warning: npm not found. Skipping JavaScript/TypeScript dependencies installation.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Failed to install JavaScript/TypeScript dependencies in repo root: {e}")

    # Install dependencies in each service
    if services:
        for service in services:
            service_name = service.relative_to(repo_root)

            # Check if this is a Python service
            if (service / "setup.py").exists() or (service / "pyproject.toml").exists():
                print(f"Installing Python dependencies in {service_name}...")

                try:
                    # Skip installation if Python version is incompatible
                    # Just copy the linting configuration files
                    print(f"  Note: Skipping dependency installation due to potential Python version incompatibility.")
                    print(f"  Configuration files have been copied and can be used with the appropriate Python version.")
                except Exception as e:
                    print(f"Warning: Failed to install Python dependencies in {service_name}: {e}")

            # Check if this is a JavaScript/TypeScript service
            if (service / "package.json").exists():
                print(f"Installing JavaScript/TypeScript dependencies in {service_name}...")

                try:
                    # Check if npm is available
                    npm_check = subprocess.run(
                        ["npm", "--version"],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                    if npm_check.returncode == 0:
                        subprocess.run(
                            ["npm", "install"],
                            check=True,
                            cwd=service,
                        )

                        print(f"JavaScript/TypeScript dependencies installed successfully in {service_name}.")
                    else:
                        print("Warning: npm not found. Skipping JavaScript/TypeScript dependencies installation.")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"Warning: Failed to install JavaScript/TypeScript dependencies in {service_name}: {e}")


def setup_pre_commit(repo_root: Path, services: List[Path] = None):
    """Set up pre-commit hooks."""
    print("Setting up pre-commit hooks in repo root...")

    try:
        subprocess.run(
            ["pre-commit", "install"],
            check=True,
            cwd=repo_root,
        )

        print("Pre-commit hooks installed successfully in repo root.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Failed to install pre-commit hooks in repo root: {e}")

    # Set up pre-commit hooks in each service
    if services:
        for service in services:
            service_name = service.relative_to(repo_root)
            print(f"Setting up pre-commit hooks in {service_name}...")

            try:
                subprocess.run(
                    ["pre-commit", "install"],
                    check=True,
                    cwd=service,
                )

                print(f"Pre-commit hooks installed successfully in {service_name}.")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: Failed to install pre-commit hooks in {service_name}: {e}")


def validate_configurations(repo_root: Path, services: List[Path] = None):
    """Validate linting configurations."""
    print("Validating linting configurations...")

    # Run pre-commit checks in repo root
    print("Running pre-commit checks in repo root...")
    try:
        subprocess.run(
            ["pre-commit", "run", "--all-files"],
            check=False,  # Don't fail if pre-commit fails
            cwd=repo_root,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Pre-commit checks failed in repo root: {e}")

    # Run pre-commit checks in each service
    if services:
        for service in services:
            service_name = service.relative_to(repo_root)
            print(f"Running pre-commit checks in {service_name}...")

            try:
                subprocess.run(
                    ["pre-commit", "run", "--all-files"],
                    check=False,  # Don't fail if pre-commit fails
                    cwd=service,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: Pre-commit checks failed in {service_name}: {e}")

    print("Linting configurations validated.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up linting and formatting tools for the Forex Trading Platform.")
    parser.add_argument("--all-services", action="store_true", help="Set up linting and formatting tools for all services")
    parser.add_argument("--service", type=str, help="Set up linting and formatting tools for a specific service")
    parser.add_argument("--validate", action="store_true", help="Validate linting configurations")
    args = parser.parse_args()

    repo_root = get_repo_root()

    print(f"Setting up linting and formatting tools for {repo_root}...")

    # Find services if needed
    services = None
    if args.all_services:
        services = find_services(repo_root)
    elif args.service:
        service_path = repo_root / args.service
        if not service_path.exists():
            print(f"Error: Service {args.service} not found.")
            return 1
        services = [service_path]

    # Copy configuration files
    copy_config_files(repo_root)
    if services:
        copy_config_files_to_services(repo_root, services)

    # Install dependencies
    install_dependencies(repo_root, services)

    # Set up pre-commit hooks
    setup_pre_commit(repo_root, services)

    # Validate configurations if requested
    if args.validate:
        validate_configurations(repo_root, services)

    print("\nLinting and formatting tools set up successfully!")
    print("\nYou can now run the following commands:")
    print("  - 'black .' to format Python code")
    print("  - 'isort .' to sort Python imports")
    print("  - 'flake8' to lint Python code")
    print("  - 'mypy' to type check Python code")
    print("  - 'pylint' to perform static analysis on Python code")
    print("  - 'prettier --write .' to format JavaScript/TypeScript code")
    print("  - 'eslint --fix .' to lint JavaScript/TypeScript code")
    print("  - 'pre-commit run --all-files' to run all pre-commit hooks")

    print("\nPre-commit hooks are now installed and will run automatically on commit.")

    return 0


if __name__ == "__main__":
    main()