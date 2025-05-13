"""
Script to identify services using the legacy configuration system.

This script scans the codebase for services using the legacy configuration system
and identifies files that need to be migrated to the standardized configuration system.
"""

import os
import re
import json
from typing import Dict, List, Set, Tuple, Optional, Any

# Service directories to scan
SERVICE_DIRS = [
    "analysis-engine-service",
    "data-pipeline-service",
    "trading-gateway-service",
    "ml-integration-service"
]

# Directories to skip
SKIP_DIRS = ['.git', '.venv', 'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist']

# Legacy configuration patterns
LEGACY_CONFIG_PATTERNS = [
    r"from\s+common_lib\.config\s+import\s+(?:[^,]+,\s*)*(?:Config|ConfigManager|ConfigLoader|ServiceSpecificConfig)",
    r"class\s+\w+\(ServiceSpecificConfig\)",
    r"config_manager\s*=\s*ConfigManager\(",
    r"\.get_service_specific_config\(\)",
    r"\.get_database_config\(\)",
    r"\.get_service_config\(\)",
    r"\.get_logging_config\(\)",
    r"\.get_service_client_config\("
]

# Standardized configuration patterns
STANDARDIZED_CONFIG_PATTERNS = [
    r"from\s+common_lib\.config\s+import\s+(?:[^,]+,\s*)*(?:BaseAppSettings|StandardizedConfigManager|get_settings)",
    r"class\s+\w+\(BaseAppSettings\)",
    r"get_settings\(",
    r"get_config_manager\("
]


class LegacyConfigFinder:
    """
    Class to find services using the legacy configuration system.
    """

    def __init__(self, root_dir: str = '.'):
        """
        Initialize the finder.

        Args:
            root_dir: Root directory to scan
        """
        self.root_dir = root_dir
        self.results = {}

    def scan_codebase(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan the codebase for services using the legacy configuration system.

        Returns:
            Dictionary mapping service names to dictionaries with legacy and standardized files
        """
        for service_dir in SERVICE_DIRS:
            service_path = os.path.join(self.root_dir, service_dir)
            if not os.path.exists(service_path):
                continue

            self.results[service_dir] = {
                "legacy_files": [],
                "standardized_files": [],
                "mixed_files": [],
                "config_files": []
            }

            for dirpath, dirnames, filenames in os.walk(service_path):
                # Skip directories
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

                for filename in filenames:
                    if not filename.endswith('.py'):
                        continue

                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, self.root_dir)

                    try:
                        self._analyze_file(file_path, rel_path, service_dir)
                    except Exception as e:
                        print(f"Error analyzing {rel_path}: {str(e)}")

        return self.results

    def _analyze_file(self, file_path: str, rel_path: str, service_dir: str) -> None:
        """
        Analyze a file for legacy and standardized configuration patterns.

        Args:
            file_path: Path to the file
            rel_path: Relative path to the file
            service_dir: Service directory
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Check if the file contains legacy configuration patterns
        legacy_matches = []
        for pattern in LEGACY_CONFIG_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                legacy_matches.extend(matches)

        # Check if the file contains standardized configuration patterns
        standardized_matches = []
        for pattern in STANDARDIZED_CONFIG_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                standardized_matches.extend(matches)

        # Check if the file is a configuration file
        is_config_file = "config" in rel_path.lower() and (
            "config.py" in rel_path.lower() or
            "settings.py" in rel_path.lower() or
            "configuration.py" in rel_path.lower()
        )

        # Categorize the file
        if legacy_matches and standardized_matches:
            self.results[service_dir]["mixed_files"].append({
                "file": rel_path,
                "legacy_matches": legacy_matches,
                "standardized_matches": standardized_matches
            })
        elif legacy_matches:
            self.results[service_dir]["legacy_files"].append({
                "file": rel_path,
                "legacy_matches": legacy_matches
            })
        elif standardized_matches:
            self.results[service_dir]["standardized_files"].append({
                "file": rel_path,
                "standardized_matches": standardized_matches
            })

        # Add to config files if it's a configuration file
        if is_config_file:
            self.results[service_dir]["config_files"].append({
                "file": rel_path,
                "legacy": bool(legacy_matches),
                "standardized": bool(standardized_matches)
            })


def main():
    """Main function."""
    finder = LegacyConfigFinder()
    results = finder.scan_codebase()

    # Print results
    print("Legacy Configuration Analysis")
    print("============================")

    total_legacy_files = 0
    total_standardized_files = 0
    total_mixed_files = 0
    total_config_files = 0

    for service_dir, files in results.items():
        legacy_files = files["legacy_files"]
        standardized_files = files["standardized_files"]
        mixed_files = files["mixed_files"]
        config_files = files["config_files"]

        print(f"\n{service_dir}:")
        print(f"  Legacy files: {len(legacy_files)}")
        print(f"  Standardized files: {len(standardized_files)}")
        print(f"  Mixed files: {len(mixed_files)}")
        print(f"  Config files: {len(config_files)}")

        if legacy_files:
            print("\n  Legacy files:")
            for file_info in legacy_files[:5]:  # Show only the first 5 files
                print(f"    {file_info['file']}")
            if len(legacy_files) > 5:
                print(f"    ... and {len(legacy_files) - 5} more")

        if mixed_files:
            print("\n  Mixed files:")
            for file_info in mixed_files[:5]:  # Show only the first 5 files
                print(f"    {file_info['file']}")
            if len(mixed_files) > 5:
                print(f"    ... and {len(mixed_files) - 5} more")

        if config_files:
            print("\n  Config files:")
            for file_info in config_files:
                config_type = "Mixed" if file_info["legacy"] and file_info["standardized"] else "Legacy" if file_info["legacy"] else "Standardized"
                print(f"    {file_info['file']} ({config_type})")

        total_legacy_files += len(legacy_files)
        total_standardized_files += len(standardized_files)
        total_mixed_files += len(mixed_files)
        total_config_files += len(config_files)

    print("\nSummary:")
    print(f"  Total legacy files: {total_legacy_files}")
    print(f"  Total standardized files: {total_standardized_files}")
    print(f"  Total mixed files: {total_mixed_files}")
    print(f"  Total config files: {total_config_files}")

    # Save results to file
    with open('legacy_config_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to legacy_config_analysis.json")


if __name__ == "__main__":
    main()
