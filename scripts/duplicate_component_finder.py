#!/usr/bin/env python3
"""
Duplicate Component Finder

This script identifies potential duplicate component implementations across
different services in the forex trading platform.
"""

import os
import re
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict


class DuplicateComponentFinder:
    """
    Finds potential duplicate component implementations.
    """

    def __init__(self, root_dir: str, config_file: str = None):
        """
        Initialize the finder.

        Args:
            root_dir: Root directory of the codebase
            config_file: Path to configuration file (optional)
        """
        self.root_dir = Path(root_dir)
        self.config = self._load_config(config_file)
        self.duplicates = defaultdict(list)
        self.stats = {
            "files_analyzed": 0,
            "potential_duplicates": 0,
            "duplicate_groups": 0,
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
            "file_extensions": [".py", ".js", ".ts"],
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
                "migrations",
                "tests",
                "test"
            ],
            "ignore_files": [
                "__init__.py",
                "setup.py",
                "conftest.py"
            ],
            "component_patterns": [
                r"class\s+([A-Za-z0-9_]+)(Indicator|Provider|Client|Service|Repository|Factory|Adapter|Handler|Processor|Validator|Formatter|Parser|Serializer|Deserializer|Converter|Transformer|Calculator|Analyzer|Generator|Manager|Controller|Middleware|Decorator|Wrapper|Proxy|Strategy|Builder|Director|Prototype|Singleton|Observer|Visitor|Command|Interpreter|Iterator|Mediator|Memento|State|Template|Flyweight|Bridge|Composite|Facade)",
                r"def\s+([a-z0-9_]+_indicator|[a-z0-9_]+_provider|[a-z0-9_]+_client|[a-z0-9_]+_service|[a-z0-9_]+_repository|[a-z0-9_]+_factory|[a-z0-9_]+_adapter|[a-z0-9_]+_handler|[a-z0-9_]+_processor|[a-z0-9_]+_validator|[a-z0-9_]+_formatter|[a-z0-9_]+_parser|[a-z0-9_]+_serializer|[a-z0-9_]+_deserializer|[a-z0-9_]+_converter|[a-z0-9_]+_transformer|[a-z0-9_]+_calculator|[a-z0-9_]+_analyzer|[a-z0-9_]+_generator|[a-z0-9_]+_manager|[a-z0-9_]+_controller|[a-z0-9_]+_middleware)",
                r"class\s+([A-Za-z0-9_]+)(RSI|MACD|EMA|SMA|ATR|Bollinger|Stochastic|Ichimoku|Fibonacci|Gann|Elliott|Momentum|ROC|CCI|Williams|Parabolic|ADX|Aroon|Keltner|Donchian|Envelope|Pivot|Support|Resistance|Trend|Volatility|Volume|OBV|MFI|Chaikin|Accumulation|Distribution|Money|Flow|Oscillator|Divergence|Convergence|Moving|Average)",
                r"def\s+(calculate|compute|get)_([a-z0-9_]+)"
            ],
            "similarity_threshold": 0.5,
            "min_component_lines": 5
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

    def find_duplicates(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find potential duplicate component implementations.

        Returns:
            Dictionary mapping component names to lists of implementations
        """
        self.duplicates = defaultdict(list)
        self._scan_directory(self.root_dir)
        self._analyze_duplicates()
        return dict(self.duplicates)

    def _scan_directory(self, directory: Path) -> None:
        """
        Scan a directory for potential duplicate components.

        Args:
            directory: Directory to scan
        """
        # Skip ignored directories
        if directory.name in self.config["ignore_dirs"]:
            return

        # Scan files and subdirectories
        for item in directory.iterdir():
            if item.is_dir():
                self._scan_directory(item)
            elif item.suffix in self.config["file_extensions"] and item.name not in self.config["ignore_files"]:
                self._analyze_file(item)

    def _analyze_file(self, file_path: Path) -> None:
        """
        Analyze a file for potential duplicate components.

        Args:
            file_path: Path to the file
        """
        self.stats["files_analyzed"] += 1

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Skip binary files
            return

        # Find components in the file
        components = self._find_components(content, file_path)

        # Add components to the duplicates dictionary
        for component_name, component_info in components.items():
            self.duplicates[component_name].append({
                "file_path": str(file_path),
                "start_line": component_info["start_line"],
                "end_line": component_info["end_line"],
                "content": component_info["content"],
                "content_hash": component_info["content_hash"],
                "service": self._get_service_name(file_path)
            })

    def _find_components(self, content: str, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Find components in a file.

        Args:
            content: File content
            file_path: Path to the file

        Returns:
            Dictionary mapping component names to component information
        """
        components = {}
        lines = content.split('\n')

        # Find components using patterns
        for pattern in self.config["component_patterns"]:
            matches = re.finditer(pattern, content)
            for match in matches:
                component_name = match.group(1)
                if component_name.lower() in ["abstract", "base", "interface"]:
                    # Skip abstract/base/interface components
                    continue

                # Find the start and end lines of the component
                start_line = content[:match.start()].count('\n') + 1
                end_line = self._find_component_end(lines, start_line)

                # Skip components that are too short
                if end_line - start_line < self.config["min_component_lines"]:
                    continue

                # Extract the component content
                component_content = '\n'.join(lines[start_line - 1:end_line])

                # Normalize the content to ignore whitespace and comments
                normalized_content = self._normalize_content(component_content)

                # Skip empty components
                if not normalized_content.strip():
                    continue

                # Calculate a hash of the normalized content
                content_hash = hashlib.md5(normalized_content.encode()).hexdigest()

                # Add the component to the dictionary
                components[component_name] = {
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": component_content,
                    "normalized_content": normalized_content,
                    "content_hash": content_hash
                }

        return components

    def _find_component_end(self, lines: List[str], start_line: int) -> int:
        """
        Find the end line of a component.

        Args:
            lines: Lines of the file
            start_line: Start line of the component

        Returns:
            End line of the component
        """
        # For Python, find the end of the indented block
        indent_level = None
        for i in range(start_line, len(lines)):
            line = lines[i]
            if not line.strip():
                continue

            # Get the indentation level of the first non-empty line
            if indent_level is None:
                indent_level = len(line) - len(line.lstrip())
                continue

            # If we find a line with less indentation, we've reached the end of the component
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                return i

        # If we reach the end of the file, return the last line
        return len(lines)

    def _normalize_content(self, content: str) -> str:
        """
        Normalize content to ignore whitespace and comments.

        Args:
            content: Content to normalize

        Returns:
            Normalized content
        """
        # Remove comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)

        # Remove empty lines
        content = re.sub(r'\n\s*\n', '\n', content)

        # Remove leading/trailing whitespace
        content = content.strip()

        return content

    def _get_service_name(self, file_path: Path) -> str:
        """
        Get the service name from a file path.

        Args:
            file_path: Path to the file

        Returns:
            Service name
        """
        # Convert the file path to a string
        path_str = str(file_path)

        # Extract the service name from the path
        service_match = re.search(r'([a-zA-Z0-9_\-]+)[-_]service', path_str)
        if service_match:
            return service_match.group(0)

        # If no service name is found, return the first directory after the root
        parts = path_str.split(os.sep)
        if len(parts) > 1:
            return parts[1]

        return "unknown"

    def _analyze_duplicates(self) -> None:
        """
        Analyze potential duplicates to find actual duplicates.
        """
        # Filter out components with only one implementation
        self.duplicates = {
            name: impls for name, impls in self.duplicates.items()
            if len(impls) > 1
        }

        # Group implementations by content hash
        for component_name, implementations in list(self.duplicates.items()):
            hash_groups = defaultdict(list)
            for impl in implementations:
                hash_groups[impl["content_hash"]].append(impl)

            # If all implementations have the same hash, they are exact duplicates
            if len(hash_groups) == 1:
                self.stats["potential_duplicates"] += len(implementations) - 1
                self.stats["duplicate_groups"] += 1
                continue

            # If there are multiple hash groups, check for similarity
            similar_groups = []
            for hash1, group1 in hash_groups.items():
                for hash2, group2 in hash_groups.items():
                    if hash1 >= hash2:
                        continue

                    # Check if the groups are similar
                    if self._are_similar(group1[0]["content"], group2[0]["content"]):
                        similar_groups.append((group1, group2))

            # If there are similar groups, count them as potential duplicates
            if similar_groups:
                self.stats["potential_duplicates"] += sum(len(g1) + len(g2) - 2 for g1, g2 in similar_groups)
                self.stats["duplicate_groups"] += len(similar_groups)
            else:
                # If there are no similar groups, remove the component from the duplicates
                del self.duplicates[component_name]

    def _are_similar(self, content1: str, content2: str) -> bool:
        """
        Check if two content strings are similar.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            True if the content strings are similar, False otherwise
        """
        # Normalize the content
        content1 = self._normalize_content(content1)
        content2 = self._normalize_content(content2)

        # Calculate the similarity using the Jaccard index
        tokens1 = set(content1.split())
        tokens2 = set(content2.split())
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        similarity = len(intersection) / len(union) if union else 0

        return similarity >= self.config["similarity_threshold"]

    def generate_report(self, output_file: str = None) -> None:
        """
        Generate a report of potential duplicate components.

        Args:
            output_file: Path to output file (optional)
        """
        report = {
            "stats": self.stats,
            "duplicates": {}
        }

        # Format the duplicates for the report
        for component_name, implementations in self.duplicates.items():
            report["duplicates"][component_name] = [
                {
                    "file_path": impl["file_path"],
                    "start_line": impl["start_line"],
                    "end_line": impl["end_line"],
                    "service": impl["service"]
                }
                for impl in implementations
            ]

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            print(json.dumps(report, indent=2))

    def generate_consolidation_plan(self, output_file: str) -> None:
        """
        Generate a plan to consolidate duplicate components.

        Args:
            output_file: Path to output file
        """
        plan = {
            "common_lib_components": [],
            "service_specific_components": [],
            "implementation_steps": []
        }

        # Analyze each duplicate component
        for component_name, implementations in self.duplicates.items():
            # Check if the component should be moved to common-lib
            services = set(impl["service"] for impl in implementations)
            if len(services) > 1:
                # Component is used in multiple services, move to common-lib
                plan["common_lib_components"].append({
                    "name": component_name,
                    "implementations": [
                        {
                            "file_path": impl["file_path"],
                            "service": impl["service"]
                        }
                        for impl in implementations
                    ],
                    "target_path": f"common-lib/common_lib/{self._get_component_type(component_name).lower()}s/{component_name.lower()}.py"
                })

                # Add implementation steps
                plan["implementation_steps"].append({
                    "step": f"Consolidate {component_name} into common-lib",
                    "description": f"Create a common implementation of {component_name} in common-lib and update all services to use it",
                    "tasks": [
                        f"Create {component_name} in common-lib",
                        f"Update services to use common-lib {component_name}",
                        f"Remove duplicate implementations from services",
                        f"Add tests for common-lib {component_name}"
                    ]
                })
            else:
                # Component is used in only one service, consolidate within the service
                service = list(services)[0]
                plan["service_specific_components"].append({
                    "name": component_name,
                    "service": service,
                    "implementations": [
                        {
                            "file_path": impl["file_path"]
                        }
                        for impl in implementations
                    ],
                    "target_path": f"{service}/{service.replace('-', '_')}/{self._get_component_type(component_name).lower()}s/{component_name.lower()}.py"
                })

                # Add implementation steps
                plan["implementation_steps"].append({
                    "step": f"Consolidate {component_name} within {service}",
                    "description": f"Create a single implementation of {component_name} in {service} and update all modules to use it",
                    "tasks": [
                        f"Create a single implementation of {component_name} in {service}",
                        f"Update modules to use the consolidated {component_name}",
                        f"Remove duplicate implementations from {service}",
                        f"Add tests for the consolidated {component_name}"
                    ]
                })

        with open(output_file, 'w') as f:
            json.dump(plan, f, indent=2)

    def _get_component_type(self, component_name: str) -> str:
        """
        Get the type of a component from its name.

        Args:
            component_name: Name of the component

        Returns:
            Type of the component
        """
        # Check for common suffixes
        suffixes = [
            "Indicator", "Provider", "Client", "Service", "Repository",
            "Factory", "Adapter", "Handler", "Processor", "Validator",
            "Formatter", "Parser", "Serializer", "Deserializer", "Converter",
            "Transformer", "Calculator", "Analyzer", "Generator", "Manager",
            "Controller", "Middleware", "Decorator", "Wrapper", "Proxy",
            "Strategy", "Builder", "Director", "Prototype", "Singleton",
            "Observer", "Visitor", "Command", "Interpreter", "Iterator",
            "Mediator", "Memento", "State", "Template", "Flyweight",
            "Bridge", "Composite", "Facade"
        ]

        for suffix in suffixes:
            if component_name.endswith(suffix):
                return suffix

        # If no suffix is found, return a default type
        return "Component"


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Find potential duplicate components")
    parser.add_argument("--root-dir", default=".", help="Root directory of the codebase")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Path to output file for the report")
    parser.add_argument("--plan", help="Path to output file for the consolidation plan")
    args = parser.parse_args()

    finder = DuplicateComponentFinder(args.root_dir, args.config)
    finder.find_duplicates()
    finder.generate_report(args.output)

    if args.plan:
        finder.generate_consolidation_plan(args.plan)


if __name__ == "__main__":
    main()