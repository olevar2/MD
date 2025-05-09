#!/usr/bin/env python3
"""
Script to help migrate API endpoints to follow the platform's standardized patterns.

This script:
1. Takes an existing API endpoint file
2. Analyzes the endpoints
3. Generates standardized versions of the endpoints
4. Creates a new file with the standardized endpoints
5. Adds backward compatibility redirects
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

# API endpoint patterns for different frameworks
FASTAPI_ROUTE_PATTERN = r'@(?:router|app)\.(?P<method>get|post|put|patch|delete)\s*\(\s*[\'"](?P<path>[^\'"]+)[\'"]'
EXPRESS_ROUTE_PATTERN = r'(?:router|app)\.(?P<method>get|post|put|patch|delete)\s*\(\s*[\'"](?P<path>[^\'"]+)[\'"]'
NESTJS_ROUTE_PATTERN = r'@(?:Get|Post|Put|Patch|Delete)\s*\(\s*[\'"](?P<path>[^\'"]+)[\'"]'

# API design standards
API_STANDARDS = {
    "url_structure": {
        "pattern": r'^/?v\d+/[a-z-]+/[a-z-]+(?:/[a-z0-9-]+(?:/[a-z-]+)?)*$',
        "description": "URLs should follow the pattern: /v{version}/{service}/{resource}/{id?}/{sub-resource?}"
    },
    "resource_naming": {
        "pattern": r'^/?v\d+/[a-z-]+/[a-z-]+',
        "description": "Resource names should be plural nouns in kebab-case"
    },
    "http_methods": {
        "get": "Read operations",
        "post": "Create operations",
        "put": "Full update operations",
        "patch": "Partial update operations",
        "delete": "Delete operations"
    },
    "action_pattern": {
        "pattern": r'^/?v\d+/[a-z-]+/[a-z-]+/[a-z0-9-]+/[a-z-]+$',
        "description": "Actions should follow the pattern: /v{version}/{service}/{resource}/{id}/{action}"
    }
}

def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/fixing
    return Path(__file__).parent.parent.parent

def extract_function_name(file_content: str, endpoint_match: re.Match) -> Optional[str]:
    """
    Extract the function name associated with an endpoint.

    Args:
        file_content: Content of the file
        endpoint_match: Regex match for the endpoint

    Returns:
        Function name if found, None otherwise
    """
    # Get the position of the endpoint in the file
    start_pos = endpoint_match.start()

    # Find the next 'def' or 'async def' after the endpoint
    def_match = re.search(r'(?:async\s+)?def\s+([a-zA-Z0-9_]+)', file_content[start_pos:])
    if def_match:
        return def_match.group(1)

    return None

def generate_standardized_path(path: str, service_name: str) -> str:
    """
    Generate a standardized path for a non-compliant endpoint.

    Args:
        path: Original path
        service_name: Service name

    Returns:
        Standardized path
    """
    # Remove service name suffix if present
    service_name = service_name.replace("-service", "")

    # Extract resource name from path
    path_parts = path.strip("/").split("/")

    # Skip 'api' prefix if present
    if path_parts and path_parts[0] == "api":
        path_parts = path_parts[1:]

    # Skip version prefix if present
    if path_parts and path_parts[0].startswith("v"):
        path_parts = path_parts[1:]

    resource_name = path_parts[0] if path_parts else "resources"

    # Check if resource name is plural
    if not resource_name.endswith("s"):
        resource_name = f"{resource_name}s"

    # Convert to kebab-case
    resource_name = re.sub(r'(?<!^)(?=[A-Z])', '-', resource_name).lower()

    # Build standardized path
    if len(path_parts) == 1:
        # Collection endpoint
        return f"/v1/{service_name}/{resource_name}"
    elif len(path_parts) == 2:
        # Resource endpoint
        return f"/v1/{service_name}/{resource_name}/{path_parts[1]}"
    elif len(path_parts) >= 3:
        # Sub-resource or action endpoint
        if path_parts[-1].isalpha():
            # Action endpoint
            return f"/v1/{service_name}/{resource_name}/{path_parts[1]}/actions/{path_parts[-1]}"
        else:
            # Sub-resource endpoint
            sub_resource = path_parts[2]
            if not sub_resource.endswith("s"):
                sub_resource = f"{sub_resource}s"
            sub_resource = re.sub(r'(?<!^)(?=[A-Z])', '-', sub_resource).lower()
            return f"/v1/{service_name}/{resource_name}/{path_parts[1]}/{sub_resource}"

    # Fallback
    return f"/v1/{service_name}/{resource_name}"

def migrate_fastapi_endpoint(
    file_content: str,
    original_path: str,
    standardized_path: str,
    method: str,
    function_name: Optional[str]
) -> Tuple[str, str]:
    """
    Migrate a FastAPI endpoint to follow the standardized pattern.

    Args:
        file_content: Content of the file
        original_path: Original path
        standardized_path: Standardized path
        method: HTTP method
        function_name: Function name

    Returns:
        Tuple of (standardized endpoint code, backward compatibility code)
    """
    # Find the endpoint decorator and function definition
    pattern = fr'@router\.{method}\s*\(\s*[\'"]({re.escape(original_path)})[\'"].*?\)\s*(?:async\s+)?def\s+{function_name}'
    endpoint_match = re.search(pattern, file_content, re.DOTALL)

    if not endpoint_match:
        return "", ""

    # Extract the full endpoint code
    endpoint_start = endpoint_match.start()

    # Find the end of the function
    function_pattern = fr'(?:async\s+)?def\s+{function_name}.*?:'
    function_match = re.search(function_pattern, file_content[endpoint_start:], re.DOTALL)

    if not function_match:
        return "", ""

    function_start = endpoint_start + function_match.start()

    # Find the next function or the end of the file
    next_function_match = re.search(r'(?:async\s+)?def\s+', file_content[function_start + 1:], re.DOTALL)

    if next_function_match:
        function_end = function_start + 1 + next_function_match.start()
    else:
        function_end = len(file_content)

    # Extract the full endpoint code
    endpoint_code = file_content[endpoint_start:function_end]

    # Create standardized endpoint
    standardized_endpoint = endpoint_code.replace(
        f'@router.{method}("{original_path}"',
        f'@router.{method}("{standardized_path}"'
    )

    # Create backward compatibility endpoint
    backward_compatibility = f"""
@router.{method}("{original_path}")
async def {function_name}_legacy(*args, **kwargs):
    \"\"\"Legacy endpoint for backward compatibility. Use {standardized_path} instead.\"\"\"
    logger.info(f"Legacy endpoint {original_path} called - consider migrating to {standardized_path}")
    return await {function_name}(*args, **kwargs)
"""

    return standardized_endpoint, backward_compatibility

def migrate_file(file_path: Path, service_name: str, output_path: Optional[Path] = None) -> None:
    """
    Migrate a file containing API endpoints to follow the standardized patterns.

    Args:
        file_path: Path to the file
        service_name: Service name
        output_path: Optional path to write the migrated file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    # Find all FastAPI endpoints
    endpoints = []
    for match in re.finditer(FASTAPI_ROUTE_PATTERN, file_content):
        method = match.group("method").lower()
        path = match.group("path")
        function_name = extract_function_name(file_content, match)

        if function_name:
            endpoints.append({
                "method": method,
                "path": path,
                "function_name": function_name
            })

    if not endpoints:
        print(f"No endpoints found in {file_path}")
        return

    # Generate standardized endpoints
    standardized_endpoints = []
    backward_compatibility = []

    for endpoint in endpoints:
        standardized_path = generate_standardized_path(endpoint["path"], service_name)

        standardized_code, compatibility_code = migrate_fastapi_endpoint(
            file_content,
            endpoint["path"],
            standardized_path,
            endpoint["method"],
            endpoint["function_name"]
        )

        if standardized_code:
            standardized_endpoints.append(standardized_code)
            backward_compatibility.append(compatibility_code)

    # Create migrated file content
    migrated_content = file_content

    # Add imports if needed
    if "import logging" not in migrated_content and "from logging import" not in migrated_content:
        migrated_content = "import logging\n" + migrated_content

    if "logger =" not in migrated_content:
        migrated_content = migrated_content.replace(
            "import logging\n",
            "import logging\n\nlogger = logging.getLogger(__name__)\n"
        )

    # Add standardized endpoints
    migrated_content += "\n\n# Standardized endpoints\n"
    migrated_content += "\n\n".join(standardized_endpoints)

    # Add backward compatibility
    migrated_content += "\n\n# Backward compatibility\n"
    migrated_content += "\n\n".join(backward_compatibility)

    # Write migrated file
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(migrated_content)
        print(f"Migrated file written to {output_path}")
    else:
        print(migrated_content)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Migrate API endpoints to follow standardized patterns")
    parser.add_argument("--file", required=True, help="Path to the file containing API endpoints")
    parser.add_argument("--service", required=True, help="Service name (e.g., analysis-engine)")
    parser.add_argument("--output", help="Path to write the migrated file")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = get_repo_root() / file_path

    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return 1

    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = get_repo_root() / output_path

    migrate_file(file_path, args.service, output_path)

    return 0

if __name__ == "__main__":
    sys.exit(main())