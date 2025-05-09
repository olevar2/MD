#!/usr/bin/env python3
"""
Script to standardize API endpoints across the platform.

This script:
1. Scans the codebase for API endpoint implementations
2. Identifies endpoints that don't follow the standardized patterns
3. Generates standardized endpoint implementations
4. Creates a migration plan for updating non-compliant endpoints
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

# FastAPI standardized endpoint template
FASTAPI_ENDPOINT_TEMPLATE = """
@router.{method}(
    "{path}",
    response_model={response_model},
    summary="{summary}",
    description="{description}"
)
async def {function_name}(
{parameters}
) -> {return_type}:
    \"\"\"
    {docstring}
    \"\"\"
    try:
        {implementation}
    except Exception as e:
        logger.error(f"{error_message}: {{str(e)}}")
        raise HTTPException(status_code=500, detail=str(e))
"""

# Express standardized endpoint template
EXPRESS_ENDPOINT_TEMPLATE = """
/**
 * {summary}
 *
 * {description}
 */
router.{method}('{path}', async (req, res) => {{
  try {{
    {implementation}
  }} catch (error) {{
    logger.error(`{error_message}: ${{error.message}}`);
    res.status(500).json({{
      error: {{
        message: error.message,
        type: error.constructor.name,
        code: error.code || 'INTERNAL_SERVER_ERROR'
      }},
      success: false
    }});
  }}
}});
"""


def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/fixing
    return Path(__file__).parent.parent.parent


def find_api_files(service_path: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Find API files in a service directory.

    Args:
        service_path: Path to the service directory

    Returns:
        Tuple of (fastapi_files, express_files, nestjs_files)
    """
    fastapi_files = []
    express_files = []
    nestjs_files = []

    # Find Python files that might contain FastAPI routes
    for py_file in service_path.glob("**/*.py"):
        if py_file.name.startswith("__"):
            continue

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            if "fastapi" in content.lower() and ("@app." in content or "@router." in content):
                fastapi_files.append(py_file)

    # Find JavaScript/TypeScript files that might contain Express routes
    for js_file in service_path.glob("**/*.js"):
        if "node_modules" in str(js_file):
            continue

        with open(js_file, "r", encoding="utf-8") as f:
            content = f.read()
            if ("express" in content.lower() or "router" in content.lower()) and ("app." in content or "router." in content):
                express_files.append(js_file)

    # Find TypeScript files that might contain NestJS routes
    for ts_file in service_path.glob("**/*.ts"):
        if "node_modules" in str(ts_file):
            continue

        with open(ts_file, "r", encoding="utf-8") as f:
            content = f.read()
            if "@nestjs" in content.lower() and any(decorator in content for decorator in ["@Get", "@Post", "@Put", "@Patch", "@Delete"]):
                nestjs_files.append(ts_file)

    return fastapi_files, express_files, nestjs_files


def extract_endpoints(
    fastapi_files: List[Path],
    express_files: List[Path],
    nestjs_files: List[Path]
) -> List[Dict[str, str]]:
    """
    Extract API endpoints from files.

    Args:
        fastapi_files: List of FastAPI files
        express_files: List of Express files
        nestjs_files: List of NestJS files

    Returns:
        List of endpoints with method, path, and file information
    """
    endpoints = []

    # Extract FastAPI endpoints
    for file_path in fastapi_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            # Find router prefix if any
            prefix = ""
            prefix_match = re.search(r'APIRouter\s*\(\s*prefix\s*=\s*[\'"]([^\'"]+)[\'"]', content)
            if prefix_match:
                prefix = prefix_match.group(1)

            # Find routes
            for match in re.finditer(FASTAPI_ROUTE_PATTERN, content):
                method = match.group("method").lower()
                path = match.group("path")

                # Combine prefix and path
                if prefix and not path.startswith("/"):
                    path = f"{prefix}/{path}"
                elif prefix:
                    path = f"{prefix}{path}"

                endpoints.append({
                    "method": method,
                    "path": path,
                    "file": str(file_path),
                    "framework": "fastapi"
                })

    # Extract Express endpoints
    for file_path in express_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            # Find router prefix if any
            prefix = ""
            prefix_match = re.search(r'router\s*=\s*express\.Router\s*\(\s*\)\s*;\s*app\.use\s*\(\s*[\'"]([^\'"]+)[\'"]', content)
            if prefix_match:
                prefix = prefix_match.group(1)

            # Find routes
            for match in re.finditer(EXPRESS_ROUTE_PATTERN, content):
                method = match.group("method").lower()
                path = match.group("path")

                # Combine prefix and path
                if prefix and not path.startswith("/"):
                    path = f"{prefix}/{path}"
                elif prefix:
                    path = f"{prefix}{path}"

                endpoints.append({
                    "method": method,
                    "path": path,
                    "file": str(file_path),
                    "framework": "express"
                })

    # Extract NestJS endpoints
    for file_path in nestjs_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            # Find controller prefix if any
            prefix = ""
            prefix_match = re.search(r'@Controller\s*\(\s*[\'"]([^\'"]+)[\'"]', content)
            if prefix_match:
                prefix = prefix_match.group(1)

            # Find routes
            for decorator in ["Get", "Post", "Put", "Patch", "Delete"]:
                method = decorator.lower()
                pattern = r'@' + decorator + r'\s*\(\s*[\'"](?P<path>[^\'"]*)[\'"]'

                for match in re.finditer(pattern, content):
                    path = match.group("path") or ""

                    # Combine prefix and path
                    if prefix and not path.startswith("/"):
                        path = f"{prefix}/{path}"
                    elif prefix:
                        path = f"{prefix}{path}"
                    elif not path:
                        path = "/"

                    endpoints.append({
                        "method": method,
                        "path": path,
                        "file": str(file_path),
                        "framework": "nestjs"
                    })

    return endpoints


def validate_endpoints(endpoints: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate API endpoints against the platform's API design standards.

    Args:
        endpoints: List of endpoints with method, path, and file information

    Returns:
        Tuple of (compliant_endpoints, non_compliant_endpoints)
    """
    compliant_endpoints = []
    non_compliant_endpoints = []

    for endpoint in endpoints:
        path = endpoint["path"]
        method = endpoint["method"]
        violations = []

        # Check URL structure
        if not re.match(API_STANDARDS["url_structure"]["pattern"], path):
            violations.append({
                "type": "url_structure",
                "description": API_STANDARDS["url_structure"]["description"],
                "expected_pattern": API_STANDARDS["url_structure"]["pattern"]
            })

        # Check resource naming
        if not re.match(API_STANDARDS["resource_naming"]["pattern"], path):
            violations.append({
                "type": "resource_naming",
                "description": API_STANDARDS["resource_naming"]["description"],
                "expected_pattern": API_STANDARDS["resource_naming"]["pattern"]
            })

        # Check if path ends with an action and method is POST
        action_match = re.match(API_STANDARDS["action_pattern"]["pattern"], path)
        if action_match and method != "post":
            violations.append({
                "type": "action_method",
                "description": "Actions should use POST method",
                "expected_method": "post",
                "actual_method": method
            })

        # Add endpoint to appropriate list
        endpoint_info = {
            "method": method,
            "path": path,
            "file": endpoint["file"],
            "framework": endpoint["framework"]
        }

        if violations:
            endpoint_info["violations"] = violations
            non_compliant_endpoints.append(endpoint_info)
        else:
            compliant_endpoints.append(endpoint_info)

    return compliant_endpoints, non_compliant_endpoints


def generate_standardized_path(endpoint: Dict[str, Any]) -> str:
    """
    Generate a standardized path for a non-compliant endpoint.

    Args:
        endpoint: Endpoint information

    Returns:
        Standardized path
    """
    path = endpoint["path"]

    # Extract service name from file path
    file_path = Path(endpoint["file"])
    service_name = file_path.parts[0] if len(file_path.parts) > 0 else "unknown-service"

    # Remove service name suffix if present
    service_name = service_name.replace("-service", "")

    # Extract resource name from path
    path_parts = path.strip("/").split("/")
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
        if endpoint["method"] == "post" and path_parts[-1].isalpha():
            # Action endpoint
            return f"/v1/{service_name}/{resource_name}/{path_parts[1]}/{path_parts[-1]}"
        else:
            # Sub-resource endpoint
            sub_resource = path_parts[2]
            if not sub_resource.endswith("s"):
                sub_resource = f"{sub_resource}s"
            sub_resource = re.sub(r'(?<!^)(?=[A-Z])', '-', sub_resource).lower()
            return f"/v1/{service_name}/{resource_name}/{path_parts[1]}/{sub_resource}"

    # Fallback
    return f"/v1/{service_name}/{resource_name}"


def generate_migration_plan(non_compliant_endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a migration plan for non-compliant endpoints.

    Args:
        non_compliant_endpoints: List of non-compliant endpoints

    Returns:
        Migration plan
    """
    migration_plan = {
        "endpoints": [],
        "files": set()
    }

    for endpoint in non_compliant_endpoints:
        standardized_path = generate_standardized_path(endpoint)

        migration_plan["endpoints"].append({
            "original": {
                "method": endpoint["method"],
                "path": endpoint["path"],
                "file": endpoint["file"],
                "framework": endpoint["framework"]
            },
            "standardized": {
                "method": "post" if re.match(API_STANDARDS["action_pattern"]["pattern"], standardized_path) else endpoint["method"],
                "path": standardized_path
            },
            "violations": endpoint.get("violations", [])
        })

        migration_plan["files"].add(endpoint["file"])

    migration_plan["files"] = list(migration_plan["files"])

    return migration_plan


def generate_migration_report(migration_plan: Dict[str, Any]) -> str:
    """
    Generate a migration report for non-compliant endpoints.

    Args:
        migration_plan: Migration plan

    Returns:
        Migration report as a string
    """
    report = [
        "# API Endpoint Migration Plan",
        "",
        "This document outlines the plan for migrating non-compliant API endpoints to follow the platform's API design standards.",
        "",
        "## Summary",
        "",
        f"- Total endpoints to migrate: {len(migration_plan['endpoints'])}",
        f"- Files to update: {len(migration_plan['files'])}",
        "",
        "## Migration Steps",
        "",
        "1. Update each endpoint to follow the standardized pattern",
        "2. Add appropriate redirects for backward compatibility",
        "3. Update client code to use the new endpoints",
        "4. Monitor for errors during the transition period",
        "5. Remove redirects after all clients have been updated",
        "",
        "## Endpoint Migrations",
        ""
    ]

    for i, migration in enumerate(migration_plan["endpoints"], 1):
        original = migration["original"]
        standardized = migration["standardized"]

        report.extend([
            f"### {i}. {original['method'].upper()} {original['path']} to {standardized['method'].upper()} {standardized['path']}",
            "",
            f"**File:** {original['file']}",
            f"**Framework:** {original['framework']}",
            "",
            "**Violations:**",
            ""
        ])

        for violation in migration["violations"]:
            report.append(f"- **{violation['type']}**: {violation['description']}")
            if "expected_pattern" in violation:
                report.append(f"  - Expected pattern: `{violation['expected_pattern']}`")
            if "expected_method" in violation:
                report.append(f"  - Expected method: `{violation['expected_method']}`, actual: `{original['method']}`")

        report.extend([
            "",
            "**Migration Code:**",
            ""
        ])

        if original["framework"] == "fastapi":
            report.extend([
                "```python",
                "# Original endpoint",
                f"@router.{original['method']}(\"{original['path']}\")",
                "async def original_function():",
                "    # Implementation",
                "    pass",
                "",
                "# Standardized endpoint",
                f"@router.{standardized['method']}(\"{standardized['path']}\")",
                "async def standardized_function():",
                "    # Implementation",
                "    pass",
                "",
                "# Redirect for backward compatibility",
                f"@router.{original['method']}(\"{original['path']}\")",
                "async def original_function_redirect():",
                f"    return RedirectResponse(url=\"{standardized['path']}\")",
                "```"
            ])
        elif original["framework"] in ["express", "nestjs"]:
            report.extend([
                "```javascript",
                "// Original endpoint",
                f"router.{original['method']}('{original['path']}', (req, res) => {{",
                "  // Implementation",
                "});",
                "",
                "// Standardized endpoint",
                f"router.{standardized['method']}('{standardized['path']}', (req, res) => {{",
                "  // Implementation",
                "});",
                "",
                "// Redirect for backward compatibility",
                f"router.{original['method']}('{original['path']}', (req, res) => {{",
                f"  res.redirect('{standardized['path']}');",
                "});",
                "```"
            ])

        report.append("")

    report.extend([
        "## Files to Update",
        ""
    ])

    for file_path in sorted(migration_plan["files"]):
        report.append(f"- {file_path}")

    return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standardize API endpoints")
    parser.add_argument("--service", help="Path to a specific service to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all services")
    parser.add_argument("--report", help="Path to save the migration report")
    args = parser.parse_args()

    repo_root = get_repo_root()

    if args.service:
        service_path = Path(args.service)
        if not service_path.is_absolute():
            service_path = repo_root / service_path

        if not service_path.exists() or not service_path.is_dir():
            print(f"Error: Service path {service_path} does not exist or is not a directory")
            return 1

        services = [service_path]
    elif args.all:
        # Find all potential service directories
        services = []
        for item in repo_root.iterdir():
            if item.is_dir() and not item.name.startswith(".") and item.name not in ["tools", "docs", "common-lib", "common-js-lib"]:
                services.append(item)
    else:
        parser.print_help()
        return 1

    all_endpoints = []

    # Analyze each service
    for service_path in services:
        service_name = service_path.name
        print(f"Analyzing API endpoints in {service_name}...")

        # Find API files
        fastapi_files, express_files, nestjs_files = find_api_files(service_path)

        if not fastapi_files and not express_files and not nestjs_files:
            print(f"  No API files found in {service_name}")
            continue

        # Extract endpoints
        endpoints = extract_endpoints(fastapi_files, express_files, nestjs_files)

        if not endpoints:
            print(f"  No API endpoints found in {service_name}")
            continue

        # Validate endpoints
        compliant_endpoints, non_compliant_endpoints = validate_endpoints(endpoints)

        # Print summary
        total_endpoints = len(compliant_endpoints) + len(non_compliant_endpoints)
        compliance_pct = int(len(compliant_endpoints) / total_endpoints * 100) if total_endpoints > 0 else 100

        print(f"  Total Endpoints: {total_endpoints}")
        print(f"  Compliant: {len(compliant_endpoints)} ({compliance_pct}%)")
        print(f"  Non-Compliant: {len(non_compliant_endpoints)} ({100 - compliance_pct}%)")

        if non_compliant_endpoints:
            print("  Non-compliant endpoints:")
            for endpoint in sorted(non_compliant_endpoints, key=lambda e: e["path"])[:5]:  # Show only first 5
                print(f"    - {endpoint['method'].upper()} {endpoint['path']}")
            if len(non_compliant_endpoints) > 5:
                print(f"    - ... and {len(non_compliant_endpoints) - 5} more")

        all_endpoints.extend(non_compliant_endpoints)

    # Generate migration plan
    if all_endpoints:
        migration_plan = generate_migration_plan(all_endpoints)

        # Generate migration report
        report = generate_migration_report(migration_plan)

        if args.report:
            report_path = Path(args.report)
            if not report_path.is_absolute():
                report_path = repo_root / report_path

            # Create directory if it doesn't exist
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(f"\nMigration report saved to {report_path}")
        else:
            print("\nMigration Plan:")
            print(report)
    else:
        print("\nAll endpoints are compliant with the API design standards.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
