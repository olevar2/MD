"""
Script to apply standardized monitoring to all services.

This script applies the standardized monitoring module to all services in the codebase.
It creates a monitoring.py file in each service based on the template in common-lib.
"""

import os
import shutil
import datetime
from typing import Dict, List, Set, Tuple, Optional, Any

# Service directories to process
SERVICE_DIRS = [
    "analysis-engine-service",
    "data-pipeline-service",
    "trading-gateway-service",
    "ml-integration-service"
]

# Template file path
TEMPLATE_FILE_PATH = "common-lib/templates/service_template/monitoring.py"

# Target file paths
TARGET_FILE_PATHS = [
    "{service_dir}/{service_name}/monitoring.py",
    "{service_dir}/monitoring.py"
]


def apply_standardized_monitoring():
    """
    Apply standardized monitoring to all services.
    """
    # Create output directory for results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tools', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "services_processed": [],
        "files_created": [],
        "files_updated": [],
        "backups_created": []
    }

    # Check if template file exists
    if not os.path.exists(TEMPLATE_FILE_PATH):
        print(f"Template file not found: {TEMPLATE_FILE_PATH}")

        # Save error results
        output_path = os.path.join(output_dir, 'standardized_monitoring_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            results["error"] = f"Template file not found: {TEMPLATE_FILE_PATH}"
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")
        return results

    # Read template file
    with open(TEMPLATE_FILE_PATH, "r", encoding="utf-8") as f:
        template_content = f.read()

    # Process each service
    for service_dir in SERVICE_DIRS:
        service_name = service_dir.replace("-", "_").replace("_service", "")

        # Add to services processed
        results["services_processed"].append(service_dir)

        # Find existing monitoring file
        existing_file = None
        for path_template in TARGET_FILE_PATHS:
            path = path_template.format(service_dir=service_dir, service_name=service_name)
            if os.path.exists(path):
                existing_file = path
                break

        if existing_file:
            # Create backup of existing file
            backup_file = f"{existing_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(existing_file, backup_file)
            print(f"Created backup of existing monitoring file: {backup_file}")
            results["backups_created"].append(backup_file)

            # Replace existing file
            with open(existing_file, "w", encoding="utf-8") as f:
                f.write(template_content)

            print(f"Updated monitoring file: {existing_file}")
            results["files_updated"].append(existing_file)
        else:
            # Create new file
            new_file = TARGET_FILE_PATHS[0].format(service_dir=service_dir, service_name=service_name)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(new_file), exist_ok=True)

            # Write new file
            with open(new_file, "w", encoding="utf-8") as f:
                f.write(template_content)

            print(f"Created new monitoring file: {new_file}")
            results["files_created"].append(new_file)

        # Update main.py to use standardized monitoring
        update_main_file(service_dir, service_name, results)


def update_main_file(service_dir: str, service_name: str, results: Dict[str, Any]):
    """
    Update main.py to use standardized monitoring.

    Args:
        service_dir: Service directory
        service_name: Service name
        results: Results dictionary to update
    """
    # Find main.py
    main_file = None
    for path in [f"{service_dir}/main.py", f"{service_dir}/{service_name}/main.py"]:
        if os.path.exists(path):
            main_file = path
            break

    if not main_file:
        print(f"Main file not found for {service_dir}")
        return

    # Read main file
    with open(main_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if monitoring is already imported
    if "from .monitoring import setup_monitoring" in content or "from monitoring import setup_monitoring" in content:
        print(f"Monitoring already imported in {main_file}")
        return

    # Create backup of main file
    backup_file = f"{main_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(main_file, backup_file)
    print(f"Created backup of main file: {backup_file}")
    results["backups_created"].append(backup_file)

    # Add import statement
    import_statement = "from .monitoring import setup_monitoring"
    if "from " not in content:
        import_statement = f"from {service_name}.monitoring import setup_monitoring"

    # Add setup_monitoring call
    setup_call = "setup_monitoring(app)"

    # Update content
    lines = content.split("\n")

    # Add import statement after other imports
    import_added = False
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            continue

        if not import_added:
            lines.insert(i, "")
            lines.insert(i + 1, import_statement)
            import_added = True
            break

    # Add setup_monitoring call after app creation
    setup_added = False
    for i, line in enumerate(lines):
        if "app = FastAPI(" in line or "app = APIRouter(" in line:
            # Find the next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            # Add setup_monitoring call
            lines.insert(j, "")
            lines.insert(j + 1, setup_call)
            setup_added = True
            break

    # Write updated content
    with open(main_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Updated main file: {main_file}")
    results["files_updated"].append(main_file)


    # Save results to file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tools', 'output')
    output_path = os.path.join(output_dir, 'standardized_monitoring_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    apply_standardized_monitoring()
