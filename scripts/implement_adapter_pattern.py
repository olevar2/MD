#!/usr/bin/env python3
"""
Script to implement the interface-based adapter pattern for services.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

def load_adapter_analysis(file_path: str) -> Dict:
    """Load adapter analysis from a markdown file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract missing adapters section
    missing_adapters_section = re.search(r'## Missing Adapters\n\n(.*?)(?:\n\n##|\Z)', content, re.DOTALL)
    if not missing_adapters_section:
        return {'missing_adapters': []}

    missing_adapters_text = missing_adapters_section.group(1)

    # Parse missing adapters
    missing_adapters = []
    for line in missing_adapters_text.strip().split('\n'):
        if '|' not in line or line.startswith('|--'):
            continue

        parts = [part.strip() for part in line.split('|')]
        if len(parts) < 5:
            continue

        service = parts[1]
        interface = parts[2]
        implementation = parts[3]
        issue = parts[4]

        if service and interface and implementation and 'Missing adapter' in issue:
            missing_adapters.append({
                'service': service,
                'interface': interface,
                'implementation': implementation
            })

    return {'missing_adapters': missing_adapters}

def create_adapter_template(service: str, interface: str, implementation: str) -> str:
    """Create a template for an adapter class."""
    interface_name = interface.split('.')[-1]
    implementation_name = implementation.split('.')[-1]

    template = f"""#!/usr/bin/env python3
\"\"\"
{implementation_name} - Adapter for {interface_name}
\"\"\"

from typing import Dict, List, Optional, Any

from common_lib.interfaces import {interface_name}
from common_lib.errors import ServiceError, NotFoundError
from common_lib.resilience.decorators import with_circuit_breaker, with_retry, with_timeout

class {implementation_name}({interface_name}):
    \"\"\"
    Adapter implementation for {interface_name}.
    \"\"\"

    def __init__(self, service_client=None):
    """
      init  .
    
    Args:
        service_client: Description of service_client
    
    """

        \"\"\"
        Initialize the adapter with an optional service client.

        Args:
            service_client: Client for the service this adapter communicates with
        \"\"\"
        self.service_client = service_client

    # TODO: Implement interface methods
    # Add methods required by the interface here
"""

    return template

def create_adapter_factory_template(service: str, adapters: List[Dict]) -> str:
    """Create a template for an adapter factory."""
    service_name = service.split('-')[0] if '-' in service else service

    imports = []
    adapter_instances = []

    for adapter in adapters:
        implementation = adapter['implementation']
        adapter_class = implementation.split('.')[-1]
        imports.append(f"from .{adapter_class.lower()} import {adapter_class}")
        adapter_instances.append(f"    @classmethod\n    def get_{adapter_class.lower()}(cls) -> {adapter_class}:\n        \"\"\"Get an instance of {adapter_class}.\"\"\"\n        return {adapter_class}()")

    template = f"""#!/usr/bin/env python3
\"\"\"
Adapter factory for {service_name} service.
\"\"\"

from typing import Dict, List, Optional, Any

{chr(10).join(imports)}

class AdapterFactory:
    \"\"\"
    Factory for creating adapter instances.
    \"\"\"

{chr(10).join(adapter_instances)}
"""

    return template

def create_adapter_files(root_dir: str, missing_adapters: List[Dict]) -> None:
    """Create adapter files for missing adapters."""
    # Group adapters by service
    service_adapters = {}
    for adapter in missing_adapters:
        service = adapter['service']
        if service not in service_adapters:
            service_adapters[service] = []
        service_adapters[service].append(adapter)

    # Create adapter files
    for service, adapters in service_adapters.items():
        # Skip common-lib adapters
        if service == 'common-lib':
            continue

        # Create service directory if it doesn't exist
        service_dir = os.path.join(root_dir, service)
        if not os.path.exists(service_dir):
            print(f"Service directory not found: {service_dir}")
            continue

        # Create adapters directory if it doesn't exist
        adapters_dir = os.path.join(service_dir, 'adapters')
        os.makedirs(adapters_dir, exist_ok=True)

        # Create __init__.py if it doesn't exist
        init_file = os.path.join(adapters_dir, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Adapters package."""\n')

        # Create adapter files
        for adapter in adapters:
            interface = adapter['interface']
            implementation = adapter['implementation']

            # Extract adapter class name
            adapter_class = implementation.split('.')[-1]

            # Create adapter file
            adapter_file = os.path.join(adapters_dir, f"{adapter_class.lower()}.py")

            if not os.path.exists(adapter_file):
                with open(adapter_file, 'w') as f:
                    f.write(create_adapter_template(service, interface, adapter_class))
                print(f"Created adapter file: {adapter_file}")

        # Create adapter factory
        factory_file = os.path.join(adapters_dir, 'adapter_factory.py')
        if not os.path.exists(factory_file):
            with open(factory_file, 'w') as f:
                f.write(create_adapter_factory_template(service, adapters))
            print(f"Created adapter factory: {factory_file}")

def main():
    """Main function to implement the adapter pattern."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    adapter_analysis_file = os.path.join(root_dir, 'adapter_analysis.md')

    # Check if adapter analysis file exists
    if not os.path.exists(adapter_analysis_file):
        print(f"Adapter analysis file not found: {adapter_analysis_file}")
        return

    # Load adapter analysis
    adapter_analysis = load_adapter_analysis(adapter_analysis_file)

    # Create adapter files
    create_adapter_files(root_dir, adapter_analysis['missing_adapters'])

    print("Adapter implementation completed.")

if __name__ == '__main__':
    main()
