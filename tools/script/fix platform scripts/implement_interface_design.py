#!/usr/bin/env python3
"""
Forex Trading Platform Interface-Based Design Implementation

This script implements interface-based design for service interactions in the forex trading platform.
It creates interface definitions and adapter implementations to reduce direct coupling between services.

Usage:
python implement_interface_design.py [--project-root <project_root>] [--service <service_name>]
"""

import os
import sys
import re
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_REPORT_PATH = r"D:\MD\forex_trading_platform\tools\output\dependency-report.json"

# Interface template
INTERFACE_TEMPLATE = """\"\"\"
Interface definition for {service_name} service.
\"\"\"

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class {interface_name}(ABC):
    \"\"\"
    Interface for {service_name} service.
    \"\"\"

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        \"\"\"
        Get the status of the service.

        Returns:
            Service status information
        \"\"\"
        pass

{methods}
"""

# Method template
METHOD_TEMPLATE = """    @abstractmethod
    def {method_name}({method_params}) -> {return_type}:
        \"\"\"
        {method_description}

{param_docs}
        Returns:
            {return_description}
        \"\"\"
        pass
"""

# Adapter template
ADAPTER_TEMPLATE = """\"\"\"
Adapter implementation for {service_name} service.
\"\"\"

from typing import Dict, List, Optional, Any
import requests
from common_lib.exceptions import ServiceException
from common_lib.resilience import circuit_breaker, retry_with_backoff
from ..interfaces.{interface_module} import {interface_name}


class {adapter_name}({interface_name}):
    \"\"\"
    Adapter for {service_name} service.
    \"\"\"

    def __init__(self, base_url: str, timeout: int = 30):
        \"\"\"
        Initialize the adapter.

        Args:
            base_url: Base URL of the service
            timeout: Request timeout in seconds
        \"\"\"
        self.base_url = base_url
        self.timeout = timeout

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def get_status(self) -> Dict[str, Any]:
        \"\"\"
        Get the status of the service.

        Returns:
            Service status information
        \"\"\"
        try:
            response = requests.get(
                f"{{self.base_url}}/api/status",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error getting service status: {{str(e)}}")

{methods}
"""

# Adapter method template
ADAPTER_METHOD_TEMPLATE = """    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def {method_name}({method_params}) -> {return_type}:
        \"\"\"
        {method_description}

{param_docs}
        Returns:
            {return_description}
        \"\"\"
        try:
            # Implement the method using requests to call the service API
            # This is a placeholder implementation
            response = requests.get(
                f"{{self.base_url}}/api/{method_endpoint}",
                params={params_dict},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error calling {method_name}: {{str(e)}}")
"""

# Factory template
FACTORY_TEMPLATE = """\"\"\"
Factory for creating service adapters.
\"\"\"

from typing import Dict, Any, Type
from common_lib.config import ConfigManager
{imports}


class AdapterFactory:
    \"\"\"
    Factory for creating service adapters.
    \"\"\"

    def __init__(self, config_manager: ConfigManager):
        \"\"\"
        Initialize the factory.

        Args:
            config_manager: Configuration manager
        \"\"\"
        self.config_manager = config_manager
        self.adapters = {{}}

{methods}
"""

# Factory method template
FACTORY_METHOD_TEMPLATE = """    def get_{service_snake}_adapter(self) -> {interface_name}:
        \"\"\"
        Get an adapter for the {service_name} service.

        Returns:
            {service_name} service adapter
        \"\"\"
        if "{service_snake}" not in self.adapters:
            config = self.config_manager.get_service_config("{service_kebab}")
            self.adapters["{service_snake}"] = {adapter_name}(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["{service_snake}"]
"""

class InterfaceDesignImplementer:
    """Implements interface-based design for service interactions."""

    def __init__(self, project_root: str, report_path: str, service_name: str = None):
        """
        Initialize the implementer.

        Args:
            project_root: Root directory of the project
            report_path: Path to the dependency report
            service_name: Name of the service to implement (None for all services)
        """
        self.project_root = project_root
        self.report_path = report_path
        self.service_name = service_name
        self.services = {}
        self.dependencies = {}
        self.implementations = []

    def load_dependency_report(self) -> None:
        """Load the dependency report."""
        try:
            with open(self.report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
                self.services = report.get('services', {})
                self.dependencies = report.get('service_dependencies', {})

            logger.info(f"Loaded dependency report from {self.report_path}")
            logger.info(f"Found {len(self.services)} services and {sum(len(deps) for deps in self.dependencies.values())} dependencies")
        except Exception as e:
            logger.error(f"Error loading dependency report: {e}")
            self.services = {}
            self.dependencies = {}

    def generate_interface(self, service_name: str) -> str:
        """
        Generate an interface for a service.

        Args:
            service_name: Name of the service

        Returns:
            Interface content
        """
        # Convert service name to camel case for interface name
        service_parts = service_name.replace('-', '_').split('_')
        interface_name = ''.join(part.capitalize() for part in service_parts) + 'Interface'

        # Generate methods
        methods = []

        # Add example methods based on service name
        if 'data' in service_name:
            methods.append(self.generate_method(
                'get_data',
                'Get data from the service.',
                'dataset_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None',
                'List[Dict[str, Any]]',
                'dataset_id: Dataset identifier\nstart_date: Start date (ISO format)\nend_date: End date (ISO format)',
                'List of data records'
            ))
        elif 'model' in service_name:
            methods.append(self.generate_method(
                'get_model',
                'Get a model from the registry.',
                'model_id: str',
                'Dict[str, Any]',
                'model_id: Model identifier',
                'Model information'
            ))
            methods.append(self.generate_method(
                'list_models',
                'List available models.',
                'tags: Optional[List[str]] = None, limit: int = 100, offset: int = 0',
                'Dict[str, Any]',
                'tags: Filter by tags\nlimit: Maximum number of results\noffset: Result offset',
                'Dictionary with models and pagination information'
            ))
        elif 'trading' in service_name:
            methods.append(self.generate_method(
                'execute_trade',
                'Execute a trade.',
                'trade_request: Dict[str, Any]',
                'Dict[str, Any]',
                'trade_request: Trade request details',
                'Trade execution result'
            ))
            methods.append(self.generate_method(
                'get_trade_status',
                'Get the status of a trade.',
                'trade_id: str',
                'Dict[str, Any]',
                'trade_id: Trade identifier',
                'Trade status information'
            ))
        else:
            # Generic methods for other services
            methods.append(self.generate_method(
                'get_info',
                'Get information from the service.',
                'resource_id: str',
                'Dict[str, Any]',
                'resource_id: Resource identifier',
                'Resource information'
            ))
            methods.append(self.generate_method(
                'list_resources',
                'List available resources.',
                'filter_params: Optional[Dict[str, Any]] = None, limit: int = 100, offset: int = 0',
                'Dict[str, Any]',
                'filter_params: Filter parameters\nlimit: Maximum number of results\noffset: Result offset',
                'Dictionary with resources and pagination information'
            ))

        # Generate interface content
        return INTERFACE_TEMPLATE.format(
            service_name=service_name,
            interface_name=interface_name,
            methods='\n'.join(methods)
        )

    def generate_method(self, method_name: str, method_description: str, method_params: str, return_type: str, param_docs: str, return_description: str) -> str:
        """
        Generate a method definition.

        Args:
            method_name: Name of the method
            method_description: Method description
            method_params: Method parameters
            return_type: Return type
            param_docs: Parameter documentation
            return_description: Return value description

        Returns:
            Method definition
        """
        # Format parameter documentation
        param_docs_formatted = '\n'.join(f'        Args:\n            {line}' for line in param_docs.split('\n'))

        return METHOD_TEMPLATE.format(
            method_name=method_name,
            method_description=method_description,
            method_params=method_params,
            return_type=return_type,
            param_docs=param_docs_formatted,
            return_description=return_description
        )

    def generate_adapter(self, service_name: str) -> str:
        """
        Generate an adapter for a service.

        Args:
            service_name: Name of the service

        Returns:
            Adapter content
        """
        # Convert service name to camel case for interface and adapter names
        service_parts = service_name.replace('-', '_').split('_')
        interface_name = ''.join(part.capitalize() for part in service_parts) + 'Interface'
        adapter_name = ''.join(part.capitalize() for part in service_parts) + 'Adapter'

        # Convert service name to snake_case for module name
        interface_module = service_name.replace('-', '_').lower() + '_interface'

        # Generate methods
        methods = []

        # Add example methods based on service name
        if 'data' in service_name:
            methods.append(self.generate_adapter_method(
                'get_data',
                'Get data from the service.',
                'dataset_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None',
                'List[Dict[str, Any]]',
                'dataset_id: Dataset identifier\nstart_date: Start date (ISO format)\nend_date: End date (ISO format)',
                'List of data records',
                'data',
                "{'dataset_id': dataset_id, 'start_date': start_date, 'end_date': end_date}"
            ))
        elif 'model' in service_name:
            methods.append(self.generate_adapter_method(
                'get_model',
                'Get a model from the registry.',
                'model_id: str',
                'Dict[str, Any]',
                'model_id: Model identifier',
                'Model information',
                'models/{model_id}',
                "{}"
            ))
            methods.append(self.generate_adapter_method(
                'list_models',
                'List available models.',
                'tags: Optional[List[str]] = None, limit: int = 100, offset: int = 0',
                'Dict[str, Any]',
                'tags: Filter by tags\nlimit: Maximum number of results\noffset: Result offset',
                'Dictionary with models and pagination information',
                'models',
                "{'tags': tags, 'limit': limit, 'offset': offset}"
            ))
        elif 'trading' in service_name:
            methods.append(self.generate_adapter_method(
                'execute_trade',
                'Execute a trade.',
                'trade_request: Dict[str, Any]',
                'Dict[str, Any]',
                'trade_request: Trade request details',
                'Trade execution result',
                'trades',
                "{}",
                'post',
                'trade_request'
            ))
            methods.append(self.generate_adapter_method(
                'get_trade_status',
                'Get the status of a trade.',
                'trade_id: str',
                'Dict[str, Any]',
                'trade_id: Trade identifier',
                'Trade status information',
                'trades/{trade_id}',
                "{}"
            ))
        else:
            # Generic methods for other services
            methods.append(self.generate_adapter_method(
                'get_info',
                'Get information from the service.',
                'resource_id: str',
                'Dict[str, Any]',
                'resource_id: Resource identifier',
                'Resource information',
                'resources/{resource_id}',
                "{}"
            ))
            methods.append(self.generate_adapter_method(
                'list_resources',
                'List available resources.',
                'filter_params: Optional[Dict[str, Any]] = None, limit: int = 100, offset: int = 0',
                'Dict[str, Any]',
                'filter_params: Filter parameters\nlimit: Maximum number of results\noffset: Result offset',
                'Dictionary with resources and pagination information',
                'resources',
                "{'limit': limit, 'offset': offset, **(filter_params or {})}"
            ))

        # Generate adapter content
        return ADAPTER_TEMPLATE.format(
            service_name=service_name,
            interface_name=interface_name,
            adapter_name=adapter_name,
            interface_module=interface_module,
            methods='\n'.join(methods)
        )

    def generate_adapter_method(self, method_name: str, method_description: str, method_params: str, return_type: str, param_docs: str, return_description: str, method_endpoint: str, params_dict: str, http_method: str = 'get', body_param: str = None) -> str:
        """
        Generate an adapter method implementation.

        Args:
            method_name: Name of the method
            method_description: Method description
            method_params: Method parameters
            return_type: Return type
            param_docs: Parameter documentation
            return_description: Return value description
            method_endpoint: API endpoint for the method
            params_dict: Dictionary of parameters to pass to the API
            http_method: HTTP method to use (get, post, put, delete)
            body_param: Parameter to use as request body

        Returns:
            Adapter method implementation
        """
        # Format parameter documentation
        param_docs_formatted = '\n'.join(f'        Args:\n            {line}' for line in param_docs.split('\n'))

        # Replace template variables in endpoint
        endpoint_params = re.findall(r'\{([^}]+)\}', method_endpoint)
        for param in endpoint_params:
            method_endpoint = method_endpoint.replace(f'{{{param}}}', f'" + str({param}) + "')

        # Generate HTTP method implementation
        if http_method == 'post':
            http_code = f"""response = requests.post(
                f"{{self.base_url}}/api/{method_endpoint}",
                json={body_param},
                timeout=self.timeout
            )"""
        elif http_method == 'put':
            http_code = f"""response = requests.put(
                f"{{self.base_url}}/api/{method_endpoint}",
                json={body_param},
                timeout=self.timeout
            )"""
        elif http_method == 'delete':
            http_code = f"""response = requests.delete(
                f"{{self.base_url}}/api/{method_endpoint}",
                params={params_dict},
                timeout=self.timeout
            )"""
        else:
            http_code = f"""response = requests.get(
                f"{{self.base_url}}/api/{method_endpoint}",
                params={params_dict},
                timeout=self.timeout
            )"""

        # Create a custom template with the correct HTTP method
        custom_template = ADAPTER_METHOD_TEMPLATE.replace(
            "response = requests.get(\n                f\"{self.base_url}/api/{method_endpoint}\",\n                params={params_dict},\n                timeout=self.timeout\n            )",
            http_code
        )

        return custom_template.format(
            method_name=method_name,
            method_description=method_description,
            method_params=method_params,
            return_type=return_type,
            param_docs=param_docs_formatted,
            return_description=return_description,
            method_endpoint=method_endpoint,
            params_dict=params_dict
        )

    def generate_factory(self) -> str:
        """
        Generate a factory for creating service adapters.

        Returns:
            Factory content
        """
        # Generate imports
        imports = []
        factory_methods = []

        for service_name in self.services:
            # Convert service name to camel case for interface and adapter names
            service_parts = service_name.replace('-', '_').split('_')
            interface_name = ''.join(part.capitalize() for part in service_parts) + 'Interface'
            adapter_name = ''.join(part.capitalize() for part in service_parts) + 'Adapter'

            # Convert service name to snake_case for module name
            service_snake = service_name.replace('-', '_').lower()
            interface_module = service_snake + '_interface'
            adapter_module = service_snake + '_adapter'

            # Add import
            imports.append(f"from ..interfaces.{interface_module} import {interface_name}")
            imports.append(f"from ..adapters.{adapter_module} import {adapter_name}")

            # Add factory method
            factory_methods.append(FACTORY_METHOD_TEMPLATE.format(
                service_name=service_name,
                service_snake=service_snake,
                service_kebab=service_name,
                interface_name=interface_name,
                adapter_name=adapter_name
            ))

        # Generate factory content
        return FACTORY_TEMPLATE.format(
            imports='\n'.join(imports),
            methods='\n'.join(factory_methods)
        )

    def implement_interface_design(self) -> List[str]:
        """
        Implement interface-based design for service interactions.

        Returns:
            List of implementations
        """
        logger.info("Starting interface-based design implementation...")

        # Load dependency report
        self.load_dependency_report()

        if not self.services:
            logger.info("No services found")
            return []

        # Filter services if a specific service was specified
        if self.service_name:
            if self.service_name in self.services:
                services_to_implement = [self.service_name]
            else:
                logger.error(f"Service not found: {self.service_name}")
                return []
        else:
            services_to_implement = list(self.services.keys())

        # Create common-lib directory if it doesn't exist
        common_lib_path = os.path.join(self.project_root, 'common-lib')
        os.makedirs(common_lib_path, exist_ok=True)

        # Create common-lib/interfaces directory if it doesn't exist
        common_lib_interfaces_path = os.path.join(common_lib_path, 'interfaces')
        os.makedirs(common_lib_interfaces_path, exist_ok=True)

        # Create __init__.py file if it doesn't exist
        init_file = os.path.join(common_lib_interfaces_path, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('"""\nInterfaces for service interactions.\n"""\n')
            self.implementations.append(f"Created file: {init_file}")

        # Implement interfaces for each service
        for service_name in services_to_implement:
            # Generate interface
            interface_content = self.generate_interface(service_name)

            # Convert service name to snake_case for file name
            file_name = service_name.replace('-', '_').lower() + '_interface.py'
            file_path = os.path.join(common_lib_interfaces_path, file_name)

            # Write interface file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(interface_content)

            self.implementations.append(f"Created interface: {file_path}")

            # Create adapters directory in each service
            service_path = os.path.join(self.project_root, service_name)
            adapters_path = os.path.join(service_path, 'adapters')
            os.makedirs(adapters_path, exist_ok=True)

            # Create interfaces directory in each service
            interfaces_path = os.path.join(service_path, 'interfaces')
            os.makedirs(interfaces_path, exist_ok=True)

            # Create __init__.py file in interfaces directory if it doesn't exist
            init_file = os.path.join(interfaces_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write('"""\nInterface definitions for the service.\n"""\n')
                self.implementations.append(f"Created file: {init_file}")

            # Create __init__.py file in adapters directory if it doesn't exist
            init_file = os.path.join(adapters_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write('"""\nAdapter implementations for external services.\n"""\n')
                self.implementations.append(f"Created file: {init_file}")

            # For each dependency, create an adapter
            for dependency in self.dependencies.get(service_name, []):
                # Generate adapter
                adapter_content = self.generate_adapter(dependency)

                # Convert dependency name to snake_case for file name
                file_name = dependency.replace('-', '_').lower() + '_adapter.py'
                file_path = os.path.join(adapters_path, file_name)

                # Write adapter file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(adapter_content)

                self.implementations.append(f"Created adapter: {file_path}")

                # Copy interface to service interfaces directory
                interface_file_name = dependency.replace('-', '_').lower() + '_interface.py'
                source_path = os.path.join(common_lib_interfaces_path, interface_file_name)
                target_path = os.path.join(interfaces_path, interface_file_name)

                if os.path.exists(source_path):
                    shutil.copy2(source_path, target_path)
                    self.implementations.append(f"Copied interface: {target_path}")

            # Generate factory
            if self.dependencies.get(service_name, []):
                factory_content = self.generate_factory()
                factory_path = os.path.join(adapters_path, 'adapter_factory.py')

                # Write factory file
                with open(factory_path, 'w', encoding='utf-8') as f:
                    f.write(factory_content)

                self.implementations.append(f"Created adapter factory: {factory_path}")

        logger.info("Interface-based design implementation complete")
        return self.implementations

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Implement interface-based design")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Path to the dependency report"
    )
    parser.add_argument(
        "--service",
        help="Name of the service to implement (default: all services)"
    )
    args = parser.parse_args()

    # Implement interface-based design
    implementer = InterfaceDesignImplementer(args.project_root, args.report_path, args.service)
    implementations = implementer.implement_interface_design()

    # Print summary
    print("\nInterface-Based Design Implementation Summary:")
    print(f"- Applied {len(implementations)} implementations")

    if implementations:
        print("\nImplementations:")
        for i, implementation in enumerate(implementations):
            print(f"  {i+1}. {implementation}")

    # Save results to file
    output_dir = os.path.join(args.project_root, 'tools', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'interface_design_implementations.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_implementations': len(implementations),
            'implementations': implementations
        }, f, indent=2)

    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
