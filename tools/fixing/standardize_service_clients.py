#!/usr/bin/env python3
"""
Script to standardize service client implementations across the platform.

This script:
1. Scans the codebase for service client implementations
2. Identifies clients that don't follow the standardized template
3. Updates these clients to follow the standardized template
4. Adds proper error handling, resilience patterns, and logging
"""

import os
import sys
import re
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


# Client implementation patterns
PYTHON_CLIENT_PATTERN = r'class\s+(\w+Client)\s*\('
JS_CLIENT_PATTERN = r'class\s+(\w+Client)\s+extends'

# Required methods and properties for standardized clients
PYTHON_CLIENT_REQUIREMENTS = {
    "base_classes": ["BaseServiceClient"],
    "methods": ["get", "post", "put", "delete", "patch"],
    "properties": ["logger", "config"],
    "error_handling": ["try", "except", "raise"],
    "resilience": ["circuit_breaker", "retry", "timeout", "bulkhead"]
}

JS_CLIENT_REQUIREMENTS = {
    "base_classes": ["BaseServiceClient"],
    "methods": ["get", "post", "put", "delete", "patch"],
    "properties": ["logger", "config"],
    "error_handling": ["try", "catch", "throw"],
    "resilience": ["circuitBreaker", "retry", "timeout", "bulkhead"]
}

# Python client template
PYTHON_CLIENT_TEMPLATE = """import logging
from typing import Dict, Any, Optional, Union, List

from common_lib.clients.base_client import BaseServiceClient, ClientConfig
from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError
)


class {client_name}(BaseServiceClient):
    """
    Client for interacting with {service_name}.
    
    This client provides methods for communicating with the {service_name}
    using standardized patterns for error handling, resilience, and logging.
    """
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """
        Initialize the client.
        
        Args:
            config: Client configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_{resource_name}(self, {resource_id_param}: str) -> Dict[str, Any]:
        """
        Get a {resource_name} by ID.
        
        Args:
            {resource_id_param}: {resource_name} ID
            
        Returns:
            {resource_name} data
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
        """
        self.logger.info(f"Getting {resource_name} with ID: {{{resource_id_param}}}")
        try:
            return await self.get(f"{resource_name}s/{{{resource_id_param}}}")
        except ClientError as e:
            self.logger.error(f"Error getting {resource_name}: {str(e)}")
            raise
    
    async def create_{resource_name}(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new {resource_name}.
        
        Args:
            data: {resource_name} data
            
        Returns:
            Created {resource_name} data
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the data is invalid
        """
        self.logger.info(f"Creating {resource_name}")
        try:
            return await self.post(f"{resource_name}s", data)
        except ClientError as e:
            self.logger.error(f"Error creating {resource_name}: {str(e)}")
            raise
    
    async def update_{resource_name}(self, {resource_id_param}: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a {resource_name}.
        
        Args:
            {resource_id_param}: {resource_name} ID
            data: Updated {resource_name} data
            
        Returns:
            Updated {resource_name} data
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the data is invalid
        """
        self.logger.info(f"Updating {resource_name} with ID: {{{resource_id_param}}}")
        try:
            return await self.put(f"{resource_name}s/{{{resource_id_param}}}", data)
        except ClientError as e:
            self.logger.error(f"Error updating {resource_name}: {str(e)}")
            raise
    
    async def delete_{resource_name}(self, {resource_id_param}: str) -> Dict[str, Any]:
        """
        Delete a {resource_name}.
        
        Args:
            {resource_id_param}: {resource_name} ID
            
        Returns:
            Deletion confirmation
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
        """
        self.logger.info(f"Deleting {resource_name} with ID: {{{resource_id_param}}}")
        try:
            return await self.delete(f"{resource_name}s/{{{resource_id_param}}}")
        except ClientError as e:
            self.logger.error(f"Error deleting {resource_name}: {str(e)}")
            raise
"""

# JavaScript client template
JS_CLIENT_TEMPLATE = """import { BaseServiceClient } from 'common-js-lib/clients/BaseServiceClient';
import { 
  ClientError,
  ClientConnectionError,
  ClientTimeoutError,
  ClientValidationError,
  ClientAuthenticationError
} from 'common-js-lib/clients/errors';

/**
 * Client for interacting with {service_name}.
 * 
 * This client provides methods for communicating with the {service_name}
 * using standardized patterns for error handling, resilience, and logging.
 */
export class {client_name} extends BaseServiceClient {
  /**
   * Initialize the client.
   * 
   * @param {Object} config - Client configuration
   */
  constructor(config) {
    super(config);
  }
  
  /**
   * Get a {resource_name} by ID.
   * 
   * @param {string} {resource_id_param} - {resource_name} ID
   * @returns {Promise<Object>} {resource_name} data
   * @throws {ClientError} If the request fails
   * @throws {ClientConnectionError} If connection to the service fails
   * @throws {ClientTimeoutError} If the request times out
   */
  async get{resource_name_pascal}({resource_id_param}) {
    this.logger.info(`Getting {resource_name} with ID: ${{{resource_id_param}}}`);
    try {
      return await this.get(`{resource_name}s/${{{resource_id_param}}}`);
    } catch (error) {
      this.logger.error(`Error getting {resource_name}: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Create a new {resource_name}.
   * 
   * @param {Object} data - {resource_name} data
   * @returns {Promise<Object>} Created {resource_name} data
   * @throws {ClientError} If the request fails
   * @throws {ClientConnectionError} If connection to the service fails
   * @throws {ClientTimeoutError} If the request times out
   * @throws {ClientValidationError} If the data is invalid
   */
  async create{resource_name_pascal}(data) {
    this.logger.info(`Creating {resource_name}`);
    try {
      return await this.post('{resource_name}s', data);
    } catch (error) {
      this.logger.error(`Error creating {resource_name}: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Update a {resource_name}.
   * 
   * @param {string} {resource_id_param} - {resource_name} ID
   * @param {Object} data - Updated {resource_name} data
   * @returns {Promise<Object>} Updated {resource_name} data
   * @throws {ClientError} If the request fails
   * @throws {ClientConnectionError} If connection to the service fails
   * @throws {ClientTimeoutError} If the request times out
   * @throws {ClientValidationError} If the data is invalid
   */
  async update{resource_name_pascal}({resource_id_param}, data) {
    this.logger.info(`Updating {resource_name} with ID: ${{{resource_id_param}}}`);
    try {
      return await this.put(`{resource_name}s/${{{resource_id_param}}}`, data);
    } catch (error) {
      this.logger.error(`Error updating {resource_name}: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Delete a {resource_name}.
   * 
   * @param {string} {resource_id_param} - {resource_name} ID
   * @returns {Promise<Object>} Deletion confirmation
   * @throws {ClientError} If the request fails
   * @throws {ClientConnectionError} If connection to the service fails
   * @throws {ClientTimeoutError} If the request times out
   */
  async delete{resource_name_pascal}({resource_id_param}) {
    this.logger.info(`Deleting {resource_name} with ID: ${{{resource_id_param}}}`);
    try {
      return await this.delete(`{resource_name}s/${{{resource_id_param}}}`);
    } catch (error) {
      this.logger.error(`Error deleting {resource_name}: ${error.message}`);
      throw error;
    }
  }
}
"""


def get_repo_root() -> Path:
    """Get the root directory of the repository."""
    # Assuming this script is in tools/fixing
    return Path(__file__).parent.parent.parent


def find_client_files(repo_root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find service client files in the repository.
    
    Args:
        repo_root: Path to the repository root
        
    Returns:
        Tuple of (python_client_files, js_client_files)
    """
    python_client_files = []
    js_client_files = []
    
    # Find Python client files
    for py_file in repo_root.glob("**/*client*.py"):
        if "node_modules" in str(py_file) or "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                if re.search(PYTHON_CLIENT_PATTERN, content):
                    python_client_files.append(py_file)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    # Find JavaScript/TypeScript client files
    for js_file in repo_root.glob("**/*[cC]lient*.[jt]s"):
        if "node_modules" in str(js_file):
            continue
        
        try:
            with open(js_file, "r", encoding="utf-8") as f:
                content = f.read()
                if re.search(JS_CLIENT_PATTERN, content):
                    js_client_files.append(js_file)
        except Exception as e:
            print(f"Error reading {js_file}: {e}")
    
    return python_client_files, js_client_files


def analyze_python_client(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a Python service client file.
    
    Args:
        file_path: Path to the Python client file
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        "file": str(file_path),
        "language": "python",
        "client_name": None,
        "base_classes": [],
        "methods": [],
        "properties": [],
        "has_error_handling": False,
        "has_logging": False,
        "has_resilience": False,
        "violations": []
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Extract client name
            client_match = re.search(PYTHON_CLIENT_PATTERN, content)
            if client_match:
                result["client_name"] = client_match.group(1)
            
            # Parse the file with ast
            tree = ast.parse(content)
            
            # Find client class
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == result["client_name"]:
                    # Get base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            result["base_classes"].append(base.id)
                        elif isinstance(base, ast.Attribute):
                            result["base_classes"].append(base.attr)
                    
                    # Get methods
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            result["methods"].append(child.name)
                            
                            # Check for error handling
                            for node in ast.walk(child):
                                if isinstance(node, ast.Try):
                                    result["has_error_handling"] = True
                                    break
                    
                    break
            
            # Check for logging
            result["has_logging"] = "logger" in content and ("self.logger" in content or "logging.getLogger" in content)
            
            # Check for resilience patterns
            result["has_resilience"] = any(pattern in content for pattern in PYTHON_CLIENT_REQUIREMENTS["resilience"])
            
            # Check for violations
            for base_class in PYTHON_CLIENT_REQUIREMENTS["base_classes"]:
                if base_class not in result["base_classes"]:
                    result["violations"].append(f"Missing base class: {base_class}")
            
            for method in PYTHON_CLIENT_REQUIREMENTS["methods"]:
                if method not in result["methods"]:
                    result["violations"].append(f"Missing method: {method}")
            
            for property_name in PYTHON_CLIENT_REQUIREMENTS["properties"]:
                if property_name not in content:
                    result["violations"].append(f"Missing property: {property_name}")
            
            if not result["has_error_handling"]:
                result["violations"].append("Missing error handling")
            
            if not result["has_logging"]:
                result["violations"].append("Missing logging")
            
            if not result["has_resilience"]:
                result["violations"].append("Missing resilience patterns")
    
    except Exception as e:
        result["violations"].append(f"Error analyzing file: {e}")
    
    return result


def analyze_js_client(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a JavaScript/TypeScript service client file.
    
    Args:
        file_path: Path to the JavaScript/TypeScript client file
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        "file": str(file_path),
        "language": "javascript" if file_path.suffix == ".js" else "typescript",
        "client_name": None,
        "base_classes": [],
        "methods": [],
        "properties": [],
        "has_error_handling": False,
        "has_logging": False,
        "has_resilience": False,
        "violations": []
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Extract client name
            client_match = re.search(JS_CLIENT_PATTERN, content)
            if client_match:
                result["client_name"] = client_match.group(1)
            
            # Extract base classes
            base_match = re.search(r'extends\s+(\w+)', content)
            if base_match:
                result["base_classes"].append(base_match.group(1))
            
            # Extract methods
            method_matches = re.finditer(r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*{', content)
            for match in method_matches:
                method_name = match.group(1)
                if method_name not in ["constructor", "super"]:
                    result["methods"].append(method_name)
            
            # Check for error handling
            result["has_error_handling"] = "try" in content and "catch" in content
            
            # Check for logging
            result["has_logging"] = "logger" in content and ("this.logger" in content or "console.log" in content)
            
            # Check for resilience patterns
            result["has_resilience"] = any(pattern in content for pattern in JS_CLIENT_REQUIREMENTS["resilience"])
            
            # Check for violations
            for base_class in JS_CLIENT_REQUIREMENTS["base_classes"]:
                if base_class not in result["base_classes"]:
                    result["violations"].append(f"Missing base class: {base_class}")
            
            for method in JS_CLIENT_REQUIREMENTS["methods"]:
                if method not in result["methods"]:
                    result["violations"].append(f"Missing method: {method}")
            
            for property_name in JS_CLIENT_REQUIREMENTS["properties"]:
                if property_name not in content:
                    result["violations"].append(f"Missing property: {property_name}")
            
            if not result["has_error_handling"]:
                result["violations"].append("Missing error handling")
            
            if not result["has_logging"]:
                result["violations"].append("Missing logging")
            
            if not result["has_resilience"]:
                result["violations"].append("Missing resilience patterns")
    
    except Exception as e:
        result["violations"].append(f"Error analyzing file: {e}")
    
    return result


def standardize_python_client(file_path: Path, analysis: Dict[str, Any]) -> bool:
    """
    Standardize a Python service client.
    
    Args:
        file_path: Path to the Python client file
        analysis: Analysis results for the client
        
    Returns:
        True if the client was standardized, False otherwise
    """
    if not analysis["violations"]:
        print(f"Client {analysis['client_name']} already follows standards")
        return False
    
    # Extract service name and resource name from client name
    client_name = analysis["client_name"]
    service_name = client_name.replace("Client", "").replace("Service", "")
    
    # Convert camel case to kebab case for service name
    service_name_kebab = re.sub(r'(?<!^)(?=[A-Z])', '-', service_name).lower()
    
    # Extract resource name (assuming it's the first part of the service name)
    resource_name = service_name_kebab.split('-')[0]
    
    # Create standardized client
    standardized_client = PYTHON_CLIENT_TEMPLATE.format(
        client_name=client_name,
        service_name=service_name,
        resource_name=resource_name,
        resource_id_param=f"{resource_name}_id"
    )
    
    # Create backup of original file
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
    file_path.rename(backup_path)
    
    # Write standardized client
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(standardized_client)
    
    print(f"Standardized client {client_name} (backup at {backup_path})")
    return True


def standardize_js_client(file_path: Path, analysis: Dict[str, Any]) -> bool:
    """
    Standardize a JavaScript/TypeScript service client.
    
    Args:
        file_path: Path to the JavaScript/TypeScript client file
        analysis: Analysis results for the client
        
    Returns:
        True if the client was standardized, False otherwise
    """
    if not analysis["violations"]:
        print(f"Client {analysis['client_name']} already follows standards")
        return False
    
    # Extract service name and resource name from client name
    client_name = analysis["client_name"]
    service_name = client_name.replace("Client", "").replace("Service", "")
    
    # Convert camel case to kebab case for service name
    service_name_kebab = re.sub(r'(?<!^)(?=[A-Z])', '-', service_name).lower()
    
    # Extract resource name (assuming it's the first part of the service name)
    resource_name = service_name_kebab.split('-')[0]
    
    # Convert resource name to pascal case for method names
    resource_name_pascal = "".join(word.capitalize() for word in resource_name.split('-'))
    
    # Create standardized client
    standardized_client = JS_CLIENT_TEMPLATE.format(
        client_name=client_name,
        service_name=service_name,
        resource_name=resource_name,
        resource_name_pascal=resource_name_pascal,
        resource_id_param=f"{resource_name}Id"
    )
    
    # Create backup of original file
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
    file_path.rename(backup_path)
    
    # Write standardized client
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(standardized_client)
    
    print(f"Standardized client {client_name} (backup at {backup_path})")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standardize service client implementations")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without making changes")
    parser.add_argument("--client", help="Path to a specific client to standardize")
    args = parser.parse_args()
    
    repo_root = get_repo_root()
    
    if args.client:
        client_path = Path(args.client)
        if not client_path.is_absolute():
            client_path = repo_root / client_path
        
        if not client_path.exists():
            print(f"Error: Client file {client_path} does not exist")
            return 1
        
        if client_path.suffix == ".py":
            analysis = analyze_python_client(client_path)
            if not args.dry_run:
                standardize_python_client(client_path, analysis)
        elif client_path.suffix in [".js", ".ts"]:
            analysis = analyze_js_client(client_path)
            if not args.dry_run:
                standardize_js_client(client_path, analysis)
        else:
            print(f"Error: Unsupported file type {client_path.suffix}")
            return 1
    else:
        # Find all client files
        python_client_files, js_client_files = find_client_files(repo_root)
        
        print(f"Found {len(python_client_files)} Python clients and {len(js_client_files)} JavaScript/TypeScript clients")
        
        # Analyze and standardize Python clients
        for file_path in python_client_files:
            analysis = analyze_python_client(file_path)
            if analysis["violations"]:
                print(f"Python client {analysis['client_name']} has violations: {analysis['violations']}")
                if not args.dry_run:
                    standardize_python_client(file_path, analysis)
        
        # Analyze and standardize JavaScript/TypeScript clients
        for file_path in js_client_files:
            analysis = analyze_js_client(file_path)
            if analysis["violations"]:
                print(f"JavaScript/TypeScript client {analysis['client_name']} has violations: {analysis['violations']}")
                if not args.dry_run:
                    standardize_js_client(file_path, analysis)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
