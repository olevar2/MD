#!/usr/bin/env python3
"""
Verification script for the API Gateway Enhancement implementation.

This script verifies that the API Gateway Enhancement implementation is correct
by checking the file structure, content, and code quality.
"""

import os
import sys
import logging
import re
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("verify_implementation")

class APIGatewayVerifier:
    """
    Verifier for the API Gateway Enhancement implementation.
    """

    def __init__(self, api_gateway_dir: str):
        """
        Initialize the verifier.

        Args:
            api_gateway_dir: Directory containing the API Gateway implementation
        """
        self.api_gateway_dir = api_gateway_dir
        self.required_files = [
            # Core files
            "core/response/standard_response.py",
            "core/response/__init__.py",
            "core/auth/auth_middleware.py",
            "core/auth/__init__.py",
            "core/rate_limit/enhanced_rate_limit.py",
            "core/rate_limit/__init__.py",
            
            # Service files
            "services/proxy/proxy_service.py",
            "services/proxy/__init__.py",
            "services/registry/service_registry.py",
            "services/registry/__init__.py",
            
            # API files
            "api/routes/proxy.py",
            "api/routes/__init__.py",
            "api/app_enhanced.py",
            
            # Config files
            "config/api-gateway-enhanced.yaml",
            ".env.example",
            
            # Documentation files
            "docs/README.md",
            "docs/ARCHITECTURE.md",
            "docs/API_REFERENCE.md"
        ]
        
        self.required_content = {
            "core/response/standard_response.py": [
                "class StandardResponse",
                "create_success_response",
                "create_error_response",
                "create_warning_response"
            ],
            "core/auth/auth_middleware.py": [
                "class EnhancedAuthMiddleware",
                "_validate_jwt",
                "_validate_api_key",
                "_check_permissions"
            ],
            "core/rate_limit/enhanced_rate_limit.py": [
                "class TokenBucket",
                "class EnhancedRateLimitMiddleware",
                "_get_rate_and_capacity"
            ],
            "services/proxy/proxy_service.py": [
                "class ProxyService",
                "proxy_request",
                "_get_circuit_breaker"
            ],
            "services/registry/service_registry.py": [
                "class ServiceRegistry",
                "_check_service_health",
                "get_service_for_endpoint"
            ],
            "api/routes/proxy.py": [
                "proxy_request",
                "service_registry",
                "proxy_service"
            ],
            "api/app_enhanced.py": [
                "app = FastAPI",
                "EnhancedAuthMiddleware",
                "EnhancedRateLimitMiddleware",
                "add_correlation_id",
                "global_exception_handler"
            ],
            "config/api-gateway-enhanced.yaml": [
                "auth:",
                "rate_limit:",
                "services:",
                "role_permissions:"
            ]
        }
        
        self.code_quality_checks = {
            "core/response/standard_response.py": [
                (r"class\s+StandardResponse\s*\(\s*BaseModel\s*\)", "StandardResponse should inherit from BaseModel"),
                (r"def\s+create_success_response", "create_success_response function should be defined"),
                (r"def\s+create_error_response", "create_error_response function should be defined")
            ],
            "core/auth/auth_middleware.py": [
                (r"class\s+EnhancedAuthMiddleware\s*\(\s*BaseHTTPMiddleware\s*\)", "EnhancedAuthMiddleware should inherit from BaseHTTPMiddleware"),
                (r"async\s+def\s+dispatch", "dispatch method should be async"),
                (r"def\s+_validate_jwt", "_validate_jwt method should be defined")
            ],
            "core/rate_limit/enhanced_rate_limit.py": [
                (r"class\s+TokenBucket", "TokenBucket class should be defined"),
                (r"async\s+def\s+consume", "consume method should be async"),
                (r"class\s+EnhancedRateLimitMiddleware\s*\(\s*BaseHTTPMiddleware\s*\)", "EnhancedRateLimitMiddleware should inherit from BaseHTTPMiddleware")
            ],
            "services/proxy/proxy_service.py": [
                (r"class\s+ProxyService", "ProxyService class should be defined"),
                (r"async\s+def\s+proxy_request", "proxy_request method should be async"),
                (r"def\s+_get_circuit_breaker", "_get_circuit_breaker method should be defined")
            ],
            "services/registry/service_registry.py": [
                (r"class\s+ServiceRegistry", "ServiceRegistry class should be defined"),
                (r"async\s+def\s+_check_service_health", "_check_service_health method should be async"),
                (r"def\s+get_service_for_endpoint", "get_service_for_endpoint method should be defined")
            ],
            "api/routes/proxy.py": [
                (r"router\s*=\s*APIRouter\(\)", "router should be initialized"),
                (r"@router\.api_route", "api_route decorator should be used"),
                (r"async\s+def\s+proxy_request", "proxy_request function should be async")
            ],
            "api/app_enhanced.py": [
                (r"app\s*=\s*FastAPI\(", "app should be initialized"),
                (r"app\.add_middleware\(\s*EnhancedAuthMiddleware", "EnhancedAuthMiddleware should be added"),
                (r"app\.add_middleware\(\s*EnhancedRateLimitMiddleware", "EnhancedRateLimitMiddleware should be added")
            ]
        }

    def verify_file_structure(self) -> Tuple[bool, List[str]]:
        """
        Verify that all required files exist.

        Returns:
            Tuple of (success, missing_files)
        """
        logger.info("Verifying file structure...")
        
        missing_files = []
        for file_path in self.required_files:
            full_path = os.path.join(self.api_gateway_dir, file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
                logger.error(f"Missing file: {file_path}")
        
        success = len(missing_files) == 0
        if success:
            logger.info("File structure verification passed")
        else:
            logger.error(f"File structure verification failed: {len(missing_files)} files missing")
        
        return success, missing_files

    def verify_file_content(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Verify that files contain required content.

        Returns:
            Tuple of (success, missing_content)
        """
        logger.info("Verifying file content...")
        
        missing_content = {}
        for file_path, required_content in self.required_content.items():
            full_path = os.path.join(self.api_gateway_dir, file_path)
            if not os.path.exists(full_path):
                # Skip files that don't exist (already reported in verify_file_structure)
                continue
            
            # Read file content
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Check for required content
            missing = []
            for required in required_content:
                if required not in content:
                    missing.append(required)
            
            if missing:
                missing_content[file_path] = missing
                logger.error(f"Missing content in {file_path}: {missing}")
        
        success = len(missing_content) == 0
        if success:
            logger.info("File content verification passed")
        else:
            logger.error(f"File content verification failed: {len(missing_content)} files with missing content")
        
        return success, missing_content

    def verify_code_quality(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Verify code quality.

        Returns:
            Tuple of (success, quality_issues)
        """
        logger.info("Verifying code quality...")
        
        quality_issues = {}
        for file_path, checks in self.code_quality_checks.items():
            full_path = os.path.join(self.api_gateway_dir, file_path)
            if not os.path.exists(full_path):
                # Skip files that don't exist (already reported in verify_file_structure)
                continue
            
            # Read file content
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Check code quality
            issues = []
            for pattern, message in checks:
                if not re.search(pattern, content):
                    issues.append(message)
            
            if issues:
                quality_issues[file_path] = issues
                logger.error(f"Code quality issues in {file_path}: {issues}")
        
        success = len(quality_issues) == 0
        if success:
            logger.info("Code quality verification passed")
        else:
            logger.error(f"Code quality verification failed: {len(quality_issues)} files with quality issues")
        
        return success, quality_issues

    def verify_implementation(self) -> bool:
        """
        Verify the API Gateway implementation.

        Returns:
            True if the implementation is correct, False otherwise
        """
        # Verify file structure
        structure_success, missing_files = self.verify_file_structure()
        
        # Verify file content
        content_success, missing_content = self.verify_file_content()
        
        # Verify code quality
        quality_success, quality_issues = self.verify_code_quality()
        
        # Overall success
        success = structure_success and content_success and quality_success
        
        if success:
            logger.info("API Gateway implementation verification passed")
        else:
            logger.error("API Gateway implementation verification failed")
        
        return success

def main():
    """
    Main function.
    """
    # Get API Gateway directory
    api_gateway_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        "../../../../api-gateway"
    ))
    
    # Create verifier
    verifier = APIGatewayVerifier(api_gateway_dir)
    
    # Verify implementation
    success = verifier.verify_implementation()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()