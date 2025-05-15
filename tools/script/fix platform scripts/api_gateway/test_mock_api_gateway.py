#!/usr/bin/env python3
"""
Test script for the mock API Gateway.

This script tests the mock API Gateway by sending requests to various endpoints
and verifying the responses.
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from typing import Dict, Any, Optional, List

import httpx


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_mock_api_gateway")


class MockAPIGatewayTester:
    """
    Tester for the mock API Gateway.
    """

    def __init__(self, base_url: str):
        """
        Initialize the tester.

        Args:
            base_url: Base URL of the API Gateway
        """
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.correlation_id = str(uuid.uuid4())
        self.request_id = str(uuid.uuid4())
        self.jwt_token = None
        self.api_key = "test_api_key"

    def _get_headers(self, use_jwt: bool = True, use_api_key: bool = False) -> Dict[str, str]:
        """
        Get headers for requests.

        Args:
            use_jwt: Whether to include JWT token
            use_api_key: Whether to include API key

        Returns:
            Headers
        """
        headers = {
            "Content-Type": "application/json",
            "X-Correlation-ID": self.correlation_id,
            "X-Request-ID": self.request_id
        }

        if use_jwt and self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"

        if use_api_key:
            headers["X-API-Key"] = self.api_key

        return headers

    def get_token(self) -> bool:
        """
        Get a JWT token.

        Returns:
            True if the token was obtained successfully, False otherwise
        """
        logger.info("Getting JWT token...")

        try:
            response = self.client.get(f"{self.base_url}/api/v1/auth/token")
            response.raise_for_status()

            # Get token
            data = response.json()
            self.jwt_token = data.get("data", {}).get("token")

            if self.jwt_token:
                logger.info("JWT token obtained successfully")
                return True
            else:
                logger.error("JWT token not found in response")
                return False
        except Exception as e:
            logger.error(f"Error getting JWT token: {str(e)}")
            return False

    def test_health_check(self) -> bool:
        """
        Test the health check endpoint.

        Returns:
            True if the test passed, False otherwise
        """
        logger.info("Testing health check endpoint...")

        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()

            # Check response
            if response.status_code == 200 and response.json().get("status") == "ok":
                logger.info("Health check test passed")
                return True
            else:
                logger.error(f"Health check test failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error testing health check: {str(e)}")
            return False

    def test_authentication(self) -> bool:
        """
        Test authentication.

        Returns:
            True if the test passed, False otherwise
        """
        logger.info("Testing authentication...")

        # Test JWT authentication
        try:
            response = self.client.get(
                f"{self.base_url}/api/v1/trading-gateway-service/test",
                headers=self._get_headers(use_jwt=True, use_api_key=False)
            )
            response.raise_for_status()
            logger.info("JWT authentication test passed")
        except Exception as e:
            logger.error(f"Error testing JWT authentication: {str(e)}")
            return False

        # Test API key authentication
        try:
            response = self.client.get(
                f"{self.base_url}/api/v1/internal/status",
                headers=self._get_headers(use_jwt=False, use_api_key=True)
            )
            if response.status_code == 200:
                logger.info("API key authentication test passed")
            else:
                logger.error(f"API key authentication test failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error testing API key authentication: {str(e)}")
            return False

        # Test authentication failure
        try:
            response = self.client.get(
                f"{self.base_url}/api/v1/trading-gateway-service/test",
                headers=self._get_headers(use_jwt=False, use_api_key=False)
            )
            if response.status_code == 401:
                logger.info("Authentication failure test passed")
            else:
                logger.error(f"Authentication failure test failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error testing authentication failure: {str(e)}")
            return False

        return True

    def test_proxy(self) -> bool:
        """
        Test proxy functionality.

        Returns:
            True if the test passed, False otherwise
        """
        logger.info("Testing proxy functionality...")

        # Test proxying to a backend service
        try:
            response = self.client.get(
                f"{self.base_url}/api/v1/trading-gateway-service/test",
                headers=self._get_headers(use_jwt=True, use_api_key=False)
            )
            response.raise_for_status()

            # Check response
            data = response.json()
            if (
                data.get("status") == "success"
                and data.get("data", {}).get("service") == "trading-gateway-service"
                and data.get("meta", {}).get("correlation_id") == self.correlation_id
                and data.get("meta", {}).get("request_id") == self.request_id
            ):
                logger.info("Proxy test passed")
                return True
            else:
                logger.error(f"Proxy test failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error testing proxy: {str(e)}")
            return False

    def test_error_handling(self) -> bool:
        """
        Test error handling.

        Returns:
            True if the test passed, False otherwise
        """
        logger.info("Testing error handling...")

        # Test 404 error
        try:
            response = self.client.get(
                f"{self.base_url}/api/v1/non-existent-service/test",
                headers=self._get_headers(use_jwt=True, use_api_key=False)
            )
            if response.status_code == 404:
                # Check error response format
                error_response = response.json()
                if (
                    error_response.get("status") == "error"
                    and error_response.get("error", {}).get("code") == "SERVICE_NOT_FOUND"
                    and error_response.get("meta", {}).get("correlation_id") == self.correlation_id
                    and error_response.get("meta", {}).get("request_id") == self.request_id
                ):
                    logger.info("Error handling test passed")
                    return True
                else:
                    logger.error(f"Error handling test failed: Invalid error response format: {error_response}")
                    return False
            else:
                logger.error(f"Error handling test failed: Expected 404, got {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing error handling: {str(e)}")
            return False

    def run_all_tests(self) -> bool:
        """
        Run all tests.

        Returns:
            True if all tests passed, False otherwise
        """
        # Get token
        if not self.get_token():
            return False

        tests = [
            self.test_health_check,
            self.test_authentication,
            self.test_proxy,
            self.test_error_handling
        ]

        results = []
        for test in tests:
            results.append(test())

        # Print summary
        logger.info("Test summary:")
        for i, result in enumerate(results):
            logger.info(f"  Test {i + 1}: {'PASSED' if result else 'FAILED'}")

        # Return overall result
        return all(results)


def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test the mock API Gateway")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the API Gateway")
    args = parser.parse_args()

    # Create tester
    tester = MockAPIGatewayTester(args.base_url)

    # Run tests
    success = tester.run_all_tests()

    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()