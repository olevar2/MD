"""
Unit tests for health check module.
"""

import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from health_check import health_check, get_system_info, get_service_status


class TestHealthCheck(unittest.TestCase):
    """Test cases for health check module."""
    
    def test_get_system_info(self):
        """Test get_system_info function."""
        system_info = get_system_info()
        
        # Check that the function returns a dictionary
        self.assertIsInstance(system_info, dict)
        
        # Check that the dictionary contains the expected keys
        self.assertIn('hostname', system_info)
        self.assertIn('platform', system_info)
        self.assertIn('python_version', system_info)
        self.assertIn('time', system_info)
    
    def test_get_service_status(self):
        """Test get_service_status function."""
        service_status = get_service_status()
        
        # Check that the function returns a dictionary
        self.assertIsInstance(service_status, dict)
        
        # Check that the dictionary contains the expected keys
        self.assertIn('status', service_status)
        self.assertIn('service', service_status)
        self.assertIn('version', service_status)
        self.assertIn('uptime', service_status)
        
        # Check that the status is UP
        self.assertEqual(service_status['status'], 'UP')
        
        # Check that the service name is correct
        self.assertEqual(service_status['service'], 'trading-gateway-service')
    
    def test_health_check(self):
        """Test health_check function."""
        # Mock the get_system_info and get_service_status functions
        with patch('health_check.get_system_info') as mock_get_system_info, \
             patch('health_check.get_service_status') as mock_get_service_status:
            
            # Set up the mock return values
            mock_get_system_info.return_value = {
                'hostname': 'test-host',
                'platform': 'test-platform',
                'python_version': '3.8.0',
                'time': '2025-05-13T00:00:00'
            }
            
            mock_get_service_status.return_value = {
                'status': 'UP',
                'service': 'trading-gateway-service',
                'version': '1.0.0',
                'uptime': 123456789.0
            }
            
            # Call the function
            result = health_check()
            
            # Check that the function returns a dictionary
            self.assertIsInstance(result, dict)
            
            # Check that the dictionary contains the expected keys
            self.assertIn('system', result)
            self.assertIn('service', result)
            self.assertIn('status', result)
            
            # Check that the status is healthy
            self.assertEqual(result['status'], 'healthy')
            
            # Check that the system and service dictionaries are correct
            self.assertEqual(result['system'], mock_get_system_info.return_value)
            self.assertEqual(result['service'], mock_get_service_status.return_value)


if __name__ == '__main__':
    unittest.main()
