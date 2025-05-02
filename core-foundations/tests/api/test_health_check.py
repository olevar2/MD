"""
Unit tests for health check API functionality.
"""

import unittest
from unittest import mock
from datetime import datetime, timezone
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import psutil

from core_foundations.api.health_check import (
    HealthCheck,
    create_health_router,
    add_health_check_to_app,
)
from core_foundations.models.schemas import HealthStatus, DependencyStatus


class TestHealthCheck(unittest.TestCase):
    """Tests for HealthCheck class and related functions."""
    
    def test_health_check_initialization(self):
        """Test HealthCheck initialization."""
        health_check = HealthCheck("test-service", "1.0.0")
        
        self.assertEqual(health_check.service_name, "test-service")
        self.assertEqual(health_check.version, "1.0.0")
        # Should have 2 default checks now: memory_usage and disk_usage
        self.assertEqual(len(health_check.checks), 2)
        self.assertEqual(len(health_check.dependencies), 0)
    
    def test_add_check(self):
        """Test adding a health check."""
        health_check = HealthCheck("test-service", "1.0.0")
        
        # Count existing checks before adding new one
        initial_check_count = len(health_check.checks)
        
        # Add a simple check
        health_check.add_check("db-connection", lambda: True, critical=True)
        
        self.assertEqual(len(health_check.checks), initial_check_count + 1)
        new_check = next(check for check in health_check.checks if check["name"] == "db-connection")
        self.assertTrue(new_check["check_func"]())
        self.assertTrue(new_check["critical"])
    
    def test_add_dependency(self):
        """Test adding a dependency check."""
        health_check = HealthCheck("test-service", "1.0.0")
        
        # Add a dependency check
        response_time = 15.5
        details = {"version": "2.0.0"}
        health_check.add_dependency(
            "auth-service", lambda: (HealthStatus.HEALTHY, response_time, details)
        )
        
        self.assertEqual(len(health_check.dependencies), 1)
        self.assertIn("auth-service", health_check.dependencies)
        
        # Test the dependency check function
        status, rt, deps = health_check.dependencies["auth-service"]()
        self.assertEqual(status, HealthStatus.HEALTHY)
        self.assertEqual(rt, response_time)
        self.assertEqual(deps, details)
    
    @mock.patch("psutil.virtual_memory")
    def test_check_memory_usage(self, mock_virtual_memory):
        """Test memory usage check."""
        # Configure mock to return healthy memory usage
        mock_memory = mock.MagicMock()
        mock_memory.percent = 80.0  # Below 90% threshold
        mock_virtual_memory.return_value = mock_memory
        
        health_check = HealthCheck("test-service", "1.0.0")
        self.assertTrue(health_check._check_memory_usage())
        
        # Test unhealthy memory usage
        mock_memory.percent = 95.0  # Above 90% threshold
        mock_virtual_memory.return_value = mock_memory
        
        self.assertFalse(health_check._check_memory_usage())
    
    @mock.patch("psutil.disk_usage")
    def test_check_disk_usage(self, mock_disk_usage):
        """Test disk usage check."""
        # Configure mock to return healthy disk usage
        mock_disk = mock.MagicMock()
        mock_disk.percent = 75.0  # Below 85% threshold
        mock_disk_usage.return_value = mock_disk
        
        health_check = HealthCheck("test-service", "1.0.0")
        self.assertTrue(health_check._check_disk_usage())
        
        # Test unhealthy disk usage
        mock_disk.percent = 90.0  # Above 85% threshold
        mock_disk_usage.return_value = mock_disk
        
        self.assertFalse(health_check._check_disk_usage())
    
    @mock.patch("psutil.cpu_percent")
    @mock.patch("psutil.virtual_memory")
    @mock.patch("psutil.disk_usage")
    @mock.patch("psutil.boot_time")
    def test_get_resource_metrics(self, mock_boot_time, mock_disk_usage, mock_virtual_memory, mock_cpu_percent):
        """Test resource metrics collection."""
        # Configure mocks
        mock_cpu_percent.return_value = 25.5
        
        mock_memory = mock.MagicMock()
        mock_memory.percent = 65.0
        mock_memory.available = 4000000000  # 4GB
        mock_virtual_memory.return_value = mock_memory
        
        mock_disk = mock.MagicMock()
        mock_disk.percent = 70.0
        mock_disk.free = 100000000000  # 100GB
        mock_disk_usage.return_value = mock_disk
        
        current_time = time.time()
        mock_boot_time.return_value = current_time - 3600  # 1 hour ago
        
        health_check = HealthCheck("test-service", "1.0.0")
        metrics = health_check.get_resource_metrics()
        
        self.assertEqual(metrics.cpu_usage, 25.5)
        self.assertEqual(metrics.memory_usage, 65.0)
        self.assertEqual(metrics.memory_available, 4000000000)
        self.assertEqual(metrics.disk_usage, 70.0)
        self.assertEqual(metrics.disk_available, 100000000000)
        self.assertEqual(metrics.uptime, 3600)
    
    @mock.patch("time.time")
    def test_check_health(self, mock_time):
        """Test health check functionality."""
        # Mock time for consistent uptime values
        start_time = 1000000
        current_time = start_time + 300  # 5 minutes later
        
        mock_time.side_effect = [start_time, current_time, current_time]
        
        health_check = HealthCheck("test-service", "1.0.0")
        
        # Override automatic resource checks for testing
        health_check.checks = []
        
        # Add a passing check
        health_check.add_check("passing-check", lambda: True, critical=True)
        
        # Add a failing non-critical check
        health_check.add_check("failing-check", lambda: False, critical=False)
        
        # Add a healthy dependency
        health_check.add_dependency(
            "healthy-dependency", lambda: (HealthStatus.HEALTHY, 10.5, {"version": "1.0.0"})
        )
        
        # Add an unhealthy dependency
        health_check.add_dependency(
            "unhealthy-dependency", lambda: (HealthStatus.UNHEALTHY, 500.0, {"error": "Connection refused"})
        )
        
        # Mock resource metrics
        health_check.get_resource_metrics = mock.MagicMock()
        
        # Run health check
        result = health_check.check_health()
        
        # Service should be DEGRADED because of the failing check and unhealthy dependency
        self.assertEqual(result.status, HealthStatus.DEGRADED)
        self.assertEqual(result.service, "test-service")
        self.assertEqual(result.version, "1.0.0")
        self.assertEqual(result.uptime, 300)  # 5 minutes
        
        # Check passing and failing checks
        self.assertTrue(result.checks["passing-check"])
        self.assertFalse(result.checks["failing-check"])
        
        # Check dependencies
        self.assertEqual(result.dependencies["healthy-dependency"]["status"], HealthStatus.HEALTHY)
        self.assertEqual(result.dependencies["unhealthy-dependency"]["status"], HealthStatus.UNHEALTHY)
    
    def test_create_health_router(self):
        """Test creating FastAPI router with health check endpoints."""
        health_check = HealthCheck("test-service", "1.0.0")
        router = create_health_router(health_check)
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Test /health endpoint
        health_check.check_health = mock.MagicMock()
        health_check.check_health.return_value = {
            "status": HealthStatus.HEALTHY,
            "service": "test-service",
            "version": "1.0.0"
        }
        
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        # Test /health/live endpoint
        response = client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "alive")
        
        # Test /health/ready endpoint when healthy
        health_check.check_health.return_value.status = HealthStatus.HEALTHY
        response = client.get("/health/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
    
    def test_add_health_check_to_app(self):
        """Test adding health check to FastAPI app."""
        app = FastAPI()
        
        # Add health check to app
        health_check = add_health_check_to_app(
            app,
            "test-service",
            "1.0.0",
            checks=[{"name": "test-check", "check_func": lambda: True}],
            dependencies={"test-dep": lambda: (HealthStatus.HEALTHY, 10.0, None)}
        )
        
        # Check that health check was created correctly
        self.assertEqual(health_check.service_name, "test-service")
        self.assertEqual(health_check.version, "1.0.0")
        
        # Verify checks and dependencies were added
        self.assertTrue(any(c["name"] == "test-check" for c in health_check.checks))
        self.assertIn("test-dep", health_check.dependencies)


if __name__ == "__main__":
    unittest.main()