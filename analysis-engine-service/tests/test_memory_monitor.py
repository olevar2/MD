"""
Tests for the memory monitoring module.

This module contains tests for the MemoryMonitor class and its functionality.
"""

import pytest
import psutil
import asyncio
from unittest.mock import MagicMock, patch
from analysis_engine.core.monitoring.memory_monitor import MemoryMonitor, get_memory_monitor
# The 'memory_monitor' fixture is now imported from conftest.py

@pytest.fixture
def mock_psutil():
    """Create a mock psutil for testing."""
    with patch("psutil.Process") as mock_process:
        process = MagicMock()
        process.memory_info.return_value = MagicMock(
            rss=1024 * 1024 * 100,  # 100MB
            vms=1024 * 1024 * 200   # 200MB
        )
        mock_process.return_value = process
        yield mock_process

@pytest.fixture
def mock_prometheus():
    """Create mock Prometheus metrics for testing."""
    with patch("analysis_engine.core.monitoring.memory_monitor.MEMORY_USAGE") as mock_usage, \
         patch("analysis_engine.core.monitoring.memory_monitor.MEMORY_LIMIT") as mock_limit, \
         patch("analysis_engine.core.monitoring.memory_monitor.MEMORY_WARNINGS") as mock_warnings:
        yield {
            "usage": mock_usage,
            "limit": mock_limit,
            "warnings": mock_warnings
        }

def test_memory_monitor_initialization(memory_monitor):
    """Test memory monitor initialization."""
    assert memory_monitor.warning_threshold == 80
    assert memory_monitor.critical_threshold == 90
    assert memory_monitor._monitoring_task is None
    assert memory_monitor._stop_event is not None

def test_get_memory_monitor_singleton():
    """Test that get_memory_monitor returns a singleton instance."""
    monitor1 = get_memory_monitor()
    monitor2 = get_memory_monitor()
    assert monitor1 is monitor2

@pytest.mark.asyncio
async def test_start_monitoring(memory_monitor, mock_psutil, mock_prometheus):
    """Test starting memory monitoring."""
    await memory_monitor.start_monitoring()
    assert memory_monitor._monitoring_task is not None
    assert not memory_monitor._monitoring_task.done()

@pytest.mark.asyncio
async def test_stop_monitoring(memory_monitor):
    """Test stopping memory monitoring."""
    await memory_monitor.start_monitoring()
    await memory_monitor.stop_monitoring()
    assert memory_monitor._monitoring_task.done()

@pytest.mark.asyncio
async def test_check_memory_usage(memory_monitor, mock_psutil, mock_prometheus):
    """Test memory usage checking."""
    # Test normal usage
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value = MagicMock(
            percent=50,
            total=1024 * 1024 * 1000  # 1GB
        )
        await memory_monitor.check_memory_usage()
        mock_prometheus["usage"].set.assert_called_once()
        mock_prometheus["warnings"].inc.assert_not_called()

    # Test warning threshold
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value = MagicMock(
            percent=85,
            total=1024 * 1024 * 1000
        )
        await memory_monitor.check_memory_usage()
        mock_prometheus["warnings"].inc.assert_called_once()

    # Test critical threshold
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value = MagicMock(
            percent=95,
            total=1024 * 1024 * 1000
        )
        await memory_monitor.check_memory_usage()
        assert mock_prometheus["warnings"].inc.call_count == 2

def test_get_memory_stats(memory_monitor, mock_psutil):
    """Test getting memory statistics."""
    stats = memory_monitor.get_memory_stats()
    assert "rss" in stats
    assert "vms" in stats
    assert "percent" in stats
    assert isinstance(stats["rss"], int)
    assert isinstance(stats["vms"], int)
    assert isinstance(stats["percent"], float)

@pytest.mark.asyncio
async def test_monitoring_loop(memory_monitor, mock_psutil, mock_prometheus):
    """Test the monitoring loop functionality."""
    await memory_monitor.start_monitoring()
    await asyncio.sleep(0.1)  # Allow the loop to run once
    await memory_monitor.stop_monitoring()
    
    assert mock_prometheus["usage"].set.called
    assert mock_prometheus["limit"].set.called

@pytest.mark.asyncio
async def test_error_handling(memory_monitor, mock_psutil):
    """Test error handling in memory monitoring."""
    with patch("psutil.virtual_memory", side_effect=Exception("Test error")):
        await memory_monitor.check_memory_usage()
        # Should not raise exception, but log error

@pytest.mark.asyncio
async def test_concurrent_monitoring(memory_monitor):
    """Test concurrent monitoring attempts."""
    await memory_monitor.start_monitoring()
    await memory_monitor.start_monitoring()  # Should not start a new task
    assert memory_monitor._monitoring_task is not None
    await memory_monitor.stop_monitoring()

def test_memory_limit_setting(memory_monitor, mock_prometheus):
    """Test memory limit setting."""
    memory_monitor.set_memory_limit(1024 * 1024 * 500)  # 500MB
    mock_prometheus["limit"].set.assert_called_once_with(1024 * 1024 * 500)

def test_threshold_validation(memory_monitor):
    """Test threshold validation."""
    with pytest.raises(ValueError):
        memory_monitor.warning_threshold = 110
    
    with pytest.raises(ValueError):
        memory_monitor.critical_threshold = 50
    
    with pytest.raises(ValueError):
        memory_monitor.critical_threshold = 70  # Less than warning threshold