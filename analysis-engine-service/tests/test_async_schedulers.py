"""
Tests for the async schedulers.

This module contains tests for the ToolEffectivenessScheduler and ReportScheduler classes.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from analysis_engine.scheduling.effectiveness_scheduler import ToolEffectivenessScheduler
from analysis_engine.scheduling.report_scheduler import ReportScheduler
from analysis_engine.scheduling.scheduler_factory import initialize_schedulers, cleanup_schedulers
from analysis_engine.core.container import ServiceContainer

@pytest.fixture
def effectiveness_scheduler():
    """Create a ToolEffectivenessScheduler instance for testing."""
    return ToolEffectivenessScheduler()

@pytest.fixture
def report_scheduler():
    """Create a ReportScheduler instance for testing."""
    db_factory = MagicMock()
    return ReportScheduler(db_factory)

@pytest.fixture
def service_container():
    """Create a ServiceContainer instance for testing."""
    return ServiceContainer()

@pytest.mark.asyncio
async def test_effectiveness_scheduler_start_stop(effectiveness_scheduler):
    """Test starting and stopping the effectiveness scheduler."""
    # Start the scheduler
    await effectiveness_scheduler.start()
    assert effectiveness_scheduler.running
    assert effectiveness_scheduler.scheduler_task is not None
    
    # Stop the scheduler
    await effectiveness_scheduler.stop()
    assert not effectiveness_scheduler.running
    assert effectiveness_scheduler.scheduler_task is None or effectiveness_scheduler.scheduler_task.done()

@pytest.mark.asyncio
async def test_report_scheduler_start_stop(report_scheduler):
    """Test starting and stopping the report scheduler."""
    # Start the scheduler
    await report_scheduler.start()
    assert report_scheduler._running
    assert report_scheduler._scheduler_task is not None
    
    # Stop the scheduler
    await report_scheduler.stop()
    assert not report_scheduler._running
    assert report_scheduler._scheduler_task is None or report_scheduler._scheduler_task.done()

@pytest.mark.asyncio
async def test_scheduler_factory_initialize(service_container):
    """Test initializing schedulers with the factory."""
    # Initialize schedulers
    await initialize_schedulers(service_container)
    
    # Check that schedulers were registered
    assert "effectiveness_scheduler" in service_container._services
    assert "report_scheduler" in service_container._services
    
    # Check that schedulers were started
    effectiveness_scheduler = service_container.get_service("effectiveness_scheduler")
    report_scheduler = service_container.get_service("report_scheduler")
    
    assert effectiveness_scheduler.running
    assert report_scheduler._running
    
    # Clean up
    await cleanup_schedulers(service_container)

@pytest.mark.asyncio
async def test_scheduler_factory_cleanup(service_container):
    """Test cleaning up schedulers with the factory."""
    # Initialize schedulers
    await initialize_schedulers(service_container)
    
    # Clean up schedulers
    await cleanup_schedulers(service_container)
    
    # Check that schedulers were stopped
    effectiveness_scheduler = service_container.get_service("effectiveness_scheduler")
    report_scheduler = service_container.get_service("report_scheduler")
    
    assert not effectiveness_scheduler.running
    assert not report_scheduler._running

@pytest.mark.asyncio
async def test_effectiveness_scheduler_calculate_metrics(effectiveness_scheduler):
    """Test calculating metrics with the effectiveness scheduler."""
    # Mock the _calculate_metrics method
    effectiveness_scheduler._calculate_metrics = MagicMock(return_value=asyncio.Future())
    effectiveness_scheduler._calculate_metrics.return_value.set_result({})
    
    # Call the calculate methods
    await effectiveness_scheduler.calculate_hourly_metrics()
    await effectiveness_scheduler.calculate_daily_metrics()
    await effectiveness_scheduler.calculate_weekly_metrics()
    await effectiveness_scheduler.calculate_monthly_metrics()
    
    # Check that _calculate_metrics was called
    assert effectiveness_scheduler._calculate_metrics.call_count == 4

@pytest.mark.asyncio
async def test_report_scheduler_generate_reports(report_scheduler):
    """Test generating reports with the report scheduler."""
    # Mock the batch calculator
    mock_batch_calculator = MagicMock()
    mock_batch_calculator.recalculate_all_metrics = MagicMock(return_value=asyncio.Future())
    mock_batch_calculator.recalculate_all_metrics.return_value.set_result({"tools_processed": 10})
    
    mock_batch_calculator.generate_periodic_reports = MagicMock(return_value=asyncio.Future())
    mock_batch_calculator.generate_periodic_reports.return_value.set_result({"reports_generated": 5})
    
    # Mock the _distribute_reports method
    report_scheduler._distribute_reports = MagicMock(return_value=asyncio.Future())
    report_scheduler._distribute_reports.return_value.set_result(None)
    
    # Patch the batch calculator
    with patch("analysis_engine.batch.metric_calculator.MetricBatchCalculator", return_value=mock_batch_calculator):
        # Call the generate methods
        await report_scheduler._generate_daily_reports()
        await report_scheduler._generate_weekly_reports()
        await report_scheduler._generate_monthly_reports()
    
    # Check that methods were called
    assert mock_batch_calculator.recalculate_all_metrics.call_count == 3
    assert mock_batch_calculator.generate_periodic_reports.call_count == 3
    assert report_scheduler._distribute_reports.call_count == 3
