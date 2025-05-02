"""
Scheduler Factory Module

This module provides factory functions for creating and initializing schedulers.
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable

from analysis_engine.core.container import ServiceContainer
from analysis_engine.scheduling.effectiveness_scheduler import ToolEffectivenessScheduler
from analysis_engine.scheduling.report_scheduler import ReportScheduler
from analysis_engine.db.connection import get_db_session

logger = logging.getLogger(__name__)

async def initialize_schedulers(container: ServiceContainer) -> None:
    """
    Initialize and register schedulers with the service container.
    
    Args:
        container: Service container to register schedulers with
    """
    try:
        # Create effectiveness scheduler
        effectiveness_scheduler = ToolEffectivenessScheduler()
        container.register_service("effectiveness_scheduler", effectiveness_scheduler)
        
        # Create report scheduler
        report_scheduler = ReportScheduler(get_db_session)
        container.register_service("report_scheduler", report_scheduler)
        
        # Start schedulers
        await effectiveness_scheduler.start()
        await report_scheduler.start()
        
        logger.info("Schedulers initialized and started")
    except Exception as e:
        logger.error(f"Error initializing schedulers: {e}", exc_info=True)
        raise

async def cleanup_schedulers(container: ServiceContainer) -> None:
    """
    Stop and clean up schedulers.
    
    Args:
        container: Service container containing the schedulers
    """
    try:
        # Get schedulers from container
        effectiveness_scheduler = container.get_service("effectiveness_scheduler")
        report_scheduler = container.get_service("report_scheduler")
        
        # Stop schedulers
        await effectiveness_scheduler.stop()
        await report_scheduler.stop()
        
        logger.info("Schedulers stopped")
    except Exception as e:
        logger.error(f"Error stopping schedulers: {e}", exc_info=True)
        raise
