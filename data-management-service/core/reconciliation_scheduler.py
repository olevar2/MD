#!/usr/bin/env python
"""
Reconciliation Scheduler.

This script schedules and runs reconciliation tasks based on their schedules.
It is designed to be run as a cron job or a scheduled task.
"""

import asyncio
import datetime
import logging
import json
import time
import os
from typing import Dict, List, Any, Optional

import httpx
import croniter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("reconciliation_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")


async def get_configs(enabled: bool = True) -> List[Dict[str, Any]]:
    """
    Get reconciliation configurations.
    
    Args:
        enabled: Filter by enabled status
        
    Returns:
        List of reconciliation configurations
    """
    params = {
        "enabled": enabled
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/reconciliation/configs", params=params)
        response.raise_for_status()
        
        return response.json()


async def schedule_task(
    config_id: str,
    scheduled_time: Optional[datetime.datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Schedule a reconciliation task.
    
    Args:
        config_id: Configuration ID
        scheduled_time: Scheduled time
        metadata: Additional metadata
        
    Returns:
        Task ID
    """
    data = {
        "config_id": config_id,
        "metadata": metadata or {}
    }
    
    if scheduled_time:
        data["scheduled_time"] = scheduled_time.isoformat()
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/reconciliation/tasks", json=data)
        response.raise_for_status()
        
        return response.json()["task_id"]


async def run_task(task_id: str) -> str:
    """
    Run a reconciliation task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Result ID
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/reconciliation/tasks/{task_id}/run")
        response.raise_for_status()
        
        return response.json()["result_id"]


async def get_tasks(
    config_id: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get reconciliation tasks.
    
    Args:
        config_id: Filter by config ID
        status: Filter by status
        start_date: Filter by scheduled time (start)
        end_date: Filter by scheduled time (end)
        limit: Maximum number of records to return
        offset: Offset for pagination
        
    Returns:
        List of reconciliation tasks
    """
    params = {
        "limit": limit,
        "offset": offset
    }
    
    if config_id:
        params["config_id"] = config_id
    
    if status:
        params["status"] = status
    
    if start_date:
        params["start_date"] = start_date.isoformat()
    
    if end_date:
        params["end_date"] = end_date.isoformat()
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/reconciliation/tasks", params=params)
        response.raise_for_status()
        
        return response.json()


async def schedule_and_run_tasks() -> None:
    """Schedule and run reconciliation tasks."""
    # Get enabled configurations
    logger.info("Getting enabled configurations")
    configs = await get_configs(enabled=True)
    
    logger.info(f"Found {len(configs)} enabled configurations")
    
    # Current time
    now = datetime.datetime.utcnow()
    
    # Process each configuration
    for config in configs:
        config_id = config.get("config_id")
        name = config.get("name")
        schedule = config.get("schedule")
        
        if not schedule:
            logger.info(f"Configuration {name} ({config_id}) has no schedule, skipping")
            continue
        
        logger.info(f"Processing configuration {name} ({config_id}) with schedule {schedule}")
        
        try:
            # Parse cron expression
            cron = croniter.croniter(schedule, now)
            
            # Get next run time
            next_run = cron.get_prev(datetime.datetime)
            
            # Get tasks for this configuration
            tasks = await get_tasks(
                config_id=config_id,
                start_date=next_run - datetime.timedelta(minutes=5),
                end_date=next_run + datetime.timedelta(minutes=5)
            )
            
            # Check if a task already exists for this run time
            task_exists = any(
                abs((datetime.datetime.fromisoformat(task.get("scheduled_time")) - next_run).total_seconds()) < 300
                for task in tasks
            )
            
            if task_exists:
                logger.info(f"Task already exists for configuration {name} ({config_id}) at {next_run}, skipping")
                continue
            
            # Schedule task
            logger.info(f"Scheduling task for configuration {name} ({config_id}) at {next_run}")
            task_id = await schedule_task(
                config_id=config_id,
                scheduled_time=next_run,
                metadata={"scheduled_by": "reconciliation_scheduler"}
            )
            
            logger.info(f"Scheduled task {task_id} for configuration {name} ({config_id}) at {next_run}")
            
            # Run task
            logger.info(f"Running task {task_id}")
            result_id = await run_task(task_id=task_id)
            
            logger.info(f"Ran task {task_id}, result: {result_id}")
        
        except Exception as e:
            logger.error(f"Error processing configuration {name} ({config_id}): {e}")


async def main() -> None:
    """Main entry point."""
    logger.info("Starting reconciliation scheduler")
    
    while True:
        try:
            # Schedule and run tasks
            await schedule_and_run_tasks()
            
            # Wait for next minute
            now = datetime.datetime.utcnow()
            next_minute = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
            sleep_seconds = (next_minute - now).total_seconds()
            
            logger.info(f"Waiting {sleep_seconds:.2f} seconds until next minute")
            await asyncio.sleep(sleep_seconds)
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying


if __name__ == "__main__":
    asyncio.run(main())
