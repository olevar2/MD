#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for scheduling.

This script shows how to create and schedule reconciliation tasks.
"""

import asyncio
import datetime
import logging
import json
import time
from typing import Dict, List, Any, Optional

import httpx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API URL
API_URL = "http://localhost:8000"


async def create_config(
    name: str,
    reconciliation_type: str,
    primary_source: Dict[str, Any],
    secondary_source: Optional[Dict[str, Any]] = None,
    rules: Optional[List[Dict[str, Any]]] = None,
    description: Optional[str] = None,
    schedule: Optional[str] = None,
    enabled: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a reconciliation configuration.
    
    Args:
        name: Configuration name
        reconciliation_type: Type of reconciliation
        primary_source: Primary data source configuration
        secondary_source: Secondary data source configuration
        rules: Reconciliation rules
        description: Configuration description
        schedule: Cron expression for scheduling
        enabled: Whether the configuration is enabled
        metadata: Additional metadata
        
    Returns:
        Config ID
    """
    data = {
        "name": name,
        "reconciliation_type": reconciliation_type,
        "primary_source": primary_source,
        "secondary_source": secondary_source,
        "rules": rules or [],
        "description": description,
        "schedule": schedule,
        "enabled": enabled,
        "metadata": metadata or {}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/reconciliation/configs", json=data)
        response.raise_for_status()
        
        return response.json()["config_id"]


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


async def get_task(task_id: str) -> Dict[str, Any]:
    """
    Get a reconciliation task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Reconciliation task
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/reconciliation/tasks/{task_id}")
        response.raise_for_status()
        
        return response.json()


async def get_result(result_id: str) -> Dict[str, Any]:
    """
    Get a reconciliation result.
    
    Args:
        result_id: Result ID
        
    Returns:
        Reconciliation result
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/reconciliation/results/{result_id}")
        response.raise_for_status()
        
        return response.json()


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


def plot_task_status(tasks: List[Dict[str, Any]], title: str) -> None:
    """
    Plot task status.
    
    Args:
        tasks: List of tasks
        title: Plot title
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count tasks by status
    status_counts = {}
    for task in tasks:
        status = task["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Plot status counts
    ax.bar(status_counts.keys(), status_counts.values())
    ax.set_title("Task Status")
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"tasks_{title.lower().replace(' ', '_')}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting scheduled reconciliation example")
    
    # Create cross-source reconciliation configuration
    logger.info("Creating cross-source reconciliation configuration")
    
    # Define primary source (OHLCV data from provider 1)
    primary_source = {
        "source_id": "provider1",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    # Define secondary source (OHLCV data from provider 2)
    secondary_source = {
        "source_id": "provider2",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    # Define reconciliation rules
    rules = [
        {
            "rule_id": "join_key_symbol",
            "name": "Symbol Join Key",
            "field": "symbol",
            "comparison_type": "join_key",
            "parameters": {},
            "severity": "ERROR"
        },
        {
            "rule_id": "join_key_timestamp",
            "name": "Timestamp Join Key",
            "field": "timestamp",
            "comparison_type": "join_key",
            "parameters": {},
            "severity": "ERROR"
        },
        {
            "rule_id": "close_price",
            "name": "Close Price",
            "field": "close",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 0.0001
            },
            "severity": "ERROR"
        }
    ]
    
    config_id = await create_config(
        name="Scheduled OHLCV Reconciliation",
        reconciliation_type="cross_source",
        primary_source=primary_source,
        secondary_source=secondary_source,
        rules=rules,
        description="Scheduled reconciliation of OHLCV data between two providers",
        schedule="0 * * * *",  # Every hour
        enabled=True
    )
    
    logger.info(f"Created reconciliation configuration: {config_id}")
    
    # Schedule multiple tasks
    logger.info("Scheduling multiple tasks")
    task_ids = []
    
    # Schedule tasks for the next 24 hours, one per hour
    for i in range(24):
        scheduled_time = datetime.datetime.utcnow() + datetime.timedelta(hours=i)
        
        task_id = await schedule_task(
            config_id=config_id,
            scheduled_time=scheduled_time,
            metadata={"hour": i}
        )
        
        task_ids.append(task_id)
        logger.info(f"Scheduled task {task_id} for {scheduled_time}")
    
    # Run the first few tasks
    logger.info("Running the first few tasks")
    for i in range(3):
        task_id = task_ids[i]
        
        logger.info(f"Running task {task_id}")
        result_id = await run_task(task_id=task_id)
        
        logger.info(f"Ran task {task_id}, result: {result_id}")
        
        # Wait a bit between tasks
        await asyncio.sleep(1)
    
    # Get all tasks
    logger.info("Getting all tasks")
    tasks = await get_tasks(config_id=config_id)
    
    logger.info(f"Retrieved {len(tasks)} tasks")
    
    # Plot task status
    logger.info("Plotting task status")
    plot_task_status(tasks, "Scheduled Reconciliation")
    
    logger.info("Scheduled reconciliation example complete")


if __name__ == "__main__":
    asyncio.run(main())
