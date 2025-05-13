#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for data quality monitoring.

This script shows how to create and run reconciliation tasks for data quality monitoring.
"""

import asyncio
import datetime
import logging
import json
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


async def get_results(
    config_id: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get reconciliation results.
    
    Args:
        config_id: Filter by config ID
        status: Filter by status
        start_date: Filter by start time (start)
        end_date: Filter by start time (end)
        limit: Maximum number of records to return
        offset: Offset for pagination
        
    Returns:
        List of reconciliation results
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
        response = await client.get(f"{API_URL}/reconciliation/results", params=params)
        response.raise_for_status()
        
        return response.json()


def plot_data_quality_trend(results: List[Dict[str, Any]], title: str) -> None:
    """
    Plot data quality trend.
    
    Args:
        results: List of results
        title: Plot title
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Extract data
    timestamps = []
    matched_percentages = []
    issue_counts = []
    
    for result in results:
        # Get timestamp
        timestamp = datetime.datetime.fromisoformat(result["start_time"])
        timestamps.append(timestamp)
        
        # Get summary
        summary = result["summary"]
        
        # Calculate matched percentage
        total_records = summary.get("total_records", 0)
        matched_records = summary.get("matched_records", 0)
        
        if total_records > 0:
            matched_percentage = matched_records / total_records * 100
        else:
            matched_percentage = 0
        
        matched_percentages.append(matched_percentage)
        
        # Get issue count
        issue_count = len(result.get("issues", []))
        issue_counts.append(issue_count)
    
    # Sort by timestamp
    sorted_data = sorted(zip(timestamps, matched_percentages, issue_counts))
    timestamps, matched_percentages, issue_counts = zip(*sorted_data) if sorted_data else ([], [], [])
    
    # Plot matched percentage trend
    ax1.plot(timestamps, matched_percentages, marker="o")
    ax1.set_title("Data Quality Trend - Matched Percentage")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Matched Percentage (%)")
    ax1.grid(True)
    
    # Plot issue count trend
    ax2.plot(timestamps, issue_counts, marker="o", color="red")
    ax2.set_title("Data Quality Trend - Issue Count")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Issue Count")
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"data_quality_{title.lower().replace(' ', '_')}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting data quality monitoring example")
    
    # Create data quality monitoring configuration
    logger.info("Creating data quality monitoring configuration")
    
    # Define primary source (current OHLCV data)
    primary_source = {
        "source_id": "current",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    # Define secondary source (expected data pattern)
    secondary_source = {
        "source_id": "expected",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    # Define reconciliation rules for data quality
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
            "rule_id": "price_range",
            "name": "Price Range Check",
            "field": "close",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 0.01  # 1% tolerance
            },
            "severity": "WARNING"
        },
        {
            "rule_id": "volume_check",
            "name": "Volume Check",
            "field": "volume",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 10.0  # 10 units tolerance
            },
            "severity": "INFO"
        }
    ]
    
    config_id = await create_config(
        name="Data Quality Monitoring",
        reconciliation_type="cross_source",
        primary_source=primary_source,
        secondary_source=secondary_source,
        rules=rules,
        description="Monitor data quality by comparing with expected patterns",
        schedule="0 * * * *",  # Every hour
        enabled=True
    )
    
    logger.info(f"Created data quality monitoring configuration: {config_id}")
    
    # Run multiple data quality checks
    logger.info("Running multiple data quality checks")
    result_ids = []
    
    # Run checks for different time periods
    for i in range(24):
        # Schedule task
        task_id = await schedule_task(
            config_id=config_id,
            scheduled_time=datetime.datetime.utcnow() - datetime.timedelta(hours=i),
            metadata={"hour": i}
        )
        
        logger.info(f"Scheduled task {task_id}")
        
        # Run task
        result_id = await run_task(task_id=task_id)
        result_ids.append(result_id)
        
        logger.info(f"Ran task {task_id}, result: {result_id}")
        
        # Wait a bit between tasks
        await asyncio.sleep(1)
    
    # Get all results
    logger.info("Getting all results")
    results = await get_results(config_id=config_id)
    
    logger.info(f"Retrieved {len(results)} results")
    
    # Plot data quality trend
    logger.info("Plotting data quality trend")
    plot_data_quality_trend(results, "Data Quality Monitoring")
    
    logger.info("Data quality monitoring example complete")


if __name__ == "__main__":
    asyncio.run(main())
