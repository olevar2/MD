#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for derived data.

This script shows how to create and run reconciliation tasks for derived data.
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


def plot_reconciliation_result(result: Dict[str, Any], title: str) -> None:
    """
    Plot reconciliation result.
    
    Args:
        result: Reconciliation result
        title: Plot title
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot summary
    summary = result["summary"]
    
    # Plot record counts
    record_counts = {
        "Primary": summary["primary_records"],
        "Secondary": summary["secondary_records"],
        "Matched": summary["matched_records"]
    }
    
    ax1.bar(record_counts.keys(), record_counts.values())
    ax1.set_title("Record Counts")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # Plot issues by severity
    if "issues_by_severity" in summary:
        issues_by_severity = summary["issues_by_severity"]
        
        if issues_by_severity:
            ax2.bar(issues_by_severity.keys(), issues_by_severity.values())
            ax2.set_title("Issues by Severity")
            ax2.set_ylabel("Count")
            ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"reconciliation_{title.lower().replace(' ', '_')}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting derived reconciliation example")
    
    # Create derived reconciliation configuration
    logger.info("Creating derived reconciliation configuration")
    
    # Define primary source (derived data - SMA values)
    primary_source = {
        "source_id": "derived_sma",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        },
        "transformations": [
            {
                "type": "add_column",
                "field": "sma_14",
                "expression": "column:close"  # Placeholder for actual SMA calculation
            }
        ]
    }
    
    # Define secondary source (source data - raw OHLCV data)
    secondary_source = {
        "source_id": "raw_ohlcv",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        },
        "transformations": [
            {
                "type": "add_column",
                "field": "sma_14",
                "expression": "column:close"  # Placeholder for actual SMA calculation
            }
        ]
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
            "rule_id": "sma_14",
            "name": "SMA 14",
            "field": "sma_14",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 0.0001
            },
            "severity": "ERROR"
        }
    ]
    
    derived_config_id = await create_config(
        name="SMA Derived Reconciliation",
        reconciliation_type="derived",
        primary_source=primary_source,
        secondary_source=secondary_source,
        rules=rules,
        description="Reconcile derived SMA values with calculated values",
        enabled=True
    )
    
    logger.info(f"Created derived reconciliation configuration: {derived_config_id}")
    
    # Schedule and run derived reconciliation task
    logger.info("Scheduling derived reconciliation task")
    derived_task_id = await schedule_task(
        config_id=derived_config_id,
        scheduled_time=datetime.datetime.utcnow()
    )
    
    logger.info(f"Scheduled derived reconciliation task: {derived_task_id}")
    
    logger.info("Running derived reconciliation task")
    derived_result_id = await run_task(task_id=derived_task_id)
    
    logger.info(f"Ran derived reconciliation task: {derived_result_id}")
    
    # Get and plot derived reconciliation result
    logger.info("Getting derived reconciliation result")
    derived_result = await get_result(result_id=derived_result_id)
    
    logger.info("Plotting derived reconciliation result")
    plot_reconciliation_result(
        result=derived_result,
        title="Derived Reconciliation"
    )
    
    logger.info("Derived reconciliation example complete")


if __name__ == "__main__":
    asyncio.run(main())
