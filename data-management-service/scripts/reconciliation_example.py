#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system.

This script shows how to create and run reconciliation tasks.
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
    logger.info("Starting reconciliation example")
    
    # Create cross-source reconciliation configuration
    logger.info("Creating cross-source reconciliation configuration")
    
    # Define primary source (OHLCV data from provider 1)
    primary_source = {
        "source_id": "provider1",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(),
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
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(),
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
        },
        {
            "rule_id": "volume",
            "name": "Volume",
            "field": "volume",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 1.0
            },
            "severity": "WARNING"
        }
    ]
    
    cross_source_config_id = await create_config(
        name="OHLCV Cross-Source Reconciliation",
        reconciliation_type="cross_source",
        primary_source=primary_source,
        secondary_source=secondary_source,
        rules=rules,
        description="Reconcile OHLCV data between two providers",
        enabled=True
    )
    
    logger.info(f"Created cross-source reconciliation configuration: {cross_source_config_id}")
    
    # Schedule and run cross-source reconciliation task
    logger.info("Scheduling cross-source reconciliation task")
    cross_source_task_id = await schedule_task(
        config_id=cross_source_config_id,
        scheduled_time=datetime.datetime.utcnow()
    )
    
    logger.info(f"Scheduled cross-source reconciliation task: {cross_source_task_id}")
    
    logger.info("Running cross-source reconciliation task")
    cross_source_result_id = await run_task(task_id=cross_source_task_id)
    
    logger.info(f"Ran cross-source reconciliation task: {cross_source_result_id}")
    
    # Get and plot cross-source reconciliation result
    logger.info("Getting cross-source reconciliation result")
    cross_source_result = await get_result(result_id=cross_source_result_id)
    
    logger.info("Plotting cross-source reconciliation result")
    plot_reconciliation_result(
        result=cross_source_result,
        title="Cross-Source Reconciliation"
    )
    
    # Create temporal reconciliation configuration
    logger.info("Creating temporal reconciliation configuration")
    
    # Define primary source (current OHLCV data)
    primary_source = {
        "source_id": "current",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(),
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
    
    temporal_config_id = await create_config(
        name="OHLCV Temporal Reconciliation",
        reconciliation_type="temporal",
        primary_source=primary_source,
        rules=rules,
        description="Reconcile current OHLCV data with historical data",
        enabled=True
    )
    
    logger.info(f"Created temporal reconciliation configuration: {temporal_config_id}")
    
    # Schedule and run temporal reconciliation task
    logger.info("Scheduling temporal reconciliation task")
    temporal_task_id = await schedule_task(
        config_id=temporal_config_id,
        scheduled_time=datetime.datetime.utcnow()
    )
    
    logger.info(f"Scheduled temporal reconciliation task: {temporal_task_id}")
    
    logger.info("Running temporal reconciliation task")
    temporal_result_id = await run_task(task_id=temporal_task_id)
    
    logger.info(f"Ran temporal reconciliation task: {temporal_result_id}")
    
    # Get and plot temporal reconciliation result
    logger.info("Getting temporal reconciliation result")
    temporal_result = await get_result(result_id=temporal_result_id)
    
    logger.info("Plotting temporal reconciliation result")
    plot_reconciliation_result(
        result=temporal_result,
        title="Temporal Reconciliation"
    )
    
    logger.info("Reconciliation example complete")


if __name__ == "__main__":
    asyncio.run(main())
