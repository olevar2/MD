#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for data consistency checking.

This script shows how to create and run reconciliation tasks for checking data consistency.
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


def plot_consistency_results(result: Dict[str, Any], title: str) -> None:
    """
    Plot consistency results.
    
    Args:
        result: Reconciliation result
        title: Plot title
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get summary
    summary = result.get("summary", {})
    
    # Plot record counts
    record_counts = {
        "Total": summary.get("total_records", 0),
        "Consistent": summary.get("matched_records", 0),
        "Inconsistent": summary.get("total_records", 0) - summary.get("matched_records", 0)
    }
    
    ax1.bar(record_counts.keys(), record_counts.values())
    ax1.set_title("Record Consistency")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # Calculate consistency percentage
    consistency = record_counts["Consistent"] / record_counts["Total"] * 100 if record_counts["Total"] > 0 else 0
    
    # Add consistency text
    ax1.text(
        0.5, 0.9,
        f"Consistency: {consistency:.2f}%",
        horizontalalignment="center",
        transform=ax1.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)
    )
    
    # Plot issues by field
    issues_by_field = summary.get("issues_by_field", {})
    
    if issues_by_field:
        ax2.bar(issues_by_field.keys(), issues_by_field.values())
        ax2.set_title("Inconsistencies by Field")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5, 0.5,
            "No inconsistencies found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
            fontsize=12
        )
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"consistency_{title.lower().replace(' ', '_')}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting data consistency example")
    
    # Create data consistency configuration
    logger.info("Creating data consistency configuration")
    
    # Define primary source (OHLCV data)
    primary_source = {
        "source_id": "ohlcv",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    # Define secondary source (derived OHLCV data from tick data)
    secondary_source = {
        "source_id": "derived_ohlcv",
        "source_type": "tick",
        "query_params": {
            "symbols": ["EURUSD"],
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        },
        "transformations": [
            {
                "type": "aggregate",
                "group_by": ["symbol", "timestamp"],
                "aggregations": {
                    "open": {"column": "bid", "function": "first"},
                    "high": {"column": "bid", "function": "max"},
                    "low": {"column": "bid", "function": "min"},
                    "close": {"column": "bid", "function": "last"},
                    "volume": {"column": "bid_volume", "function": "sum"}
                }
            }
        ]
    }
    
    # Define rules for consistency checking
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
            "rule_id": "open_price",
            "name": "Open Price Consistency",
            "field": "open",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 0.0001  # 0.01% tolerance
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "high_price",
            "name": "High Price Consistency",
            "field": "high",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 0.0001  # 0.01% tolerance
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "low_price",
            "name": "Low Price Consistency",
            "field": "low",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 0.0001  # 0.01% tolerance
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "close_price",
            "name": "Close Price Consistency",
            "field": "close",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 0.0001  # 0.01% tolerance
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "volume",
            "name": "Volume Consistency",
            "field": "volume",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 1.0  # 1 unit tolerance
            },
            "severity": "WARNING"
        }
    ]
    
    config_id = await create_config(
        name="OHLCV Data Consistency",
        reconciliation_type="cross_source",
        primary_source=primary_source,
        secondary_source=secondary_source,
        rules=rules,
        description="Check consistency between OHLCV data and derived OHLCV data from tick data",
        enabled=True
    )
    
    logger.info(f"Created data consistency configuration: {config_id}")
    
    # Schedule and run data consistency task
    logger.info("Scheduling data consistency task")
    task_id = await schedule_task(
        config_id=config_id,
        scheduled_time=datetime.datetime.utcnow()
    )
    
    logger.info(f"Scheduled data consistency task: {task_id}")
    
    logger.info("Running data consistency task")
    result_id = await run_task(task_id=task_id)
    
    logger.info(f"Ran data consistency task: {result_id}")
    
    # Get data consistency result
    logger.info("Getting data consistency result")
    result = await get_result(result_id=result_id)
    
    # Extract consistency issues
    issues = result.get("issues", [])
    
    logger.info(f"Found {len(issues)} consistency issues")
    
    # Group issues by field
    issues_by_field = {}
    for issue in issues:
        field = issue.get("field", "unknown")
        if field not in issues_by_field:
            issues_by_field[field] = []
        
        issues_by_field[field].append(issue)
    
    # Print issues by field
    for field, field_issues in issues_by_field.items():
        logger.info(f"Field: {field}, Issues: {len(field_issues)}")
        
        # Print first few issues
        for i, issue in enumerate(field_issues[:3]):
            logger.info(f"  Issue {i+1}: {issue.get('description')}")
        
        if len(field_issues) > 3:
            logger.info(f"  ... and {len(field_issues) - 3} more issues")
    
    # Plot consistency results
    logger.info("Plotting consistency results")
    plot_consistency_results(result, "OHLCV Data Consistency")
    
    logger.info("Data consistency example complete")


if __name__ == "__main__":
    asyncio.run(main())
