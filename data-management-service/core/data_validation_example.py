#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for data validation.

This script shows how to create and run reconciliation tasks for validating data.
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


def plot_validation_results(result: Dict[str, Any], title: str) -> None:
    """
    Plot validation results.
    
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
        "Valid": summary.get("matched_records", 0),
        "Invalid": summary.get("total_records", 0) - summary.get("matched_records", 0)
    }
    
    ax1.bar(record_counts.keys(), record_counts.values())
    ax1.set_title("Record Validation")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # Plot issues by field
    issues_by_field = summary.get("issues_by_field", {})
    
    if issues_by_field:
        ax2.bar(issues_by_field.keys(), issues_by_field.values())
        ax2.set_title("Issues by Field")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"validation_{title.lower().replace(' ', '_')}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting data validation example")
    
    # Create data validation configuration
    logger.info("Creating data validation configuration")
    
    # Define primary source (data to validate)
    primary_source = {
        "source_id": "data_to_validate",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    # Define validation rules
    rules = [
        {
            "rule_id": "price_range",
            "name": "Price Range Validation",
            "field": "close",
            "comparison_type": "custom",
            "parameters": {
                "min_value": 0.5,
                "max_value": 2.0
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "volume_range",
            "name": "Volume Range Validation",
            "field": "volume",
            "comparison_type": "custom",
            "parameters": {
                "min_value": 0.0,
                "max_value": 10000.0
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "high_low_check",
            "name": "High-Low Check",
            "field": "high",
            "comparison_type": "custom",
            "parameters": {
                "compare_with": "low",
                "condition": "greater_than"
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "open_range_check",
            "name": "Open Range Check",
            "field": "open",
            "comparison_type": "custom",
            "parameters": {
                "compare_with": ["high", "low"],
                "condition": "between"
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "close_range_check",
            "name": "Close Range Check",
            "field": "close",
            "comparison_type": "custom",
            "parameters": {
                "compare_with": ["high", "low"],
                "condition": "between"
            },
            "severity": "ERROR"
        }
    ]
    
    config_id = await create_config(
        name="OHLCV Data Validation",
        reconciliation_type="custom",
        primary_source=primary_source,
        rules=rules,
        description="Validate OHLCV data against business rules",
        enabled=True
    )
    
    logger.info(f"Created data validation configuration: {config_id}")
    
    # Schedule and run data validation task
    logger.info("Scheduling data validation task")
    task_id = await schedule_task(
        config_id=config_id,
        scheduled_time=datetime.datetime.utcnow()
    )
    
    logger.info(f"Scheduled data validation task: {task_id}")
    
    logger.info("Running data validation task")
    result_id = await run_task(task_id=task_id)
    
    logger.info(f"Ran data validation task: {result_id}")
    
    # Get data validation result
    logger.info("Getting data validation result")
    result = await get_result(result_id=result_id)
    
    # Extract validation issues
    issues = result.get("issues", [])
    
    logger.info(f"Found {len(issues)} validation issues")
    
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
    
    # Plot validation results
    logger.info("Plotting validation results")
    plot_validation_results(result, "OHLCV Data Validation")
    
    logger.info("Data validation example complete")


if __name__ == "__main__":
    asyncio.run(main())
