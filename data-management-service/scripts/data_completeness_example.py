#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for data completeness checking.

This script shows how to create and run reconciliation tasks for checking data completeness.
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


async def get_ohlcv_data(
    symbol: str,
    timeframe: str,
    start_timestamp: datetime.datetime,
    end_timestamp: datetime.datetime
) -> pd.DataFrame:
    """
    Get OHLCV data.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_timestamp: Start timestamp
        end_timestamp: End timestamp
        
    Returns:
        DataFrame with OHLCV data
    """
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start_timestamp": start_timestamp.isoformat(),
        "end_timestamp": end_timestamp.isoformat(),
        "format": "json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/historical/ohlcv", params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df


def generate_expected_timestamps(
    timeframe: str,
    start_timestamp: datetime.datetime,
    end_timestamp: datetime.datetime
) -> List[datetime.datetime]:
    """
    Generate expected timestamps for a timeframe.
    
    Args:
        timeframe: Timeframe
        start_timestamp: Start timestamp
        end_timestamp: End timestamp
        
    Returns:
        List of expected timestamps
    """
    # Map timeframe to timedelta
    timeframe_map = {
        "1m": datetime.timedelta(minutes=1),
        "5m": datetime.timedelta(minutes=5),
        "15m": datetime.timedelta(minutes=15),
        "30m": datetime.timedelta(minutes=30),
        "1h": datetime.timedelta(hours=1),
        "4h": datetime.timedelta(hours=4),
        "1d": datetime.timedelta(days=1)
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    delta = timeframe_map[timeframe]
    
    # Generate timestamps
    timestamps = []
    current = start_timestamp
    
    while current <= end_timestamp:
        timestamps.append(current)
        current += delta
    
    return timestamps


def plot_completeness_results(
    actual_timestamps: List[datetime.datetime],
    expected_timestamps: List[datetime.datetime],
    title: str
) -> None:
    """
    Plot completeness results.
    
    Args:
        actual_timestamps: Actual timestamps
        expected_timestamps: Expected timestamps
        title: Plot title
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert to sets for comparison
    actual_set = set(actual_timestamps)
    expected_set = set(expected_timestamps)
    
    # Find missing timestamps
    missing_set = expected_set - actual_set
    missing_timestamps = sorted(list(missing_set))
    
    # Plot record counts
    record_counts = {
        "Expected": len(expected_timestamps),
        "Actual": len(actual_timestamps),
        "Missing": len(missing_timestamps)
    }
    
    ax1.bar(record_counts.keys(), record_counts.values())
    ax1.set_title("Record Counts")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # Calculate completeness percentage
    completeness = len(actual_timestamps) / len(expected_timestamps) * 100 if expected_timestamps else 0
    
    # Add completeness text
    ax1.text(
        0.5, 0.9,
        f"Completeness: {completeness:.2f}%",
        horizontalalignment="center",
        transform=ax1.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)
    )
    
    # Plot missing timestamps
    if missing_timestamps:
        # Group missing timestamps by day
        missing_df = pd.DataFrame({"timestamp": missing_timestamps})
        missing_df["date"] = missing_df["timestamp"].dt.date
        missing_counts = missing_df.groupby("date").size()
        
        ax2.bar(missing_counts.index, missing_counts.values)
        ax2.set_title("Missing Records by Day")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.text(
            0.5, 0.5,
            "No missing records",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
            fontsize=12
        )
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"completeness_{title.lower().replace(' ', '_')}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting data completeness example")
    
    # Define time range
    end_timestamp = datetime.datetime.utcnow()
    start_timestamp = end_timestamp - datetime.timedelta(days=7)
    
    # Define symbol and timeframe
    symbol = "EURUSD"
    timeframe = "1h"
    
    # Get actual data
    logger.info(f"Getting actual data for {symbol} {timeframe}")
    df = await get_ohlcv_data(
        symbol=symbol,
        timeframe=timeframe,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp
    )
    
    if df.empty:
        logger.error("No data found")
        return
    
    logger.info(f"Retrieved {len(df)} records")
    
    # Get actual timestamps
    actual_timestamps = df["timestamp"].tolist()
    
    # Generate expected timestamps
    logger.info("Generating expected timestamps")
    expected_timestamps = generate_expected_timestamps(
        timeframe=timeframe,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp
    )
    
    logger.info(f"Generated {len(expected_timestamps)} expected timestamps")
    
    # Create data completeness configuration
    logger.info("Creating data completeness configuration")
    
    # Define primary source (actual data)
    primary_source = {
        "source_id": "actual",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": [symbol],
            "timeframe": timeframe,
            "start_timestamp": start_timestamp.isoformat(),
            "end_timestamp": end_timestamp.isoformat()
        }
    }
    
    # Define secondary source (expected data pattern)
    secondary_source = {
        "source_id": "expected",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": [symbol],
            "timeframe": timeframe,
            "start_timestamp": start_timestamp.isoformat(),
            "end_timestamp": end_timestamp.isoformat()
        }
    }
    
    # Define rules for completeness checking
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
        }
    ]
    
    config_id = await create_config(
        name="Data Completeness Check",
        reconciliation_type="cross_source",
        primary_source=primary_source,
        secondary_source=secondary_source,
        rules=rules,
        description="Check data completeness",
        enabled=True
    )
    
    logger.info(f"Created data completeness configuration: {config_id}")
    
    # Schedule and run data completeness task
    logger.info("Scheduling data completeness task")
    task_id = await schedule_task(
        config_id=config_id,
        scheduled_time=datetime.datetime.utcnow()
    )
    
    logger.info(f"Scheduled data completeness task: {task_id}")
    
    logger.info("Running data completeness task")
    result_id = await run_task(task_id=task_id)
    
    logger.info(f"Ran data completeness task: {result_id}")
    
    # Get data completeness result
    logger.info("Getting data completeness result")
    result = await get_result(result_id=result_id)
    
    # Extract completeness issues
    issues = result.get("issues", [])
    
    logger.info(f"Found {len(issues)} completeness issues")
    
    # Plot completeness results
    logger.info("Plotting completeness results")
    plot_completeness_results(
        actual_timestamps=actual_timestamps,
        expected_timestamps=expected_timestamps,
        title=f"{symbol} {timeframe} Completeness"
    )
    
    logger.info("Data completeness example complete")


if __name__ == "__main__":
    asyncio.run(main())
