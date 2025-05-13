#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for anomaly detection.

This script shows how to create and run reconciliation tasks for detecting anomalies in data.
"""

import asyncio
import datetime
import logging
import json
import random
from typing import Dict, List, Any, Optional

import httpx
import pandas as pd
import numpy as np
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


def plot_anomalies(
    df: pd.DataFrame,
    anomalies: List[Dict[str, Any]],
    title: str
) -> None:
    """
    Plot anomalies.
    
    Args:
        df: DataFrame with data
        anomalies: List of anomalies
        title: Plot title
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    ax.plot(df["timestamp"], df["close"], label="Close Price")
    
    # Plot anomalies
    anomaly_timestamps = []
    anomaly_values = []
    anomaly_severities = []
    
    for anomaly in anomalies:
        # Get timestamp from metadata
        timestamp_str = anomaly.get("metadata", {}).get("timestamp")
        if timestamp_str:
            timestamp = datetime.datetime.fromisoformat(timestamp_str)
            anomaly_timestamps.append(timestamp)
            
            # Get value
            value = anomaly.get("primary_value")
            anomaly_values.append(value)
            
            # Get severity
            severity = anomaly.get("severity")
            anomaly_severities.append(severity)
    
    # Define colors for severities
    severity_colors = {
        "INFO": "blue",
        "WARNING": "orange",
        "ERROR": "red",
        "CRITICAL": "purple"
    }
    
    # Plot anomalies
    for timestamp, value, severity in zip(anomaly_timestamps, anomaly_values, anomaly_severities):
        color = severity_colors.get(severity, "red")
        ax.scatter(timestamp, value, color=color, s=100, marker="x", label=f"{severity} Anomaly")
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"anomalies_{title.lower().replace(' ', '_')}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting anomaly detection example")
    
    # Get OHLCV data
    logger.info("Getting OHLCV data")
    end_timestamp = datetime.datetime.utcnow()
    start_timestamp = end_timestamp - datetime.timedelta(days=7)
    
    df = await get_ohlcv_data(
        symbol="EURUSD",
        timeframe="1h",
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp
    )
    
    if df.empty:
        logger.error("No OHLCV data found")
        return
    
    logger.info(f"Retrieved {len(df)} OHLCV records")
    
    # Create anomaly detection configuration
    logger.info("Creating anomaly detection configuration")
    
    # Define primary source (current data)
    primary_source = {
        "source_id": "current",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD"],
            "timeframe": "1h",
            "start_timestamp": start_timestamp.isoformat(),
            "end_timestamp": end_timestamp.isoformat()
        }
    }
    
    # Define secondary source (expected data pattern)
    secondary_source = {
        "source_id": "expected",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD"],
            "timeframe": "1h",
            "start_timestamp": start_timestamp.isoformat(),
            "end_timestamp": end_timestamp.isoformat()
        },
        "transformations": [
            {
                "type": "add_column",
                "field": "expected_close",
                "expression": "column:close"
            }
        ]
    }
    
    # Define rules for anomaly detection
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
            "rule_id": "price_anomaly",
            "name": "Price Anomaly Detection",
            "field": "close",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 0.001  # 0.1% tolerance
            },
            "severity": "WARNING"
        }
    ]
    
    config_id = await create_config(
        name="Price Anomaly Detection",
        reconciliation_type="cross_source",
        primary_source=primary_source,
        secondary_source=secondary_source,
        rules=rules,
        description="Detect anomalies in price data",
        enabled=True
    )
    
    logger.info(f"Created anomaly detection configuration: {config_id}")
    
    # Schedule and run anomaly detection task
    logger.info("Scheduling anomaly detection task")
    task_id = await schedule_task(
        config_id=config_id,
        scheduled_time=datetime.datetime.utcnow()
    )
    
    logger.info(f"Scheduled anomaly detection task: {task_id}")
    
    logger.info("Running anomaly detection task")
    result_id = await run_task(task_id=task_id)
    
    logger.info(f"Ran anomaly detection task: {result_id}")
    
    # Get anomaly detection result
    logger.info("Getting anomaly detection result")
    result = await get_result(result_id=result_id)
    
    # Extract anomalies
    anomalies = result.get("issues", [])
    
    logger.info(f"Found {len(anomalies)} anomalies")
    
    # Add timestamp to anomalies for plotting
    for anomaly in anomalies:
        # Get primary value
        primary_value = anomaly.get("primary_value")
        
        # Find matching record in DataFrame
        matching_record = df[df["close"] == primary_value]
        
        if not matching_record.empty:
            # Get timestamp
            timestamp = matching_record.iloc[0]["timestamp"]
            
            # Add timestamp to metadata
            if "metadata" not in anomaly:
                anomaly["metadata"] = {}
            
            anomaly["metadata"]["timestamp"] = timestamp.isoformat()
    
    # Plot anomalies
    logger.info("Plotting anomalies")
    plot_anomalies(df, anomalies, "Price Anomaly Detection")
    
    logger.info("Anomaly detection example complete")


if __name__ == "__main__":
    asyncio.run(main())
