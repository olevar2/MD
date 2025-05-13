#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for data lineage tracking.

This script shows how to create and run reconciliation tasks for tracking data lineage.
"""

import asyncio
import datetime
import logging
import json
from typing import Dict, List, Any, Optional

import httpx
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

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


async def get_configs() -> List[Dict[str, Any]]:
    """
    Get all reconciliation configurations.
    
    Returns:
        List of reconciliation configurations
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/reconciliation/configs")
        response.raise_for_status()
        
        return response.json()


def plot_data_lineage(configs: List[Dict[str, Any]], title: str) -> None:
    """
    Plot data lineage.
    
    Args:
        configs: List of reconciliation configurations
        title: Plot title
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for config in configs:
        # Get sources
        primary_source = config.get("primary_source", {})
        secondary_source = config.get("secondary_source", {})
        
        # Get source IDs
        primary_id = primary_source.get("source_id", "")
        secondary_id = secondary_source.get("source_id", "")
        
        # Add nodes
        if primary_id:
            G.add_node(primary_id, type=primary_source.get("source_type", ""))
        
        if secondary_id:
            G.add_node(secondary_id, type=secondary_source.get("source_type", ""))
        
        # Add edge
        if primary_id and secondary_id:
            G.add_edge(secondary_id, primary_id, config=config.get("name", ""))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define node colors based on type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "")
        if node_type == "ohlcv":
            node_colors.append("lightblue")
        elif node_type == "tick":
            node_colors.append("lightgreen")
        elif node_type == "alternative":
            node_colors.append("lightcoral")
        else:
            node_colors.append("lightgray")
    
    # Define layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    
    # Draw edge labels
    edge_labels = {(u, v): d["config"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"data_lineage_{title.lower().replace(' ', '_')}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting data lineage example")
    
    # Create data lineage configurations
    logger.info("Creating data lineage configurations")
    
    # Define sources
    raw_tick_source = {
        "source_id": "raw_tick_data",
        "source_type": "tick",
        "query_params": {
            "symbols": ["EURUSD"],
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    processed_tick_source = {
        "source_id": "processed_tick_data",
        "source_type": "tick",
        "query_params": {
            "symbols": ["EURUSD"],
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    ohlcv_1m_source = {
        "source_id": "ohlcv_1m_data",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD"],
            "timeframe": "1m",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    ohlcv_5m_source = {
        "source_id": "ohlcv_5m_data",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD"],
            "timeframe": "5m",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    ohlcv_1h_source = {
        "source_id": "ohlcv_1h_data",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD"],
            "timeframe": "1h",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    indicator_source = {
        "source_id": "indicator_data",
        "source_type": "alternative",
        "query_params": {
            "symbols": ["EURUSD"],
            "data_type": "indicator",
            "start_timestamp": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
            "end_timestamp": datetime.datetime.utcnow().isoformat()
        }
    }
    
    # Create configurations for data lineage
    config_ids = []
    
    # Raw tick to processed tick
    config_id = await create_config(
        name="Raw Tick to Processed Tick",
        reconciliation_type="derived",
        primary_source=processed_tick_source,
        secondary_source=raw_tick_source,
        description="Track lineage from raw tick data to processed tick data",
        enabled=True,
        metadata={"lineage": True}
    )
    config_ids.append(config_id)
    
    # Processed tick to OHLCV 1m
    config_id = await create_config(
        name="Processed Tick to OHLCV 1m",
        reconciliation_type="derived",
        primary_source=ohlcv_1m_source,
        secondary_source=processed_tick_source,
        description="Track lineage from processed tick data to OHLCV 1m data",
        enabled=True,
        metadata={"lineage": True}
    )
    config_ids.append(config_id)
    
    # OHLCV 1m to OHLCV 5m
    config_id = await create_config(
        name="OHLCV 1m to OHLCV 5m",
        reconciliation_type="derived",
        primary_source=ohlcv_5m_source,
        secondary_source=ohlcv_1m_source,
        description="Track lineage from OHLCV 1m data to OHLCV 5m data",
        enabled=True,
        metadata={"lineage": True}
    )
    config_ids.append(config_id)
    
    # OHLCV 5m to OHLCV 1h
    config_id = await create_config(
        name="OHLCV 5m to OHLCV 1h",
        reconciliation_type="derived",
        primary_source=ohlcv_1h_source,
        secondary_source=ohlcv_5m_source,
        description="Track lineage from OHLCV 5m data to OHLCV 1h data",
        enabled=True,
        metadata={"lineage": True}
    )
    config_ids.append(config_id)
    
    # OHLCV 1h to Indicator
    config_id = await create_config(
        name="OHLCV 1h to Indicator",
        reconciliation_type="derived",
        primary_source=indicator_source,
        secondary_source=ohlcv_1h_source,
        description="Track lineage from OHLCV 1h data to indicator data",
        enabled=True,
        metadata={"lineage": True}
    )
    config_ids.append(config_id)
    
    logger.info(f"Created {len(config_ids)} data lineage configurations")
    
    # Get all configurations
    logger.info("Getting all configurations")
    configs = await get_configs()
    
    # Filter configurations for data lineage
    lineage_configs = [
        config for config in configs
        if config.get("metadata", {}).get("lineage", False)
    ]
    
    logger.info(f"Retrieved {len(lineage_configs)} data lineage configurations")
    
    # Plot data lineage
    logger.info("Plotting data lineage")
    plot_data_lineage(lineage_configs, "Data Lineage")
    
    logger.info("Data lineage example complete")


if __name__ == "__main__":
    asyncio.run(main())
