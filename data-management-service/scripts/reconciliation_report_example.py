#!/usr/bin/env python
"""
Example script demonstrating how to use the Data Reconciliation system for generating reports.

This script shows how to create and run reconciliation tasks and generate reports.
"""

import asyncio
import datetime
import logging
import json
import os
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


def generate_report(
    results: List[Dict[str, Any]],
    report_dir: str,
    title: str
) -> str:
    """
    Generate a reconciliation report.
    
    Args:
        results: List of reconciliation results
        report_dir: Directory to save the report
        title: Report title
        
    Returns:
        Path to the report file
    """
    # Create report directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    # Create report file
    report_file = os.path.join(report_dir, f"{title.lower().replace(' ', '_')}.html")
    
    # Create HTML report
    with open(report_file, "w") as f:
        # Write header
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .summary {{
                    margin-bottom: 20px;
                    padding: 10px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                }}
                .issues {{
                    margin-bottom: 20px;
                }}
                .error {{
                    color: red;
                }}
                .warning {{
                    color: orange;
                }}
                .info {{
                    color: blue;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Report generated on {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        """)
        
        # Write summary
        f.write("""
            <h2>Summary</h2>
            <div class="summary">
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Results</td>
                        <td>{}</td>
                    </tr>
                    <tr>
                        <td>Total Records</td>
                        <td>{}</td>
                    </tr>
                    <tr>
                        <td>Matched Records</td>
                        <td>{}</td>
                    </tr>
                    <tr>
                        <td>Match Percentage</td>
                        <td>{:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Total Issues</td>
                        <td>{}</td>
                    </tr>
                </table>
            </div>
        """.format(
            len(results),
            sum(result.get("total_records", 0) for result in results),
            sum(result.get("matched_records", 0) for result in results),
            sum(result.get("matched_records", 0)) / sum(result.get("total_records", 0)) * 100 if sum(result.get("total_records", 0)) > 0 else 0,
            sum(len(result.get("issues", [])) for result in results)
        ))
        
        # Write results
        f.write("""
            <h2>Results</h2>
            <table>
                <tr>
                    <th>Result ID</th>
                    <th>Status</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Total Records</th>
                    <th>Matched Records</th>
                    <th>Match Percentage</th>
                    <th>Issues</th>
                </tr>
        """)
        
        for result in results:
            total_records = result.get("total_records", 0)
            matched_records = result.get("matched_records", 0)
            match_percentage = matched_records / total_records * 100 if total_records > 0 else 0
            
            f.write(f"""
                <tr>
                    <td>{result.get("result_id", "")}</td>
                    <td>{result.get("status", "")}</td>
                    <td>{result.get("start_time", "")}</td>
                    <td>{result.get("end_time", "")}</td>
                    <td>{total_records}</td>
                    <td>{matched_records}</td>
                    <td>{match_percentage:.2f}%</td>
                    <td>{len(result.get("issues", []))}</td>
                </tr>
            """)
        
        f.write("</table>")
        
        # Write issues
        f.write("<h2>Issues</h2>")
        
        for i, result in enumerate(results):
            issues = result.get("issues", [])
            
            if not issues:
                continue
            
            f.write(f"""
                <h3>Result {i+1}: {result.get("result_id", "")}</h3>
                <div class="issues">
                    <table>
                        <tr>
                            <th>Issue ID</th>
                            <th>Field</th>
                            <th>Severity</th>
                            <th>Description</th>
                        </tr>
            """)
            
            for issue in issues[:100]:  # Limit to 100 issues per result
                severity = issue.get("severity", "")
                severity_class = "error" if severity == "ERROR" else "warning" if severity == "WARNING" else "info"
                
                f.write(f"""
                    <tr>
                        <td>{issue.get("issue_id", "")}</td>
                        <td>{issue.get("field", "")}</td>
                        <td class="{severity_class}">{severity}</td>
                        <td>{issue.get("description", "")}</td>
                    </tr>
                """)
            
            if len(issues) > 100:
                f.write(f"""
                    <tr>
                        <td colspan="4">... and {len(issues) - 100} more issues</td>
                    </tr>
                """)
            
            f.write("</table></div>")
        
        # Write footer
        f.write("""
        </body>
        </html>
        """)
    
    return report_file


async def main() -> None:
    """Main entry point."""
    logger.info("Starting reconciliation report example")
    
    # Create data reconciliation configuration
    logger.info("Creating data reconciliation configuration")
    
    # Define primary source (OHLCV data from provider 1)
    primary_source = {
        "source_id": "provider1",
        "source_type": "ohlcv",
        "query_params": {
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
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
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
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
                "tolerance": 0.0001  # 0.01% tolerance
            },
            "severity": "ERROR"
        },
        {
            "rule_id": "volume",
            "name": "Volume",
            "field": "volume",
            "comparison_type": "tolerance",
            "parameters": {
                "tolerance": 1.0  # 1 unit tolerance
            },
            "severity": "WARNING"
        }
    ]
    
    config_id = await create_config(
        name="OHLCV Data Reconciliation",
        reconciliation_type="cross_source",
        primary_source=primary_source,
        secondary_source=secondary_source,
        rules=rules,
        description="Reconcile OHLCV data between two providers",
        enabled=True
    )
    
    logger.info(f"Created data reconciliation configuration: {config_id}")
    
    # Run multiple reconciliation tasks
    logger.info("Running multiple reconciliation tasks")
    result_ids = []
    
    # Run tasks for different symbols
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    for symbol in symbols:
        # Update primary source
        primary_source["query_params"]["symbols"] = [symbol]
        
        # Update secondary source
        secondary_source["query_params"]["symbols"] = [symbol]
        
        # Schedule task
        task_id = await schedule_task(
            config_id=config_id,
            scheduled_time=datetime.datetime.utcnow(),
            metadata={"symbol": symbol}
        )
        
        logger.info(f"Scheduled task {task_id} for {symbol}")
        
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
    
    # Generate report
    logger.info("Generating report")
    report_file = generate_report(
        results=results,
        report_dir="reports",
        title="OHLCV Data Reconciliation Report"
    )
    
    logger.info(f"Generated report: {report_file}")
    
    logger.info("Reconciliation report example complete")


if __name__ == "__main__":
    asyncio.run(main())
