#!/usr/bin/env python
"""
Example script demonstrating how to use the Historical Data Management service for data quality reporting.

This script shows how to generate and analyze data quality reports.
"""

import asyncio
import datetime
import logging
from typing import Dict, List, Any

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


async def generate_quality_report(
    symbol: str,
    source_type: str,
    timeframe: str = None,
    start_date: datetime.datetime = None,
    end_date: datetime.datetime = None
) -> str:
    """
    Generate a data quality report.
    
    Args:
        symbol: Trading symbol
        source_type: Type of data source
        timeframe: Timeframe (for OHLCV data)
        start_date: Start date
        end_date: End date
        
    Returns:
        Report ID
    """
    data = {
        "symbol": symbol,
        "source_type": source_type,
        "timeframe": timeframe
    }
    
    if start_date:
        data["start_timestamp"] = start_date.isoformat()
    
    if end_date:
        data["end_timestamp"] = end_date.isoformat()
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/historical/quality-report", json=data)
        response.raise_for_status()
        
        return response.json()["report_id"]


async def get_quality_reports(
    symbol: str = None,
    source_type: str = None,
    timeframe: str = None,
    start_date: datetime.datetime = None,
    end_date: datetime.datetime = None
) -> List[Dict[str, Any]]:
    """
    Get data quality reports.
    
    Args:
        symbol: Symbol filter
        source_type: Source type filter
        timeframe: Timeframe filter
        start_date: Start date filter
        end_date: End date filter
        
    Returns:
        List of quality report records
    """
    params = {}
    
    if symbol:
        params["symbol"] = symbol
    
    if source_type:
        params["source_type"] = source_type
    
    if timeframe:
        params["timeframe"] = timeframe
    
    if start_date:
        params["start_timestamp"] = start_date.isoformat()
    
    if end_date:
        params["end_timestamp"] = end_date.isoformat()
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/historical/quality-reports", params=params)
        response.raise_for_status()
        
        return response.json()


def plot_quality_metrics(reports: List[Dict[str, Any]], symbol: str) -> None:
    """
    Plot data quality metrics.
    
    Args:
        reports: List of quality report records
        symbol: Trading symbol
    """
    # Convert to DataFrame
    df = pd.DataFrame(reports)
    
    # Convert timestamps to datetime
    df["report_timestamp"] = pd.to_datetime(df["report_timestamp"])
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"])
    
    # Sort by timeframe and report timestamp
    df = df.sort_values(["timeframe", "report_timestamp"])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot total records and missing records
    timeframes = df["timeframe"].unique()
    
    for i, timeframe in enumerate(timeframes):
        timeframe_df = df[df["timeframe"] == timeframe]
        
        ax1.bar(
            [i - 0.2 + 0.4 * j / len(timeframe_df) for j in range(len(timeframe_df))],
            timeframe_df["total_records"],
            width=0.4 / len(timeframe_df),
            label=f"{timeframe} - Total" if i == 0 else None,
            alpha=0.7
        )
        
        ax1.bar(
            [i + 0.4 * j / len(timeframe_df) for j in range(len(timeframe_df))],
            timeframe_df["missing_records"],
            width=0.4 / len(timeframe_df),
            label=f"{timeframe} - Missing" if i == 0 else None,
            alpha=0.7,
            color="red"
        )
    
    ax1.set_xticks(range(len(timeframes)))
    ax1.set_xticklabels(timeframes)
    ax1.set_xlabel("Timeframe")
    ax1.set_ylabel("Number of Records")
    ax1.set_title(f"Data Completeness - {symbol}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot data quality over time
    for timeframe in timeframes:
        timeframe_df = df[df["timeframe"] == timeframe]
        
        if len(timeframe_df) > 1:
            # Calculate completeness ratio
            timeframe_df["completeness"] = 1 - (timeframe_df["missing_records"] / timeframe_df["total_records"])
            
            ax2.plot(
                timeframe_df["report_timestamp"],
                timeframe_df["completeness"],
                marker="o",
                label=timeframe
            )
    
    ax2.set_xlabel("Report Timestamp")
    ax2.set_ylabel("Data Completeness (1 - missing/total)")
    ax2.set_title(f"Data Quality Over Time - {symbol}")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"quality_report_{symbol}.png")
    plt.show()


def plot_quality_issues(reports: List[Dict[str, Any]], symbol: str) -> None:
    """
    Plot data quality issues.
    
    Args:
        reports: List of quality report records
        symbol: Trading symbol
    """
    # Extract quality issues
    all_issues = []
    
    for report in reports:
        for issue in report.get("quality_issues", []):
            issue_data = {
                "symbol": report["symbol"],
                "timeframe": report["timeframe"],
                "type": issue["type"],
                "timestamp": issue.get("timestamp"),
                "description": issue.get("description")
            }
            all_issues.append(issue_data)
    
    if not all_issues:
        logger.warning("No quality issues found")
        return
    
    # Convert to DataFrame
    issues_df = pd.DataFrame(all_issues)
    
    # Convert timestamp to datetime
    if "timestamp" in issues_df.columns:
        issues_df["timestamp"] = pd.to_datetime(issues_df["timestamp"])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot issue types
    issue_counts = issues_df["type"].value_counts()
    issue_counts.plot(kind="bar", ax=ax1)
    ax1.set_title(f"Quality Issue Types - {symbol}")
    ax1.set_xlabel("Issue Type")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # Plot issues by timeframe
    timeframe_issue_counts = issues_df.groupby(["timeframe", "type"]).size().unstack()
    timeframe_issue_counts.plot(kind="bar", stacked=True, ax=ax2)
    ax2.set_title(f"Quality Issues by Timeframe - {symbol}")
    ax2.set_xlabel("Timeframe")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"quality_issues_{symbol}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting quality report example")
    
    # Set parameters
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=30)
    
    # Generate quality reports
    for symbol in symbols:
        logger.info(f"Generating quality reports for {symbol}")
        
        for timeframe in timeframes:
            logger.info(f"Generating quality report for {symbol} {timeframe}")
            
            report_id = await generate_quality_report(
                symbol=symbol,
                source_type="OHLCV",
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            logger.info(f"Generated report {report_id}")
    
    # Get quality reports
    for symbol in symbols:
        logger.info(f"Getting quality reports for {symbol}")
        
        reports = await get_quality_reports(
            symbol=symbol,
            source_type="OHLCV"
        )
        
        if not reports:
            logger.warning(f"No reports found for {symbol}")
            continue
        
        logger.info(f"Retrieved {len(reports)} reports")
        
        # Plot quality metrics
        logger.info(f"Plotting quality metrics for {symbol}")
        plot_quality_metrics(reports, symbol)
        
        # Plot quality issues
        logger.info(f"Plotting quality issues for {symbol}")
        plot_quality_issues(reports, symbol)
    
    logger.info("Quality report example complete")


if __name__ == "__main__":
    asyncio.run(main())
