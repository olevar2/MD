#!/usr/bin/env python
"""
Example script demonstrating how to use the Historical Data Management service for data corrections.

This script shows how to create and track corrections to historical data.
"""

import asyncio
import datetime
import logging
from typing import Dict, List, Any, Optional

import httpx
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API URL
API_URL = "http://localhost:8000"


async def get_ohlcv_data(
    symbol: str,
    timeframe: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    include_corrections: bool = True
) -> pd.DataFrame:
    """
    Get OHLCV data.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date
        end_date: End date
        include_corrections: Whether to include corrections
        
    Returns:
        DataFrame with OHLCV data
    """
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start_timestamp": start_date.isoformat(),
        "end_timestamp": end_date.isoformat(),
        "include_corrections": include_corrections,
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


async def create_correction(
    record_id: str,
    correction_data: Dict[str, Any],
    correction_type: str,
    correction_reason: str,
    source_type: str
) -> Dict[str, str]:
    """
    Create a correction for a record.
    
    Args:
        record_id: ID of the record to correct
        correction_data: Corrected data
        correction_type: Type of correction
        correction_reason: Reason for correction
        source_type: Type of data source
        
    Returns:
        Dictionary with correction IDs
    """
    data = {
        "original_record_id": record_id,
        "correction_data": correction_data,
        "correction_type": correction_type,
        "correction_reason": correction_reason,
        "corrected_by": "correction_example.py",
        "source_type": source_type
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/historical/correction", json=data)
        response.raise_for_status()
        
        return response.json()


async def get_record_history(
    record_id: str,
    source_type: str
) -> List[Dict[str, Any]]:
    """
    Get the history of a record, including all corrections.
    
    Args:
        record_id: ID of the record
        source_type: Type of data source
        
    Returns:
        List of records in chronological order
    """
    params = {
        "source_type": source_type
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/historical/record-history/{record_id}", params=params)
        response.raise_for_status()
        
        return response.json()


def find_outliers(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Find outliers in a DataFrame column using z-score.
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        threshold: Z-score threshold
        
    Returns:
        DataFrame with outliers
    """
    # Calculate z-score
    z_score = (df[column] - df[column].mean()) / df[column].std()
    
    # Find outliers
    outliers = df[abs(z_score) > threshold].copy()
    
    return outliers


def plot_data_with_corrections(
    original_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    outliers: pd.DataFrame,
    symbol: str,
    timeframe: str
) -> None:
    """
    Plot original and corrected data.
    
    Args:
        original_df: DataFrame with original data
        corrected_df: DataFrame with corrected data
        outliers: DataFrame with outliers
        symbol: Trading symbol
        timeframe: Timeframe
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original data
    ax.plot(
        original_df["timestamp"],
        original_df["close"],
        label="Original Data",
        color="blue",
        alpha=0.7
    )
    
    # Plot corrected data
    ax.plot(
        corrected_df["timestamp"],
        corrected_df["close"],
        label="Corrected Data",
        color="green",
        linestyle="--"
    )
    
    # Highlight outliers
    ax.scatter(
        outliers["timestamp"],
        outliers["close"],
        color="red",
        marker="x",
        s=100,
        label="Outliers (Corrected)"
    )
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"Original vs. Corrected Data - {symbol} {timeframe}")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"corrections_{symbol}_{timeframe}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting correction example")
    
    # Set parameters
    symbol = "EURUSD"
    timeframe = "1h"
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=7)
    
    # Get original data
    logger.info(f"Getting original data for {symbol} {timeframe}")
    original_df = await get_ohlcv_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        include_corrections=False
    )
    
    if original_df.empty:
        logger.error(f"No data found for {symbol} {timeframe}")
        return
    
    logger.info(f"Retrieved {len(original_df)} records")
    
    # Find outliers
    logger.info("Finding outliers")
    outliers = find_outliers(original_df, "close", threshold=2.0)
    
    if outliers.empty:
        logger.warning("No outliers found, creating artificial outliers for demonstration")
        
        # Create artificial outliers for demonstration
        outlier_indices = [len(original_df) // 4, len(original_df) // 2, 3 * len(original_df) // 4]
        outliers = original_df.iloc[outlier_indices].copy()
        
        # Modify close price to create outliers
        outliers["close"] = outliers["close"] * 1.05  # 5% increase
    
    logger.info(f"Found {len(outliers)} outliers")
    
    # Create corrections
    logger.info("Creating corrections")
    for _, outlier in outliers.iterrows():
        record_id = outlier["record_id"]
        
        # Calculate corrected price (average of previous and next)
        timestamp = outlier["timestamp"]
        prev_record = original_df[original_df["timestamp"] < timestamp].iloc[-1] if not original_df[original_df["timestamp"] < timestamp].empty else None
        next_record = original_df[original_df["timestamp"] > timestamp].iloc[0] if not original_df[original_df["timestamp"] > timestamp].empty else None
        
        if prev_record is not None and next_record is not None:
            corrected_close = (prev_record["close"] + next_record["close"]) / 2
        else:
            # If no previous or next record, use a simple adjustment
            corrected_close = outlier["close"] * 0.95  # Reduce by 5%
        
        # Create correction
        correction_data = {
            "data": {
                "close": corrected_close
            }
        }
        
        correction_result = await create_correction(
            record_id=record_id,
            correction_data=correction_data,
            correction_type="AUTOMATED_CORRECTION",
            correction_reason="Outlier detected and corrected",
            source_type="OHLCV"
        )
        
        logger.info(f"Created correction: {correction_result}")
        
        # Get record history
        history = await get_record_history(record_id=record_id, source_type="OHLCV")
        logger.info(f"Record history: {len(history)} versions")
    
    # Get corrected data
    logger.info("Getting corrected data")
    corrected_df = await get_ohlcv_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        include_corrections=True
    )
    
    # Plot results
    logger.info("Plotting results")
    plot_data_with_corrections(
        original_df=original_df,
        corrected_df=corrected_df,
        outliers=outliers,
        symbol=symbol,
        timeframe=timeframe
    )
    
    logger.info("Correction example complete")


if __name__ == "__main__":
    asyncio.run(main())
