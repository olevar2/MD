#!/usr/bin/env python
"""
Initialize the database with sample data.

This script populates the database with sample historical data for testing purposes.
"""

import asyncio
import datetime
import logging
import random
from typing import List

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API URL
API_URL = "http://localhost:8000"

# Sample data
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


async def generate_ohlcv_data(
    symbol: str,
    timeframe: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime
) -> List[dict]:
    """
    Generate sample OHLCV data for a symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date
        end_date: End date
        
    Returns:
        List of OHLCV data records
    """
    # Determine time delta based on timeframe
    if timeframe == "1m":
        delta = datetime.timedelta(minutes=1)
    elif timeframe == "5m":
        delta = datetime.timedelta(minutes=5)
    elif timeframe == "15m":
        delta = datetime.timedelta(minutes=15)
    elif timeframe == "30m":
        delta = datetime.timedelta(minutes=30)
    elif timeframe == "1h":
        delta = datetime.timedelta(hours=1)
    elif timeframe == "4h":
        delta = datetime.timedelta(hours=4)
    elif timeframe == "1d":
        delta = datetime.timedelta(days=1)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Generate data
    data = []
    current_date = start_date
    
    # Initial price
    if symbol == "EURUSD":
        price = 1.1200
    elif symbol == "GBPUSD":
        price = 1.3000
    elif symbol == "USDJPY":
        price = 110.00
    elif symbol == "AUDUSD":
        price = 0.7500
    elif symbol == "USDCAD":
        price = 1.2500
    else:
        price = 1.0000
    
    while current_date <= end_date:
        # Generate random price movement
        change = random.uniform(-0.002, 0.002)
        price = price * (1 + change)
        
        # Generate OHLCV data
        open_price = price
        high_price = price * (1 + random.uniform(0, 0.001))
        low_price = price * (1 - random.uniform(0, 0.001))
        close_price = price * (1 + random.uniform(-0.0005, 0.0005))
        volume = random.uniform(100, 1000)
        
        # Create record
        record = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": current_date.isoformat(),
            "open_price": open_price,
            "high_price": high_price,
            "low_price": low_price,
            "close_price": close_price,
            "volume": volume,
            "source_id": "sample_data",
            "metadata": {
                "generated": True,
                "generator": "init_sample_data.py"
            },
            "created_by": "system"
        }
        
        data.append(record)
        
        # Move to next time period
        current_date += delta
    
    return data


async def generate_tick_data(
    symbol: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    num_ticks: int = 1000
) -> List[dict]:
    """
    Generate sample tick data for a symbol.
    
    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        num_ticks: Number of ticks to generate
        
    Returns:
        List of tick data records
    """
    # Generate data
    data = []
    
    # Initial price
    if symbol == "EURUSD":
        price = 1.1200
    elif symbol == "GBPUSD":
        price = 1.3000
    elif symbol == "USDJPY":
        price = 110.00
    elif symbol == "AUDUSD":
        price = 0.7500
    elif symbol == "USDCAD":
        price = 1.2500
    else:
        price = 1.0000
    
    # Generate ticks
    for _ in range(num_ticks):
        # Generate random timestamp
        timestamp = start_date + datetime.timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Generate random price movement
        change = random.uniform(-0.0001, 0.0001)
        price = price * (1 + change)
        
        # Generate bid/ask
        spread = random.uniform(0.0001, 0.0003)
        bid = price - spread / 2
        ask = price + spread / 2
        
        # Generate volumes
        bid_volume = random.uniform(10, 100)
        ask_volume = random.uniform(10, 100)
        
        # Create record
        record = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "bid": bid,
            "ask": ask,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "source_id": "sample_data",
            "metadata": {
                "generated": True,
                "generator": "init_sample_data.py"
            },
            "created_by": "system"
        }
        
        data.append(record)
    
    # Sort by timestamp
    data.sort(key=lambda x: x["timestamp"])
    
    return data


async def upload_ohlcv_data(data: List[dict]) -> None:
    """
    Upload OHLCV data to the API.
    
    Args:
        data: List of OHLCV data records
    """
    async with httpx.AsyncClient() as client:
        for record in data:
            try:
                response = await client.post(f"{API_URL}/historical/ohlcv", json=record)
                response.raise_for_status()
                logger.info(f"Uploaded OHLCV data for {record['symbol']} {record['timeframe']} at {record['timestamp']}")
            except Exception as e:
                logger.error(f"Failed to upload OHLCV data: {e}")


async def upload_tick_data(data: List[dict]) -> None:
    """
    Upload tick data to the API.
    
    Args:
        data: List of tick data records
    """
    async with httpx.AsyncClient() as client:
        for record in data:
            try:
                response = await client.post(f"{API_URL}/historical/tick", json=record)
                response.raise_for_status()
                logger.info(f"Uploaded tick data for {record['symbol']} at {record['timestamp']}")
            except Exception as e:
                logger.error(f"Failed to upload tick data: {e}")


async def main() -> None:
    """Main entry point."""
    logger.info("Initializing sample data")
    
    # Set date range
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=30)
    
    # Generate and upload OHLCV data
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            logger.info(f"Generating OHLCV data for {symbol} {timeframe}")
            ohlcv_data = await generate_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            logger.info(f"Uploading {len(ohlcv_data)} OHLCV records for {symbol} {timeframe}")
            await upload_ohlcv_data(ohlcv_data)
    
    # Generate and upload tick data
    for symbol in SYMBOLS:
        logger.info(f"Generating tick data for {symbol}")
        tick_data = await generate_tick_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            num_ticks=1000
        )
        
        logger.info(f"Uploading {len(tick_data)} tick records for {symbol}")
        await upload_tick_data(tick_data)
    
    logger.info("Sample data initialization complete")


if __name__ == "__main__":
    asyncio.run(main())
