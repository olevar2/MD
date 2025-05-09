"""
Example of using the Manipulation Detection client

This module demonstrates how to use the standardized Manipulation Detection client
to interact with the Analysis Engine Service API.
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime

from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Example OHLCV data
EXAMPLE_OHLCV = [
    {"timestamp": "2025-04-01T00:00:00", "open": 1.0765, "high": 1.0780, "low": 1.0760, "close": 1.0775, "volume": 1000},
    {"timestamp": "2025-04-01T01:00:00", "open": 1.0775, "high": 1.0790, "low": 1.0770, "close": 1.0785, "volume": 1200},
    {"timestamp": "2025-04-01T02:00:00", "open": 1.0785, "high": 1.0795, "low": 1.0780, "close": 1.0790, "volume": 1100},
    {"timestamp": "2025-04-01T03:00:00", "open": 1.0790, "high": 1.0800, "low": 1.0785, "close": 1.0795, "volume": 1300},
    {"timestamp": "2025-04-01T04:00:00", "open": 1.0795, "high": 1.0810, "low": 1.0790, "close": 1.0805, "volume": 1400},
    {"timestamp": "2025-04-01T05:00:00", "open": 1.0805, "high": 1.0820, "low": 1.0800, "close": 1.0815, "volume": 1500},
    {"timestamp": "2025-04-01T06:00:00", "open": 1.0815, "high": 1.0830, "low": 1.0810, "close": 1.0825, "volume": 1600},
    {"timestamp": "2025-04-01T07:00:00", "open": 1.0825, "high": 1.0840, "low": 1.0820, "close": 1.0835, "volume": 1700},
    {"timestamp": "2025-04-01T08:00:00", "open": 1.0835, "high": 1.0850, "low": 1.0830, "close": 1.0845, "volume": 1800},
    {"timestamp": "2025-04-01T09:00:00", "open": 1.0845, "high": 1.0860, "low": 1.0840, "close": 1.0855, "volume": 1900},
    {"timestamp": "2025-04-01T10:00:00", "open": 1.0855, "high": 1.0870, "low": 1.0850, "close": 1.0865, "volume": 2000},
    {"timestamp": "2025-04-01T11:00:00", "open": 1.0865, "high": 1.0880, "low": 1.0860, "close": 1.0875, "volume": 2100},
    {"timestamp": "2025-04-01T12:00:00", "open": 1.0875, "high": 1.0890, "low": 1.0870, "close": 1.0885, "volume": 2200},
    {"timestamp": "2025-04-01T13:00:00", "open": 1.0885, "high": 1.0900, "low": 1.0880, "close": 1.0895, "volume": 2300},
    {"timestamp": "2025-04-01T14:00:00", "open": 1.0895, "high": 1.0910, "low": 1.0890, "close": 1.0905, "volume": 2400},
    {"timestamp": "2025-04-01T15:00:00", "open": 1.0905, "high": 1.0920, "low": 1.0900, "close": 1.0915, "volume": 2500},
    {"timestamp": "2025-04-01T16:00:00", "open": 1.0915, "high": 1.0930, "low": 1.0910, "close": 1.0925, "volume": 2600},
    {"timestamp": "2025-04-01T17:00:00", "open": 1.0925, "high": 1.0940, "low": 1.0920, "close": 1.0935, "volume": 2700},
    {"timestamp": "2025-04-01T18:00:00", "open": 1.0935, "high": 1.0950, "low": 1.0930, "close": 1.0945, "volume": 2800},
    {"timestamp": "2025-04-01T19:00:00", "open": 1.0945, "high": 1.0960, "low": 1.0940, "close": 1.0955, "volume": 2900},
    {"timestamp": "2025-04-01T20:00:00", "open": 1.0955, "high": 1.0970, "low": 1.0950, "close": 1.0965, "volume": 3000},
    {"timestamp": "2025-04-01T21:00:00", "open": 1.0965, "high": 1.0980, "low": 1.0960, "close": 1.0975, "volume": 3100},
    {"timestamp": "2025-04-01T22:00:00", "open": 1.0975, "high": 1.0990, "low": 1.0970, "close": 1.0985, "volume": 3200},
    {"timestamp": "2025-04-01T23:00:00", "open": 1.0985, "high": 1.1000, "low": 1.0980, "close": 1.0995, "volume": 3300},
    # Add more data points to meet the minimum requirement of 100 data points
    # This is just a placeholder - in a real scenario, you would have actual market data
]

# Example metadata
EXAMPLE_METADATA = {
    "symbol": "EUR/USD",
    "timeframe": "1h"
}

# Generate more data points to meet the minimum requirement of 100 data points
for i in range(100):
    last_data = EXAMPLE_OHLCV[-1]
    new_data = {
        "timestamp": f"2025-04-02T{i % 24:02d}:00:00",
        "open": last_data["close"],
        "high": last_data["close"] + 0.0015,
        "low": last_data["close"] - 0.0010,
        "close": last_data["close"] + 0.0005,
        "volume": last_data["volume"] + 100
    }
    EXAMPLE_OHLCV.append(new_data)

async def detect_manipulation_patterns_example():
    """
    Example of detecting manipulation patterns using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the manipulation detection client
    client = factory.get_manipulation_detection_client()
    
    # Example parameters
    sensitivity = 1.2
    include_protection = True
    
    try:
        # Detect manipulation patterns
        result = await client.detect_manipulation_patterns(
            ohlcv=EXAMPLE_OHLCV,
            metadata=EXAMPLE_METADATA,
            sensitivity=sensitivity,
            include_protection=include_protection
        )
        
        logger.info(f"Detected manipulation patterns: {result}")
        return result
    except Exception as e:
        logger.error(f"Error detecting manipulation patterns: {str(e)}")
        raise

async def detect_stop_hunting_example():
    """
    Example of detecting stop hunting patterns using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the manipulation detection client
    client = factory.get_manipulation_detection_client()
    
    # Example parameters
    lookback = 30
    recovery_threshold = 0.5
    
    try:
        # Detect stop hunting patterns
        result = await client.detect_stop_hunting(
            ohlcv=EXAMPLE_OHLCV,
            metadata=EXAMPLE_METADATA,
            lookback=lookback,
            recovery_threshold=recovery_threshold
        )
        
        logger.info(f"Detected stop hunting patterns: {result}")
        return result
    except Exception as e:
        logger.error(f"Error detecting stop hunting patterns: {str(e)}")
        raise

async def detect_fake_breakouts_example():
    """
    Example of detecting fake breakout patterns using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the manipulation detection client
    client = factory.get_manipulation_detection_client()
    
    # Example parameters
    threshold = 0.7
    
    try:
        # Detect fake breakout patterns
        result = await client.detect_fake_breakouts(
            ohlcv=EXAMPLE_OHLCV,
            metadata=EXAMPLE_METADATA,
            threshold=threshold
        )
        
        logger.info(f"Detected fake breakout patterns: {result}")
        return result
    except Exception as e:
        logger.error(f"Error detecting fake breakout patterns: {str(e)}")
        raise

async def detect_volume_anomalies_example():
    """
    Example of detecting volume anomalies using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the manipulation detection client
    client = factory.get_manipulation_detection_client()
    
    # Example parameters
    z_threshold = 2.0
    
    try:
        # Detect volume anomalies
        result = await client.detect_volume_anomalies(
            ohlcv=EXAMPLE_OHLCV,
            metadata=EXAMPLE_METADATA,
            z_threshold=z_threshold
        )
        
        logger.info(f"Detected volume anomalies: {result}")
        return result
    except Exception as e:
        logger.error(f"Error detecting volume anomalies: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running Manipulation Detection client examples")
    
    # Run the examples
    await detect_manipulation_patterns_example()
    await detect_stop_hunting_example()
    await detect_fake_breakouts_example()
    await detect_volume_anomalies_example()
    
    logger.info("Completed Manipulation Detection client examples")

if __name__ == "__main__":
    asyncio.run(main())
