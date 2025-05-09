"""
Example of using the Correlation Analysis client

This module demonstrates how to use the standardized Correlation Analysis client
to interact with the Analysis Engine Service API.
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime

from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Example currency pair data
EXAMPLE_DATA = {
    "EUR/USD": {
        "ohlc": [
            {"timestamp": "2025-04-01T00:00:00", "open": 1.0765, "high": 1.0780, "low": 1.0760, "close": 1.0775, "volume": 1000},
            {"timestamp": "2025-04-01T01:00:00", "open": 1.0775, "high": 1.0790, "low": 1.0770, "close": 1.0785, "volume": 1200},
            {"timestamp": "2025-04-01T02:00:00", "open": 1.0785, "high": 1.0795, "low": 1.0780, "close": 1.0790, "volume": 1100},
            {"timestamp": "2025-04-01T03:00:00", "open": 1.0790, "high": 1.0800, "low": 1.0785, "close": 1.0795, "volume": 1300},
            {"timestamp": "2025-04-01T04:00:00", "open": 1.0795, "high": 1.0810, "low": 1.0790, "close": 1.0805, "volume": 1400}
        ],
        "metadata": {"timeframe": "1h"}
    },
    "GBP/USD": {
        "ohlc": [
            {"timestamp": "2025-04-01T00:00:00", "open": 1.2765, "high": 1.2780, "low": 1.2760, "close": 1.2775, "volume": 800},
            {"timestamp": "2025-04-01T01:00:00", "open": 1.2775, "high": 1.2790, "low": 1.2770, "close": 1.2785, "volume": 900},
            {"timestamp": "2025-04-01T02:00:00", "open": 1.2785, "high": 1.2795, "low": 1.2780, "close": 1.2790, "volume": 850},
            {"timestamp": "2025-04-01T03:00:00", "open": 1.2790, "high": 1.2800, "low": 1.2785, "close": 1.2795, "volume": 950},
            {"timestamp": "2025-04-01T04:00:00", "open": 1.2795, "high": 1.2810, "low": 1.2790, "close": 1.2805, "volume": 1000}
        ],
        "metadata": {"timeframe": "1h"}
    },
    "USD/JPY": {
        "ohlc": [
            {"timestamp": "2025-04-01T00:00:00", "open": 110.65, "high": 110.80, "low": 110.60, "close": 110.75, "volume": 1200},
            {"timestamp": "2025-04-01T01:00:00", "open": 110.75, "high": 110.90, "low": 110.70, "close": 110.85, "volume": 1300},
            {"timestamp": "2025-04-01T02:00:00", "open": 110.85, "high": 110.95, "low": 110.80, "close": 110.90, "volume": 1250},
            {"timestamp": "2025-04-01T03:00:00", "open": 110.90, "high": 111.00, "low": 110.85, "close": 110.95, "volume": 1350},
            {"timestamp": "2025-04-01T04:00:00", "open": 110.95, "high": 111.10, "low": 110.90, "close": 111.05, "volume": 1400}
        ],
        "metadata": {"timeframe": "1h"}
    }
}

async def analyze_currency_correlations_example():
    """
    Example of analyzing currency correlations using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the correlation analysis client
    client = factory.get_correlation_analysis_client()
    
    # Example parameters
    window_sizes = [5, 20, 60]
    correlation_method = "pearson"
    significance_threshold = 0.7
    
    try:
        # Analyze currency correlations
        result = await client.analyze_currency_correlations(
            data=EXAMPLE_DATA,
            window_sizes=window_sizes,
            correlation_method=correlation_method,
            significance_threshold=significance_threshold
        )
        
        logger.info(f"Analyzed currency correlations: {result}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing currency correlations: {str(e)}")
        raise

async def analyze_lead_lag_relationships_example():
    """
    Example of analyzing lead-lag relationships using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the correlation analysis client
    client = factory.get_correlation_analysis_client()
    
    # Example parameters
    max_lag = 10
    significance = 0.05
    
    try:
        # Analyze lead-lag relationships
        result = await client.analyze_lead_lag_relationships(
            data=EXAMPLE_DATA,
            max_lag=max_lag,
            significance=significance
        )
        
        logger.info(f"Analyzed lead-lag relationships: {result}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing lead-lag relationships: {str(e)}")
        raise

async def detect_correlation_breakdowns_example():
    """
    Example of detecting correlation breakdowns using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the correlation analysis client
    client = factory.get_correlation_analysis_client()
    
    # Example parameters
    short_window = 5
    long_window = 60
    change_threshold = 0.3
    
    try:
        # Detect correlation breakdowns
        result = await client.detect_correlation_breakdowns(
            data=EXAMPLE_DATA,
            short_window=short_window,
            long_window=long_window,
            change_threshold=change_threshold
        )
        
        logger.info(f"Detected correlation breakdowns: {result}")
        return result
    except Exception as e:
        logger.error(f"Error detecting correlation breakdowns: {str(e)}")
        raise

async def test_pair_cointegration_example():
    """
    Example of testing pair cointegration using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the correlation analysis client
    client = factory.get_correlation_analysis_client()
    
    # Example parameters
    significance = 0.05
    
    try:
        # Test pair cointegration
        result = await client.test_pair_cointegration(
            data=EXAMPLE_DATA,
            significance=significance
        )
        
        logger.info(f"Tested pair cointegration: {result}")
        return result
    except Exception as e:
        logger.error(f"Error testing pair cointegration: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running Correlation Analysis client examples")
    
    # Run the examples
    await analyze_currency_correlations_example()
    await analyze_lead_lag_relationships_example()
    await detect_correlation_breakdowns_example()
    await test_pair_cointegration_example()
    
    logger.info("Completed Correlation Analysis client examples")

if __name__ == "__main__":
    asyncio.run(main())
