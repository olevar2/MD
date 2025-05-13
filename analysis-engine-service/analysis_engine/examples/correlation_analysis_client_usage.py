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
EXAMPLE_DATA = {'EUR/USD': {'ohlc': [{'timestamp': '2025-04-01T00:00:00',
    'open': 1.0765, 'high': 1.078, 'low': 1.076, 'close': 1.0775, 'volume':
    1000}, {'timestamp': '2025-04-01T01:00:00', 'open': 1.0775, 'high': 
    1.079, 'low': 1.077, 'close': 1.0785, 'volume': 1200}, {'timestamp':
    '2025-04-01T02:00:00', 'open': 1.0785, 'high': 1.0795, 'low': 1.078,
    'close': 1.079, 'volume': 1100}, {'timestamp': '2025-04-01T03:00:00',
    'open': 1.079, 'high': 1.08, 'low': 1.0785, 'close': 1.0795, 'volume': 
    1300}, {'timestamp': '2025-04-01T04:00:00', 'open': 1.0795, 'high': 
    1.081, 'low': 1.079, 'close': 1.0805, 'volume': 1400}], 'metadata': {
    'timeframe': '1h'}}, 'GBP/USD': {'ohlc': [{'timestamp':
    '2025-04-01T00:00:00', 'open': 1.2765, 'high': 1.278, 'low': 1.276,
    'close': 1.2775, 'volume': 800}, {'timestamp': '2025-04-01T01:00:00',
    'open': 1.2775, 'high': 1.279, 'low': 1.277, 'close': 1.2785, 'volume':
    900}, {'timestamp': '2025-04-01T02:00:00', 'open': 1.2785, 'high': 
    1.2795, 'low': 1.278, 'close': 1.279, 'volume': 850}, {'timestamp':
    '2025-04-01T03:00:00', 'open': 1.279, 'high': 1.28, 'low': 1.2785,
    'close': 1.2795, 'volume': 950}, {'timestamp': '2025-04-01T04:00:00',
    'open': 1.2795, 'high': 1.281, 'low': 1.279, 'close': 1.2805, 'volume':
    1000}], 'metadata': {'timeframe': '1h'}}, 'USD/JPY': {'ohlc': [{
    'timestamp': '2025-04-01T00:00:00', 'open': 110.65, 'high': 110.8,
    'low': 110.6, 'close': 110.75, 'volume': 1200}, {'timestamp':
    '2025-04-01T01:00:00', 'open': 110.75, 'high': 110.9, 'low': 110.7,
    'close': 110.85, 'volume': 1300}, {'timestamp': '2025-04-01T02:00:00',
    'open': 110.85, 'high': 110.95, 'low': 110.8, 'close': 110.9, 'volume':
    1250}, {'timestamp': '2025-04-01T03:00:00', 'open': 110.9, 'high': 
    111.0, 'low': 110.85, 'close': 110.95, 'volume': 1350}, {'timestamp':
    '2025-04-01T04:00:00', 'open': 110.95, 'high': 111.1, 'low': 110.9,
    'close': 111.05, 'volume': 1400}], 'metadata': {'timeframe': '1h'}}}


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def analyze_currency_correlations_example():
    """
    Example of analyzing currency correlations using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_correlation_analysis_client()
    window_sizes = [5, 20, 60]
    correlation_method = 'pearson'
    significance_threshold = 0.7
    try:
        result = await client.analyze_currency_correlations(data=
            EXAMPLE_DATA, window_sizes=window_sizes, correlation_method=
            correlation_method, significance_threshold=significance_threshold)
        logger.info(f'Analyzed currency correlations: {result}')
        return result
    except Exception as e:
        logger.error(f'Error analyzing currency correlations: {str(e)}')
        raise


@async_with_exception_handling
async def analyze_lead_lag_relationships_example():
    """
    Example of analyzing lead-lag relationships using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_correlation_analysis_client()
    max_lag = 10
    significance = 0.05
    try:
        result = await client.analyze_lead_lag_relationships(data=
            EXAMPLE_DATA, max_lag=max_lag, significance=significance)
        logger.info(f'Analyzed lead-lag relationships: {result}')
        return result
    except Exception as e:
        logger.error(f'Error analyzing lead-lag relationships: {str(e)}')
        raise


@async_with_exception_handling
async def detect_correlation_breakdowns_example():
    """
    Example of detecting correlation breakdowns using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_correlation_analysis_client()
    short_window = 5
    long_window = 60
    change_threshold = 0.3
    try:
        result = await client.detect_correlation_breakdowns(data=
            EXAMPLE_DATA, short_window=short_window, long_window=
            long_window, change_threshold=change_threshold)
        logger.info(f'Detected correlation breakdowns: {result}')
        return result
    except Exception as e:
        logger.error(f'Error detecting correlation breakdowns: {str(e)}')
        raise


@async_with_exception_handling
async def test_pair_cointegration_example():
    """
    Example of testing pair cointegration using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_correlation_analysis_client()
    significance = 0.05
    try:
        result = await client.test_pair_cointegration(data=EXAMPLE_DATA,
            significance=significance)
        logger.info(f'Tested pair cointegration: {result}')
        return result
    except Exception as e:
        logger.error(f'Error testing pair cointegration: {str(e)}')
        raise


async def main():
    """
    Run the examples.
    """
    logger.info('Running Correlation Analysis client examples')
    await analyze_currency_correlations_example()
    await analyze_lead_lag_relationships_example()
    await detect_correlation_breakdowns_example()
    await test_pair_cointegration_example()
    logger.info('Completed Correlation Analysis client examples')


if __name__ == '__main__':
    asyncio.run(main())
