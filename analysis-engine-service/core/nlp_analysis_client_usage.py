"""
Example of using the NLP Analysis client

This module demonstrates how to use the standardized NLP Analysis client
to interact with the Analysis Engine Service API.
"""
import asyncio
from typing import Dict, List, Any
from datetime import datetime
from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger
logger = get_structured_logger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def analyze_news_example():
    """
    Example of analyzing news using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_nlp_analysis_client()
    news_items = [{'id': 'news_12345', 'title':
        'Federal Reserve Raises Interest Rates by 25 Basis Points',
        'content':
        'The Federal Reserve announced today that it will raise interest rates by 25 basis points, citing concerns about inflation and strong economic growth. The move was widely expected by market participants.'
        , 'source': 'Financial Times', 'timestamp': '2025-04-01T14:30:00Z'},
        {'id': 'news_12346', 'title':
        'ECB Signals Potential Rate Cut in June', 'content':
        'The European Central Bank has signaled that it may cut interest rates in June, as inflation in the eurozone continues to moderate and economic growth remains sluggish.'
        , 'source': 'Bloomberg', 'timestamp': '2025-04-01T10:15:00Z'}]
    try:
        result = await client.analyze_news(news_items)
        logger.info(f'Analyzed news: {result}')
        return result
    except Exception as e:
        logger.error(f'Error analyzing news: {str(e)}')
        raise


@async_with_exception_handling
async def analyze_economic_report_example():
    """
    Example of analyzing an economic report using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_nlp_analysis_client()
    report = {'title': 'Non-Farm Payroll Report - April 2025', 'content':
        'The U.S. economy added 200,000 jobs in April 2025, exceeding expectations of 180,000. The unemployment rate remained steady at 3.8%. Average hourly earnings increased by 0.3% month-over-month, in line with expectations.'
        , 'type': 'NFP', 'source': 'Bureau of Labor Statistics',
        'timestamp': '2025-05-01T12:00:00Z'}
    try:
        result = await client.analyze_economic_report(report)
        logger.info(f'Analyzed economic report: {result}')
        return result
    except Exception as e:
        logger.error(f'Error analyzing economic report: {str(e)}')
        raise


@async_with_exception_handling
async def get_combined_insights_example():
    """
    Example of getting combined insights using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_nlp_analysis_client()
    news_data = {'news_items': [{'id': 'news_12345', 'title':
        'Federal Reserve Raises Interest Rates by 25 Basis Points',
        'content':
        'The Federal Reserve announced today that it will raise interest rates by 25 basis points, citing concerns about inflation and strong economic growth. The move was widely expected by market participants.'
        , 'source': 'Financial Times', 'timestamp': '2025-04-01T14:30:00Z'}]}
    economic_reports = [{'title': 'Non-Farm Payroll Report - April 2025',
        'content':
        'The U.S. economy added 200,000 jobs in April 2025, exceeding expectations of 180,000. The unemployment rate remained steady at 3.8%. Average hourly earnings increased by 0.3% month-over-month, in line with expectations.'
        , 'type': 'NFP', 'source': 'Bureau of Labor Statistics',
        'timestamp': '2025-05-01T12:00:00Z'}]
    currency_pairs = ['EUR/USD', 'GBP/USD']
    try:
        result = await client.get_combined_insights(news_data=news_data,
            economic_reports=economic_reports, currency_pairs=currency_pairs)
        logger.info(f'Got combined insights: {result}')
        return result
    except Exception as e:
        logger.error(f'Error getting combined insights: {str(e)}')
        raise


@async_with_exception_handling
async def get_market_sentiment_example():
    """
    Example of getting market sentiment using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_nlp_analysis_client()
    currency_pair = 'EUR/USD'
    try:
        result = await client.get_market_sentiment(currency_pair)
        logger.info(f'Got market sentiment: {result}')
        return result
    except Exception as e:
        logger.error(f'Error getting market sentiment: {str(e)}')
        raise


async def main():
    """
    Run the examples.
    """
    logger.info('Running NLP Analysis client examples')
    await analyze_news_example()
    await analyze_economic_report_example()
    await get_combined_insights_example()
    await get_market_sentiment_example()
    logger.info('Completed NLP Analysis client examples')


if __name__ == '__main__':
    asyncio.run(main())
