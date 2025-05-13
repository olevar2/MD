"""
Example of using the Feedback client

This module demonstrates how to use the standardized Feedback client
to interact with the Analysis Engine Service API.
"""
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
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
async def get_feedback_statistics_example():
    """
    Example of getting feedback statistics using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_feedback_client()
    strategy_id = 'macd_crossover_strategy'
    model_id = 'macd_model_v1'
    instrument = 'EUR_USD'
    start_time = datetime.utcnow() - timedelta(days=30)
    end_time = datetime.utcnow()
    try:
        result = await client.get_feedback_statistics(strategy_id=
            strategy_id, model_id=model_id, instrument=instrument,
            start_time=start_time, end_time=end_time)
        logger.info(f'Got feedback statistics: {result}')
        return result
    except Exception as e:
        logger.error(f'Error getting feedback statistics: {str(e)}')
        raise


@async_with_exception_handling
async def trigger_model_retraining_example():
    """
    Example of triggering model retraining using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_feedback_client()
    model_id = 'macd_model_v1'
    try:
        result = await client.trigger_model_retraining(model_id)
        logger.info(f'Triggered model retraining: {result}')
        return result
    except Exception as e:
        logger.error(f'Error triggering model retraining: {str(e)}')
        raise


@async_with_exception_handling
async def update_feedback_rules_example():
    """
    Example of updating feedback rules using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_feedback_client()
    rule_updates = [{'rule_id': 'high_confidence_signals', 'enabled': True,
        'priority': 1, 'conditions': {'confidence_threshold': 0.85,
        'signal_types': ['buy', 'sell']}, 'actions': {'weight': 2.0,
        'auto_approve': True}}, {'rule_id': 'low_confidence_signals',
        'enabled': True, 'priority': 2, 'conditions': {
        'confidence_threshold': 0.5, 'signal_types': ['buy', 'sell']},
        'actions': {'weight': 0.5, 'auto_approve': False}}]
    try:
        result = await client.update_feedback_rules(rule_updates)
        logger.info(f'Updated feedback rules: {result}')
        return result
    except Exception as e:
        logger.error(f'Error updating feedback rules: {str(e)}')
        raise


@async_with_exception_handling
async def get_parameter_performance_example():
    """
    Example of getting parameter performance using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_feedback_client()
    strategy_id = 'macd_crossover_strategy'
    min_samples = 10
    try:
        result = await client.get_parameter_performance(strategy_id=
            strategy_id, min_samples=min_samples)
        logger.info(f'Got parameter performance: {result}')
        return result
    except Exception as e:
        logger.error(f'Error getting parameter performance: {str(e)}')
        raise


@async_with_exception_handling
async def submit_feedback_example():
    """
    Example of submitting feedback using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_feedback_client()
    source = 'user'
    target_id = 'signal_12345'
    feedback_type = 'accuracy'
    content = {'rating': 4, 'comments':
        'Signal was accurate but slightly delayed', 'market_conditions':
        'trending'}
    timestamp = datetime.utcnow()
    try:
        result = await client.submit_feedback(source=source, target_id=
            target_id, feedback_type=feedback_type, content=content,
            timestamp=timestamp)
        logger.info(f'Submitted feedback: {result}')
        return result
    except Exception as e:
        logger.error(f'Error submitting feedback: {str(e)}')
        raise


async def main():
    """
    Run the examples.
    """
    logger.info('Running Feedback client examples')
    await get_feedback_statistics_example()
    await trigger_model_retraining_example()
    await update_feedback_rules_example()
    await get_parameter_performance_example()
    await submit_feedback_example()
    logger.info('Completed Feedback client examples')


if __name__ == '__main__':
    asyncio.run(main())
