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

async def get_feedback_statistics_example():
    """
    Example of getting feedback statistics using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the feedback client
    client = factory.get_feedback_client()
    
    # Example filter parameters
    strategy_id = "macd_crossover_strategy"
    model_id = "macd_model_v1"
    instrument = "EUR_USD"
    start_time = datetime.utcnow() - timedelta(days=30)
    end_time = datetime.utcnow()
    
    try:
        # Get feedback statistics
        result = await client.get_feedback_statistics(
            strategy_id=strategy_id,
            model_id=model_id,
            instrument=instrument,
            start_time=start_time,
            end_time=end_time
        )
        
        logger.info(f"Got feedback statistics: {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting feedback statistics: {str(e)}")
        raise

async def trigger_model_retraining_example():
    """
    Example of triggering model retraining using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the feedback client
    client = factory.get_feedback_client()
    
    # Example model ID
    model_id = "macd_model_v1"
    
    try:
        # Trigger model retraining
        result = await client.trigger_model_retraining(model_id)
        
        logger.info(f"Triggered model retraining: {result}")
        return result
    except Exception as e:
        logger.error(f"Error triggering model retraining: {str(e)}")
        raise

async def update_feedback_rules_example():
    """
    Example of updating feedback rules using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the feedback client
    client = factory.get_feedback_client()
    
    # Example rule updates
    rule_updates = [
        {
            "rule_id": "high_confidence_signals",
            "enabled": True,
            "priority": 1,
            "conditions": {
                "confidence_threshold": 0.85,
                "signal_types": ["buy", "sell"]
            },
            "actions": {
                "weight": 2.0,
                "auto_approve": True
            }
        },
        {
            "rule_id": "low_confidence_signals",
            "enabled": True,
            "priority": 2,
            "conditions": {
                "confidence_threshold": 0.5,
                "signal_types": ["buy", "sell"]
            },
            "actions": {
                "weight": 0.5,
                "auto_approve": False
            }
        }
    ]
    
    try:
        # Update feedback rules
        result = await client.update_feedback_rules(rule_updates)
        
        logger.info(f"Updated feedback rules: {result}")
        return result
    except Exception as e:
        logger.error(f"Error updating feedback rules: {str(e)}")
        raise

async def get_parameter_performance_example():
    """
    Example of getting parameter performance using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the feedback client
    client = factory.get_feedback_client()
    
    # Example strategy ID and minimum samples
    strategy_id = "macd_crossover_strategy"
    min_samples = 10
    
    try:
        # Get parameter performance
        result = await client.get_parameter_performance(
            strategy_id=strategy_id,
            min_samples=min_samples
        )
        
        logger.info(f"Got parameter performance: {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting parameter performance: {str(e)}")
        raise

async def submit_feedback_example():
    """
    Example of submitting feedback using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the feedback client
    client = factory.get_feedback_client()
    
    # Example feedback data
    source = "user"
    target_id = "signal_12345"
    feedback_type = "accuracy"
    content = {
        "rating": 4,
        "comments": "Signal was accurate but slightly delayed",
        "market_conditions": "trending"
    }
    timestamp = datetime.utcnow()
    
    try:
        # Submit feedback
        result = await client.submit_feedback(
            source=source,
            target_id=target_id,
            feedback_type=feedback_type,
            content=content,
            timestamp=timestamp
        )
        
        logger.info(f"Submitted feedback: {result}")
        return result
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running Feedback client examples")
    
    # Get feedback statistics
    await get_feedback_statistics_example()
    
    # Trigger model retraining
    await trigger_model_retraining_example()
    
    # Update feedback rules
    await update_feedback_rules_example()
    
    # Get parameter performance
    await get_parameter_performance_example()
    
    # Submit feedback
    await submit_feedback_example()
    
    logger.info("Completed Feedback client examples")

if __name__ == "__main__":
    asyncio.run(main())
