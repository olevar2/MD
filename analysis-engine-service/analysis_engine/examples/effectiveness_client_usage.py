"""
Example of using the Tool Effectiveness client

This module demonstrates how to use the standardized Tool Effectiveness client
to interact with the Analysis Engine Service API.
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta

from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

async def register_signal_example():
    """
    Example of registering a signal using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the effectiveness client
    client = factory.get_effectiveness_client()
    
    # Example signal data
    tool_id = "macd_crossover_v1"
    signal_type = "buy"
    instrument = "EUR_USD"
    timestamp = datetime.utcnow()
    confidence = 0.85
    timeframe = "1H"
    market_regime = "trending"
    additional_data = {
        "macd_value": 0.0025,
        "signal_line": 0.0010,
        "histogram": 0.0015
    }
    
    try:
        # Register signal
        result = await client.register_signal(
            tool_id=tool_id,
            signal_type=signal_type,
            instrument=instrument,
            timestamp=timestamp,
            confidence=confidence,
            timeframe=timeframe,
            market_regime=market_regime,
            additional_data=additional_data
        )
        
        logger.info(f"Registered signal: {result}")
        return result
    except Exception as e:
        logger.error(f"Error registering signal: {str(e)}")
        raise

async def register_outcome_example(signal_id: str):
    """
    Example of registering an outcome using the standardized client.
    
    Args:
        signal_id: ID of the signal to register an outcome for
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the effectiveness client
    client = factory.get_effectiveness_client()
    
    # Example outcome data
    success = True
    realized_profit = 0.75
    timestamp = datetime.utcnow() + timedelta(hours=2)  # 2 hours after signal
    additional_data = {
        "exit_reason": "take_profit",
        "pips": 15,
        "trade_duration_minutes": 120
    }
    
    try:
        # Register outcome
        result = await client.register_outcome(
            signal_id=signal_id,
            success=success,
            realized_profit=realized_profit,
            timestamp=timestamp,
            additional_data=additional_data
        )
        
        logger.info(f"Registered outcome: {result}")
        return result
    except Exception as e:
        logger.error(f"Error registering outcome: {str(e)}")
        raise

async def get_effectiveness_metrics_example():
    """
    Example of getting effectiveness metrics using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the effectiveness client
    client = factory.get_effectiveness_client()
    
    # Example filter parameters
    tool_id = "macd_crossover_v1"
    timeframe = "1H"
    instrument = "EUR_USD"
    market_regime = "trending"
    from_date = datetime.utcnow() - timedelta(days=30)
    to_date = datetime.utcnow()
    
    try:
        # Get effectiveness metrics
        result = await client.get_effectiveness_metrics(
            tool_id=tool_id,
            timeframe=timeframe,
            instrument=instrument,
            market_regime=market_regime,
            from_date=from_date,
            to_date=to_date
        )
        
        logger.info(f"Got effectiveness metrics: {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting effectiveness metrics: {str(e)}")
        raise

async def get_dashboard_data_example():
    """
    Example of getting dashboard data using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the effectiveness client
    client = factory.get_effectiveness_client()
    
    try:
        # Get dashboard data
        result = await client.get_dashboard_data()
        
        logger.info(f"Got dashboard data: {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        raise

async def save_effectiveness_report_example():
    """
    Example of saving an effectiveness report using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the effectiveness client
    client = factory.get_effectiveness_client()
    
    # Example report parameters
    name = "Monthly MACD Performance Report"
    description = "Performance analysis of MACD crossover strategy for the past month"
    tool_id = "macd_crossover_v1"
    timeframe = "1H"
    instrument = "EUR_USD"
    market_regime = "trending"
    from_date = datetime.utcnow() - timedelta(days=30)
    to_date = datetime.utcnow()
    
    try:
        # Save effectiveness report
        result = await client.save_effectiveness_report(
            name=name,
            description=description,
            tool_id=tool_id,
            timeframe=timeframe,
            instrument=instrument,
            market_regime=market_regime,
            from_date=from_date,
            to_date=to_date
        )
        
        logger.info(f"Saved effectiveness report: {result}")
        return result
    except Exception as e:
        logger.error(f"Error saving effectiveness report: {str(e)}")
        raise

async def get_effectiveness_reports_example():
    """
    Example of getting effectiveness reports using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the effectiveness client
    client = factory.get_effectiveness_client()
    
    # Example pagination parameters
    skip = 0
    limit = 10
    
    try:
        # Get effectiveness reports
        result = await client.get_effectiveness_reports(
            skip=skip,
            limit=limit
        )
        
        logger.info(f"Got effectiveness reports: {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting effectiveness reports: {str(e)}")
        raise

async def get_effectiveness_report_example(report_id: int):
    """
    Example of getting a specific effectiveness report using the standardized client.
    
    Args:
        report_id: ID of the report to retrieve
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the effectiveness client
    client = factory.get_effectiveness_client()
    
    try:
        # Get effectiveness report
        result = await client.get_effectiveness_report(report_id)
        
        logger.info(f"Got effectiveness report: {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting effectiveness report: {str(e)}")
        raise

async def clear_tool_data_example(tool_id: str):
    """
    Example of clearing tool data using the standardized client.
    
    Args:
        tool_id: ID of the tool to clear data for
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the effectiveness client
    client = factory.get_effectiveness_client()
    
    try:
        # Clear tool data
        result = await client.clear_tool_data(tool_id)
        
        logger.info(f"Cleared tool data: {result}")
        return result
    except Exception as e:
        logger.error(f"Error clearing tool data: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running Tool Effectiveness client examples")
    
    # Register a signal
    signal_result = await register_signal_example()
    signal_id = signal_result.get("signal_id")
    
    # Register an outcome for the signal
    if signal_id:
        await register_outcome_example(signal_id)
    
    # Get effectiveness metrics
    await get_effectiveness_metrics_example()
    
    # Get dashboard data
    await get_dashboard_data_example()
    
    # Save an effectiveness report
    report_result = await save_effectiveness_report_example()
    report_id = report_result.get("report_id")
    
    # Get all effectiveness reports
    await get_effectiveness_reports_example()
    
    # Get a specific effectiveness report
    if report_id:
        await get_effectiveness_report_example(report_id)
    
    # Clear tool data (commented out to avoid deleting data)
    # await clear_tool_data_example("macd_crossover_v1")
    
    logger.info("Completed Tool Effectiveness client examples")

if __name__ == "__main__":
    asyncio.run(main())
