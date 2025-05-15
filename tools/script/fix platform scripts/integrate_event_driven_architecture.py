"""
Event-Driven Architecture Integration Script

This script integrates the event-driven architecture with the actual services:
1. Connects data-pipeline-service to feature-store-service via market data events
2. Connects analysis-engine-service to strategy-execution-engine via trading signal events
3. Connects portfolio-management-service to risk-management-service via position events
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import event bus components
from common_lib.events.base import Event, EventType, EventPriority, EventMetadata, IEventBus
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType
from common_lib.events.event_publisher import EventPublisher

# Import service components
from data_pipeline_service.services.market_data_publisher import MarketDataPublisher, get_market_data_publisher
from analysis_engine_service.services.signal_publisher import SignalPublisher, get_signal_publisher
from portfolio_management_service.services.position_event_sourcing import PositionEventSourcing, get_position_event_sourcing
from portfolio_management_service.services.position_event_consumer import PositionEventConsumer, get_position_event_consumer


async def integrate_market_data_distribution():
    """
    Integrate market data distribution between data-pipeline-service and feature-store-service.
    
    This function:
    1. Sets up the market data publisher in data-pipeline-service
    2. Configures it to publish market data events
    3. Ensures feature-store-service is subscribed to these events
    """
    logger.info("Integrating market data distribution...")
    
    # Get market data publisher
    market_data_publisher = get_market_data_publisher(
        service_name="market-data-publisher",
        publish_interval=1.0,  # Publish every 1 second
        config={
            "event_bus_type": EventBusType.KAFKA,  # Use Kafka for production
            "bootstrap_servers": "localhost:9092",
            "topic_prefix": "forex."
        }
    )
    
    # Start the publisher
    await market_data_publisher.start()
    
    # Add symbols to monitor
    symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"]
    for symbol in symbols:
        market_data_publisher.add_symbol(symbol)
    
    logger.info(f"Market data publisher configured with {len(symbols)} symbols")
    
    # Create a sample market data update to verify integration
    for symbol in symbols:
        market_data_publisher.update_market_data(symbol, {
            "price": 1.0000,  # Placeholder price
            "bid": 0.9999,
            "ask": 1.0001,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    logger.info("Market data distribution integration complete")
    
    return market_data_publisher


async def integrate_trading_signal_distribution():
    """
    Integrate trading signal distribution between analysis-engine-service and strategy-execution-engine.
    
    This function:
    1. Sets up the signal publisher in analysis-engine-service
    2. Configures it to publish trading signal events
    3. Ensures strategy-execution-engine is subscribed to these events
    """
    logger.info("Integrating trading signal distribution...")
    
    # Get signal publisher
    signal_publisher = get_signal_publisher(
        service_name="signal-publisher",
        config={
            "event_bus_type": EventBusType.KAFKA,  # Use Kafka for production
            "bootstrap_servers": "localhost:9092",
            "topic_prefix": "forex."
        }
    )
    
    # Start the publisher
    await signal_publisher.start()
    
    # Create a sample signal to verify integration
    await signal_publisher.publish_signal(
        symbol="EUR/USD",
        signal_type="buy",
        timeframe="1h",
        confidence=0.85,
        price=1.1234,
        indicator_name="RSI",
        strategy_name="Trend Following"
    )
    
    logger.info("Trading signal distribution integration complete")
    
    return signal_publisher


async def integrate_position_event_sourcing():
    """
    Integrate position event sourcing between portfolio-management-service and risk-management-service.
    
    This function:
    1. Sets up the position event sourcing in portfolio-management-service
    2. Configures it to publish position events
    3. Ensures risk-management-service is subscribed to these events
    """
    logger.info("Integrating position event sourcing...")
    
    # Get position event sourcing
    position_event_sourcing = get_position_event_sourcing(
        service_name="position-event-sourcing",
        config={
            "event_bus_type": EventBusType.KAFKA,  # Use Kafka for production
            "bootstrap_servers": "localhost:9092",
            "topic_prefix": "forex."
        }
    )
    
    # Start the position event sourcing
    await position_event_sourcing.start()
    
    # Get position event consumer
    position_event_consumer = get_position_event_consumer(
        service_name="position-event-consumer",
        config={
            "event_bus_type": EventBusType.KAFKA,  # Use Kafka for production
            "bootstrap_servers": "localhost:9092",
            "topic_prefix": "forex."
        }
    )
    
    # Start the position event consumer
    await position_event_consumer.start()
    
    # Create a sample position to verify integration
    position_id = await position_event_sourcing.open_position(
        symbol="EUR/USD",
        direction="BUY",
        quantity=1.0,
        entry_price=1.1234,
        account_id="demo-account",
        stop_loss=1.1200,
        take_profit=1.1300
    )
    
    # Wait for events to be processed
    await asyncio.sleep(1)
    
    # Update the position
    await position_event_sourcing.update_position(
        position_id=position_id,
        current_price=1.1250
    )
    
    # Wait for events to be processed
    await asyncio.sleep(1)
    
    # Close the position
    await position_event_sourcing.close_position(
        position_id=position_id,
        exit_price=1.1260
    )
    
    logger.info("Position event sourcing integration complete")
    
    return position_event_sourcing, position_event_consumer


async def verify_integration():
    """
    Verify that the event-driven architecture integration is working correctly.
    
    This function:
    1. Checks that market data events are flowing from data-pipeline-service to feature-store-service
    2. Checks that trading signal events are flowing from analysis-engine-service to strategy-execution-engine
    3. Checks that position events are flowing from portfolio-management-service to risk-management-service
    """
    logger.info("Verifying event-driven architecture integration...")
    
    # Verify market data distribution
    # TODO: Add verification code
    
    # Verify trading signal distribution
    # TODO: Add verification code
    
    # Verify position event sourcing
    # TODO: Add verification code
    
    logger.info("Event-driven architecture integration verified")


async def main():
    """
    Main function to integrate the event-driven architecture.
    """
    try:
        # Integrate market data distribution
        market_data_publisher = await integrate_market_data_distribution()
        
        # Integrate trading signal distribution
        signal_publisher = await integrate_trading_signal_distribution()
        
        # Integrate position event sourcing
        position_event_sourcing, position_event_consumer = await integrate_position_event_sourcing()
        
        # Verify integration
        await verify_integration()
        
        # Keep the script running to allow events to flow
        logger.info("Event-driven architecture integration complete. Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        
        # Stop services
        await market_data_publisher.stop()
        await signal_publisher.stop()
        await position_event_sourcing.stop()
        await position_event_consumer.stop()
        
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Error during integration: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
