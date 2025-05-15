"""
Update Service APIs for Event-Driven Architecture

This script updates the service APIs to use the event-driven architecture:
1. Updates data-pipeline-service API to publish market data events
2. Updates analysis-engine-service API to publish trading signal events
3. Updates portfolio-management-service API to use position event sourcing
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


def update_data_pipeline_service_api():
    """
    Update data-pipeline-service API to publish market data events.
    
    This function:
    1. Updates the OHLCV endpoint to publish market data events
    2. Updates the tick data endpoint to publish market data events
    """
    logger.info("Updating data-pipeline-service API...")
    
    # Path to the API file
    api_file = os.path.join("data-pipeline-service", "api", "v1", "ohlcv_api.py")
    
    # Check if the file exists
    if not os.path.exists(api_file):
        logger.warning(f"API file not found: {api_file}")
        return
    
    # Read the file
    with open(api_file, "r") as f:
        content = f.read()
    
    # Check if the file already has event publishing
    if "market_data_publisher" in content:
        logger.info("API file already has event publishing")
        return
    
    # Add import for market data publisher
    import_statement = """
from services.market_data_publisher import get_market_data_publisher
"""
    
    # Add code to publish market data events
    publish_code = """
    # Publish market data event
    market_data_publisher = get_market_data_publisher()
    market_data_publisher.update_market_data(symbol, {
        "price": data[-1]["close"],
        "bid": data[-1]["close"] - 0.0001,
        "ask": data[-1]["close"] + 0.0001,
        "timestamp": data[-1]["timestamp"],
        "open": data[-1]["open"],
        "high": data[-1]["high"],
        "low": data[-1]["low"],
        "close": data[-1]["close"],
        "volume": data[-1]["volume"]
    })
"""
    
    # Find the right place to add the import
    import_index = content.find("from fastapi import")
    if import_index == -1:
        import_index = content.find("import ")
    
    # Find the right place to add the publish code
    return_index = content.find("return ")
    
    # Update the content
    if import_index != -1 and return_index != -1:
        new_content = (
            content[:import_index] + 
            import_statement + 
            content[import_index:return_index] + 
            publish_code + 
            content[return_index:]
        )
        
        # Write the updated content
        with open(api_file, "w") as f:
            f.write(new_content)
        
        logger.info(f"Updated API file: {api_file}")
    else:
        logger.warning(f"Could not update API file: {api_file}")


def update_analysis_engine_service_api():
    """
    Update analysis-engine-service API to publish trading signal events.
    
    This function:
    1. Updates the analysis endpoint to publish trading signal events
    2. Updates the signal generation endpoint to publish trading signal events
    """
    logger.info("Updating analysis-engine-service API...")
    
    # Path to the API file
    api_file = os.path.join("analysis-engine-service", "api", "v1", "analysis_api.py")
    
    # Check if the file exists
    if not os.path.exists(api_file):
        logger.warning(f"API file not found: {api_file}")
        return
    
    # Read the file
    with open(api_file, "r") as f:
        content = f.read()
    
    # Check if the file already has event publishing
    if "signal_publisher" in content:
        logger.info("API file already has event publishing")
        return
    
    # Add import for signal publisher
    import_statement = """
from services.signal_publisher import get_signal_publisher
"""
    
    # Add code to publish trading signal events
    publish_code = """
    # Publish trading signal event
    signal_publisher = get_signal_publisher()
    await signal_publisher.publish_signal(
        symbol=analysis_result["symbol"],
        signal_type=analysis_result["signal"],
        timeframe=analysis_result["timeframe"],
        confidence=analysis_result["confidence"],
        price=analysis_result["price"],
        indicator_name=analysis_result.get("indicator"),
        strategy_name=analysis_result.get("strategy")
    )
"""
    
    # Find the right place to add the import
    import_index = content.find("from fastapi import")
    if import_index == -1:
        import_index = content.find("import ")
    
    # Find the right place to add the publish code
    return_index = content.find("return ")
    
    # Update the content
    if import_index != -1 and return_index != -1:
        new_content = (
            content[:import_index] + 
            import_statement + 
            content[import_index:return_index] + 
            publish_code + 
            content[return_index:]
        )
        
        # Write the updated content
        with open(api_file, "w") as f:
            f.write(new_content)
        
        logger.info(f"Updated API file: {api_file}")
    else:
        logger.warning(f"Could not update API file: {api_file}")


def update_portfolio_management_service_api():
    """
    Update portfolio-management-service API to use position event sourcing.
    
    This function:
    1. Updates the position creation endpoint to use position event sourcing
    2. Updates the position update endpoint to use position event sourcing
    3. Updates the position close endpoint to use position event sourcing
    """
    logger.info("Updating portfolio-management-service API...")
    
    # Path to the API file
    api_file = os.path.join("portfolio-management-service", "api", "v1", "position_api.py")
    
    # Check if the file exists
    if not os.path.exists(api_file):
        logger.warning(f"API file not found: {api_file}")
        return
    
    # Read the file
    with open(api_file, "r") as f:
        content = f.read()
    
    # Check if the file already has event sourcing
    if "position_event_sourcing" in content:
        logger.info("API file already has event sourcing")
        return
    
    # Add import for position event sourcing
    import_statement = """
from services.position_event_sourcing import get_position_event_sourcing
"""
    
    # Add code to use position event sourcing for creating positions
    create_position_code = """
    # Use position event sourcing to open position
    position_event_sourcing = get_position_event_sourcing()
    position_id = await position_event_sourcing.open_position(
        symbol=position.symbol,
        direction=position.direction,
        quantity=position.quantity,
        entry_price=position.entry_price,
        account_id=position.account_id,
        stop_loss=position.stop_loss,
        take_profit=position.take_profit,
        strategy_id=position.strategy_id,
        metadata=position.metadata
    )
    
    # Return the position ID
    return {"position_id": position_id}
"""
    
    # Add code to use position event sourcing for updating positions
    update_position_code = """
    # Use position event sourcing to update position
    position_event_sourcing = get_position_event_sourcing()
    await position_event_sourcing.update_position(
        position_id=position_id,
        current_price=position_update.current_price,
        stop_loss=position_update.stop_loss,
        take_profit=position_update.take_profit,
        metadata=position_update.metadata
    )
    
    # Return success
    return {"status": "success"}
"""
    
    # Add code to use position event sourcing for closing positions
    close_position_code = """
    # Use position event sourcing to close position
    position_event_sourcing = get_position_event_sourcing()
    await position_event_sourcing.close_position(
        position_id=position_id,
        exit_price=close_request.exit_price,
        metadata=close_request.metadata
    )
    
    # Return success
    return {"status": "success"}
"""
    
    # Find the right place to add the import
    import_index = content.find("from fastapi import")
    if import_index == -1:
        import_index = content.find("import ")
    
    # Find the right places to add the event sourcing code
    create_position_index = content.find("async def create_position")
    update_position_index = content.find("async def update_position")
    close_position_index = content.find("async def close_position")
    
    # Update the content
    if import_index != -1 and create_position_index != -1 and update_position_index != -1 and close_position_index != -1:
        # Find the function bodies
        create_position_body_start = content.find("    # Create position", create_position_index)
        create_position_body_end = content.find("    return ", create_position_index)
        
        update_position_body_start = content.find("    # Update position", update_position_index)
        update_position_body_end = content.find("    return ", update_position_index)
        
        close_position_body_start = content.find("    # Close position", close_position_index)
        close_position_body_end = content.find("    return ", close_position_index)
        
        # Replace the function bodies
        if (create_position_body_start != -1 and create_position_body_end != -1 and
            update_position_body_start != -1 and update_position_body_end != -1 and
            close_position_body_start != -1 and close_position_body_end != -1):
            
            new_content = (
                content[:import_index] + 
                import_statement + 
                content[import_index:create_position_body_start] + 
                create_position_code + 
                content[create_position_body_end:update_position_body_start] + 
                update_position_code + 
                content[update_position_body_end:close_position_body_start] + 
                close_position_code + 
                content[close_position_body_end:]
            )
            
            # Write the updated content
            with open(api_file, "w") as f:
                f.write(new_content)
            
            logger.info(f"Updated API file: {api_file}")
        else:
            logger.warning(f"Could not update API file: {api_file}")
    else:
        logger.warning(f"Could not update API file: {api_file}")


def main():
    """
    Main function to update service APIs.
    """
    try:
        # Update data-pipeline-service API
        update_data_pipeline_service_api()
        
        # Update analysis-engine-service API
        update_analysis_engine_service_api()
        
        # Update portfolio-management-service API
        update_portfolio_management_service_api()
        
        logger.info("Service API updates complete")
        
    except Exception as e:
        logger.error(f"Error during API updates: {str(e)}")
        raise


if __name__ == "__main__":
    main()
