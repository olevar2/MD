"""
Adapter Usage Example

This module demonstrates how to use the adapter pattern to reduce direct dependencies.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

from common_lib.interfaces.data_interfaces import IFeatureProvider, IDataPipeline
from common_lib.interfaces.ml_interfaces import IModelProvider
from common_lib.interfaces.trading_interfaces import IRiskManager, ITradingGateway

from analysis_engine.adapters.adapter_factory import adapter_factory


async def analyze_market_conditions(symbols: List[str], timeframe: str) -> Dict[str, Any]:
    """
    Analyze market conditions for a list of symbols.
    
    Args:
        symbols: List of symbols to analyze
        timeframe: Timeframe for the analysis
        
    Returns:
        Analysis results
    """
    # Get adapters from the factory
    feature_provider = adapter_factory.get_adapter(IFeatureProvider)
    data_pipeline = adapter_factory.get_adapter(IDataPipeline)
    model_provider = adapter_factory.get_adapter(IModelProvider)
    risk_manager = adapter_factory.get_adapter(IRiskManager)
    trading_gateway = adapter_factory.get_adapter(ITradingGateway)
    
    # Define time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Get market data
    market_data = await data_pipeline.get_market_data(symbols, timeframe, start_time, end_time)
    
    # Get features
    features = await feature_provider.get_features(["volatility", "trend", "momentum"], start_time, end_time)
    
    # Get model prediction
    prediction = await model_provider.get_model_prediction("market_conditions_model", features)
    
    # Evaluate risk
    risk = await risk_manager.evaluate_risk({"symbols": symbols, "prediction": prediction})
    
    # Get market status
    market_status = await trading_gateway.get_market_status(symbols)
    
    # Combine results
    return {
        "symbols": symbols,
        "timeframe": timeframe,
        "prediction": prediction,
        "risk": risk,
        "market_status": market_status
    }


async def main():
    """Main function."""
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframe = "1h"
    
    results = await analyze_market_conditions(symbols, timeframe)
    print(f"Analysis results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
