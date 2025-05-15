"""
Example Usage of Common Adapters.

This module demonstrates how to use the common adapter factory to interact with other services
using the common interfaces defined in common-lib. This approach helps to break circular dependencies
between services.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

from common_lib.interfaces.trading_gateway import ITradingGateway
from common_lib.interfaces.ml_integration import IMLModelRegistry, IMLMetricsProvider
from common_lib.interfaces.ml_workbench import IExperimentManager, IModelEvaluator
from common_lib.interfaces.risk_management import IRiskManager
from common_lib.interfaces.feature_store import IFeatureProvider
from common_lib.models.trading import Order, OrderType, OrderSide

from analysis_engine.adapters.common_adapter_factory import get_common_adapter_factory


logger = logging.getLogger(__name__)


async def trading_gateway_example():
    """Example of using the Trading Gateway adapter."""
    logger.info("Running Trading Gateway adapter example")
    
    # Get the adapter factory
    factory = get_common_adapter_factory()
    
    # Get the Trading Gateway adapter
    trading_gateway = factory.get_adapter(ITradingGateway)
    
    try:
        # Get available symbols
        symbols = await trading_gateway.get_available_symbols()
        logger.info(f"Available symbols: {symbols[:5]}")
        
        # Get market data
        symbol = "EURUSD"
        timeframe = "1h"
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        
        market_data = await trading_gateway.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=100
        )
        logger.info(f"Market data: {market_data.data[:5]}")
        
        # Get account info
        account_id = "demo-account"
        account_info = await trading_gateway.get_account_info(account_id)
        logger.info(f"Account info: {account_info}")
        
        # Get open positions
        positions = await trading_gateway.get_positions(account_id)
        logger.info(f"Open positions: {positions}")
        
        # Place a test order (commented out to avoid actual order placement)
        """
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.01,
            account_id=account_id
        )
        execution_report = await trading_gateway.place_order(order)
        logger.info(f"Execution report: {execution_report}")
        """
    except Exception as e:
        logger.error(f"Error in Trading Gateway example: {str(e)}")


async def ml_integration_example():
    """Example of using the ML Integration adapter."""
    logger.info("Running ML Integration adapter example")
    
    # Get the adapter factory
    factory = get_common_adapter_factory()
    
    # Get the ML Model Registry adapter
    model_registry = factory.get_adapter(IMLModelRegistry)
    
    # Get the ML Metrics Provider adapter
    metrics_provider = factory.get_adapter(IMLMetricsProvider)
    
    try:
        # List models
        models = await model_registry.list_models()
        logger.info(f"Available models: {models[:5]}")
        
        # Get model info
        if models:
            model_id = models[0]["id"]
            model_info = await model_registry.get_model_info(model_id)
            logger.info(f"Model info: {model_info}")
            
            # Get model metrics
            metrics = await metrics_provider.get_model_metrics(
                model_id=model_id,
                start_time=datetime.utcnow() - timedelta(days=30)
            )
            logger.info(f"Model metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error in ML Integration example: {str(e)}")


async def ml_workbench_example():
    """Example of using the ML Workbench adapter."""
    logger.info("Running ML Workbench adapter example")
    
    # Get the adapter factory
    factory = get_common_adapter_factory()
    
    # Get the Experiment Manager adapter
    experiment_manager = factory.get_adapter(IExperimentManager)
    
    # Get the Model Evaluator adapter
    model_evaluator = factory.get_adapter(IModelEvaluator)
    
    try:
        # List experiments
        experiments = await experiment_manager.list_experiments(limit=5)
        logger.info(f"Available experiments: {experiments}")
        
        # Get experiment info
        if experiments.get("experiments", []):
            experiment_id = experiments["experiments"][0]["id"]
            experiment_info = await experiment_manager.get_experiment(experiment_id)
            logger.info(f"Experiment info: {experiment_info}")
            
            # Get evaluation history for a model
            model_id = experiment_info.get("model_id")
            if model_id:
                evaluation_history = await model_evaluator.get_evaluation_history(model_id)
                logger.info(f"Evaluation history: {evaluation_history}")
    except Exception as e:
        logger.error(f"Error in ML Workbench example: {str(e)}")


async def risk_management_example():
    """Example of using the Risk Management adapter."""
    logger.info("Running Risk Management adapter example")
    
    # Get the adapter factory
    factory = get_common_adapter_factory()
    
    # Get the Risk Manager adapter
    risk_manager = factory.get_adapter(IRiskManager)
    
    try:
        # Get risk limits
        account_id = "demo-account"
        risk_limits = await risk_manager.get_risk_limits(account_id)
        logger.info(f"Risk limits: {risk_limits}")
        
        # Check portfolio risk
        portfolio_risk = await risk_manager.get_portfolio_risk(account_id)
        logger.info(f"Portfolio risk: {portfolio_risk}")
        
        # Calculate position risk
        symbol = "EURUSD"
        position_risk = await risk_manager.get_position_risk(symbol, account_id)
        logger.info(f"Position risk: {position_risk}")
    except Exception as e:
        logger.error(f"Error in Risk Management example: {str(e)}")


async def feature_store_example():
    """Example of using the Feature Store adapter."""
    logger.info("Running Feature Store adapter example")
    
    # Get the adapter factory
    factory = get_common_adapter_factory()
    
    # Get the Feature Provider adapter
    feature_provider = factory.get_adapter(IFeatureProvider)
    
    try:
        # Get available features
        features = await feature_provider.get_available_features()
        logger.info(f"Available features: {features[:5]}")
        
        # Get feature data
        if features:
            feature_name = features[0]
            symbol = "EURUSD"
            timeframe = "1h"
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            
            feature_data = await feature_provider.get_feature_data(
                feature_name=feature_name,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            logger.info(f"Feature data: {feature_data.head()}")
    except Exception as e:
        logger.error(f"Error in Feature Store example: {str(e)}")


async def main():
    """Run all examples."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run examples
    await trading_gateway_example()
    await ml_integration_example()
    await ml_workbench_example()
    await risk_management_example()
    await feature_store_example()


if __name__ == "__main__":
    asyncio.run(main())
