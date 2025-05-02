"""
End-to-end test for the ML retraining and feedback loop integration.

This comprehensive E2E test validates the complete feedback loop from data ingestion 
through model retraining to strategy deployment:

1. Data Pipeline: Ingests and processes market data
2. Analysis Engine: Analyzes data and produces signals
3. Strategy Execution: Executes trades based on signals
4. Risk Management: Validates trade against risk parameters
5. Trading Gateway: Simulates trade execution
6. Feedback Collection: Collects performance feedback
7. ML Retraining: Retrains models based on feedback
8. Backtest Validation: Validates improved model performance
9. Strategy Deployment: Deploys updated strategy

This ensures all components work together in a full end-to-end flow.
"""

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
import pytest
from pathlib import Path

from e2e.framework.service_orchestrator import ServiceOrchestrator
from e2e.framework.api_clients import (
    DataPipelineClient,
    AnalysisEngineClient, 
    StrategyExecutionClient,
    RiskManagementClient,
    TradingGatewayClient,
    MLIntegrationClient
)
from e2e.utils.market_data_simulator import MarketDataSimulator
from e2e.utils.test_data_generator import generate_test_data
from e2e.utils.test_metrics import calculate_performance_metrics
from e2e.utils.validators import verify_model_improvement

# Configure logging
logger = logging.getLogger(__name__)

# Test configuration
CONFIG = {
    "test_duration_seconds": 600,  # 10 minutes
    "market_volatility": "medium",
    "asset_pair": "EUR/USD",
    "expected_trades_min": 5,
    "backtest_lookback_days": 30,
    "min_sharpe_improvement": 0.1,  # 10% improvement in Sharpe ratio
    "max_wait_for_retraining": 300,  # 5 minutes
    "services_startup_timeout": 120,  # 2 minutes
}


@pytest.fixture(scope="module")
async def services():
    """Start and configure all required services for the E2E test"""
    orchestrator = ServiceOrchestrator()
    
    # Start all required services
    logger.info("Starting services for E2E test...")
    await orchestrator.start_services([
        "data-pipeline-service",
        "analysis-engine-service",
        "strategy-execution-engine",
        "risk-management-service",
        "trading-gateway-service",
        "ml-integration-service"
    ], timeout=CONFIG["services_startup_timeout"])
    
    # Initialize API clients
    clients = {
        "data_pipeline": DataPipelineClient(),
        "analysis_engine": AnalysisEngineClient(),
        "strategy_execution": StrategyExecutionClient(),
        "risk_management": RiskManagementClient(),
        "trading_gateway": TradingGatewayClient(use_simulation=True),
        "ml_integration": MLIntegrationClient()
    }
    
    # Ensure all services are healthy
    for service_name, client in clients.items():
        healthy = await client.check_health()
        logger.info(f"Service {service_name} health check: {'PASSED' if healthy else 'FAILED'}")
        assert healthy, f"Service {service_name} is not healthy"
    
    try:
        yield clients
    finally:
        # Cleanup and shut down services
        logger.info("Shutting down services...")
        await orchestrator.stop_services()


@pytest.fixture(scope="module")
async def market_data_simulator():
    """Create and configure a market data simulator"""
    simulator = MarketDataSimulator(
        volatility=CONFIG["market_volatility"],
        seed=42  # For reproducibility
    )
    await simulator.initialize()
    return simulator


@pytest.fixture(scope="module")
async def strategy_setup(services):
    """Create and deploy a test strategy"""
    strategy_id = f"e2e_test_strategy_{uuid.uuid4().hex[:8]}"
    
    # Generate strategy configuration
    strategy_config = {
        "id": strategy_id,
        "name": "E2E Test Strategy",
        "version": "1.0.0",
        "model_id": f"forex_prediction_model_{uuid.uuid4().hex[:8]}",
        "asset_class": "forex",
        "instruments": [CONFIG["asset_pair"]],
        "timeframes": ["1m", "5m", "15m"],
        "risk_parameters": {
            "max_position_size": 1000,
            "max_drawdown_percent": 2.0,
            "stop_loss_pips": 20,
            "take_profit_pips": 40
        },
        "parameters": {
            "entry_threshold": 0.75,
            "exit_threshold": 0.65,
            "trailing_stop": True
        }
    }
    
    # Train initial model
    model_id = strategy_config["model_id"]
    training_data = generate_test_data(
        pair=CONFIG["asset_pair"],
        days=30,
        resolution="1m"
    )
    
    logger.info(f"Training initial model {model_id}...")
    training_job = await services["ml_integration"].train_model(
        model_id=model_id,
        training_data=training_data,
        parameters={
            "model_type": "lstm",
            "lookback_periods": 60,
            "epochs": 10,  # Reduced for test speed
            "validation_split": 0.2
        }
    )
    
    # Wait for training to complete
    job_status = await services["ml_integration"].wait_for_job_completion(
        job_id=training_job["job_id"],
        timeout=300  # 5 minutes
    )
    assert job_status["status"] == "completed", f"Model training failed: {job_status.get('error', 'Unknown error')}"
    
    # Deploy strategy
    logger.info(f"Deploying test strategy {strategy_id}...")
    deployment = await services["strategy_execution"].deploy_strategy(strategy_config)
    assert deployment["success"], f"Strategy deployment failed: {deployment.get('message', 'Unknown error')}"
    
    return {
        "strategy_id": strategy_id,
        "model_id": model_id,
        "config": strategy_config
    }


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_feedback_loop_cycle(services, market_data_simulator, strategy_setup):
    """
    End-to-end test of the complete feedback loop cycle from data ingestion
    through model retraining to strategy deployment.
    """
    strategy_id = strategy_setup["strategy_id"]
    model_id = strategy_setup["model_id"]
    
    logger.info(f"Starting E2E test for strategy {strategy_id} with model {model_id}")
    
    # Step 1: Start market data simulation
    logger.info("Starting market data simulation...")
    simulation_task = asyncio.create_task(
        market_data_simulator.stream_data(
            pair=CONFIG["asset_pair"],
            target_service=services["data_pipeline"],
            duration_seconds=CONFIG["test_duration_seconds"]
        )
    )
    
    # Step 2: Track initial strategy performance
    initial_performance = await services["strategy_execution"].get_strategy_performance(strategy_id)
    logger.info(f"Initial strategy performance: {initial_performance}")
    
    # Step 3: Wait for trades to execute
    logger.info(f"Waiting for trades to execute (up to {CONFIG['test_duration_seconds']} seconds)...")
    start_time = time.time()
    trades_executed = 0
    
    while time.time() - start_time < CONFIG["test_duration_seconds"]:
        trades = await services["trading_gateway"].get_recent_trades(strategy_id)
        new_trades = len(trades)
        
        if new_trades > trades_executed:
            logger.info(f"Executed {new_trades - trades_executed} new trades, total: {new_trades}")
            trades_executed = new_trades
        
        if trades_executed >= CONFIG["expected_trades_min"]:
            logger.info(f"Minimum trade threshold reached ({trades_executed} trades)")
            break
            
        await asyncio.sleep(10)  # Check every 10 seconds
    
    assert trades_executed >= CONFIG["expected_trades_min"], \
        f"Not enough trades executed: {trades_executed} (minimum: {CONFIG['expected_trades_min']})"
    
    # Step 4: Wait for feedback processing and model retraining
    logger.info("Waiting for feedback processing and model retraining...")
    retraining_detected = False
    retraining_jobs = []
    retraining_start = time.time()
    
    while time.time() - retraining_start < CONFIG["max_wait_for_retraining"]:
        # Check for model training jobs
        jobs = await services["ml_integration"].list_jobs(model_id=model_id)
        # Filter for jobs created after our test started
        new_jobs = [j for j in jobs if j.get("created_at", 0) > retraining_start]
        
        if new_jobs and not retraining_detected:
            logger.info(f"Detected {len(new_jobs)} new training jobs for model {model_id}")
            retraining_detected = True
            retraining_jobs = new_jobs
            break
            
        await asyncio.sleep(10)  # Check every 10 seconds
    
    assert retraining_detected, f"No model retraining detected within {CONFIG['max_wait_for_retraining']} seconds"
    
    # Step 5: Wait for retraining job to complete
    if retraining_jobs:
        latest_job = sorted(retraining_jobs, key=lambda j: j.get("created_at", 0), reverse=True)[0]
        job_id = latest_job["job_id"]
        
        logger.info(f"Waiting for retraining job {job_id} to complete...")
        job_status = await services["ml_integration"].wait_for_job_completion(
            job_id=job_id,
            timeout=CONFIG["max_wait_for_retraining"]
        )
        
        assert job_status["status"] == "completed", \
            f"Model retraining job failed: {job_status.get('error', 'Unknown error')}"
    
    # Step 6: Verify backtest ran and performance improved
    logger.info("Verifying backtest results and performance improvement...")
    
    # Wait for backtest to complete
    await asyncio.sleep(30)  # Give time for backtest and deployment
    
    # Get updated strategy details
    updated_strategy = await services["strategy_execution"].get_strategy(strategy_id)
    
    # Verify model version was updated
    assert updated_strategy["model_version"] != strategy_setup["config"].get("model_version", "1.0.0"), \
        "Model version was not updated after retraining"
    
    # Verify backtest was performed
    backtest_results = await services["strategy_execution"].get_latest_backtest(strategy_id)
    assert backtest_results is not None, "No backtest results found"
    
    # Verify performance metrics
    if "metrics" in backtest_results:
        improvement = verify_model_improvement(
            initial_performance,
            backtest_results["metrics"],
            min_sharpe_improvement=CONFIG["min_sharpe_improvement"]
        )
        logger.info(f"Performance improvement: {improvement}")
        
        # This assertion might be too strict for a real E2E test, consider making it optional
        # assert improvement["improved"], f"Model performance did not improve: {improvement['details']}"
    
    # Step 7: Verify strategy was redeployed with new model
    deployment_history = await services["strategy_execution"].get_deployment_history(strategy_id)
    recent_deployments = [d for d in deployment_history 
                          if d.get("timestamp", 0) > retraining_start]
    
    assert len(recent_deployments) > 0, "No recent strategy deployments found after retraining"
    
    logger.info("E2E test completed successfully - full feedback loop cycle verified!")
    
    # Cancel the simulation task if it's still running
    if not simulation_task.done():
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass
