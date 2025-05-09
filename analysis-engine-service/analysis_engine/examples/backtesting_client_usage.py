"""
Example of using the Backtesting client

This module demonstrates how to use the standardized Backtesting client
to interact with the Analysis Engine Service API.
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta

from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

async def run_backtest_example():
    """
    Example of running a backtest using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the backtesting client
    client = factory.get_backtesting_client()
    
    # Example strategy configuration
    strategy_config = {
        "strategy_id": "moving_average_crossover",
        "parameters": {
            "fast_period": 10,
            "slow_period": 30
        },
        "risk_settings": {
            "max_drawdown_pct": 20,
            "max_risk_per_trade_pct": 2
        },
        "position_sizing": {
            "method": "fixed_risk",
            "risk_pct": 1
        }
    }
    
    # Example backtest parameters
    start_date = datetime.utcnow() - timedelta(days=365)  # 1 year ago
    end_date = datetime.utcnow()
    instruments = ["EUR_USD", "GBP_USD"]
    initial_capital = 10000.0
    
    try:
        # Run backtest
        result = await client.run_backtest(
            strategy_config=strategy_config,
            start_date=start_date,
            end_date=end_date,
            instruments=instruments,
            initial_capital=initial_capital,
            commission_model="fixed",
            commission_settings={"fixed_commission": 5.0},
            slippage_model="fixed",
            slippage_settings={"fixed_pips": 1},
            data_source="historical",
            data_parameters={"timeframe": "H1"}
        )
        
        logger.info(f"Backtest completed with ID: {result.get('backtest_id')}")
        
        # Extract key metrics
        backtest_id = result.get("backtest_id")
        final_capital = result.get("final_capital")
        trade_metrics = result.get("trade_metrics", {})
        performance_metrics = result.get("performance_metrics", {})
        
        logger.info(f"Initial capital: ${initial_capital:.2f}")
        logger.info(f"Final capital: ${final_capital:.2f}")
        logger.info(f"Total return: {performance_metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Total trades: {trade_metrics.get('total_trades', 0)}")
        logger.info(f"Win rate: {trade_metrics.get('win_rate', 0) * 100:.2f}%")
        
        return backtest_id, result
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise

async def get_backtest_results_example(backtest_id: str):
    """
    Example of getting backtest results using the standardized client.
    
    Args:
        backtest_id: ID of the backtest to retrieve
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the backtesting client
    client = factory.get_backtesting_client()
    
    try:
        # Get backtest results
        result = await client.get_backtest_results(
            backtest_id=backtest_id,
            include_trades=True,
            include_equity_curve=True
        )
        
        logger.info(f"Retrieved backtest results for {backtest_id}")
        
        # Extract key metrics
        strategy_id = result.get("strategy_id")
        final_capital = result.get("final_capital")
        trade_metrics = result.get("trade_metrics", {})
        performance_metrics = result.get("performance_metrics", {})
        
        logger.info(f"Strategy: {strategy_id}")
        logger.info(f"Final capital: ${final_capital:.2f}")
        logger.info(f"Total return: {performance_metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Sharpe ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max drawdown: {performance_metrics.get('max_drawdown_pct', 0):.2f}%")
        
        # Analyze trades
        trades = result.get("trades", [])
        if trades:
            profitable_trades = [t for t in trades if t.get("profit", 0) > 0]
            losing_trades = [t for t in trades if t.get("profit", 0) <= 0]
            
            logger.info(f"Analyzed {len(trades)} trades:")
            logger.info(f"  Profitable trades: {len(profitable_trades)}")
            logger.info(f"  Losing trades: {len(losing_trades)}")
            
            # Calculate average profit and loss
            if profitable_trades:
                avg_profit = sum(t.get("profit", 0) for t in profitable_trades) / len(profitable_trades)
                logger.info(f"  Average profit: ${avg_profit:.2f}")
            
            if losing_trades:
                avg_loss = sum(t.get("profit", 0) for t in losing_trades) / len(losing_trades)
                logger.info(f"  Average loss: ${avg_loss:.2f}")
        
        return result
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        raise

async def run_walk_forward_optimization_example():
    """
    Example of running walk-forward optimization using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the backtesting client
    client = factory.get_backtesting_client()
    
    # Example strategy configuration
    strategy_config = {
        "strategy_id": "moving_average_crossover",
        "parameters": {
            "fast_period": 10,
            "slow_period": 30
        }
    }
    
    # Example parameter ranges for optimization
    parameter_ranges = {
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 40, 50]
    }
    
    # Example optimization parameters
    start_date = datetime.utcnow() - timedelta(days=365)  # 1 year ago
    end_date = datetime.utcnow()
    instruments = ["EUR_USD"]
    
    try:
        # Run walk-forward optimization
        result = await client.run_walk_forward_optimization(
            strategy_config=strategy_config,
            parameter_ranges=parameter_ranges,
            start_date=start_date,
            end_date=end_date,
            instruments=instruments,
            initial_capital=10000.0,
            window_size_days=90,
            anchor_size_days=30,
            optimization_metric="sharpe_ratio",
            parallel_jobs=2
        )
        
        logger.info(f"Walk-forward optimization completed with ID: {result.get('optimization_id')}")
        
        # Extract key results
        optimization_id = result.get("optimization_id")
        optimal_parameters = result.get("optimal_parameters", {})
        in_sample_metrics = result.get("in_sample_metrics", {})
        out_of_sample_metrics = result.get("out_of_sample_metrics", {})
        robustness_score = result.get("robustness_score", 0)
        
        logger.info(f"Optimal parameters: {optimal_parameters}")
        logger.info(f"In-sample return: {in_sample_metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Out-of-sample return: {out_of_sample_metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Robustness score: {robustness_score:.2f}")
        
        return optimization_id, result
    except Exception as e:
        logger.error(f"Error running walk-forward optimization: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running Backtesting client examples")
    
    # Run backtest
    backtest_id, _ = await run_backtest_example()
    
    # Get backtest results
    await get_backtest_results_example(backtest_id)
    
    # Run walk-forward optimization
    await run_walk_forward_optimization_example()
    
    logger.info("Completed Backtesting client examples")

if __name__ == "__main__":
    asyncio.run(main())
