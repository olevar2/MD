"""
Automated Backtester

This module provides a high-level interface for running strategy backtests
and standardizes the format of backtest results for use in automated
deployment workflows.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.backtest_exceptions import (
    BacktestConfigError,
    BacktestDataError,
    BacktestExecutionError
)

# Import local modules
from .backtest_engine import BacktestEngine
from .reporting import BacktestReportGenerator
from ..multi_asset.asset_registry import AssetRegistry

logger = get_logger(__name__)


class Backtester:
    """
    High-level backtesting service that coordinates the execution of backtests
    and standardizes their results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtester with configuration.
        
        Args:
            config: Configuration dictionary for the backtester
        """
        self.config = config or {}
        self.backtest_engine = BacktestEngine()
        self.report_generator = BacktestReportGenerator()
        self.asset_registry = AssetRegistry()
        
        # Default configuration
        self.default_lookback_days = self.config.get("default_lookback_days", 365)
        self.include_market_regimes = self.config.get("include_market_regimes", True)
        self.data_source = self.config.get("data_source", "default")
        self.temp_dir = self.config.get("temp_dir", "tmp")
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info("Backtester initialized")
    
    async def run_backtest(self, 
                    config_path: str, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a backtest for a strategy configuration over the specified date range.
        
        Args:
            config_path: Path to the strategy configuration file or config object
            start_date: Start date for backtest in YYYY-MM-DD format
            end_date: End date for backtest in YYYY-MM-DD format
            assets: Optional list of asset symbols to include in backtest
            
        Returns:
            Dict: Standardized backtest results including performance metrics
            
        Raises:
            BacktestConfigError: If strategy configuration is invalid
            BacktestDataError: If data is missing or invalid
            BacktestExecutionError: If backtest execution fails
        """
        logger.info(f"Starting backtest for config: {config_path}")
        
        try:
            # Process dates
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
            
            if start_date:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                # Use default lookback period if no start date provided
                start_date_obj = end_date_obj - timedelta(days=self.default_lookback_days)
            
            # Load strategy configuration
            strategy_config = self._load_strategy_config(config_path)
            
            # Determine assets to test if not explicitly specified
            if not assets and "instruments" in strategy_config:
                assets = strategy_config["instruments"]
            elif not assets:
                # Default to all forex majors if not specified
                assets = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
            
            # Get market data for the backtest period
            market_data = await self._get_market_data(assets, start_date_obj, end_date_obj)
            
            # Run the backtest
            backtest_result = await self.backtest_engine.execute(
                strategy_config=strategy_config,
                market_data=market_data,
                start_date=start_date_obj,
                end_date=end_date_obj
            )
            
            # Add market regime information if requested
            if self.include_market_regimes:
                backtest_result = await self._add_market_regime_analysis(backtest_result, market_data, assets)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(backtest_result)
            
            # Generate report if specified in config
            if self.config.get("generate_reports", False):
                report_path = await self.report_generator.generate_report(backtest_result, metrics)
                metrics["report_path"] = report_path
            
            logger.info(f"Backtest completed for {config_path}")
            
            # Return standardized result format
            return {
                "success": True,
                "metadata": {
                    "strategy_id": strategy_config.get("id", "unknown"),
                    "version": strategy_config.get("version", "1.0"),
                    "start_date": start_date_obj.isoformat(),
                    "end_date": end_date_obj.isoformat(),
                    "assets": assets,
                    "execution_time": datetime.now().isoformat()
                },
                "metrics": metrics
            }
            
        except BacktestConfigError as e:
            logger.error(f"Invalid strategy configuration: {e}")
            return {"success": False, "error": f"Configuration error: {str(e)}"}
        
        except BacktestDataError as e:
            logger.error(f"Data error during backtest: {e}")
            return {"success": False, "error": f"Data error: {str(e)}"}
        
        except BacktestExecutionError as e:
            logger.error(f"Error executing backtest: {e}")
            return {"success": False, "error": f"Execution error: {str(e)}"}
        
        except Exception as e:
            logger.error(f"Unexpected error during backtest: {e}", exc_info=True)
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def _load_strategy_config(self, config_path: Union[str, Dict]) -> Dict[str, Any]:
        """
        Load strategy configuration from a file path or use directly if dict.
        
        Args:
            config_path: Path to JSON config file or config dictionary
            
        Returns:
            Dict: Strategy configuration
            
        Raises:
            BacktestConfigError: If config cannot be loaded or is invalid
        """
        if isinstance(config_path, dict):
            return config_path
        
        try:
            if isinstance(config_path, str):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config
        except (IOError, json.JSONDecodeError) as e:
            raise BacktestConfigError(f"Failed to load strategy config: {e}")
        
        raise BacktestConfigError(f"Invalid config_path type: {type(config_path)}")
    
    async def _get_market_data(self, 
                       assets: List[str], 
                       start_date: datetime, 
                       end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Retrieve historical market data for the specified assets and date range.
        
        Args:
            assets: List of asset symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dict: Mapping of asset symbols to pandas DataFrames with market data
            
        Raises:
            BacktestDataError: If data cannot be retrieved
        """
        # This is a placeholder - in a real implementation, this would fetch data
        # from your data pipeline service, market data provider, or database
        
        market_data = {}
        
        try:
            # TODO: Replace with actual data retrieval
            # Example implementation:
            # from ..integration.data_client import DataClient
            # data_client = DataClient()
            # for asset in assets:
            #     market_data[asset] = await data_client.get_historical_data(
            #         asset, start_date, end_date
            #     )
            
            # For now, generate some dummy data for the example
            for asset in assets:
                # Create date range
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Generate random price data
                np.random.seed(42)  # For reproducibility
                n = len(date_range)
                close_prices = np.random.random(n) * 10 + 100  # Random prices around 100
                
                # Add some trend to make it more realistic
                trend = np.linspace(0, 5, n)
                close_prices = close_prices + trend
                
                # Calculate OHLC from close
                open_prices = close_prices * np.random.uniform(0.99, 1.01, n)
                high_prices = np.maximum(close_prices, open_prices) * np.random.uniform(1.001, 1.02, n)
                low_prices = np.minimum(close_prices, open_prices) * np.random.uniform(0.98, 0.999, n)
                volumes = np.random.random(n) * 1000000 + 100000
                
                # Create DataFrame
                df = pd.DataFrame({
                    'open': open_prices,
                    'high': high_prices,
                    'low': low_prices,
                    'close': close_prices,
                    'volume': volumes
                }, index=date_range)
                
                market_data[asset] = df
            
            return market_data
            
        except Exception as e:
            raise BacktestDataError(f"Failed to retrieve market data: {e}")
    
    async def _add_market_regime_analysis(self, 
                                 backtest_result: Dict[str, Any], 
                                 market_data: Dict[str, pd.DataFrame],
                                 assets: List[str]) -> Dict[str, Any]:
        """
        Enhance backtest results with market regime analysis.
        
        Args:
            backtest_result: Backtest execution results
            market_data: Market data used for the backtest
            assets: List of asset symbols
            
        Returns:
            Dict: Enhanced backtest results with market regime information
        """
        try:
            # TODO: Replace with actual market regime analysis
            # This is a placeholder implementation
            backtest_result["market_regimes"] = {
                "trend": 0.6,          # 60% trending
                "mean_reverting": 0.3,  # 30% mean reverting
                "volatile": 0.1         # 10% volatile
            }
            
            return backtest_result
        except Exception as e:
            logger.warning(f"Could not add market regime analysis: {e}")
            return backtest_result
    
    def _calculate_performance_metrics(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from backtest results.
        
        Args:
            backtest_result: Raw backtest results
            
        Returns:
            Dict: Performance metrics including Sharpe, drawdown, etc.
        """
        try:
            # Extract trades and equity curve from results
            trades = backtest_result.get("trades", [])
            equity_curve = backtest_result.get("equity_curve", [])
            
            # If no trades or equity data, return placeholder metrics
            if not trades or not equity_curve:
                logger.warning("No trades or equity curve data found in backtest results")
                return self._generate_placeholder_metrics()
            
            # Calculate actual metrics based on backtest data
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Convert equity curve to numpy array if it's a list
            if isinstance(equity_curve, list):
                equity_values = np.array([point.get("equity", 0) for point in equity_curve])
            else:
                equity_values = equity_curve
                
            # Calculate returns
            returns = np.diff(equity_values) / equity_values[:-1]
            
            # Calculate metrics
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
            
            # Max drawdown
            peak = np.maximum.accumulate(equity_values)
            drawdown = (peak - equity_values) / peak
            max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
            
            # Calculate profit metrics
            profit_trades = [trade for trade in trades if trade.get("pnl", 0) > 0]
            loss_trades = [trade for trade in trades if trade.get("pnl", 0) < 0]
            
            avg_profit = sum(trade.get("pnl", 0) for trade in profit_trades) / len(profit_trades) if profit_trades else 0
            avg_loss = sum(trade.get("pnl", 0) for trade in loss_trades) / len(loss_trades) if loss_trades else 0
            
            # Profit factor
            gross_profit = sum(trade.get("pnl", 0) for trade in profit_trades)
            gross_loss = abs(sum(trade.get("pnl", 0) for trade in loss_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Return calculated metrics
            return {
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "total_trades": total_trades,
                "profit_factor": float(profit_factor),
                "avg_profit": float(avg_profit),
                "avg_loss": float(avg_loss),
                "expectancy": float(avg_profit * win_rate + avg_loss * (1 - win_rate)),
                "net_profit": float(gross_profit - gross_loss),
                "volatility": float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            return self._generate_placeholder_metrics()
    
    def _generate_placeholder_metrics(self) -> Dict[str, Any]:
        """
        Generate placeholder metrics when calculation fails.
        
        Returns:
            Dict: Default performance metrics
        """
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 100.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "net_profit": 0.0,
            "volatility": 0.0
        }


# Create a singleton instance for easier imports
backtester = Backtester()


async def run_backtest(config_path: str, 
                start_date: Optional[str] = None, 
                end_date: Optional[str] = None,
                assets: Optional[List[str]] = None,
                custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to run a backtest without directly instantiating the Backtester.
    
    Args:
        config_path: Path to strategy configuration file or config dictionary
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        assets: Optional list of asset symbols to test
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Dict: Standardized backtest results
    """
    if custom_config:
        bt = Backtester(config=custom_config)
        return await bt.run_backtest(config_path, start_date, end_date, assets)
    else:
        return await backtester.run_backtest(config_path, start_date, end_date, assets)
