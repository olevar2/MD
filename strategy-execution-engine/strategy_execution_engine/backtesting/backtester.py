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

# Import error handling
from ..error import (
    BacktestConfigError,
    BacktestDataError,
    BacktestExecutionError,
    BacktestReportError,
    with_error_handling,
    async_with_error_handling
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

    @async_with_error_handling
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
            BacktestReportError: If report generation fails
        """
        logger.info(f"Starting backtest for config: {config_path}")

        # Validate inputs
        if not config_path:
            raise BacktestConfigError(
                message="Strategy configuration path cannot be empty",
                details={"config_path": config_path}
            )

        # Process dates
        try:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

            if start_date:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                # Use default lookback period if no start date provided
                start_date_obj = end_date_obj - timedelta(days=self.default_lookback_days)

            # Validate date range
            if start_date_obj >= end_date_obj:
                raise BacktestConfigError(
                    message="Start date must be before end date",
                    details={
                        "start_date": start_date,
                        "end_date": end_date
                    }
                )
        except ValueError as e:
            raise BacktestConfigError(
                message=f"Invalid date format: {str(e)}",
                details={
                    "start_date": start_date,
                    "end_date": end_date,
                    "expected_format": "YYYY-MM-DD"
                }
            )

        # Load strategy configuration
        strategy_config = self._load_strategy_config(config_path)

        # Get strategy ID for error reporting
        strategy_id = strategy_config.get("id", "unknown")

        # Determine assets to test if not explicitly specified
        if not assets and "instruments" in strategy_config:
            assets = strategy_config["instruments"]
        elif not assets:
            # Default to all forex majors if not specified
            assets = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]

        # Validate assets
        if not assets or not isinstance(assets, list) or not all(isinstance(a, str) for a in assets):
            raise BacktestConfigError(
                message="Invalid assets specification",
                strategy_id=strategy_id,
                details={"assets": assets}
            )

        # Get market data for the backtest period
        market_data = await self._get_market_data(assets, start_date_obj, end_date_obj)

        # Run the backtest
        try:
            backtest_result = await self.backtest_engine.execute(
                strategy_config=strategy_config,
                market_data=market_data,
                start_date=start_date_obj,
                end_date=end_date_obj
            )
        except Exception as e:
            raise BacktestExecutionError(
                message=f"Failed to execute backtest: {str(e)}",
                strategy_id=strategy_id,
                details={
                    "original_error": str(e),
                    "start_date": start_date_obj.isoformat(),
                    "end_date": end_date_obj.isoformat(),
                    "assets": assets
                }
            )

        # Add market regime information if requested
        if self.include_market_regimes:
            try:
                backtest_result = await self._add_market_regime_analysis(backtest_result, market_data, assets)
            except Exception as e:
                logger.warning(f"Failed to add market regime analysis: {e}")
                # Continue without market regime analysis

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(backtest_result)

        # Generate report if specified in config
        if self.config.get("generate_reports", False):
            try:
                report_path = await self.report_generator.generate_report(backtest_result, metrics)
                metrics["report_path"] = report_path
            except Exception as e:
                raise BacktestReportError(
                    message=f"Failed to generate backtest report: {str(e)}",
                    strategy_id=strategy_id,
                    details={"original_error": str(e)}
                )

        logger.info(f"Backtest completed for {config_path}")

        # Return standardized result format
        return {
            "success": True,
            "metadata": {
                "strategy_id": strategy_id,
                "version": strategy_config.get("version", "1.0"),
                "start_date": start_date_obj.isoformat(),
                "end_date": end_date_obj.isoformat(),
                "assets": assets,
                "execution_time": datetime.now().isoformat()
            },
            "metrics": metrics
        }

    @with_error_handling
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
            # Validate the dictionary has required fields
            if not self._validate_strategy_config(config_path):
                raise BacktestConfigError(
                    message="Invalid strategy configuration format",
                    details={"missing_fields": "Required fields missing in strategy configuration"}
                )
            return config_path

        if not isinstance(config_path, str):
            raise BacktestConfigError(
                message=f"Invalid config_path type: {type(config_path)}",
                details={"expected_types": ["str", "dict"], "received_type": str(type(config_path))}
            )

        # Check if file exists
        if not os.path.exists(config_path):
            raise BacktestConfigError(
                message=f"Strategy configuration file not found: {config_path}",
                details={"config_path": config_path}
            )

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Validate the loaded configuration
            if not self._validate_strategy_config(config):
                raise BacktestConfigError(
                    message="Invalid strategy configuration format",
                    config_name=os.path.basename(config_path),
                    details={"config_path": config_path, "missing_fields": "Required fields missing in strategy configuration"}
                )

            return config
        except IOError as e:
            raise BacktestConfigError(
                message=f"Failed to read strategy config file: {str(e)}",
                config_name=os.path.basename(config_path),
                details={"config_path": config_path, "error": str(e)}
            )
        except json.JSONDecodeError as e:
            raise BacktestConfigError(
                message=f"Invalid JSON in strategy config file: {str(e)}",
                config_name=os.path.basename(config_path),
                details={"config_path": config_path, "error": str(e), "line": e.lineno, "column": e.colno}
            )

    def _validate_strategy_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate that a strategy configuration has the required fields.

        Args:
            config: Strategy configuration to validate

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check for required fields
        required_fields = ["id", "name", "version"]
        for field in required_fields:
            if field not in config:
                logger.warning(f"Missing required field in strategy config: {field}")
                return False

        # Check for strategy type
        if "type" not in config:
            logger.warning("Missing strategy type in configuration")
            return False

        return True

    @async_with_error_handling
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
        # Validate inputs
        if not assets:
            raise BacktestDataError(
                message="No assets specified for market data retrieval",
                details={"assets": assets}
            )

        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise BacktestDataError(
                message="Invalid date types for market data retrieval",
                details={
                    "start_date_type": str(type(start_date)),
                    "end_date_type": str(type(end_date))
                }
            )

        if start_date >= end_date:
            raise BacktestDataError(
                message="Start date must be before end date for market data retrieval",
                details={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            )

        # This is a placeholder - in a real implementation, this would fetch data
        # from your data pipeline service, market data provider, or database
        market_data = {}
        failed_assets = []

        try:
            # TODO: Replace with actual data retrieval
            # Example implementation:
            # from ..integration.data_client import DataClient
            # data_client = DataClient()
            # for asset in assets:
            #     try:
            #         market_data[asset] = await data_client.get_historical_data(
            #             asset, start_date, end_date
            #         )
            #     except Exception as e:
            #         failed_assets.append({"asset": asset, "error": str(e)})

            # For now, generate some dummy data for the example
            for asset in assets:
                try:
                    # Create date range
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

                    # Check if we have enough data points
                    if len(date_range) < 2:
                        raise ValueError(f"Insufficient data points for asset {asset}: date range too small")

                    # Generate random price data
                    np.random.seed(42 + hash(asset) % 100)  # Different seed for each asset
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
                except Exception as e:
                    failed_assets.append({"asset": asset, "error": str(e)})

            # Check if we have any data
            if not market_data:
                if failed_assets:
                    raise BacktestDataError(
                        message="Failed to retrieve market data for all assets",
                        details={"failed_assets": failed_assets}
                    )
                else:
                    raise BacktestDataError(
                        message="No market data retrieved",
                        details={
                            "assets": assets,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat()
                        }
                    )

            # Log warnings for any failed assets
            if failed_assets:
                logger.warning(f"Failed to retrieve market data for some assets: {failed_assets}")

            return market_data

        except Exception as e:
            if isinstance(e, BacktestDataError):
                raise

            raise BacktestDataError(
                message=f"Failed to retrieve market data: {str(e)}",
                details={
                    "assets": assets,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "error": str(e)
                }
            )

    @async_with_error_handling
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
        # Validate inputs
        if not backtest_result:
            logger.warning("Empty backtest result provided for market regime analysis")
            return backtest_result

        if not market_data:
            logger.warning("No market data provided for market regime analysis")
            return backtest_result

        if not assets:
            logger.warning("No assets provided for market regime analysis")
            return backtest_result

        try:
            # TODO: Replace with actual market regime analysis
            # This is a placeholder implementation

            # Calculate regime percentages based on price movements
            regime_percentages = {"trend": 0.0, "mean_reverting": 0.0, "volatile": 0.0}
            asset_regimes = {}

            for asset, data in market_data.items():
                if asset not in assets:
                    continue

                if len(data) < 10:  # Need enough data points for analysis
                    logger.warning(f"Insufficient data for market regime analysis of {asset}")
                    continue

                # Calculate returns
                returns = data['close'].pct_change().dropna()

                if len(returns) == 0:
                    continue

                # Calculate metrics
                volatility = returns.std()
                autocorrelation = returns.autocorr(lag=1)
                abs_mean_return = abs(returns.mean())

                # Determine regime based on metrics
                if volatility > 0.015:  # High volatility
                    regime = "volatile"
                    confidence = min(volatility * 50, 1.0)  # Scale volatility to confidence
                elif abs_mean_return > 0.003 and abs(autocorrelation) < 0.2:  # Strong trend, low mean reversion
                    regime = "trend"
                    confidence = min(abs_mean_return * 200, 1.0)  # Scale mean return to confidence
                else:  # Low volatility, high mean reversion
                    regime = "mean_reverting"
                    confidence = min(abs(autocorrelation) * 2, 1.0)  # Scale autocorrelation to confidence

                # Store regime for this asset
                asset_regimes[asset] = {
                    "regime": regime,
                    "confidence": float(confidence),
                    "metrics": {
                        "volatility": float(volatility),
                        "autocorrelation": float(autocorrelation),
                        "mean_return": float(returns.mean())
                    }
                }

                # Update overall percentages
                regime_percentages[regime] += 1

            # Normalize percentages
            total_assets = len(asset_regimes)
            if total_assets > 0:
                for regime in regime_percentages:
                    regime_percentages[regime] = regime_percentages[regime] / total_assets
            else:
                # Default values if no assets could be analyzed
                regime_percentages = {
                    "trend": 0.6,          # 60% trending
                    "mean_reverting": 0.3,  # 30% mean reverting
                    "volatile": 0.1         # 10% volatile
                }

            # Add to backtest result
            backtest_result["market_regimes"] = {
                "overall": regime_percentages,
                "assets": asset_regimes
            }

            return backtest_result
        except Exception as e:
            logger.warning(f"Could not add market regime analysis: {e}")
            # Don't let market regime analysis failure stop the backtest
            return backtest_result

    @with_error_handling
    def _calculate_performance_metrics(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from backtest results.

        Args:
            backtest_result: Raw backtest results

        Returns:
            Dict: Performance metrics including Sharpe, drawdown, etc.

        Raises:
            BacktestExecutionError: If metrics calculation fails
        """
        # Validate input
        if not backtest_result:
            logger.warning("Empty backtest result provided for metrics calculation")
            return self._generate_placeholder_metrics()

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
            try:
                equity_values = np.array([point.get("equity", 0) for point in equity_curve])
            except Exception as e:
                raise BacktestExecutionError(
                    message=f"Failed to process equity curve: {str(e)}",
                    details={"error": str(e)}
                )
        else:
            equity_values = equity_curve

        # Validate equity values
        if len(equity_values) < 2:
            logger.warning("Insufficient equity data points for metrics calculation")
            return self._generate_placeholder_metrics()

        # Calculate returns
        try:
            returns = np.diff(equity_values) / equity_values[:-1]
        except Exception as e:
            raise BacktestExecutionError(
                message=f"Failed to calculate returns: {str(e)}",
                details={"error": str(e)}
            )

        # Calculate metrics
        try:
            # Sharpe ratio
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0

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

            # Calculate additional metrics
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
            expectancy = avg_profit * win_rate + avg_loss * (1 - win_rate)
            net_profit = gross_profit - gross_loss

            # Calculate drawdown duration
            drawdown_duration = 0
            max_drawdown_duration = 0
            in_drawdown = False

            for i in range(1, len(equity_values)):
                if equity_values[i] < peak[i-1]:
                    if not in_drawdown:
                        in_drawdown = True
                        drawdown_duration = 1
                    else:
                        drawdown_duration += 1
                else:
                    in_drawdown = False
                    max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
                    drawdown_duration = 0

            # Include max drawdown duration in metrics
            max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)

            # Return calculated metrics
            return {
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "max_drawdown_duration": int(max_drawdown_duration),
                "win_rate": float(win_rate),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "profit_factor": float(profit_factor),
                "avg_profit": float(avg_profit),
                "avg_loss": float(avg_loss),
                "expectancy": float(expectancy),
                "net_profit": float(net_profit),
                "volatility": float(volatility),
                "calmar_ratio": float(net_profit / max_drawdown) if max_drawdown > 0 else float('inf'),
                "avg_trade": float((gross_profit - gross_loss) / total_trades) if total_trades > 0 else 0
            }
        except Exception as e:
            raise BacktestExecutionError(
                message=f"Error calculating performance metrics: {str(e)}",
                details={"error": str(e)}
            )

    @with_error_handling
    def _generate_placeholder_metrics(self) -> Dict[str, Any]:
        """
        Generate placeholder metrics when calculation fails.

        Returns:
            Dict: Default performance metrics
        """
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 100.0,
            "max_drawdown_duration": 0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "net_profit": 0.0,
            "volatility": 0.0,
            "calmar_ratio": 0.0,
            "avg_trade": 0.0,
            "_placeholder": True  # Flag to indicate these are placeholder metrics
        }


# Create a singleton instance for easier imports
backtester = Backtester()


@async_with_error_handling
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

    Raises:
        BacktestConfigError: If strategy configuration is invalid
        BacktestDataError: If data is missing or invalid
        BacktestExecutionError: If backtest execution fails
        BacktestReportError: If report generation fails
    """
    # Validate inputs
    if not config_path:
        raise BacktestConfigError(
            message="Strategy configuration path cannot be empty",
            details={"config_path": config_path}
        )

    try:
        if custom_config:
            bt = Backtester(config=custom_config)
            return await bt.run_backtest(config_path, start_date, end_date, assets)
        else:
            return await backtester.run_backtest(config_path, start_date, end_date, assets)
    except Exception as e:
        # If it's already a BacktestError, just re-raise it
        if isinstance(e, (BacktestConfigError, BacktestDataError, BacktestExecutionError, BacktestReportError)):
            raise

        # Otherwise, wrap it in a BacktestExecutionError
        raise BacktestExecutionError(
            message=f"Unexpected error during backtest: {str(e)}",
            details={"error": str(e)}
        )
