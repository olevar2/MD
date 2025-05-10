"""
Backtesting Engine for Forex Trading Strategies

This module provides a backtesting engine for evaluating trading strategies
with historical data, including technical indicator effectiveness tracking.
"""
import pandas as pd
import numpy as np
import uuid
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable

# Import error handling
from ..error import (
    BacktestConfigError,
    BacktestDataError,
    BacktestExecutionError,
    BacktestReportError,
    with_error_handling,
    async_with_error_handling
)

from .tool_effectiveness_evaluator import BacktestToolEffectivenessEvaluator
from .reporting import BacktestReport

class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical forex data

    The backtesting engine simulates the execution of trading strategies on
    historical data and provides performance metrics and analysis tools.
    It includes support for evaluating the effectiveness of technical analysis tools.
    """

    def __init__(
        self,
        data: pd.DataFrame = None,
        initial_balance: float = 10000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        spread: float = 0.0,
        track_tool_effectiveness: bool = True,
        backtest_id: Optional[str] = None
    ):
        """
        Initialize the backtesting engine

        Args:
            data: Historical price data as pandas DataFrame
            initial_balance: Starting account balance
            commission: Commission per trade (percentage)
            slippage: Slippage per trade (pips)
            spread: Spread in pips
            track_tool_effectiveness: Whether to track tool effectiveness
            backtest_id: Optional ID for the backtest, auto-generated if None
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.spread = spread

        # Generate or set backtest ID
        self.backtest_id = backtest_id or f"backtest_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create logger
        self.logger = logging.getLogger(f"backtest_engine.{self.backtest_id}")

        # Set data if provided
        self.data = data

        # Trading state
        self.positions = []
        self.closed_positions = []
        self.orders = []
        self.balance_history = []
        self.equity_history = []

        # Performance metrics
        self.metrics = {}

        # Tool effectiveness tracking
        self.track_tool_effectiveness = track_tool_effectiveness
        if track_tool_effectiveness:
            self.tool_evaluator = BacktestToolEffectivenessEvaluator(self.backtest_id)
        else:
            self.tool_evaluator = None

        # Output directory
        self.output_dir = os.path.join("output", "backtests", self.backtest_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def set_data(self, data: pd.DataFrame):
        """
        Set historical price data for backtesting

        Args:
            data: DataFrame with historical price data
        """
        self.data = data

    @with_error_handling
    def run_strategy(self, strategy_func: Callable, **strategy_params):
        """
        Run a trading strategy on the historical data

        Args:
            strategy_func: Strategy function to execute
            strategy_params: Additional parameters for the strategy

        Returns:
            Dict with backtest results

        Raises:
            BacktestDataError: If no data is provided for backtesting
            BacktestExecutionError: If strategy execution fails
            BacktestReportError: If report generation fails
        """
        if self.data is None:
            raise BacktestDataError(
                message="No data provided for backtesting",
                details={"backtest_id": self.backtest_id}
            )

        if not callable(strategy_func):
            raise BacktestConfigError(
                message="Strategy function is not callable",
                details={"backtest_id": self.backtest_id}
            )

        # Reset state
        self.balance = self.initial_balance
        self.positions = []
        self.closed_positions = []
        self.orders = []
        self.balance_history = []
        self.equity_history = []

        try:
            # Execute strategy
            strategy_results = strategy_func(
                data=self.data,
                engine=self,
                **strategy_params
            )

            # Calculate performance metrics
            self._calculate_metrics()

            # Evaluate tool effectiveness if needed
            if self.track_tool_effectiveness and self.tool_evaluator:
                try:
                    # Automatically evaluate any pending signals
                    evaluated_count = self.tool_evaluator.bulk_evaluate_pending_signals(
                        price_data=self.data,
                        evaluation_params=strategy_params.get("effectiveness_params", None)
                    )
                    self.logger.info(f"Evaluated {evaluated_count} pending tool signals")

                    # Generate effectiveness report
                    effectiveness_report_path = self.tool_evaluator.export_report(
                        output_format="json"
                    )
                    self.logger.info(f"Exported tool effectiveness report to {effectiveness_report_path}")
                except Exception as e:
                    self.logger.warning(f"Error evaluating tool effectiveness: {str(e)}")
                    # Continue without tool effectiveness evaluation

            # Save backtest results
            try:
                results_path = self._save_results()
            except Exception as e:
                raise BacktestReportError(
                    message=f"Failed to save backtest results: {str(e)}",
                    backtest_id=self.backtest_id,
                    details={"error": str(e)}
                )

            return {
                "backtest_id": self.backtest_id,
                "metrics": self.metrics,
                "results_path": results_path,
                "strategy_results": strategy_results,
                "success": True
            }

        except Exception as e:
            if isinstance(e, (BacktestConfigError, BacktestDataError, BacktestExecutionError, BacktestReportError)):
                raise

            raise BacktestExecutionError(
                message=f"Error running strategy: {str(e)}",
                backtest_id=self.backtest_id,
                details={"error": str(e)}
            )

    @with_error_handling
    def _calculate_metrics(self):
        """
        Calculate performance metrics for the backtest

        Raises:
            BacktestExecutionError: If metrics calculation fails
        """
        try:
            # Calculate basic metrics
            total_trades = len(self.closed_positions)
            winning_trades = sum(1 for p in self.closed_positions if p.get("profit", 0) > 0)
            losing_trades = sum(1 for p in self.closed_positions if p.get("profit", 0) <= 0)

            # Avoid division by zero
            win_rate = winning_trades / max(1, total_trades)

            # Calculate profit metrics
            profit_loss = self.balance - self.initial_balance
            return_pct = (self.balance / self.initial_balance - 1) * 100 if self.initial_balance > 0 else 0

            # Calculate advanced metrics if we have trades
            if total_trades > 0:
                # Average profit and loss
                profit_trades = [p for p in self.closed_positions if p.get("profit", 0) > 0]
                loss_trades = [p for p in self.closed_positions if p.get("profit", 0) <= 0]

                avg_profit = sum(p.get("profit", 0) for p in profit_trades) / max(1, len(profit_trades))
                avg_loss = sum(p.get("profit", 0) for p in loss_trades) / max(1, len(loss_trades))

                # Profit factor
                gross_profit = sum(p.get("profit", 0) for p in profit_trades)
                gross_loss = abs(sum(p.get("profit", 0) for p in loss_trades))
                profit_factor = gross_profit / max(0.01, gross_loss)  # Avoid division by zero

                # Expectancy
                expectancy = (avg_profit * win_rate) + (avg_loss * (1 - win_rate))

                # Store advanced metrics
                self.metrics = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "final_balance": self.balance,
                    "profit_loss": profit_loss,
                    "return_pct": return_pct,
                    "avg_profit": avg_profit,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor,
                    "expectancy": expectancy,
                    "gross_profit": gross_profit,
                    "gross_loss": gross_loss
                }
            else:
                # Basic metrics if no trades
                self.metrics = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "final_balance": self.balance,
                    "profit_loss": profit_loss,
                    "return_pct": return_pct
                }
        except Exception as e:
            raise BacktestExecutionError(
                message=f"Failed to calculate performance metrics: {str(e)}",
                backtest_id=self.backtest_id,
                details={"error": str(e)}
            )

    @with_error_handling
    def _save_results(self) -> str:
        """
        Save backtest results to file

        Returns:
            Path to the saved results file

        Raises:
            BacktestReportError: If saving results fails
        """
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)

            # Prepare results data
            results = {
                "backtest_id": self.backtest_id,
                "timestamp": datetime.now().isoformat(),
                "initial_balance": self.initial_balance,
                "final_balance": self.balance,
                "parameters": {
                    "commission": self.commission,
                    "slippage": self.slippage,
                    "spread": self.spread
                },
                "metrics": self.metrics,
                "closed_positions": self.closed_positions
            }

            # Define file path
            file_path = os.path.join(self.output_dir, "results.json")

            # Write to file
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)

            return file_path
        except Exception as e:
            raise BacktestReportError(
                message=f"Failed to save backtest results: {str(e)}",
                backtest_id=self.backtest_id,
                details={"error": str(e), "output_dir": self.output_dir}
            )

    # Trading methods
    @with_error_handling
    def open_position(self, timestamp, symbol, direction, size, price=None, stop_loss=None, take_profit=None, metadata=None):
        """
        Open a new trading position

        Args:
            timestamp: Time of position opening
            symbol: Trading symbol
            direction: Trade direction ('buy' or 'sell')
            size: Position size
            price: Entry price (optional)
            stop_loss: Stop loss level (optional)
            take_profit: Take profit level (optional)
            metadata: Additional position metadata (optional)

        Returns:
            Position ID

        Raises:
            BacktestExecutionError: If position opening fails
        """
        # Validate inputs
        if not symbol:
            raise BacktestExecutionError(
                message="Symbol cannot be empty when opening a position",
                backtest_id=self.backtest_id,
                details={"timestamp": timestamp}
            )

        if direction not in ['buy', 'sell']:
            raise BacktestExecutionError(
                message=f"Invalid direction: {direction}. Must be 'buy' or 'sell'",
                backtest_id=self.backtest_id,
                details={"symbol": symbol, "timestamp": timestamp}
            )

        if size <= 0:
            raise BacktestExecutionError(
                message=f"Position size must be positive: {size}",
                backtest_id=self.backtest_id,
                details={"symbol": symbol, "timestamp": timestamp}
            )

        try:
            # Create position object
            position = {
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "direction": direction,
                "size": size,
                "entry_time": timestamp,
                "entry_price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "status": "open",
                "metadata": metadata or {}
            }

            # Add to positions list
            self.positions.append(position)

            # Log position opening
            self.logger.info(f"Opened {direction} position for {symbol} at {price}, size: {size}")

            return position["id"]
        except Exception as e:
            raise BacktestExecutionError(
                message=f"Failed to open position: {str(e)}",
                backtest_id=self.backtest_id,
                details={
                    "symbol": symbol,
                    "direction": direction,
                    "size": size,
                    "timestamp": timestamp,
                    "error": str(e)
                }
            )

    @with_error_handling
    def close_position(self, position_id, timestamp, price=None, reason=None):
        """
        Close an existing position

        Args:
            position_id: ID of the position to close
            timestamp: Time of position closing
            price: Exit price (optional)
            reason: Reason for closing (optional)

        Returns:
            Closed position details or None if position not found

        Raises:
            BacktestExecutionError: If position closing fails
        """
        # Validate inputs
        if not position_id:
            raise BacktestExecutionError(
                message="Position ID cannot be empty when closing a position",
                backtest_id=self.backtest_id,
                details={"timestamp": timestamp}
            )

        # Find position
        position = next((p for p in self.positions if p["id"] == position_id), None)
        if not position:
            self.logger.warning(f"Position not found for closing: {position_id}")
            return None

        try:
            # Calculate profit
            direction = position["direction"]
            entry_price = position["entry_price"]
            exit_price = price
            size = position["size"]

            if direction == "buy":
                profit = (exit_price - entry_price) * size
            else:
                profit = (entry_price - exit_price) * size

            # Apply commission
            profit -= (entry_price + exit_price) * size * self.commission

            # Update position
            position["exit_time"] = timestamp
            position["exit_price"] = exit_price
            position["profit"] = profit
            position["status"] = "closed"
            position["close_reason"] = reason

            # Move to closed positions
            self.positions = [p for p in self.positions if p["id"] != position_id]
            self.closed_positions.append(position)

            # Update balance
            self.balance += profit

            # Log position closing
            self.logger.info(f"Closed {position['direction']} position for {position['symbol']} at {exit_price}, profit: {profit}")

            return position
        except Exception as e:
            raise BacktestExecutionError(
                message=f"Failed to close position: {str(e)}",
                backtest_id=self.backtest_id,
                details={
                    "position_id": position_id,
                    "timestamp": timestamp,
                    "error": str(e)
                }
            )

    # Tool effectiveness tracking methods
    @with_error_handling
    def register_tool_signal(
        self,
        tool_name: str,
        signal_type: str,
        direction: str,
        strength: float,
        timestamp: datetime,
        symbol: str,
        timeframe: str,
        price: float,
        market_regime: str = "unknown",
        metadata: Dict[str, Any] = None,
        internal_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Register a signal from a technical analysis tool

        Args:
            tool_name: Name of the tool generating the signal
            signal_type: Type of signal (entry, exit, alert, etc.)
            direction: Signal direction (buy, sell, neutral)
            strength: Signal strength (0.0 to 1.0)
            timestamp: Timestamp of the signal
            symbol: Trading symbol
            timeframe: Timeframe of the analysis
            price: Price at the time of the signal
            market_regime: Market regime at the time of the signal
            metadata: Additional tool-specific data
            internal_id: Optional internal ID to associate with the signal

        Returns:
            Signal ID if registered successfully, None otherwise

        Raises:
            BacktestExecutionError: If signal registration fails
        """
        if not self.track_tool_effectiveness or not self.tool_evaluator:
            return None

        # Validate inputs
        if not tool_name:
            raise BacktestExecutionError(
                message="Tool name cannot be empty when registering a signal",
                backtest_id=self.backtest_id,
                details={"symbol": symbol, "timestamp": timestamp}
            )

        if not symbol:
            raise BacktestExecutionError(
                message="Symbol cannot be empty when registering a signal",
                backtest_id=self.backtest_id,
                details={"tool_name": tool_name, "timestamp": timestamp}
            )

        if direction not in ['buy', 'sell', 'neutral']:
            raise BacktestExecutionError(
                message=f"Invalid direction: {direction}. Must be 'buy', 'sell', or 'neutral'",
                backtest_id=self.backtest_id,
                details={"tool_name": tool_name, "symbol": symbol, "timestamp": timestamp}
            )

        if not (0.0 <= strength <= 1.0):
            raise BacktestExecutionError(
                message=f"Signal strength must be between 0.0 and 1.0: {strength}",
                backtest_id=self.backtest_id,
                details={"tool_name": tool_name, "symbol": symbol, "timestamp": timestamp}
            )

        try:
            signal_id = self.tool_evaluator.register_signal(
                tool_name=tool_name,
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                timestamp=timestamp,
                symbol=symbol,
                timeframe=timeframe,
                price=price,
                market_regime=market_regime,
                metadata=metadata,
                internal_id=internal_id
            )

            self.logger.debug(f"Registered {direction} signal from {tool_name} for {symbol} at {timestamp}")
            return signal_id
        except Exception as e:
            self.logger.error(f"Error registering tool signal: {str(e)}")
            # Don't raise exception for tool effectiveness tracking failures
            # as they shouldn't stop the backtest
            return None

    @with_error_handling
    def register_tool_outcome(
        self,
        signal_id: str,
        outcome: str,
        exit_price: Optional[float] = None,
        exit_timestamp: Optional[datetime] = None,
        max_favorable_price: Optional[float] = None,
        max_adverse_price: Optional[float] = None,
        profit_loss: Optional[float] = None,
        notes: str = "",
        internal_id: Optional[str] = None
    ) -> bool:
        """
        Register the outcome of a previously registered signal

        Args:
            signal_id: ID from the effectiveness tracker, or internal ID if mapped
            outcome: "success", "failure", or "undetermined"
            exit_price: Price at exit
            exit_timestamp: Timestamp at exit
            max_favorable_price: Most favorable price reached
            max_adverse_price: Most adverse price reached
            profit_loss: P&L if the signal was traded
            notes: Additional notes on the outcome
            internal_id: Optional internal ID used by the backtester

        Returns:
            True if outcome was registered successfully, False otherwise

        Raises:
            BacktestExecutionError: If outcome registration fails
        """
        if not self.track_tool_effectiveness or not self.tool_evaluator:
            return False

        # Validate inputs
        if not signal_id and not internal_id:
            raise BacktestExecutionError(
                message="Either signal_id or internal_id must be provided when registering an outcome",
                backtest_id=self.backtest_id
            )

        if outcome not in ['success', 'failure', 'undetermined']:
            raise BacktestExecutionError(
                message=f"Invalid outcome: {outcome}. Must be 'success', 'failure', or 'undetermined'",
                backtest_id=self.backtest_id,
                details={"signal_id": signal_id, "internal_id": internal_id}
            )

        try:
            result = self.tool_evaluator.register_outcome(
                signal_id=signal_id,
                outcome=outcome,
                exit_price=exit_price,
                exit_timestamp=exit_timestamp,
                max_favorable_price=max_favorable_price,
                max_adverse_price=max_adverse_price,
                profit_loss=profit_loss,
                notes=notes,
                internal_id=internal_id
            )

            self.logger.debug(f"Registered {outcome} outcome for signal {signal_id or internal_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error registering tool outcome: {str(e)}")
            # Don't raise exception for tool effectiveness tracking failures
            # as they shouldn't stop the backtest
            return False

    @with_error_handling
    def get_tool_effectiveness_metrics(
        self,
        tool_name: Optional[str] = None,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness metrics for technical analysis tools

        Args:
            tool_name: Name of tool to evaluate, or None for all tools
            market_regime: Optional filter by market regime
            timeframe: Optional filter by timeframe
            symbol: Optional filter by symbol

        Returns:
            Dictionary with effectiveness metrics

        Raises:
            BacktestExecutionError: If metrics calculation fails
        """
        if not self.track_tool_effectiveness or not self.tool_evaluator:
            return {}

        try:
            metrics = self.tool_evaluator.calculate_tool_metrics(
                tool_name=tool_name,
                market_regime=market_regime,
                timeframe=timeframe,
                symbol=symbol
            )

            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating tool metrics: {str(e)}")
            # Don't raise exception for tool effectiveness tracking failures
            # as they shouldn't stop the backtest
            return {
                "error": str(e),
                "tool_name": tool_name,
                "market_regime": market_regime,
                "timeframe": timeframe,
                "symbol": symbol
            }

    # Advanced reporting methods
    @with_error_handling
    def generate_performance_report(self, include_trades: bool = True,
                                   include_drawdowns: bool = True,
                                   include_metrics: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report with attribution analysis

        Args:
            include_trades: Include detailed trade analysis
            include_drawdowns: Include drawdown analysis
            include_metrics: Include detailed performance metrics

        Returns:
            Dictionary containing performance report data

        Raises:
            BacktestReportError: If report generation fails
        """
        try:
            report = BacktestReport(self)
            return report.generate_performance_report(
                include_trades=include_trades,
                include_drawdowns=include_drawdowns,
                include_metrics=include_metrics
            )
        except Exception as e:
            raise BacktestReportError(
                message=f"Failed to generate performance report: {str(e)}",
                backtest_id=self.backtest_id,
                details={"error": str(e)}
            )

    @with_error_handling
    def create_interactive_dashboard(self, report_data: Dict[str, Any] = None) -> str:
        """
        Create an interactive HTML dashboard for backtest results

        Args:
            report_data: Report data (generated if not provided)

        Returns:
            Path to the generated HTML dashboard

        Raises:
            BacktestReportError: If dashboard creation fails
        """
        try:
            report = BacktestReport(self)
            return report.create_interactive_dashboard(report_data)
        except Exception as e:
            raise BacktestReportError(
                message=f"Failed to create interactive dashboard: {str(e)}",
                backtest_id=self.backtest_id,
                report_type="html",
                details={"error": str(e)}
            )

    @with_error_handling
    def export_pdf_report(self, report_data: Dict[str, Any] = None) -> str:
        """
        Export backtest results to a PDF report

        Args:
            report_data: Report data (generated if not provided)

        Returns:
            Path to the exported PDF report

        Raises:
            BacktestReportError: If PDF export fails
        """
        try:
            report = BacktestReport(self)
            return report.export_pdf_report(report_data)
        except Exception as e:
            raise BacktestReportError(
                message=f"Failed to export PDF report: {str(e)}",
                backtest_id=self.backtest_id,
                report_type="pdf",
                details={"error": str(e)}
            )

    @with_error_handling
    def export_excel_report(self, report_data: Dict[str, Any] = None) -> str:
        """
        Export backtest results to an Excel report

        Args:
            report_data: Report data (generated if not provided)

        Returns:
            Path to the exported Excel report

        Raises:
            BacktestReportError: If Excel export fails
        """
        try:
            report = BacktestReport(self)
            return report.export_excel_report(report_data)
        except Exception as e:
            raise BacktestReportError(
                message=f"Failed to export Excel report: {str(e)}",
                backtest_id=self.backtest_id,
                report_type="excel",
                details={"error": str(e)}
            )