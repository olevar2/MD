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
    
    def run_strategy(self, strategy_func: Callable, **strategy_params):
        """
        Run a trading strategy on the historical data
        
        Args:
            strategy_func: Strategy function to execute
            strategy_params: Additional parameters for the strategy
            
        Returns:
            Dict with backtest results
        """
        if self.data is None:
            self.logger.error("No data provided for backtesting")
            return {"error": "No data provided for backtesting"}
        
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
            
            # Save backtest results
            results_path = self._save_results()
            
            return {
                "backtest_id": self.backtest_id,
                "metrics": self.metrics,
                "results_path": results_path,
                "strategy_results": strategy_results,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error running strategy: {str(e)}", exc_info=True)
            return {
                "backtest_id": self.backtest_id,
                "error": str(e),
                "success": False
            }
    
    def _calculate_metrics(self):
        """Calculate performance metrics for the backtest"""
        # Implement detailed performance metrics calculation
        # Placeholder for now
        self.metrics = {
            "total_trades": len(self.closed_positions),
            "winning_trades": sum(1 for p in self.closed_positions if p["profit"] > 0),
            "losing_trades": sum(1 for p in self.closed_positions if p["profit"] <= 0),
            "win_rate": sum(1 for p in self.closed_positions if p["profit"] > 0) / max(1, len(self.closed_positions)),
            "final_balance": self.balance,
            "profit_loss": self.balance - self.initial_balance,
            "return_pct": (self.balance / self.initial_balance - 1) * 100
        }
    
    def _save_results(self) -> str:
        """Save backtest results to file"""
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
        
        file_path = os.path.join(self.output_dir, "results.json")
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return file_path
    
    # Trading methods
    def open_position(self, timestamp, symbol, direction, size, price=None, stop_loss=None, take_profit=None, metadata=None):
        """Open a new trading position"""
        # Implementation details for opening a position
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
        self.positions.append(position)
        return position["id"]
    
    def close_position(self, position_id, timestamp, price=None, reason=None):
        """Close an existing position"""
        # Implementation details for closing a position
        position = next((p for p in self.positions if p["id"] == position_id), None)
        if not position:
            return None
            
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
        
        return position
    
    # Tool effectiveness tracking methods
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
        """
        if not self.track_tool_effectiveness or not self.tool_evaluator:
            return None
            
        try:
            return self.tool_evaluator.register_signal(
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
        except Exception as e:
            self.logger.error(f"Error registering tool signal: {str(e)}")
            return None
    
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
        """
        if not self.track_tool_effectiveness or not self.tool_evaluator:
            return False
            
        try:
            return self.tool_evaluator.register_outcome(
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
        except Exception as e:
            self.logger.error(f"Error registering tool outcome: {str(e)}")
            return False
    
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
        """
        if not self.track_tool_effectiveness or not self.tool_evaluator:
            return {}
            
        try:
            return self.tool_evaluator.calculate_tool_metrics(
                tool_name=tool_name,
                market_regime=market_regime,
                timeframe=timeframe,
                symbol=symbol
            )
        except Exception as e:
            self.logger.error(f"Error calculating tool metrics: {str(e)}")
            return {}
    
    # Advanced reporting methods
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
        """
        report = BacktestReport(self)
        return report.generate_performance_report(
            include_trades=include_trades,
            include_drawdowns=include_drawdowns,
            include_metrics=include_metrics
        )
    
    def create_interactive_dashboard(self, report_data: Dict[str, Any] = None) -> str:
        """
        Create an interactive HTML dashboard for backtest results
        
        Args:
            report_data: Report data (generated if not provided)
            
        Returns:
            Path to the generated HTML dashboard
        """
        report = BacktestReport(self)
        return report.create_interactive_dashboard(report_data)
    
    def export_pdf_report(self, report_data: Dict[str, Any] = None) -> str:
        """
        Export backtest results to a PDF report
        
        Args:
            report_data: Report data (generated if not provided)
            
        Returns:
            Path to the exported PDF report
        """
        report = BacktestReport(self)
        return report.export_pdf_report(report_data)
    
    def export_excel_report(self, report_data: Dict[str, Any] = None) -> str:
        """
        Export backtest results to an Excel report
        
        Args:
            report_data: Report data (generated if not provided)
            
        Returns:
            Path to the exported Excel report
        """
        report = BacktestReport(self)
        return report.export_excel_report(report_data)