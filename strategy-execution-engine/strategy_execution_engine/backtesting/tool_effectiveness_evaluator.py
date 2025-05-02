"""
Tool Effectiveness Evaluator for Backtesting

This module integrates tool effectiveness metrics into the backtesting engine,
allowing technical analysis tools to be evaluated during backtesting.
"""

from typing import Dict, List, Optional, Any, Set, Callable, Union
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os

# Import tool effectiveness tracking from analysis engine service
# NOTE: In a real system, this would be imported from a shared package or accessed via API
from analysis_engine.services.tool_effectiveness import (
    ToolEffectivenessTracker, SignalEvent, TimeFrame, MarketRegime
)

class BacktestToolEffectivenessEvaluator:
    """
    Evaluates and tracks the effectiveness of technical analysis tools during backtesting
    
    This class integrates with both the backtesting engine and the tool effectiveness metrics
    framework to provide historical evaluation of technical tools.
    """
    
    def __init__(self, backtest_id: str):
        """
        Initialize the tool effectiveness evaluator
        
        Args:
            backtest_id: Unique identifier for the backtest run
        """
        self.backtest_id = backtest_id
        self.tracker = ToolEffectivenessTracker()
        self.signals_map = {}  # Maps internal signal IDs to tool effectiveness signal IDs
        self.pending_signals = {}  # Signals awaiting outcome evaluation
        self.logger = logging.getLogger(__name__)
        self.output_dir = os.path.join("output", "backtests", backtest_id, "effectiveness")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def register_signal(
        self,
        tool_name: str,
        signal_type: str,
        direction: str,
        strength: float,
        timestamp: datetime,
        symbol: str,
        timeframe: Union[str, TimeFrame],
        price: float,
        market_regime: Union[str, MarketRegime] = MarketRegime.UNKNOWN,
        metadata: Dict[str, Any] = None,
        internal_id: Optional[str] = None
    ) -> str:
        """
        Register a signal from a technical analysis tool during backtesting
        
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
            internal_id: Optional internal ID used by the backtester
            
        Returns:
            Signal ID from the effectiveness tracker
        """
        # Convert timeframe to TimeFrame enum if it's a string
        if isinstance(timeframe, str):
            try:
                timeframe = TimeFrame(timeframe)
            except ValueError:
                self.logger.warning(f"Invalid timeframe: {timeframe}, using nearest match")
                # Try to find closest match
                if timeframe in ("1m", "1min", "M1"):
                    timeframe = TimeFrame.M1
                elif timeframe in ("5m", "5min", "M5"):
                    timeframe = TimeFrame.M5
                elif timeframe in ("15m", "15min", "M15"):
                    timeframe = TimeFrame.M15
                elif timeframe in ("30m", "30min", "M30"):
                    timeframe = TimeFrame.M30
                elif timeframe in ("1h", "60m", "H1"):
                    timeframe = TimeFrame.H1
                elif timeframe in ("4h", "240m", "H4"):
                    timeframe = TimeFrame.H4
                elif timeframe in ("1d", "D", "D1"):
                    timeframe = TimeFrame.D1
                elif timeframe in ("1w", "W", "W1"):
                    timeframe = TimeFrame.W1
                else:
                    timeframe = TimeFrame.H1  # Default to H1 if no match
        
        # Convert market_regime to MarketRegime enum if it's a string
        if isinstance(market_regime, str):
            try:
                market_regime = MarketRegime(market_regime)
            except ValueError:
                self.logger.warning(f"Invalid market regime: {market_regime}, using UNKNOWN")
                market_regime = MarketRegime.UNKNOWN
        
        # Create market context with regime information
        market_context = {
            "regime": market_regime,
            "backtest_id": self.backtest_id
        }
        
        # Create signal event
        signal = SignalEvent(
            tool_name=tool_name,
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            price_at_signal=price,
            metadata=metadata or {},
            market_context=market_context
        )
        
        # Register signal with effectiveness tracker
        signal_id = self.tracker.register_signal(signal)
        
        # Store in pending signals
        self.pending_signals[signal_id] = {
            "signal": signal,
            "registered_at": datetime.now(),
            "status": "pending"
        }
        
        # Map internal ID to effectiveness signal ID if provided
        if internal_id:
            self.signals_map[internal_id] = signal_id
            
        self.logger.info(f"Registered signal from {tool_name}: {signal_id} (direction: {direction})")
        
        return signal_id
    
    def register_outcome(
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
        # If internal_id provided, try to map it to signal_id
        if internal_id and signal_id is None:
            if internal_id in self.signals_map:
                signal_id = self.signals_map[internal_id]
            else:
                self.logger.warning(f"No mapped signal found for internal ID: {internal_id}")
                return False
        
        # Validate outcome value
        if outcome not in ["success", "failure", "undetermined"]:
            self.logger.warning(f"Invalid outcome: {outcome}, must be success, failure, or undetermined")
            outcome = "undetermined"
        
        # Register outcome
        result = self.tracker.register_outcome(
            signal_id=signal_id,
            outcome=outcome,
            exit_price=exit_price,
            exit_timestamp=exit_timestamp,
            max_favorable_price=max_favorable_price,
            max_adverse_price=max_adverse_price,
            profit_loss=profit_loss,
            notes=notes
        )
        
        # Update pending signals
        if signal_id in self.pending_signals:
            if result:
                self.pending_signals[signal_id]["status"] = "completed"
                self.pending_signals[signal_id]["outcome"] = outcome
                self.pending_signals[signal_id]["completed_at"] = datetime.now()
            else:
                self.pending_signals[signal_id]["status"] = "error"
                
        return result is not None
    
    def bulk_evaluate_pending_signals(
        self,
        price_data: pd.DataFrame,
        evaluation_params: Dict[str, Any] = None
    ) -> int:
        """
        Bulk evaluate all pending signals against price data
        
        Args:
            price_data: DataFrame with OHLCV data, must include timestamp column
            evaluation_params: Configuration for evaluation
                max_bars_forward: Maximum bars to look forward for evaluation
                success_threshold_pips: For buy signals, pips above entry to consider success
                failure_threshold_pips: For buy signals, pips below entry to consider failure
                pip_value: Value of 1 pip (e.g., 0.0001 for most pairs)
                
        Returns:
            Number of signals evaluated
        """
        params = evaluation_params or {}
        max_bars = params.get("max_bars_forward", 20)
        success_threshold = params.get("success_threshold_pips", 10) * params.get("pip_value", 0.0001)
        failure_threshold = params.get("failure_threshold_pips", 10) * params.get("pip_value", 0.0001)
        
        # Ensure timestamp column exists and is datetime
        if "timestamp" not in price_data.columns:
            self.logger.error("Price data must include timestamp column")
            return 0
        
        # Ensure price columns exist
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
            self.logger.error(f"Price data missing required columns: {missing_cols}")
            return 0
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_dtype(price_data["timestamp"]):
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
        
        # Sort by timestamp
        price_data = price_data.sort_values("timestamp")
        
        evaluated_count = 0
        for signal_id, signal_data in list(self.pending_signals.items()):
            if signal_data["status"] != "pending":
                continue
                
            signal = signal_data["signal"]
            signal_ts = signal.timestamp
            
            # Find index of signal timestamp
            idx = price_data[price_data["timestamp"] >= signal_ts].index
            if len(idx) == 0:
                continue
                
            start_idx = idx[0]
            
            # Get forward data for evaluation
            end_idx = min(start_idx + max_bars, len(price_data) - 1)
            if end_idx <= start_idx:
                continue
                
            forward_data = price_data.iloc[start_idx:end_idx + 1]
            
            # Evaluate based on direction
            if signal.direction == "buy":
                entry_price = signal.price_at_signal
                max_high = forward_data["high"].max()
                max_low = forward_data["low"].min()
                
                # Determine outcome
                if max_high >= entry_price + success_threshold:
                    outcome = "success"
                    exit_price = entry_price + success_threshold  # Assume take profit
                elif max_low <= entry_price - failure_threshold:
                    outcome = "failure"
                    exit_price = entry_price - failure_threshold  # Assume stop loss
                else:
                    # No clear outcome, use last price
                    if forward_data.iloc[-1]["close"] > entry_price:
                        outcome = "success"
                    elif forward_data.iloc[-1]["close"] < entry_price:
                        outcome = "failure"
                    else:
                        outcome = "undetermined"
                    exit_price = forward_data.iloc[-1]["close"]
                
                # Register outcome
                self.register_outcome(
                    signal_id=signal_id,
                    outcome=outcome,
                    exit_price=exit_price,
                    exit_timestamp=forward_data.iloc[-1]["timestamp"],
                    max_favorable_price=max_high,
                    max_adverse_price=max_low,
                    profit_loss=(exit_price - entry_price) if outcome != "undetermined" else 0
                )
                evaluated_count += 1
                
            elif signal.direction == "sell":
                entry_price = signal.price_at_signal
                max_high = forward_data["high"].max()
                max_low = forward_data["low"].min()
                
                # Determine outcome
                if max_low <= entry_price - success_threshold:
                    outcome = "success"
                    exit_price = entry_price - success_threshold  # Assume take profit
                elif max_high >= entry_price + failure_threshold:
                    outcome = "failure"
                    exit_price = entry_price + failure_threshold  # Assume stop loss
                else:
                    # No clear outcome, use last price
                    if forward_data.iloc[-1]["close"] < entry_price:
                        outcome = "success"
                    elif forward_data.iloc[-1]["close"] > entry_price:
                        outcome = "failure"
                    else:
                        outcome = "undetermined"
                    exit_price = forward_data.iloc[-1]["close"]
                
                # Register outcome
                self.register_outcome(
                    signal_id=signal_id,
                    outcome=outcome,
                    exit_price=exit_price,
                    exit_timestamp=forward_data.iloc[-1]["timestamp"],
                    max_favorable_price=max_low,  # For sell, low is favorable
                    max_adverse_price=max_high,   # For sell, high is adverse
                    profit_loss=(entry_price - exit_price) if outcome != "undetermined" else 0
                )
                evaluated_count += 1
        
        return evaluated_count
    
    def calculate_tool_metrics(
        self,
        tool_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        market_regime: Optional[MarketRegime] = None,
        timeframe: Optional[TimeFrame] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate effectiveness metrics for tools
        
        Args:
            tool_name: Name of the tool to evaluate, or None to evaluate all
            metrics: List of metric names to calculate, or None for all
            market_regime: Optional filter by market regime
            timeframe: Optional filter by timeframe
            symbol: Optional filter by symbol
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            
        Returns:
            Dictionary with calculated metrics
        """
        if tool_name:
            return self.tracker.calculate_metrics(
                tool_name=tool_name,
                metric_names=metrics,
                market_regime=market_regime,
                timeframe=timeframe,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
        else:
            # Calculate for all tools
            tool_names = self.tracker.get_tool_names()
            results = {}
            for name in tool_names:
                results[name] = self.tracker.calculate_metrics(
                    tool_name=name,
                    metric_names=metrics,
                    market_regime=market_regime,
                    timeframe=timeframe,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            return results
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Get summary report of tool effectiveness across all tools
        
        Returns:
            Dictionary with summary report
        """
        return self.tracker.get_summary_report()
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save effectiveness results to a JSON file
        
        Args:
            filename: Optional custom filename, or None to use default
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"effectiveness_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Save to JSON
        self.tracker.save_to_json(file_path)
        
        self.logger.info(f"Saved effectiveness results to {file_path}")
        return file_path
    
    def load_results(self, file_path: str) -> bool:
        """
        Load effectiveness results from a JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        return self.tracker.load_from_json(file_path)
    
    def generate_effectiveness_report(self, output_format: str = "json") -> Union[Dict[str, Any], str]:
        """
        Generate a comprehensive effectiveness report
        
        Args:
            output_format: "json" or "markdown"
            
        Returns:
            Report in the specified format
        """
        # Calculate metrics for all tools
        tool_names = self.tracker.get_tool_names()
        report_data = {
            "summary": {
                "backtest_id": self.backtest_id,
                "generated_at": datetime.now().isoformat(),
                "total_signals": len(self.tracker.signals),
                "total_outcomes": len(self.tracker.outcomes),
                "tools_evaluated": len(tool_names)
            },
            "tools": {}
        }
        
        # Calculate metrics by tool
        for tool_name in tool_names:
            tool_metrics = self.tracker.calculate_metrics(tool_name)
            report_data["tools"][tool_name] = tool_metrics
            
            # Calculate metrics by market regime for this tool
            regimes = {}
            for regime in MarketRegime:
                regime_metrics = self.tracker.calculate_metrics(
                    tool_name=tool_name,
                    market_regime=regime
                )
                if regime_metrics.get("sample_size", 0) > 0:
                    regimes[regime] = regime_metrics
            
            if regimes:
                report_data["tools"][tool_name]["by_regime"] = regimes
        
        # Generate report in specified format
        if output_format == "markdown":
            md_report = [f"# Tool Effectiveness Report - Backtest {self.backtest_id}"]
            md_report.append(f"\nGenerated: {report_data['summary']['generated_at']}")
            md_report.append(f"\nTotal Signals: {report_data['summary']['total_signals']}")
            md_report.append(f"Total Outcomes: {report_data['summary']['total_outcomes']}")
            md_report.append(f"Tools Evaluated: {report_data['summary']['tools_evaluated']}")
            
            for tool_name, tool_data in report_data["tools"].items():
                md_report.append(f"\n## {tool_name}")
                md_report.append(f"Sample Size: {tool_data.get('sample_size', 0)}")
                
                # Add metrics
                md_report.append("\n### Metrics")
                metrics_table = ["| Metric | Value | Sample Size |", "|--------|-------|------------|"]
                
                for metric_name, metric_data in tool_data.get("metrics", {}).items():
                    value = metric_data.get("value")
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    metrics_table.append(
                        f"| {metric_name} | {value} | {metric_data.get('sample_size', 0)} |"
                    )
                
                md_report.extend(metrics_table)
                
                # Add regime breakdown if available
                if "by_regime" in tool_data:
                    md_report.append("\n### Performance by Market Regime")
                    regime_table = ["| Regime | Win Rate | Sample Size |", "|--------|----------|------------|"]
                    
                    for regime, regime_data in tool_data["by_regime"].items():
                        win_rate = regime_data.get("metrics", {}).get("win_rate", {}).get("value", 0)
                        if win_rate:
                            win_rate = f"{win_rate:.2%}"
                        regime_table.append(
                            f"| {regime} | {win_rate} | {regime_data.get('sample_size', 0)} |"
                        )
                    
                    md_report.extend(regime_table)
            
            return "\n".join(md_report)
        else:
            # Default to JSON
            return report_data
    
    def export_report(self, output_format: str = "json", filename: Optional[str] = None) -> str:
        """
        Export effectiveness report to a file
        
        Args:
            output_format: "json" or "markdown"
            filename: Optional filename, or None to use default
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = f"effectiveness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Add extension based on format
        if output_format == "markdown":
            filename = f"{filename}.md"
        else:
            filename = f"{filename}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Generate report
        report = self.generate_effectiveness_report(output_format)
        
        # Save to file
        with open(file_path, 'w') as f:
            if output_format == "markdown":
                f.write(report)
            else:
                json.dump(report, f, indent=2)
        
        self.logger.info(f"Exported effectiveness report to {file_path}")
        return file_path