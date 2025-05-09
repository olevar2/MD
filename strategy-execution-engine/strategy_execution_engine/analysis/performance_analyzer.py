"""
Performance Analyzer for Strategy Execution Engine

This module provides functionality for analyzing the performance of trading strategies
based on their backtesting results.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from strategy_execution_engine.core.config import get_settings
from strategy_execution_engine.error import (
    AnalysisError,
    DataFetchError,
    async_with_error_handling
)

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Analyzer for strategy performance based on backtest results.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self._settings = get_settings()
        self._backtest_data_dir = self._settings.backtest_data_dir
        
        logger.info("Performance analyzer initialized")
    
    def get_backtest_results(self, backtest_id: str) -> Dict[str, Any]:
        """
        Get backtest results by ID.
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            Dict: Backtest results
            
        Raises:
            DataFetchError: If backtest results not found
        """
        file_path = os.path.join(self._backtest_data_dir, f"{backtest_id}.json")
        
        if not os.path.exists(file_path):
            raise DataFetchError(f"Backtest results not found: {backtest_id}")
        
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load backtest results: {e}", exc_info=True)
            raise DataFetchError(f"Failed to load backtest results: {str(e)}")
    
    def get_all_backtest_results(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all backtest results, optionally filtered by strategy ID.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            
        Returns:
            List: Backtest results
        """
        results = []
        
        try:
            for filename in os.listdir(self._backtest_data_dir):
                if not filename.endswith(".json"):
                    continue
                
                file_path = os.path.join(self._backtest_data_dir, filename)
                
                try:
                    with open(file_path, "r") as f:
                        result = json.load(f)
                        
                        # Filter by strategy ID if provided
                        if strategy_id and result.get("strategy_id") != strategy_id:
                            continue
                        
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load backtest result {filename}: {e}")
        except Exception as e:
            logger.error(f"Failed to list backtest results: {e}", exc_info=True)
        
        return results
    
    def analyze_backtest(self, backtest_id: str) -> Dict[str, Any]:
        """
        Analyze a backtest by ID.
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            Dict: Analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        try:
            # Get backtest results
            backtest = self.get_backtest_results(backtest_id)
            
            # Extract metrics
            metrics = backtest.get("metrics", {})
            
            # Extract trades
            trades = backtest.get("trades", [])
            
            # Extract equity curve
            equity_curve = backtest.get("equity_curve", [])
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(trades, equity_curve)
            
            # Combine metrics
            combined_metrics = {**metrics, **additional_metrics}
            
            # Calculate trade statistics
            trade_stats = self._calculate_trade_statistics(trades)
            
            # Calculate drawdown statistics
            drawdown_stats = self._calculate_drawdown_statistics(equity_curve)
            
            # Calculate monthly returns
            monthly_returns = self._calculate_monthly_returns(equity_curve)
            
            # Return analysis results
            return {
                "backtest_id": backtest_id,
                "strategy_id": backtest.get("strategy_id"),
                "start_date": backtest.get("start_date"),
                "end_date": backtest.get("end_date"),
                "metrics": combined_metrics,
                "trade_stats": trade_stats,
                "drawdown_stats": drawdown_stats,
                "monthly_returns": monthly_returns
            }
        except DataFetchError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to analyze backtest: {e}", exc_info=True)
            raise AnalysisError(f"Failed to analyze backtest: {str(e)}")
    
    def compare_strategies(self, strategy_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple strategies based on their backtest results.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            
        Returns:
            Dict: Comparison results
            
        Raises:
            AnalysisError: If comparison fails
        """
        try:
            # Get latest backtest for each strategy
            strategy_results = {}
            
            for strategy_id in strategy_ids:
                backtests = self.get_all_backtest_results(strategy_id)
                
                if not backtests:
                    logger.warning(f"No backtest results found for strategy {strategy_id}")
                    continue
                
                # Sort by date (newest first)
                backtests.sort(key=lambda x: x.get("end_date", ""), reverse=True)
                
                # Get latest backtest
                latest_backtest = backtests[0]
                
                # Analyze backtest
                analysis = self.analyze_backtest(latest_backtest.get("backtest_id"))
                
                strategy_results[strategy_id] = analysis
            
            # Compare metrics
            metrics_comparison = self._compare_metrics(strategy_results)
            
            # Compare trade statistics
            trade_stats_comparison = self._compare_trade_statistics(strategy_results)
            
            # Compare drawdown statistics
            drawdown_comparison = self._compare_drawdown_statistics(strategy_results)
            
            # Return comparison results
            return {
                "strategies": list(strategy_results.keys()),
                "metrics_comparison": metrics_comparison,
                "trade_stats_comparison": trade_stats_comparison,
                "drawdown_comparison": drawdown_comparison,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to compare strategies: {e}", exc_info=True)
            raise AnalysisError(f"Failed to compare strategies: {str(e)}")
    
    def _calculate_additional_metrics(self, trades: List[Dict[str, Any]], 
                                     equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate additional performance metrics.
        
        Args:
            trades: List of trades
            equity_curve: Equity curve
            
        Returns:
            Dict: Additional metrics
        """
        if not trades or not equity_curve:
            return {}
        
        try:
            # Calculate Sharpe ratio
            returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i-1].get("equity", 0)
                curr_equity = equity_curve[i].get("equity", 0)
                
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            sharpe_ratio = 0
            if returns and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate Sortino ratio
            negative_returns = [r for r in returns if r < 0]
            sortino_ratio = 0
            if negative_returns and np.std(negative_returns) > 0:
                sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)  # Annualized
            
            # Calculate Calmar ratio
            max_drawdown = 0
            if equity_curve:
                peak = equity_curve[0].get("equity", 0)
                for point in equity_curve:
                    equity = point.get("equity", 0)
                    if equity > peak:
                        peak = equity
                    else:
                        drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
                        max_drawdown = max(max_drawdown, drawdown)
            
            calmar_ratio = 0
            if max_drawdown > 0:
                calmar_ratio = (equity_curve[-1].get("equity", 0) - equity_curve[0].get("equity", 0)) / equity_curve[0].get("equity", 0) * 100 / max_drawdown
            
            return {
                "sharpe_ratio": round(sharpe_ratio, 2),
                "sortino_ratio": round(sortino_ratio, 2),
                "calmar_ratio": round(calmar_ratio, 2)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate additional metrics: {e}")
            return {}
    
    def _calculate_trade_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trade statistics.
        
        Args:
            trades: List of trades
            
        Returns:
            Dict: Trade statistics
        """
        if not trades:
            return {}
        
        try:
            # Calculate trade durations
            durations = []
            for trade in trades:
                entry_time = trade.get("entry_time")
                exit_time = trade.get("exit_time")
                
                if entry_time and exit_time:
                    try:
                        entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                        exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                        duration = (exit_dt - entry_dt).total_seconds() / 3600  # Hours
                        durations.append(duration)
                    except Exception as e:
                        logger.warning(f"Failed to parse trade times: {e}")
            
            # Calculate profit/loss statistics
            profit_loss = [trade.get("profit_loss", 0) for trade in trades]
            profit_loss_pct = [trade.get("profit_loss_pct", 0) for trade in trades]
            
            # Calculate consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            current_streak = 0
            
            for trade in trades:
                pl = trade.get("profit_loss", 0)
                
                if pl > 0:
                    if current_streak > 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                    
                    consecutive_wins = max(consecutive_wins, current_streak)
                elif pl < 0:
                    if current_streak < 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                    
                    consecutive_losses = max(consecutive_losses, abs(current_streak))
            
            return {
                "avg_trade_duration": round(np.mean(durations), 2) if durations else 0,
                "max_trade_duration": round(max(durations), 2) if durations else 0,
                "min_trade_duration": round(min(durations), 2) if durations else 0,
                "avg_profit": round(np.mean([p for p in profit_loss if p > 0]), 2) if any(p > 0 for p in profit_loss) else 0,
                "avg_loss": round(np.mean([p for p in profit_loss if p < 0]), 2) if any(p < 0 for p in profit_loss) else 0,
                "max_profit": round(max(profit_loss), 2) if profit_loss else 0,
                "max_loss": round(min(profit_loss), 2) if profit_loss else 0,
                "avg_profit_pct": round(np.mean([p for p in profit_loss_pct if p > 0]), 2) if any(p > 0 for p in profit_loss_pct) else 0,
                "avg_loss_pct": round(np.mean([p for p in profit_loss_pct if p < 0]), 2) if any(p < 0 for p in profit_loss_pct) else 0,
                "max_profit_pct": round(max(profit_loss_pct), 2) if profit_loss_pct else 0,
                "max_loss_pct": round(min(profit_loss_pct), 2) if profit_loss_pct else 0,
                "consecutive_wins": consecutive_wins,
                "consecutive_losses": consecutive_losses
            }
        except Exception as e:
            logger.warning(f"Failed to calculate trade statistics: {e}")
            return {}
    
    def _calculate_drawdown_statistics(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate drawdown statistics.
        
        Args:
            equity_curve: Equity curve
            
        Returns:
            Dict: Drawdown statistics
        """
        if not equity_curve:
            return {}
        
        try:
            # Extract equity values
            equity_values = [point.get("equity", 0) for point in equity_curve]
            timestamps = [point.get("timestamp", "") for point in equity_curve]
            
            # Calculate drawdowns
            drawdowns = []
            peak = equity_values[0]
            peak_idx = 0
            
            for i, equity in enumerate(equity_values):
                if equity > peak:
                    peak = equity
                    peak_idx = i
                elif peak > 0:
                    drawdown = (peak - equity) / peak * 100
                    if drawdown > 0:
                        drawdowns.append({
                            "drawdown": drawdown,
                            "peak_equity": peak,
                            "trough_equity": equity,
                            "peak_timestamp": timestamps[peak_idx],
                            "trough_timestamp": timestamps[i]
                        })
            
            # Sort drawdowns by size (largest first)
            drawdowns.sort(key=lambda x: x["drawdown"], reverse=True)
            
            # Calculate recovery times
            for i, drawdown in enumerate(drawdowns[:10]):  # Only analyze top 10 drawdowns
                peak_idx = timestamps.index(drawdown["peak_timestamp"])
                trough_idx = timestamps.index(drawdown["trough_timestamp"])
                
                # Find recovery point (if any)
                recovery_idx = None
                for j in range(trough_idx + 1, len(equity_values)):
                    if equity_values[j] >= drawdown["peak_equity"]:
                        recovery_idx = j
                        break
                
                if recovery_idx is not None:
                    try:
                        trough_dt = datetime.fromisoformat(timestamps[trough_idx].replace("Z", "+00:00"))
                        recovery_dt = datetime.fromisoformat(timestamps[recovery_idx].replace("Z", "+00:00"))
                        recovery_time = (recovery_dt - trough_dt).total_seconds() / 86400  # Days
                        drawdowns[i]["recovery_time"] = recovery_time
                        drawdowns[i]["recovery_timestamp"] = timestamps[recovery_idx]
                    except Exception as e:
                        logger.warning(f"Failed to calculate recovery time: {e}")
            
            # Calculate underwater periods
            underwater_periods = []
            underwater = False
            start_idx = None
            
            for i, equity in enumerate(equity_values):
                if equity < peak and not underwater:
                    underwater = True
                    start_idx = i
                elif equity >= peak and underwater:
                    underwater = False
                    try:
                        start_dt = datetime.fromisoformat(timestamps[start_idx].replace("Z", "+00:00"))
                        end_dt = datetime.fromisoformat(timestamps[i].replace("Z", "+00:00"))
                        duration = (end_dt - start_dt).total_seconds() / 86400  # Days
                        underwater_periods.append(duration)
                    except Exception as e:
                        logger.warning(f"Failed to calculate underwater period: {e}")
                
                if equity > peak:
                    peak = equity
            
            return {
                "max_drawdown": round(drawdowns[0]["drawdown"], 2) if drawdowns else 0,
                "avg_drawdown": round(np.mean([d["drawdown"] for d in drawdowns]), 2) if drawdowns else 0,
                "top_drawdowns": drawdowns[:5],  # Top 5 drawdowns
                "avg_recovery_time": round(np.mean([d.get("recovery_time", 0) for d in drawdowns if "recovery_time" in d]), 2) if any("recovery_time" in d for d in drawdowns) else 0,
                "max_recovery_time": round(max([d.get("recovery_time", 0) for d in drawdowns if "recovery_time" in d]), 2) if any("recovery_time" in d for d in drawdowns) else 0,
                "avg_underwater_period": round(np.mean(underwater_periods), 2) if underwater_periods else 0,
                "max_underwater_period": round(max(underwater_periods), 2) if underwater_periods else 0
            }
        except Exception as e:
            logger.warning(f"Failed to calculate drawdown statistics: {e}")
            return {}
    
    def _calculate_monthly_returns(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate monthly returns.
        
        Args:
            equity_curve: Equity curve
            
        Returns:
            Dict: Monthly returns
        """
        if not equity_curve:
            return {}
        
        try:
            # Group equity points by month
            monthly_data = {}
            
            for point in equity_curve:
                timestamp = point.get("timestamp", "")
                equity = point.get("equity", 0)
                
                if not timestamp:
                    continue
                
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    month_key = f"{dt.year}-{dt.month:02d}"
                    
                    if month_key not in monthly_data:
                        monthly_data[month_key] = {"first": equity, "last": equity}
                    else:
                        monthly_data[month_key]["last"] = equity
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp: {e}")
            
            # Calculate monthly returns
            monthly_returns = {}
            
            for month, data in monthly_data.items():
                if data["first"] > 0:
                    monthly_returns[month] = round((data["last"] - data["first"]) / data["first"] * 100, 2)
            
            return monthly_returns
        except Exception as e:
            logger.warning(f"Failed to calculate monthly returns: {e}")
            return {}
    
    def _compare_metrics(self, strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Compare metrics across strategies.
        
        Args:
            strategy_results: Results for each strategy
            
        Returns:
            Dict: Metrics comparison
        """
        metrics_comparison = {}
        
        for metric in ["win_rate", "profit_factor", "net_profit_pct", "max_drawdown", 
                      "sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
            metrics_comparison[metric] = {}
            
            for strategy_id, result in strategy_results.items():
                metrics = result.get("metrics", {})
                metrics_comparison[metric][strategy_id] = metrics.get(metric, 0)
        
        return metrics_comparison
    
    def _compare_trade_statistics(self, strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Compare trade statistics across strategies.
        
        Args:
            strategy_results: Results for each strategy
            
        Returns:
            Dict: Trade statistics comparison
        """
        trade_stats_comparison = {}
        
        for stat in ["avg_trade_duration", "avg_profit", "avg_loss", 
                    "avg_profit_pct", "avg_loss_pct", "consecutive_wins", "consecutive_losses"]:
            trade_stats_comparison[stat] = {}
            
            for strategy_id, result in strategy_results.items():
                trade_stats = result.get("trade_stats", {})
                trade_stats_comparison[stat][strategy_id] = trade_stats.get(stat, 0)
        
        return trade_stats_comparison
    
    def _compare_drawdown_statistics(self, strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Compare drawdown statistics across strategies.
        
        Args:
            strategy_results: Results for each strategy
            
        Returns:
            Dict: Drawdown statistics comparison
        """
        drawdown_comparison = {}
        
        for stat in ["max_drawdown", "avg_drawdown", "avg_recovery_time", "max_recovery_time"]:
            drawdown_comparison[stat] = {}
            
            for strategy_id, result in strategy_results.items():
                drawdown_stats = result.get("drawdown_stats", {})
                drawdown_comparison[stat][strategy_id] = drawdown_stats.get(stat, 0)
        
        return drawdown_comparison

# Create singleton instance
performance_analyzer = PerformanceAnalyzer()
