"""
News Event Backtester for Forex Trading Platform.

This module provides a backtesting framework that incorporates historical news events
to evaluate trading strategies under realistic market conditions during economic releases,
central bank decisions, and other significant news events.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import json
import os
import uuid
import logging

from trading_gateway_service.simulation.news_sentiment_simulator import (
    NewsAndSentimentSimulator, NewsEvent, NewsImpactLevel, NewsEventType, SentimentLevel
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class NewsEventBacktester:
    """
    Backtester that incorporates news events and their market impact.
    
    This class extends traditional backtesting by integrating historical news events 
    and their impact on market prices, volatility, slippage, and gaps.
    """
    
    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        initial_balance: float = 10000.0,
        commission: float = 0.0,
        base_slippage: float = 0.0,
        base_spread: float = 0.0,
        seed: Optional[int] = None,
        backtest_id: Optional[str] = None
    ):
        """
        Initialize the news event backtester.
        
        Args:
            historical_data: Dictionary of DataFrames with market data, keyed by instrument
            initial_balance: Starting account balance
            commission: Commission per trade (percentage)
            base_slippage: Base slippage per trade (pips)
            base_spread: Base spread in pips
            seed: Random seed for reproducibility
            backtest_id: Optional ID for the backtest, auto-generated if None
        """
        self.historical_data = historical_data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.base_slippage = base_slippage
        self.base_spread = base_spread
        
        # Initialize news simulator
        self.news_simulator = NewsAndSentimentSimulator(seed=seed)
        
        # Generate or set backtest ID
        self.backtest_id = backtest_id or f"news_backtest_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create logger
        self.logger = logger
        
        # Trading state
        self.positions = []
        self.closed_positions = []
        self.balance_history = []
        self.equity_history = []
        self.metrics = {}
        
        # News events storage
        self.news_events = []
        self.news_impacts = {}
        
        # Output directory
        self.output_dir = os.path.join("output", "news_backtests", self.backtest_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def load_historical_news_events(self, news_events_file: str) -> bool:
        """
        Load historical news events from a file.
        
        Args:
            news_events_file: Path to JSON file containing historical news events
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(news_events_file, 'r') as f:
                events_data = json.load(f)
            
            # Parse events
            events = []
            for event_data in events_data:
                event = NewsEvent.from_dict(event_data)
                events.append(event)
            
            self.news_events = events
            self.logger.info(f"Loaded {len(events)} news events from {news_events_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading news events: {str(e)}")
            return False
            
    def generate_synthetic_news_events(
        self,
        num_events: int = 50,
        currency_pairs: Optional[List[str]] = None
    ) -> List[NewsEvent]:
        """
        Generate synthetic news events for backtesting when historical events aren't available.
        
        Args:
            num_events: Number of news events to generate
            currency_pairs: Currency pairs to generate events for (defaults to keys in historical_data)
            
        Returns:
            List of generated NewsEvent objects
        """
        if currency_pairs is None:
            currency_pairs = list(self.historical_data.keys())
        
        # Get min and max dates from historical data
        all_dates = []
        for df in self.historical_data.values():
            if not df.empty:
                all_dates.extend([df.index[0], df.index[-1]])
        
        if not all_dates:
            self.logger.error("No valid dates found in historical data")
            return []
            
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        # Generate events
        events = self.news_simulator.generate_random_economic_calendar(
            start_date, 
            end_date,
            currency_pairs,
            num_events=num_events
        )
        
        self.news_events = events
        self.logger.info(f"Generated {len(events)} synthetic news events")
        return events
        
    def calculate_news_impacts(self) -> None:
        """
        Calculate the impact of news events on each timepoint in the historical data.
        
        This pre-calculates news impacts for each instrument at each timestamp for efficiency.
        """
        self.news_impacts = {}
        
        for instrument, df in self.historical_data.items():
            self.news_impacts[instrument] = {}
            
            # Process each timestamp
            for timestamp in df.index:
                # Set simulator to current time
                self.news_simulator.set_current_time(timestamp)
                
                # Get impact for this instrument at this time
                impact = self.news_simulator.calculate_price_impact(
                    instrument, 
                    1.0,  # Base price
                    0.0001  # Base volatility
                )
                
                # Check for gaps
                gap_prob = self.news_simulator.calculate_gap_probability(instrument)
                
                # Store impact data
                self.news_impacts[instrument][timestamp] = {
                    "price_change_pct": impact["price_change_pct"],
                    "volatility_multiplier": impact["volatility_multiplier"],
                    "spread_multiplier": impact["spread_multiplier"],
                    "gap_probability": gap_prob
                }
        
        self.logger.info("Pre-calculated news impacts for all instruments and timestamps")
                
    def apply_news_impact_to_prices(self) -> Dict[str, pd.DataFrame]:
        """
        Apply news impacts to historical price data.
        
        Returns:
            Dictionary of DataFrames with news-adjusted price data
        """
        adjusted_data = {}
        
        for instrument, df in self.historical_data.items():
            adjusted_df = df.copy()
            
            # Apply news impacts to each row
            for i, timestamp in enumerate(adjusted_df.index):
                if i == 0:
                    # Skip first row as we need previous close for gap analysis
                    continue
                    
                # Get news impact for this timestamp
                if timestamp in self.news_impacts[instrument]:
                    impact = self.news_impacts[instrument][timestamp]
                    
                    # Check for gaps
                    gap_occurred = False
                    if i > 0 and np.random.random() < impact["gap_probability"]:
                        gap_size = self.news_simulator.generate_gap_size(
                            instrument, 
                            adjusted_df.iloc[i-1]['close']
                        )
                        gap_occurred = True
                        
                        # Apply gap to open price
                        adjusted_df.at[adjusted_df.index[i], 'open'] = adjusted_df.iloc[i-1]['close'] + gap_size
                    
                    # Apply price impact (percentage change)
                    price_factor = 1.0 + impact["price_change_pct"]
                    
                    # Apply to OHLC
                    if not gap_occurred:
                        adjusted_df.at[adjusted_df.index[i], 'open'] *= price_factor
                    adjusted_df.at[adjusted_df.index[i], 'high'] *= price_factor
                    adjusted_df.at[adjusted_df.index[i], 'low'] *= price_factor
                    adjusted_df.at[adjusted_df.index[i], 'close'] *= price_factor
                    
                    # Apply volatility impact (increase range between high and low)
                    if impact["volatility_multiplier"] > 1.01:
                        mid_price = (adjusted_df.at[adjusted_df.index[i], 'high'] + 
                                    adjusted_df.at[adjusted_df.index[i], 'low']) / 2
                        current_range = adjusted_df.at[adjusted_df.index[i], 'high'] - adjusted_df.at[adjusted_df.index[i], 'low']
                        new_range = current_range * impact["volatility_multiplier"]
                        adjusted_df.at[adjusted_df.index[i], 'high'] = mid_price + new_range/2
                        adjusted_df.at[adjusted_df.index[i], 'low'] = mid_price - new_range/2
                        
                        # Make sure high >= close >= low
                        adjusted_df.at[adjusted_df.index[i], 'high'] = max(
                            adjusted_df.at[adjusted_df.index[i], 'high'],
                            adjusted_df.at[adjusted_df.index[i], 'close']
                        )
                        adjusted_df.at[adjusted_df.index[i], 'low'] = min(
                            adjusted_df.at[adjusted_df.index[i], 'low'],
                            adjusted_df.at[adjusted_df.index[i], 'close']
                        )
            
            adjusted_data[instrument] = adjusted_df
            
        return adjusted_data
        
    def get_active_news_events(self, timestamp: datetime, instrument: str) -> List[NewsEvent]:
        """
        Get a list of news events active at a specific time for an instrument.
        
        Args:
            timestamp: The timestamp to check
            instrument: The instrument to check
            
        Returns:
            List of active NewsEvent objects
        """
        active_events = []
        
        # Check each event
        for event in self.news_events:
            # Check if the event affects this instrument
            if instrument in event.currencies_affected:
                # Check if the event is active at this time
                event_end = event.timestamp + timedelta(minutes=event.duration_minutes)
                if event.timestamp <= timestamp <= event_end:
                    active_events.append(event)
        
        return active_events
        
    def calculate_slippage(self, timestamp: datetime, instrument: str, order_size: float) -> float:
        """
        Calculate slippage for an order taking into account news impacts.
        
        Args:
            timestamp: The timestamp of the order
            instrument: The instrument being traded
            order_size: Size of the order in lots
            
        Returns:
            Slippage amount in pips
        """
        # Set simulator to current time
        self.news_simulator.set_current_time(timestamp)
        
        # Calculate slippage using the news simulator
        slippage = self.news_simulator.calculate_slippage_impact(
            instrument, 
            order_size, 
            self.base_slippage
        )
        
        return slippage
        
    def calculate_spread(self, timestamp: datetime, instrument: str) -> float:
        """
        Calculate spread for an instrument taking into account news impacts.
        
        Args:
            timestamp: The timestamp to check
            instrument: The instrument to check
            
        Returns:
            Spread in pips
        """
        if timestamp in self.news_impacts[instrument]:
            impact = self.news_impacts[instrument][timestamp]
            return self.base_spread * impact["spread_multiplier"]
        
        return self.base_spread
        
    def run_backtest(self, strategy_func: Callable, **strategy_params) -> Dict[str, Any]:
        """
        Run a backtest with news event impact incorporated.
        
        Args:
            strategy_func: Strategy function to execute
            strategy_params: Additional parameters for the strategy
            
        Returns:
            Dict with backtest results
        """
        # Calculate news impacts if not already done
        if not self.news_impacts:
            self.calculate_news_impacts()
        
        # Apply news impacts to get adjusted historical data
        adjusted_data = self.apply_news_impact_to_prices()
        
        # Reset state
        self.balance = self.initial_balance
        self.positions = []
        self.closed_positions = []
        self.balance_history = []
        self.equity_history = []
        
        try:
            # Execute strategy with news-adjusted data
            strategy_results = strategy_func(
                data=adjusted_data,
                backtester=self,
                **strategy_params
            )
            
            # Calculate performance metrics
            self._calculate_metrics()
            
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
            
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics for the backtest."""
        if not self.closed_positions:
            self.metrics = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "profit_factor": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0,
                "net_profit": 0.0,
                "return_pct": 0.0,
                "max_drawdown_pct": 0.0
            }
            return
        
        # Basic counts
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p["profit"] > 0])
        losing_trades = len([p for p in self.closed_positions if p["profit"] < 0])
        
        # PnL statistics
        total_profit = sum([p["profit"] for p in self.closed_positions if p["profit"] > 0])
        total_loss = sum([p["profit"] for p in self.closed_positions if p["profit"] < 0])
        net_profit = sum([p["profit"] for p in self.closed_positions])
        
        # Performance ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        return_pct = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate max drawdown
        max_balance = self.initial_balance
        max_drawdown = 0.0
        
        for balance in self.balance_history:
            if balance > max_balance:
                max_balance = balance
            else:
                drawdown = (max_balance - balance) / max_balance * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        self.metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "profit_factor": profit_factor,
            "average_profit": avg_profit,
            "average_loss": avg_loss,
            "net_profit": net_profit,
            "return_pct": return_pct,
            "max_drawdown_pct": max_drawdown
        }
        
    def _save_results(self) -> str:
        """
        Save backtest results to file
        
        Returns:
            Path to saved results file
        """
        results = {
            "backtest_id": self.backtest_id,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "metrics": self.metrics,
            "trades": self.closed_positions,
            "balance_history": self.balance_history,
            "equity_history": self.equity_history,
            "news_events": [e.to_dict() for e in self.news_events]
        }
        
        results_path = os.path.join(self.output_dir, "results.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Saved backtest results to {results_path}")
        
        return results_path
        
    # Trading methods
    def open_position(
        self, 
        timestamp: datetime, 
        instrument: str, 
        direction: str, 
        size: float, 
        price: float = None, 
        stop_loss: float = None, 
        take_profit: float = None, 
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Open a new position with news-aware execution modeling.
        
        Args:
            timestamp: Entry time
            instrument: Symbol to trade
            direction: Trade direction ('long' or 'short')
            size: Position size in lots
            price: Entry price (None for market orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional position metadata
            
        Returns:
            Position details dictionary
        """
        if metadata is None:
            metadata = {}
            
        # Calculate spread
        spread = self.calculate_spread(timestamp, instrument)
        
        # Calculate slippage
        slippage_pips = self.calculate_slippage(timestamp, instrument, size)
        
        # Determine execution price
        if price is None:
            # Use close price as default for backtest
            for df in self.historical_data.values():
                if instrument in df.columns:
                    price = df.loc[timestamp]['close']
                    break
        
        # Apply slippage and spread to price
        if direction == "long":
            execution_price = price + spread / 2 + slippage_pips
        else:  # short
            execution_price = price - spread / 2 - slippage_pips
            
        # Generate position ID
        position_id = f"pos_{uuid.uuid4().hex[:8]}"
        
        # Create position
        position = {
            "id": position_id,
            "instrument": instrument,
            "direction": direction,
            "size": size,
            "entry_time": timestamp,
            "entry_price": execution_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "slippage": slippage_pips,
            "metadata": metadata
        }
        
        # Calculate commission
        commission = self.commission * execution_price * size
        
        # Update balance
        self.balance -= commission
        self.balance_history.append(self.balance)
        
        # Log active news during position opening
        active_news = self.get_active_news_events(timestamp, instrument)
        if active_news:
            news_info = []
            for event in active_news:
                news_info.append({
                    "title": event.title,
                    "impact": event.impact_level.value,
                    "price_impact": event.price_impact,
                    "volatility_impact": event.volatility_impact
                })
            position["active_news_at_entry"] = news_info
        
        # Store position
        self.positions.append(position)
        
        return position
        
    def close_position(
        self, 
        position_id: str, 
        timestamp: datetime, 
        price: float = None, 
        reason: str = None
    ) -> Dict[str, Any]:
        """
        Close an existing position with news-aware execution modeling.
        
        Args:
            position_id: ID of position to close
            timestamp: Exit time
            price: Exit price (None for market orders)
            reason: Reason for closing position
            
        Returns:
            Updated position details
        """
        # Find position
        position = None
        for p in self.positions:
            if p["id"] == position_id:
                position = p
                break
                
        if position is None:
            self.logger.error(f"Position with ID {position_id} not found")
            return None
            
        # Determine instrument
        instrument = position["instrument"]
        
        # Calculate spread
        spread = self.calculate_spread(timestamp, instrument)
        
        # Calculate slippage
        slippage_pips = self.calculate_slippage(timestamp, instrument, position["size"])
        
        # Determine execution price
        if price is None:
            # Use close price as default for backtest
            for df in self.historical_data.values():
                if instrument in df.columns:
                    price = df.loc[timestamp]['close']
                    break
        
        # Apply slippage and spread to price
        if position["direction"] == "long":
            execution_price = price - spread / 2 - slippage_pips
        else:  # short
            execution_price = price + spread / 2 + slippage_pips
            
        # Calculate profit
        if position["direction"] == "long":
            profit = (execution_price - position["entry_price"]) * position["size"]
        else:  # short
            profit = (position["entry_price"] - execution_price) * position["size"]
            
        # Calculate commission
        commission = self.commission * execution_price * position["size"]
        
        # Update position data
        position["exit_time"] = timestamp
        position["exit_price"] = execution_price
        position["profit"] = profit
        position["commission"] = commission
        position["exit_slippage"] = slippage_pips
        position["exit_reason"] = reason
        
        # Log active news during position closing
        active_news = self.get_active_news_events(timestamp, instrument)
        if active_news:
            news_info = []
            for event in active_news:
                news_info.append({
                    "title": event.title,
                    "impact": event.impact_level.value,
                    "price_impact": event.price_impact,
                    "volatility_impact": event.volatility_impact
                })
            position["active_news_at_exit"] = news_info
        
        # Move to closed positions
        self.positions = [p for p in self.positions if p["id"] != position_id]
        self.closed_positions.append(position)
        
        # Update balance
        self.balance = self.balance + profit - commission
        self.balance_history.append(self.balance)
        
        return position
        
    def analyze_performance_during_news(self) -> Dict[str, Any]:
        """
        Analyze trading performance during news events versus normal periods.
        
        Returns:
            Dictionary with news performance analysis
        """
        if not self.closed_positions:
            return {"error": "No closed positions to analyze"}
            
        # Separate trades into news-affected and normal periods
        news_trades = []
        normal_trades = []
        
        for trade in self.closed_positions:
            # Check for news at entry or exit
            if trade.get("active_news_at_entry") or trade.get("active_news_at_exit"):
                news_trades.append(trade)
            else:
                normal_trades.append(trade)
                
        # Calculate metrics for each group
        news_metrics = self._calculate_trade_group_metrics(news_trades)
        normal_metrics = self._calculate_trade_group_metrics(normal_trades)
        
        # Calculate metrics by news impact level
        impact_level_trades = {
            "low": [],
            "medium": [],
            "high": [],
            "critical": []
        }
        
        for trade in news_trades:
            # Find the highest impact level of news during this trade
            max_impact = "low"
            
            for news_event in trade.get("active_news_at_entry", []) + trade.get("active_news_at_exit", []):
                impact = news_event.get("impact", "low")
                if impact == "critical":
                    max_impact = "critical"
                    break
                elif impact == "high" and max_impact != "critical":
                    max_impact = "high"
                elif impact == "medium" and max_impact not in ["critical", "high"]:
                    max_impact = "medium"
                    
            impact_level_trades[max_impact].append(trade)
                
        # Calculate metrics by impact level
        impact_metrics = {}
        for level, trades in impact_level_trades.items():
            impact_metrics[level] = self._calculate_trade_group_metrics(trades)
            
        return {
            "news_trades_count": len(news_trades),
            "normal_trades_count": len(normal_trades),
            "news_trades_metrics": news_metrics,
            "normal_trades_metrics": normal_metrics,
            "impact_level_metrics": impact_metrics
        }
        
    def _calculate_trade_group_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for a group of trades."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "profit_factor": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0,
                "net_profit": 0.0
            }
            
        # Basic counts
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t["profit"] > 0])
        losing_trades = len([t for t in trades if t["profit"] < 0])
        
        # PnL statistics
        total_profit = sum([t["profit"] for t in trades if t["profit"] > 0])
        total_loss = sum([t["profit"] for t in trades if t["profit"] < 0])
        net_profit = sum([t["profit"] for t in trades])
        
        # Performance ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "profit_factor": profit_factor,
            "average_profit": avg_profit,
            "average_loss": avg_loss,
            "net_profit": net_profit
        }
