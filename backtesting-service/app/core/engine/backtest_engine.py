"""
Backtesting Engine

This module provides the core backtesting engine for the backtesting service.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from app.models.backtest_models import (
    BacktestRequest,
    BacktestResult,
    TradeResult,
    PerformanceMetrics
)

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for running backtests on trading strategies.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtest engine.
        
        Args:
            config: Configuration parameters for the engine
        """
        self.config = config or {}
        self.commission_rate = self.config.get('commission_rate', 0.0001)
        self.slippage = self.config.get('slippage', 0.0001)
        self.initial_balance = self.config.get('initial_balance', 10000.0)
    
    async def run_backtest(self, strategy: Any, data: pd.DataFrame, 
                         parameters: Dict[str, Any]) -> BacktestResult:
        """
        Run a backtest for a trading strategy.
        
        Args:
            strategy: The trading strategy to backtest
            data: Market data for backtesting
            parameters: Parameters for the backtest
            
        Returns:
            BacktestResult: The result of the backtest
        """
        # Initialize backtest state
        balance = self.initial_balance
        positions = []
        trades = []
        equity_curve = []
        
        # Apply strategy to data
        signals = self._apply_strategy(strategy, data, parameters)
        
        # Process signals and generate trades
        for i, (timestamp, signal) in enumerate(signals.items()):
            # Skip if no signal
            if signal == 0:
                # Update equity curve
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': balance + self._calculate_unrealized_pnl(positions, data, timestamp)
                })
                continue
            
            # Close positions if signal is opposite to current position
            for position in positions[:]:
                if (position['direction'] == 'long' and signal < 0) or \
                   (position['direction'] == 'short' and signal > 0):
                    # Close position
                    trade = self._close_position(position, data, timestamp)
                    trades.append(trade)
                    balance += trade['pnl']
                    positions.remove(position)
            
            # Open new position if signal is non-zero
            if signal != 0:
                # Calculate position size
                position_size = self._calculate_position_size(balance, data, timestamp, parameters)
                
                # Open position
                position = self._open_position(
                    direction='long' if signal > 0 else 'short',
                    size=position_size,
                    data=data,
                    timestamp=timestamp
                )
                positions.append(position)
            
            # Update equity curve
            equity_curve.append({
                'timestamp': timestamp,
                'equity': balance + self._calculate_unrealized_pnl(positions, data, timestamp)
            })
        
        # Close any remaining positions at the end of the backtest
        last_timestamp = data.index[-1]
        for position in positions[:]:
            trade = self._close_position(position, data, last_timestamp)
            trades.append(trade)
            balance += trade['pnl']
            positions.remove(position)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(trades, equity_curve, parameters)
        
        # Create backtest result
        backtest_result = BacktestResult(
            backtest_id=parameters.get('backtest_id', ''),
            strategy_id=parameters.get('strategy_id', ''),
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_balance=self.initial_balance,
            final_balance=balance,
            total_trades=len(trades),
            winning_trades=sum(1 for trade in trades if trade['pnl'] > 0),
            losing_trades=sum(1 for trade in trades if trade['pnl'] <= 0),
            performance_metrics=performance_metrics,
            trades=trades,
            equity_curve=equity_curve,
            parameters=parameters
        )
        
        return backtest_result
    
    def _apply_strategy(self, strategy: Any, data: pd.DataFrame, 
                      parameters: Dict[str, Any]) -> Dict[datetime, int]:
        """
        Apply a trading strategy to market data.
        
        Args:
            strategy: The trading strategy to apply
            data: Market data
            parameters: Strategy parameters
            
        Returns:
            Dict[datetime, int]: Trading signals for each timestamp
        """
        # This is a placeholder implementation
        # In a real scenario, this would apply the actual strategy logic
        
        # For demonstration, we'll use a simple moving average crossover strategy
        short_window = parameters.get('short_window', 10)
        long_window = parameters.get('long_window', 50)
        
        # Calculate moving averages
        data['short_ma'] = data['close'].rolling(window=short_window).mean()
        data['long_ma'] = data['close'].rolling(window=long_window).mean()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1  # Buy signal
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1  # Sell signal
        
        # Convert to dictionary
        signals = data['signal'].to_dict()
        
        return signals
    
    def _open_position(self, direction: str, size: float, data: pd.DataFrame, 
                     timestamp: datetime) -> Dict[str, Any]:
        """
        Open a new trading position.
        
        Args:
            direction: Direction of the position ('long' or 'short')
            size: Size of the position
            data: Market data
            timestamp: Timestamp for the position
            
        Returns:
            Dict[str, Any]: The opened position
        """
        # Get price at timestamp
        price = data.loc[timestamp, 'close']
        
        # Apply slippage
        if direction == 'long':
            entry_price = price * (1 + self.slippage)
        else:
            entry_price = price * (1 - self.slippage)
        
        # Calculate commission
        commission = size * entry_price * self.commission_rate
        
        # Create position
        position = {
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'commission': commission
        }
        
        return position
    
    def _close_position(self, position: Dict[str, Any], data: pd.DataFrame, 
                      timestamp: datetime) -> Dict[str, Any]:
        """
        Close a trading position.
        
        Args:
            position: The position to close
            data: Market data
            timestamp: Timestamp for closing the position
            
        Returns:
            Dict[str, Any]: The trade result
        """
        # Get price at timestamp
        price = data.loc[timestamp, 'close']
        
        # Apply slippage
        if position['direction'] == 'long':
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)
        
        # Calculate commission
        commission = position['size'] * exit_price * self.commission_rate
        
        # Calculate PnL
        if position['direction'] == 'long':
            pnl = position['size'] * (exit_price - position['entry_price'])
        else:
            pnl = position['size'] * (position['entry_price'] - exit_price)
        
        # Subtract commissions
        pnl -= (position['commission'] + commission)
        
        # Create trade result
        trade = {
            'direction': position['direction'],
            'size': position['size'],
            'entry_price': position['entry_price'],
            'entry_time': position['entry_time'],
            'exit_price': exit_price,
            'exit_time': timestamp,
            'pnl': pnl,
            'commission': position['commission'] + commission,
            'duration': (timestamp - position['entry_time']).total_seconds() / 3600  # Duration in hours
        }
        
        return trade
    
    def _calculate_unrealized_pnl(self, positions: List[Dict[str, Any]], 
                                data: pd.DataFrame, timestamp: datetime) -> float:
        """
        Calculate unrealized PnL for open positions.
        
        Args:
            positions: List of open positions
            data: Market data
            timestamp: Current timestamp
            
        Returns:
            float: Unrealized PnL
        """
        unrealized_pnl = 0.0
        
        # Get price at timestamp
        price = data.loc[timestamp, 'close']
        
        for position in positions:
            if position['direction'] == 'long':
                unrealized_pnl += position['size'] * (price - position['entry_price'])
            else:
                unrealized_pnl += position['size'] * (position['entry_price'] - price)
            
            # Subtract commission for entry (exit commission will be applied when position is closed)
            unrealized_pnl -= position['commission']
        
        return unrealized_pnl
    
    def _calculate_position_size(self, balance: float, data: pd.DataFrame, 
                               timestamp: datetime, parameters: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk management parameters.
        
        Args:
            balance: Current account balance
            data: Market data
            timestamp: Current timestamp
            parameters: Risk management parameters
            
        Returns:
            float: Position size
        """
        # Get risk percentage
        risk_percentage = parameters.get('risk_percentage', 0.02)  # Default to 2% risk
        
        # Get price at timestamp
        price = data.loc[timestamp, 'close']
        
        # Calculate position size based on risk
        position_size = (balance * risk_percentage) / price
        
        return position_size
    
    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]], 
                                     equity_curve: List[Dict[str, Any]], 
                                     parameters: Dict[str, Any]) -> PerformanceMetrics:
        """
        Calculate performance metrics for the backtest.
        
        Args:
            trades: List of trades
            equity_curve: Equity curve
            parameters: Backtest parameters
            
        Returns:
            PerformanceMetrics: Performance metrics
        """
        # Extract equity values
        equity_values = [point['equity'] for point in equity_curve]
        
        # Calculate returns
        returns = pd.Series(equity_values).pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_values[-1] / equity_values[0]) - 1 if equity_values else 0
        
        # Calculate annualized return
        days = (equity_curve[-1]['timestamp'] - equity_curve[0]['timestamp']).days if equity_curve else 0
        annualized_return = (1 + total_return) ** (365 / max(1, days)) - 1 if days > 0 else 0
        
        # Calculate Sharpe ratio
        risk_free_rate = parameters.get('risk_free_rate', 0.0)
        sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() if len(returns) > 0 and returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        drawdowns = 1 - cumulative_returns / cumulative_returns.cummax()
        max_drawdown = drawdowns.max() if not drawdowns.empty else 0
        
        # Calculate win rate
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Calculate average trade
        average_trade = sum(t['pnl'] for t in trades) / len(trades) if trades else 0
        
        # Calculate average winning trade
        average_winning_trade = sum(t['pnl'] for t in trades if t['pnl'] > 0) / len([t for t in trades if t['pnl'] > 0]) if [t for t in trades if t['pnl'] > 0] else 0
        
        # Calculate average losing trade
        average_losing_trade = sum(t['pnl'] for t in trades if t['pnl'] <= 0) / len([t for t in trades if t['pnl'] <= 0]) if [t for t in trades if t['pnl'] <= 0] else 0
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_trade=average_trade,
            average_winning_trade=average_winning_trade,
            average_losing_trade=average_losing_trade
        )
        
        return performance_metrics