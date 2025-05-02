"""
Portfolio Performance Metrics Module.

Provides functionality for calculating performance metrics for trading portfolios.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core_foundations.utils.logger import get_logger
from portfolio_management_service.models.position import Position, PositionStatus

logger = get_logger("portfolio-performance")


class PerformanceMetrics:
    """Performance metrics calculation for portfolios."""
    
    @staticmethod
    def calculate_position_metrics(position: Position) -> Dict[str, Any]:
        """
        Calculate performance metrics for a single position.
        
        Args:
            position: Position data
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Skip positions that haven't been closed yet
        if position.status != PositionStatus.CLOSED:
            return {
                "realized_pnl": 0,
                "pnl_percentage": 0,
                "holding_period_hours": 0,
                "win": False
            }
        
        # Calculate realized PnL
        metrics["realized_pnl"] = position.realized_pnl
        
        # Calculate PnL as percentage of position size
        if position.size > 0:
            metrics["pnl_percentage"] = (position.realized_pnl / (position.entry_price * position.size)) * 100
        else:
            metrics["pnl_percentage"] = 0
        
        # Calculate holding period
        if position.entry_time and position.exit_time:
            holding_period = position.exit_time - position.entry_time
            metrics["holding_period_hours"] = holding_period.total_seconds() / 3600
        else:
            metrics["holding_period_hours"] = 0
        
        # Determine if position was a win
        metrics["win"] = position.realized_pnl > 0
        
        return metrics
    
    @staticmethod
    def calculate_overall_metrics(positions: List[Position]) -> Dict[str, Any]:
        """
        Calculate portfolio-wide performance metrics.
        
        Args:
            positions: List of positions to analyze
            
        Returns:
            Dictionary with overall performance metrics
        """
        # Skip analysis if no positions
        if not positions:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "average_holding_time_hours": 0,
                "net_profit": 0,
                "sharpe_ratio": 0
            }
        
        # Filter only closed positions
        closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
        
        if not closed_positions:
            # Return empty metrics if no closed positions
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "average_holding_time_hours": 0,
                "net_profit": 0,
                "sharpe_ratio": 0
            }
        
        # Calculate win rate
        wins = [p for p in closed_positions if p.realized_pnl > 0]
        losses = [p for p in closed_positions if p.realized_pnl <= 0]
        
        win_rate = len(wins) / len(closed_positions) if closed_positions else 0
        
        # Calculate profit factor
        total_profit = sum(p.realized_pnl for p in wins)
        total_loss = abs(sum(p.realized_pnl for p in losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate average win and loss
        average_win = total_profit / len(wins) if wins else 0
        average_loss = total_loss / len(losses) if losses else 0
        
        # Calculate largest win and loss
        largest_win = max([p.realized_pnl for p in wins]) if wins else 0
        largest_loss = min([p.realized_pnl for p in losses]) if losses else 0
        
        # Calculate average holding time
        holding_times = []
        for p in closed_positions:
            if p.entry_time and p.exit_time:
                holding_time = (p.exit_time - p.entry_time).total_seconds() / 3600
                holding_times.append(holding_time)
        
        average_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
        
        # Calculate net profit
        net_profit = sum(p.realized_pnl for p in closed_positions)
        
        # Calculate Sharpe ratio (simplified)
        daily_returns = []
        for p in closed_positions:
            if p.exit_time:
                daily_returns.append(p.realized_pnl)
        
        if daily_returns:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns) if len(daily_returns) > 1 else 1
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            "total_trades": len(closed_positions),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "average_holding_time_hours": average_holding_time,
            "net_profit": net_profit,
            "sharpe_ratio": sharpe_ratio
        }
    
    @staticmethod
    def calculate_metrics_by_period(positions: List[Position], 
                                   period_type: str = 'monthly') -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics broken down by time periods.
        
        Args:
            positions: List of positions
            period_type: Type of period breakdown ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary with metrics by period
        """
        # Create a DataFrame for easier analysis
        position_data = []
        
        for p in positions:
            if p.status == PositionStatus.CLOSED and p.entry_time and p.exit_time:
                position_data.append({
                    'entry_time': p.entry_time,
                    'exit_time': p.exit_time,
                    'realized_pnl': p.realized_pnl
                })
        
        if not position_data:
            return {}
        
        df = pd.DataFrame(position_data)
        
        # Add period columns
        if period_type == 'daily':
            df['period'] = df['exit_time'].dt.strftime('%Y-%m-%d')
        elif period_type == 'weekly':
            df['period'] = df['exit_time'].dt.to_period('W').apply(lambda x: str(x))
        else:  # monthly
            df['period'] = df['exit_time'].dt.strftime('%Y-%m')
        
        # Group by period
        result = {}
        
        for period, group in df.groupby('period'):
            wins = len(group[group['realized_pnl'] > 0])
            total = len(group)
            win_rate = wins / total if total > 0 else 0
            net_profit = group['realized_pnl'].sum()
            
            result[period] = {
                'trades': total,
                'win_rate': win_rate,
                'net_profit': net_profit
            }
        
        return result
    
    @staticmethod
    def calculate_drawdown(positions: List[Position], account_balance: float) -> Dict[str, Any]:
        """
        Calculate maximum drawdown metrics.
        
        Args:
            positions: List of positions
            account_balance: Current account balance
            
        Returns:
            Dictionary with drawdown metrics
        """
        if not positions:
            return {
                "max_drawdown_amount": 0,
                "max_drawdown_percentage": 0,
                "current_drawdown_amount": 0,
                "current_drawdown_percentage": 0
            }
        
        # Create a series of cumulative PnL to calculate drawdown
        position_data = []
        
        for p in positions:
            if p.status == PositionStatus.CLOSED and p.exit_time:
                position_data.append({
                    'time': p.exit_time,
                    'pnl': p.realized_pnl
                })
        
        if not position_data:
            return {
                "max_drawdown_amount": 0,
                "max_drawdown_percentage": 0,
                "current_drawdown_amount": 0,
                "current_drawdown_percentage": 0
            }
        
        # Sort by time
        position_data.sort(key=lambda x: x['time'])
        
        # Calculate cumulative PnL and drawdown
        cumulative_pnl = []
        running_pnl = 0
        
        for p in position_data:
            running_pnl += p['pnl']
            cumulative_pnl.append(running_pnl)
        
        # Calculate drawdown
        max_cumulative = 0
        max_drawdown = 0
        
        for pnl in cumulative_pnl:
            max_cumulative = max(max_cumulative, pnl)
            drawdown = max_cumulative - pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate current drawdown
        current_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
        max_cumulative = max(max(cumulative_pnl) if cumulative_pnl else 0, current_pnl)
        current_drawdown = max_cumulative - current_pnl
        
        # Calculate percentages
        initial_balance = account_balance - current_pnl
        
        if initial_balance > 0:
            max_drawdown_percentage = (max_drawdown / initial_balance) * 100
            current_drawdown_percentage = (current_drawdown / initial_balance) * 100
        else:
            max_drawdown_percentage = 0
            current_drawdown_percentage = 0
        
        return {
            "max_drawdown_amount": max_drawdown,
            "max_drawdown_percentage": max_drawdown_percentage,
            "current_drawdown_amount": current_drawdown,
            "current_drawdown_percentage": current_drawdown_percentage
        }
