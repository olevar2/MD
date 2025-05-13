"""
Historical Portfolio Tracking Module.

Provides functionality for tracking and analyzing historical portfolio performance.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, date, timezone
import pandas as pd

from core_foundations.utils.logger import get_logger
from core.connection import get_db_session
from repositories.historical_repository import HistoricalRepository
from core.historical import PortfolioSnapshot, PerformanceRecord
from core.performance_metrics import PerformanceMetrics

logger = get_logger("portfolio-historical")


class HistoricalTracking:
    """Historical tracking service for portfolio performance."""
    
    def __init__(self):
        """Initialize the historical tracking service."""
        self.metrics_calculator = PerformanceMetrics()
    
    async def create_daily_snapshot(self, account_id: str, timestamp: Optional[datetime] = None) -> PortfolioSnapshot:
        """
        Create a daily snapshot of portfolio state for historical tracking.
        
        Args:
            account_id: ID of the account to snapshot
            timestamp: Optional timestamp for the snapshot (defaults to current time)
            
        Returns:
            Created snapshot
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        async with get_db_session() as session:
            hist_repo = HistoricalRepository(session)
            
            # Get current account state
            from repositories.account_repository import AccountRepository
            from repositories.position_repository import PositionRepository
            
            account_repo = AccountRepository(session)
            position_repo = PositionRepository(session)
            
            account = await account_repo.get_by_id(account_id)
            if not account:
                logger.error(f"Account {account_id} not found for snapshot creation")
                raise ValueError(f"Account {account_id} not found")
            
            # Get open positions
            open_positions = await position_repo.get_open_positions(account_id)
            closed_positions = await position_repo.get_closed_positions(
                account_id,
                start_date=timestamp - timedelta(days=30),  # Last 30 days of closed positions
                end_date=timestamp
            )
            
            # Calculate unrealized PnL for open positions
            unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
            
            # Create snapshot object
            snapshot = PortfolioSnapshot(
                account_id=account_id,
                timestamp=timestamp,
                balance=account.balance,
                equity=account.balance + unrealized_pnl,
                open_positions_count=len(open_positions),
                margin_used=account.margin_used,
                free_margin=account.balance - account.margin_used,
                unrealized_pnl=unrealized_pnl
            )
            
            # Calculate performance metrics for the last 30 days
            if closed_positions:
                metrics = self.metrics_calculator.calculate_overall_metrics(closed_positions)
                
                # Create performance record
                performance = PerformanceRecord(
                    account_id=account_id,
                    timestamp=timestamp,
                    period="30D",  # 30-day rolling period
                    win_rate=metrics["win_rate"],
                    profit_factor=metrics["profit_factor"],
                    net_profit=metrics["net_profit"],
                    total_trades=metrics["total_trades"],
                    sharpe_ratio=metrics["sharpe_ratio"]
                )
                
                # Save performance record
                await hist_repo.create_performance_record(performance)
            
            # Save and return snapshot
            return await hist_repo.create_snapshot(snapshot)
    
    async def get_historical_equity(self, account_id: str, 
                             start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
        """
        Get historical equity curve data.
        
        Args:
            account_id: Account ID to get data for
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical equity data
        """
        async with get_db_session() as session:
            hist_repo = HistoricalRepository(session)
            snapshots = await hist_repo.get_snapshots(account_id, start_date, end_date)
            
            # Convert to DataFrame for easier analysis
            data = []
            for snapshot in snapshots:
                data.append({
                    'timestamp': snapshot.timestamp,
                    'balance': snapshot.balance,
                    'equity': snapshot.equity,
                    'unrealized_pnl': snapshot.unrealized_pnl
                })
            
            if not data:
                return pd.DataFrame(columns=['timestamp', 'balance', 'equity', 'unrealized_pnl'])
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
    
    async def get_performance_metrics_history(self, account_id: str,
                                       start_date: datetime,
                                       end_date: datetime) -> pd.DataFrame:
        """
        Get historical performance metrics.
        
        Args:
            account_id: Account ID to get data for
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical performance metrics
        """
        async with get_db_session() as session:
            hist_repo = HistoricalRepository(session)
            records = await hist_repo.get_performance_records(account_id, start_date, end_date)
            
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    'timestamp': record.timestamp,
                    'period': record.period,
                    'win_rate': record.win_rate,
                    'profit_factor': record.profit_factor,
                    'net_profit': record.net_profit,
                    'total_trades': record.total_trades,
                    'sharpe_ratio': record.sharpe_ratio
                })
            
            if not data:
                return pd.DataFrame(columns=['timestamp', 'period', 'win_rate', 'profit_factor', 
                                           'net_profit', 'total_trades', 'sharpe_ratio'])
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
    
    async def analyze_drawdown_history(self, account_id: str,
                                start_date: datetime,
                                end_date: datetime) -> Dict[str, Any]:
        """
        Analyze historical drawdowns.
        
        Args:
            account_id: Account ID to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with drawdown analysis
        """
        equity_curve = await self.get_historical_equity(account_id, start_date, end_date)
        
        if equity_curve.empty:
            return {
                "max_drawdown": 0,
                "max_drawdown_percentage": 0,
                "max_drawdown_start": None,
                "max_drawdown_end": None,
                "average_drawdown": 0,
                "average_drawdown_length_days": 0,
                "drawdowns": []
            }
        
        # Analyze drawdowns
        equity = equity_curve['equity'].values
        peak = equity[0]
        max_drawdown = 0
        max_dd_pct = 0
        drawdown_start = None
        drawdown_end = None
        drawdowns = []
        
        for i, value in enumerate(equity):
            if value > peak:
                peak = value
                
                # If we were in a drawdown and now making new highs, record it
                if drawdown_start is not None:
                    drawdown_end = equity_curve.index[i-1]
                    dd_amount = peak - min(equity[drawdown_start:i])
                    dd_pct = dd_amount / peak * 100
                    
                    drawdowns.append({
                        "start": drawdown_start,
                        "end": drawdown_end,
                        "amount": dd_amount,
                        "percentage": dd_pct,
                        "duration_days": (drawdown_end - drawdown_start).days
                    })
                    
                    drawdown_start = None
                
            elif value < peak:
                # Start of a drawdown
                if drawdown_start is None:
                    drawdown_start = equity_curve.index[i]
                
                # Calculate current drawdown
                dd = peak - value
                dd_pct = dd / peak * 100
                
                # Check if this is a new max drawdown
                if dd > max_drawdown:
                    max_drawdown = dd
                    max_dd_pct = dd_pct
                    max_dd_start = drawdown_start
                    max_dd_end = equity_curve.index[i]
        
        # Handle any ongoing drawdown at the end of the data
        if drawdown_start is not None:
            drawdown_end = equity_curve.index[-1]
            dd_amount = peak - min(equity[equity_curve.index.get_loc(drawdown_start):])
            dd_pct = dd_amount / peak * 100
            
            drawdowns.append({
                "start": drawdown_start,
                "end": drawdown_end,
                "amount": dd_amount,
                "percentage": dd_pct,
                "duration_days": (drawdown_end - drawdown_start).days
            })
        
        # Calculate average drawdown metrics
        if drawdowns:
            avg_dd = sum(d["amount"] for d in drawdowns) / len(drawdowns)
            avg_dd_days = sum(d["duration_days"] for d in drawdowns) / len(drawdowns)
        else:
            avg_dd = 0
            avg_dd_days = 0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_percentage": max_dd_pct,
            "max_drawdown_start": max_dd_start,
            "max_drawdown_end": max_dd_end,
            "average_drawdown": avg_dd,
            "average_drawdown_length_days": avg_dd_days,
            "drawdowns": drawdowns
        }
