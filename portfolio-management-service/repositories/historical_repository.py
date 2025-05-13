"""
Historical Repository Module.

Repository for accessing and storing historical portfolio data.
"""
from typing import List, Optional
from datetime import datetime, timezone
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from core.historical import PortfolioSnapshot, PerformanceRecord
from core_foundations.utils.logger import get_logger

logger = get_logger("historical-repository")


class HistoricalRepository:
    """Repository for historical portfolio data."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async Database session
        """
        self.session = session
        
    async def create_snapshot(self, snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
        """
        Create a new portfolio snapshot.
        
        Args:
            snapshot: Portfolio snapshot to create
            
        Returns:
            Created snapshot with assigned ID
        """
        # Convert Pydantic model to dict
        snapshot_dict = snapshot.dict(exclude={"id"})
        
        # Create SQL query
        query = """
        INSERT INTO portfolio_snapshots 
        (account_id, timestamp, balance, equity, open_positions_count, margin_used, free_margin, unrealized_pnl)
        VALUES (%(account_id)s, %(timestamp)s, %(balance)s, %(equity)s, %(open_positions_count)s, 
                %(margin_used)s, %(free_margin)s, %(unrealized_pnl)s)
        RETURNING id
        """
        
        # Execute query
        result = await self.session.execute(query, snapshot_dict)
        snapshot_id = result.fetchone()[0]
        
        # Commit the transaction
        await self.session.commit()
        
        # Update model with ID
        snapshot.id = snapshot_id
        
        logger.info(f"Created portfolio snapshot with ID {snapshot_id}")
        return snapshot
    
    async def create_performance_record(self, record: PerformanceRecord) -> PerformanceRecord:
        """
        Create a new performance record.
        
        Args:
            record: Performance record to create
            
        Returns:
            Created record with assigned ID
        """
        # Convert Pydantic model to dict
        record_dict = record.dict(exclude={"id"})
        
        # Create SQL query
        query = """
        INSERT INTO performance_records 
        (account_id, timestamp, period, win_rate, profit_factor, net_profit, total_trades, sharpe_ratio)
        VALUES (%(account_id)s, %(timestamp)s, %(period)s, %(win_rate)s, %(profit_factor)s, 
                %(net_profit)s, %(total_trades)s, %(sharpe_ratio)s)
        RETURNING id
        """
        
        # Execute query
        result = await self.session.execute(query, record_dict)
        record_id = result.fetchone()[0]
        
        # Commit the transaction
        await self.session.commit()
        
        # Update model with ID
        record.id = record_id
        
        logger.info(f"Created performance record with ID {record_id}")
        return record
    
    async def get_snapshots(self, account_id: str, start_date: datetime, end_date: datetime) -> List[PortfolioSnapshot]:
        """
        Get portfolio snapshots within a date range.
        
        Args:
            account_id: Account ID to get snapshots for
            start_date: Start date
            end_date: End date
            
        Returns:
            List of portfolio snapshots
        """
        query = """
        SELECT id, account_id, timestamp, balance, equity, open_positions_count, 
               margin_used, free_margin, unrealized_pnl
        FROM portfolio_snapshots
        WHERE account_id = %(account_id)s
          AND timestamp >= %(start_date)s
          AND timestamp <= %(end_date)s
        ORDER BY timestamp
        """
        
        params = {
            "account_id": account_id,
            "start_date": start_date,
            "end_date": end_date
        }
        
        result = await self.session.execute(query, params)
        
        snapshots = []
        for row in result:
            snapshot = PortfolioSnapshot(
                id=row[0],
                account_id=row[1],
                timestamp=row[2],
                balance=row[3],
                equity=row[4],
                open_positions_count=row[5],
                margin_used=row[6],
                free_margin=row[7],
                unrealized_pnl=row[8]
            )
            snapshots.append(snapshot)
        
        logger.info(f"Retrieved {len(snapshots)} portfolio snapshots for account {account_id}")
        return snapshots
    
    async def get_performance_records(self, account_id: str, start_date: datetime, end_date: datetime) -> List[PerformanceRecord]:
        """
        Get performance records within a date range.
        
        Args:
            account_id: Account ID to get records for
            start_date: Start date
            end_date: End date
            
        Returns:
            List of performance records
        """
        query = """
        SELECT id, account_id, timestamp, period, win_rate, profit_factor, 
               net_profit, total_trades, sharpe_ratio
        FROM performance_records
        WHERE account_id = %(account_id)s
          AND timestamp >= %(start_date)s
          AND timestamp <= %(end_date)s
        ORDER BY timestamp
        """
        
        params = {
            "account_id": account_id,
            "start_date": start_date,
            "end_date": end_date
        }
        
        result = await self.session.execute(query, params)
        
        records = []
        for row in result:
            record = PerformanceRecord(
                id=row[0],
                account_id=row[1],
                timestamp=row[2],
                period=row[3],
                win_rate=row[4],
                profit_factor=row[5],
                net_profit=row[6],
                total_trades=row[7],
                sharpe_ratio=row[8]
            )
            records.append(record)
        
        logger.info(f"Retrieved {len(records)} performance records for account {account_id}")
        return records
    
    async def get_latest_snapshot(self, account_id: str) -> Optional[PortfolioSnapshot]:
        """
        Get the latest portfolio snapshot for an account.
        
        Args:
            account_id: Account ID to get snapshot for
            
        Returns:
            Latest portfolio snapshot or None if no snapshots exist
        """
        query = """
        SELECT id, account_id, timestamp, balance, equity, open_positions_count, 
               margin_used, free_margin, unrealized_pnl
        FROM portfolio_snapshots
        WHERE account_id = %(account_id)s
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        params = {"account_id": account_id}
        
        result = await self.session.execute(query, params)
        row = result.fetchone()
        
        if not row:
            logger.warning(f"No portfolio snapshots found for account {account_id}")
            return None
        
        snapshot = PortfolioSnapshot(
            id=row[0],
            account_id=row[1],
            timestamp=row[2],
            balance=row[3],
            equity=row[4],
            open_positions_count=row[5],
            margin_used=row[6],
            free_margin=row[7],
            unrealized_pnl=row[8]
        )
        
        return snapshot
