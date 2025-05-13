"""
Position Repository Module.

Repository for accessing and managing trading positions.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from core.position import (
    Position, PositionCreate, PositionUpdate, PositionStatus
)
from core_foundations.utils.logger import get_logger

logger = get_logger("position-repository")


class PositionRepository:
    """Repository for position data."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async database session
        """
        self.session = session
        
    async def create(self, position_data: PositionCreate) -> Position:
        """
        Create a new position.
        
        Args:
            position_data: Position data to create
            
        Returns:
            Created position with assigned ID
        """
        # Convert Pydantic model to dict
        data = position_data.dict(exclude={"id"})
        
        # Initialize position based on provided data
        if "entry_time" not in data or data["entry_time"] is None:
            data["entry_time"] = datetime.now(timezone.utc)
        
        if "status" not in data or data["status"] is None:
            data["status"] = PositionStatus.OPEN.value
        
        if "unrealized_pnl" not in data:
            data["unrealized_pnl"] = 0.0
        
        if "realized_pnl" not in data:
            data["realized_pnl"] = 0.0
        
        # Create SQL query
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f"%({k})s" for k in data.keys()])
        
        query = f"""
        INSERT INTO positions ({columns})
        VALUES ({placeholders})
        RETURNING id
        """
        
        # Execute query
        result = await self.session.execute(query, data)
        position_id = result.fetchone()[0]
        
        # Commit the transaction
        await self.session.commit()
        
        # Get created position
        position = await self.get_by_id(position_id)
        logger.info(f"Created position {position_id} for {position_data.symbol}")
        
        return position
    
    async def get_by_id(self, position_id: str) -> Optional[Position]:
        """
        Get position by ID.
        
        Args:
            position_id: Position ID to find
            
        Returns:
            Position if found, None otherwise
        """
        query = """
        SELECT 
            id, 
            account_id, 
            symbol, 
            direction, 
            size, 
            entry_price, 
            exit_price, 
            stop_loss, 
            take_profit, 
            entry_time, 
            exit_time, 
            status, 
            unrealized_pnl, 
            realized_pnl, 
            meta_data
        FROM positions
        WHERE id = %(position_id)s
        """
        
        result = await self.session.execute(query, {"position_id": position_id})
        row = result.fetchone()
        
        if not row:
            logger.warning(f"Position {position_id} not found")
            return None
        
        # Convert row to Position object
        position = Position(
            id=row[0],
            account_id=row[1],
            symbol=row[2],
            direction=row[3],
            size=row[4],
            entry_price=row[5],
            exit_price=row[6],
            stop_loss=row[7],
            take_profit=row[8],
            entry_time=row[9],
            exit_time=row[10],
            status=row[11],
            unrealized_pnl=row[12],
            realized_pnl=row[13],
            meta_data=row[14] or {}
        )
        
        return position
    
    async def update(self, position_id: str, update_data: PositionUpdate) -> Optional[Position]:
        """
        Update position with new data.
        
        Args:
            position_id: ID of position to update
            update_data: New data for position
            
        Returns:
            Updated position or None if position not found
        """
        # Get current position
        position = await self.get_by_id(position_id)
        if not position:
            logger.error(f"Cannot update position {position_id}: not found")
            return None
        
        # Convert update data to dict and remove None values
        data = update_data.dict(exclude_unset=True)
        if not data:
            logger.warning(f"No data provided to update position {position_id}")
            return position
        
        # Build SET clause
        set_clause = ", ".join([f"{k} = %({k})s" for k in data.keys()])
        query = f"""
        UPDATE positions
        SET {set_clause}
        WHERE id = %(position_id)s
        """
        
        # Add position ID to data
        data["position_id"] = position_id
        
        # Execute update
        await self.session.execute(query, data)
        
        # Commit the transaction
        await self.session.commit()
        
        # Get updated position
        updated = await self.get_by_id(position_id)
        logger.info(f"Updated position {position_id}")
        
        return updated
    
    async def close_position(self, position_id: str, exit_price: float) -> Optional[Position]:
        """
        Close an open position.
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price for the position
            
        Returns:
            Updated position or None if position not found
        """
        # Get position
        position = await self.get_by_id(position_id)
        if not position:
            logger.error(f"Cannot close position {position_id}: not found")
            return None
        
        if position.status != PositionStatus.OPEN:
            logger.warning(f"Position {position_id} is already {position.status}, cannot close")
            return position
        
        # Calculate realized PnL
        if position.direction == "BUY":
            realized_pnl = (exit_price - position.entry_price) * position.size
        else:  # SELL
            realized_pnl = (position.entry_price - exit_price) * position.size
        
        # Update position
        update_data = PositionUpdate(
            exit_price=exit_price,
            exit_time=datetime.now(timezone.utc),
            status=PositionStatus.CLOSED,
            realized_pnl=realized_pnl,
            unrealized_pnl=0.0
        )
        
        updated = await self.update(position_id, update_data)
        logger.info(f"Closed position {position_id} with realized PnL: {realized_pnl}")
        
        return updated
    
    async def get_open_positions(self, account_id: str) -> List[Position]:
        """
        Get all open positions for an account.
        
        Args:
            account_id: Account ID to filter by
            
        Returns:
            List of open positions
        """
        query = """
        SELECT 
            id, 
            account_id, 
            symbol, 
            direction, 
            size, 
            entry_price, 
            exit_price, 
            stop_loss, 
            take_profit, 
            entry_time, 
            exit_time, 
            status, 
            unrealized_pnl, 
            realized_pnl, 
            meta_data
        FROM positions
        WHERE account_id = %(account_id)s
        AND status = 'OPEN'
        """
        
        result = await self.session.execute(query, {"account_id": account_id})
        
        positions = []
        for row in result:
            position = Position(
                id=row[0],
                account_id=row[1],
                symbol=row[2],
                direction=row[3],
                size=row[4],
                entry_price=row[5],
                exit_price=row[6],
                stop_loss=row[7],
                take_profit=row[8],
                entry_time=row[9],
                exit_time=row[10],
                status=row[11],
                unrealized_pnl=row[12],
                realized_pnl=row[13],
                meta_data=row[14] or {}
            )
            positions.append(position)
        
        logger.info(f"Retrieved {len(positions)} open positions for account {account_id}")
        return positions
    
    async def get_closed_positions(self, account_id: str, start_date: datetime = None, end_date: datetime = None) -> List[Position]:
        """
        Get closed positions for an account in a date range.
        
        Args:
            account_id: Account ID to filter by
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            List of closed positions
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)  # Default to last 30 days
            
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        query = """
        SELECT 
            id, 
            account_id, 
            symbol, 
            direction, 
            size, 
            entry_price, 
            exit_price, 
            stop_loss, 
            take_profit, 
            entry_time, 
            exit_time, 
            status, 
            unrealized_pnl, 
            realized_pnl, 
            meta_data
        FROM positions
        WHERE account_id = %(account_id)s
        AND status = 'CLOSED'
        AND exit_time >= %(start_date)s
        AND exit_time <= %(end_date)s
        """
        
        params = {
            "account_id": account_id,
            "start_date": start_date,
            "end_date": end_date
        }
        
        result = await self.session.execute(query, params)
        
        positions = []
        for row in result:
            position = Position(
                id=row[0],
                account_id=row[1],
                symbol=row[2],
                direction=row[3],
                size=row[4],
                entry_price=row[5],
                exit_price=row[6],
                stop_loss=row[7],
                take_profit=row[8],
                entry_time=row[9],
                exit_time=row[10],
                status=row[11],
                unrealized_pnl=row[12],
                realized_pnl=row[13],
                meta_data=row[14] or {}
            )
            positions.append(position)
        
        logger.info(f"Retrieved {len(positions)} closed positions for account {account_id} from {start_date} to {end_date}")
        return positions
    
    async def update_unrealized_pnl(self, position_id: str, current_price: float) -> float:
        """
        Update unrealized PnL for a position based on current price.
        
        Args:
            position_id: Position ID to update
            current_price: Current market price
            
        Returns:
            Updated unrealized PnL
        """
        # Get position
        position = await self.get_by_id(position_id)
        if not position or position.status != PositionStatus.OPEN:
            logger.warning(f"Cannot update unrealized PnL: Position {position_id} not found or not open")
            return 0.0
        
        # Calculate unrealized PnL
        if position.direction == "BUY":
            unrealized_pnl = (current_price - position.entry_price) * position.size
        else:  # SELL
            unrealized_pnl = (position.entry_price - current_price) * position.size
        
        # Update position
        query = """
        UPDATE positions
        SET unrealized_pnl = %(unrealized_pnl)s
        WHERE id = %(position_id)s
        """
        
        params = {
            "position_id": position_id,
            "unrealized_pnl": unrealized_pnl
        }
        
        await self.session.execute(query, params)
        
        # Commit the transaction
        await self.session.commit()
        
        logger.debug(f"Updated unrealized PnL for position {position_id}: {unrealized_pnl}")
        
        return unrealized_pnl
    
    async def get_positions_by_symbol(self, account_id: str, symbol: str) -> List[Position]:
        """
        Get all positions for an account filtered by symbol.
        
        Args:
            account_id: Account ID to filter by
            symbol: Symbol to filter by
            
        Returns:
            List of positions for the symbol
        """
        query = """
        SELECT 
            id, 
            account_id, 
            symbol, 
            direction, 
            size, 
            entry_price, 
            exit_price, 
            stop_loss, 
            take_profit, 
            entry_time, 
            exit_time, 
            status, 
            unrealized_pnl, 
            realized_pnl, 
            meta_data
        FROM positions
        WHERE account_id = %(account_id)s
        AND symbol = %(symbol)s
        """
        
        params = {
            "account_id": account_id,
            "symbol": symbol
        }
        
        result = await self.session.execute(query, params)
        
        positions = []
        for row in result:
            position = Position(
                id=row[0],
                account_id=row[1],
                symbol=row[2],
                direction=row[3],
                size=row[4],
                entry_price=row[5],
                exit_price=row[6],
                stop_loss=row[7],
                take_profit=row[8],
                entry_time=row[9],
                exit_time=row[10],
                status=row[11],
                unrealized_pnl=row[12],
                realized_pnl=row[13],
                meta_data=row[14] or {}
            )
            positions.append(position)
            
        return positions
