"""
Account Repository Module.

Repository for accessing and managing account data.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from core.account import AccountBalance, BalanceChange, AccountDetails
from core_foundations.utils.logger import get_logger

logger = get_logger("account-repository")


class AccountRepository:
    """Repository for account data."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async Database session
        """
        self.session = session
        
    async def get_by_id(self, account_id: str) -> Optional[AccountBalance]:
        """
        Get account balance by ID.
        
        Args:
            account_id: Account ID
            
        Returns:
            Account balance or None if not found
        """
        query = """
        SELECT id, user_id, balance, margin_used, last_updated
        FROM account_balances
        WHERE id = %(account_id)s
        """
        
        result = await self.session.execute(query, {"account_id": account_id})
        row = result.fetchone()
        
        if not row:
            logger.warning(f"Account {account_id} not found")
            return None
        
        account = AccountBalance(
            id=row[0],
            user_id=row[1],
            balance=row[2],
            margin_used=row[3],
            last_updated=row[4]
        )
        
        logger.debug(f"Retrieved account balance for {account_id}")
        return account
    
    async def get_by_user_id(self, user_id: str) -> List[AccountBalance]:
        """
        Get all accounts for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of account balances
        """
        query = """
        SELECT id, user_id, balance, margin_used, last_updated
        FROM account_balances
        WHERE user_id = %(user_id)s
        """
        
        result = await self.session.execute(query, {"user_id": user_id})
        
        accounts = []
        for row in result:
            account = AccountBalance(
                id=row[0],
                user_id=row[1],
                balance=row[2],
                margin_used=row[3],
                last_updated=row[4]
            )
            accounts.append(account)
        
        logger.debug(f"Retrieved {len(accounts)} accounts for user {user_id}")
        return accounts
    
    async def create_account(self, account: AccountBalance) -> AccountBalance:
        """
        Create a new account.
        
        Args:
            account: Account balance data
            
        Returns:
            Created account with assigned ID
        """
        query = """
        INSERT INTO account_balances (user_id, balance, margin_used, last_updated)
        VALUES (%(user_id)s, %(balance)s, %(margin_used)s, %(last_updated)s)
        RETURNING id
        """
        
        account_dict = account.dict(exclude={"id"})
        if "last_updated" not in account_dict or account_dict["last_updated"] is None:
            account_dict["last_updated"] = datetime.now(timezone.utc)
            
        result = await self.session.execute(query, account_dict)
        account_id = result.fetchone()[0]
        
        # Update account with ID
        account.id = account_id
        
        # Commit the transaction
        await self.session.commit()
        
        logger.info(f"Created new account with ID {account_id}")
        return account
    
    async def update_balance(self, account_id: str, change_amount: float, reason: str) -> AccountBalance:
        """
        Update account balance and record the change.
        
        Args:
            account_id: Account ID
            change_amount: Amount to change the balance by (positive or negative)
            reason: Reason for the balance change
            
        Returns:
            Updated account balance
        """
        # Get current balance
        account = await self.get_by_id(account_id)
        if not account:
            logger.error(f"Cannot update balance: Account {account_id} not found")
            raise ValueError(f"Account {account_id} not found")
        
        # Calculate new balance
        new_balance = account.balance + change_amount
        
        # Update balance in database
        update_query = """
        UPDATE account_balances
        SET balance = %(new_balance)s, last_updated = %(last_updated)s
        WHERE id = %(account_id)s
        """
        
        update_params = {
            "account_id": account_id,
            "new_balance": new_balance,
            "last_updated": datetime.now(timezone.utc)
        }
        
        await self.session.execute(update_query, update_params)
        
        # Record balance change
        change_query = """
        INSERT INTO balance_changes (account_id, amount, balance_before, balance_after, reason, timestamp)
        VALUES (%(account_id)s, %(amount)s, %(balance_before)s, %(balance_after)s, %(reason)s, %(timestamp)s)
        """
        
        change_params = {
            "account_id": account_id,
            "amount": change_amount,
            "balance_before": account.balance,
            "balance_after": new_balance,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc)
        }
        
        await self.session.execute(change_query, change_params)
        
        # Commit the transaction
        await self.session.commit()
        
        # Update account object
        account.balance = new_balance
        account.last_updated = update_params["last_updated"]
        
        logger.info(f"Updated balance for account {account_id}: {change_amount} ({reason})")
        return account
    
    async def update_margin(self, account_id: str, margin_used: float) -> AccountBalance:
        """
        Update account margin used.
        
        Args:
            account_id: Account ID
            margin_used: New margin used value
            
        Returns:
            Updated account balance
        """
        # Get current account
        account = await self.get_by_id(account_id)
        if not account:
            logger.error(f"Cannot update margin: Account {account_id} not found")
            raise ValueError(f"Account {account_id} not found")
        
        # Update margin in database
        update_query = """
        UPDATE account_balances
        SET margin_used = %(margin_used)s, last_updated = %(last_updated)s
        WHERE id = %(account_id)s
        """
        
        update_params = {
            "account_id": account_id,
            "margin_used": margin_used,
            "last_updated": datetime.now(timezone.utc)
        }
        
        await self.session.execute(update_query, update_params)
        
        # Commit the transaction
        await self.session.commit()
        
        # Update account object
        account.margin_used = margin_used
        account.last_updated = update_params["last_updated"]
        
        logger.info(f"Updated margin for account {account_id}: {margin_used}")
        return account
    
    async def get_balance_history(self, account_id: str, start_date: datetime, end_date: datetime) -> List[BalanceChange]:
        """
        Get history of balance changes.
        
        Args:
            account_id: Account ID
            start_date: Start date
            end_date: End date
            
        Returns:
            List of balance changes
        """
        query = """
        SELECT id, account_id, amount, balance_before, balance_after, reason, timestamp
        FROM balance_changes
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
        
        changes = []
        for row in result:
            change = BalanceChange(
                id=row[0],
                account_id=row[1],
                amount=row[2],
                balance_before=row[3],
                balance_after=row[4],
                reason=row[5],
                timestamp=row[6]
            )
            changes.append(change)
        
        logger.debug(f"Retrieved {len(changes)} balance changes for account {account_id}")
        return changes
    
    async def get_account_details(self, account_id: str) -> Optional[AccountDetails]:
        """
        Get detailed account information including balance history.
        
        Args:
            account_id: Account ID
            
        Returns:
            Detailed account information or None if account not found
        """
        # Get basic account balance
        account = await self.get_by_id(account_id)
        if not account:
            return None
        
        # Get recent balance changes (last 30 days)
        now = datetime.now(timezone.utc)
        start_date = now.replace(day=1)  # First day of current month
        
        balance_changes = await self.get_balance_history(account_id, start_date, now)
        
        # Get position information
        query = """
        SELECT 
            COUNT(*) as total_positions,
            SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_positions,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_positions,
            SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losing_positions,
            SUM(realized_pnl) as total_pnl
        FROM positions
        WHERE account_id = %(account_id)s
        """
        
        result = await self.session.execute(query, {"account_id": account_id})
        position_stats = result.fetchone()
        
        # Create account details
        if position_stats:
            total_positions = position_stats[0] or 0
            win_rate = position_stats[3] / total_positions if total_positions > 0 else 0
            
            details = AccountDetails(
                id=account.id,
                user_id=account.user_id,
                balance=account.balance,
                margin_used=account.margin_used,
                free_margin=account.balance - account.margin_used,
                equity=account.balance,  # Will be updated with unrealized PnL
                last_updated=account.last_updated,
                total_positions=total_positions,
                open_positions=position_stats[1] or 0,
                winning_positions=position_stats[2] or 0,
                losing_positions=position_stats[3] or 0,
                win_rate=win_rate,
                total_pnl=position_stats[4] or 0,
                recent_changes=balance_changes
            )
            
            # Get unrealized PnL for open positions
            query = """
            SELECT COALESCE(SUM(unrealized_pnl), 0) as total_unrealized
            FROM positions
            WHERE account_id = %(account_id)s
            AND status = 'OPEN'
            """
            
            result = await self.session.execute(query, {"account_id": account_id})
            unrealized_pnl = result.fetchone()[0] or 0
            
            # Update equity with unrealized PnL
            details.equity = details.balance + unrealized_pnl
            details.unrealized_pnl = unrealized_pnl
            
            return details
        
        # Return basic details if no positions found
        return AccountDetails(
            id=account.id,
            user_id=account.user_id,
            balance=account.balance,
            margin_used=account.margin_used,
            free_margin=account.balance - account.margin_used,
            equity=account.balance,
            last_updated=account.last_updated,
            total_positions=0,
            open_positions=0,
            winning_positions=0,
            losing_positions=0,
            win_rate=0,
            total_pnl=0,
            recent_changes=balance_changes
        )
