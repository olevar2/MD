"""
Portfolio Service Module.

Provides business logic for managing portfolios, positions, and account balances.
"""
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

from core_foundations.utils.logger import get_logger
from core.connection import get_db_session
from repositories.position_repository import PositionRepository
from repositories.account_repository import AccountRepository
from repositories.historical_repository import HistoricalRepository
from core.position import Position, PositionCreate, PositionUpdate, PositionStatus, PositionPerformance
from core.account import AccountBalance, BalanceChange, AccountDetails
from core.historical import PortfolioSnapshot, DrawdownAnalysis
from core.performance_metrics import PerformanceMetrics
from core.historical_tracking import HistoricalTracking
from portfolio_management_service.error import (
    async_with_exception_handling,
    PortfolioManagementError,
    PortfolioNotFoundError,
    PositionNotFoundError,
    InsufficientBalanceError,
    PortfolioOperationError,
    AccountReconciliationError
)

logger = get_logger("portfolio-service")


class PortfolioService:
    """Service for managing portfolio operations."""
    
    def __init__(self):
        """Initialize portfolio service with required components."""
        self.performance_metrics = PerformanceMetrics()
        self.historical_tracking = HistoricalTracking()
    
    @async_with_exception_handling
    async def create_position(self, position_data: PositionCreate) -> Position:
        """
        Create a new trading position.
        
        Args:
            position_data: Data for the new position
            
        Returns:
            Created position
            
        Raises:
            InsufficientBalanceError: If account has insufficient balance for the position
            PortfolioOperationError: If position creation fails
        """
        async with get_db_session() as session:
            # Check account balance
            account_repo = AccountRepository(session)
            account = await account_repo.get_by_id(position_data.account_id)
            
            if not account:
                raise PortfolioNotFoundError(
                    message=f"Account {position_data.account_id} not found",
                    portfolio_id=position_data.account_id
                )
            
            # Calculate required margin
            required_margin = position_data.size * 0.05  # 5% margin requirement
            
            # Check if account has sufficient free margin
            if account.free_margin < required_margin:
                raise InsufficientBalanceError(
                    message="Insufficient free margin for position",
                    required_amount=required_margin,
                    available_amount=account.free_margin,
                    currency=account.currency
                )
            
            # Create position
            position_repo = PositionRepository(session)
            try:
                position = await position_repo.create(position_data)
            except Exception as e:
                raise PortfolioOperationError(
                    message=f"Failed to create position: {str(e)}",
                    operation="create_position",
                    details={"symbol": position_data.symbol, "error": str(e)}
                )
            
            # Update margin usage
            await self._update_account_margin(
                account_repo, 
                position_data.account_id, 
                calculate_margin=True
            )
            
            logger.info(f"Created position for {position_data.symbol} in account {position_data.account_id}")
            return position
            
    @async_with_exception_handling
    async def get_position(self, position_id: str) -> Position:
        """
        Get a position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position if found
            
        Raises:
            PositionNotFoundError: If position is not found
        """
        async with get_db_session() as session:
            position_repo = PositionRepository(session)
            position = await position_repo.get_by_id(position_id)
            
            if not position:
                raise PositionNotFoundError(
                    message=f"Position {position_id} not found",
                    position_id=position_id
                )
                
            return position
            
    @async_with_exception_handling
    async def update_position(self, position_id: str, position_update: PositionUpdate) -> Position:
        """
        Update a position.
        
        Args:
            position_id: Position ID
            position_update: Data to update
            
        Returns:
            Updated position
            
        Raises:
            PositionNotFoundError: If position is not found
            PortfolioOperationError: If position update fails
        """
        async with get_db_session() as session:
            position_repo = PositionRepository(session)
            position = await position_repo.update(position_id, position_update)
            
            if not position:
                raise PositionNotFoundError(
                    message=f"Position {position_id} not found",
                    position_id=position_id
                )
            
            if position_update.current_price is not None:
                # If price was updated, recalculate equity
                account_repo = AccountRepository(session)
                await self._update_account_equity(
                    account_repo, 
                    position.account_id
                )
                
            return position
            
    @async_with_exception_handling
    async def close_position(self, position_id: str, exit_price: float) -> Position:
        """
        Close an open position.
        
        Args:
            position_id: ID of position to close
            exit_price: Exit price for the position
            
        Returns:
            Closed position
            
        Raises:
            PositionNotFoundError: If position is not found or already closed
            PortfolioOperationError: If position closing fails
        """
        async with get_db_session() as session:
            # Close position
            position_repo = PositionRepository(session)
            position = await position_repo.close_position(position_id, exit_price)
            
            if not position:
                raise PositionNotFoundError(
                    message=f"Position {position_id} not found or already closed",
                    position_id=position_id
                )
            
            # Update account balance with realized PnL
            account_repo = AccountRepository(session)
            try:
                await account_repo.update_balance(
                    position.account_id,
                    position.realized_pnl,
                    f"Position closed: {position.symbol} {position.direction}"
                )
            except Exception as e:
                raise PortfolioOperationError(
                    message=f"Failed to update account balance: {str(e)}",
                    operation="close_position",
                    details={"position_id": position_id, "error": str(e)}
                )
            
            # Update account margin
            await self._update_account_margin(
                account_repo,
                position.account_id,
                calculate_margin=True
            )
            
            # Create a daily snapshot to track the portfolio change
            try:
                await self.historical_tracking.create_daily_snapshot(position.account_id)
            except Exception as e:
                logger.warning(f"Failed to create daily snapshot: {str(e)}")
                # Don't fail the operation if snapshot creation fails
            
            logger.info(f"Closed position {position_id} with realized PnL: {position.realized_pnl}")
            return position
    
    @async_with_exception_handling
    async def get_portfolio_summary(self, account_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the portfolio for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Dictionary with portfolio summary information
            
        Raises:
            PortfolioNotFoundError: If account is not found
        """
        async with get_db_session() as session:
            # Get account details
            account_repo = AccountRepository(session)
            account_details = await account_repo.get_account_details(account_id)
            
            if not account_details:
                raise PortfolioNotFoundError(
                    message=f"Account {account_id} not found",
                    portfolio_id=account_id
                )
            
            # Get positions
            position_repo = PositionRepository(session)
            open_positions = await position_repo.get_open_positions(account_id)
            
            # Get closed positions for the last 30 days
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            closed_positions = await position_repo.get_closed_positions(
                account_id, start_date, end_date
            )
            
            # Calculate performance metrics
            all_positions = open_positions + closed_positions
            performance_metrics = self.performance_metrics.calculate_overall_metrics(all_positions)
            
            # Calculate drawdown
            drawdown_metrics = self.performance_metrics.calculate_drawdown(
                all_positions, account_details.balance
            )
            
            # Create summary
            summary = {
                "account": {
                    "id": account_details.id,
                    "balance": account_details.balance,
                    "equity": account_details.equity,
                    "margin_used": account_details.margin_used,
                    "free_margin": account_details.free_margin,
                    "margin_level": (account_details.equity / account_details.margin_used * 100) 
                              if account_details.margin_used > 0 else None
                },
                "positions": {
                    "open_count": len(open_positions),
                    "closed_count": len(closed_positions),
                    "open_positions": [self._position_to_dict(p) for p in open_positions]
                },
                "performance": performance_metrics,
                "drawdown": drawdown_metrics,
                "updated_at": datetime.now(timezone.utc)
            }
            
            return summary
    
    @async_with_exception_handling
    async def get_historical_performance(self, account_id: str, period_days: int = 90) -> Dict[str, Any]:
        """
        Get historical performance data for an account.
        
        Args:
            account_id: Account ID
            period_days: Number of days to look back
            
        Returns:
            Dictionary with historical performance data
            
        Raises:
            PortfolioNotFoundError: If account is not found
            PortfolioOperationError: If historical data retrieval fails
        """
        # Check if account exists
        async with get_db_session() as session:
            account_repo = AccountRepository(session)
            account = await account_repo.get_by_id(account_id)
            
            if not account:
                raise PortfolioNotFoundError(
                    message=f"Account {account_id} not found",
                    portfolio_id=account_id
                )
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=period_days)
        
        try:
            # Get equity curve
            equity_df = await self.historical_tracking.get_historical_equity(
                account_id, start_date, end_date
            )
            
            # Get performance metrics history
            metrics_df = await self.historical_tracking.get_performance_metrics_history(
                account_id, start_date, end_date
            )
        except Exception as e:
            raise PortfolioOperationError(
                message=f"Failed to retrieve historical data: {str(e)}",
                operation="get_historical_performance",
                details={"account_id": account_id, "error": str(e)}
            )
        
        # Calculate periodic returns
        async with get_db_session() as session:
            position_repo = PositionRepository(session)
            closed_positions = await position_repo.get_closed_positions(
                account_id, start_date, end_date
            )
            
            monthly_metrics = self.performance_metrics.calculate_metrics_by_period(
                closed_positions, period_type='monthly'
            )
            
            weekly_metrics = self.performance_metrics.calculate_metrics_by_period(
                closed_positions, period_type='weekly'
            )
        
        # Format results
        if not equity_df.empty:
            equity_data = equity_df.reset_index().to_dict(orient='records')
        else:
            equity_data = []
            
        if not metrics_df.empty:
            metrics_data = metrics_df.reset_index().to_dict(orient='records')
        else:
            metrics_data = []
            
        result = {
            "equity_history": equity_data,
            "performance_metrics": metrics_data,
            "monthly_performance": monthly_metrics,
            "weekly_performance": weekly_metrics,
            # "drawdown_analysis": drawdown_analysis,
            "period_days": period_days,
            "start_date": start_date,
            "end_date": end_date
        }
        
        return result
    
    @async_with_exception_handling
    async def _update_account_margin(self, account_repo: AccountRepository, 
                                  account_id: str, calculate_margin: bool = False) -> float:
        """
        Update account margin used.
        
        Args:
            account_repo: Account repository
            account_id: Account ID
            calculate_margin: Whether to recalculate margin from open positions
            
        Returns:
            Updated margin used value
            
        Raises:
            PortfolioNotFoundError: If account is not found
            PortfolioOperationError: If margin update fails
        """
        if calculate_margin:
            # Calculate current margin use from open positions
            async with get_db_session() as session:
                position_repo = PositionRepository(session)
                open_positions = await position_repo.get_open_positions(account_id)
                
                # Simple margin calculation (for example purposes)
                # In a real system, this would be more complex based on leverage, position size, etc.
                margin_used = sum(p.size * 0.05 for p in open_positions)  # 5% margin requirement
                
                try:
                    # Update account margin
                    account = await account_repo.update_margin(account_id, margin_used)
                    if not account:
                        raise PortfolioNotFoundError(
                            message=f"Account {account_id} not found",
                            portfolio_id=account_id
                        )
                    return margin_used
                except Exception as e:
                    raise PortfolioOperationError(
                        message=f"Failed to update account margin: {str(e)}",
                        operation="update_account_margin",
                        details={"account_id": account_id, "error": str(e)}
                    )
        else:
            # Get current margin from account
            account = await account_repo.get_by_id(account_id)
            if not account:
                raise PortfolioNotFoundError(
                    message=f"Account {account_id} not found",
                    portfolio_id=account_id
                )
            return account.margin_used
            
    @async_with_exception_handling
    async def _update_account_equity(self, account_repo: AccountRepository, account_id: str) -> float:
        """
        Update account equity based on unrealized PnL of open positions.
        
        Args:
            account_repo: Account repository
            account_id: Account ID
            
        Returns:
            Updated equity value
            
        Raises:
            PortfolioNotFoundError: If account is not found
            PortfolioOperationError: If equity update fails
        """
        async with get_db_session() as session:
            position_repo = PositionRepository(session)
            open_positions = await position_repo.get_open_positions(account_id)
            
            # Calculate unrealized PnL
            unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
            
            # Get account balance
            account = await account_repo.get_by_id(account_id)
            if not account:
                raise PortfolioNotFoundError(
                    message=f"Account {account_id} not found for equity update",
                    portfolio_id=account_id
                )
                
            # Calculate equity
            equity = account.balance + unrealized_pnl
            
            # In a more complex system, we might update some 'equity' field in the account
            # For now, we just return the calculated value
            return equity
            
    def _position_to_dict(self, position: Position) -> Dict[str, Any]:
        """
        Convert position to dictionary for easier serialization.
        
        Args:
            position: Position object
            
        Returns:
            Dictionary representation of position
        """
        return {
            "id": position.id,
            "symbol": position.symbol,
            "direction": position.direction,
            "size": position.size,
            "entry_price": position.entry_price,
            "current_price": position.exit_price,  # Use exit_price if set, otherwise needs market data
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "entry_time": position.entry_time,
            "unrealized_pnl": position.unrealized_pnl,
            "status": position.status
        }