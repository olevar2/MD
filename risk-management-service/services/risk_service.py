"""
Risk Service Module.

Provides business logic for risk management, limit checking, and breach handling.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import uuid4

from core_foundations.utils.logger import get_logger
from core.connection import DatabaseSession
from models.risk_limit_repository import RiskLimitRepository
from repositories.risk_breach_repository import RiskBreachRepository
from core.risk_limit import (
    RiskLimit, RiskLimitCreate, RiskLimitUpdate, RiskLimitBreach,
    LimitType, LimitScope, LimitBreachSeverity, LimitBreachStatus, LimitBreachAction
)

logger = get_logger("risk-service")


class RiskService:
    """Service for managing risk limits and checks."""

    def __init__(self):
        """Initialize the risk service with required clients."""
        from adapters.portfolio_management_client import PortfolioManagementClient
        self.portfolio_client = PortfolioManagementClient()

    def create_limit(self, limit_data: RiskLimitCreate) -> RiskLimit:
        """
        Create a new risk limit.

        Args:
            limit_data: Data for the new limit

        Returns:
            Created limit
        """
        with DatabaseSession() as session:
            repo = RiskLimitRepository(session)
            limit = repo.create(limit_data)
            logger.info(f"Created risk limit: {limit.name} ({limit.limit_id})")
            return limit

    def get_limit(self, limit_id: str) -> Optional[RiskLimit]:
        """
        Get a risk limit by ID.

        Args:
            limit_id: Limit ID

        Returns:
            Limit if found, None otherwise
        """
        with DatabaseSession() as session:
            repo = RiskLimitRepository(session)
            return repo.get_by_id(limit_id)

    def update_limit(self, limit_id: str, limit_update: RiskLimitUpdate) -> Optional[RiskLimit]:
        """
        Update a risk limit.

        Args:
            limit_id: Limit ID
            limit_update: Data to update

        Returns:
            Updated limit if found, None otherwise
        """
        with DatabaseSession() as session:
            repo = RiskLimitRepository(session)
            updated_limit = repo.update(limit_id, limit_update)
            if updated_limit:
                logger.info(f"Updated risk limit: {updated_limit.name} ({limit_id})")
            return updated_limit

    def list_limits(
        self,
        limit_type: Optional[str] = None,
        scope: Optional[str] = None,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[RiskLimit]:
        """
        List risk limits with optional filtering.

        Args:
            limit_type: Optional limit type filter
            scope: Optional scope filter
            account_id: Optional account ID filter
            strategy_id: Optional strategy ID filter
            is_active: Optional active status filter
            limit: Maximum number of limits to return
            offset: Number of limits to skip

        Returns:
            List of risk limits
        """
        with DatabaseSession() as session:
            repo = RiskLimitRepository(session)
            return repo.list_limits(
                limit_type=limit_type,
                scope=scope,
                account_id=account_id,
                strategy_id=strategy_id,
                is_active=is_active,
                limit=limit,
                offset=offset
            )

    def delete_limit(self, limit_id: str) -> bool:
        """
        Delete a risk limit.

        Args:
            limit_id: Limit ID

        Returns:
            True if deleted, False if not found
        """
        with DatabaseSession() as session:
            repo = RiskLimitRepository(session)
            result = repo.delete(limit_id)
            if result:
                logger.info(f"Deleted risk limit: {limit_id}")
            return result

    def check_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an order complies with risk limits.

        Args:
            order_data: Order data including account_id, symbol, side, quantity, price

        Returns:
            Dictionary with check results
        """
        account_id = order_data.get("account_id")
        symbol = order_data.get("symbol")
        side = order_data.get("side")
        quantity = float(order_data.get("quantity", 0))
        price = float(order_data.get("price", 0))
        strategy_id = order_data.get("strategy_id")

        # Get applicable limits
        with DatabaseSession() as session:
            limit_repo = RiskLimitRepository(session)
            applicable_limits = limit_repo.get_applicable_limits(
                account_id=account_id,
                strategy_id=strategy_id
            )

            # Check each limit
            results = []
            all_passed = True

            for limit in applicable_limits:
                result = self._check_limit(limit, order_data)
                results.append(result)

                if not result["passed"]:
                    all_passed = False

                    # Record breach if limit was violated
                    if result["record_breach"]:
                        self._record_breach(
                            session=session,
                            limit=limit,
                            account_id=account_id,
                            strategy_id=strategy_id,
                            current_value=result["current_value"],
                            description=result["message"]
                        )

            return {
                "passed": all_passed,
                "order_id": order_data.get("order_id"),
                "account_id": account_id,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "results": results,
                "timestamp": datetime.utcnow()
            }

    def _check_limit(self, limit: RiskLimit, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an order complies with a specific risk limit.

        Args:
            limit: Risk limit to check
            order_data: Order data

        Returns:
            Dictionary with check result
        """
        limit_type = limit.limit_type
        account_id = order_data.get("account_id")
        symbol = order_data.get("symbol")
        side = order_data.get("side")
        quantity = float(order_data.get("quantity", 0))
        price = float(order_data.get("price", 0))

        passed = True
        current_value = 0
        message = ""
        record_breach = False

        # Different checks based on limit type
        if limit_type == LimitType.MAX_POSITION_SIZE:
            # Example: Check if position size exceeds limit
            position_value = quantity * price
            current_value = position_value

            if position_value > limit.value:
                passed = False
                message = f"Position size ({position_value} {limit.unit}) exceeds limit of {limit.value} {limit.unit}"
                record_breach = True

        elif limit_type == LimitType.MAX_TOTAL_EXPOSURE:
            # Would need to fetch current positions to calculate total exposure
            # This is a simplified example
            position_value = quantity * price
            current_exposure = self._get_current_exposure_sync(account_id)
            new_exposure = current_exposure + position_value
            current_value = new_exposure

            if new_exposure > limit.value:
                passed = False
                message = f"Total exposure ({new_exposure} {limit.unit}) would exceed limit of {limit.value} {limit.unit}"
                record_breach = True

        elif limit_type == LimitType.MAX_SYMBOL_EXPOSURE:
            # Check exposure per symbol
            if symbol:
                current_symbol_exposure = self._get_symbol_exposure_sync(account_id, symbol)
                position_value = quantity * price
                new_symbol_exposure = current_symbol_exposure + position_value
                current_value = new_symbol_exposure

                if new_symbol_exposure > limit.value:
                    passed = False
                    message = f"Symbol exposure for {symbol} ({new_symbol_exposure} {limit.unit}) would exceed limit of {limit.value} {limit.unit}"
                    record_breach = True

        # Additional limit types would be checked similarly

        return {
            "limit_id": limit.limit_id,
            "limit_name": limit.name,
            "limit_type": limit.limit_type.value,
            "limit_value": limit.value,
            "current_value": current_value,
            "passed": passed,
            "message": message,
            "record_breach": record_breach
        }

    def _record_breach(
        self,
        session,
        limit: RiskLimit,
        account_id: Optional[str],
        strategy_id: Optional[str],
        current_value: float,
        description: str
    ) -> RiskLimitBreach:
        """
        Record a risk limit breach.

        Args:
            session: Database session
            limit: Breached limit
            account_id: Account ID
            strategy_id: Strategy ID
            current_value: Current value that breached the limit
            description: Description of the breach

        Returns:
            Created breach record
        """
        # Determine severity based on how much the limit was exceeded
        severity_ratio = current_value / limit.value if limit.value != 0 else float('inf')

        if severity_ratio > 1.5:
            severity = LimitBreachSeverity.HARD
            action = LimitBreachAction.PREVENT_NEW_POSITIONS
        elif severity_ratio > 1.2:
            severity = LimitBreachSeverity.SOFT
            action = LimitBreachAction.NOTIFY_ONLY
        else:
            severity = LimitBreachSeverity.WARNING
            action = LimitBreachAction.NOTIFY_ONLY

        # Create breach record
        breach = RiskLimitBreach(
            limit_id=limit.limit_id,
            account_id=account_id,
            strategy_id=strategy_id,
            breach_time=datetime.utcnow(),
            severity=severity,
            status=LimitBreachStatus.ACTIVE,
            current_value=current_value,
            limit_value=limit.value,
            action_taken=action,
            description=description
        )

        # Record in database
        breach_repo = RiskBreachRepository(session)
        breach_repo.create(breach)

        logger.warning(
            f"Risk limit breach: {limit.name} ({limit.limit_id}), "
            f"Value: {current_value}/{limit.value}, "
            f"Severity: {severity.value}"
        )

        return breach

    def list_breaches(
        self,
        limit_id: Optional[str] = None,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[RiskLimitBreach]:
        """
        List risk limit breaches with optional filtering.

        Args:
            limit_id: Optional limit ID filter
            account_id: Optional account ID filter
            strategy_id: Optional strategy ID filter
            severity: Optional severity filter
            status: Optional status filter
            from_time: Optional start time filter
            to_time: Optional end time filter
            limit: Maximum number of breaches to return
            offset: Number of breaches to skip

        Returns:
            List of risk breaches
        """
        with DatabaseSession() as session:
            repo = RiskBreachRepository(session)
            return repo.list_breaches(
                limit_id=limit_id,
                account_id=account_id,
                strategy_id=strategy_id,
                severity=severity,
                status=status,
                from_time=from_time,
                to_time=to_time,
                limit=limit,
                offset=offset
            )

    def get_breach(self, breach_id: str) -> Optional[RiskLimitBreach]:
        """
        Get a risk breach by ID.

        Args:
            breach_id: Breach ID

        Returns:
            Breach if found, None otherwise
        """
        with DatabaseSession() as session:
            repo = RiskBreachRepository(session)
            return repo.get_by_id(breach_id)

    def resolve_breach(
        self,
        breach_id: str,
        override: bool = False,
        override_reason: Optional[str] = None,
        override_by: Optional[str] = None
    ) -> Optional[RiskLimitBreach]:
        """
        Resolve or override a risk breach.

        Args:
            breach_id: Breach ID
            override: Whether this is an override (True) or resolution (False)
            override_reason: Reason for override, if overriding
            override_by: User who overrode the breach, if overriding

        Returns:
            Updated breach if found, None otherwise
        """
        with DatabaseSession() as session:
            repo = RiskBreachRepository(session)

            status = LimitBreachStatus.OVERRIDDEN if override else LimitBreachStatus.RESOLVED

            updated_breach = repo.update_status(
                breach_id=breach_id,
                status=status,
                resolved_time=datetime.utcnow(),
                override_reason=override_reason,
                override_by=override_by
            )

            if updated_breach:
                action_type = "Overridden" if override else "Resolved"
                logger.info(f"{action_type} risk breach: {breach_id}")

            return updated_breach

    def get_active_breaches(
        self,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> List[RiskLimitBreach]:
        """
        Get all active risk breaches.

        Args:
            account_id: Optional account ID filter
            strategy_id: Optional strategy ID filter

        Returns:
            List of active breaches
        """
        with DatabaseSession() as session:
            repo = RiskBreachRepository(session)
            return repo.get_active_breaches(
                account_id=account_id,
                strategy_id=strategy_id
            )

    def get_breach_statistics(
        self,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about risk breaches.

        Args:
            account_id: Optional account ID filter
            strategy_id: Optional strategy ID filter
            from_time: Optional start time filter
            to_time: Optional end time filter

        Returns:
            Dictionary with breach statistics
        """
        with DatabaseSession() as session:
            repo = RiskBreachRepository(session)
            return repo.get_breach_statistics(
                account_id=account_id,
                strategy_id=strategy_id,
                from_time=from_time,
                to_time=to_time
            )    # Helper methods for getting account/position data from Portfolio Management Service
    async def _get_current_exposure(self, account_id: str) -> float:
        """
        Get current total exposure for an account.

        Args:
            account_id: Account ID

        Returns:
            Total exposure value
        """
        return await self.portfolio_client.get_total_exposure(account_id)

    async def _get_symbol_exposure(self, account_id: str, symbol: str) -> float:
        """
        Get current exposure for a specific symbol.

        Args:
            account_id: Account ID
            symbol: Symbol to check

        Returns:
            Symbol exposure value
        """
        return await self.portfolio_client.get_symbol_exposure(account_id, symbol)

    def _get_current_exposure_sync(self, account_id: str) -> float:
        """
        Synchronous version of _get_current_exposure for use in non-async methods.

        Args:
            account_id: Account ID

        Returns:
            Total exposure value
        """
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._get_current_exposure(account_id))
        finally:
            loop.close()

    def _get_symbol_exposure_sync(self, account_id: str, symbol: str) -> float:
        """
        Synchronous version of _get_symbol_exposure for use in non-async methods.

        Args:
            account_id: Account ID
            symbol: Symbol to check

        Returns:
            Symbol exposure value
        """
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._get_symbol_exposure(account_id, symbol))
        finally:
            loop.close()

    async def close(self):
        """Close service resources."""
        await self.portfolio_client.close()