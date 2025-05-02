"""
Risk Limit Breach Repository Module.

Handles database operations for risk limit breaches.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import desc, and_
from sqlalchemy.orm import Session

from core_foundations.utils.logger import get_logger
from risk_management_service.db.models import RiskLimitBreachDb
from risk_management_service.models.risk_limit import (
    RiskLimitBreach, LimitBreachSeverity, LimitBreachStatus, LimitBreachAction
)

logger = get_logger("risk-breach-repository")


class RiskBreachRepository:
    """Repository for managing risk limit breaches in the database."""

    def __init__(self, session: Session):
        """
        Initialize the repository with a database session.
        
        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def create(self, breach_data: RiskLimitBreach) -> RiskLimitBreach:
        """
        Record a new risk limit breach in the database.
        
        Args:
            breach_data: Data for the risk limit breach
            
        Returns:
            Created risk breach record
        """
        # Create database model
        db_breach = RiskLimitBreachDb(
            breach_id=breach_data.breach_id,
            limit_id=breach_data.limit_id,
            account_id=breach_data.account_id,
            strategy_id=breach_data.strategy_id,
            breach_time=breach_data.breach_time,
            severity=breach_data.severity.value,
            status=breach_data.status.value,
            current_value=breach_data.current_value,
            limit_value=breach_data.limit_value,
            action_taken=breach_data.action_taken.value,
            description=breach_data.description,
            resolved_time=breach_data.resolved_time,
            override_reason=breach_data.override_reason,
            override_by=breach_data.override_by,
            metadata=breach_data.metadata
        )
        
        try:
            # Add to database
            self.session.add(db_breach)
            self.session.flush()
            logger.info(f"Recorded risk breach {breach_data.breach_id} for limit {breach_data.limit_id}")
            return breach_data
        except Exception as e:
            logger.error(f"Failed to record risk breach: {str(e)}")
            raise

    def get_by_id(self, breach_id: str) -> Optional[RiskLimitBreach]:
        """
        Get a risk breach by its ID.
        
        Args:
            breach_id: Risk breach ID
            
        Returns:
            Risk breach if found, None otherwise
        """
        db_breach = self.session.query(RiskLimitBreachDb).filter(
            RiskLimitBreachDb.breach_id == breach_id
        ).first()
        
        if not db_breach:
            return None
            
        return self._map_to_model(db_breach)

    def update_status(
        self, 
        breach_id: str, 
        status: LimitBreachStatus, 
        resolved_time: Optional[datetime] = None,
        override_reason: Optional[str] = None,
        override_by: Optional[str] = None
    ) -> Optional[RiskLimitBreach]:
        """
        Update the status of a risk breach.
        
        Args:
            breach_id: Risk breach ID
            status: New status
            resolved_time: Time of resolution, if resolved
            override_reason: Reason for override, if overridden
            override_by: User who overrode the breach, if overridden
            
        Returns:
            Updated risk breach if found, None otherwise
        """
        db_breach = self.session.query(RiskLimitBreachDb).filter(
            RiskLimitBreachDb.breach_id == breach_id
        ).first()
        
        if not db_breach:
            logger.warning(f"Risk breach not found for update: {breach_id}")
            return None
            
        # Update fields
        db_breach.status = status.value
        
        if status == LimitBreachStatus.RESOLVED and resolved_time is not None:
            db_breach.resolved_time = resolved_time
        elif status == LimitBreachStatus.RESOLVED and resolved_time is None:
            db_breach.resolved_time = datetime.utcnow()
            
        if status == LimitBreachStatus.OVERRIDDEN:
            db_breach.override_reason = override_reason
            db_breach.override_by = override_by
            if not db_breach.resolved_time:
                db_breach.resolved_time = datetime.utcnow()
        
        try:
            self.session.flush()
            logger.info(f"Updated risk breach {breach_id} status to {status.value}")
            return self._map_to_model(db_breach)
        except Exception as e:
            logger.error(f"Failed to update risk breach {breach_id}: {str(e)}")
            raise

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
        List risk breaches with optional filtering.
        
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
        query = self.session.query(RiskLimitBreachDb)
        
        # Apply filters
        if limit_id:
            query = query.filter(RiskLimitBreachDb.limit_id == limit_id)
            
        if account_id:
            query = query.filter(RiskLimitBreachDb.account_id == account_id)
            
        if strategy_id:
            query = query.filter(RiskLimitBreachDb.strategy_id == strategy_id)
            
        if severity:
            query = query.filter(RiskLimitBreachDb.severity == severity)
            
        if status:
            query = query.filter(RiskLimitBreachDb.status == status)
            
        if from_time:
            query = query.filter(RiskLimitBreachDb.breach_time >= from_time)
            
        if to_time:
            query = query.filter(RiskLimitBreachDb.breach_time <= to_time)
            
        # Order by breach_time descending (newest first)
        query = query.order_by(desc(RiskLimitBreachDb.breach_time))
        
        # Apply pagination
        query = query.limit(limit).offset(offset)
        
        # Execute query
        db_breaches = query.all()
        
        # Map to domain models
        return [self._map_to_model(db_breach) for db_breach in db_breaches]

    def count_breaches(
        self,
        limit_id: Optional[str] = None,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> int:
        """
        Count risk breaches with optional filtering.
        
        Args:
            limit_id: Optional limit ID filter
            account_id: Optional account ID filter
            strategy_id: Optional strategy ID filter
            severity: Optional severity filter
            status: Optional status filter
            from_time: Optional start time filter
            to_time: Optional end time filter
            
        Returns:
            Number of risk breaches
        """
        query = self.session.query(RiskLimitBreachDb)
        
        # Apply filters
        if limit_id:
            query = query.filter(RiskLimitBreachDb.limit_id == limit_id)
            
        if account_id:
            query = query.filter(RiskLimitBreachDb.account_id == account_id)
            
        if strategy_id:
            query = query.filter(RiskLimitBreachDb.strategy_id == strategy_id)
            
        if severity:
            query = query.filter(RiskLimitBreachDb.severity == severity)
            
        if status:
            query = query.filter(RiskLimitBreachDb.status == status)
            
        if from_time:
            query = query.filter(RiskLimitBreachDb.breach_time >= from_time)
            
        if to_time:
            query = query.filter(RiskLimitBreachDb.breach_time <= to_time)
            
        # Execute count query
        return query.count()

    def get_active_breaches(
        self,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> List[RiskLimitBreach]:
        """
        Get all active limit breaches for an account or strategy.
        
        Args:
            account_id: Optional account ID
            strategy_id: Optional strategy ID
            
        Returns:
            List of active risk breaches
        """
        query = self.session.query(RiskLimitBreachDb).filter(
            RiskLimitBreachDb.status == LimitBreachStatus.ACTIVE.value
        )
        
        if account_id:
            query = query.filter(RiskLimitBreachDb.account_id == account_id)
            
        if strategy_id:
            query = query.filter(RiskLimitBreachDb.strategy_id == strategy_id)
            
        # Order by breach_time (newest first)
        query = query.order_by(desc(RiskLimitBreachDb.breach_time))
        
        # Execute query
        db_breaches = query.all()
        
        # Map to domain models
        return [self._map_to_model(db_breach) for db_breach in db_breaches]

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
        base_query = self.session.query(RiskLimitBreachDb)
        
        # Apply filters to base query
        if account_id:
            base_query = base_query.filter(RiskLimitBreachDb.account_id == account_id)
            
        if strategy_id:
            base_query = base_query.filter(RiskLimitBreachDb.strategy_id == strategy_id)
            
        if from_time:
            base_query = base_query.filter(RiskLimitBreachDb.breach_time >= from_time)
            
        if to_time:
            base_query = base_query.filter(RiskLimitBreachDb.breach_time <= to_time)
            
        # Get counts by status
        active_count = base_query.filter(RiskLimitBreachDb.status == LimitBreachStatus.ACTIVE.value).count()
        resolved_count = base_query.filter(RiskLimitBreachDb.status == LimitBreachStatus.RESOLVED.value).count()
        overridden_count = base_query.filter(RiskLimitBreachDb.status == LimitBreachStatus.OVERRIDDEN.value).count()
        
        # Get counts by severity
        warning_count = base_query.filter(RiskLimitBreachDb.severity == LimitBreachSeverity.WARNING.value).count()
        soft_count = base_query.filter(RiskLimitBreachDb.severity == LimitBreachSeverity.SOFT.value).count()
        hard_count = base_query.filter(RiskLimitBreachDb.severity == LimitBreachSeverity.HARD.value).count()
        
        # Get counts by action type
        notify_count = base_query.filter(RiskLimitBreachDb.action_taken == LimitBreachAction.NOTIFY_ONLY.value).count()
        prevent_count = base_query.filter(RiskLimitBreachDb.action_taken == LimitBreachAction.PREVENT_NEW_POSITIONS.value).count()
        close_count = base_query.filter(RiskLimitBreachDb.action_taken == LimitBreachAction.CLOSE_POSITIONS.value).count()
        reduce_count = base_query.filter(RiskLimitBreachDb.action_taken == LimitBreachAction.REDUCE_POSITION_SIZE.value).count()
        suspend_count = base_query.filter(RiskLimitBreachDb.action_taken == LimitBreachAction.SUSPEND_ACCOUNT.value).count()
        
        # Compile statistics
        return {
            "total_breaches": active_count + resolved_count + overridden_count,
            "status": {
                "active": active_count,
                "resolved": resolved_count,
                "overridden": overridden_count
            },
            "severity": {
                "warning": warning_count,
                "soft": soft_count,
                "hard": hard_count
            },
            "actions": {
                "notify_only": notify_count,
                "prevent_new_positions": prevent_count,
                "close_positions": close_count,
                "reduce_position_size": reduce_count,
                "suspend_account": suspend_count
            },
            "account_id": account_id,
            "strategy_id": strategy_id,
            "from_time": from_time,
            "to_time": to_time
        }

    def _map_to_model(self, db_breach: RiskLimitBreachDb) -> RiskLimitBreach:
        """
        Map database model to domain model.
        
        Args:
            db_breach: Database model
            
        Returns:
            Domain model
        """
        return RiskLimitBreach(
            breach_id=db_breach.breach_id,
            limit_id=db_breach.limit_id,
            account_id=db_breach.account_id,
            strategy_id=db_breach.strategy_id,
            breach_time=db_breach.breach_time,
            severity=LimitBreachSeverity(db_breach.severity),
            status=LimitBreachStatus(db_breach.status),
            current_value=db_breach.current_value,
            limit_value=db_breach.limit_value,
            action_taken=LimitBreachAction(db_breach.action_taken),
            description=db_breach.description,
            resolved_time=db_breach.resolved_time,
            override_reason=db_breach.override_reason,
            override_by=db_breach.override_by,
            metadata=db_breach.metadata or {}
        )