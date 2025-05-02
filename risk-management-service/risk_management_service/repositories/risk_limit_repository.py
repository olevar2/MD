"""
Risk Limit Repository Module.

Provides data access operations for risk limits.
"""
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError

from core_foundations.utils.logger import get_logger
from risk_management_service.db.connection import get_db_session
from risk_management_service.db.models import RiskLimitDb
from risk_management_service.models.risk_limit import RiskLimit, LimitScope, LimitType

logger = get_logger("risk-limit-repository")


class RiskLimitRepository:
    """Repository for managing risk limits in the database."""
    
    def create_risk_limit(self, risk_limit: RiskLimit) -> RiskLimit:
        """
        Create a new risk limit.
        
        Args:
            risk_limit: Risk limit to create
            
        Returns:
            Created risk limit with ID
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        # Generate UUID if not provided
        if not risk_limit.limit_id:
            risk_limit.limit_id = str(uuid.uuid4())
        
        db_model = self._to_db_model(risk_limit)
        
        try:
            with get_db_session() as session:
                session.add(db_model)
                session.flush()  # Flush to get the ID if auto-generated
                
                # Convert back to domain model
                created_limit = self._to_domain_model(db_model)
                logger.info(f"Created risk limit: {created_limit.limit_id} - {created_limit.name}")
                return created_limit
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create risk limit: {str(e)}")
            raise
    
    def get_risk_limit(self, limit_id: str) -> Optional[RiskLimit]:
        """
        Get a risk limit by ID.
        
        Args:
            limit_id: Risk limit ID
            
        Returns:
            Risk limit if found, None otherwise
        """
        try:
            with get_db_session() as session:
                risk_limit_db = session.query(RiskLimitDb).filter(RiskLimitDb.limit_id == limit_id).first()
                
                if risk_limit_db:
                    return self._to_domain_model(risk_limit_db)
                return None
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get risk limit {limit_id}: {str(e)}")
            raise
    
    def update_risk_limit(self, risk_limit: RiskLimit) -> Optional[RiskLimit]:
        """
        Update an existing risk limit.
        
        Args:
            risk_limit: Risk limit to update
            
        Returns:
            Updated risk limit if successful, None if not found
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            with get_db_session() as session:
                risk_limit_db = session.query(RiskLimitDb).filter(RiskLimitDb.limit_id == risk_limit.limit_id).first()
                
                if not risk_limit_db:
                    logger.warning(f"Risk limit not found for update: {risk_limit.limit_id}")
                    return None
                
                # Update fields
                risk_limit_db.name = risk_limit.name
                risk_limit_db.description = risk_limit.description
                risk_limit_db.limit_type = risk_limit.limit_type.value
                risk_limit_db.scope = risk_limit.scope.value
                risk_limit_db.value = risk_limit.value
                risk_limit_db.unit = risk_limit.unit
                risk_limit_db.is_active = risk_limit.is_active
                risk_limit_db.account_id = risk_limit.account_id
                risk_limit_db.strategy_id = risk_limit.strategy_id
                risk_limit_db.metadata = risk_limit.metadata
                risk_limit_db.updated_at = datetime.utcnow()
                
                session.flush()
                updated_limit = self._to_domain_model(risk_limit_db)
                logger.info(f"Updated risk limit: {updated_limit.limit_id}")
                return updated_limit
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to update risk limit {risk_limit.limit_id}: {str(e)}")
            raise
    
    def delete_risk_limit(self, limit_id: str) -> bool:
        """
        Delete a risk limit by ID.
        
        Args:
            limit_id: Risk limit ID
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            with get_db_session() as session:
                risk_limit = session.query(RiskLimitDb).filter(RiskLimitDb.limit_id == limit_id).first()
                
                if not risk_limit:
                    logger.warning(f"Risk limit not found for deletion: {limit_id}")
                    return False
                
                session.delete(risk_limit)
                logger.info(f"Deleted risk limit: {limit_id}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete risk limit {limit_id}: {str(e)}")
            raise
    
    def get_all_risk_limits(self) -> List[RiskLimit]:
        """
        Get all risk limits.
        
        Returns:
            List of risk limits
        """
        try:
            with get_db_session() as session:
                risk_limits = session.query(RiskLimitDb).all()
                return [self._to_domain_model(limit) for limit in risk_limits]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get all risk limits: {str(e)}")
            raise
    
    def get_risk_limits_by_scope(self, scope: LimitScope, 
                                account_id: Optional[str] = None, 
                                strategy_id: Optional[str] = None) -> List[RiskLimit]:
        """
        Get risk limits by scope.
        
        Args:
            scope: Limit scope
            account_id: Optional account ID for ACCOUNT scope
            strategy_id: Optional strategy ID for STRATEGY scope
            
        Returns:
            List of risk limits
        """
        try:
            with get_db_session() as session:
                query = session.query(RiskLimitDb).filter(RiskLimitDb.scope == scope.value)
                
                if scope == LimitScope.ACCOUNT and account_id:
                    query = query.filter(RiskLimitDb.account_id == account_id)
                elif scope == LimitScope.STRATEGY and strategy_id:
                    query = query.filter(RiskLimitDb.strategy_id == strategy_id)
                
                risk_limits = query.all()
                return [self._to_domain_model(limit) for limit in risk_limits]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get risk limits by scope {scope.value}: {str(e)}")
            raise
    
    def get_risk_limits_by_type(self, limit_type: LimitType) -> List[RiskLimit]:
        """
        Get risk limits by type.
        
        Args:
            limit_type: Limit type
            
        Returns:
            List of risk limits
        """
        try:
            with get_db_session() as session:
                risk_limits = session.query(RiskLimitDb).filter(
                    RiskLimitDb.limit_type == limit_type.value
                ).all()
                
                return [self._to_domain_model(limit) for limit in risk_limits]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get risk limits by type {limit_type.value}: {str(e)}")
            raise
    
    def get_risk_limits_for_account(self, account_id: str) -> List[RiskLimit]:
        """
        Get all risk limits applicable for an account.
        This includes:
          - Global limits
          - Account-specific limits for this account
          - Strategy limits for strategies used by this account
        
        Args:
            account_id: Account ID
            
        Returns:
            List of applicable risk limits
        """
        try:
            with get_db_session() as session:
                # Get global limits and account-specific limits
                risk_limits = session.query(RiskLimitDb).filter(
                    ((RiskLimitDb.scope == LimitScope.GLOBAL.value) |
                     ((RiskLimitDb.scope == LimitScope.ACCOUNT.value) & 
                      (RiskLimitDb.account_id == account_id)))
                ).all()
                
                return [self._to_domain_model(limit) for limit in risk_limits]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get risk limits for account {account_id}: {str(e)}")
            raise
    
    def _to_db_model(self, risk_limit: RiskLimit) -> RiskLimitDb:
        """
        Convert domain model to database model.
        
        Args:
            risk_limit: Domain model
            
        Returns:
            Database model
        """
        return RiskLimitDb(
            limit_id=risk_limit.limit_id,
            name=risk_limit.name,
            description=risk_limit.description,
            limit_type=risk_limit.limit_type.value,
            scope=risk_limit.scope.value,
            account_id=risk_limit.account_id,
            strategy_id=risk_limit.strategy_id,
            value=risk_limit.value,
            unit=risk_limit.unit,
            is_active=risk_limit.is_active,
            created_at=risk_limit.created_at or datetime.utcnow(),
            updated_at=risk_limit.updated_at or datetime.utcnow(),
            metadata=risk_limit.metadata
        )
    
    def _to_domain_model(self, db_model: RiskLimitDb) -> RiskLimit:
        """
        Convert database model to domain model.
        
        Args:
            db_model: Database model
            
        Returns:
            Domain model
        """
        return RiskLimit(
            limit_id=db_model.limit_id,
            name=db_model.name,
            description=db_model.description,
            limit_type=LimitType(db_model.limit_type),
            scope=LimitScope(db_model.scope),
            account_id=db_model.account_id,
            strategy_id=db_model.strategy_id,
            value=db_model.value,
            unit=db_model.unit,
            is_active=db_model.is_active,
            created_at=db_model.created_at,
            updated_at=db_model.updated_at,
            metadata=db_model.metadata or {}
        )