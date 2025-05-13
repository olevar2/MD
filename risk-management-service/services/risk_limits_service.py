"""
Risk Limits Service Module.

Provides functionality for managing and enforcing risk limits across accounts.
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from core_foundations.utils.logger import get_logger
from core.connection import DatabaseSession
from repositories.risk_repository import RiskRepository
from repositories.limits_repository import LimitsRepository
from core.risk_limits import (
    RiskLimit, RiskLimitCreate, RiskLimitUpdate, LimitType, RiskProfile
)

logger = get_logger("risk-limits-service")


class RiskViolationSeverity(str, Enum):
    """Enum for risk limit violation severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskLimitsService:
    """
    Service for managing and enforcing risk limits.
    
    This service provides functionality to:
    - Define and manage risk limits for accounts
    - Check if operations would violate risk limits
    - Create and manage risk profiles
    """
    
    def create_risk_limit(self, risk_limit: RiskLimitCreate) -> RiskLimit:
        """
        Create a new risk limit.
        
        Args:
            risk_limit: Risk limit to create
            
        Returns:
            Created risk limit
        """
        with DatabaseSession() as session:
            limits_repo = LimitsRepository(session)
            created_limit = limits_repo.create_limit(risk_limit)
            logger.info(f"Created risk limit: {created_limit.limit_type} for account {created_limit.account_id}")
            return created_limit
    
    def update_risk_limit(self, limit_id: str, update_data: RiskLimitUpdate) -> Optional[RiskLimit]:
        """
        Update an existing risk limit.
        
        Args:
            limit_id: ID of the risk limit to update
            update_data: New data for the risk limit
            
        Returns:
            Updated risk limit or None if not found
        """
        with DatabaseSession() as session:
            limits_repo = LimitsRepository(session)
            updated_limit = limits_repo.update_limit(limit_id, update_data)
            if updated_limit:
                logger.info(f"Updated risk limit {limit_id}")
            else:
                logger.warning(f"Risk limit {limit_id} not found for update")
            return updated_limit
    
    def get_risk_limit(self, limit_id: str) -> Optional[RiskLimit]:
        """
        Get a risk limit by ID.
        
        Args:
            limit_id: ID of the risk limit
            
        Returns:
            Risk limit if found, None otherwise
        """
        with DatabaseSession() as session:
            limits_repo = LimitsRepository(session)
            return limits_repo.get_limit_by_id(limit_id)
    
    def get_account_limits(self, account_id: str) -> Dict[str, RiskLimit]:
        """
        Get all risk limits for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Dictionary mapping limit types to RiskLimit objects
        """
        with DatabaseSession() as session:
            limits_repo = LimitsRepository(session)
            limits = limits_repo.get_limits_by_account(account_id)
            
            # Create a dictionary mapping limit types to limit objects
            limits_dict = {limit.limit_type: limit for limit in limits}
            return limits_dict
    
    def check_position_risk(
        self, 
        account_id: str, 
        position_size: float, 
        symbol: str, 
        entry_price: float
    ) -> Dict[str, Any]:
        """
        Check if a new position would violate risk limits.
        
        Args:
            account_id: Account ID
            position_size: Size of the position
            symbol: Trading symbol
            entry_price: Entry price
            
        Returns:
            Dictionary with risk check results
        """
        with DatabaseSession() as session:
            # Get account risk limits
            limits_repo = LimitsRepository(session)
            risk_repo = RiskRepository(session)
            
            account_limits = limits_repo.get_limits_by_account(account_id)
            account_info = risk_repo.get_account_info(account_id)
            
            if not account_info:
                logger.warning(f"Account {account_id} not found in risk repository")
                return {
                    "allowed": False,
                    "reason": "Account not found"
                }
            
            # Extract relevant limits
            max_position_size_limit = None
            max_single_exposure_pct_limit = None
            max_total_exposure_pct_limit = None
            max_positions_count_limit = None
            
            for limit in account_limits:
                if limit.limit_type == LimitType.MAX_POSITION_SIZE:
                    max_position_size_limit = limit
                elif limit.limit_type == LimitType.MAX_SINGLE_EXPOSURE_PCT:
                    max_single_exposure_pct_limit = limit
                elif limit.limit_type == LimitType.MAX_TOTAL_EXPOSURE_PCT:
                    max_total_exposure_pct_limit = limit
                elif limit.limit_type == LimitType.MAX_POSITIONS_COUNT:
                    max_positions_count_limit = limit
            
            # Check position size limit
            if max_position_size_limit and position_size > max_position_size_limit.limit_value:
                return {
                    "allowed": False,
                    "reason": f"Position size {position_size} exceeds limit {max_position_size_limit.limit_value}"
                }
            
            # Check single exposure percentage limit
            if max_single_exposure_pct_limit:
                exposure_value = position_size * entry_price
                exposure_pct = (exposure_value / account_info.balance) * 100
                
                if exposure_pct > max_single_exposure_pct_limit.limit_value:
                    return {
                        "allowed": False,
                        "reason": f"Position exposure {exposure_pct}% exceeds limit {max_single_exposure_pct_limit.limit_value}%"
                    }
            
            # Check total exposure percentage limit
            if max_total_exposure_pct_limit:
                current_exposure = risk_repo.get_total_exposure(account_id)
                new_exposure = current_exposure + (position_size * entry_price)
                new_exposure_pct = (new_exposure / account_info.balance) * 100
                
                if new_exposure_pct > max_total_exposure_pct_limit.limit_value:
                    return {
                        "allowed": False,
                        "reason": f"Total exposure {new_exposure_pct}% would exceed limit {max_total_exposure_pct_limit.limit_value}%"
                    }
            
            # Check max positions count limit
            if max_positions_count_limit:
                current_positions_count = risk_repo.get_open_positions_count(account_id)
                
                if current_positions_count >= max_positions_count_limit.limit_value:
                    return {
                        "allowed": False,
                        "reason": f"Max positions count {max_positions_count_limit.limit_value} would be exceeded"
                    }
            
            # All checks passed
            return {
                "allowed": True,
                "reason": "All risk checks passed"
            }
    
    def check_portfolio_risk(self, account_id: str) -> Dict[str, Any]:
        """
        Check overall portfolio risk against limits.
        
        Args:
            account_id: Account ID
            
        Returns:
            Dictionary with portfolio risk check results
        """
        with DatabaseSession() as session:
            limits_repo = LimitsRepository(session)
            risk_repo = RiskRepository(session)
            
            account_limits = limits_repo.get_limits_by_account(account_id)
            account_info = risk_repo.get_account_info(account_id)
            
            if not account_info:
                logger.warning(f"Account {account_id} not found in risk repository")
                return {
                    "status": "error",
                    "reason": "Account not found"
                }
            
            # Initialize results
            results = {
                "account_id": account_id,
                "timestamp": datetime.utcnow(),
                "violations": [],
                "warnings": [],
                "overall_risk_level": RiskViolationSeverity.LOW
            }
            
            # Get current risk metrics
            metrics = risk_repo.get_risk_metrics(account_id)
            
            # Check all applicable limits
            for limit in account_limits:
                current_value = None
                
                if limit.limit_type == LimitType.MAX_DRAWDOWN_PCT and metrics.get("current_drawdown_pct") is not None:
                    current_value = metrics["current_drawdown_pct"]
                elif limit.limit_type == LimitType.MAX_TOTAL_EXPOSURE_PCT and metrics.get("total_exposure_pct") is not None:
                    current_value = metrics["total_exposure_pct"]
                elif limit.limit_type == LimitType.MAX_DAILY_LOSS_PCT and metrics.get("daily_loss_pct") is not None:
                    current_value = metrics["daily_loss_pct"]
                elif limit.limit_type == LimitType.MAX_VAR_PCT and metrics.get("var_pct") is not None:
                    current_value = metrics["var_pct"]
                
                if current_value is not None and limit.limit_value is not None:
                    # Calculate proximity to limit as percentage
                    limit_proximity = (current_value / limit.limit_value) * 100
                    
                    # Determine severity based on proximity to limit
                    severity = RiskViolationSeverity.LOW
                    if current_value > limit.limit_value:
                        severity = RiskViolationSeverity.HIGH
                    elif limit_proximity > 90:
                        severity = RiskViolationSeverity.MEDIUM
                    elif limit_proximity > 75:
                        severity = RiskViolationSeverity.LOW
                    
                    # Record violation or warning
                    if current_value > limit.limit_value:
                        results["violations"].append({
                            "limit_type": limit.limit_type,
                            "limit_value": limit.limit_value,
                            "current_value": current_value,
                            "severity": severity
                        })
                    elif limit_proximity > 75:
                        results["warnings"].append({
                            "limit_type": limit.limit_type,
                            "limit_value": limit.limit_value,
                            "current_value": current_value,
                            "proximity_pct": limit_proximity,
                            "severity": severity
                        })
            
            # Determine overall risk level
            if any(v["severity"] == RiskViolationSeverity.HIGH for v in results["violations"]):
                results["overall_risk_level"] = RiskViolationSeverity.HIGH
            elif any(v["severity"] == RiskViolationSeverity.MEDIUM for v in results["violations"]):
                results["overall_risk_level"] = RiskViolationSeverity.MEDIUM
            elif results["violations"]:
                results["overall_risk_level"] = RiskViolationSeverity.LOW
            
            # Record check in repository
            risk_repo.record_risk_check(
                account_id=account_id,
                check_type="portfolio",
                result=results["overall_risk_level"],
                violations_count=len(results["violations"])
            )
            
            return results
    
    def create_risk_profile(self, profile_data: Dict[str, Any]) -> RiskProfile:
        """
        Create a risk profile with predefined limits.
        
        Args:
            profile_data: Risk profile data including name, description, and limits
            
        Returns:
            Created risk profile
        """
        with DatabaseSession() as session:
            limits_repo = LimitsRepository(session)
            profile = limits_repo.create_risk_profile(profile_data)
            logger.info(f"Created risk profile: {profile.name}")
            return profile
    
    def apply_risk_profile(self, account_id: str, profile_id: str) -> List[RiskLimit]:
        """
        Apply a risk profile to an account.
        
        Args:
            account_id: Account ID
            profile_id: Risk profile ID
            
        Returns:
            List of created risk limits
        """
        with DatabaseSession() as session:
            limits_repo = LimitsRepository(session)
            
            # Get the profile
            profile = limits_repo.get_risk_profile_by_id(profile_id)
            if not profile:
                logger.error(f"Risk profile {profile_id} not found")
                return []
            
            # Get existing limits for the account
            existing_limits = limits_repo.get_limits_by_account(account_id)
            
            # Create or update limits based on profile
            created_limits = []
            for profile_limit in profile.limits:
                # Check if this limit type already exists
                existing_limit = next(
                    (l for l in existing_limits if l.limit_type == profile_limit.limit_type),
                    None
                )
                
                if existing_limit:
                    # Update the existing limit
                    updated = limits_repo.update_limit(
                        existing_limit.id,
                        RiskLimitUpdate(limit_value=profile_limit.limit_value)
                    )
                    if updated:
                        created_limits.append(updated)
                else:
                    # Create a new limit
                    new_limit = RiskLimitCreate(
                        account_id=account_id,
                        limit_type=profile_limit.limit_type,
                        limit_value=profile_limit.limit_value,
                        description=f"Applied from profile: {profile.name}"
                    )
                    created = limits_repo.create_limit(new_limit)
                    created_limits.append(created)
            
            logger.info(f"Applied risk profile {profile.name} to account {account_id}")
            return created_limits
