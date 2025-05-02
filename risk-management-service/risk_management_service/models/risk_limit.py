"""
Risk Limit Models Module.

Defines domain models for risk limits and risk limit breaches.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from uuid import uuid4


class LimitType(Enum):
    """Enumeration of risk limit types."""
    
    MAX_POSITION_SIZE = "max_position_size"
    MAX_TOTAL_EXPOSURE = "max_total_exposure"
    MAX_SYMBOL_EXPOSURE = "max_symbol_exposure"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_LEVERAGE = "max_leverage"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_OVERNIGHT_POSITIONS = "max_overnight_positions"
    MAX_TRADES_PER_DAY = "max_trades_per_day"
    MAX_TRADE_VOLUME = "max_trade_volume"
    MIN_ACCOUNT_BALANCE = "min_account_balance"


class LimitScope(Enum):
    """Enumeration of risk limit scopes."""
    
    GLOBAL = "global"          # Applies to all accounts and strategies
    ACCOUNT = "account"        # Applies to a specific account
    STRATEGY = "strategy"      # Applies to a specific strategy


class LimitBreachSeverity(Enum):
    """Enumeration of risk limit breach severities."""
    
    WARNING = "warning"        # Warning level, notification only
    SOFT = "soft"              # Soft limit breach, may require action
    HARD = "hard"              # Hard limit breach, requires immediate action


class LimitBreachStatus(Enum):
    """Enumeration of risk limit breach statuses."""
    
    ACTIVE = "active"          # Breach is currently active
    RESOLVED = "resolved"      # Breach has been resolved
    OVERRIDDEN = "overridden"  # Breach was overridden by an authorized user


class LimitBreachAction(Enum):
    """Enumeration of actions taken for risk limit breaches."""
    
    NOTIFY_ONLY = "notify_only"              # Just send notification
    PREVENT_NEW_POSITIONS = "prevent_new"    # Prevent opening new positions
    CLOSE_POSITIONS = "close_positions"      # Close existing positions
    REDUCE_POSITION_SIZE = "reduce_size"     # Reduce position size
    SUSPEND_ACCOUNT = "suspend_account"      # Suspend trading for the account


class RiskLimit:
    """Domain model for a risk limit."""
    
    def __init__(
        self,
        name: str,
        limit_type: LimitType,
        scope: LimitScope,
        value: float,
        unit: str,
        description: Optional[str] = None,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        is_active: bool = True,
        limit_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a risk limit.
        
        Args:
            name: Name of the limit
            limit_type: Type of limit
            scope: Scope of the limit (global, account, strategy)
            value: Value of the limit
            unit: Unit for the limit value (e.g., USD, percentage)
            description: Optional description
            account_id: Account ID if scope is ACCOUNT
            strategy_id: Strategy ID if scope is STRATEGY
            is_active: Whether the limit is active
            limit_id: Optional ID for the limit (generated if not provided)
            created_at: Creation timestamp (set to now if not provided)
            updated_at: Last update timestamp (set to now if not provided)
            metadata: Optional additional metadata
        """
        self.name = name
        self.description = description
        self.limit_type = limit_type
        self.scope = scope
        self.account_id = account_id
        self.strategy_id = strategy_id
        self.value = value
        self.unit = unit
        self.is_active = is_active
        self.limit_id = limit_id if limit_id is not None else str(uuid4())
        self.created_at = created_at if created_at is not None else datetime.utcnow()
        self.updated_at = updated_at if updated_at is not None else datetime.utcnow()
        self.metadata = metadata if metadata is not None else {}
        
        # Validate based on scope
        if self.scope == LimitScope.ACCOUNT and not self.account_id:
            raise ValueError("Account ID must be provided for account-scoped limits")
        
        if self.scope == LimitScope.STRATEGY and not self.strategy_id:
            raise ValueError("Strategy ID must be provided for strategy-scoped limits")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "limit_id": self.limit_id,
            "name": self.name,
            "description": self.description,
            "limit_type": self.limit_type.value,
            "scope": self.scope.value,
            "account_id": self.account_id,
            "strategy_id": self.strategy_id,
            "value": self.value,
            "unit": self.unit,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }


class RiskLimitCreate:
    """Data model for creating a new risk limit."""
    
    def __init__(
        self,
        name: str,
        limit_type: LimitType,
        scope: LimitScope,
        value: float,
        unit: str,
        description: Optional[str] = None,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        is_active: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize risk limit creation data.
        
        Args:
            name: Name of the limit
            limit_type: Type of limit
            scope: Scope of the limit (global, account, strategy)
            value: Value of the limit
            unit: Unit for the limit value (e.g., USD, percentage)
            description: Optional description
            account_id: Account ID if scope is ACCOUNT
            strategy_id: Strategy ID if scope is STRATEGY
            is_active: Whether the limit is active
            metadata: Optional additional metadata
        """
        self.name = name
        self.description = description
        self.limit_type = limit_type
        self.scope = scope
        self.account_id = account_id
        self.strategy_id = strategy_id
        self.value = value
        self.unit = unit
        self.is_active = is_active
        self.metadata = metadata if metadata is not None else {}
        
        # Validate based on scope
        if self.scope == LimitScope.ACCOUNT and not self.account_id:
            raise ValueError("Account ID must be provided for account-scoped limits")
        
        if self.scope == LimitScope.STRATEGY and not self.strategy_id:
            raise ValueError("Strategy ID must be provided for strategy-scoped limits")


class RiskLimitUpdate:
    """Data model for updating an existing risk limit."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        value: Optional[float] = None,
        is_active: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize risk limit update data.
        
        Args:
            name: New name (if provided)
            description: New description (if provided)
            value: New value (if provided)
            is_active: New active status (if provided)
            metadata: New metadata (if provided)
        """
        self.name = name
        self.description = description
        self.value = value
        self.is_active = is_active
        self.metadata = metadata


class RiskLimitBreach:
    """Domain model for a risk limit breach."""
    
    def __init__(
        self,
        limit_id: str,
        severity: LimitBreachSeverity,
        current_value: float,
        limit_value: float,
        action_taken: LimitBreachAction,
        description: str,
        breach_time: Optional[datetime] = None,
        status: LimitBreachStatus = LimitBreachStatus.ACTIVE,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        resolved_time: Optional[datetime] = None,
        override_reason: Optional[str] = None,
        override_by: Optional[str] = None,
        breach_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a risk limit breach.
        
        Args:
            limit_id: ID of the breached limit
            severity: Breach severity
            current_value: Current value that caused the breach
            limit_value: Limit value that was breached
            action_taken: Action taken in response to the breach
            description: Description of the breach
            breach_time: Time of the breach (set to now if not provided)
            status: Status of the breach
            account_id: Account ID related to the breach (if applicable)
            strategy_id: Strategy ID related to the breach (if applicable)
            resolved_time: Time when the breach was resolved (if applicable)
            override_reason: Reason for override (if applicable)
            override_by: User who overrode the breach (if applicable)
            breach_id: Optional ID for the breach (generated if not provided)
            metadata: Optional additional metadata
        """
        self.limit_id = limit_id
        self.severity = severity
        self.current_value = current_value
        self.limit_value = limit_value
        self.action_taken = action_taken
        self.description = description
        self.breach_time = breach_time if breach_time is not None else datetime.utcnow()
        self.status = status
        self.account_id = account_id
        self.strategy_id = strategy_id
        self.resolved_time = resolved_time
        self.override_reason = override_reason
        self.override_by = override_by
        self.breach_id = breach_id if breach_id is not None else str(uuid4())
        self.metadata = metadata if metadata is not None else {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "breach_id": self.breach_id,
            "limit_id": self.limit_id,
            "severity": self.severity.value,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "action_taken": self.action_taken.value,
            "description": self.description,
            "breach_time": self.breach_time,
            "status": self.status.value,
            "account_id": self.account_id,
            "strategy_id": self.strategy_id,
            "resolved_time": self.resolved_time,
            "override_reason": self.override_reason,
            "override_by": self.override_by,
            "metadata": self.metadata
        }