"""
Risk Management Package for Forex Trading Platform

This package provides comprehensive risk management functionality for the forex trading platform,
including risk parameters, risk regime detection, dynamic risk tuning, and risk management clients.
"""

# Import from interfaces module
from common_lib.risk.interfaces import (
    RiskRegimeType,
    RiskParameterType,
    IRiskParameters,
    IRiskRegimeDetector,
    IDynamicRiskTuner
)

# Import from models module
from common_lib.risk.models import (
    RiskLimit,
    RiskProfile,
    RiskMetrics,
    PositionRisk,
    RiskParameters,
    RiskLimitBreachAction,
    RiskLimitBreachSeverity,
    RiskLimitBreachStatus,
    RiskLimitBreachNotification
)

# Import from client module
from common_lib.risk.client import RiskManagementClient

# Import from adapters module
from common_lib.risk.adapters import (
    RiskManagementAdapter,
    StandardRiskParameters
)

# Re-export from common_lib.interfaces.risk_management
from common_lib.interfaces.risk_management import (
    RiskLimitType,
    RiskCheckResult,
    IRiskManager
)

__all__ = [
    # Interfaces
    'RiskRegimeType',
    'RiskParameterType',
    'IRiskParameters',
    'IRiskRegimeDetector',
    'IDynamicRiskTuner',
    'IRiskManager',
    
    # Models
    'RiskLimit',
    'RiskProfile',
    'RiskMetrics',
    'PositionRisk',
    'RiskParameters',
    'RiskLimitBreachAction',
    'RiskLimitBreachSeverity',
    'RiskLimitBreachStatus',
    'RiskLimitBreachNotification',
    'RiskLimitType',
    'RiskCheckResult',
    
    # Client
    'RiskManagementClient',
    
    # Adapters
    'RiskManagementAdapter',
    'StandardRiskParameters'
]
"""
