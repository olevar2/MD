"""
Risk Management Interfaces Module

This module provides interfaces for risk management components used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime


class RiskRegimeType(str, Enum):
    """Types of risk regimes for dynamic adaptation."""
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    EXTREME_RISK = "extreme_risk"
    CRISIS = "crisis"


class RiskParameterType(str, Enum):
    """Types of risk parameters that can be dynamically adjusted."""
    POSITION_SIZE = "position_size"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_EXPOSURE = "max_exposure"
    VOLATILITY_SCALING = "volatility_scaling"
    CORRELATION_LIMIT = "correlation_limit"
    NEWS_SENSITIVITY = "news_sensitivity"


class IRiskParameters(ABC):
    """Interface for risk parameters."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to dictionary.
        
        Returns:
            Dictionary representation of parameters
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRiskParameters':
        """
        Create parameters from dictionary.
        
        Args:
            data: Dictionary representation of parameters
            
        Returns:
            Risk parameters instance
        """
        pass
    
    @abstractmethod
    def adjust_for_regime(self, regime_type: RiskRegimeType) -> 'IRiskParameters':
        """
        Create a copy of risk parameters adjusted for the given risk regime.
        
        Args:
            regime_type: Risk regime type
            
        Returns:
            Adjusted risk parameters
        """
        pass


class IRiskRegimeDetector(ABC):
    """Interface for risk regime detectors."""
    
    @abstractmethod
    def detect_regime(self, market_data: Any) -> RiskRegimeType:
        """
        Detect current risk regime based on market data.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Detected risk regime type
        """
        pass
    
    @abstractmethod
    def get_regime_history(self) -> List[Tuple[datetime, RiskRegimeType]]:
        """
        Get history of detected regimes.
        
        Returns:
            List of (timestamp, regime) tuples
        """
        pass
    
    @abstractmethod
    def get_current_risk_parameters(self) -> IRiskParameters:
        """
        Get risk parameters adjusted for current regime.
        
        Returns:
            Adjusted risk parameters
        """
        pass


class IDynamicRiskTuner(ABC):
    """Interface for dynamic risk parameter tuners."""
    
    @abstractmethod
    def update_parameters(
        self,
        market_data: Optional[Any] = None,
        current_time: Optional[datetime] = None,
        rl_recommendations: Optional[Dict[str, float]] = None,
        force_update: bool = False
    ) -> IRiskParameters:
        """
        Update risk parameters based on regime, RL insights, and time.
        
        Args:
            market_data: Optional market data for regime detection
            current_time: Current time (defaults to now)
            rl_recommendations: Optional parameter recommendations from RL
            force_update: Whether to force update regardless of time
            
        Returns:
            Updated risk parameters
        """
        pass
    
    @abstractmethod
    def get_current_parameters(self) -> IRiskParameters:
        """
        Get current risk parameters.
        
        Returns:
            Current risk parameters
        """
        pass
    
    @abstractmethod
    def get_parameter_history(self) -> List[Tuple[datetime, Dict[str, Any]]]:
        """
        Get history of parameter changes.
        
        Returns:
            List of (timestamp, parameters) tuples
        """
        pass
