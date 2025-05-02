"""
Risk Management Service for Forex Trading.

Implements comprehensive risk management including position sizing,
exposure limits, drawdown protection, and risk metrics calculation.
"""
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from decimal import Decimal
from enum import Enum
from common_lib.exceptions import ServiceError

logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(str, Enum):
    """Types of risk to monitor."""
    MARKET = "market_risk"
    LEVERAGE = "leverage_risk"
    EXPOSURE = "exposure_risk"
    CONCENTRATION = "concentration_risk"
    DRAWDOWN = "drawdown_risk"
    VOLATILITY = "volatility_risk"

class RiskLimit:
    """Represents a risk limit with threshold and action."""
    
    def __init__(
        self,
        risk_type: RiskType,
        threshold: float,
        risk_level: RiskLevel,
        action: Optional[str] = None
    ):
        self.risk_type = risk_type
        self.threshold = threshold
        self.risk_level = risk_level
        self.action = action or "Alert"
        self.is_breached = False
        self.breach_time = None
        
    def check_breach(self, value: float) -> bool:
        """Check if value breaches the risk limit."""
        is_breach = value > self.threshold
        
        if is_breach and not self.is_breached:
            self.is_breached = True
            self.breach_time = datetime.utcnow()
        elif not is_breach and self.is_breached:
            self.is_breached = False
            self.breach_time = None
            
        return is_breach

class Position:
    """Represents a trading position."""
    
    def __init__(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        direction: str,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.symbol = symbol
        self.size = size
        self.entry_price = entry_price
        self.direction = direction
        self.leverage = leverage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.utcnow()
        
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate position P&L."""
        price_diff = current_price - self.entry_price
        if self.direction == "short":
            price_diff = -price_diff
        return price_diff * self.size * self.leverage

    def calculate_risk(self, current_price: float) -> float:
        """Calculate position risk based on stop loss."""
        if not self.stop_loss:
            return float('inf')
            
        risk = abs(current_price - self.stop_loss)
        return risk * self.size * self.leverage

class RiskManager:
    """
    Core risk management system for forex trading.
    
    Handles position sizing, risk limits, exposure tracking,
    and implements circuit breakers for risk events.
    """
    
    def __init__(
        self,
        initial_balance: float,
        max_position_size: float,
        max_leverage: float = 20.0,
        max_drawdown: float = 0.20,  # 20% max drawdown
        risk_per_trade: float = 0.02  # 2% risk per trade
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        
        self.positions: Dict[str, Position] = {}
        self.risk_limits: Dict[RiskType, List[RiskLimit]] = {
            risk_type: [] for risk_type in RiskType
        }
        
        self.peak_balance = initial_balance
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Initialize default risk limits
        self._initialize_risk_limits()
        
    def _initialize_risk_limits(self) -> None:
        """Initialize default risk limits."""
        # Leverage risk limits
        self.add_risk_limit(
            RiskType.LEVERAGE,
            threshold=self.max_leverage * 0.8,
            risk_level=RiskLevel.HIGH,
            action="Reduce leverage"
        )
        self.add_risk_limit(
            RiskType.LEVERAGE,
            threshold=self.max_leverage,
            risk_level=RiskLevel.CRITICAL,
            action="Close positions"
        )
        
        # Drawdown risk limits
        self.add_risk_limit(
            RiskType.DRAWDOWN,
            threshold=self.max_drawdown * 0.5,
            risk_level=RiskLevel.MEDIUM,
            action="Review strategy"
        )
        self.add_risk_limit(
            RiskType.DRAWDOWN,
            threshold=self.max_drawdown * 0.8,
            risk_level=RiskLevel.HIGH,
            action="Reduce exposure"
        )
        self.add_risk_limit(
            RiskType.DRAWDOWN,
            threshold=self.max_drawdown,
            risk_level=RiskLevel.CRITICAL,
            action="Stop trading"
        )
        
        # Exposure risk limits
        total_exposure_limit = self.initial_balance * 2  # 200% max exposure
        self.add_risk_limit(
            RiskType.EXPOSURE,
            threshold=total_exposure_limit * 0.8,
            risk_level=RiskLevel.HIGH,
            action="Reduce exposure"
        )
        self.add_risk_limit(
            RiskType.EXPOSURE,
            threshold=total_exposure_limit,
            risk_level=RiskLevel.CRITICAL,
            action="Close positions"
        )
        
    def add_risk_limit(
        self,
        risk_type: RiskType,
        threshold: float,
        risk_level: RiskLevel,
        action: Optional[str] = None
    ) -> None:
        """Add a new risk limit."""
        limit = RiskLimit(risk_type, threshold, risk_level, action)
        self.risk_limits[risk_type].append(limit)
        
        # Sort limits by threshold
        self.risk_limits[risk_type].sort(key=lambda x: x.threshold)
        
    def check_position_entry(
        self,
        symbol: str,
        size: float,
        price: float,
        direction: str,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check if a new position can be entered.
        
        Args:
            symbol: Trading symbol
            size: Position size
            price: Entry price
            direction: Trade direction ('long' or 'short')
            leverage: Position leverage
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Dict containing validation result and any warnings/errors
        """
        result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check leverage
        if leverage > self.max_leverage:
            result['valid'] = False
            result['errors'].append(
                f"Leverage {leverage} exceeds maximum {self.max_leverage}"
            )
            
        # Check position size
        if size > self.max_position_size:
            result['valid'] = False
            result['errors'].append(
                f"Position size {size} exceeds maximum {self.max_position_size}"
            )
            
        # Calculate position value
        position_value = size * price * leverage
        
        # Check if within risk per trade
        if stop_loss:
            risk = abs(price - stop_loss) * size * leverage
            max_risk = self.current_balance * self.risk_per_trade
            if risk > max_risk:
                result['valid'] = False
                result['errors'].append(
                    f"Position risk {risk:.2f} exceeds max risk per trade {max_risk:.2f}"
                )
                
        # Check total exposure
        current_exposure = sum(
            p.size * p.entry_price * p.leverage
            for p in self.positions.values()
        )
        total_exposure = current_exposure + position_value
        
        for limit in self.risk_limits[RiskType.EXPOSURE]:
            if total_exposure > limit.threshold:
                if limit.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    result['valid'] = False
                    result['errors'].append(
                        f"Total exposure {total_exposure:.2f} exceeds {limit.risk_level} threshold"
                    )
                else:
                    result['warnings'].append(
                        f"Total exposure {total_exposure:.2f} exceeds {limit.risk_level} threshold"
                    )
                    
        # Check concentration in single symbol
        symbol_exposure = position_value
        if symbol in self.positions:
            symbol_exposure += (
                self.positions[symbol].size *
                self.positions[symbol].entry_price *
                self.positions[symbol].leverage
            )
            
        concentration = symbol_exposure / self.current_balance
        if concentration > 0.25:  # max 25% in single symbol
            result['warnings'].append(
                f"High concentration in {symbol}: {concentration*100:.1f}%"
            )
            
        return result
        
    def add_position(
        self,
        symbol: str,
        size: float,
        price: float,
        direction: str,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """Add a new position."""
        position = Position(
            symbol=symbol,
            size=size,
            entry_price=price,
            direction=direction,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        self.positions[symbol] = position
        
        self._update_metrics()
        
    def update_position(
        self,
        symbol: str,
        current_price: float,
        size: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """Update an existing position."""
        if symbol not in self.positions:
            raise ServiceError(f"No position found for {symbol}", service_name="risk-management-service")
            
        position = self.positions[symbol]
        
        if size is not None:
            position.size = size
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
            
        # Update balance with P&L
        pnl = position.calculate_pnl(current_price)
        self.current_balance += pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)
        
        self._update_metrics()
        
    def close_position(self, symbol: str, current_price: float) -> float:
        """Close a position and return realized P&L."""
        if symbol not in self.positions:
            raise ServiceError(f"No position found for {symbol}", service_name="risk-management-service")
            
        position = self.positions[symbol]
        pnl = position.calculate_pnl(current_price)
        
        self.current_balance += pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)
        
        del self.positions[symbol]
        self._update_metrics()
        
        return pnl
        
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get current portfolio metrics."""
        total_exposure = sum(
            p.size * p.entry_price * p.leverage
            for p in self.positions.values()
        )
        
        unrealized_pnl = 0.0  # Would need current prices to calculate
        
        drawdown = (
            (self.peak_balance - self.current_balance) /
            self.peak_balance
            if self.peak_balance > 0 else 0
        )
        
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'drawdown': drawdown,
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / self.current_balance if self.current_balance > 0 else 0,
            'position_count': len(self.positions),
            'unrealized_pnl': unrealized_pnl
        }
        
    def check_risk_limits(self) -> List[Dict[str, Any]]:
        """Check all risk limits and return breached limits."""
        metrics = self.get_portfolio_metrics()
        breached_limits = []
        
        # Check drawdown limits
        for limit in self.risk_limits[RiskType.DRAWDOWN]:
            if limit.check_breach(metrics['drawdown']):
                breached_limits.append({
                    'risk_type': RiskType.DRAWDOWN,
                    'threshold': limit.threshold,
                    'current_value': metrics['drawdown'],
                    'risk_level': limit.risk_level,
                    'action': limit.action,
                    'breach_time': limit.breach_time
                })
                
        # Check exposure limits
        exposure_ratio = metrics['exposure_ratio']
        for limit in self.risk_limits[RiskType.EXPOSURE]:
            if limit.check_breach(exposure_ratio):
                breached_limits.append({
                    'risk_type': RiskType.EXPOSURE,
                    'threshold': limit.threshold,
                    'current_value': exposure_ratio,
                    'risk_level': limit.risk_level,
                    'action': limit.action,
                    'breach_time': limit.breach_time
                })
                
        # Check leverage limits
        max_leverage = max(
            (p.leverage for p in self.positions.values()),
            default=0
        )
        for limit in self.risk_limits[RiskType.LEVERAGE]:
            if limit.check_breach(max_leverage):
                breached_limits.append({
                    'risk_type': RiskType.LEVERAGE,
                    'threshold': limit.threshold,
                    'current_value': max_leverage,
                    'risk_level': limit.risk_level,
                    'action': limit.action,
                    'breach_time': limit.breach_time
                })
                
        return breached_limits
        
    def _update_metrics(self) -> None:
        """Update metrics history."""
        metrics = self.get_portfolio_metrics()
        metrics['timestamp'] = datetime.utcnow()
        self.metrics_history.append(metrics)
        
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical metrics as a DataFrame."""
        df = pd.DataFrame(self.metrics_history)
        
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
            
        return df

    def calculate_var(
        self,
        confidence_level: float = 0.95,
        lookback_days: int = 30
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            confidence_level: VaR confidence level
            lookback_days: Historical lookback period in days
            
        Returns:
            VaR value
        """
        if not self.metrics_history:
            return 0.0
            
        df = self.get_metrics_history(
            start_time=datetime.utcnow() - timedelta(days=lookback_days)
        )
        
        if len(df) < 2:
            return 0.0
            
        # Calculate daily returns
        df['returns'] = df['current_balance'].pct_change()
        
        # Calculate VaR
        var = np.percentile(df['returns'].dropna(), (1 - confidence_level) * 100)
        
        return -var * self.current_balance  # Convert to absolute value
