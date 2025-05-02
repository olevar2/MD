"""
Risk Management API Module.

Provides API endpoints for risk management operations.
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Path, Query, HTTPException, Depends, Body

from core_foundations.utils.logger import get_logger
from risk_management_service.services.risk_calculator import RiskCalculator
from risk_management_service.services.risk_limits_service import RiskLimitsService
from risk_management_service.models.risk_limits import (
    RiskLimit, RiskLimitCreate, RiskLimitUpdate, LimitType, RiskProfile
)
from risk_management_service.models.risk_metrics import AccountRiskInfo, RiskMetrics
from risk_management_service.api.auth import get_api_key

logger = get_logger("risk-management-api")

router = APIRouter(
    prefix="/api/v1/risk",
    tags=["Risk Management"]
)

# Initialize services
risk_calculator = RiskCalculator()
risk_limits_service = RiskLimitsService()


@router.post("/limits", response_model=RiskLimit)
async def create_risk_limit(
    limit: RiskLimitCreate,
    api_key: str = Depends(get_api_key)
):
    """Create a new risk limit for an account."""
    try:
        result = risk_limits_service.create_risk_limit(limit)
        return result
    except Exception as e:
        logger.error(f"Error creating risk limit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create risk limit: {str(e)}")


@router.get("/limits/{limit_id}", response_model=RiskLimit)
async def get_risk_limit(
    limit_id: str = Path(..., description="ID of the risk limit to retrieve"),
    api_key: str = Depends(get_api_key)
):
    """Get a risk limit by ID."""
    limit = risk_limits_service.get_risk_limit(limit_id)
    if not limit:
        raise HTTPException(status_code=404, detail=f"Risk limit {limit_id} not found")
    return limit


@router.put("/limits/{limit_id}", response_model=RiskLimit)
async def update_risk_limit(
    update_data: RiskLimitUpdate,
    limit_id: str = Path(..., description="ID of the risk limit to update"),
    api_key: str = Depends(get_api_key)
):
    """Update a risk limit."""
    limit = risk_limits_service.update_risk_limit(limit_id, update_data)
    if not limit:
        raise HTTPException(status_code=404, detail=f"Risk limit {limit_id} not found")
    return limit


@router.get("/accounts/{account_id}/limits", response_model=Dict[str, RiskLimit])
async def get_account_limits(
    account_id: str = Path(..., description="Account ID"),
    api_key: str = Depends(get_api_key)
):
    """Get all risk limits for an account."""
    return risk_limits_service.get_account_limits(account_id)


@router.post("/check/position", response_model=Dict[str, Any])
async def check_position_risk(
    account_id: str = Query(..., description="Account ID"),
    position_size: float = Query(..., description="Position size"),
    symbol: str = Query(..., description="Trading symbol"),
    entry_price: float = Query(..., description="Entry price"),
    api_key: str = Depends(get_api_key)
):
    """Check if a new position would violate risk limits."""
    result = risk_limits_service.check_position_risk(
        account_id=account_id,
        position_size=position_size,
        symbol=symbol,
        entry_price=entry_price
    )
    return result


@router.post("/check/portfolio/{account_id}", response_model=Dict[str, Any])
async def check_portfolio_risk(
    account_id: str = Path(..., description="Account ID"),
    api_key: str = Depends(get_api_key)
):
    """Check overall portfolio risk against limits."""
    result = risk_limits_service.check_portfolio_risk(account_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("reason"))
    return result


@router.post("/calculate/position-size", response_model=Dict[str, float])
async def calculate_position_size(
    account_balance: float = Query(..., description="Account balance"),
    risk_per_trade_pct: float = Query(..., description="Risk percentage per trade"),
    stop_loss_pips: float = Query(..., description="Stop loss in pips"),
    pip_value: float = Query(..., description="Pip value in account currency"),
    leverage: float = Query(1.0, description="Account leverage (default: 1.0)"),
    api_key: str = Depends(get_api_key)
):
    """Calculate position size based on account risk percentage."""
    result = risk_calculator.calculate_position_size(
        account_balance=account_balance,
        risk_per_trade_pct=risk_per_trade_pct,
        stop_loss_pips=stop_loss_pips,
        pip_value=pip_value,
        leverage=leverage
    )
    return result


@router.post("/calculate/var", response_model=Dict[str, float])
async def calculate_value_at_risk(
    portfolio_value: float = Query(..., description="Portfolio value"),
    daily_returns: List[float] = Body(..., description="Historical daily returns as percentages"),
    confidence_level: float = Query(0.95, description="Confidence level (default: 0.95)"),
    time_horizon_days: int = Query(1, description="Time horizon in days (default: 1)"),
    api_key: str = Depends(get_api_key)
):
    """Calculate Value at Risk (VaR) for a portfolio."""
    result = risk_calculator.calculate_value_at_risk(
        portfolio_value=portfolio_value,
        daily_returns=daily_returns,
        confidence_level=confidence_level,
        time_horizon_days=time_horizon_days
    )
    return result


@router.post("/calculate/drawdown", response_model=Dict[str, Any])
async def calculate_drawdown_risk(
    current_balance: float = Query(..., description="Current account balance"),
    historical_balances: List[float] = Body(..., description="List of historical account balances"),
    max_drawdown_limit_pct: float = Query(20.0, description="Maximum allowed drawdown percentage"),
    api_key: str = Depends(get_api_key)
):
    """Calculate drawdown risk metrics."""
    result = risk_calculator.calculate_drawdown_risk(
        current_balance=current_balance,
        historical_balances=historical_balances,
        max_drawdown_limit_pct=max_drawdown_limit_pct
    )
    return result


@router.post("/calculate/correlation", response_model=Dict[str, Any])
async def calculate_correlation_risk(
    symbols_returns: Dict[str, List[float]] = Body(..., description="Dictionary mapping symbols to their historical returns"),
    positions: Dict[str, float] = Body(..., description="Dictionary mapping symbols to their position sizes"),
    api_key: str = Depends(get_api_key)
):
    """Calculate correlation risk for a portfolio of positions."""
    result = risk_calculator.calculate_correlation_risk(
        symbols_returns=symbols_returns,
        positions=positions
    )
    return result


@router.post("/calculate/max-trades", response_model=Dict[str, int])
async def calculate_max_trades(
    account_balance: float = Query(..., description="Current account balance"),
    risk_per_trade_pct: float = Query(..., description="Risk percentage per trade"),
    portfolio_heat_limit_pct: float = Query(20.0, description="Maximum total portfolio risk percentage"),
    api_key: str = Depends(get_api_key)
):
    """Calculate maximum number of simultaneous trades based on risk limits."""
    result = risk_calculator.calculate_max_trades(
        account_balance=account_balance,
        risk_per_trade_pct=risk_per_trade_pct,
        portfolio_heat_limit_pct=portfolio_heat_limit_pct
    )
    return result


@router.post("/profiles", response_model=RiskProfile)
async def create_risk_profile(
    profile_data: Dict[str, Any] = Body(..., description="Risk profile data"),
    api_key: str = Depends(get_api_key)
):
    """Create a risk profile with predefined limits."""
    try:
        result = risk_limits_service.create_risk_profile(profile_data)
        return result
    except Exception as e:
        logger.error(f"Error creating risk profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create risk profile: {str(e)}")


@router.post("/accounts/{account_id}/apply-profile/{profile_id}", response_model=List[RiskLimit])
async def apply_risk_profile_to_account(
    account_id: str = Path(..., description="Account ID"),
    profile_id: str = Path(..., description="Risk profile ID"),
    api_key: str = Depends(get_api_key)
):
    """Apply a risk profile to an account."""
    try:
        result = risk_limits_service.apply_risk_profile(account_id, profile_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Risk profile {profile_id} not found")
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error applying risk profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to apply risk profile: {str(e)}")
