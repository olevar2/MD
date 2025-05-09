"""
API router for the Python components of the Trading Gateway Service.

This module defines the API endpoints for the Python components of the
Trading Gateway Service, which handle market data processing, order
reconciliation, and other backend functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import Dict, List, Any, Optional

from trading_gateway_service.error import (
    ForexTradingPlatformError,
    BrokerConnectionError,
    OrderValidationError,
    MarketDataError,
    ServiceError,
    with_exception_handling,
    async_with_exception_handling
)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Trading Gateway"])

# Market data endpoints
@router.get("/market-data/{symbol}")
@async_with_exception_handling
async def get_market_data(
    request: Request,
    symbol: str,
    timeframe: Optional[str] = "1m"
):
    """
    Get market data for a specific symbol and timeframe.

    Args:
        symbol: The currency pair symbol (e.g., EURUSD)
        timeframe: The timeframe for the data (e.g., 1m, 5m, 1h)

    Returns:
        Market data for the specified symbol and timeframe
    """
    market_data_service = request.app.state.market_data_service

    try:
        data = await market_data_service.get_market_data(symbol, timeframe)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data
        }
    except Exception as e:
        raise MarketDataError(
            message=f"Failed to fetch market data for {symbol}",
            symbol=symbol,
            details={"timeframe": timeframe, "error": str(e)}
        )

# Order reconciliation endpoints
@router.post("/reconcile/orders")
@async_with_exception_handling
async def reconcile_orders(request: Request):
    """
    Trigger order reconciliation process.

    Returns:
        Reconciliation results
    """
    reconciliation_service = request.app.state.order_reconciliation_service

    try:
        result = await reconciliation_service.reconcile_orders()
        return {
            "status": "success",
            "reconciled_orders": result.get("reconciled_orders", 0),
            "discrepancies": result.get("discrepancies", [])
        }
    except Exception as e:
        raise ServiceError(
            message=f"Failed to reconcile orders: {str(e)}",
            details={"error": str(e)}
        )

# Monitoring endpoints
@router.get("/monitoring/performance")
@async_with_exception_handling
async def get_performance_metrics(request: Request):
    """
    Get performance metrics for the trading gateway.

    Returns:
        Performance metrics
    """
    monitoring = request.app.state.monitoring

    try:
        metrics = monitoring.get_metrics()
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        raise ServiceError(
            message=f"Failed to get performance metrics: {str(e)}",
            details={"error": str(e)}
        )

# Degraded mode endpoints
@router.get("/status/degraded-mode")
@async_with_exception_handling
async def get_degraded_mode_status(request: Request):
    """
    Get current degraded mode status.

    Returns:
        Current degraded mode status
    """
    degraded_mode_manager = request.app.state.degraded_mode_manager

    try:
        status = degraded_mode_manager.get_status()
        return {
            "is_degraded": status.get("is_degraded", False),
            "level": status.get("level", "NORMAL"),
            "reason": status.get("reason", None),
            "message": status.get("message", None),
            "active_fallbacks": status.get("active_fallbacks", [])
        }
    except Exception as e:
        raise ServiceError(
            message=f"Failed to get degraded mode status: {str(e)}",
            details={"error": str(e)}
        )
