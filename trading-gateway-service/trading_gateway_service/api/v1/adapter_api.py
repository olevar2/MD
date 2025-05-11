"""
Adapter API Module

This module provides API endpoints that use the adapter pattern.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

from common_lib.interfaces.trading import (
    ITradingProvider,
    IOrderBookProvider,
    OrderType,
    OrderSide,
    OrderStatus
)
from common_lib.interfaces.risk_management import IRiskManager
from common_lib.errors.base_exceptions import (
    BaseError, ValidationError, DataError, ServiceError
)

from trading_gateway_service.api.dependencies import (
    get_trading_provider,
    get_order_book_provider,
    get_risk_manager
)
from trading_gateway_service.core.logging import get_logger

# Configure logging
logger = get_logger("trading-gateway-service.adapter-api")

# Create router
adapter_router = APIRouter(
    prefix="/api/v1/adapter",
    tags=["adapter"],
    responses={404: {"description": "Not found"}}
)


# Request models
class OrderRequest(BaseModel):
    """Request model for placing an order."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'EURUSD')")
    order_type: OrderType = Field(..., description="Type of order")
    side: OrderSide = Field(..., description="Order side")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Order price (required for LIMIT orders)")
    stop_price: Optional[float] = Field(None, description="Stop price (required for STOP orders)")
    time_in_force: Optional[str] = Field("GTC", description="Time in force (e.g., 'GTC', 'IOC', 'FOK')")
    client_order_id: Optional[str] = Field(None, description="Client-generated order ID")


# Response models
class OrderResponse(BaseModel):
    """Response model for order operations."""
    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    order_type: str
    side: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    filled_quantity: float = 0
    average_price: Optional[float] = None
    message: Optional[str] = None


class OrderBookResponse(BaseModel):
    """Response model for order book data."""
    symbol: str
    timestamp: datetime
    bids: List[Dict[str, float]]
    asks: List[Dict[str, float]]


class RiskCheckResponse(BaseModel):
    """Response model for risk checks."""
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None


# API endpoints
@adapter_router.post("/orders", response_model=OrderResponse)
async def place_order(
    order_request: OrderRequest,
    trading_provider: ITradingProvider = Depends(get_trading_provider)
):
    """
    Place a new order.

    Args:
        order_request: The order to place
        trading_provider: The trading provider adapter

    Returns:
        OrderResponse containing the order details
    """
    try:
        # Call the trading provider
        result = await trading_provider.place_order(
            symbol=order_request.symbol,
            order_type=order_request.order_type,
            side=order_request.side,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            time_in_force=order_request.time_in_force,
            client_order_id=order_request.client_order_id
        )

        # Convert to response model
        return OrderResponse(
            order_id=result.get("order_id", ""),
            client_order_id=result.get("client_order_id", order_request.client_order_id),
            symbol=order_request.symbol,
            order_type=order_request.order_type.value,
            side=order_request.side.value,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            status=result.get("status", "PENDING"),
            created_at=result.get("created_at", datetime.utcnow()),
            updated_at=result.get("updated_at"),
            filled_quantity=result.get("filled_quantity", 0),
            average_price=result.get("average_price"),
            message=result.get("message")
        )
    except ServiceError as e:
        logger.error(f"Service error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error(f"Base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@adapter_router.delete("/orders/{order_id}", response_model=Dict[str, Any])
async def cancel_order(
    order_id: str = Path(..., description="ID of the order to cancel"),
    trading_provider: ITradingProvider = Depends(get_trading_provider)
):
    """
    Cancel an order.

    Args:
        order_id: ID of the order to cancel
        trading_provider: The trading provider adapter

    Returns:
        Dictionary with cancellation result
    """
    try:
        # Call the trading provider
        result = await trading_provider.cancel_order(order_id)

        if result:
            return {"success": True, "message": f"Order {order_id} cancelled successfully"}
        else:
            return {"success": False, "message": f"Failed to cancel order {order_id}"}
    except ServiceError as e:
        logger.error(f"Service error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except DataError as e:
        logger.error(f"Data error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except BaseError as e:
        logger.error(f"Base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@adapter_router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str = Path(..., description="ID of the order to get"),
    trading_provider: ITradingProvider = Depends(get_trading_provider)
):
    """
    Get order details.

    Args:
        order_id: ID of the order to get
        trading_provider: The trading provider adapter

    Returns:
        OrderResponse containing the order details
    """
    try:
        # Call the trading provider
        result = await trading_provider.get_order(order_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Convert to response model
        return OrderResponse(
            order_id=result.get("order_id", order_id),
            client_order_id=result.get("client_order_id"),
            symbol=result.get("symbol", ""),
            order_type=result.get("order_type", ""),
            side=result.get("side", ""),
            quantity=result.get("quantity", 0),
            price=result.get("price"),
            stop_price=result.get("stop_price"),
            status=result.get("status", "UNKNOWN"),
            created_at=result.get("created_at", datetime.utcnow()),
            updated_at=result.get("updated_at"),
            filled_quantity=result.get("filled_quantity", 0),
            average_price=result.get("average_price")
        )
    except ServiceError as e:
        logger.error(f"Service error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except DataError as e:
        logger.error(f"Data error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except BaseError as e:
        logger.error(f"Base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@adapter_router.get("/orderbook/{symbol}", response_model=OrderBookResponse)
async def get_order_book(
    symbol: str = Path(..., description="Trading symbol (e.g., 'EURUSD')"),
    depth: int = Query(10, description="Depth of the order book"),
    order_book_provider: IOrderBookProvider = Depends(get_order_book_provider)
):
    """
    Get the order book for a symbol.

    Args:
        symbol: Trading symbol
        depth: Depth of the order book
        order_book_provider: The order book provider adapter

    Returns:
        OrderBookResponse containing the order book data
    """
    try:
        # Call the order book provider
        result = await order_book_provider.get_order_book(
            symbol=symbol,
            depth=depth
        )

        # Convert to response model
        return OrderBookResponse(
            symbol=symbol,
            timestamp=datetime.fromisoformat(result.get("timestamp")) if isinstance(result.get("timestamp"), str) else result.get("timestamp", datetime.utcnow()),
            bids=result.get("bids", []),
            asks=result.get("asks", [])
        )
    except ServiceError as e:
        logger.error(f"Service error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except DataError as e:
        logger.error(f"Data error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except BaseError as e:
        logger.error(f"Base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@adapter_router.post("/risk/check", response_model=RiskCheckResponse)
async def check_risk(
    order_request: OrderRequest,
    risk_manager: IRiskManager = Depends(get_risk_manager)
):
    """
    Check if an order meets risk criteria.

    Args:
        order_request: The order to check
        risk_manager: The risk manager adapter

    Returns:
        RiskCheckResponse containing the risk check result
    """
    try:
        # Call the risk manager
        result = await risk_manager.check_order(
            symbol=order_request.symbol,
            order_type=order_request.order_type,
            side=order_request.side,
            quantity=order_request.quantity,
            price=order_request.price
        )

        # Convert to response model
        return RiskCheckResponse(
            is_valid=result.get("is_valid", False),
            message=result.get("message", ""),
            details=result.get("details")
        )
    except ServiceError as e:
        logger.error(f"Service error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error(f"Base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
