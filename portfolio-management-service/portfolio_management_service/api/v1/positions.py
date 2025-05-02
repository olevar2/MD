"""
Positions API Module.

API endpoints for managing trading positions.
"""
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path, status

from core_foundations.utils.logger import get_logger
from portfolio_management_service.models.position import Position, PositionCreate, PositionUpdate
from portfolio_management_service.services.portfolio_service import PortfolioService

# Import common-lib exceptions
from common_lib.exceptions import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    TradingError,
    OrderExecutionError
)

# Initialize logger
logger = get_logger("positions-api")

# Create router
router = APIRouter()

# Initialize service
portfolio_service = PortfolioService()

# Constants
POSITION_ID_DESC = "Position ID"
POSITION_NOT_FOUND = "Position not found"


@router.post("/", response_model=Position, status_code=201)
async def create_position(position_data: PositionCreate) -> Position:
    """
    Create a new trading position.

    Args:
        position_data: Data for the new position

    Returns:
        Created position
    """
    try:
        position = portfolio_service.create_position(position_data)
        logger.info(f"API: Created position for {position_data.symbol}")
        return position
    except DataValidationError as e:
        logger.error(f"API: Data validation error during position creation: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Data validation error: {e.message}")
    except TradingError as e:
        logger.error(f"API: Trading error during position creation: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Trading error: {e.message}")
    except OrderExecutionError as e:
        logger.error(f"API: Order execution error during position creation: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Order execution error: {e.message}")
    except DataStorageError as e:
        logger.error(f"API: Data storage error during position creation: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data storage error: {e.message}")
    except ForexTradingPlatformError as e:
        logger.error(f"API: Platform error during position creation: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Platform error: {e.message}")
    except ValueError as e:
        logger.error(f"API: Failed to create position: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"API: Unexpected error creating position: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create position")


@router.get("/{position_id}", response_model=Position)
async def get_position(position_id: str = Path(..., description=POSITION_ID_DESC)) -> Position:
    """
    Get a position by ID.

    Args:
        position_id: Position ID

    Returns:
        Position details
    """
    try:
        position = portfolio_service.get_position(position_id)
        if not position:
            logger.warning(f"API: Position not found: {position_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=POSITION_NOT_FOUND)
        return position
    except DataFetchError as e:
        logger.error(f"API: Data fetch error during position retrieval: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data fetch error: {e.message}")
    except ForexTradingPlatformError as e:
        logger.error(f"API: Platform error during position retrieval: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Platform error: {e.message}")
    except Exception as e:
        logger.error(f"API: Unexpected error retrieving position: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve position")


@router.patch("/{position_id}", response_model=Position)
async def update_position(
    position_update: PositionUpdate,
    position_id: str = Path(..., description=POSITION_ID_DESC)
) -> Position:
    """
    Update a position.

    Args:
        position_id: Position ID
        position_update: Data to update

    Returns:
        Updated position
    """
    try:
        position = portfolio_service.update_position(position_id, position_update)
        if not position:
            logger.warning(f"API: Position not found for update: {position_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=POSITION_NOT_FOUND)
        logger.info(f"API: Updated position {position_id}")
        return position
    except DataValidationError as e:
        logger.error(f"API: Data validation error during position update: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Data validation error: {e.message}")
    except TradingError as e:
        logger.error(f"API: Trading error during position update: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Trading error: {e.message}")
    except DataFetchError as e:
        logger.error(f"API: Data fetch error during position update: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data fetch error: {e.message}")
    except DataStorageError as e:
        logger.error(f"API: Data storage error during position update: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data storage error: {e.message}")
    except ForexTradingPlatformError as e:
        logger.error(f"API: Platform error during position update: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Platform error: {e.message}")
    except ValueError as e:
        logger.error(f"API: Failed to update position: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"API: Unexpected error updating position: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update position")


@router.post("/{position_id}/close", response_model=Dict[str, Any])
async def close_position(
    position_id: str = Path(..., description=POSITION_ID_DESC),
    close_price: float = Query(..., description="Closing price"),
    quantity: Optional[float] = Query(None, description="Quantity to close, if None, close all")
) -> Dict[str, Any]:
    """
    Close a position or part of it.

    Args:
        position_id: Position ID
        close_price: Closing price
        quantity: Quantity to close, if None, close all

    Returns:
        Dictionary with position and P&L details
    """
    try:
        result = portfolio_service.close_position(position_id, close_price, quantity)
        if not result:
            logger.warning(f"API: Position not found for closing: {position_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=POSITION_NOT_FOUND)

        position, realized_pnl = result
        logger.info(f"API: Closed position {position_id} with realized P&L: {realized_pnl}")

        return {
            "position": position,
            "realized_pnl": realized_pnl,
            "status": position.status,
            "close_price": close_price
        }
    except DataValidationError as e:
        logger.error(f"API: Data validation error during position closing: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Data validation error: {e.message}")
    except TradingError as e:
        logger.error(f"API: Trading error during position closing: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Trading error: {e.message}")
    except OrderExecutionError as e:
        logger.error(f"API: Order execution error during position closing: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Order execution error: {e.message}")
    except DataFetchError as e:
        logger.error(f"API: Data fetch error during position closing: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data fetch error: {e.message}")
    except DataStorageError as e:
        logger.error(f"API: Data storage error during position closing: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data storage error: {e.message}")
    except ForexTradingPlatformError as e:
        logger.error(f"API: Platform error during position closing: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Platform error: {e.message}")
    except ValueError as e:
        logger.error(f"API: Failed to close position: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"API: Unexpected error closing position: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to close position")


@router.get("/", response_model=List[Position])
async def list_positions(
    account_id: Optional[str] = Query(None, description="Filter by account ID"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status"),
    side: Optional[str] = Query(None, description="Filter by side"),
    limit: int = Query(100, description="Maximum number of positions to return"),
    offset: int = Query(0, description="Number of positions to skip")
) -> List[Position]:
    """
    List positions with optional filtering.

    Args:
        account_id: Optional account ID filter
        symbol: Optional symbol filter
        status: Optional status filter
        side: Optional side filter
        limit: Maximum number of positions to return
        offset: Number of positions to skip

    Returns:
        List of positions
    """
    try:
        positions = portfolio_service.list_positions(
            account_id=account_id,
            symbol=symbol,
            status=status,
            side=side,
            limit=limit,
            offset=offset
        )
        return positions
    except DataValidationError as e:
        logger.error(f"API: Data validation error during position listing: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Data validation error: {e.message}")
    except DataFetchError as e:
        logger.error(f"API: Data fetch error during position listing: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data fetch error: {e.message}")
    except ForexTradingPlatformError as e:
        logger.error(f"API: Platform error during position listing: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Platform error: {e.message}")
    except Exception as e:
        logger.error(f"API: Unexpected error listing positions: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list positions")


@router.post("/prices/update", response_model=Dict[str, Any])
async def update_prices(symbol_prices: Dict[str, float]) -> Dict[str, Any]:
    """
    Update position prices based on new market prices.

    Args:
        symbol_prices: Dictionary mapping symbols to current prices

    Returns:
        Dictionary with update statistics
    """
    try:
        result = portfolio_service.update_prices(symbol_prices)
        logger.info(f"API: Updated {result['updated_positions']} position prices")
        return result
    except DataValidationError as e:
        logger.error(f"API: Data validation error during price update: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Data validation error: {e.message}")
    except DataFetchError as e:
        logger.error(f"API: Data fetch error during price update: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data fetch error: {e.message}")
    except DataStorageError as e:
        logger.error(f"API: Data storage error during price update: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data storage error: {e.message}")
    except ForexTradingPlatformError as e:
        logger.error(f"API: Platform error during price update: {e.message}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Platform error: {e.message}")
    except Exception as e:
        logger.error(f"API: Unexpected error updating prices: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update prices")