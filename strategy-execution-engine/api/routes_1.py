"""
API Routes for Strategy Execution Engine

This module sets up all API routes for the Strategy Execution Engine.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, FastAPI, Depends, HTTPException, Query, Body, status
from pydantic import BaseModel, Field

from config.config_1 import get_settings
from core.strategy_loader import StrategyLoader
from core.backtester import backtester
from core.error import (
    async_with_error_handling,
    StrategyExecutionError,
    StrategyConfigurationError,
    StrategyLoadError,
    BacktestError
)

logger = logging.getLogger(__name__)

# Create main router
main_router = APIRouter()

# Root endpoint
@main_router.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Strategy Execution Engine is running",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Create API router for strategies
strategies_router = APIRouter(prefix="/api/v1/strategies", tags=["strategies"])

# Request/Response Models
class StrategyInfo(BaseModel):
    """Strategy information model"""
    id: str
    name: str
    type: str
    status: str
    instruments: List[str]
    timeframe: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class StrategyListResponse(BaseModel):
    """Response model for strategy list"""
    strategies: List[StrategyInfo]

class StrategyRegisterRequest(BaseModel):
    """Request model for registering a strategy"""
    name: str
    description: Optional[str] = None
    instruments: List[str]
    timeframe: str
    parameters: Optional[Dict[str, Any]] = None
    code: str

class StrategyResponse(BaseModel):
    """Response model for strategy operations"""
    id: str
    name: str
    status: str
    message: str

class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    strategy_id: str
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    parameters: Optional[Dict[str, Any]] = None

class BacktestResponse(BaseModel):
    """Response model for backtest results"""
    backtest_id: str
    strategy_id: str
    start_date: str
    end_date: str
    metrics: Dict[str, Any]
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]

# Strategy endpoints
@strategies_router.get("", response_model=StrategyListResponse)
@async_with_error_handling
async def list_strategies():
    """
    List all registered strategies.
    """
    strategy_loader = StrategyLoader()
    strategies = strategy_loader.get_available_strategies()
    
    strategy_list = []
    for strategy_id, strategy_info in strategies.items():
        strategy_list.append(
            StrategyInfo(
                id=strategy_id,
                name=strategy_info.get("name", "Unknown"),
                type=strategy_info.get("type", "custom"),
                status=strategy_info.get("status", "active"),
                instruments=strategy_info.get("instruments", []),
                timeframe=strategy_info.get("timeframe", "1h"),
                description=strategy_info.get("description"),
                parameters=strategy_info.get("parameters")
            )
        )
    
    return {"strategies": strategy_list}

@strategies_router.get("/{strategy_id}", response_model=StrategyInfo)
@async_with_error_handling
async def get_strategy(strategy_id: str):
    """
    Get details of a specific strategy.
    """
    strategy_loader = StrategyLoader()
    strategy = strategy_loader.get_strategy(strategy_id)
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found"
        )
    
    return StrategyInfo(
        id=strategy_id,
        name=strategy.name,
        type=getattr(strategy, "type", "custom"),
        status="active" if getattr(strategy, "is_active", True) else "inactive",
        instruments=getattr(strategy, "instruments", []),
        timeframe=getattr(strategy, "timeframe", "1h"),
        description=getattr(strategy, "description", None),
        parameters=getattr(strategy, "parameters", {})
    )

@strategies_router.post("/register", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
@async_with_error_handling
async def register_strategy(request: StrategyRegisterRequest):
    """
    Register a new strategy.
    """
    strategy_loader = StrategyLoader()
    
    # Register the strategy
    strategy_id = await strategy_loader.register_strategy(
        name=request.name,
        code=request.code,
        instruments=request.instruments,
        timeframe=request.timeframe,
        parameters=request.parameters,
        description=request.description
    )
    
    return {
        "id": strategy_id,
        "name": request.name,
        "status": "active",
        "message": f"Strategy {request.name} registered successfully"
    }

# Create API router for backtesting
backtest_router = APIRouter(prefix="/api/v1/backtest", tags=["backtesting"])

@backtest_router.post("", response_model=BacktestResponse)
@async_with_error_handling
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest for a strategy.
    """
    # Run the backtest
    result = await backtester.run_backtest(
        strategy_id=request.strategy_id,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        parameters=request.parameters
    )
    
    return {
        "backtest_id": result["backtest_id"],
        "strategy_id": request.strategy_id,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "metrics": result["metrics"],
        "trades": result["trades"],
        "equity_curve": result["equity_curve"]
    }

def setup_routes(app: FastAPI) -> None:
    """
    Set up all API routes for the application.

    Args:
        app (FastAPI): The FastAPI application instance
    """
    # Include main router
    app.include_router(main_router)
    
    # Include strategies router
    app.include_router(strategies_router)
    
    # Include backtest router
    app.include_router(backtest_router)
    
    logger.info("API routes configured")
