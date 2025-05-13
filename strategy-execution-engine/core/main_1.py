"""
Mock main module for testing.
"""

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, AsyncMock

# Create mock objects
mock_service_container = MagicMock()
mock_service_container.initialize = AsyncMock()
mock_service_container.shutdown = AsyncMock()

mock_strategy_loader = MagicMock()
mock_strategy_loader.load_strategies = AsyncMock()

mock_backtester = MagicMock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Args:
        app: FastAPI application
    """
    # Set up app state
    app.state.service_container = mock_service_container
    app.state.strategy_loader = mock_strategy_loader
    app.state.backtester = mock_backtester
    
    # Initialize services
    await mock_service_container.initialize()
    await mock_strategy_loader.load_strategies()
    
    yield
    
    # Shutdown services
    await mock_service_container.shutdown()

def create_app() -> FastAPI:
    """
    Create FastAPI application.
    
    Returns:
        FastAPI: Application instance
    """
    app = FastAPI(
        title="Strategy Execution Engine",
        description="API for strategy execution and backtesting",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routes
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Strategy Execution Engine is running",
            "version": "0.1.0"
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": "0.1.0"
        }
    
    @app.get("/api/v1/strategies")
    async def list_strategies():
        """List all strategies"""
        return {
            "strategies": [
                {
                    "id": "strategy1",
                    "name": "Test Strategy 1",
                    "type": "custom",
                    "status": "active",
                    "instruments": ["EUR/USD", "GBP/USD"],
                    "timeframe": "1h",
                    "description": "Test strategy 1",
                    "parameters": {"param1": 10, "param2": "value"}
                },
                {
                    "id": "strategy2",
                    "name": "Test Strategy 2",
                    "type": "custom",
                    "status": "active",
                    "instruments": ["USD/JPY"],
                    "timeframe": "4h",
                    "description": "Test strategy 2",
                    "parameters": {"param1": 20, "param2": "value2"}
                }
            ]
        }
    
    @app.get("/api/v1/strategies/{strategy_id}")
    async def get_strategy(strategy_id: str):
        """Get strategy details"""
        if strategy_id == "non_existent":
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "id": strategy_id,
            "name": "Test Strategy 1",
            "type": "custom",
            "status": "active",
            "instruments": ["EUR/USD", "GBP/USD"],
            "timeframe": "1h",
            "description": "Test strategy 1",
            "parameters": {"param1": 10, "param2": "value"}
        }
    
    @app.post("/api/v1/strategies/register", status_code=201)
    async def register_strategy(request: Request):
        """Register a new strategy"""
        data = await request.json()
        
        return {
            "id": "new_strategy_id",
            "name": data.get("name", "New Strategy"),
            "status": "active",
            "message": f"Strategy {data.get('name', 'New Strategy')} registered successfully"
        }
    
    @app.post("/api/v1/backtest")
    async def run_backtest(request: Request):
        """Run a backtest"""
        data = await request.json()
        
        return {
            "backtest_id": "test_backtest_id",
            "strategy_id": data.get("strategy_id", "strategy1"),
            "start_date": data.get("start_date", "2023-01-01"),
            "end_date": data.get("end_date", "2023-12-31"),
            "metrics": {
                "total_trades": 50,
                "winning_trades": 30,
                "losing_trades": 20,
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "net_profit": 5000,
                "net_profit_pct": 50,
                "max_drawdown": 10
            },
            "trades": [
                {
                    "id": "trade1",
                    "position_id": "position1",
                    "instrument": "EUR/USD",
                    "type": "long",
                    "entry_price": 1.1000,
                    "entry_time": "2023-01-05T10:00:00",
                    "exit_price": 1.1100,
                    "exit_time": "2023-01-06T10:00:00",
                    "size": 1.0,
                    "profit_loss": 100,
                    "profit_loss_pct": 1.0
                }
            ],
            "equity_curve": [
                {
                    "timestamp": "2023-01-01T00:00:00",
                    "equity": 10000
                },
                {
                    "timestamp": "2023-12-31T00:00:00",
                    "equity": 15000
                }
            ]
        }
    
    return app
