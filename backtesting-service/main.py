"""
Main entry point for the backtesting service.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import datetime

app = FastAPI(title="Backtesting Service")

class BacktestRequest(BaseModel):
    """Model for backtest request."""
    strategy_id: str
    start_date: str
    end_date: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]

class BacktestResult(BaseModel):
    """Model for backtest result."""
    id: str
    strategy_id: str
    start_date: str
    end_date: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: str

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/api/v1/backtest", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """Run a backtest."""
    # This is a simplified implementation for testing
    # In a real service, this would run a backtest and store the results
    
    # Validate input
    try:
        datetime.datetime.strptime(request.start_date, "%Y-%m-%d")
        datetime.datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Create a sample result
    result = BacktestResult(
        id="test_backtest_id",
        strategy_id=request.strategy_id,
        start_date=request.start_date,
        end_date=request.end_date,
        symbol=request.symbol,
        timeframe=request.timeframe,
        parameters=request.parameters,
        results={
            "trades": [
                {"entry_time": "2023-01-05T10:00:00", "exit_time": "2023-01-05T14:00:00", "entry_price": 1.0750, "exit_price": 1.0800, "profit_pips": 50, "profit_percent": 0.47},
                {"entry_time": "2023-01-10T12:00:00", "exit_time": "2023-01-10T16:00:00", "entry_price": 1.0800, "exit_price": 1.0750, "profit_pips": -50, "profit_percent": -0.46},
                {"entry_time": "2023-01-15T09:00:00", "exit_time": "2023-01-15T17:00:00", "entry_price": 1.0700, "exit_price": 1.0800, "profit_pips": 100, "profit_percent": 0.93}
            ],
            "equity_curve": [10000, 10047, 10000, 10093]
        },
        metrics={
            "total_trades": 3,
            "win_rate": 0.67,
            "profit_factor": 1.5,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.46,
            "average_trade": 0.31
        },
        created_at=datetime.datetime.now().isoformat()
    )
    
    return result

@app.get("/api/v1/backtest/{backtest_id}", response_model=BacktestResult)
async def get_backtest(backtest_id: str):
    """Get a backtest by ID."""
    # This is a simplified implementation for testing
    # In a real service, this would retrieve the backtest from a database
    
    if backtest_id != "test_backtest_id":
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    # Create a sample result
    result = BacktestResult(
        id=backtest_id,
        strategy_id="test_strategy",
        start_date="2023-01-01",
        end_date="2023-01-31",
        symbol="EURUSD",
        timeframe="1h",
        parameters={
            "risk_reward_ratio": 2.0,
            "stop_loss_pips": 20,
            "take_profit_pips": 40
        },
        results={
            "trades": [
                {"entry_time": "2023-01-05T10:00:00", "exit_time": "2023-01-05T14:00:00", "entry_price": 1.0750, "exit_price": 1.0800, "profit_pips": 50, "profit_percent": 0.47},
                {"entry_time": "2023-01-10T12:00:00", "exit_time": "2023-01-10T16:00:00", "entry_price": 1.0800, "exit_price": 1.0750, "profit_pips": -50, "profit_percent": -0.46},
                {"entry_time": "2023-01-15T09:00:00", "exit_time": "2023-01-15T17:00:00", "entry_price": 1.0700, "exit_price": 1.0800, "profit_pips": 100, "profit_percent": 0.93}
            ],
            "equity_curve": [10000, 10047, 10000, 10093]
        },
        metrics={
            "total_trades": 3,
            "win_rate": 0.67,
            "profit_factor": 1.5,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.46,
            "average_trade": 0.31
        },
        created_at=datetime.datetime.now().isoformat()
    )
    
    return result

@app.get("/api/v1/backtest/{backtest_id}/metrics")
async def get_backtest_metrics(backtest_id: str):
    """Get metrics for a backtest."""
    # This is a simplified implementation for testing
    # In a real service, this would retrieve the metrics from a database
    
    if backtest_id != "test_backtest_id":
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    return {
        "total_trades": 3,
        "win_rate": 0.67,
        "profit_factor": 1.5,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.46,
        "average_trade": 0.31
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
