from typing import Dict, List, Any, Optional
import json
import os
import asyncio
from datetime import datetime

from backtesting_service.models.backtest_models import BacktestResult, BacktestStatus

class BacktestRepository:
    """
    Repository for storing and retrieving backtest results.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the repository with a storage path.
        """
        self.storage_path = storage_path or os.environ.get("BACKTEST_STORAGE_PATH", "./data/backtests")
        os.makedirs(self.storage_path, exist_ok=True)
        self.backtests: Dict[str, BacktestResult] = {}
        self.lock = asyncio.Lock()
    
    async def save_backtest(self, backtest_id: str, backtest_result: BacktestResult) -> None:
        """
        Save a backtest result to the repository.
        """
        async with self.lock:
            self.backtests[backtest_id] = backtest_result
            
            # Also persist to disk for durability
            file_path = os.path.join(self.storage_path, f"{backtest_id}.json")
            with open(file_path, "w") as f:
                # Convert to dict for JSON serialization
                backtest_dict = backtest_result.dict()
                # Convert datetime objects to strings
                backtest_dict["start_time"] = backtest_dict["start_time"].isoformat()
                if backtest_dict["end_time"]:
                    backtest_dict["end_time"] = backtest_dict["end_time"].isoformat()
                
                # Convert trade datetime objects
                for trade in backtest_dict["trades"]:
                    trade["entry_time"] = trade["entry_time"].isoformat()
                    if trade["exit_time"]:
                        trade["exit_time"] = trade["exit_time"].isoformat()
                
                json.dump(backtest_dict, f, indent=2)
    
    async def get_backtest(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        Retrieve a backtest result from the repository.
        """
        async with self.lock:
            # Try to get from in-memory cache first
            if backtest_id in self.backtests:
                return self.backtests[backtest_id]
            
            # If not in memory, try to load from disk
            file_path = os.path.join(self.storage_path, f"{backtest_id}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    backtest_dict = json.load(f)
                    
                    # Convert string back to datetime
                    backtest_dict["start_time"] = datetime.fromisoformat(backtest_dict["start_time"])
                    if backtest_dict["end_time"]:
                        backtest_dict["end_time"] = datetime.fromisoformat(backtest_dict["end_time"])
                    
                    # Convert trade datetime strings
                    for trade in backtest_dict["trades"]:
                        trade["entry_time"] = datetime.fromisoformat(trade["entry_time"])
                        if trade["exit_time"]:
                            trade["exit_time"] = datetime.fromisoformat(trade["exit_time"])
                    
                    backtest_result = BacktestResult(**backtest_dict)
                    # Cache in memory for future use
                    self.backtests[backtest_id] = backtest_result
                    return backtest_result
            
            return None
    
    async def delete_backtest(self, backtest_id: str) -> bool:
        """
        Delete a backtest result from the repository.
        """
        async with self.lock:
            if backtest_id in self.backtests:
                del self.backtests[backtest_id]
            
            file_path = os.path.join(self.storage_path, f"{backtest_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            
            return False
    
    async def list_backtests(
        self,
        limit: int = 10,
        offset: int = 0,
        strategy_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[BacktestResult]:
        """
        List all backtest results with optional filtering.
        """
        async with self.lock:
            # List files in the storage directory
            backtest_files = [f for f in os.listdir(self.storage_path) if f.endswith(".json")]
            
            # Load all backtests
            backtests = []
            for filename in backtest_files:
                backtest_id = filename[:-5]  # Remove .json extension
                backtest = await self.get_backtest(backtest_id)
                if backtest:
                    # Apply filters
                    if strategy_id and backtest.request.strategy.strategy_id != strategy_id:
                        continue
                    if status and backtest.status != status:
                        continue
                    
                    backtests.append(backtest)
            
            # Sort by start time (newest first)
            backtests.sort(key=lambda x: x.start_time, reverse=True)
            
            # Apply pagination
            return backtests[offset:offset+limit]