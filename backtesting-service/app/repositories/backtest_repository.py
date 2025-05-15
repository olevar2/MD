"""
Backtest Repository

This module provides repository classes for storing and retrieving backtest results.
"""
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from app.models.backtest_models import (
    BacktestResult,
    BacktestStatus,
    OptimizationResult,
    WalkForwardTestResult
)

logger = logging.getLogger(__name__)

class BacktestRepository:
    """
    Repository for backtest results.
    """
    def __init__(self, db_connection=None):
        """
        Initialize the backtest repository.
        
        Args:
            db_connection: Database connection object
        """
        self.db_connection = db_connection
        self.backtests = {}  # In-memory storage for backtests
        self.optimizations = {}  # In-memory storage for optimizations
        self.walk_forward_tests = {}  # In-memory storage for walk-forward tests
    
    async def save_backtest(self, backtest_id: str, backtest_result: Dict[str, Any]) -> None:
        """
        Save a backtest result.
        
        Args:
            backtest_id: ID of the backtest
            backtest_result: Backtest result data
        """
        # Store in memory
        self.backtests[backtest_id] = backtest_result
        
        # If database connection is available, store in database
        if self.db_connection:
            try:
                # Convert to JSON for storage
                backtest_json = json.dumps(backtest_result)
                
                # Store in database
                # This is a placeholder for actual database storage
                # await self.db_connection.execute(
                #     "INSERT INTO backtests (backtest_id, data) VALUES ($1, $2)",
                #     backtest_id, backtest_json
                # )
                
                logger.info(f"Saved backtest {backtest_id} to database")
            except Exception as e:
                logger.error(f"Error saving backtest to database: {e}")
    
    async def get_backtest(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a backtest result by ID.
        
        Args:
            backtest_id: ID of the backtest
            
        Returns:
            Backtest result or None if not found
        """
        # Check in-memory storage
        if backtest_id in self.backtests:
            return self.backtests[backtest_id]
        
        # If database connection is available, check database
        if self.db_connection:
            try:
                # This is a placeholder for actual database retrieval
                # result = await self.db_connection.fetchrow(
                #     "SELECT data FROM backtests WHERE backtest_id = $1",
                #     backtest_id
                # )
                
                # if result:
                #     backtest_data = json.loads(result['data'])
                #     
                #     # Cache in memory
                #     self.backtests[backtest_id] = backtest_data
                #     
                #     return backtest_data
                
                logger.info(f"Retrieved backtest {backtest_id} from database")
            except Exception as e:
                logger.error(f"Error retrieving backtest from database: {e}")
        
        return None
    
    async def update_backtest_status(self, backtest_id: str, status: str, message: Optional[str] = None) -> None:
        """
        Update the status of a backtest.
        
        Args:
            backtest_id: ID of the backtest
            status: New status
            message: Optional message
        """
        # Check if backtest exists
        backtest = await self.get_backtest(backtest_id)
        
        if backtest:
            # Update status
            backtest['status'] = status
            
            # Update message if provided
            if message:
                backtest['message'] = message
            
            # Save updated backtest
            await self.save_backtest(backtest_id, backtest)
            
            logger.info(f"Updated backtest {backtest_id} status to {status}")
        else:
            logger.warning(f"Backtest {backtest_id} not found for status update")
    
    async def list_backtests(self, strategy_id: Optional[str] = None, 
                           symbol: Optional[str] = None, 
                           limit: int = 100, 
                           offset: int = 0) -> List[Dict[str, Any]]:
        """
        List backtests with optional filtering.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            symbol: Optional symbol to filter by
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of backtests
        """
        # If database connection is available, query database
        if self.db_connection:
            try:
                # This is a placeholder for actual database query
                # query = "SELECT data FROM backtests"
                # params = []
                # 
                # if strategy_id:
                #     query += " WHERE data->>'strategy_id' = $1"
                #     params.append(strategy_id)
                # 
                # if symbol:
                #     if strategy_id:
                #         query += " AND data->>'symbol' = $2"
                #     else:
                #         query += " WHERE data->>'symbol' = $1"
                #     params.append(symbol)
                # 
                # query += " ORDER BY (data->>'created_at')::timestamp DESC LIMIT $" + str(len(params) + 1) + " OFFSET $" + str(len(params) + 2)
                # params.extend([limit, offset])
                # 
                # results = await self.db_connection.fetch(query, *params)
                # 
                # backtests = [json.loads(row['data']) for row in results]
                # 
                # return backtests
                
                logger.info(f"Listed backtests from database")
            except Exception as e:
                logger.error(f"Error listing backtests from database: {e}")
        
        # Fall back to in-memory storage
        backtests = list(self.backtests.values())
        
        # Apply filters
        if strategy_id:
            backtests = [b for b in backtests if b.get('strategy_id') == strategy_id]
        
        if symbol:
            backtests = [b for b in backtests if b.get('symbol') == symbol]
        
        # Sort by created_at (descending)
        backtests.sort(key=lambda b: b.get('created_at', ''), reverse=True)
        
        # Apply pagination
        backtests = backtests[offset:offset + limit]
        
        return backtests
    
    async def save_optimization(self, optimization_id: str, optimization_result: Dict[str, Any]) -> None:
        """
        Save an optimization result.
        
        Args:
            optimization_id: ID of the optimization
            optimization_result: Optimization result data
        """
        # Store in memory
        self.optimizations[optimization_id] = optimization_result
        
        # If database connection is available, store in database
        if self.db_connection:
            try:
                # Convert to JSON for storage
                optimization_json = json.dumps(optimization_result)
                
                # Store in database
                # This is a placeholder for actual database storage
                # await self.db_connection.execute(
                #     "INSERT INTO optimizations (optimization_id, data) VALUES ($1, $2)",
                #     optimization_id, optimization_json
                # )
                
                logger.info(f"Saved optimization {optimization_id} to database")
            except Exception as e:
                logger.error(f"Error saving optimization to database: {e}")
    
    async def get_optimization(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an optimization result by ID.
        
        Args:
            optimization_id: ID of the optimization
            
        Returns:
            Optimization result or None if not found
        """
        # Check in-memory storage
        if optimization_id in self.optimizations:
            return self.optimizations[optimization_id]
        
        # If database connection is available, check database
        if self.db_connection:
            try:
                # This is a placeholder for actual database retrieval
                # result = await self.db_connection.fetchrow(
                #     "SELECT data FROM optimizations WHERE optimization_id = $1",
                #     optimization_id
                # )
                
                # if result:
                #     optimization_data = json.loads(result['data'])
                #     
                #     # Cache in memory
                #     self.optimizations[optimization_id] = optimization_data
                #     
                #     return optimization_data
                
                logger.info(f"Retrieved optimization {optimization_id} from database")
            except Exception as e:
                logger.error(f"Error retrieving optimization from database: {e}")
        
        return None
    
    async def save_walk_forward_test(self, test_id: str, test_result: Dict[str, Any]) -> None:
        """
        Save a walk-forward test result.
        
        Args:
            test_id: ID of the walk-forward test
            test_result: Walk-forward test result data
        """
        # Store in memory
        self.walk_forward_tests[test_id] = test_result
        
        # If database connection is available, store in database
        if self.db_connection:
            try:
                # Convert to JSON for storage
                test_json = json.dumps(test_result)
                
                # Store in database
                # This is a placeholder for actual database storage
                # await self.db_connection.execute(
                #     "INSERT INTO walk_forward_tests (test_id, data) VALUES ($1, $2)",
                #     test_id, test_json
                # )
                
                logger.info(f"Saved walk-forward test {test_id} to database")
            except Exception as e:
                logger.error(f"Error saving walk-forward test to database: {e}")
    
    async def get_walk_forward_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a walk-forward test result by ID.
        
        Args:
            test_id: ID of the walk-forward test
            
        Returns:
            Walk-forward test result or None if not found
        """
        # Check in-memory storage
        if test_id in self.walk_forward_tests:
            return self.walk_forward_tests[test_id]
        
        # If database connection is available, check database
        if self.db_connection:
            try:
                # This is a placeholder for actual database retrieval
                # result = await self.db_connection.fetchrow(
                #     "SELECT data FROM walk_forward_tests WHERE test_id = $1",
                #     test_id
                # )
                
                # if result:
                #     test_data = json.loads(result['data'])
                #     
                #     # Cache in memory
                #     self.walk_forward_tests[test_id] = test_data
                #     
                #     return test_data
                
                logger.info(f"Retrieved walk-forward test {test_id} from database")
            except Exception as e:
                logger.error(f"Error retrieving walk-forward test from database: {e}")
        
        return None