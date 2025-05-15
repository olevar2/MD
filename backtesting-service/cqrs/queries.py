"""
Query models for the Backtesting Service.

This module provides the query models for the Backtesting Service.
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from common_lib.cqrs.queries import Query


class GetBacktestQuery(Query):
    """Query to get a backtest by ID."""
    
    backtest_id: str = Field(..., description="ID of the backtest to retrieve")


class ListBacktestsQuery(Query):
    """Query to list backtests."""
    
    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    limit: int = Field(10, description="Maximum number of backtests to return")
    offset: int = Field(0, description="Offset for pagination")


class GetOptimizationQuery(Query):
    """Query to get an optimization by ID."""
    
    optimization_id: str = Field(..., description="ID of the optimization to retrieve")


class ListOptimizationsQuery(Query):
    """Query to list optimizations."""
    
    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    limit: int = Field(10, description="Maximum number of optimizations to return")
    offset: int = Field(0, description="Offset for pagination")


class GetWalkForwardTestQuery(Query):
    """Query to get a walk-forward test by ID."""
    
    test_id: str = Field(..., description="ID of the walk-forward test to retrieve")


class ListWalkForwardTestsQuery(Query):
    """Query to list walk-forward tests."""
    
    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    limit: int = Field(10, description="Maximum number of walk-forward tests to return")
    offset: int = Field(0, description="Offset for pagination")


class ListStrategiesQuery(Query):
    """Query to list available strategies."""
    
    category: Optional[str] = Field(None, description="Filter by strategy category")
    limit: int = Field(10, description="Maximum number of strategies to return")
    offset: int = Field(0, description="Offset for pagination")