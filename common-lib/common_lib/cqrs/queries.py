"""
Query-related classes for CQRS pattern.

This module provides the base classes for queries and query handlers
in the CQRS pattern.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Any, Optional
from pydantic import BaseModel

# Type variables for generic typing
TQuery = TypeVar('TQuery', bound=BaseModel)
TResult = TypeVar('TResult')

class Query(BaseModel):
    """Base class for all queries."""
    correlation_id: Optional[str] = None


class QueryHandler(Generic[TQuery, TResult], ABC):
    """Base class for all query handlers."""
    
    @abstractmethod
    async def handle(self, query: TQuery) -> TResult:
        """
        Handle the query.
        
        Args:
            query: The query to handle
            
        Returns:
            The result of handling the query
        """
        pass


class QueryBus:
    """
    Query bus for dispatching queries to their handlers.
    """
    def __init__(self):
        self._handlers: Dict[type, QueryHandler] = {}
    
    def register_handler(self, query_type: type, handler: QueryHandler) -> None:
        """
        Register a handler for a query type.
        
        Args:
            query_type: The type of query
            handler: The handler for the query
        """
        self._handlers[query_type] = handler
    
    async def dispatch(self, query: Query) -> Any:
        """
        Dispatch a query to its handler.
        
        Args:
            query: The query to dispatch
            
        Returns:
            The result of handling the query
            
        Raises:
            ValueError: If no handler is registered for the query type
        """
        handler = self._handlers.get(type(query))
        if not handler:
            raise ValueError(f"No handler registered for query type {type(query).__name__}")
        
        return await handler.handle(query)