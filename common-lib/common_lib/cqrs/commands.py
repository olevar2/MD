"""
Command-related classes for CQRS pattern.

This module provides the base classes for commands and command handlers
in the CQRS pattern.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Any, Optional
from pydantic import BaseModel

# Type variables for generic typing
TCommand = TypeVar('TCommand', bound=BaseModel)
TResult = TypeVar('TResult')

class Command(BaseModel):
    """Base class for all commands."""
    correlation_id: Optional[str] = None


class CommandHandler(Generic[TCommand, TResult], ABC):
    """Base class for all command handlers."""
    
    @abstractmethod
    async def handle(self, command: TCommand) -> TResult:
        """
        Handle the command.
        
        Args:
            command: The command to handle
            
        Returns:
            The result of handling the command
        """
        pass


class CommandBus:
    """
    Command bus for dispatching commands to their handlers.
    """
    def __init__(self):
        self._handlers: Dict[type, CommandHandler] = {}
    
    def register_handler(self, command_type: type, handler: CommandHandler) -> None:
        """
        Register a handler for a command type.
        
        Args:
            command_type: The type of command
            handler: The handler for the command
        """
        self._handlers[command_type] = handler
    
    async def dispatch(self, command: Command) -> Any:
        """
        Dispatch a command to its handler.
        
        Args:
            command: The command to dispatch
            
        Returns:
            The result of handling the command
            
        Raises:
            ValueError: If no handler is registered for the command type
        """
        handler = self._handlers.get(type(command))
        if not handler:
            raise ValueError(f"No handler registered for command type {type(command).__name__}")
        
        return await handler.handle(command)