"""
Database Connection Module for Risk Management Service.

Re-exports database utilities from common-lib.
"""

# Import and re-export from common-lib
from common_lib.database import (
    get_db_session, 
    get_async_db_session, 
    Base, 
    DatabaseManager, 
    DatabaseConnectionConfig,
    retry_on_db_error,
    SQLAlchemyError,
    OperationalError,
    IntegrityError
)
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

# Optional: Configure a specific manager instance if needed, 
# otherwise services can directly use get_db_session
# risk_db_config = DatabaseConnectionConfig.from_env(prefix="RISK_MANAGEMENT_")
# risk_db_manager = DatabaseManager(config=risk_db_config)

# Expose the necessary components for the service
__all__ = [
    "get_db_session",
    "get_async_db_session",
    "Base",
    "Session",
    "AsyncSession",
    "DatabaseManager", 
    "DatabaseConnectionConfig",
    "retry_on_db_error",
    "SQLAlchemyError",
    "OperationalError",
    "IntegrityError"
    # "risk_db_manager" # Uncomment if specific manager instance is created
]