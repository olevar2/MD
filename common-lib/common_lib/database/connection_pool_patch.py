"""
Patch for the connection_pool.py file to fix issues with the get_async_session method.
"""
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the connection_pool.py file
connection_pool_path = os.path.join(os.path.dirname(__file__), "connection_pool.py")

# Read the file
with open(connection_pool_path, "r") as f:
    content = f.read()

# Find the get_async_session method
get_async_session_pattern = r"@asynccontextmanager\s+@async_with_database_resilience\(\"get_async_session\"\)\s+@async_with_exception_handling\s+async def get_async_session\(self\) -> AsyncSession:.*?# Check if we're using mocks.*?if USE_MOCKS:.*?# Create a mock session.*?session = mock\.MagicMock\(spec=AsyncSession\).*?# Create mock result for session\.execute.*?result_mock = mock\.MagicMock\(\).*?result_mock\.fetchall = mock\.MagicMock\(return_value=\[\{\"id\": 1, \"name\": \"test\"\}\]\).*?result_mock\.fetchone = mock\.MagicMock\(return_value=\{\"id\": 1, \"name\": \"test\"\}\).*?result_mock\.scalar_one = mock\.MagicMock\(return_value=1\).*?result_mock\.scalar_one_or_none = mock\.MagicMock\(return_value=1\).*?result_mock\.scalars = mock\.MagicMock\(return_value=result_mock\).*?result_mock\.first = mock\.MagicMock\(return_value=\{\"id\": 1, \"name\": \"test\"\}\).*?result_mock\.all = mock\.MagicMock\(return_value=\[\{\"id\": 1, \"name\": \"test\"\}\]\).*?# Set up session methods.*?session\.execute = mock\.AsyncMock\(return_value=result_mock\).*?session\.commit = mock\.AsyncMock\(\).*?session\.rollback = mock\.AsyncMock\(\).*?session\.close = mock\.AsyncMock\(\).*?yield session.*?return"

# Replace the get_async_session method
get_async_session_replacement = """@asynccontextmanager
    @async_with_database_resilience("get_async_session")
    @async_with_exception_handling
    async def get_async_session(self) -> AsyncSession:
        \"\"\"
        Get an asynchronous database session.
        
        Yields:
            SQLAlchemy AsyncSession
        \"\"\"
        # Check if we're using mocks
        if USE_MOCKS:
            # Create a mock session
            session = mock.MagicMock(spec=AsyncSession)
            
            # Create mock result for session.execute
            result_mock = mock.MagicMock()
            result_mock.fetchall = mock.MagicMock(return_value=[{"id": 1, "name": "test"}])
            result_mock.fetchone = mock.MagicMock(return_value={"id": 1, "name": "test"})
            result_mock.scalar_one = mock.MagicMock(return_value=1)
            result_mock.scalar_one_or_none = mock.MagicMock(return_value=1)
            result_mock.scalars = mock.MagicMock(return_value=result_mock)
            result_mock.first = mock.MagicMock(return_value={"id": 1, "name": "test"})
            result_mock.all = mock.MagicMock(return_value=[{"id": 1, "name": "test"}])
            
            # Set up session methods
            session.execute = mock.AsyncMock(return_value=result_mock)
            session.commit = mock.AsyncMock()
            session.rollback = mock.AsyncMock()
            session.close = mock.AsyncMock()
            
            # Create an async context manager for the session
            class AsyncSessionContextManager:
                def __init__(self, session):
                    self.session = session
                    
                async def __aenter__(self):
                    return self.session
                    
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            # Return the async context manager
            yield session
            return"""

# Replace the get_async_session method
content = re.sub(get_async_session_pattern, get_async_session_replacement, content, flags=re.DOTALL)

# Write the file
with open(connection_pool_path, "w") as f:
    f.write(content)

logger.info(f"Patched {connection_pool_path}")