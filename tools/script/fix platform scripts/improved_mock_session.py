"""
Improved mock session implementation for database testing.

This module provides a more sophisticated mock session that can parse and respond to
different types of queries.
"""
import re
import logging
import unittest.mock as mock
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockQueryParser:
    """Parse SQL queries and return appropriate results."""
    
    def __init__(self):
        """Initialize the query parser."""
        # In-memory database schema
        self.tables = {}
        
        # Initialize with some default tables
        self.initialize_default_schema()
    
    def initialize_default_schema(self):
        """Initialize the default schema."""
        # Create a users table
        self.tables["users"] = {
            "columns": ["id", "username", "email", "created_at", "updated_at"],
            "rows": [
                {"id": 1, "username": "user1", "email": "user1@example.com", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 2, "username": "user2", "email": "user2@example.com", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 3, "username": "user3", "email": "user3@example.com", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 4, "username": "user4", "email": "user4@example.com", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 5, "username": "user5", "email": "user5@example.com", "created_at": datetime.now(), "updated_at": datetime.now()},
            ],
        }
        
        # Create a posts table
        self.tables["posts"] = {
            "columns": ["id", "user_id", "title", "content", "created_at", "updated_at"],
            "rows": [
                {"id": 1, "user_id": 1, "title": "Post 1", "content": "Content 1", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 2, "user_id": 1, "title": "Post 2", "content": "Content 2", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 3, "user_id": 2, "title": "Post 3", "content": "Content 3", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 4, "user_id": 3, "title": "Post 4", "content": "Content 4", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 5, "user_id": 3, "title": "Post 5", "content": "Content 5", "created_at": datetime.now(), "updated_at": datetime.now()},
            ],
        }
        
        # Create a comments table
        self.tables["comments"] = {
            "columns": ["id", "post_id", "user_id", "content", "created_at", "updated_at"],
            "rows": [
                {"id": 1, "post_id": 1, "user_id": 2, "content": "Comment 1", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 2, "post_id": 1, "user_id": 3, "content": "Comment 2", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 3, "post_id": 2, "user_id": 4, "content": "Comment 3", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 4, "post_id": 3, "user_id": 5, "content": "Comment 4", "created_at": datetime.now(), "updated_at": datetime.now()},
                {"id": 5, "post_id": 4, "user_id": 1, "content": "Comment 5", "created_at": datetime.now(), "updated_at": datetime.now()},
            ],
        }
    
    def add_table(self, table_name: str, columns: List[str], rows: List[Dict[str, Any]]):
        """
        Add a table to the in-memory database.
        
        Args:
            table_name: Name of the table
            columns: List of column names
            rows: List of rows
        """
        self.tables[table_name] = {
            "columns": columns,
            "rows": rows,
        }
    
    def add_row(self, table_name: str, row: Dict[str, Any]):
        """
        Add a row to a table.
        
        Args:
            table_name: Name of the table
            row: Row to add
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist")
        
        self.tables[table_name]["rows"].append(row)
    
    def update_row(self, table_name: str, row_id: int, updates: Dict[str, Any]):
        """
        Update a row in a table.
        
        Args:
            table_name: Name of the table
            row_id: ID of the row to update
            updates: Updates to apply
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist")
        
        for i, row in enumerate(self.tables[table_name]["rows"]):
            if row["id"] == row_id:
                self.tables[table_name]["rows"][i].update(updates)
                return
        
        raise ValueError(f"Row with ID {row_id} not found in table {table_name}")
    
    def delete_row(self, table_name: str, row_id: int):
        """
        Delete a row from a table.
        
        Args:
            table_name: Name of the table
            row_id: ID of the row to delete
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist")
        
        for i, row in enumerate(self.tables[table_name]["rows"]):
            if row["id"] == row_id:
                del self.tables[table_name]["rows"][i]
                return
        
        raise ValueError(f"Row with ID {row_id} not found in table {table_name}")
    
    def parse_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse a SQL query and return appropriate results.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        # Convert query to lowercase for easier parsing
        query_lower = query.lower()
        
        # Parse SELECT queries
        if query_lower.startswith("select"):
            return self.parse_select_query(query, params)
        
        # Parse INSERT queries
        elif query_lower.startswith("insert"):
            return self.parse_insert_query(query, params)
        
        # Parse UPDATE queries
        elif query_lower.startswith("update"):
            return self.parse_update_query(query, params)
        
        # Parse DELETE queries
        elif query_lower.startswith("delete"):
            return self.parse_delete_query(query, params)
        
        # Default response for other queries
        else:
            return {
                "type": "unknown",
                "rowcount": 0,
                "rows": [],
            }
    
    def parse_select_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse a SELECT query and return appropriate results.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        # Check if it's a COUNT query
        if "count(" in query.lower():
            return {
                "type": "select",
                "count": True,
                "rowcount": 1,
                "rows": [{"count": 10}],
            }
        
        # Extract table name from the query
        table_match = re.search(r"from\s+(\w+)", query.lower())
        if table_match:
            table_name = table_match.group(1)
            
            # Check if the table exists
            if table_name in self.tables:
                # Get the rows from the table
                rows = self.tables[table_name]["rows"]
                
                # Apply WHERE clause if present
                where_match = re.search(r"where\s+(.*?)(?:order by|group by|limit|$)", query.lower())
                if where_match and params:
                    where_clause = where_match.group(1).strip()
                    
                    # Extract conditions from the WHERE clause
                    conditions = []
                    for condition in where_clause.split("and"):
                        condition = condition.strip()
                        
                        # Extract column name and parameter name
                        column_match = re.search(r"(\w+)\s*=\s*:(\w+)", condition)
                        if column_match:
                            column_name = column_match.group(1)
                            param_name = column_match.group(2)
                            
                            # Check if the parameter exists
                            if param_name in params:
                                conditions.append((column_name, params[param_name]))
                    
                    # Filter rows based on conditions
                    filtered_rows = []
                    for row in rows:
                        match = True
                        for column_name, value in conditions:
                            if column_name in row and row[column_name] != value:
                                match = False
                                break
                        
                        if match:
                            filtered_rows.append(row)
                    
                    rows = filtered_rows
                
                return {
                    "type": "select",
                    "count": False,
                    "rowcount": len(rows),
                    "rows": rows,
                }
        
        # Default response for SELECT queries
        return {
            "type": "select",
            "count": False,
            "rowcount": 10,
            "rows": [{"id": i, "name": f"test_{i}", "value": i * 10} for i in range(1, 11)],
        }
    
    def parse_insert_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse an INSERT query and return appropriate results.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        # Extract table name from the query
        table_match = re.search(r"insert\s+into\s+(\w+)", query.lower())
        if table_match:
            table_name = table_match.group(1)
            
            # Check if the table exists
            if table_name in self.tables:
                # Get the next ID
                next_id = 1
                if self.tables[table_name]["rows"]:
                    next_id = max(row["id"] for row in self.tables[table_name]["rows"]) + 1
                
                # Create a new row
                new_row = {"id": next_id}
                
                # Extract column names and parameter names from the query
                columns_match = re.search(r"\(([^)]+)\)\s+values\s+\(([^)]+)\)", query.lower())
                if columns_match and params:
                    column_names = [col.strip() for col in columns_match.group(1).split(",")]
                    param_names = [param.strip() for param in columns_match.group(2).split(",")]
                    
                    # Map column names to parameter values
                    for i, column_name in enumerate(column_names):
                        if i < len(param_names):
                            param_name = param_names[i].strip(":")
                            if param_name in params:
                                new_row[column_name] = params[param_name]
                
                # Add created_at and updated_at if they exist in the table
                if "created_at" in self.tables[table_name]["columns"] and "created_at" not in new_row:
                    new_row["created_at"] = datetime.now()
                if "updated_at" in self.tables[table_name]["columns"] and "updated_at" not in new_row:
                    new_row["updated_at"] = datetime.now()
                
                # Add the new row to the table
                self.tables[table_name]["rows"].append(new_row)
                
                return {
                    "type": "insert",
                    "rowcount": 1,
                    "rows": [{"id": next_id}],
                }
        
        # Default response for INSERT queries
        return {
            "type": "insert",
            "rowcount": 1,
            "rows": [{"id": 1}],
        }
    
    def parse_update_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse an UPDATE query and return appropriate results.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        # Extract table name from the query
        table_match = re.search(r"update\s+(\w+)", query.lower())
        if table_match:
            table_name = table_match.group(1)
            
            # Check if the table exists
            if table_name in self.tables:
                # Extract SET clause from the query
                set_match = re.search(r"set\s+(.*?)(?:where|$)", query.lower())
                if set_match and params:
                    set_clause = set_match.group(1).strip()
                    
                    # Extract column names and parameter names from the SET clause
                    updates = {}
                    for update in set_clause.split(","):
                        update = update.strip()
                        
                        # Extract column name and parameter name
                        column_match = re.search(r"(\w+)\s*=\s*:(\w+)", update)
                        if column_match:
                            column_name = column_match.group(1)
                            param_name = column_match.group(2)
                            
                            # Check if the parameter exists
                            if param_name in params:
                                updates[column_name] = params[param_name]
                    
                    # Add updated_at if it exists in the table
                    if "updated_at" in self.tables[table_name]["columns"] and "updated_at" not in updates:
                        updates["updated_at"] = datetime.now()
                    
                    # Extract WHERE clause from the query
                    where_match = re.search(r"where\s+(.*?)(?:order by|group by|limit|$)", query.lower())
                    if where_match and params:
                        where_clause = where_match.group(1).strip()
                        
                        # Extract conditions from the WHERE clause
                        conditions = []
                        for condition in where_clause.split("and"):
                            condition = condition.strip()
                            
                            # Extract column name and parameter name
                            column_match = re.search(r"(\w+)\s*=\s*:(\w+)", condition)
                            if column_match:
                                column_name = column_match.group(1)
                                param_name = column_match.group(2)
                                
                                # Check if the parameter exists
                                if param_name in params:
                                    conditions.append((column_name, params[param_name]))
                        
                        # Update rows based on conditions
                        updated_rows = 0
                        for row in self.tables[table_name]["rows"]:
                            match = True
                            for column_name, value in conditions:
                                if column_name in row and row[column_name] != value:
                                    match = False
                                    break
                            
                            if match:
                                row.update(updates)
                                updated_rows += 1
                        
                        return {
                            "type": "update",
                            "rowcount": updated_rows,
                            "rows": [],
                        }
                    else:
                        # Update all rows
                        for row in self.tables[table_name]["rows"]:
                            row.update(updates)
                        
                        return {
                            "type": "update",
                            "rowcount": len(self.tables[table_name]["rows"]),
                            "rows": [],
                        }
        
        # Default response for UPDATE queries
        return {
            "type": "update",
            "rowcount": 10,
            "rows": [],
        }
    
    def parse_delete_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse a DELETE query and return appropriate results.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        # Extract table name from the query
        table_match = re.search(r"delete\s+from\s+(\w+)", query.lower())
        if table_match:
            table_name = table_match.group(1)
            
            # Check if the table exists
            if table_name in self.tables:
                # Extract WHERE clause from the query
                where_match = re.search(r"where\s+(.*?)(?:order by|group by|limit|$)", query.lower())
                if where_match and params:
                    where_clause = where_match.group(1).strip()
                    
                    # Extract conditions from the WHERE clause
                    conditions = []
                    for condition in where_clause.split("and"):
                        condition = condition.strip()
                        
                        # Extract column name and parameter name
                        column_match = re.search(r"(\w+)\s*=\s*:(\w+)", condition)
                        if column_match:
                            column_name = column_match.group(1)
                            param_name = column_match.group(2)
                            
                            # Check if the parameter exists
                            if param_name in params:
                                conditions.append((column_name, params[param_name]))
                    
                    # Delete rows based on conditions
                    deleted_rows = 0
                    rows_to_keep = []
                    for row in self.tables[table_name]["rows"]:
                        match = True
                        for column_name, value in conditions:
                            if column_name in row and row[column_name] != value:
                                match = False
                                break
                        
                        if match:
                            deleted_rows += 1
                        else:
                            rows_to_keep.append(row)
                    
                    self.tables[table_name]["rows"] = rows_to_keep
                    
                    return {
                        "type": "delete",
                        "rowcount": deleted_rows,
                        "rows": [],
                    }
                else:
                    # Delete all rows
                    deleted_rows = len(self.tables[table_name]["rows"])
                    self.tables[table_name]["rows"] = []
                    
                    return {
                        "type": "delete",
                        "rowcount": deleted_rows,
                        "rows": [],
                    }
        
        # Default response for DELETE queries
        return {
            "type": "delete",
            "rowcount": 10,
            "rows": [],
        }


class ImprovedMockSession:
    """Improved mock session for database testing."""
    
    def __init__(self):
        """Initialize the mock session."""
        self.query_parser = MockQueryParser()
        self.committed = False
        self.rolled_back = False
        self.closed = False
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        Execute a query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        # Parse the query
        result = self.query_parser.parse_query(query, params)
        
        # Create a mock result
        mock_result = mock.MagicMock()
        
        # Configure the mock result based on the query type
        if result["type"] == "select":
            if result["count"]:
                mock_result.fetchall = mock.MagicMock(return_value=[{"count": result["rows"][0]["count"]}])
                mock_result.fetchone = mock.MagicMock(return_value={"count": result["rows"][0]["count"]})
                mock_result.scalar_one = mock.MagicMock(return_value=result["rows"][0]["count"])
                mock_result.scalar_one_or_none = mock.MagicMock(return_value=result["rows"][0]["count"])
            else:
                mock_result.fetchall = mock.MagicMock(return_value=result["rows"])
                mock_result.fetchone = mock.MagicMock(return_value=result["rows"][0] if result["rows"] else None)
                mock_result.scalar_one = mock.MagicMock(return_value=result["rows"][0]["id"] if result["rows"] else None)
                mock_result.scalar_one_or_none = mock.MagicMock(return_value=result["rows"][0]["id"] if result["rows"] else None)
        elif result["type"] == "insert":
            mock_result.fetchall = mock.MagicMock(return_value=result["rows"])
            mock_result.fetchone = mock.MagicMock(return_value=result["rows"][0] if result["rows"] else None)
            mock_result.scalar_one = mock.MagicMock(return_value=result["rows"][0]["id"] if result["rows"] else None)
            mock_result.scalar_one_or_none = mock.MagicMock(return_value=result["rows"][0]["id"] if result["rows"] else None)
        
        # Set the rowcount
        mock_result.rowcount = result["rowcount"]
        
        # Add additional result methods
        mock_result.scalars = mock.MagicMock(return_value=mock_result)
        mock_result.first = mock.MagicMock(return_value=result["rows"][0] if result["rows"] else None)
        mock_result.all = mock.MagicMock(return_value=result["rows"])
        
        return mock_result
    
    async def commit(self):
        """Commit the session."""
        self.committed = True
    
    async def rollback(self):
        """Roll back the session."""
        self.rolled_back = True
    
    async def close(self):
        """Close the session."""
        self.closed = True


# Test the improved mock session
async def test_improved_mock_session():
    """Test the improved mock session."""
    # Create a mock session
    session = ImprovedMockSession()
    
    # Test SELECT query
    result = await session.execute("SELECT * FROM users WHERE id = :id", {"id": 1})
    rows = result.fetchall()
    print(f"SELECT result: {rows}")
    
    # Test INSERT query
    result = await session.execute(
        "INSERT INTO users (username, email) VALUES (:username, :email)",
        {"username": "new_user", "email": "new_user@example.com"},
    )
    user_id = result.scalar_one()
    print(f"INSERT result: {user_id}")
    
    # Test UPDATE query
    result = await session.execute(
        "UPDATE users SET email = :email WHERE id = :id",
        {"id": 1, "email": "updated_email@example.com"},
    )
    print(f"UPDATE result: {result.rowcount} rows updated")
    
    # Test DELETE query
    result = await session.execute(
        "DELETE FROM users WHERE id = :id",
        {"id": 2},
    )
    print(f"DELETE result: {result.rowcount} rows deleted")
    
    # Test COUNT query
    result = await session.execute("SELECT COUNT(*) FROM users")
    count = result.scalar_one()
    print(f"COUNT result: {count}")
    
    # Commit the session
    await session.commit()
    print(f"Session committed: {session.committed}")
    
    # Roll back the session
    await session.rollback()
    print(f"Session rolled back: {session.rolled_back}")
    
    # Close the session
    await session.close()
    print(f"Session closed: {session.closed}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_improved_mock_session())