#!/usr/bin/env python
"""
Script to apply database optimization to a service.

This script applies the database utilities from common_lib.database to a service.
It updates the database connection code to use the standardized connection pool,
adds prepared statements to repository methods, and adds monitoring to database operations.

Usage:
    python apply_database_optimization.py --service <service_name>

Example:
    python apply_database_optimization.py --service market-analysis-service
"""
import os
import sys
import argparse
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Apply database optimization to a service')
    parser.add_argument('--service', required=True, help='Service name (e.g., market-analysis-service)')
    return parser.parse_args()


def find_repository_files(service_path):
    """
    Find repository files in a service.
    
    Args:
        service_path: Path to the service
        
    Returns:
        List of repository file paths
    """
    repository_files = []
    
    # Find all Python files in the repositories directory
    for root, _, files in os.walk(service_path):
        if 'repositories' in root and '__pycache__' not in root:
            for file in files:
                if file.endswith('.py'):
                    repository_files.append(os.path.join(root, file))
    
    return repository_files


def find_database_connection_files(service_path):
    """
    Find database connection files in a service.
    
    Args:
        service_path: Path to the service
        
    Returns:
        List of database connection file paths
    """
    connection_files = []
    
    # Find all Python files that might contain database connection code
    for root, _, files in os.walk(service_path):
        if '__pycache__' not in root:
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # Check if the file contains database connection code
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if re.search(r'(create_engine|create_async_engine|asyncpg\.create_pool)', content):
                            connection_files.append(file_path)
    
    return connection_files


def update_database_connection(file_path, service_name):
    """
    Update database connection code to use the standardized connection pool.
    
    Args:
        file_path: Path to the file
        service_name: Name of the service
        
    Returns:
        True if the file was updated, False otherwise
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file already uses the standardized connection pool
    if 'common_lib.database' in content and 'get_connection_pool' in content:
        logger.info(f"File {file_path} already uses the standardized connection pool")
        return False
    
    # Replace synchronous database connection code
    sync_pattern = r'(engine\s*=\s*create_engine\([^)]+\))'
    sync_replacement = f"""# Use standardized connection pool
from common_lib.database import get_connection_pool, get_sync_db_session

# Get connection pool for {service_name}
connection_pool = get_connection_pool("{service_name}")"""
    
    # Replace asynchronous database connection code
    async_pattern = r'(engine\s*=\s*create_async_engine\([^)]+\))'
    async_replacement = f"""# Use standardized connection pool
from common_lib.database import get_connection_pool, get_async_db_session

# Get connection pool for {service_name}
connection_pool = get_connection_pool("{service_name}")"""
    
    # Replace asyncpg connection code
    asyncpg_pattern = r'(pool\s*=\s*await\s*asyncpg\.create_pool\([^)]+\))'
    asyncpg_replacement = f"""# Use standardized connection pool
from common_lib.database import get_connection_pool, get_asyncpg_connection

# Get connection pool for {service_name}
connection_pool = get_connection_pool("{service_name}")"""
    
    # Apply replacements
    new_content = content
    new_content = re.sub(sync_pattern, sync_replacement, new_content)
    new_content = re.sub(async_pattern, async_replacement, new_content)
    new_content = re.sub(asyncpg_pattern, asyncpg_replacement, new_content)
    
    # Replace session creation code
    session_pattern = r'(with\s+sessionmaker\([^)]+\)\(\)\s+as\s+session:)'
    session_replacement = f"""with get_sync_db_session("{service_name}") as session:"""
    
    async_session_pattern = r'(async\s+with\s+AsyncSession\([^)]+\)\s+as\s+session:)'
    async_session_replacement = f"""async with get_async_db_session("{service_name}") as session:"""
    
    # Apply replacements
    new_content = re.sub(session_pattern, session_replacement, new_content)
    new_content = re.sub(async_session_pattern, async_session_replacement, new_content)
    
    # Write the updated content
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        logger.info(f"Updated database connection in {file_path}")
        return True
    
    return False


def update_repository_methods(file_path, service_name):
    """
    Update repository methods to use prepared statements and monitoring.
    
    Args:
        file_path: Path to the file
        service_name: Name of the service
        
    Returns:
        True if the file was updated, False otherwise
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file already uses prepared statements and monitoring
    if 'common_lib.database' in content and ('execute_prepared_statement' in content or 'track_query_performance' in content):
        logger.info(f"File {file_path} already uses prepared statements and monitoring")
        return False
    
    # Add imports
    imports_to_add = """
from common_lib.database import (
    execute_prepared_statement,
    execute_prepared_statement_async,
    execute_prepared_statement_asyncpg,
    fetch_prepared_statement_asyncpg,
    with_prepared_statement,
    async_with_prepared_statement,
    track_query_performance,
    async_track_query_performance,
    track_transaction,
    async_track_transaction,
    bulk_insert,
    bulk_insert_async,
    bulk_update,
    bulk_update_async,
    bulk_delete,
    bulk_delete_async,
)
"""
    
    # Add imports to the file
    import_pattern = r'(import\s+[^\n]+\n|from\s+[^\n]+\n)+'
    if re.search(import_pattern, content):
        # Add after the last import
        last_import = re.search(import_pattern, content).group(0)
        new_content = content.replace(last_import, last_import + imports_to_add)
    else:
        # Add at the beginning of the file
        new_content = imports_to_add + content
    
    # Find and update repository methods
    # This is a simplified approach - in a real implementation, you would need to parse the Python code
    # and update the methods more carefully
    
    # Update synchronous query methods
    query_pattern = r'(def\s+get_by_id\([^)]+\):)'
    query_replacement = r"""@track_query_performance("select", "entity", "{service_name}")
\1""".format(service_name=service_name)
    
    new_content = re.sub(query_pattern, query_replacement, new_content)
    
    # Update asynchronous query methods
    async_query_pattern = r'(async\s+def\s+get_by_id\([^)]+\):)'
    async_query_replacement = r"""@async_track_query_performance("select", "entity", "{service_name}")
\1""".format(service_name=service_name)
    
    new_content = re.sub(async_query_pattern, async_query_replacement, new_content)
    
    # Update session.execute calls to use prepared statements
    execute_pattern = r'(session\.execute\([^)]+\))'
    execute_replacement = r"""execute_prepared_statement(
            session,
            query,
            params,
            "{service_name}",
            "get_by_id"
        )""".format(service_name=service_name)
    
    new_content = re.sub(execute_pattern, execute_replacement, new_content)
    
    # Update async session.execute calls to use prepared statements
    async_execute_pattern = r'(await\s+session\.execute\([^)]+\))'
    async_execute_replacement = r"""await execute_prepared_statement_async(
            session,
            query,
            params,
            "{service_name}",
            "get_by_id"
        )""".format(service_name=service_name)
    
    new_content = re.sub(async_execute_pattern, async_execute_replacement, new_content)
    
    # Write the updated content
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        logger.info(f"Updated repository methods in {file_path}")
        return True
    
    return False


def add_bulk_operations(file_path, service_name):
    """
    Add bulk operations to repository methods.
    
    Args:
        file_path: Path to the file
        service_name: Name of the service
        
    Returns:
        True if the file was updated, False otherwise
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file already uses bulk operations
    if 'bulk_insert' in content or 'bulk_update' in content or 'bulk_delete' in content:
        logger.info(f"File {file_path} already uses bulk operations")
        return False
    
    # Find and update repository methods that could benefit from bulk operations
    # This is a simplified approach - in a real implementation, you would need to parse the Python code
    # and update the methods more carefully
    
    # Look for methods that insert, update, or delete multiple entities
    bulk_insert_pattern = r'(def\s+add_many\([^)]+\):)'
    bulk_insert_replacement = r"""@track_query_performance("insert", "entity", "{service_name}")
\1""".format(service_name=service_name)
    
    new_content = re.sub(bulk_insert_pattern, bulk_insert_replacement, new_content)
    
    # Look for loops that insert, update, or delete entities
    loop_insert_pattern = r'(for\s+entity\s+in\s+entities:\s+[^\n]+\.add\([^)]+\))'
    loop_insert_replacement = r"""# Use bulk insert instead of loop
        bulk_insert(
            session,
            entity_table,
            [entity.dict() for entity in entities],
            "{service_name}",
        )""".format(service_name=service_name)
    
    new_content = re.sub(loop_insert_pattern, loop_insert_replacement, new_content)
    
    # Write the updated content
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        logger.info(f"Added bulk operations to {file_path}")
        return True
    
    return False


def main():
    """Main function."""
    args = parse_args()
    service_name = args.service
    
    # Get the service path
    service_path = os.path.join(os.getcwd(), service_name)
    if not os.path.exists(service_path):
        logger.error(f"Service path {service_path} does not exist")
        sys.exit(1)
    
    logger.info(f"Applying database optimization to {service_name}")
    
    # Find database connection files
    connection_files = find_database_connection_files(service_path)
    logger.info(f"Found {len(connection_files)} database connection files")
    
    # Update database connection files
    for file_path in connection_files:
        update_database_connection(file_path, service_name)
    
    # Find repository files
    repository_files = find_repository_files(service_path)
    logger.info(f"Found {len(repository_files)} repository files")
    
    # Update repository files
    for file_path in repository_files:
        update_repository_methods(file_path, service_name)
        add_bulk_operations(file_path, service_name)
    
    logger.info(f"Database optimization applied to {service_name}")


if __name__ == "__main__":
    main()