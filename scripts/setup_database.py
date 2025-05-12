#!/usr/bin/env python
"""
Database Setup Script for Forex Trading Platform

This script initializes all required databases for the forex trading platform.
It handles PostgreSQL with TimescaleDB extension, creates necessary databases,
users, schemas, and tables, and configures permissions.

Usage:
    python setup_database.py [--host HOST] [--port PORT] [--admin-user ADMIN_USER] 
                            [--admin-password ADMIN_PASSWORD] [--skip-install]

Options:
    --host HOST                PostgreSQL host (default: localhost)
    --port PORT                PostgreSQL port (default: 5432)
    --admin-user ADMIN_USER    PostgreSQL admin username (default: postgres)
    --admin-password ADMIN_PW  PostgreSQL admin password (default: postgres)
    --skip-install             Skip PostgreSQL and TimescaleDB installation
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
import platform

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("setup_database")

# Database configuration
DB_CONFIG = {
    "forex_platform": {
        "user": "forex_user",
        "password": "forex_password",
        "schemas": ["market_data", "historical_data", "indicators"],
    },
    "data_pipeline": {
        "user": "pipeline_user",
        "password": "pipeline_password",
        "schemas": ["pipeline", "staging", "processed"],
    },
    "feature_store": {
        "user": "feature_user",
        "password": "feature_password",
        "schemas": ["features", "metadata", "registry"],
    },
    "portfolio": {
        "user": "portfolio_user",
        "password": "portfolio_password",
        "schemas": ["portfolio", "accounts", "performance"],
    },
    "analysis_engine": {
        "user": "analysis_user",
        "password": "analysis_password",
        "schemas": ["analysis", "models", "results"],
    },
    "ml_models": {
        "user": "ml_user",
        "password": "ml_password",
        "schemas": ["models", "training", "evaluation"],
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup databases for Forex Trading Platform")
    parser.add_argument("--host", type=str, default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--admin-user", type=str, default="postgres", help="PostgreSQL admin username")
    parser.add_argument("--admin-password", type=str, default="postgres", help="PostgreSQL admin password")
    parser.add_argument("--skip-install", action="store_true", help="Skip PostgreSQL and TimescaleDB installation")
    return parser.parse_args()


def check_postgres_installed():
    """Check if PostgreSQL is installed."""
    try:
        result = subprocess.run(
            ["psql", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"PostgreSQL is installed: {result.stdout.strip()}")
            return True
        else:
            logger.warning("PostgreSQL is not installed or not in PATH")
            return False
    except FileNotFoundError:
        logger.warning("PostgreSQL is not installed or not in PATH")
        return False


def install_postgres_and_timescaledb():
    """Install PostgreSQL and TimescaleDB."""
    system = platform.system().lower()
    
    if system == "linux":
        logger.info("Installing PostgreSQL and TimescaleDB on Linux...")
        try:
            # Add TimescaleDB repository
            subprocess.run([
                "sudo", "sh", "-c", 
                "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
            ], check=True)
            
            # Add TimescaleDB GPG key
            subprocess.run([
                "wget", "--quiet", "-O", "-", 
                "https://packagecloud.io/timescale/timescaledb/gpgkey", 
                "|", "sudo", "apt-key", "add", "-"
            ], check=True)
            
            # Update package lists
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            
            # Install PostgreSQL and TimescaleDB
            subprocess.run([
                "sudo", "apt-get", "install", "-y", 
                "postgresql-14", "timescaledb-2-postgresql-14"
            ], check=True)
            
            # Enable TimescaleDB
            subprocess.run([
                "sudo", "sh", "-c", 
                "echo \"shared_preload_libraries = 'timescaledb'\" >> /etc/postgresql/14/main/postgresql.conf"
            ], check=True)
            
            # Restart PostgreSQL
            subprocess.run(["sudo", "systemctl", "restart", "postgresql"], check=True)
            
            logger.info("PostgreSQL and TimescaleDB installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install PostgreSQL and TimescaleDB: {e}")
            return False
    
    elif system == "windows":
        logger.info("For Windows, please install PostgreSQL and TimescaleDB manually:")
        logger.info("1. Download and install PostgreSQL from https://www.postgresql.org/download/windows/")
        logger.info("2. Download and install TimescaleDB from https://docs.timescale.com/install/latest/self-hosted/installation-windows/")
        return False
    
    elif system == "darwin":  # macOS
        logger.info("Installing PostgreSQL and TimescaleDB on macOS...")
        try:
            # Install PostgreSQL and TimescaleDB using Homebrew
            subprocess.run(["brew", "install", "postgresql@14"], check=True)
            subprocess.run(["brew", "install", "timescaledb"], check=True)
            
            # Enable TimescaleDB
            subprocess.run([
                "sh", "-c", 
                "echo \"shared_preload_libraries = 'timescaledb'\" >> $(brew --prefix)/var/postgres/postgresql.conf"
            ], check=True)
            
            # Restart PostgreSQL
            subprocess.run(["brew", "services", "restart", "postgresql@14"], check=True)
            
            logger.info("PostgreSQL and TimescaleDB installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install PostgreSQL and TimescaleDB: {e}")
            return False
    
    else:
        logger.error(f"Unsupported operating system: {system}")
        return False


def connect_to_postgres(host, port, user, password, database="postgres"):
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        logger.info(f"Connected to PostgreSQL database: {database}")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to PostgreSQL database {database}: {e}")
        return None


def check_timescaledb_extension(conn):
    """Check if TimescaleDB extension is available."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'")
            count = cur.fetchone()[0]
            if count > 0:
                logger.info("TimescaleDB extension is already installed")
                return True
            else:
                logger.info("TimescaleDB extension is not installed")
                return False
    except psycopg2.Error as e:
        logger.error(f"Failed to check TimescaleDB extension: {e}")
        return False


def create_timescaledb_extension(conn):
    """Create TimescaleDB extension."""
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            logger.info("TimescaleDB extension created successfully")
            return True
    except psycopg2.Error as e:
        logger.error(f"Failed to create TimescaleDB extension: {e}")
        return False


def create_database(conn, db_name, owner=None):
    """Create a database if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cur.fetchone()
            
            if not exists:
                # Create database
                if owner:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {} WITH OWNER = {}").format(
                            sql.Identifier(db_name),
                            sql.Identifier(owner)
                        )
                    )
                else:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(db_name)
                        )
                    )
                logger.info(f"Database '{db_name}' created successfully")
            else:
                logger.info(f"Database '{db_name}' already exists")
            
            return True
    except psycopg2.Error as e:
        logger.error(f"Failed to create database '{db_name}': {e}")
        return False


def create_user(conn, username, password):
    """Create a user if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            # Check if user exists
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (username,))
            exists = cur.fetchone()
            
            if not exists:
                # Create user
                cur.execute(
                    sql.SQL("CREATE USER {} WITH PASSWORD {}").format(
                        sql.Identifier(username),
                        sql.Literal(password)
                    )
                )
                logger.info(f"User '{username}' created successfully")
            else:
                # Update password
                cur.execute(
                    sql.SQL("ALTER USER {} WITH PASSWORD {}").format(
                        sql.Identifier(username),
                        sql.Literal(password)
                    )
                )
                logger.info(f"User '{username}' already exists, password updated")
            
            return True
    except psycopg2.Error as e:
        logger.error(f"Failed to create/update user '{username}': {e}")
        return False


def create_schema(conn, schema_name):
    """Create a schema if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(schema_name)
                )
            )
            logger.info(f"Schema '{schema_name}' created successfully")
            return True
    except psycopg2.Error as e:
        logger.error(f"Failed to create schema '{schema_name}': {e}")
        return False


def grant_privileges(conn, db_name, schema_name, username):
    """Grant privileges to a user on a schema."""
    try:
        with conn.cursor() as cur:
            # Grant usage on schema
            cur.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(username)
                )
            )
            
            # Grant all privileges on all tables in schema
            cur.execute(
                sql.SQL("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA {} TO {}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(username)
                )
            )
            
            # Grant all privileges on all sequences in schema
            cur.execute(
                sql.SQL("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {} TO {}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(username)
                )
            )
            
            # Grant all privileges on schema
            cur.execute(
                sql.SQL("GRANT ALL PRIVILEGES ON SCHEMA {} TO {}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(username)
                )
            )
            
            logger.info(f"Privileges granted to '{username}' on schema '{schema_name}'")
            return True
    except psycopg2.Error as e:
        logger.error(f"Failed to grant privileges to '{username}' on schema '{schema_name}': {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Check if PostgreSQL is installed
    if not args.skip_install and not check_postgres_installed():
        logger.info("Installing PostgreSQL and TimescaleDB...")
        if not install_postgres_and_timescaledb():
            logger.error("Failed to install PostgreSQL and TimescaleDB. Please install manually.")
            return 1
    
    # Connect to PostgreSQL
    conn = connect_to_postgres(args.host, args.port, args.admin_user, args.admin_password)
    if not conn:
        logger.error("Failed to connect to PostgreSQL. Please check your credentials and try again.")
        return 1
    
    # Check and create TimescaleDB extension
    if not check_timescaledb_extension(conn):
        if not create_timescaledb_extension(conn):
            logger.error("Failed to create TimescaleDB extension. Please install TimescaleDB and try again.")
            conn.close()
            return 1
    
    # Create users, databases, and schemas
    for db_name, config in DB_CONFIG.items():
        # Create user
        if not create_user(conn, config["user"], config["password"]):
            logger.error(f"Failed to create user for database '{db_name}'")
            continue
        
        # Create database
        if not create_database(conn, db_name, config["user"]):
            logger.error(f"Failed to create database '{db_name}'")
            continue
        
        # Connect to the new database
        db_conn = connect_to_postgres(args.host, args.port, args.admin_user, args.admin_password, db_name)
        if not db_conn:
            logger.error(f"Failed to connect to database '{db_name}'")
            continue
        
        # Create schemas
        for schema in config["schemas"]:
            if not create_schema(db_conn, schema):
                logger.error(f"Failed to create schema '{schema}' in database '{db_name}'")
                continue
            
            # Grant privileges
            if not grant_privileges(db_conn, db_name, schema, config["user"]):
                logger.error(f"Failed to grant privileges on schema '{schema}' to user '{config['user']}'")
                continue
        
        # Close database connection
        db_conn.close()
    
    # Close main connection
    conn.close()
    
    logger.info("Database setup completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
