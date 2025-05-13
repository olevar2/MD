#!/usr/bin/env python
"""
Run the Historical Data Management service.

This script runs the service with the specified configuration.
"""

import argparse
import logging
import os
import sys

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Historical Data Management service")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (default: False)"
    )
    
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/forex_platform"
        ),
        help="Database connection URL"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set environment variables
    os.environ["DATABASE_URL"] = args.db_url
    
    logger.info(f"Starting service on {args.host}:{args.port}")
    logger.info(f"Database URL: {args.db_url}")
    
    # Run the service
    uvicorn.run(
        "data_management_service.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
