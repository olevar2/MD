#!/usr/bin/env python
"""
Symbol Data Loader for Forex Trading Platform

This module loads symbol data into the forex trading platform.
It reads symbol data from CSV files and loads it into the database.

Functions:
    load_data: Load symbol data from CSV files
    validate_symbol_data: Validate symbol data before loading
    upload_symbols: Upload symbols to the data pipeline service
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("symbol_loader")

# API endpoints
DATA_PIPELINE_API = "http://localhost:8001/api/v1"


def validate_symbol_data(symbols: List[Dict[str, Any]]) -> bool:
    """
    Validate symbol data before loading.
    
    Args:
        symbols: List of symbol dictionaries
        
    Returns:
        True if valid, False otherwise
    """
    if not symbols:
        logger.error("No symbols to validate")
        return False
    
    # Check required fields
    required_fields = ["symbol", "description", "type", "pip_value", "lot_size"]
    
    for symbol in symbols:
        for field in required_fields:
            if field not in symbol:
                logger.error(f"Symbol {symbol.get('symbol', 'unknown')} missing required field: {field}")
                return False
        
        # Validate symbol format
        if not symbol["symbol"].isalpha():
            logger.error(f"Invalid symbol format: {symbol['symbol']}")
            return False
        
        # Validate numeric fields
        numeric_fields = ["pip_value", "lot_size", "min_lot", "max_lot"]
        for field in numeric_fields:
            if field in symbol and not isinstance(symbol[field], (int, float)):
                try:
                    symbol[field] = float(symbol[field])
                except (ValueError, TypeError):
                    logger.error(f"Symbol {symbol['symbol']} has invalid {field}: {symbol[field]}")
                    return False
    
    logger.info(f"Validated {len(symbols)} symbols")
    return True


def upload_symbols(symbols: List[Dict[str, Any]], verbose: bool = False) -> bool:
    """
    Upload symbols to the data pipeline service.
    
    Args:
        symbols: List of symbol dictionaries
        verbose: Enable verbose output
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Uploading {len(symbols)} symbols to data pipeline service...")
    
    # Upload each symbol
    success_count = 0
    for symbol in symbols:
        try:
            # Check if symbol already exists
            response = requests.get(
                f"{DATA_PIPELINE_API}/symbols/{symbol['symbol']}",
                timeout=10
            )
            
            if response.status_code == 200:
                # Symbol exists, update it
                if verbose:
                    logger.info(f"Updating symbol: {symbol['symbol']}")
                
                response = requests.put(
                    f"{DATA_PIPELINE_API}/symbols/{symbol['symbol']}",
                    json=symbol,
                    timeout=10
                )
            else:
                # Symbol doesn't exist, create it
                if verbose:
                    logger.info(f"Creating symbol: {symbol['symbol']}")
                
                response = requests.post(
                    f"{DATA_PIPELINE_API}/symbols",
                    json=symbol,
                    timeout=10
                )
            
            if response.status_code in (200, 201):
                success_count += 1
            else:
                logger.error(f"Failed to upload symbol {symbol['symbol']}: {response.status_code} {response.text}")
        except requests.RequestException as e:
            logger.error(f"Error uploading symbol {symbol['symbol']}: {e}")
    
    logger.info(f"Successfully uploaded {success_count} out of {len(symbols)} symbols")
    return success_count == len(symbols)


def load_data(data_dir: str, verify: bool = False, verbose: bool = False) -> bool:
    """
    Load symbol data from CSV files.
    
    Args:
        data_dir: Directory containing sample data files
        verify: Verify data after loading
        verbose: Enable verbose output
        
    Returns:
        True if successful, False otherwise
    """
    # Set log level
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if symbols.csv exists
    symbols_file = os.path.join(data_dir, "symbols.csv")
    if not os.path.isfile(symbols_file):
        logger.error(f"Symbols file not found: {symbols_file}")
        return False
    
    # Read symbols from CSV
    symbols = []
    try:
        with open(symbols_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert empty strings to None
                for key, value in row.items():
                    if value == "":
                        row[key] = None
                
                symbols.append(row)
    except Exception as e:
        logger.error(f"Error reading symbols file: {e}")
        return False
    
    logger.info(f"Read {len(symbols)} symbols from {symbols_file}")
    
    # Validate symbols
    if not validate_symbol_data(symbols):
        logger.error("Symbol validation failed")
        return False
    
    # Upload symbols
    if not upload_symbols(symbols, verbose):
        logger.error("Symbol upload failed")
        return False
    
    # Verify symbols
    if verify:
        logger.info("Verifying symbols...")
        
        try:
            response = requests.get(
                f"{DATA_PIPELINE_API}/symbols",
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to verify symbols: {response.status_code} {response.text}")
                return False
            
            uploaded_symbols = response.json()
            
            if len(uploaded_symbols) < len(symbols):
                logger.error(f"Verification failed: Expected {len(symbols)} symbols, found {len(uploaded_symbols)}")
                return False
            
            logger.info(f"Verified {len(uploaded_symbols)} symbols")
        except requests.RequestException as e:
            logger.error(f"Error verifying symbols: {e}")
            return False
    
    logger.info("Symbol data loaded successfully")
    return True


if __name__ == "__main__":
    # This allows the module to be run directly for testing
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
    verify = "--verify" in sys.argv
    verbose = "--verbose" in sys.argv
    
    if load_data(data_dir, verify, verbose):
        sys.exit(0)
    else:
        sys.exit(1)
