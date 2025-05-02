"""
JSON Optimization Module

This module provides optimized JSON parsing functions for the feature store client.
"""
import logging
from typing import Any, Dict, List, Optional, Union

# Try to import ujson for faster JSON parsing
try:
    import ujson as json
    has_ujson = True
except ImportError:
    import json
    has_ujson = False

# Try to import orjson for even faster JSON parsing
try:
    import orjson
    has_orjson = True
except ImportError:
    has_orjson = False


async def parse_json_response(response) -> Any:
    """
    Parse a JSON response using the fastest available JSON parser.
    
    Args:
        response: aiohttp response object
        
    Returns:
        Parsed JSON data
    """
    try:
        # Use the fastest available JSON parser
        if has_orjson:
            return orjson.loads(await response.read())
        elif has_ujson:
            return json.loads(await response.text())
        else:
            return await response.json()
    except Exception as e:
        logging.error(f"Error parsing JSON response: {str(e)}")
        # Fall back to standard json parser
        return await response.json()


def dumps(data: Any) -> str:
    """
    Serialize data to JSON string using the fastest available JSON serializer.
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON string
    """
    try:
        if has_orjson:
            return orjson.dumps(data).decode('utf-8')
        elif has_ujson:
            return json.dumps(data)
        else:
            return json.dumps(data)
    except Exception as e:
        logging.error(f"Error serializing JSON: {str(e)}")
        # Fall back to standard json serializer
        return json.dumps(data)


def loads(data: str) -> Any:
    """
    Parse a JSON string using the fastest available JSON parser.
    
    Args:
        data: JSON string to parse
        
    Returns:
        Parsed JSON data
    """
    try:
        if has_orjson:
            return orjson.loads(data)
        elif has_ujson:
            return json.loads(data)
        else:
            return json.loads(data)
    except Exception as e:
        logging.error(f"Error parsing JSON: {str(e)}")
        # Fall back to standard json parser
        return json.loads(data)
