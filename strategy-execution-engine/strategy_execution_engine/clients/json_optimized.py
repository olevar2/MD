"""
JSON Optimization Module

This module provides optimized JSON parsing functions for the feature store client.
"""
import logging
from typing import Any, Dict, List, Optional, Union
try:
    import ujson as json
    has_ujson = True
except ImportError:
    import json
    has_ujson = False
try:
    import orjson
    has_orjson = True
except ImportError:
    has_orjson = False


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def parse_json_response(response) ->Any:
    """
    Parse a JSON response using the fastest available JSON parser.
    
    Args:
        response: aiohttp response object
        
    Returns:
        Parsed JSON data
    """
    try:
        if has_orjson:
            return orjson.loads(await response.read())
        elif has_ujson:
            return json.loads(await response.text())
        else:
            return await response.json()
    except Exception as e:
        logging.error(f'Error parsing JSON response: {str(e)}')
        return await response.json()


@with_exception_handling
def dumps(data: Any) ->str:
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
        logging.error(f'Error serializing JSON: {str(e)}')
        return json.dumps(data)


@with_exception_handling
def loads(data: str) ->Any:
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
        logging.error(f'Error parsing JSON: {str(e)}')
        return json.loads(data)
