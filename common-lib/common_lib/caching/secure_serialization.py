#!/usr/bin/env python3
"""
Secure serialization module for replacing pickle with JSON.

This module provides secure serialization and deserialization functions
that use JSON instead of pickle, with special handling for common types
like pandas DataFrames and numpy arrays.
"""

import json
import base64
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Union

class SecureSerializer:
    """
    Secure serializer that uses JSON instead of pickle.
    
    This class provides methods for serializing and deserializing objects
    using JSON instead of pickle, with special handling for common types
    like pandas DataFrames and numpy arrays.
    """
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """
        Serialize an object to JSON.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized object as bytes
        """
        # Convert object to JSON-serializable format
        json_obj = SecureSerializer._convert_to_json(obj)
        
        # Serialize to JSON and encode as bytes
        return json.dumps(json_obj).encode('utf-8')
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        Deserialize an object from JSON.
        
        Args:
            data: Serialized object as bytes
            
        Returns:
            Deserialized object
        """
        # Decode bytes to string and parse JSON
        json_obj = json.loads(data.decode('utf-8'))
        
        # Convert from JSON-serializable format
        return SecureSerializer._convert_from_json(json_obj)
    
    @staticmethod
    def _convert_to_json(obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        # Handle None
        if obj is None:
            return None
        
        # Handle basic types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return {
                "__type__": "list" if isinstance(obj, list) else "tuple",
                "data": [SecureSerializer._convert_to_json(item) for item in obj]
            }
        
        # Handle dictionaries
        if isinstance(obj, dict):
            # Convert keys to strings (JSON requires string keys)
            return {
                "__type__": "dict",
                "data": {
                    str(k): SecureSerializer._convert_to_json(v) for k, v in obj.items()
                }
            }
        
        # Handle pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "pandas.DataFrame",
                "data": obj.to_json(orient="split")
            }
        
        # Handle pandas Series
        if isinstance(obj, pd.Series):
            return {
                "__type__": "pandas.Series",
                "data": obj.to_json(orient="split")
            }
        
        # Handle numpy array
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "numpy.ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": base64.b64encode(obj.tobytes()).decode('ascii')
            }
        
        # Handle numpy scalar types
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return {
                "__type__": "numpy.scalar",
                "dtype": str(obj.dtype),
                "data": obj.item()
            }
        
        # Handle datetime
        if hasattr(obj, 'isoformat'):
            return {
                "__type__": "datetime",
                "data": obj.isoformat()
            }
        
        # Handle sets
        if isinstance(obj, set):
            return {
                "__type__": "set",
                "data": list(obj)
            }
        
        # Handle custom objects with __dict__
        if hasattr(obj, '__dict__'):
            return {
                "__type__": "object",
                "class": obj.__class__.__name__,
                "module": obj.__class__.__module__,
                "data": SecureSerializer._convert_to_json(obj.__dict__)
            }
        
        # Fallback: convert to string
        return {
            "__type__": "string",
            "data": str(obj)
        }
    
    @staticmethod
    def _convert_from_json(obj: Any) -> Any:
        """
        Convert an object from a JSON-serializable format.
        
        Args:
            obj: JSON-serializable object
            
        Returns:
            Deserialized object
        """
        # Handle None
        if obj is None:
            return None
        
        # Handle basic types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle special types
        if isinstance(obj, dict) and "__type__" in obj:
            obj_type = obj["__type__"]
            
            # Handle lists and tuples
            if obj_type == "list":
                return [SecureSerializer._convert_from_json(item) for item in obj["data"]]
            elif obj_type == "tuple":
                return tuple(SecureSerializer._convert_from_json(item) for item in obj["data"])
            
            # Handle dictionaries
            elif obj_type == "dict":
                return {k: SecureSerializer._convert_from_json(v) for k, v in obj["data"].items()}
            
            # Handle pandas DataFrame
            elif obj_type == "pandas.DataFrame":
                return pd.read_json(obj["data"], orient="split")
            
            # Handle pandas Series
            elif obj_type == "pandas.Series":
                return pd.read_json(obj["data"], orient="split", typ="series")
            
            # Handle numpy array
            elif obj_type == "numpy.ndarray":
                data = base64.b64decode(obj["data"])
                return np.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"])
            
            # Handle numpy scalar types
            elif obj_type == "numpy.scalar":
                return np.array([obj["data"]], dtype=obj["dtype"])[0]
            
            # Handle datetime
            elif obj_type == "datetime":
                from datetime import datetime
                return datetime.fromisoformat(obj["data"])
            
            # Handle sets
            elif obj_type == "set":
                return set(obj["data"])
            
            # Handle custom objects with __dict__
            elif obj_type == "object":
                # Note: This is a simplified implementation that doesn't handle
                # all possible custom objects. In a real-world scenario, you might
                # want to use a whitelist of allowed classes.
                return {
                    "class": obj["class"],
                    "module": obj["module"],
                    "data": SecureSerializer._convert_from_json(obj["data"])
                }
            
            # Handle string fallback
            elif obj_type == "string":
                return obj["data"]
        
        # Handle regular dictionaries
        if isinstance(obj, dict):
            return {k: SecureSerializer._convert_from_json(v) for k, v in obj.items()}
        
        # Handle lists
        if isinstance(obj, list):
            return [SecureSerializer._convert_from_json(item) for item in obj]
        
        return obj