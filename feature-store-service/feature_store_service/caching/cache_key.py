"""
Cache key implementation for feature store caching system.
"""
import json
from datetime import datetime
from typing import Dict, Any


class CacheKey:
    """
    Class representing a unique key for caching indicator results.
    Keys are composed of the indicator type, parameters, symbol, timeframe,
    and time range.
    """
    def __init__(
        self,
        indicator_type: str,
        params: Dict[str, Any],
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ):
        self.indicator_type = indicator_type
        self.params = params
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        
    def to_string(self) -> str:
        """Convert key to string format for storage."""
        params_str = json.dumps(self.params, sort_keys=True)
        return f"{self.indicator_type}:{params_str}:{self.symbol}:{self.timeframe}:{self.start_time.isoformat()}:{self.end_time.isoformat()}"
        
    @classmethod
    def from_string(cls, key_str: str) -> 'CacheKey':
        """Parse key from string format."""
        parts = key_str.split(':')
        indicator_type = parts[0]
        params = json.loads(parts[1])
        symbol = parts[2]
        timeframe = parts[3]
        start_time = datetime.fromisoformat(parts[4])
        end_time = datetime.fromisoformat(parts[5])
        
        return cls(
            indicator_type,
            params,
            symbol,
            timeframe,
            start_time,
            end_time
        )

    def __eq__(self, other):
        if not isinstance(other, CacheKey):
            return False
        return self.to_string() == other.to_string()
    
    def __hash__(self):
        return hash(self.to_string())
