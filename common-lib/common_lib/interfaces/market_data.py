
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime

class IMarketDataProvider(ABC):
    """Interface for market data providers"""

    @abstractmethod
    async def get_historical_data(self,
                                 symbol: str,
                                 timeframe: str,
                                 start_time: datetime,
                                 end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical market data for a symbol"""
        pass

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Dict[str, float]:
        """Get the latest price for a symbol"""
        pass

    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """Get available symbols"""
        pass

class IMarketDataCache(ABC):
    """Interface for market data caching"""

    @abstractmethod
    async def get_cached_data(self,
                             symbol: str,
                             timeframe: str,
                             start_time: datetime,
                             end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Get cached market data if available"""
        pass

    @abstractmethod
    async def cache_data(self,
                        symbol: str,
                        timeframe: str,
                        data: pd.DataFrame) -> bool:
        """Cache market data"""
        pass

    @abstractmethod
    async def invalidate_cache(self,
                              symbol: Optional[str] = None,
                              timeframe: Optional[str] = None) -> bool:
        """Invalidate cache entries"""
        pass
