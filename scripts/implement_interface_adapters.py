#!/usr/bin/env python3
"""
Interface-Based Adapter Pattern Implementation Script

This script implements the interface-based adapter pattern to resolve circular dependencies
between services by creating standardized interfaces in the common-lib module.
"""

import os
import sys
from pathlib import Path
import shutil

# Constants
COMMON_LIB_PATH = "common-lib/common_lib"
INTERFACES_DIR = f"{COMMON_LIB_PATH}/interfaces"
ADAPTERS_DIR = f"{COMMON_LIB_PATH}/adapters"

# Interface definitions
MARKET_DATA_INTERFACE = '''
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
'''

FEATURE_STORE_INTERFACE = '''
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
from datetime import datetime

class IFeatureProvider(ABC):
    """Interface for feature providers"""

    @abstractmethod
    async def get_feature(self,
                         feature_name: str,
                         symbol: str,
                         timeframe: str,
                         start_time: datetime,
                         end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get a specific feature for a symbol"""
        pass

    @abstractmethod
    async def get_features(self,
                          feature_names: List[str],
                          symbol: str,
                          timeframe: str,
                          start_time: datetime,
                          end_time: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Get multiple features for a symbol"""
        pass

    @abstractmethod
    async def list_features(self) -> List[Dict[str, Any]]:
        """List available features"""
        pass

class IFeatureStore(ABC):
    """Interface for feature storage"""

    @abstractmethod
    async def store_feature(self,
                           feature_name: str,
                           symbol: str,
                           timeframe: str,
                           data: pd.DataFrame) -> bool:
        """Store a feature in the feature store"""
        pass

    @abstractmethod
    async def get_feature(self,
                         feature_name: str,
                         symbol: str,
                         timeframe: str,
                         start_time: datetime,
                         end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Retrieve a feature from the feature store"""
        pass

    @abstractmethod
    async def delete_feature(self,
                            feature_name: str,
                            symbol: Optional[str] = None,
                            timeframe: Optional[str] = None) -> bool:
        """Delete a feature from the feature store"""
        pass

class IFeatureGenerator(ABC):
    """Interface for feature generators"""

    @abstractmethod
    async def generate_feature(self,
                              feature_name: str,
                              symbol: str,
                              timeframe: str,
                              start_time: datetime,
                              end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Generate a feature for a symbol"""
        pass

    @abstractmethod
    async def register_generator(self,
                                feature_name: str,
                                generator_func: Callable) -> bool:
        """Register a new feature generator function"""
        pass

    @abstractmethod
    async def list_generators(self) -> List[str]:
        """List available feature generators"""
        pass
'''

ANALYSIS_ENGINE_INTERFACE = '''
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime

class IAnalysisProvider(ABC):
    """Interface for analysis providers"""

    @abstractmethod
    async def analyze_market(self,
                            symbol: str,
                            timeframe: str,
                            analysis_type: str,
                            start_time: datetime,
                            end_time: Optional[datetime] = None,
                            parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform market analysis"""
        pass

    @abstractmethod
    async def get_analysis_types(self) -> List[Dict[str, Any]]:
        """Get available analysis types"""
        pass

    @abstractmethod
    async def backtest_strategy(self,
                               strategy_id: str,
                               symbol: str,
                               timeframe: str,
                               start_time: datetime,
                               end_time: datetime,
                               parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backtest a trading strategy"""
        pass

class IIndicatorProvider(ABC):
    """Interface for indicator providers"""

    @abstractmethod
    async def calculate_indicator(self,
                                 indicator_name: str,
                                 data: pd.DataFrame,
                                 parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Calculate a technical indicator"""
        pass

    @abstractmethod
    async def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        """Get information about an indicator"""
        pass

    @abstractmethod
    async def list_indicators(self) -> List[str]:
        """List available indicators"""
        pass

class IPatternRecognizer(ABC):
    """Interface for pattern recognition"""

    @abstractmethod
    async def recognize_patterns(self,
                                data: pd.DataFrame,
                                pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Recognize patterns in market data"""
        pass

    @abstractmethod
    async def get_pattern_types(self) -> List[Dict[str, Any]]:
        """Get available pattern types"""
        pass
'''

TRADING_INTERFACE = '''
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    """
    OrderType class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """
    OrderSide class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

class ITradingProvider(ABC):
    """Interface for trading providers"""

    @abstractmethod
    async def place_order(self,
                         symbol: str,
                         order_type: OrderType,
                         side: OrderSide,
                         quantity: float,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: Optional[str] = "GTC",
                         client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Place a trading order"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details"""
        pass

    @abstractmethod
    async def get_orders(self,
                        symbol: Optional[str] = None,
                        status: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """Get orders"""
        pass

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass

class IOrderBookProvider(ABC):
    """Interface for order book providers"""

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: Optional[int] = 10) -> Dict[str, Any]:
        """Get order book for a symbol"""
        pass

    @abstractmethod
    async def subscribe_order_book(self, symbol: str, callback: callable) -> Any:
        """Subscribe to order book updates"""
        pass

    @abstractmethod
    async def unsubscribe_order_book(self, subscription_id: Any) -> bool:
        """Unsubscribe from order book updates"""
        pass

class IRiskManager(ABC):
    """Interface for risk management"""

    @abstractmethod
    async def check_risk_limits(self,
                               symbol: str,
                               order_type: OrderType,
                               side: OrderSide,
                               quantity: float,
                               price: Optional[float] = None) -> Dict[str, Any]:
        """Check if an order complies with risk limits"""
        pass

    @abstractmethod
    async def get_position_risk(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get risk metrics for current positions"""
        pass

    @abstractmethod
    async def set_risk_limit(self,
                            limit_type: str,
                            symbol: Optional[str] = None,
                            value: Optional[float] = None) -> bool:
        """Set a risk limit"""
        pass
'''

def create_directory(path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content):
    """Create a file with the given content"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created file: {path}")

def create_init_file(path):
    """Create an __init__.py file"""
    init_path = os.path.join(path, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write('"""Interface and adapter definitions"""\n')
        print(f"Created file: {init_path}")

def implement_interfaces():
    """Implement the interface-based adapter pattern"""
    # Create directories
    create_directory(INTERFACES_DIR)
    create_directory(ADAPTERS_DIR)

    # Create __init__.py files
    create_init_file(INTERFACES_DIR)
    create_init_file(ADAPTERS_DIR)

    # Create interface files
    create_file(os.path.join(INTERFACES_DIR, "market_data.py"), MARKET_DATA_INTERFACE)
    create_file(os.path.join(INTERFACES_DIR, "feature_store.py"), FEATURE_STORE_INTERFACE)
    create_file(os.path.join(INTERFACES_DIR, "analysis_engine.py"), ANALYSIS_ENGINE_INTERFACE)
    create_file(os.path.join(INTERFACES_DIR, "trading.py"), TRADING_INTERFACE)

    # Update the interfaces __init__.py to export all interfaces
    interfaces_init_content = '''
"""Interface definitions for service integration"""

# Market Data interfaces
from .market_data import IMarketDataProvider, IMarketDataCache

# Feature Store interfaces
from .feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator

# Analysis Engine interfaces
from .analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer

# Trading interfaces
from .trading import ITradingProvider, IOrderBookProvider, IRiskManager, OrderType, OrderSide, OrderStatus

__all__ = [
    'IMarketDataProvider',
    'IMarketDataCache',
    'IFeatureProvider',
    'IFeatureStore',
    'IFeatureGenerator',
    'IAnalysisProvider',
    'IIndicatorProvider',
    'IPatternRecognizer',
    'ITradingProvider',
    'IOrderBookProvider',
    'IRiskManager',
    'OrderType',
    'OrderSide',
    'OrderStatus'
]
'''

    create_file(os.path.join(INTERFACES_DIR, "__init__.py"), interfaces_init_content)

    print("Interface-based adapter pattern implemented successfully")

def main():
    """
    Main.
    
    """

    implement_interfaces()

if __name__ == "__main__":
    main()
