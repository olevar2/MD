"""
Feature Store Client Module

This module provides a client to interact with the feature store service,
for retrieving OHLCV data, technical indicators, and other features.
"""
import pandas as pd
import requests
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import os
import json

class FeatureStoreClient:
    """
    A client for interacting with the feature store service.
    
    This class provides methods to retrieve OHLCV data, technical indicators,
    and other features from the feature store service.
    
    Attributes:
        base_url: Base URL of the feature store service
        headers: HTTP headers for API requests
        logger: Logger instance
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the FeatureStoreClient with connection settings.
        
        Args:
            base_url: Base URL of the feature store service API.
                     If None, uses FEATURE_STORE_URL environment variable or default.
            api_key: API key for authentication.
                    If None, uses FEATURE_STORE_API_KEY environment variable.
        """
        self.base_url = base_url or os.getenv('FEATURE_STORE_URL', 'http://localhost:8001/api/v1')
        api_key = api_key or os.getenv('FEATURE_STORE_API_KEY')
        
        # Set up headers for API requests
        self.headers = {
            'Content-Type': 'application/json'
        }
        
        # Add API key to headers if available
        if api_key:
            self.headers['X-API-Key'] = api_key
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FeatureStoreClient initialized with base URL: {self.base_url}")
    
    def _format_date(self, date: Union[str, datetime]) -> str:
        """
        Format a date for use in API requests.
        
        Args:
            date: Date to format, either as string or datetime object
            
        Returns:
            str: Formatted date string in ISO format
        """
        if isinstance(date, str):
            return date
        elif isinstance(date, datetime):
            return date.isoformat()
        else:
            raise ValueError(f"Unsupported date format: {type(date)}")
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and raise exceptions for errors.
        
        Args:
            response: Response from the API
            
        Returns:
            Dict[str, Any]: JSON response data
            
        Raises:
            HTTPError: If the response contains an error
        """
        try:
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            error_msg = f"HTTP error occurred: {e}"
            try:
                error_detail = response.json().get('detail', 'No detail available')
                error_msg += f" - {error_detail}"
            except:
                pass
            
            self.logger.error(error_msg)
            raise
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON response from API: {response.text}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def get_ohlcv_data(
        self, 
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data from the feature store.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR_USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data
        """
        try:
            # Format dates
            start_date_str = self._format_date(start_date)
            end_date_str = self._format_date(end_date)
            
            # Build URL and parameters
            url = f"{self.base_url}/data/ohlcv"
            params = {
                "symbol": symbol,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "timeframe": timeframe
            }
            
            self.logger.info(f"Requesting OHLCV data for {symbol} from {start_date_str} to {end_date_str} ({timeframe})")
            
            # Make the request
            response = requests.get(url, params=params, headers=self.headers)
            data = self._handle_response(response)
            
            # Convert to DataFrame
            if 'data' in data and data['data']:
                df = pd.DataFrame(data['data'])
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                self.logger.info(f"Retrieved {len(df)} OHLCV records")
                return df
            else:
                self.logger.warning(f"No OHLCV data returned for {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error retrieving OHLCV data: {str(e)}")
            # For development, return sample data instead of failing
            if os.getenv('ENVIRONMENT', 'development') == 'development':
                self.logger.warning("Returning sample data for development")
                return self._get_sample_ohlcv_data(symbol, start_date, end_date, timeframe)
            else:
                raise
    
    def get_indicators(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1h',
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve technical indicators from the feature store.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR_USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data
            indicators: List of indicator names to retrieve
            
        Returns:
            pd.DataFrame: DataFrame containing indicator values
        """
        try:
            # Format dates
            start_date_str = self._format_date(start_date)
            end_date_str = self._format_date(end_date)
            
            # Build URL and parameters
            url = f"{self.base_url}/features/indicators"
            params = {
                "symbol": symbol,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "timeframe": timeframe
            }
            
            # Add indicators to parameters if specified
            if indicators:
                params["indicators"] = ",".join(indicators)
            
            self.logger.info(f"Requesting indicators for {symbol} from {start_date_str} to {end_date_str} ({timeframe})")
            if indicators:
                self.logger.info(f"Requested indicators: {indicators}")
            
            # Make the request
            response = requests.get(url, params=params, headers=self.headers)
            data = self._handle_response(response)
            
            # Convert to DataFrame
            if 'data' in data and data['data']:
                df = pd.DataFrame(data['data'])
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                self.logger.info(f"Retrieved {len(df)} indicator records")
                return df
            else:
                self.logger.warning(f"No indicator data returned for {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error retrieving indicator data: {str(e)}")
            # For development, return sample data instead of failing
            if os.getenv('ENVIRONMENT', 'development') == 'development':
                self.logger.warning("Returning sample indicator data for development")
                return self._get_sample_indicator_data(symbol, start_date, end_date, timeframe, indicators)
            else:
                raise
    
    def get_available_indicators(self) -> List[Dict[str, Any]]:
        """
        Get information about all available indicators in the feature store.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with indicator information
        """
        try:
            url = f"{self.base_url}/features/available-indicators"
            
            self.logger.info("Requesting available indicators information")
            
            response = requests.get(url, headers=self.headers)
            data = self._handle_response(response)
            
            if 'data' in data and isinstance(data['data'], list):
                self.logger.info(f"Retrieved information for {len(data['data'])} indicators")
                return data['data']
            else:
                self.logger.warning("No indicator information returned")
                return []
            
        except Exception as e:
            self.logger.error(f"Error retrieving indicator information: {str(e)}")
            # For development, return sample data
            if os.getenv('ENVIRONMENT', 'development') == 'development':
                self.logger.warning("Returning sample indicator information for development")
                return self._get_sample_indicator_info()
            else:
                raise
    
    def compute_feature(
        self,
        feature_name: str,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1h',
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Request computation of a specific feature for a given timeframe.
        
        Args:
            feature_name: Name of the feature to compute
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe for the data
            parameters: Optional parameters for feature computation
            
        Returns:
            pd.DataFrame: DataFrame with computed feature values
        """
        try:
            # Format dates
            start_date_str = self._format_date(start_date)
            end_date_str = self._format_date(end_date)
            
            # Build URL and request body
            url = f"{self.base_url}/features/compute"
            body = {
                "feature_name": feature_name,
                "symbol": symbol,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "timeframe": timeframe
            }
            
            # Add parameters if provided
            if parameters:
                body["parameters"] = parameters
            
            self.logger.info(f"Requesting computation of feature '{feature_name}' for {symbol}")
            
            # Make the POST request
            response = requests.post(url, json=body, headers=self.headers)
            data = self._handle_response(response)
            
            # Convert to DataFrame
            if 'data' in data and data['data']:
                df = pd.DataFrame(data['data'])
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                self.logger.info(f"Retrieved {len(df)} computed feature records")
                return df
            else:
                self.logger.warning(f"No computed feature data returned for {feature_name}")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error computing feature: {str(e)}")
            # For development purposes
            if os.getenv('ENVIRONMENT', 'development') == 'development':
                return pd.DataFrame()  # Return empty DataFrame in development
            else:
                raise
    
    def _get_sample_ohlcv_data(
        self, 
        symbol: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """
        Generate sample OHLCV data for development when the API is unavailable.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            pd.DataFrame: Sample OHLCV data
        """
        # Convert string dates to datetime if necessary
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determine frequency based on timeframe
        if timeframe == '1m':
            freq = '1min'
        elif timeframe == '5m':
            freq = '5min'
        elif timeframe == '15m':
            freq = '15min'
        elif timeframe == '1h':
            freq = '1H'
        elif timeframe == '4h':
            freq = '4H'
        elif timeframe == '1d':
            freq = '1D'
        else:
            freq = '1H'  # Default to hourly
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Base price and volatility based on symbol
        if symbol == 'EUR_USD':
            base_price = 1.1
            volatility = 0.001
        elif symbol == 'GBP_USD':
            base_price = 1.3
            volatility = 0.0015
        elif symbol == 'USD_JPY':
            base_price = 110.0
            volatility = 0.1
        else:
            base_price = 1.0
            volatility = 0.001
        
        import numpy as np
        
        # Generate random price data
        np.random.seed(42)  # For reproducibility
        n = len(dates)
        
        # Generate random walk for close prices
        changes = np.random.normal(0, volatility, n)
        closes = base_price + np.cumsum(changes)
        
        # Generate other OHLCV data
        highs = closes + np.random.uniform(0, volatility * 2, n)
        lows = closes - np.random.uniform(0, volatility * 2, n)
        opens = np.zeros(n)
        opens[0] = base_price
        opens[1:] = closes[:-1]
        
        # Ensure high is always >= close and open, low is always <= close and open
        for i in range(n):
            high_val = max(highs[i], closes[i], opens[i])
            low_val = min(lows[i], closes[i], opens[i])
            highs[i] = high_val
            lows[i] = low_val
        
        # Generate volume data
        volumes = np.random.randint(1000, 10000, n)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        return df
    
    def _get_sample_indicator_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate sample indicator data for development when the API is unavailable.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            indicators: List of indicator names
            
        Returns:
            pd.DataFrame: Sample indicator data
        """
        # First, get sample OHLCV data to base indicators on
        ohlcv_data = self._get_sample_ohlcv_data(symbol, start_date, end_date, timeframe)
        
        if ohlcv_data.empty:
            return pd.DataFrame()
        
        # If no indicators are specified, use a default set
        if not indicators:
            indicators = ['sma_5', 'sma_20', 'ema_5', 'ema_20', 'rsi_14']
        
        # Create DataFrame with timestamp
        df = pd.DataFrame({'timestamp': ohlcv_data['timestamp']})
        
        import numpy as np
        
        # Generate indicator data based on OHLCV
        for indicator in indicators:
            if 'sma' in indicator:
                # Extract window from indicator name, or default to 5
                window = int(indicator.split('_')[1]) if '_' in indicator else 5
                df[indicator] = ohlcv_data['close'].rolling(window=min(window, 3)).mean()
            
            elif 'ema' in indicator:
                window = int(indicator.split('_')[1]) if '_' in indicator else 5
                df[indicator] = ohlcv_data['close'].ewm(span=min(window, 3)).mean()
            
            elif 'rsi' in indicator:
                # Simplified RSI calculation for sample data
                df[indicator] = 50 + np.random.normal(0, 10, len(ohlcv_data))
                df[indicator] = df[indicator].clip(0, 100)  # Ensure RSI is between 0 and 100
            
            elif 'macd' in indicator:
                if indicator == 'macd':
                    df[indicator] = np.random.normal(0, 0.001, len(ohlcv_data))
                elif indicator == 'macd_signal':
                    df[indicator] = np.random.normal(0, 0.001, len(ohlcv_data))
                elif indicator == 'macd_hist':
                    df[indicator] = np.random.normal(0, 0.0005, len(ohlcv_data))
            
            elif 'bb_' in indicator:
                if indicator == 'bb_upper':
                    df[indicator] = ohlcv_data['close'] + np.random.uniform(0.001, 0.003, len(ohlcv_data))
                elif indicator == 'bb_middle':
                    df[indicator] = ohlcv_data['close']
                elif indicator == 'bb_lower':
                    df[indicator] = ohlcv_data['close'] - np.random.uniform(0.001, 0.003, len(ohlcv_data))
            
            elif 'atr' in indicator:
                window = int(indicator.split('_')[1]) if '_' in indicator else 14
                df[indicator] = np.random.uniform(0.001, 0.002, len(ohlcv_data))
            
            else:
                # Generic random indicator data
                df[indicator] = np.random.normal(0, 1, len(ohlcv_data))
        
        # Fill NaN values that might have been created by rolling windows
        df = df.fillna(method='bfill')
        
        return df
    
    def _get_sample_indicator_info(self) -> List[Dict[str, Any]]:
        """
        Generate sample indicator information for development.
        
        Returns:
            List[Dict[str, Any]]: List of sample indicator information
        """
        return [
            {
                "id": "sma_5",
                "name": "Simple Moving Average (5)",
                "description": "5-period Simple Moving Average",
                "category": "trend",
                "default_parameters": {"window": 5},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "sma_10",
                "name": "Simple Moving Average (10)",
                "description": "10-period Simple Moving Average",
                "category": "trend",
                "default_parameters": {"window": 10},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "sma_20",
                "name": "Simple Moving Average (20)",
                "description": "20-period Simple Moving Average",
                "category": "trend",
                "default_parameters": {"window": 20},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "ema_5",
                "name": "Exponential Moving Average (5)",
                "description": "5-period Exponential Moving Average",
                "category": "trend",
                "default_parameters": {"window": 5},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "ema_10",
                "name": "Exponential Moving Average (10)",
                "description": "10-period Exponential Moving Average",
                "category": "trend",
                "default_parameters": {"window": 10},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "ema_20",
                "name": "Exponential Moving Average (20)",
                "description": "20-period Exponential Moving Average",
                "category": "trend",
                "default_parameters": {"window": 20},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "rsi_14",
                "name": "Relative Strength Index (14)",
                "description": "14-period Relative Strength Index",
                "category": "momentum",
                "default_parameters": {"window": 14},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "macd",
                "name": "MACD Line",
                "description": "Moving Average Convergence Divergence Line",
                "category": "momentum",
                "default_parameters": {"fast_window": 12, "slow_window": 26},
                "available_timeframes": ["5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "macd_signal",
                "name": "MACD Signal Line",
                "description": "Signal Line for Moving Average Convergence Divergence",
                "category": "momentum",
                "default_parameters": {"fast_window": 12, "slow_window": 26, "signal_window": 9},
                "available_timeframes": ["5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "macd_hist",
                "name": "MACD Histogram",
                "description": "Histogram showing difference between MACD Line and Signal Line",
                "category": "momentum",
                "default_parameters": {"fast_window": 12, "slow_window": 26, "signal_window": 9},
                "available_timeframes": ["5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "bb_upper",
                "name": "Bollinger Bands (Upper)",
                "description": "Upper band of Bollinger Bands",
                "category": "volatility",
                "default_parameters": {"window": 20, "std_dev": 2},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "bb_middle",
                "name": "Bollinger Bands (Middle)",
                "description": "Middle band of Bollinger Bands (SMA)",
                "category": "volatility",
                "default_parameters": {"window": 20},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "bb_lower",
                "name": "Bollinger Bands (Lower)",
                "description": "Lower band of Bollinger Bands",
                "category": "volatility",
                "default_parameters": {"window": 20, "std_dev": 2},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            {
                "id": "atr_14",
                "name": "Average True Range (14)",
                "description": "14-period Average True Range",
                "category": "volatility",
                "default_parameters": {"window": 14},
                "available_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
        ]