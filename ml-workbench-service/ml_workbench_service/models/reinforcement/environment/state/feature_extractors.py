"""
Feature Extractors for RL Environments

This module provides feature extractors for different types of data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    
    Feature extractors transform raw data into features for the RL agent.
    """
    
    @abstractmethod
    def extract(self, data: Any) -> np.ndarray:
        """
        Extract features from data.
        
        Args:
            data: Raw data
            
        Returns:
            Extracted features as a numpy array
        """
        pass


class MarketDataFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for market data (OHLCV).
    
    This extractor processes OHLCV data and applies optional normalization.
    """
    
    def __init__(self, 
                 features: List[str], 
                 lookback_periods: int, 
                 normalize: bool = True):
        """
        Initialize the market data feature extractor.
        
        Args:
            features: List of features to extract (e.g., 'open', 'high', 'low', 'close')
            lookback_periods: Number of past periods to include
            normalize: Whether to normalize the features
        """
        self.features = features
        self.lookback_periods = lookback_periods
        self.normalize = normalize
    
    def extract(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Extracted features as a numpy array
        """
        if len(data) < self.lookback_periods:
            # Not enough data, pad with zeros
            return np.zeros(len(self.features) * self.lookback_periods)
        
        # Extract the most recent data
        recent_data = data.iloc[-self.lookback_periods:]
        
        features = []
        
        for feature in self.features:
            if feature in recent_data.columns:
                values = recent_data[feature].values
                
                # Apply normalization if enabled
                if self.normalize:
                    # Z-score normalization within the window
                    mean = np.mean(values)
                    std = np.std(values)
                    if std > 0:
                        values = (values - mean) / std
                
                features.extend(values)
            else:
                # Feature not available, pad with zeros
                features.extend([0] * self.lookback_periods)
        
        return np.array(features, dtype=np.float32)


class TechnicalIndicatorExtractor(FeatureExtractor):
    """
    Feature extractor for technical indicators.
    
    This extractor calculates technical indicators from market data.
    """
    
    def __init__(self, indicators: List[str], normalize: bool = True):
        """
        Initialize the technical indicator extractor.
        
        Args:
            indicators: List of indicators to calculate
            normalize: Whether to normalize the indicators
        """
        self.indicators = indicators
        self.normalize = normalize
    
    def extract(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract technical indicators from market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Extracted indicators as a numpy array
        """
        if len(data) < 20:  # Minimum data required for most indicators
            return np.zeros(len(self.indicators))
        
        # Make a copy to avoid modifying the original dataframe
        df = data.copy()
        
        # Calculate indicators
        for indicator in self.indicators:
            if indicator == 'sma5':
                df['sma5'] = df['close'].rolling(window=5).mean()
            elif indicator == 'sma20':
                df['sma20'] = df['close'].rolling(window=20).mean()
            elif indicator == 'ema5':
                df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
            elif indicator == 'ema20':
                df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            elif indicator == 'rsi':
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            elif indicator == 'macd':
                df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
            elif indicator == 'macd_signal':
                if 'macd' not in df.columns:
                    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            elif indicator == 'macd_hist':
                if 'macd' not in df.columns:
                    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
                if 'macd_signal' not in df.columns:
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
            elif indicator == 'bb_upper':
                if 'bb_middle' not in df.columns:
                    df['bb_middle'] = df['close'].rolling(window=20).mean()
                if 'bb_std' not in df.columns:
                    df['bb_std'] = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            elif indicator == 'bb_lower':
                if 'bb_middle' not in df.columns:
                    df['bb_middle'] = df['close'].rolling(window=20).mean()
                if 'bb_std' not in df.columns:
                    df['bb_std'] = df['close'].rolling(window=20).std()
                df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Extract the latest values
        features = []
        
        for indicator in self.indicators:
            if indicator in df.columns:
                value = df[indicator].iloc[-1]
                
                # Handle NaN values
                if pd.isna(value):
                    value = 0.0
                
                # Apply normalization if enabled
                if self.normalize:
                    if indicator == 'rsi':
                        # RSI is already in [0, 100], normalize to [0, 1]
                        value = value / 100
                    elif indicator in ['macd', 'macd_signal', 'macd_hist']:
                        # MACD can be both positive and negative, normalize to [-1, 1]
                        max_abs = df[indicator].abs().max()
                        if max_abs > 0:
                            value = value / max_abs
                    else:
                        # Price-based indicators, normalize to last close price
                        last_close = df['close'].iloc[-1]
                        if last_close > 0:
                            value = (value / last_close - 1)
                
                features.append(value)
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)


class BrokerStateExtractor(FeatureExtractor):
    """
    Feature extractor for broker state.
    
    This extractor processes broker state information.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize the broker state extractor.
        
        Args:
            normalize: Whether to normalize the features
        """
        self.normalize = normalize
    
    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from broker state.
        
        Args:
            state: Broker state dictionary
            
        Returns:
            Extracted features as a numpy array
        """
        features = []
        
        # Extract common broker state features
        account_balance = state.get('account_balance', 0.0)
        current_position = state.get('current_position', 0.0)
        current_pnl = state.get('current_pnl', 0.0)
        total_trades = state.get('total_trades', 0)
        profitable_trades = state.get('profitable_trades', 0)
        
        # Apply normalization if enabled
        if self.normalize:
            features.append(account_balance / 10000.0)  # Normalize by initial balance
            features.append(current_position / state.get('max_position_size', 1.0))  # Normalize by max position
            features.append(current_pnl / 1000.0)  # Normalize P&L
            features.append(total_trades / 100.0)  # Normalize trade count
            features.append(profitable_trades / max(1, total_trades))  # Win rate
        else:
            features.extend([account_balance, current_position, current_pnl, total_trades, profitable_trades])
        
        return np.array(features, dtype=np.float32)


class OrderBookExtractor(FeatureExtractor):
    """
    Feature extractor for order book data.
    
    This extractor processes order book information.
    """
    
    def __init__(self, levels: int = 5, normalize: bool = True):
        """
        Initialize the order book extractor.
        
        Args:
            levels: Number of order book levels to include
            normalize: Whether to normalize the features
        """
        self.levels = levels
        self.normalize = normalize
    
    def extract(self, order_book: Dict[str, List[Tuple[float, float]]]) -> np.ndarray:
        """
        Extract features from order book.
        
        Args:
            order_book: Order book dictionary with 'bids' and 'asks' lists
            
        Returns:
            Extracted features as a numpy array
        """
        features = []
        
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            # No order book data, return zeros
            return np.zeros(self.levels * 4)
        
        bids = order_book['bids']
        asks = order_book['asks']
        
        # Get mid price for normalization
        if self.normalize and bids and asks:
            mid_price = (bids[0][0] + asks[0][0]) / 2
        else:
            mid_price = 1.0
        
        # Process bid levels
        for i in range(min(self.levels, len(bids))):
            price, volume = bids[i]
            
            if self.normalize:
                # Normalize price to mid price
                features.append(price / mid_price - 1)
                # Normalize volume (simple approach)
                features.append(min(volume / 100.0, 1.0))
            else:
                features.extend([price, volume])
        
        # Fill with zeros if there are fewer than levels
        for i in range(self.levels - min(self.levels, len(bids))):
            features.extend([0, 0])
        
        # Process ask levels
        for i in range(min(self.levels, len(asks))):
            price, volume = asks[i]
            
            if self.normalize:
                # Normalize price to mid price
                features.append(price / mid_price - 1)
                # Normalize volume (simple approach)
                features.append(min(volume / 100.0, 1.0))
            else:
                features.extend([price, volume])
        
        # Fill with zeros if there are fewer than levels
        for i in range(self.levels - min(self.levels, len(asks))):
            features.extend([0, 0])
        
        return np.array(features, dtype=np.float32)


class NewsSentimentExtractor(FeatureExtractor):
    """
    Feature extractor for news sentiment data.
    
    This extractor processes news sentiment information.
    """
    
    def __init__(self, num_features: int):
        """
        Initialize the news sentiment extractor.
        
        Args:
            num_features: Number of features to extract
        """
        self.num_features = num_features
    
    def extract(self, news_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from news sentiment data.
        
        Args:
            news_data: News sentiment data
            
        Returns:
            Extracted features as a numpy array
        """
        if not news_data or 'events' not in news_data:
            return np.zeros(self.num_features)
        
        events = news_data['events']
        
        if not events:
            return np.zeros(self.num_features)
        
        features = []
        
        # Basic features
        features.append(len(events))  # Number of recent events
        
        # Average sentiment (normalized to [-1, 1])
        avg_sentiment = sum(event.get('sentiment_score', 0) for event in events) / len(events)
        features.append(avg_sentiment)
        
        # Highest impact (normalized to [0, 1])
        max_impact = max(event.get('impact', 0) for event in events) / 3  # Assuming impact is 0-3
        features.append(max_impact)
        
        # Additional features can be added here
        
        # Ensure we have the right number of features
        if len(features) < self.num_features:
            features.extend([0] * (self.num_features - len(features)))
        elif len(features) > self.num_features:
            features = features[:self.num_features]
        
        return np.array(features, dtype=np.float32)