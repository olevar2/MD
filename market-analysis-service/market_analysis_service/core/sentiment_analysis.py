"""
Sentiment Analysis module for Market Analysis Service.

This module provides algorithms for analyzing market sentiment.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Class for analyzing market sentiment.
    """
    
    def __init__(self):
        """
        Initialize the Sentiment Analyzer.
        """
        pass
        
    def analyze_sentiment(
        self,
        data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        additional_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment.
        
        Args:
            data: Market data
            sentiment_data: External sentiment data (if available)
            additional_parameters: Additional parameters for analysis
            
        Returns:
            Sentiment analysis results
        """
        if additional_parameters is None:
            additional_parameters = {}
            
        # Calculate technical sentiment indicators
        technical_sentiment = self._calculate_technical_sentiment(data, additional_parameters)
        
        # Calculate market sentiment from price action
        price_sentiment = self._calculate_price_sentiment(data, additional_parameters)
        
        # Calculate sentiment from external data (if available)
        external_sentiment = self._calculate_external_sentiment(sentiment_data, additional_parameters)
        
        # Calculate combined sentiment
        combined_sentiment = self._calculate_combined_sentiment(
            technical_sentiment, price_sentiment, external_sentiment, additional_parameters
        )
        
        return {
            "technical_sentiment": technical_sentiment,
            "price_sentiment": price_sentiment,
            "external_sentiment": external_sentiment,
            "combined_sentiment": combined_sentiment
        }
        
    def _calculate_technical_sentiment(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate sentiment based on technical indicators.
        
        Args:
            data: Market data
            parameters: Additional parameters
            
        Returns:
            Technical sentiment data
        """
        # Ensure we have enough data
        if len(data) < 50:
            return {
                "sentiment": 0,
                "indicators": {}
            }
            
        # Calculate technical indicators
        indicators = {}
        
        # 1. Moving Average Crossover
        short_ma = data["close"].rolling(window=20).mean()
        long_ma = data["close"].rolling(window=50).mean()
        
        ma_crossover = 1 if short_ma.iloc[-1] > long_ma.iloc[-1] else -1
        indicators["ma_crossover"] = ma_crossover
        
        # 2. RSI (Relative Strength Index)
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_value = rsi.iloc[-1]
        rsi_sentiment = 0
        
        if rsi_value < 30:
            rsi_sentiment = 1  # Oversold (bullish)
        elif rsi_value > 70:
            rsi_sentiment = -1  # Overbought (bearish)
            
        indicators["rsi"] = {
            "value": float(rsi_value),
            "sentiment": rsi_sentiment
        }
        
        # 3. MACD (Moving Average Convergence Divergence)
        ema_12 = data["close"].ewm(span=12).mean()
        ema_26 = data["close"].ewm(span=26).mean()
        
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        
        macd_value = macd.iloc[-1]
        signal_value = signal.iloc[-1]
        
        macd_sentiment = 1 if macd_value > signal_value else -1
        
        indicators["macd"] = {
            "value": float(macd_value),
            "signal": float(signal_value),
            "sentiment": macd_sentiment
        }
        
        # 4. Bollinger Bands
        middle_band = data["close"].rolling(window=20).mean()
        std_dev = data["close"].rolling(window=20).std()
        
        upper_band = middle_band + 2 * std_dev
        lower_band = middle_band - 2 * std_dev
        
        current_price = data["close"].iloc[-1]
        
        bb_sentiment = 0
        
        if current_price > upper_band.iloc[-1]:
            bb_sentiment = -1  # Overbought (bearish)
        elif current_price < lower_band.iloc[-1]:
            bb_sentiment = 1  # Oversold (bullish)
            
        indicators["bollinger_bands"] = {
            "upper": float(upper_band.iloc[-1]),
            "middle": float(middle_band.iloc[-1]),
            "lower": float(lower_band.iloc[-1]),
            "sentiment": bb_sentiment
        }
        
        # Calculate overall technical sentiment
        sentiment_values = [
            ma_crossover,
            rsi_sentiment,
            macd_sentiment,
            bb_sentiment
        ]
        
        # Remove neutral values
        sentiment_values = [v for v in sentiment_values if v != 0]
        
        # Calculate average sentiment
        if sentiment_values:
            overall_sentiment = sum(sentiment_values) / len(sentiment_values)
        else:
            overall_sentiment = 0
            
        return {
            "sentiment": float(overall_sentiment),
            "indicators": indicators
        }
        
    def _calculate_price_sentiment(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate sentiment based on price action.
        
        Args:
            data: Market data
            parameters: Additional parameters
            
        Returns:
            Price sentiment data
        """
        # Ensure we have enough data
        if len(data) < 20:
            return {
                "sentiment": 0,
                "components": {}
            }
            
        components = {}
        
        # 1. Price Momentum
        returns = data["close"].pct_change()
        momentum = returns.rolling(window=10).mean().iloc[-1]
        
        momentum_sentiment = np.tanh(momentum * 100)  # Scale and bound between -1 and 1
        
        components["momentum"] = {
            "value": float(momentum),
            "sentiment": float(momentum_sentiment)
        }
        
        # 2. Price Trend
        price_20d_ago = data["close"].iloc[-21] if len(data) >= 21 else data["close"].iloc[0]
        current_price = data["close"].iloc[-1]
        
        price_change = (current_price - price_20d_ago) / price_20d_ago
        trend_sentiment = np.tanh(price_change * 5)  # Scale and bound between -1 and 1
        
        components["trend"] = {
            "value": float(price_change),
            "sentiment": float(trend_sentiment)
        }
        
        # 3. Volume Analysis
        if "volume" in data.columns:
            volume = data["volume"]
            avg_volume = volume.rolling(window=20).mean()
            
            volume_ratio = volume.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            
            # Volume sentiment depends on price direction
            if returns.iloc[-1] > 0:
                # Rising price with high volume is bullish
                volume_sentiment = (volume_ratio - 1) * 0.5
            else:
                # Falling price with high volume is bearish
                volume_sentiment = (1 - volume_ratio) * 0.5
                
            volume_sentiment = max(-1, min(1, volume_sentiment))  # Bound between -1 and 1
            
            components["volume"] = {
                "value": float(volume_ratio),
                "sentiment": float(volume_sentiment)
            }
        else:
            components["volume"] = {
                "value": None,
                "sentiment": 0
            }
            
        # 4. Candlestick Patterns
        # Simple bullish/bearish candle detection
        open_price = data["open"].iloc[-1]
        close_price = data["close"].iloc[-1]
        high_price = data["high"].iloc[-1]
        low_price = data["low"].iloc[-1]
        
        body_size = abs(close_price - open_price)
        total_size = high_price - low_price
        
        if total_size > 0:
            body_ratio = body_size / total_size
        else:
            body_ratio = 0
            
        if close_price > open_price:
            # Bullish candle
            candle_sentiment = body_ratio
        else:
            # Bearish candle
            candle_sentiment = -body_ratio
            
        components["candlestick"] = {
            "value": float(body_ratio),
            "sentiment": float(candle_sentiment)
        }
        
        # Calculate overall price sentiment
        sentiment_values = [
            momentum_sentiment,
            trend_sentiment,
            components["volume"]["sentiment"],
            candle_sentiment
        ]
        
        # Calculate weighted average sentiment
        weights = [0.3, 0.3, 0.2, 0.2]
        overall_sentiment = sum(s * w for s, w in zip(sentiment_values, weights))
        
        return {
            "sentiment": float(overall_sentiment),
            "components": components
        }
        
    def _calculate_external_sentiment(
        self,
        sentiment_data: Optional[pd.DataFrame],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate sentiment based on external data.
        
        Args:
            sentiment_data: External sentiment data
            parameters: Additional parameters
            
        Returns:
            External sentiment data
        """
        if sentiment_data is None or len(sentiment_data) == 0:
            return {
                "sentiment": 0,
                "sources": {}
            }
            
        # Process external sentiment data
        sources = {}
        
        # Example: Process sentiment data from different sources
        if "news_sentiment" in sentiment_data.columns:
            news_sentiment = sentiment_data["news_sentiment"].iloc[-1]
            sources["news"] = float(news_sentiment)
            
        if "social_sentiment" in sentiment_data.columns:
            social_sentiment = sentiment_data["social_sentiment"].iloc[-1]
            sources["social"] = float(social_sentiment)
            
        if "analyst_sentiment" in sentiment_data.columns:
            analyst_sentiment = sentiment_data["analyst_sentiment"].iloc[-1]
            sources["analyst"] = float(analyst_sentiment)
            
        # Calculate overall external sentiment
        if sources:
            overall_sentiment = sum(sources.values()) / len(sources)
        else:
            overall_sentiment = 0
            
        return {
            "sentiment": float(overall_sentiment),
            "sources": sources
        }
        
    def _calculate_combined_sentiment(
        self,
        technical_sentiment: Dict[str, Any],
        price_sentiment: Dict[str, Any],
        external_sentiment: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate combined sentiment from all sources.
        
        Args:
            technical_sentiment: Technical sentiment data
            price_sentiment: Price sentiment data
            external_sentiment: External sentiment data
            parameters: Additional parameters
            
        Returns:
            Combined sentiment data
        """
        # Get sentiment values
        technical_value = technical_sentiment["sentiment"]
        price_value = price_sentiment["sentiment"]
        external_value = external_sentiment["sentiment"]
        
        # Get weights from parameters or use defaults
        technical_weight = parameters.get("technical_weight", 0.4)
        price_weight = parameters.get("price_weight", 0.4)
        external_weight = parameters.get("external_weight", 0.2)
        
        # Adjust weights if external sentiment is not available
        if external_value == 0 and "sources" in external_sentiment and not external_sentiment["sources"]:
            technical_weight = technical_weight / (technical_weight + price_weight)
            price_weight = price_weight / (technical_weight + price_weight)
            external_weight = 0
            
        # Calculate weighted average
        combined_value = (
            technical_value * technical_weight +
            price_value * price_weight +
            external_value * external_weight
        )
        
        # Determine sentiment category
        if combined_value > 0.5:
            category = "strongly_bullish"
        elif combined_value > 0.1:
            category = "bullish"
        elif combined_value > -0.1:
            category = "neutral"
        elif combined_value > -0.5:
            category = "bearish"
        else:
            category = "strongly_bearish"
            
        return {
            "sentiment": float(combined_value),
            "category": category,
            "components": {
                "technical": {
                    "value": float(technical_value),
                    "weight": float(technical_weight)
                },
                "price": {
                    "value": float(price_value),
                    "weight": float(price_weight)
                },
                "external": {
                    "value": float(external_value),
                    "weight": float(external_weight)
                }
            }
        }