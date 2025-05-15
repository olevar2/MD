"""
Unit tests for sentiment analysis.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_analysis_service.core.sentiment_analysis import SentimentAnalyzer

class TestSentimentAnalysis(unittest.TestCase):
    """
    Unit tests for sentiment analysis.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Create test data
        self.create_test_data()
        
    def create_test_data(self):
        """
        Create test data for sentiment analysis.
        """
        # Create a DataFrame with OHLCV data
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        
        # Create price data with different sentiment patterns
        close = []
        high = []
        low = []
        
        for i in range(100):
            if i < 20:
                # Bullish trend
                c = 100 + i * 0.5 + np.random.normal(0, 1)
            elif i < 40:
                # Bearish trend
                c = 110 - (i - 20) * 0.5 + np.random.normal(0, 1)
            elif i < 60:
                # Sideways with bullish bias
                c = 100 + 5 * np.sin((i - 40) / 10) + np.random.normal(0, 1) + 0.1 * (i - 40)
            elif i < 80:
                # Sideways with bearish bias
                c = 100 + 5 * np.sin((i - 60) / 10) + np.random.normal(0, 1) - 0.1 * (i - 60)
            else:
                # Strong bullish trend
                c = 90 + (i - 80) * 1.0 + np.random.normal(0, 1)
                
            # Create high and low
            h = c + abs(np.random.normal(0, 1))
            l = c - abs(np.random.normal(0, 1))
            
            close.append(c)
            high.append(h)
            low.append(l)
            
        # Create DataFrame
        self.test_data = pd.DataFrame({
            "timestamp": dates,
            "open": close,  # Use close as open for simplicity
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, 100)
        })
        
        # Create external sentiment data
        self.sentiment_data = pd.DataFrame({
            "timestamp": dates,
            "news_sentiment": np.random.uniform(-1, 1, 100),
            "social_sentiment": np.random.uniform(-1, 1, 100),
            "analyst_sentiment": np.random.uniform(-1, 1, 100)
        })
        
    def test_analyze_sentiment(self):
        """
        Test analyzing sentiment.
        """
        # Analyze sentiment
        sentiment_results = self.sentiment_analyzer.analyze_sentiment(
            data=self.test_data,
            sentiment_data=self.sentiment_data
        )
        
        # Check that we have technical sentiment
        self.assertIn("technical_sentiment", sentiment_results)
        
        # Check that we have price sentiment
        self.assertIn("price_sentiment", sentiment_results)
        
        # Check that we have external sentiment
        self.assertIn("external_sentiment", sentiment_results)
        
        # Check that we have combined sentiment
        self.assertIn("combined_sentiment", sentiment_results)
        
        # Check that the technical sentiment has the required fields
        technical_sentiment = sentiment_results["technical_sentiment"]
        self.assertIn("sentiment", technical_sentiment)
        self.assertIn("indicators", technical_sentiment)
        
        # Check that the sentiment is between -1 and 1
        self.assertGreaterEqual(technical_sentiment["sentiment"], -1)
        self.assertLessEqual(technical_sentiment["sentiment"], 1)
        
        # Check that the price sentiment has the required fields
        price_sentiment = sentiment_results["price_sentiment"]
        self.assertIn("sentiment", price_sentiment)
        self.assertIn("components", price_sentiment)
        
        # Check that the sentiment is between -1 and 1
        self.assertGreaterEqual(price_sentiment["sentiment"], -1)
        self.assertLessEqual(price_sentiment["sentiment"], 1)
        
        # Check that the external sentiment has the required fields
        external_sentiment = sentiment_results["external_sentiment"]
        self.assertIn("sentiment", external_sentiment)
        self.assertIn("sources", external_sentiment)
        
        # Check that the sentiment is between -1 and 1
        self.assertGreaterEqual(external_sentiment["sentiment"], -1)
        self.assertLessEqual(external_sentiment["sentiment"], 1)
        
        # Check that the combined sentiment has the required fields
        combined_sentiment = sentiment_results["combined_sentiment"]
        self.assertIn("sentiment", combined_sentiment)
        self.assertIn("category", combined_sentiment)
        self.assertIn("components", combined_sentiment)
        
        # Check that the sentiment is between -1 and 1
        self.assertGreaterEqual(combined_sentiment["sentiment"], -1)
        self.assertLessEqual(combined_sentiment["sentiment"], 1)
        
        # Check that the category is valid
        self.assertIn(combined_sentiment["category"], ["strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"])
        
    def test_calculate_technical_sentiment(self):
        """
        Test calculating technical sentiment.
        """
        # Calculate technical sentiment
        technical_sentiment = self.sentiment_analyzer._calculate_technical_sentiment(
            data=self.test_data,
            parameters={}
        )
        
        # Check that we have the required fields
        self.assertIn("sentiment", technical_sentiment)
        self.assertIn("indicators", technical_sentiment)
        
        # Check that the sentiment is between -1 and 1
        self.assertGreaterEqual(technical_sentiment["sentiment"], -1)
        self.assertLessEqual(technical_sentiment["sentiment"], 1)
        
        # Check that we have the expected indicators
        self.assertIn("ma_crossover", technical_sentiment["indicators"])
        self.assertIn("rsi", technical_sentiment["indicators"])
        self.assertIn("macd", technical_sentiment["indicators"])
        self.assertIn("bollinger_bands", technical_sentiment["indicators"])
        
    def test_calculate_price_sentiment(self):
        """
        Test calculating price sentiment.
        """
        # Calculate price sentiment
        price_sentiment = self.sentiment_analyzer._calculate_price_sentiment(
            data=self.test_data,
            parameters={}
        )
        
        # Check that we have the required fields
        self.assertIn("sentiment", price_sentiment)
        self.assertIn("components", price_sentiment)
        
        # Check that the sentiment is between -1 and 1
        self.assertGreaterEqual(price_sentiment["sentiment"], -1)
        self.assertLessEqual(price_sentiment["sentiment"], 1)
        
        # Check that we have the expected components
        self.assertIn("momentum", price_sentiment["components"])
        self.assertIn("trend", price_sentiment["components"])
        self.assertIn("volume", price_sentiment["components"])
        self.assertIn("candlestick", price_sentiment["components"])
        
    def test_calculate_external_sentiment(self):
        """
        Test calculating external sentiment.
        """
        # Calculate external sentiment
        external_sentiment = self.sentiment_analyzer._calculate_external_sentiment(
            sentiment_data=self.sentiment_data,
            parameters={}
        )
        
        # Check that we have the required fields
        self.assertIn("sentiment", external_sentiment)
        self.assertIn("sources", external_sentiment)
        
        # Check that the sentiment is between -1 and 1
        self.assertGreaterEqual(external_sentiment["sentiment"], -1)
        self.assertLessEqual(external_sentiment["sentiment"], 1)
        
        # Check that we have the expected sources
        self.assertIn("news", external_sentiment["sources"])
        self.assertIn("social", external_sentiment["sources"])
        self.assertIn("analyst", external_sentiment["sources"])
        
    def test_calculate_combined_sentiment(self):
        """
        Test calculating combined sentiment.
        """
        # Calculate individual sentiments
        technical_sentiment = self.sentiment_analyzer._calculate_technical_sentiment(
            data=self.test_data,
            parameters={}
        )
        
        price_sentiment = self.sentiment_analyzer._calculate_price_sentiment(
            data=self.test_data,
            parameters={}
        )
        
        external_sentiment = self.sentiment_analyzer._calculate_external_sentiment(
            sentiment_data=self.sentiment_data,
            parameters={}
        )
        
        # Calculate combined sentiment
        combined_sentiment = self.sentiment_analyzer._calculate_combined_sentiment(
            technical_sentiment=technical_sentiment,
            price_sentiment=price_sentiment,
            external_sentiment=external_sentiment,
            parameters={}
        )
        
        # Check that we have the required fields
        self.assertIn("sentiment", combined_sentiment)
        self.assertIn("category", combined_sentiment)
        self.assertIn("components", combined_sentiment)
        
        # Check that the sentiment is between -1 and 1
        self.assertGreaterEqual(combined_sentiment["sentiment"], -1)
        self.assertLessEqual(combined_sentiment["sentiment"], 1)
        
        # Check that the category is valid
        self.assertIn(combined_sentiment["category"], ["strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"])
        
        # Check that we have the expected components
        self.assertIn("technical", combined_sentiment["components"])
        self.assertIn("price", combined_sentiment["components"])
        self.assertIn("external", combined_sentiment["components"])
        
if __name__ == "__main__":
    unittest.main()
