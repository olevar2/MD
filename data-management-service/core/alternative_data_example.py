"""
Alternative Data Integration Example.

This script demonstrates the usage of the Alternative Data Integration framework.
"""
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.models import AlternativeDataType
from services.service import AlternativeDataService
from adapters.adapter_factory_1 import MultiSourceAdapterFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def get_news_data(service: AlternativeDataService):
    """
    Get news data example.
    
    Args:
        service: Alternative data service
    """
    logger.info("Getting news data...")
    
    # Define parameters
    symbols = ["EUR/USD", "GBP/USD", "USD/JPY"]
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    # Get news data
    news_data = await service.get_data(
        data_type=AlternativeDataType.NEWS,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    logger.info(f"Retrieved {len(news_data)} news items")
    
    # Display sample
    if not news_data.empty:
        logger.info("\nSample news data:")
        print(news_data.head())
    
    return news_data


async def extract_news_features(service: AlternativeDataService, news_data: pd.DataFrame):
    """
    Extract features from news data example.
    
    Args:
        service: Alternative data service
        news_data: News data
    """
    logger.info("\nExtracting features from news data...")
    
    # Define feature extraction configuration
    feature_config = {
        "extraction_method": "nlp",
        "resample_freq": "D",
        "positive_keywords": ["increase", "rise", "gain", "positive", "growth", "bullish", "up"],
        "negative_keywords": ["decrease", "fall", "drop", "negative", "decline", "bearish", "down"]
    }
    
    # Extract features
    features = await service.extract_features(
        data=news_data,
        data_type=AlternativeDataType.NEWS,
        feature_config=feature_config
    )
    
    logger.info(f"Extracted {len(features)} feature rows")
    
    # Display sample
    if not features.empty:
        logger.info("\nSample features:")
        print(features.head())
    
    return features


async def get_economic_data(service: AlternativeDataService):
    """
    Get economic data example.
    
    Args:
        service: Alternative data service
    """
    logger.info("\nGetting economic data...")
    
    # Define parameters
    symbols = ["USD", "EUR", "GBP", "JPY"]
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    # Get economic data
    economic_data = await service.get_data(
        data_type=AlternativeDataType.ECONOMIC,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
        indicators="gdp,inflation,interest_rate,unemployment"
    )
    
    logger.info(f"Retrieved {len(economic_data)} economic data points")
    
    # Display sample
    if not economic_data.empty:
        logger.info("\nSample economic data:")
        print(economic_data.head())
    
    return economic_data


async def extract_economic_features(service: AlternativeDataService, economic_data: pd.DataFrame):
    """
    Extract features from economic data example.
    
    Args:
        service: Alternative data service
        economic_data: Economic data
    """
    logger.info("\nExtracting features from economic data...")
    
    # Define feature extraction configuration
    feature_config = {
        "extraction_method": "surprise",
        "resample_freq": "M",
        "indicators": ["gdp", "inflation", "interest_rate", "unemployment"]
    }
    
    # Extract features
    features = await service.extract_features(
        data=economic_data,
        data_type=AlternativeDataType.ECONOMIC,
        feature_config=feature_config
    )
    
    logger.info(f"Extracted {len(features)} feature rows")
    
    # Display sample
    if not features.empty:
        logger.info("\nSample features:")
        print(features.head())
    
    return features


async def get_sentiment_data(service: AlternativeDataService):
    """
    Get sentiment data example.
    
    Args:
        service: Alternative data service
    """
    logger.info("\nGetting sentiment data...")
    
    # Define parameters
    symbols = ["EUR/USD", "GBP/USD", "USD/JPY"]
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    # Get sentiment data
    sentiment_data = await service.get_data(
        data_type=AlternativeDataType.SENTIMENT,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
        sources="news,social_media,analyst_ratings"
    )
    
    logger.info(f"Retrieved {len(sentiment_data)} sentiment data points")
    
    # Display sample
    if not sentiment_data.empty:
        logger.info("\nSample sentiment data:")
        print(sentiment_data.head())
    
    return sentiment_data


async def visualize_data(news_features: pd.DataFrame, economic_features: pd.DataFrame, sentiment_data: pd.DataFrame):
    """
    Visualize alternative data example.
    
    Args:
        news_features: News features
        economic_features: Economic features
        sentiment_data: Sentiment data
    """
    logger.info("\nVisualizing alternative data...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot news sentiment
    if not news_features.empty and "sentiment_score" in news_features.columns:
        for symbol in news_features["symbols"].unique():
            symbol_data = news_features[news_features["symbols"] == symbol]
            axes[0].plot(symbol_data["timestamp"], symbol_data["sentiment_score"], label=symbol)
        
        axes[0].set_title("News Sentiment Over Time")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Sentiment Score")
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot economic data
    if not economic_features.empty:
        # Find columns with "surprise" in the name
        surprise_cols = [col for col in economic_features.columns if "surprise" in col and "USD" in col]
        
        if surprise_cols:
            economic_features.set_index("timestamp")[surprise_cols].plot(ax=axes[1])
            axes[1].set_title("Economic Data Surprises (USD)")
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel("Surprise Value")
            axes[1].legend()
            axes[1].grid(True)
    
    # Plot sentiment data
    if not sentiment_data.empty:
        for symbol in sentiment_data["symbol"].unique():
            symbol_data = sentiment_data[sentiment_data["symbol"] == symbol]
            axes[2].plot(symbol_data["timestamp"], symbol_data["sentiment_score"], label=symbol)
        
        axes[2].set_title("Market Sentiment Over Time")
        axes[2].set_xlabel("Date")
        axes[2].set_ylabel("Sentiment Score")
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("alternative_data_visualization.png")
    logger.info("Visualization saved to 'alternative_data_visualization.png'")


async def main():
    """Main function."""
    logger.info("Starting Alternative Data Integration Example")
    
    # Create configuration
    config = {
        "adapters": {
            AlternativeDataType.NEWS: [
                {
                    "name": "MockNewsAdapter",
                    "source_id": "mock_news",
                    "use_mock": True
                }
            ],
            AlternativeDataType.ECONOMIC: [
                {
                    "name": "MockEconomicAdapter",
                    "source_id": "mock_economic",
                    "use_mock": True
                }
            ],
            AlternativeDataType.SENTIMENT: [
                {
                    "name": "MockSentimentAdapter",
                    "source_id": "mock_sentiment",
                    "use_mock": True
                }
            ]
        }
    }
    
    # Create service
    service = AlternativeDataService(config)
    
    # Get news data
    news_data = await get_news_data(service)
    
    # Extract news features
    news_features = await extract_news_features(service, news_data)
    
    # Get economic data
    economic_data = await get_economic_data(service)
    
    # Extract economic features
    economic_features = await extract_economic_features(service, economic_data)
    
    # Get sentiment data
    sentiment_data = await get_sentiment_data(service)
    
    # Visualize data
    await visualize_data(news_features, economic_features, sentiment_data)
    
    logger.info("Alternative Data Integration Example completed")


if __name__ == "__main__":
    asyncio.run(main())
