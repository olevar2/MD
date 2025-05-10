#!/usr/bin/env python
"""
Real Market Data Test

This script tests the optimized components with real market data from various sources.
It supports multiple data providers and fallback mechanisms.
"""

import os
import sys
import time
import logging
import json
import argparse
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import optimized components
from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.utils.adaptive_cache_manager import AdaptiveCacheManager
from analysis_engine.utils.optimized_parallel_processor import OptimizedParallelProcessor
from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
from analysis_engine.utils.distributed_tracing import DistributedTracer
from analysis_engine.utils.gpu_accelerator import GPUAccelerator
from analysis_engine.utils.predictive_cache_manager import PredictiveCacheManager

# Import ML components
from analysis_engine.ml.pattern_recognition_model import PatternRecognitionModel
from analysis_engine.ml.price_prediction_model import PricePredictionModel
from analysis_engine.ml.ml_confluence_detector import MLConfluenceDetector
from analysis_engine.ml.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data providers
class DataProvider:
    """Base class for data providers."""
    
    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get price data for the given symbol and timeframe.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "H1", "D1")
            start_date: Start date (if None, use default)
            end_date: End date (if None, use current date)
            
        Returns:
            DataFrame with OHLCV data
        """
        raise NotImplementedError("Subclasses must implement get_price_data")

class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Alpha Vantage data provider.
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
        # Map timeframes to Alpha Vantage intervals
        self.timeframe_map = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "60min",
            "H4": "4hour",  # Not directly supported, will use 60min and resample
            "D1": "daily"
        }
    
    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get price data from Alpha Vantage.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "H1", "D1")
            start_date: Start date (if None, use default)
            end_date: End date (if None, use current date)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Map timeframe to Alpha Vantage interval
        interval = self.timeframe_map.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Format symbol for Alpha Vantage
        from_currency = symbol[:3]
        to_currency = symbol[3:6]
        
        # Set up parameters
        params = {
            "function": "FX_INTRADAY" if timeframe != "D1" else "FX_DAILY",
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "interval": interval if timeframe != "D1" else None,
            "outputsize": "full",
            "apikey": self.api_key
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Make request
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Error fetching data: {response.status}")
                
                data = await response.json()
                
                # Check for error
                if "Error Message" in data:
                    raise Exception(f"Alpha Vantage error: {data['Error Message']}")
                
                # Extract time series data
                time_series_key = f"Time Series FX ({interval})" if timeframe != "D1" else "Time Series FX (Daily)"
                if time_series_key not in data:
                    raise Exception(f"No time series data found: {data}")
                
                time_series = data[time_series_key]
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient="index")
                
                # Rename columns
                df.columns = [col.split(". ")[1].lower() for col in df.columns]
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                
                # Add timestamp column
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Filter by date range
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                # Resample if needed (for H4)
                if timeframe == "H4":
                    df = df.resample("4H").agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum"
                    })
                
                # Reset index
                df = df.reset_index()
                df = df.rename(columns={"index": "timestamp"})
                
                return df

class SyntheticDataProvider(DataProvider):
    """Synthetic data provider for testing."""
    
    def __init__(self, volatility: float = 0.01, trend: float = 0.0001):
        """
        Initialize the synthetic data provider.
        
        Args:
            volatility: Volatility parameter for price generation
            trend: Trend parameter for price generation
        """
        self.volatility = volatility
        self.trend = trend
        
        # Base prices for different pairs
        self.base_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.3000,
            "USDJPY": 110.00,
            "AUDUSD": 0.7000,
            "USDCAD": 1.3000,
            "EURGBP": 0.8500,
            "EURJPY": 130.00,
            "GBPJPY": 150.00
        }
    
    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic price data.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "H1", "D1")
            start_date: Start date (if None, use 1 year ago)
            end_date: End date (if None, use current date)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # Map timeframe to timedelta
        timeframe_map = {
            "M1": timedelta(minutes=1),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1)
        }
        
        delta = timeframe_map.get(timeframe)
        if delta is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Generate timestamps
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += delta
        
        # Get base price for the symbol
        base_price = self.base_prices.get(symbol, 1.0)
        
        # Generate price data
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate random walk
        returns = np.random.normal(self.trend, self.volatility, len(timestamps))
        cumulative_returns = np.cumsum(returns)
        
        # Add some cyclical patterns
        t = np.linspace(0, 10 * np.pi, len(timestamps))
        cycles = 0.005 * np.sin(t) + 0.003 * np.sin(2 * t) + 0.002 * np.sin(3 * t)
        
        # Combine components
        close_prices = base_price * np.exp(cumulative_returns + cycles)
        
        # Generate OHLC data
        high_prices = close_prices * np.exp(np.random.uniform(0, self.volatility, len(timestamps)))
        low_prices = close_prices * np.exp(-np.random.uniform(0, self.volatility, len(timestamps)))
        open_prices = low_prices + np.random.uniform(0, 1, len(timestamps)) * (high_prices - low_prices)
        
        # Generate volume data
        volume = np.random.uniform(1000, 10000, len(timestamps))
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume
        })
        
        return df

class MultiSourceDataProvider(DataProvider):
    """Data provider that tries multiple sources with fallback."""
    
    def __init__(self, providers: List[DataProvider]):
        """
        Initialize the multi-source data provider.
        
        Args:
            providers: List of data providers to try
        """
        self.providers = providers
    
    async def get_price_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get price data from multiple sources with fallback.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "H1", "D1")
            start_date: Start date (if None, use default)
            end_date: End date (if None, use current date)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try each provider in order
        for i, provider in enumerate(self.providers):
            try:
                logger.info(f"Trying data provider {i+1}/{len(self.providers)} for {symbol} {timeframe}")
                return await provider.get_price_data(symbol, timeframe, start_date, end_date)
            except Exception as e:
                logger.warning(f"Error with provider {i+1}: {e}")
                if i == len(self.providers) - 1:
                    raise Exception(f"All data providers failed for {symbol} {timeframe}")

async def run_test(
    symbols: List[str],
    timeframes: List[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    alpha_vantage_api_key: Optional[str] = None,
    use_gpu: bool = False,
    use_ml: bool = False,
    output_dir: str = "performance_results"
):
    """
    Run the test with real market data.
    
    Args:
        symbols: List of currency pairs to test
        timeframes: List of timeframes to test
        start_date: Start date for data
        end_date: End date for data
        alpha_vantage_api_key: Alpha Vantage API key (if None, use synthetic data only)
        use_gpu: Whether to use GPU acceleration
        use_ml: Whether to use machine learning models
        output_dir: Directory for output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up data provider
    providers = []
    if alpha_vantage_api_key:
        providers.append(AlphaVantageProvider(alpha_vantage_api_key))
    providers.append(SyntheticDataProvider())
    data_provider = MultiSourceDataProvider(providers)
    
    # Set up components
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = CurrencyStrengthAnalyzer()
    
    # Set up optimized components
    optimized_detector = OptimizedConfluenceDetector(
        correlation_service=correlation_service,
        currency_strength_analyzer=currency_strength_analyzer,
        correlation_threshold=0.7,
        lookback_periods=20,
        cache_ttl_minutes=60,
        max_workers=4
    )
    
    # Set up ML components if requested
    ml_detector = None
    if use_ml:
        model_manager = ModelManager(
            model_dir="models",
            use_gpu=use_gpu,
            correlation_service=correlation_service,
            currency_strength_analyzer=currency_strength_analyzer
        )
        
        ml_detector = model_manager.load_ml_confluence_detector()
    
    # Set up GPU accelerator if requested
    gpu_accelerator = None
    if use_gpu:
        gpu_accelerator = GPUAccelerator(
            enable_gpu=True,
            memory_limit_mb=1024,
            batch_size=1000
        )
    
    # Set up tracer
    tracer = DistributedTracer(
        service_name="real-market-data-test",
        enable_tracing=True,
        sampling_rate=1.0
    )
    
    # Fetch price data for all symbols and timeframes
    logger.info("Fetching price data...")
    price_data = {}
    for symbol in tqdm(symbols, desc="Symbols"):
        price_data[symbol] = {}
        for timeframe in tqdm(timeframes, desc=f"Timeframes for {symbol}", leave=False):
            try:
                df = await data_provider.get_price_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                price_data[symbol][timeframe] = df
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
    
    # Run tests
    logger.info("Running tests...")
    results = {
        "optimized": {
            "confluence": {},
            "divergence": {}
        }
    }
    
    if use_ml:
        results["ml"] = {
            "confluence": {},
            "divergence": {}
        }
    
    # Test signal types and directions
    signal_types = ["trend", "reversal", "breakout"]
    signal_directions = ["bullish", "bearish"]
    
    # Run confluence detection tests
    for symbol in tqdm(symbols, desc="Testing confluence detection"):
        for timeframe in timeframes:
            if symbol not in price_data or timeframe not in price_data[symbol]:
                continue
            
            df = price_data[symbol][timeframe]
            
            # Get related pairs
            related_pairs = await optimized_detector.find_related_pairs(symbol)
            
            # Prepare price data for all pairs
            all_price_data = {}
            for pair in [symbol] + list(related_pairs.keys()):
                if pair in price_data and timeframe in price_data[pair]:
                    all_price_data[pair] = price_data[pair][timeframe]
            
            for signal_type in signal_types:
                for signal_direction in signal_directions:
                    # Test optimized confluence detector
                    with tracer.start_span("optimized_confluence_detection") as span:
                        span.set_attribute("symbol", symbol)
                        span.set_attribute("timeframe", timeframe)
                        span.set_attribute("signal_type", signal_type)
                        span.set_attribute("signal_direction", signal_direction)
                        
                        try:
                            start_time = time.time()
                            result = optimized_detector.detect_confluence_optimized(
                                symbol=symbol,
                                price_data=all_price_data,
                                signal_type=signal_type,
                                signal_direction=signal_direction,
                                related_pairs=related_pairs
                            )
                            execution_time = time.time() - start_time
                            
                            key = f"{symbol}_{timeframe}_{signal_type}_{signal_direction}"
                            results["optimized"]["confluence"][key] = {
                                "execution_time": execution_time,
                                "confluence_score": result["confluence_score"],
                                "confirmation_count": result["confirmation_count"],
                                "contradiction_count": result["contradiction_count"]
                            }
                        except Exception as e:
                            logger.error(f"Error in optimized confluence detection: {e}")
                    
                    # Test ML confluence detector if requested
                    if use_ml and ml_detector:
                        with tracer.start_span("ml_confluence_detection") as span:
                            span.set_attribute("symbol", symbol)
                            span.set_attribute("timeframe", timeframe)
                            span.set_attribute("signal_type", signal_type)
                            span.set_attribute("signal_direction", signal_direction)
                            
                            try:
                                start_time = time.time()
                                result = ml_detector.detect_confluence_ml(
                                    symbol=symbol,
                                    price_data=all_price_data,
                                    signal_type=signal_type,
                                    signal_direction=signal_direction,
                                    related_pairs=related_pairs
                                )
                                execution_time = time.time() - start_time
                                
                                key = f"{symbol}_{timeframe}_{signal_type}_{signal_direction}"
                                results["ml"]["confluence"][key] = {
                                    "execution_time": execution_time,
                                    "confluence_score": result["confluence_score"],
                                    "confirmation_count": result["confirmation_count"],
                                    "contradiction_count": result["contradiction_count"],
                                    "pattern_score": result["pattern_score"],
                                    "prediction_score": result["prediction_score"]
                                }
                            except Exception as e:
                                logger.error(f"Error in ML confluence detection: {e}")
    
    # Run divergence analysis tests
    for symbol in tqdm(symbols, desc="Testing divergence analysis"):
        for timeframe in timeframes:
            if symbol not in price_data or timeframe not in price_data[symbol]:
                continue
            
            df = price_data[symbol][timeframe]
            
            # Get related pairs
            related_pairs = await optimized_detector.find_related_pairs(symbol)
            
            # Prepare price data for all pairs
            all_price_data = {}
            for pair in [symbol] + list(related_pairs.keys()):
                if pair in price_data and timeframe in price_data[pair]:
                    all_price_data[pair] = price_data[pair][timeframe]
            
            # Test optimized divergence analyzer
            with tracer.start_span("optimized_divergence_analysis") as span:
                span.set_attribute("symbol", symbol)
                span.set_attribute("timeframe", timeframe)
                
                try:
                    start_time = time.time()
                    result = optimized_detector.analyze_divergence_optimized(
                        symbol=symbol,
                        price_data=all_price_data,
                        related_pairs=related_pairs
                    )
                    execution_time = time.time() - start_time
                    
                    key = f"{symbol}_{timeframe}"
                    results["optimized"]["divergence"][key] = {
                        "execution_time": execution_time,
                        "divergence_score": result["divergence_score"],
                        "divergences_found": result["divergences_found"]
                    }
                except Exception as e:
                    logger.error(f"Error in optimized divergence analysis: {e}")
            
            # Test ML divergence analyzer if requested
            if use_ml and ml_detector:
                with tracer.start_span("ml_divergence_analysis") as span:
                    span.set_attribute("symbol", symbol)
                    span.set_attribute("timeframe", timeframe)
                    
                    try:
                        start_time = time.time()
                        result = ml_detector.analyze_divergence_ml(
                            symbol=symbol,
                            price_data=all_price_data,
                            related_pairs=related_pairs
                        )
                        execution_time = time.time() - start_time
                        
                        key = f"{symbol}_{timeframe}"
                        results["ml"]["divergence"][key] = {
                            "execution_time": execution_time,
                            "divergence_score": result["divergence_score"],
                            "divergences_found": result["divergences_found"]
                        }
                    except Exception as e:
                        logger.error(f"Error in ML divergence analysis: {e}")
    
    # Save results
    logger.info("Saving results...")
    with open(os.path.join(output_dir, "real_market_data_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    logger.info("Generating plots...")
    
    # Plot confluence detection execution times
    plt.figure(figsize=(12, 6))
    
    optimized_times = [v["execution_time"] for v in results["optimized"]["confluence"].values()]
    plt.hist(optimized_times, bins=20, alpha=0.7, label="Optimized")
    
    if use_ml:
        ml_times = [v["execution_time"] for v in results["ml"]["confluence"].values()]
        plt.hist(ml_times, bins=20, alpha=0.7, label="ML")
    
    plt.title("Confluence Detection Execution Times")
    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "confluence_execution_times.png"))
    
    # Plot divergence analysis execution times
    plt.figure(figsize=(12, 6))
    
    optimized_times = [v["execution_time"] for v in results["optimized"]["divergence"].values()]
    plt.hist(optimized_times, bins=20, alpha=0.7, label="Optimized")
    
    if use_ml:
        ml_times = [v["execution_time"] for v in results["ml"]["divergence"].values()]
        plt.hist(ml_times, bins=20, alpha=0.7, label="ML")
    
    plt.title("Divergence Analysis Execution Times")
    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "divergence_execution_times.png"))
    
    logger.info("Test completed successfully!")

# Mock correlation service for testing
class MockCorrelationService:
    """Mock correlation service for testing."""
    
    async def get_all_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Get all correlations between pairs.
        
        Returns:
            Dictionary mapping pairs to dictionaries of correlations
        """
        return {
            "EURUSD": {
                "GBPUSD": 0.85,
                "AUDUSD": 0.75,
                "USDCAD": -0.65,
                "USDJPY": -0.55,
                "EURGBP": 0.62,
                "EURJPY": 0.78
            },
            "GBPUSD": {
                "EURUSD": 0.85,
                "AUDUSD": 0.70,
                "USDCAD": -0.60,
                "USDJPY": -0.50,
                "EURGBP": -0.58,
                "GBPJPY": 0.75
            },
            "USDJPY": {
                "EURUSD": -0.55,
                "GBPUSD": -0.50,
                "AUDUSD": -0.45,
                "USDCAD": 0.40,
                "EURJPY": 0.65,
                "GBPJPY": 0.70
            },
            "AUDUSD": {
                "EURUSD": 0.75,
                "GBPUSD": 0.70,
                "USDCAD": -0.55,
                "USDJPY": -0.45
            },
            "USDCAD": {
                "EURUSD": -0.65,
                "GBPUSD": -0.60,
                "AUDUSD": -0.55,
                "USDJPY": 0.40
            },
            "EURGBP": {
                "EURUSD": 0.62,
                "GBPUSD": -0.58,
                "EURJPY": 0.55,
                "GBPJPY": -0.50
            },
            "EURJPY": {
                "EURUSD": 0.78,
                "USDJPY": 0.65,
                "EURGBP": 0.55,
                "GBPJPY": 0.60
            },
            "GBPJPY": {
                "GBPUSD": 0.75,
                "USDJPY": 0.70,
                "EURGBP": -0.50,
                "EURJPY": 0.60
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test optimized components with real market data")
    parser.add_argument("--symbols", type=str, default="EURUSD,GBPUSD,USDJPY,AUDUSD", help="Comma-separated list of symbols")
    parser.add_argument("--timeframes", type=str, default="H1,H4,D1", help="Comma-separated list of timeframes")
    parser.add_argument("--days", type=int, default=30, help="Number of days of data to fetch")
    parser.add_argument("--api-key", type=str, help="Alpha Vantage API key")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--use-ml", action="store_true", help="Use machine learning models")
    parser.add_argument("--output-dir", type=str, default="performance_results", help="Output directory")
    args = parser.parse_args()
    
    # Parse arguments
    symbols = args.symbols.split(",")
    timeframes = args.timeframes.split(",")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Run test
    asyncio.run(run_test(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        alpha_vantage_api_key=args.api_key,
        use_gpu=args.use_gpu,
        use_ml=args.use_ml,
        output_dir=args.output_dir
    ))
