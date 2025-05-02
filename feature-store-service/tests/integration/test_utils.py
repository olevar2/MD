"""
Test utilities and fixtures for integration testing.

Provides common test data generators, mock components, and helper functions
for integration testing of the feature store service.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

class MarketDataGenerator:
    """Generates realistic market data for testing."""
    
    @staticmethod
    def generate_ohlcv(
        periods: int = 1000,
        frequency: str = '1min',
        volatility: float = 0.01,
        trend: float = 0.0,
        gaps: bool = False,
        missing_values: bool = False
    ) -> pd.DataFrame:
        """
        Generate OHLCV data with realistic properties.
        
        Args:
            periods: Number of periods to generate
            frequency: Time frequency for the data
            volatility: Price volatility factor
            trend: Price trend factor (-1.0 to 1.0)
            gaps: Whether to include market gaps
            missing_values: Whether to include missing values
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate timestamps
        timestamps = pd.date_range(
            start='2025-01-01',
            periods=periods,
            freq=frequency
        )
        
        # Add gaps if requested
        if gaps:
            # Remove some random periods to simulate gaps
            gap_indices = random.sample(
                range(len(timestamps)),
                k=int(len(timestamps) * 0.05)  # 5% gaps
            )
            timestamps = timestamps.delete(gap_indices)
            
        # Generate price movement
        returns = np.random.normal(
            loc=trend * volatility,
            scale=volatility,
            size=len(timestamps)
        )
        price = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': price * (1 + np.random.normal(0, volatility/2, len(price))),
            'high': price * (1 + abs(np.random.normal(0, volatility, len(price)))),
            'low': price * (1 - abs(np.random.normal(0, volatility, len(price)))),
            'close': price,
            'volume': np.random.lognormal(10, 1, len(price))
        })
        
        # Ensure high/low consistency
        data['high'] = np.maximum.reduce([
            data['high'],
            data['open'],
            data['close']
        ])
        data['low'] = np.minimum.reduce([
            data['low'],
            data['open'],
            data['close']
        ])
        
        # Add missing values if requested
        if missing_values:
            # Randomly set some values to NaN
            mask = np.random.random(size=data.shape) < 0.02  # 2% missing
            data.mask(mask, np.nan, inplace=True)
            
        return data

    @staticmethod
    def generate_regime_changes(
        data: pd.DataFrame,
        n_regimes: int = 3
    ) -> pd.DataFrame:
        """
        Add market regime labels to OHLCV data.
        
        Args:
            data: OHLCV DataFrame
            n_regimes: Number of different regimes
            
        Returns:
            DataFrame with added regime column
        """
        # Calculate some basic features
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        trend = returns.rolling(50).mean()
        
        # Create regime labels based on volatility and trend
        conditions = [
            (volatility > volatility.quantile(0.7)) & (trend > 0),  # Volatile bullish
            (volatility > volatility.quantile(0.7)) & (trend < 0),  # Volatile bearish
            (volatility <= volatility.quantile(0.7)) & (trend > 0),  # Stable bullish
            (volatility <= volatility.quantile(0.7)) & (trend < 0),  # Stable bearish
        ]
        regime_labels = ['volatile_bull', 'volatile_bear', 'stable_bull', 'stable_bear']
        
        result = data.copy()
        result['regime'] = np.select(
            conditions,
            regime_labels,
            default='neutral'
        )
        
        return result


class IndicatorTestData:
    """Provides test data for different types of indicators."""
    
    @staticmethod
    def generate_momentum_data(
        periods: int = 100,
        trend_strength: float = 0.6
    ) -> pd.DataFrame:
        """Generate data suitable for momentum indicators."""
        # Create trending data
        data = MarketDataGenerator.generate_ohlcv(
            periods=periods,
            trend=trend_strength,
            volatility=0.02
        )
        return data

    @staticmethod
    def generate_volatility_data(
        periods: int = 100,
        volatility_clusters: bool = True
    ) -> pd.DataFrame:
        """Generate data suitable for volatility indicators."""
        if volatility_clusters:
            # Create data with volatility clustering
            volatilities = np.concatenate([
                np.random.uniform(0.01, 0.02, periods // 2),
                np.random.uniform(0.03, 0.05, periods // 2)
            ])
            np.random.shuffle(volatilities)
        else:
            volatilities = np.random.uniform(0.01, 0.03, periods)
            
        data = pd.DataFrame()
        for i in range(periods):
            data_chunk = MarketDataGenerator.generate_ohlcv(
                periods=1,
                volatility=volatilities[i]
            )
            data = pd.concat([data, data_chunk])
            
        return data.reset_index(drop=True)

    @staticmethod
    def generate_oscillator_data(
        periods: int = 100,
        cycle_length: int = 20
    ) -> pd.DataFrame:
        """Generate data suitable for oscillator indicators."""
        # Create cyclic price movements
        t = np.linspace(0, 4*np.pi, periods)
        cycle = np.sin(t * (2*np.pi/cycle_length))
        
        # Add noise
        noise = np.random.normal(0, 0.1, periods)
        price = 100 * (1 + cycle + noise)
        
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=periods),
            'open': price * (1 + np.random.normal(0, 0.005, periods)),
            'high': price * (1 + abs(np.random.normal(0, 0.01, periods))),
            'low': price * (1 - abs(np.random.normal(0, 0.01, periods))),
            'close': price,
            'volume': np.random.lognormal(10, 1, periods)
        })
        
        # Ensure high/low consistency
        data['high'] = np.maximum.reduce([
            data['high'],
            data['open'],
            data['close']
        ])
        data['low'] = np.minimum.reduce([
            data['low'],
            data['open'],
            data['close']
        ])
        
        return data


def create_test_environment() -> Dict[str, Any]:
    """
    Create a complete test environment with all necessary components.
    
    Returns:
        Dictionary containing test environment components
    """
    import tempfile
    from feature_store_service.validation.data_validator import DataValidationService
    from feature_store_service.error.error_manager import IndicatorErrorManager
    from feature_store_service.error.recovery_service import ErrorRecoveryService
    from feature_store_service.error.monitoring_service import ErrorMonitoringService
    from feature_store_service.optimization.resource_manager import AdaptiveResourceManager
    from feature_store_service.optimization.performance_optimizer import (
        PerformanceOptimizer,
        PerformanceMonitor
    )
    from feature_store_service.logging.indicator_logging import (
        IndicatorLogger,
        IndicatorReport
    )
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    log_dir = Path(temp_dir) / "logs"
    cache_dir = Path(temp_dir) / "cache"
    profile_dir = Path(temp_dir) / "profiles"
    
    # Initialize components
    components = {
        'temp_dir': temp_dir,
        'validator': DataValidationService(),
        'error_manager': IndicatorErrorManager(),
        'recovery_service': ErrorRecoveryService(),
        'monitoring_service': ErrorMonitoringService(
            storage_dir=str(temp_dir / "monitoring")
        ),
        'resource_manager': AdaptiveResourceManager(
            cache_dir=str(cache_dir)
        ),
        'performance_optimizer': PerformanceOptimizer(
            profile_dir=str(profile_dir)
        ),
        'logger': IndicatorLogger(log_dir=str(log_dir)),
        'reporter': IndicatorReport(log_dir=str(log_dir))
    }
    
    return components


def cleanup_test_environment(components: Dict[str, Any]) -> None:
    """
    Clean up test environment.
    
    Args:
        components: Dictionary of test environment components
    """
    import shutil
    
    # Clean up temporary directory
    if 'temp_dir' in components:
        shutil.rmtree(components['temp_dir'])
        
    # Clean up any other resources
    if 'resource_manager' in components:
        components['resource_manager'].cleanup()
