"""
Market Regime Service for Trading Gateway.

This service provides market regime detection and analysis for trading algorithms.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from ..error import (
    async_with_exception_handling,
    MarketDataError,
    ServiceUnavailableError
)

class MarketRegimeService:
    """
    Service for detecting and analyzing market regimes.
    
    This service uses the standardized Analysis Engine API to detect market regimes
    and provide regime-specific analysis for trading algorithms.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the market regime service.
        
        Args:
            logger: Logger instance
            config: Service configuration
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Cache for market regimes
        self.regime_cache: Dict[str, str] = {}
        self.last_regime_update: Dict[str, float] = {}
        
        # Cache expiration (in seconds)
        self.regime_cache_ttl = self.config.get('regime_cache_ttl', 3600)  # 1 hour
        
        # Analysis Engine URL
        self.analysis_engine_url = self.config.get(
            'analysis_engine_url', 
            'http://analysis-engine-service:8000'
        )
    
    @async_with_exception_handling
    async def detect_regime(
        self,
        instrument: str,
        timeframe: str = '1h',
        ohlc_data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Detect the current market regime for an instrument.
        
        Args:
            instrument: The instrument to detect regime for
            timeframe: Timeframe for analysis
            ohlc_data: Optional OHLC data to use for detection
            
        Returns:
            Detected market regime
            
        Raises:
            MarketDataError: If there's an error detecting market regime
        """
        if not instrument:
            raise MarketDataError(
                message="Instrument cannot be empty",
                symbol=instrument
            )
        
        # Check cache
        now = time.time()
        cache_key = f"{instrument}_{timeframe}"
        if (cache_key in self.regime_cache and
            cache_key in self.last_regime_update and
            now - self.last_regime_update[cache_key] < self.regime_cache_ttl):
            return self.regime_cache[cache_key]
        
        try:
            # Use standardized client to get market regime
            try:
                from analysis_engine.clients.standardized import get_client_factory
                client = get_client_factory().get_market_regime_client()
                
                # Detect market regime
                if ohlc_data is not None:
                    result = await client.detect_market_regime(
                        symbol=instrument,
                        timeframe=timeframe,
                        ohlc_data=ohlc_data
                    )
                else:
                    # Get recent price data from market data service
                    from ..services.market_data_service import MarketDataService
                    market_data_service = MarketDataService({})  # Empty adapter dict for now
                    
                    # Get recent price data
                    end_time = datetime.utcnow()
                    start_time = end_time - timedelta(days=7)  # Get 7 days of data
                    
                    ohlc_data = await market_data_service.get_historical_data(
                        instrument=instrument,
                        start_time=start_time,
                        end_time=end_time,
                        timeframe=timeframe
                    )
                    
                    result = await client.detect_market_regime(
                        symbol=instrument,
                        timeframe=timeframe,
                        ohlc_data=ohlc_data
                    )
                
                # Update cache
                regime = result.get("regime", "unknown")
                self.regime_cache[cache_key] = regime
                self.last_regime_update[cache_key] = now
                
                return regime
            except ImportError:
                # Fall back to legacy implementation if standardized client is not available
                self.logger.warning("Standardized client not available, using legacy implementation")
                return await self._detect_regime_legacy(instrument, timeframe, ohlc_data)
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            raise MarketDataError(
                message=f"Failed to detect market regime for {instrument}: {str(e)}",
                symbol=instrument,
                details={"error": str(e)}
            )
    
    async def _detect_regime_legacy(
        self,
        instrument: str,
        timeframe: str,
        ohlc_data: Optional[pd.DataFrame]
    ) -> str:
        """
        Legacy implementation of market regime detection.
        
        Args:
            instrument: The instrument to detect regime for
            timeframe: Timeframe for analysis
            ohlc_data: Optional OHLC data to use for detection
            
        Returns:
            Detected market regime
        """
        # Simple regime detection based on volatility and trend
        if ohlc_data is None or len(ohlc_data) < 2:
            return "unknown"
        
        # Calculate returns
        ohlc_data['returns'] = ohlc_data['close'].pct_change().dropna()
        
        # Calculate volatility (standard deviation of returns)
        volatility = ohlc_data['returns'].std()
        
        # Calculate trend strength (absolute mean of returns)
        trend_strength = abs(ohlc_data['returns'].mean())
        
        # Determine regime based on volatility and trend strength
        if volatility > 0.002:  # High volatility
            return 'volatile'
        elif trend_strength > 0.0001:  # Strong trend
            return 'trending'
        else:  # Low volatility, weak trend
            return 'ranging'
