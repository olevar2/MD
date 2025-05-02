"""
Market Regime Detection

This module provides functionality to detect and classify market regimes
based on price action, volatility, and trend characteristics.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session

# Assuming we have a PriceRepository to fetch price data
# This would need to be created or integrated with an existing component
from analysis_engine.repositories.price_repository import PriceRepository
from analysis_engine.caching.cache_service import cache_result # Added import

# --- Add imports for event publishing ---
import logging
from analysis_engine.config import settings
from analysis_engine.events.publisher import EventPublisher
from analysis_engine.events.schemas import MarketRegimeChangeEvent, MarketRegimeChangePayload
# --- End Add imports ---

# --- Initialize Logger and Event Publisher ---
logger = logging.getLogger(__name__)
try:
    event_publisher = EventPublisher()
except Exception as e:
    logger.error(f"Failed to initialize EventPublisher in MarketRegimeAnalyzer: {e}", exc_info=True)
    event_publisher = None
# --- End Initialize ---

class MarketRegimeAnalyzer:
    """
    Detector for market regimes based on price action analysis
    """
    
    def __init__(self, db: Session):
        self.price_repository = PriceRepository(db)
    
    @cache_result(ttl=900) # Cache for 15 minutes
    def detect_regime(
        self,
        instrument: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Detect market regime for a specific instrument and timeframe
        
        Returns a dictionary with:
        - regime: 'trending', 'ranging', or 'volatile'
        - direction: 'bullish', 'bearish', or 'neutral'
        - volatility: 'low', 'medium', or 'high'
        - strength: value between 0-1 indicating strength of classification
        """
        # Default date range if not provided
        if to_date is None:
            to_date = datetime.utcnow()
        if from_date is None:
            from_date = to_date - timedelta(days=30)  # Default to 30 days
        
        # Get price data
        prices = self.price_repository.get_prices(
            instrument=instrument,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date
        )
        
        if not prices or len(prices) < 20:
            return {
                "instrument": instrument,
                "timeframe": timeframe,
                "regime": "unknown",
                "direction": "neutral",
                "volatility": "unknown",
                "strength": 0,
                "detected_at": datetime.utcnow()
            }
        
        # Extract price series
        close_prices = np.array([p.close for p in prices])
        
        # Calculate indicators
        atr = self._calculate_atr(prices)
        adx = self._calculate_adx(prices)
        rsi = self._calculate_rsi(close_prices)
        
        # Determine trend direction
        sma20 = np.mean(close_prices[-20:])
        sma50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else sma20
        
        if close_prices[-1] > sma20 and sma20 > sma50:
            direction = "bullish"
            direction_strength = min(1.0, (close_prices[-1] / sma20 - 1) * 10)
        elif close_prices[-1] < sma20 and sma20 < sma50:
            direction = "bearish"
            direction_strength = min(1.0, (1 - close_prices[-1] / sma20) * 10)
        else:
            direction = "neutral"
            direction_strength = 0.5
        
        # Classify volatility
        avg_range = np.mean([p.high - p.low for p in prices])
        avg_price = np.mean(close_prices)
        volatility_ratio = (avg_range / avg_price) * 100  # As percentage of price
        
        if volatility_ratio < 0.5:
            volatility = "low"
        elif volatility_ratio < 1.2:
            volatility = "medium"
        else:
            volatility = "high"
        
        # Determine regime
        if adx > 25:
            # Strong trend
            regime = "trending"
            strength = min(1.0, adx / 50)
        elif adx < 15 and volatility_ratio < 0.8:
            # Low volatility, no trend
            regime = "ranging"
            strength = min(1.0, (25 - adx) / 25)
        else:
            # High volatility without strong trend
            regime = "volatile"
            strength = min(1.0, volatility_ratio / 2)
        
        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "regime": regime,
            "direction": direction,
            "volatility": volatility,
            "strength": round(strength, 2),
            "metrics": {
                "adx": round(adx, 2),
                "rsi": round(rsi, 2),
                "volatility_ratio": round(volatility_ratio, 2)
            },
            "detected_at": datetime.utcnow()
        }
    
    def _calculate_atr(self, prices: List[Any], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i].high
            low = prices[i].low
            prev_close = prices[i-1].close
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        return np.mean(true_ranges[-period:])
    
    def _calculate_adx(self, prices: List[Any], period: int = 14) -> float:
        """Calculate Average Directional Index"""
        if len(prices) < period * 2:
            return 0.0
        
        # Calculate +DI and -DI
        plus_dm = []
        minus_dm = []
        tr = []
        
        for i in range(1, len(prices)):
            high = prices[i].high
            low = prices[i].low
            prev_high = prices[i-1].high
            prev_low = prices[i-1].low
            prev_close = prices[i-1].close
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr.append(max(tr1, tr2, tr3))
            
            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
                
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
        
        # Smooth with EMA
        tr_ema = self._calculate_ema(tr, period)
        plus_dm_ema = self._calculate_ema(plus_dm, period)
        minus_dm_ema = self._calculate_ema(minus_dm, period)
        
        # Calculate DI
        plus_di = 100 * plus_dm_ema / tr_ema if tr_ema > 0 else 0
        minus_di = 100 * minus_dm_ema / tr_ema if tr_ema > 0 else 0
        
        # Calculate DX
        if plus_di + minus_di > 0:
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        else:
            dx = 0
        
        # Calculate ADX (smoothed DX)
        adx_values = [dx]
        for i in range(1, period):
            if i >= len(prices) - period:
                break
            adx_values.append((adx_values[-1] * (period - 1) + dx) / period)
        
        return adx_values[-1] if adx_values else 0
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) <= period:
            return 50.0  # Default neutral value
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss if avg_loss > 0 else 1000
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, values: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if not values or period <= 0 or len(values) < period:
            return 0.0
            
        ema = sum(values[:period]) / period
        multiplier = 2 / (period + 1)
        
        for i in range(period, len(values)):
            ema = (values[i] - ema) * multiplier + ema
            
        return ema
    
    # No caching needed for detect_regime_change as it calls detect_regime internally,
    # which is already cached.
    def detect_regime_change(
        self,
        instrument: str,
        timeframe: str,
        days_back: int = 60
    ) -> Dict[str, Any]:
        """
        Detect if there has been a market regime change
        by comparing current and previous periods
        """
        now = datetime.utcnow()
        mid_point = now - timedelta(days=days_back//2)
        start_date = now - timedelta(days=days_back)
        
        # Get current regime (last half of period)
        current_regime = self.detect_regime(
            instrument=instrument,
            timeframe=timeframe,
            from_date=mid_point,
            to_date=now
        )
        
        # Get previous regime (first half of period)
        previous_regime = self.detect_regime(
            instrument=instrument,
            timeframe=timeframe,
            from_date=start_date,
            to_date=mid_point
        )
        
        # Determine if there's been a change
        regime_changed = current_regime["regime"] != previous_regime["regime"]
        direction_changed = current_regime["direction"] != previous_regime["direction"]
        volatility_changed = current_regime["volatility"] != previous_regime["volatility"]
        
        # Calculate significance of change
        significance = 0
        if regime_changed:
            significance += 0.6
        if direction_changed:
            significance += 0.3
        if volatility_changed:
            significance += 0.1

        # --- Publish Market Regime Change Event ---
        if regime_changed and event_publisher:
            try:
                payload = MarketRegimeChangePayload(
                    symbol=instrument,
                    timeframe=timeframe,
                    previous_regime=previous_regime["regime"],
                    current_regime=current_regime["regime"],
                    confidence_score=current_regime["strength"], # Use current regime strength as confidence
                    detection_timestamp=current_regime["detected_at"]
                )
                event = MarketRegimeChangeEvent(payload=payload)
                event_publisher.publish(topic=settings.KAFKA_REGIME_TOPIC, event=event)
            except Exception as pub_exc:
                logger.error(f"Failed to publish MarketRegimeChangeEvent for {instrument}/{timeframe}: {pub_exc}", exc_info=True)
        # --- End Publish Event ---

        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "period_days": days_back,
            "regime_changed": regime_changed,
            "direction_changed": direction_changed,
            "volatility_changed": volatility_changed,
            "change_significance": round(significance, 2),
            "current_regime": current_regime,
            "previous_regime": previous_regime,
            "analyzed_at": now
        }
