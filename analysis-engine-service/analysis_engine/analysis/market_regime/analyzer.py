"""
Market Regime Analyzer

This module provides the main analyzer class that coordinates detection
and classification of market regimes.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from analysis_engine.caching.cache_service import cache_result
from analysis_engine.repositories.price_repository import PriceRepository
from analysis_engine.events.publisher import EventPublisher
from analysis_engine.events.schemas import MarketRegimeChangeEvent, MarketRegimeChangePayload
from analysis_engine.config import settings

from .models import MarketRegimeResult, RegimeChangeResult
from .detector import MarketRegimeDetector
from .classifier import MarketRegimeClassifier

logger = logging.getLogger(__name__)

# Initialize Event Publisher
try:
    event_publisher = EventPublisher()
except Exception as e:
    logger.error(f"Failed to initialize EventPublisher in MarketRegimeAnalyzer: {e}", exc_info=True)
    event_publisher = None


class MarketRegimeAnalyzer:
    """
    Analyzer for market regimes.
    
    This class coordinates the detection and classification of market regimes
    and provides methods for analyzing regime changes.
    """
    
    def __init__(self, db: Session):
        """
        Initialize the market regime analyzer.
        
        Args:
            db: Database session
        """
        price_repository = PriceRepository(db)
        self.detector = MarketRegimeDetector(price_repository)
        self.classifier = MarketRegimeClassifier()
        logger.info("MarketRegimeAnalyzer initialized")
    
    @cache_result(ttl=900)  # Cache for 15 minutes
    async def detect_regime(
        self,
        instrument: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> MarketRegimeResult:
        """
        Detect market regime for a specific instrument and timeframe.
        
        Args:
            instrument: The instrument to analyze
            timeframe: The timeframe to analyze
            from_date: Start date for analysis
            to_date: End date for analysis
            
        Returns:
            MarketRegimeResult: Detected regime
        """
        # Default date range if not provided
        if to_date is None:
            to_date = datetime.utcnow()
        if from_date is None:
            from_date = to_date - timedelta(days=30)  # Default to 30 days
        
        # Get price data
        prices = await self.detector.get_price_data(
            instrument=instrument,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date
        )
        
        if not prices or len(prices) < 20:
            return MarketRegimeResult(
                instrument=instrument,
                timeframe=timeframe,
                regime="unknown",
                direction="neutral",
                volatility="unknown",
                strength=0,
                detected_at=datetime.utcnow()
            )
        
        # Extract features
        features = self.detector.extract_features(prices)
        
        # Classify regime
        return self.classifier.classify_regime(
            instrument=instrument,
            timeframe=timeframe,
            features=features
        )
    
    async def detect_regime_change(
        self,
        instrument: str,
        timeframe: str,
        days_back: int = 60
    ) -> RegimeChangeResult:
        """
        Detect if there has been a market regime change
        by comparing current and previous periods.
        
        Args:
            instrument: The instrument to analyze
            timeframe: The timeframe to analyze
            days_back: Number of days to look back
            
        Returns:
            RegimeChangeResult: Detected regime change
        """
        now = datetime.utcnow()
        mid_point = now - timedelta(days=days_back//2)
        start_date = now - timedelta(days=days_back)
        
        # Get current regime (last half of period)
        current_regime = await self.detect_regime(
            instrument=instrument,
            timeframe=timeframe,
            from_date=mid_point,
            to_date=now
        )
        
        # Get previous regime (first half of period)
        previous_regime = await self.detect_regime(
            instrument=instrument,
            timeframe=timeframe,
            from_date=start_date,
            to_date=mid_point
        )
        
        # Classify change
        change_result = self.classifier.classify_regime_change(
            current_regime=current_regime,
            previous_regime=previous_regime
        )
        
        # Create result
        result = RegimeChangeResult(
            instrument=instrument,
            timeframe=timeframe,
            period_days=days_back,
            regime_changed=change_result["regime_changed"],
            direction_changed=change_result["direction_changed"],
            volatility_changed=change_result["volatility_changed"],
            change_significance=change_result["change_significance"],
            current_regime=current_regime,
            previous_regime=previous_regime,
            analyzed_at=now
        )
        
        # Publish event if regime changed
        if result.regime_changed and event_publisher:
            try:
                payload = MarketRegimeChangePayload(
                    symbol=instrument,
                    timeframe=timeframe,
                    previous_regime=previous_regime.regime.value,
                    current_regime=current_regime.regime.value,
                    confidence_score=current_regime.strength,
                    detection_timestamp=current_regime.detected_at
                )
                event = MarketRegimeChangeEvent(payload=payload)
                event_publisher.publish(topic=settings.KAFKA_REGIME_TOPIC, event=event)
            except Exception as e:
                logger.error(f"Failed to publish MarketRegimeChangeEvent: {e}", exc_info=True)
        
        return result