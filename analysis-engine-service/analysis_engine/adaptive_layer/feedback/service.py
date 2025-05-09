"""
Timeframe Feedback Service

This module implements the main TimeframeFeedbackService class that coordinates
feedback collection, analysis, and adjustment.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from core_foundations.models.feedback import TradeFeedback
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.exceptions.feedback_exceptions import TimeframeFeedbackError

from .models import TimeframeCorrelation, TimeframeAdjustment, TimeframeInsight
from .collectors.timeframe_collector import TimeframeFeedbackCollector
from .analyzers.correlation import TimeframeCorrelationAnalyzer
from .analyzers.temporal import TemporalFeedbackAnalyzer
from .processors.adjustment import TimeframeAdjustmentProcessor

logger = get_logger(__name__)


class TimeframeFeedbackService:
    """
    Service for collecting, analyzing, and applying timeframe-specific feedback.
    
    This service coordinates the collection of feedback from various sources,
    analyzes patterns and correlations between timeframes, and calculates
    adjustments to improve prediction accuracy.
    """
    
    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        analysis_interval_hours: int = 6,
        max_items_per_timeframe: int = 1000,
        min_sample_size: int = 20
    ):
        """
        Initialize the timeframe feedback service.
        
        Args:
            event_publisher: Event publisher for broadcasting events
            analysis_interval_hours: Interval for automatic analysis
            max_items_per_timeframe: Maximum items to keep per timeframe
            min_sample_size: Minimum sample size for analysis
        """
        self.event_publisher = event_publisher
        self.analysis_interval_hours = analysis_interval_hours
        
        # Initialize components
        self.collector = TimeframeFeedbackCollector(
            event_publisher=event_publisher,
            max_items_per_timeframe=max_items_per_timeframe
        )
        
        self.correlation_analyzer = TimeframeCorrelationAnalyzer(
            min_sample_size=min_sample_size
        )
        
        self.temporal_analyzer = TemporalFeedbackAnalyzer(
            min_sample_size=min_sample_size
        )
        
        self.adjustment_processor = TimeframeAdjustmentProcessor()
        
        # Store analysis results
        self.correlations: Dict[str, List[TimeframeCorrelation]] = {}
        self.insights: Dict[str, TimeframeInsight] = {}
        self.adjustments: Dict[str, Dict[str, TimeframeAdjustment]] = {}
        
        # Track last analysis time
        self.last_analysis_time: Dict[str, datetime] = {}
        
        # Background task
        self._analysis_task = None
        self._running = False
        
        logger.info("TimeframeFeedbackService initialized")
    
    async def start(self) -> None:
        """Start the feedback service."""
        if self._running:
            logger.warning("TimeframeFeedbackService is already running")
            return
        
        self._running = True
        self._analysis_task = asyncio.create_task(self._periodic_analysis())
        logger.info("TimeframeFeedbackService started")
    
    async def stop(self) -> None:
        """Stop the feedback service."""
        if not self._running:
            logger.warning("TimeframeFeedbackService is already stopped")
            return
        
        self._running = False
        
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
            
        logger.info("TimeframeFeedbackService stopped")
    
    async def collect_feedback(self, feedback: TradeFeedback) -> str:
        """
        Collect timeframe-specific feedback.
        
        Args:
            feedback: The feedback to collect
            
        Returns:
            str: The feedback ID
        """
        return await self.collector.collect_feedback(feedback)
    
    async def analyze_instrument(self, instrument: str) -> Dict[str, Any]:
        """
        Analyze feedback for a specific instrument.
        
        Args:
            instrument: The instrument to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Get feedback data by timeframe
            feedback_data = await self.collector.get_feedback_by_timeframe(
                instrument=instrument
            )
            
            if feedback_data["status"] != "success":
                return {
                    "instrument": instrument,
                    "status": feedback_data["status"],
                    "message": "No feedback data available for analysis"
                }
            
            timeframe_data = {}
            for timeframe, data in feedback_data["timeframes"].items():
                timeframe_data[timeframe] = data["feedback"]
            
            # Analyze correlations
            correlations = self.correlation_analyzer.analyze_correlations(
                instrument=instrument,
                timeframe_data=timeframe_data
            )
            
            # Analyze temporal patterns
            insights = self.temporal_analyzer.analyze_temporal_patterns(
                instrument=instrument,
                timeframe_data=timeframe_data
            )
            
            # Calculate adjustments
            adjustments = self.adjustment_processor.calculate_adjustments(
                instrument=instrument,
                correlations=correlations,
                insights=insights
            )
            
            # Store results
            self.correlations[instrument] = correlations
            self.insights[instrument] = insights
            self.adjustments[instrument] = adjustments
            self.last_analysis_time[instrument] = datetime.utcnow()
            
            # Prepare response
            result = {
                "instrument": instrument,
                "status": "success",
                "analysis_time": datetime.utcnow().isoformat(),
                "timeframes_analyzed": list(timeframe_data.keys()),
                "correlations": [c.to_dict() for c in correlations],
                "insights": insights.to_dict(),
                "adjustments": {tf: adj.to_dict() for tf, adj in adjustments.items()}
            }
            
            # Publish event if available
            if self.event_publisher:
                await self.event_publisher.publish(
                    "feedback.timeframe.analyzed",
                    {
                        "instrument": instrument,
                        "analysis_time": result["analysis_time"],
                        "timeframes_analyzed": result["timeframes_analyzed"]
                    }
                )
            
            logger.info(f"Completed analysis for {instrument} with {len(timeframe_data)} timeframes")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing instrument {instrument}: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to analyze instrument {instrument}: {str(e)}")
    
    async def get_adjustment(
        self, instrument: str, timeframe: str
    ) -> Optional[TimeframeAdjustment]:
        """
        Get adjustment for a specific instrument and timeframe.
        
        Args:
            instrument: The instrument
            timeframe: The timeframe
            
        Returns:
            Optional[TimeframeAdjustment]: Adjustment or None if not available
        """
        # Check if we have adjustments for this instrument
        if instrument not in self.adjustments:
            # Try to analyze if we haven't yet
            if instrument not in self.last_analysis_time:
                try:
                    await self.analyze_instrument(instrument)
                except:
                    return None
            
            # Still not available
            if instrument not in self.adjustments:
                return None
        
        # Check if we have adjustment for this timeframe
        if timeframe not in self.adjustments[instrument]:
            return None
        
        return self.adjustments[instrument][timeframe]
    
    async def get_insights(self, instrument: str) -> Optional[TimeframeInsight]:
        """
        Get insights for a specific instrument.
        
        Args:
            instrument: The instrument
            
        Returns:
            Optional[TimeframeInsight]: Insights or None if not available
        """
        # Check if we have insights for this instrument
        if instrument not in self.insights:
            # Try to analyze if we haven't yet
            if instrument not in self.last_analysis_time:
                try:
                    await self.analyze_instrument(instrument)
                except:
                    return None
            
            # Still not available
            if instrument not in self.insights:
                return None
        
        return self.insights[instrument]
    
    async def get_correlations(self, instrument: str) -> List[TimeframeCorrelation]:
        """
        Get correlations for a specific instrument.
        
        Args:
            instrument: The instrument
            
        Returns:
            List[TimeframeCorrelation]: Correlations
        """
        # Check if we have correlations for this instrument
        if instrument not in self.correlations:
            # Try to analyze if we haven't yet
            if instrument not in self.last_analysis_time:
                try:
                    await self.analyze_instrument(instrument)
                except:
                    return []
            
            # Still not available
            if instrument not in self.correlations:
                return []
        
        return self.correlations[instrument]
    
    async def get_analysis_status(self) -> Dict[str, Any]:
        """
        Get status of feedback analysis.
        
        Returns:
            Dict[str, Any]: Analysis status
        """
        instruments = set(self.last_analysis_time.keys())
        instruments.update(self.correlations.keys())
        instruments.update(self.insights.keys())
        instruments.update(self.adjustments.keys())
        
        status = {
            "instruments_analyzed": len(instruments),
            "instruments": [],
            "last_analysis": None
        }
        
        for instrument in instruments:
            instrument_status = {
                "instrument": instrument,
                "last_analysis": self.last_analysis_time.get(instrument, None)
            }
            
            if instrument_status["last_analysis"]:
                instrument_status["last_analysis"] = instrument_status["last_analysis"].isoformat()
                
                # Update overall last analysis
                if status["last_analysis"] is None or instrument_status["last_analysis"] > status["last_analysis"]:
                    status["last_analysis"] = instrument_status["last_analysis"]
            
            if instrument in self.correlations:
                instrument_status["correlations_count"] = len(self.correlations[instrument])
            
            if instrument in self.insights:
                instrument_status["timeframes_analyzed"] = self.insights[instrument].timeframes_analyzed
                instrument_status["recommendations_count"] = len(self.insights[instrument].recommendations)
            
            if instrument in self.adjustments:
                instrument_status["adjustments_count"] = len(self.adjustments[instrument])
            
            status["instruments"].append(instrument_status)
        
        return status
    
    async def _periodic_analysis(self) -> None:
        """Periodically analyze feedback for all instruments."""
        while self._running:
            try:
                # Get all instruments with feedback
                instruments = set()
                
                for instrument in self.collector.recent_feedback.keys():
                    instruments.add(instrument)
                
                # Analyze each instrument if needed
                for instrument in instruments:
                    # Skip if analyzed recently
                    last_analysis = self.last_analysis_time.get(instrument)
                    if last_analysis and (datetime.utcnow() - last_analysis).total_seconds() < self.analysis_interval_hours * 3600:
                        continue
                    
                    try:
                        await self.analyze_instrument(instrument)
                        logger.info(f"Periodic analysis completed for {instrument}")
                    except Exception as e:
                        logger.error(f"Error in periodic analysis for {instrument}: {str(e)}")
                
                # Clean up old feedback
                self.collector.clear_old_feedback()
                
            except Exception as e:
                logger.error(f"Error in periodic analysis task: {str(e)}")
            
            # Sleep until next analysis
            try:
                await asyncio.sleep(60 * 60)  # Check every hour
            except asyncio.CancelledError:
                break