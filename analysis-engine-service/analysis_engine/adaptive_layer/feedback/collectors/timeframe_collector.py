"""
Timeframe Feedback Collector

This module implements the collection of timeframe-specific feedback.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from core_foundations.models.feedback import TradeFeedback
from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.exceptions.feedback_exceptions import TimeframeFeedbackError

from ..models import extract_timeframe_from_feedback

logger = get_logger(__name__)


class TimeframeFeedbackCollector:
    """Collects and manages timeframe-specific feedback."""
    
    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        max_items_per_timeframe: int = 1000
    ):
        """
        Initialize the timeframe feedback collector.
        
        Args:
            event_publisher: Event publisher for broadcasting events
            max_items_per_timeframe: Maximum items to keep per timeframe
        """
        self.event_publisher = event_publisher
        self.max_items_per_timeframe = max_items_per_timeframe
        
        # Store recent feedback by timeframe
        # Structure: {instrument: {timeframe: [feedback_items]}}
        self.recent_feedback = {}
        
        logger.info("TimeframeFeedbackCollector initialized")
    
    async def collect_feedback(self, feedback: TradeFeedback) -> str:
        """
        Collect timeframe-specific feedback.
        
        Args:
            feedback: The feedback to collect
            
        Returns:
            str: The feedback ID
        """
        try:
            # Generate ID if not provided
            if not feedback.id:
                feedback.id = str(uuid.uuid4())
            
            # Set timestamp if not provided
            if not feedback.timestamp:
                feedback.timestamp = datetime.utcnow().isoformat()
            
            # Extract instrument and timeframe information
            instrument = getattr(feedback, "instrument", None)
            timeframe = extract_timeframe_from_feedback(feedback)
            
            if not instrument:
                logger.warning(f"Missing instrument in feedback {feedback.id}, using 'unknown'")
                instrument = "unknown"
                
            if not timeframe:
                logger.warning(f"Missing timeframe in feedback {feedback.id}, using 'unknown'")
                timeframe = "unknown"
            
            # Initialize data structures if needed
            if instrument not in self.recent_feedback:
                self.recent_feedback[instrument] = {}
            
            if timeframe not in self.recent_feedback[instrument]:
                self.recent_feedback[instrument][timeframe] = []
            
            # Add feedback to appropriate timeframe bucket
            feedback_list = self.recent_feedback[instrument][timeframe]
            feedback_list.append(feedback)
            
            # Trim if needed
            if len(feedback_list) > self.max_items_per_timeframe:
                # Remove oldest items
                feedback_list.sort(key=lambda fb: fb.timestamp)
                self.recent_feedback[instrument][timeframe] = feedback_list[-self.max_items_per_timeframe:]
            
            # Publish event if available
            if self.event_publisher:
                await self.event_publisher.publish(
                    "feedback.timeframe.collected",
                    {
                        "feedback_id": feedback.id,
                        "instrument": instrument,
                        "timeframe": timeframe,
                        "timestamp": feedback.timestamp
                    }
                )
            
            logger.debug(f"Collected {timeframe} feedback for {instrument}")
            
            return feedback.id
            
        except Exception as e:
            logger.error(f"Error collecting timeframe feedback: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to collect timeframe feedback: {str(e)}")
    
    async def get_feedback_by_timeframe(
        self,
        instrument: str,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get feedback data for a specific timeframe.
        
        Args:
            instrument: The instrument to get feedback for
            timeframe: The specific timeframe (or None for all)
            start_time: Start time for filtering
            end_time: End time for filtering
            limit: Maximum items to return per timeframe
            
        Returns:
            Dict[str, Any]: Feedback data by timeframe
        """
        try:
            if instrument not in self.recent_feedback:
                return {"instrument": instrument, "status": "no_data"}
            
            result = {
                "instrument": instrument,
                "status": "success",
                "timeframes": {},
                "query_time": datetime.utcnow().isoformat()
            }
            
            # Determine which timeframes to include
            timeframes_to_check = []
            if timeframe:
                # Specific timeframe requested
                if timeframe in self.recent_feedback[instrument]:
                    timeframes_to_check = [timeframe]
                else:
                    return {
                        "instrument": instrument,
                        "status": "timeframe_not_found",
                        "available_timeframes": list(self.recent_feedback[instrument].keys())
                    }
            else:
                # All timeframes
                timeframes_to_check = list(self.recent_feedback[instrument].keys())
            
            # Process each timeframe
            for tf in timeframes_to_check:
                # Filter by time if specified
                filtered_feedback = self.recent_feedback[instrument][tf]
                
                if start_time or end_time:
                    filtered_feedback = []
                    for fb in self.recent_feedback[instrument][tf]:
                        try:
                            fb_time = datetime.fromisoformat(fb.timestamp)
                            if start_time and fb_time < start_time:
                                continue
                            if end_time and fb_time > end_time:
                                continue
                            filtered_feedback.append(fb)
                        except:
                            continue
                
                # Sort by timestamp (newest first)
                filtered_feedback = sorted(
                    filtered_feedback,
                    key=lambda fb: fb.timestamp if hasattr(fb, "timestamp") else "",
                    reverse=True
                )
                
                # Apply limit
                filtered_feedback = filtered_feedback[:limit]
                
                # Extract key information
                feedback_data = []
                for fb in filtered_feedback:
                    item = {
                        "id": fb.id,
                        "timestamp": fb.timestamp,
                        "source": fb.source.value if hasattr(fb, "source") else "unknown",
                        "category": fb.category.value if hasattr(fb, "category") else "unknown"
                    }
                    
                    # Add error magnitude if available
                    if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None:
                        item["error_magnitude"] = fb.error_magnitude
                    
                    # Add metadata if available
                    if hasattr(fb, "metadata") and fb.metadata:
                        if isinstance(fb.metadata, str):
                            try:
                                item["metadata"] = json.loads(fb.metadata)
                            except:
                                pass
                        elif isinstance(fb.metadata, dict):
                            item["metadata"] = fb.metadata
                    
                    feedback_data.append(item)
                
                # Calculate statistics
                error_magnitudes = [
                    fb.error_magnitude for fb in filtered_feedback 
                    if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None
                ]
                
                stats = {
                    "count": len(filtered_feedback),
                    "has_error_data": len(error_magnitudes) > 0
                }
                
                if error_magnitudes:
                    stats["avg_error"] = sum(error_magnitudes) / len(error_magnitudes)
                    stats["min_error"] = min(error_magnitudes)
                    stats["max_error"] = max(error_magnitudes)
                    
                    if len(error_magnitudes) > 1:
                        mean = stats["avg_error"]
                        variance = sum((x - mean) ** 2 for x in error_magnitudes) / len(error_magnitudes)
                        stats["std_dev"] = variance ** 0.5
                
                # Add to result
                result["timeframes"][tf] = {
                    "feedback": feedback_data,
                    "statistics": stats
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting feedback by timeframe: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to get feedback by timeframe: {str(e)}")
    
    def clear_old_feedback(self, max_items: Optional[int] = None) -> Dict[str, int]:
        """
        Clear old feedback to prevent memory growth.
        
        Args:
            max_items: Maximum items to keep per timeframe (overrides instance setting)
            
        Returns:
            Dict[str, int]: Count of items removed by instrument
        """
        max_items = max_items or self.max_items_per_timeframe
        removed_counts = {}
        
        for instrument, timeframes in self.recent_feedback.items():
            removed_counts[instrument] = 0
            
            for timeframe, feedback_list in timeframes.items():
                if len(feedback_list) > max_items:
                    # Sort by timestamp and keep only the most recent
                    feedback_list.sort(key=lambda fb: fb.timestamp)
                    removed = len(feedback_list) - max_items
                    self.recent_feedback[instrument][timeframe] = feedback_list[-max_items:]
                    removed_counts[instrument] += removed
        
        return removed_counts