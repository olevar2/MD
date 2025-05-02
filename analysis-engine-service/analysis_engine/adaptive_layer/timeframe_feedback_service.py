"""
Timeframe Feedback Service

This module implements specialized feedback mechanisms for multi-timeframe predictions,
including cross-timeframe correlation analysis and temporal feedback adjustments.
"""

from typing import Dict, List, Any, Optional, Union, Set, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import uuid
import json
import numpy as np
from collections import defaultdict

from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_schema import Event, EventType
from core_foundations.exceptions.feedback_exceptions import FeedbackProcessingError, TimeframeFeedbackError

logger = get_logger(__name__)


class TimeframeFeedbackService:
    """
    Service for handling multi-timeframe prediction feedback and analysis.
    
    This service provides specialized feedback mechanisms for multi-timeframe predictions,
    focusing on cross-timeframe correlation analysis and temporal feedback adjustments.
    """
    
    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the timeframe feedback service.
        
        Args:
            event_publisher: Event publisher for broadcasting events
            config: Configuration parameters
        """
        self.event_publisher = event_publisher
        self.config = config or {}
        
        # Set default configuration
        self._set_default_config()
        
        # Store recent feedback by timeframe
        # Structure: {instrument: {timeframe: [feedback_items]}}
        self.recent_feedback = {}
        
        # Store cross-timeframe correlations
        # Structure: {instrument: {(tf1, tf2): correlation_data}}
        self.timeframe_correlations = {}
        
        # Store common timeframes for easy reference
        self.common_timeframes = [
            "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"
        ]
        
        # Background task for periodic analysis
        self._analysis_task = None
        self._is_running = False
        
        logger.info("TimeframeFeedbackService initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {
            "max_feedback_items_per_timeframe": 1000,   # Maximum items to keep per timeframe
            "correlation_min_samples": 30,              # Minimum samples for correlation calculation
            "analysis_interval": 3600,                  # Run analysis every 1 hour
            "temporal_decay_factor": 0.95,              # Decay factor for older feedback
            "default_lookback_days": 30,                # Default lookback period in days
            "min_correlation_significance": 0.3,        # Minimum correlation to consider significant
            "enable_periodic_analysis": True,           # Whether to enable periodic analysis
            "timeframe_importance_weights": {           # Weights for different timeframes
                "1m": 0.6,
                "5m": 0.7,
                "15m": 0.8,
                "30m": 0.85,
                "1h": 0.9,
                "4h": 0.95,
                "1d": 1.0,
                "1w": 1.0
            }
        }
        
        # Apply defaults for missing config values
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
        
        # Ensure we have weights for all common timeframes
        for tf in self.common_timeframes:
            if tf not in self.config["timeframe_importance_weights"]:
                self.config["timeframe_importance_weights"][tf] = 0.8  # Default weight
    
    async def start(self):
        """Start the timeframe feedback service."""
        self._is_running = True
        
        # Start background analysis task if enabled
        if self.config["enable_periodic_analysis"]:
            self._analysis_task = asyncio.create_task(self._periodic_analysis())
            logger.info("Started periodic analysis task")
            
        logger.info("TimeframeFeedbackService started")
    
    async def stop(self):
        """Stop the timeframe feedback service."""
        self._is_running = False
        
        # Cancel background task
        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
            
        logger.info("TimeframeFeedbackService stopped")
    
    async def _periodic_analysis(self):
        """Periodically analyze cross-timeframe correlations."""
        try:
            while self._is_running:
                # Sleep until next analysis interval
                await asyncio.sleep(self.config["analysis_interval"])
                
                logger.info("Running periodic cross-timeframe analysis")
                
                # Analyze all instruments with feedback
                for instrument in self.recent_feedback.keys():
                    try:
                        await self.analyze_cross_timeframe_correlations(instrument)
                        logger.debug(f"Completed cross-timeframe analysis for {instrument}")
                    except Exception as e:
                        logger.error(f"Error analyzing cross-timeframe correlations for {instrument}: {str(e)}", exc_info=True)
                
                logger.info("Completed periodic cross-timeframe analysis")
                
        except asyncio.CancelledError:
            logger.info("Periodic analysis task canceled")
        except Exception as e:
            logger.error(f"Error in periodic analysis task: {str(e)}", exc_info=True)
    
    async def collect_timeframe_feedback(self, feedback: TradeFeedback) -> str:
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
            timeframe = None
            
            # Try to extract timeframe from metadata
            if hasattr(feedback, "metadata"):
                metadata = feedback.metadata
                if isinstance(metadata, str):
                    try:
                        metadata_dict = json.loads(metadata)
                        timeframe = metadata_dict.get("timeframe")
                    except:
                        pass
                elif isinstance(metadata, dict):
                    timeframe = metadata.get("timeframe")
            
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
            max_items = self.config["max_feedback_items_per_timeframe"]
            if len(feedback_list) > max_items:
                # Remove oldest items
                feedback_list.sort(key=lambda fb: fb.timestamp)
                self.recent_feedback[instrument][timeframe] = feedback_list[-max_items:]
            
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
    
    async def analyze_cross_timeframe_correlations(self, instrument: str) -> Dict[str, Any]:
        """
        Analyze cross-timeframe correlations for a specific instrument.
        
        Args:
            instrument: The instrument to analyze
            
        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        try:
            if instrument not in self.recent_feedback:
                logger.warning(f"No feedback data for instrument {instrument}")
                return {"instrument": instrument, "status": "no_data"}
            
            # Get all timeframes for this instrument
            timeframes = list(self.recent_feedback[instrument].keys())
            if len(timeframes) < 2:
                logger.info(f"Insufficient timeframes for correlation analysis on {instrument}")
                return {
                    "instrument": instrument,
                    "status": "insufficient_timeframes",
                    "timeframes_found": timeframes
                }
            
            # Extract error data by timeframe
            timeframe_data = {}
            for tf in timeframes:
                # Get error magnitudes and timestamps
                data = []
                for fb in self.recent_feedback[instrument][tf]:
                    if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None:
                        try:
                            timestamp = datetime.fromisoformat(fb.timestamp)
                            data.append((timestamp, fb.error_magnitude))
                        except:
                            continue
                
                if data:
                    # Sort by timestamp
                    data.sort(key=lambda x: x[0])
                    timeframe_data[tf] = data
            
            # Calculate correlations between timeframes
            correlations = {}
            for i, tf1 in enumerate(timeframe_data.keys()):
                for tf2 in list(timeframe_data.keys())[i+1:]:
                    # Calculate correlation between these timeframes
                    correlation_result = self._calculate_timeframe_correlation(
                        instrument, tf1, tf2, timeframe_data[tf1], timeframe_data[tf2]
                    )
                    
                    # Store the correlation
                    correlation_key = (tf1, tf2)
                    correlations[correlation_key] = correlation_result
                    
                    # Also update the stored correlations
                    if instrument not in self.timeframe_correlations:
                        self.timeframe_correlations[instrument] = {}
                    
                    self.timeframe_correlations[instrument][correlation_key] = correlation_result
            
            # Calculate average error by timeframe
            avg_errors = {}
            for tf, data in timeframe_data.items():
                if data:
                    avg_errors[tf] = sum(error for _, error in data) / len(data)
            
            # Publish analysis results if available
            if self.event_publisher:
                await self.event_publisher.publish(
                    "feedback.timeframe.analysis",
                    {
                        "instrument": instrument,
                        "timestamp": datetime.utcnow().isoformat(),
                        "timeframes_analyzed": list(timeframe_data.keys()),
                        "correlation_count": len(correlations),
                        "avg_errors": avg_errors
                    }
                )
            
            return {
                "instrument": instrument,
                "status": "analyzed",
                "timeframes_analyzed": list(timeframe_data.keys()),
                "correlations": {
                    f"{tf1}-{tf2}": {
                        "correlation": result["correlation"],
                        "significance": result["significance"],
                        "sample_size": result["sample_size"]
                    }
                    for (tf1, tf2), result in correlations.items()
                },
                "avg_errors": avg_errors,
                "analysis_time": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error analyzing cross-timeframe correlations for {instrument}: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to analyze cross-timeframe correlations: {str(e)}")
    
    def _calculate_timeframe_correlation(
        self,
        instrument: str,
        timeframe1: str,
        timeframe2: str,
        data1: List[Tuple[datetime, float]],
        data2: List[Tuple[datetime, float]]
    ) -> Dict[str, Any]:
        """
        Calculate correlation between errors in two timeframes.
        
        Args:
            instrument: The instrument
            timeframe1: First timeframe
            timeframe2: Second timeframe
            data1: Error data for first timeframe [(timestamp, error), ...]
            data2: Error data for second timeframe [(timestamp, error), ...]
            
        Returns:
            Dict[str, Any]: Correlation results
        """
        # Basic correlation info
        result = {
            "instrument": instrument,
            "timeframe1": timeframe1,
            "timeframe2": timeframe2,
            "correlation": 0.0,
            "significance": "none",
            "sample_size": 0,
            "calculation_time": datetime.utcnow().isoformat()
        }
        
        # If not enough data points, return early
        if not data1 or not data2:
            return result
        
        try:
            # Extract timestamps and errors
            timestamps1 = [d[0] for d in data1]
            errors1 = [d[1] for d in data1]
            
            timestamps2 = [d[0] for d in data2]
            errors2 = [d[1] for d in data2]
            
            # Find common time window
            min_time = max(min(timestamps1), min(timestamps2))
            max_time = min(max(timestamps1), max(timestamps2))
            
            # Filter data to common time window
            filtered_data1 = [(t, e) for t, e in data1 if min_time <= t <= max_time]
            filtered_data2 = [(t, e) for t, e in data2 if min_time <= t <= max_time]
            
            # Ensure we still have enough data
            min_samples = self.config["correlation_min_samples"]
            if len(filtered_data1) < min_samples or len(filtered_data2) < min_samples:
                logger.debug(f"Insufficient data for correlation between {timeframe1} and {timeframe2} for {instrument}")
                result["sample_size"] = min(len(filtered_data1), len(filtered_data2))
                return result
            
            # Align data points by timestamp
            # This is a simple approach - in practice, you might need more sophisticated time-based alignment
            aligned_errors1 = []
            aligned_errors2 = []
            
            # Create lookup dictionary for quick access
            lookup2 = {t.isoformat(): e for t, e in filtered_data2}
            
            # For each point in timeframe1, find closest matching point in timeframe2
            for t1, e1 in filtered_data1:
                t1_iso = t1.isoformat()
                
                if t1_iso in lookup2:
                    # Direct timestamp match
                    aligned_errors1.append(e1)
                    aligned_errors2.append(lookup2[t1_iso])
                else:
                    # Find closest timestamp in timeframe2
                    # Skip this for now as it requires more complex logic to find the closest timestamp
                    continue
            
            # Calculate correlation if we have enough aligned points
            if len(aligned_errors1) >= min_samples:
                # Calculate Pearson correlation
                correlation = np.corrcoef(aligned_errors1, aligned_errors2)[0, 1]
                
                # Handle NaN (can happen if all values in a series are identical)
                if np.isnan(correlation):
                    correlation = 0.0
                
                result["correlation"] = float(correlation)
                result["sample_size"] = len(aligned_errors1)
                
                # Determine significance
                abs_corr = abs(correlation)
                if abs_corr >= 0.7:
                    result["significance"] = "strong"
                elif abs_corr >= self.config["min_correlation_significance"]:
                    result["significance"] = "moderate"
                else:
                    result["significance"] = "weak"
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating timeframe correlation: {str(e)}", exc_info=True)
            return result
    
    async def get_temporal_feedback_adjustments(
        self,
        instrument: str,
        target_timeframe: str,
        reference_timeframes: Optional[List[str]] = None,
        lookback_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get temporal feedback adjustments based on multi-timeframe analysis.
        
        Args:
            instrument: The instrument to analyze
            target_timeframe: The target timeframe to get adjustments for
            reference_timeframes: List of timeframes to use as reference (optional)
            lookback_days: Number of days to look back (optional)
            
        Returns:
            Dict[str, Any]: Temporal adjustment recommendations
        """
        try:
            if instrument not in self.recent_feedback:
                logger.warning(f"No feedback data for instrument {instrument}")
                return {"instrument": instrument, "status": "no_data"}
            
            if target_timeframe not in self.recent_feedback[instrument]:
                logger.warning(f"No feedback data for {target_timeframe} timeframe on {instrument}")
                return {
                    "instrument": instrument,
                    "timeframe": target_timeframe,
                    "status": "no_timeframe_data"
                }
            
            # Set defaults
            if not reference_timeframes:
                reference_timeframes = [tf for tf in self.recent_feedback[instrument].keys() 
                                      if tf != target_timeframe]
            else:
                # Filter to timeframes that actually have data
                reference_timeframes = [tf for tf in reference_timeframes 
                                      if tf in self.recent_feedback[instrument]]
            
            if not lookback_days:
                lookback_days = self.config["default_lookback_days"]
            
            # Calculate lookback date
            lookback_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            # Get target timeframe feedback within lookback period
            target_feedback = []
            for fb in self.recent_feedback[instrument][target_timeframe]:
                try:
                    fb_time = datetime.fromisoformat(fb.timestamp)
                    if fb_time >= lookback_date:
                        target_feedback.append(fb)
                except:
                    continue
            
            # Check if we have enough target data
            if not target_feedback:
                return {
                    "instrument": instrument,
                    "timeframe": target_timeframe,
                    "status": "insufficient_target_data"
                }
            
            # Calculate average target error
            target_errors = [
                fb.error_magnitude for fb in target_feedback 
                if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None
            ]
            
            if not target_errors:
                return {
                    "instrument": instrument,
                    "timeframe": target_timeframe,
                    "status": "no_error_data"
                }
            
            avg_target_error = sum(target_errors) / len(target_errors)
            
            # Get reference timeframe data
            reference_data = {}
            for tf in reference_timeframes:
                if tf not in self.recent_feedback[instrument]:
                    continue
                
                # Filter to lookback period
                ref_feedback = []
                for fb in self.recent_feedback[instrument][tf]:
                    try:
                        fb_time = datetime.fromisoformat(fb.timestamp)
                        if fb_time >= lookback_date:
                            ref_feedback.append(fb)
                    except:
                        continue
                
                # Calculate average error
                ref_errors = [
                    fb.error_magnitude for fb in ref_feedback 
                    if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None
                ]
                
                if ref_errors:
                    avg_ref_error = sum(ref_errors) / len(ref_errors)
                    
                    # Store reference data
                    reference_data[tf] = {
                        "avg_error": avg_ref_error,
                        "sample_size": len(ref_errors)
                    }
            
            # Check if we have enough reference data
            if not reference_data:
                return {
                    "instrument": instrument,
                    "timeframe": target_timeframe,
                    "status": "insufficient_reference_data"
                }
            
            # Look up correlations
            correlations = {}
            for ref_tf in reference_data.keys():
                # Try both orderings of the timeframe pair
                key1 = (target_timeframe, ref_tf)
                key2 = (ref_tf, target_timeframe)
                
                if instrument in self.timeframe_correlations:
                    if key1 in self.timeframe_correlations[instrument]:
                        correlations[ref_tf] = self.timeframe_correlations[instrument][key1]
                    elif key2 in self.timeframe_correlations[instrument]:
                        correlations[ref_tf] = self.timeframe_correlations[instrument][key2]
            
            # Calculate temporal adjustments
            adjustments = self._calculate_temporal_adjustments(
                target_timeframe, avg_target_error, reference_data, correlations
            )
            
            return {
                "instrument": instrument,
                "target_timeframe": target_timeframe,
                "status": "success",
                "reference_timeframes": list(reference_data.keys()),
                "lookback_days": lookback_days,
                "target_data": {
                    "avg_error": avg_target_error,
                    "sample_size": len(target_errors)
                },
                "reference_data": reference_data,
                "adjustments": adjustments,
                "calculation_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting temporal feedback adjustments: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to get temporal feedback adjustments: {str(e)}")
    
    def _calculate_temporal_adjustments(
        self,
        target_timeframe: str,
        avg_target_error: float,
        reference_data: Dict[str, Dict[str, Any]],
        correlations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate temporal adjustments based on multi-timeframe analysis.
        
        Args:
            target_timeframe: The target timeframe
            avg_target_error: Average error for target timeframe
            reference_data: Reference timeframe data
            correlations: Correlation data between timeframes
            
        Returns:
            Dict[str, Any]: Adjustment recommendations
        """
        # Get target timeframe weight
        target_weight = self.config["timeframe_importance_weights"].get(
            target_timeframe, 0.8  # Default weight if not specified
        )
        
        # Start with base adjustments
        adjustments = {
            "confidence_adjustment": 0.0,
            "error_magnitude_adjustment": 0.0,
            "prediction_bias_adjustment": 0.0,
            "recommended_actions": []
        }
        
        # Analyze each reference timeframe
        weighted_error_diffs = []
        weighted_correlations = []
        total_weight = 0.0
        
        for ref_tf, ref_data in reference_data.items():
            # Get weight for this timeframe
            ref_weight = self.config["timeframe_importance_weights"].get(ref_tf, 0.8)
            
            # Get correlation if available
            correlation_value = 0.0
            correlation_significance = "none"
            if ref_tf in correlations:
                correlation_value = correlations[ref_tf].get("correlation", 0.0)
                correlation_significance = correlations[ref_tf].get("significance", "none")
            
            # Adjust weight based on correlation significance
            if correlation_significance == "strong":
                adjusted_weight = ref_weight * 1.5
            elif correlation_significance == "moderate":
                adjusted_weight = ref_weight * 1.2
            else:
                adjusted_weight = ref_weight * 0.8
            
            # Calculate error difference
            avg_ref_error = ref_data["avg_error"]
            error_diff = avg_ref_error - avg_target_error
            
            # Add weighted values
            weighted_error_diffs.append(error_diff * adjusted_weight)
            weighted_correlations.append(abs(correlation_value) * adjusted_weight)
            total_weight += adjusted_weight
        
        # Calculate weighted averages
        if total_weight > 0:
            avg_weighted_error_diff = sum(weighted_error_diffs) / total_weight
            avg_weighted_correlation = sum(weighted_correlations) / total_weight
            
            # Calculate adjustments
            adjustments["confidence_adjustment"] = min(0.3, avg_weighted_correlation * 0.5)
            
            # Error magnitude adjustment
            if abs(avg_weighted_error_diff) > 0.05:  # Only adjust if difference is significant
                # Positive diff means reference errors are higher than target
                adjustments["error_magnitude_adjustment"] = avg_weighted_error_diff * 0.25
            
            # Prediction bias adjustment
            # This would be more complex in practice, based on directional bias in errors
            # For now, we'll use a simple approach based on error diff
            if avg_weighted_error_diff > 0.1:
                adjustments["prediction_bias_adjustment"] = 0.05  # Small upward bias
            elif avg_weighted_error_diff < -0.1:
                adjustments["prediction_bias_adjustment"] = -0.05  # Small downward bias
        
        # Generate recommended actions
        if abs(adjustments["error_magnitude_adjustment"]) > 0.1:
            if adjustments["error_magnitude_adjustment"] > 0:
                adjustments["recommended_actions"].append(
                    "Increase error tolerance for predictions"
                )
            else:
                adjustments["recommended_actions"].append(
                    "Decrease error tolerance for predictions"
                )
        
        if abs(adjustments["prediction_bias_adjustment"]) > 0.03:
            if adjustments["prediction_bias_adjustment"] > 0:
                adjustments["recommended_actions"].append(
                    "Apply upward bias correction to predictions"
                )
            else:
                adjustments["recommended_actions"].append(
                    "Apply downward bias correction to predictions"
                )
        
        if adjustments["confidence_adjustment"] > 0.15:
            adjustments["recommended_actions"].append(
                "Increase confidence in predictions based on cross-timeframe correlation"
            )
        
        return adjustments
    
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
    
    async def get_correlation_matrix(
        self,
        instrument: str,
        min_significance: str = "weak"
    ) -> Dict[str, Any]:
        """
        Get the correlation matrix between timeframes for a specific instrument.
        
        Args:
            instrument: The instrument to get correlations for
            min_significance: Minimum correlation significance to include (weak, moderate, strong)
            
        Returns:
            Dict[str, Any]: Correlation matrix data
        """
        try:
            if instrument not in self.timeframe_correlations:
                # No correlations calculated yet
                return {
                    "instrument": instrument,
                    "status": "no_correlations",
                    "recommendation": "Run analyze_cross_timeframe_correlations first"
                }
            
            # Map significance to minimum value
            min_value = 0.0
            if min_significance == "moderate":
                min_value = self.config["min_correlation_significance"]
            elif min_significance == "strong":
                min_value = 0.7
            
            # Get all unique timeframes
            timeframes = set()
            for (tf1, tf2) in self.timeframe_correlations[instrument].keys():
                timeframes.add(tf1)
                timeframes.add(tf2)
            
            # Sort timeframes by common order if possible
            ordered_timeframes = [tf for tf in self.common_timeframes if tf in timeframes]
            for tf in timeframes:
                if tf not in ordered_timeframes:
                    ordered_timeframes.append(tf)
            
            # Create correlation matrix
            matrix = {}
            for tf1 in ordered_timeframes:
                matrix[tf1] = {}
                for tf2 in ordered_timeframes:
                    if tf1 == tf2:
                        # Self correlation is always 1.0
                        matrix[tf1][tf2] = 1.0
                        continue
                    
                    # Check if we have correlation data
                    key1 = (tf1, tf2)
                    key2 = (tf2, tf1)
                    
                    if key1 in self.timeframe_correlations[instrument]:
                        corr = self.timeframe_correlations[instrument][key1].get("correlation", 0.0)
                        significance = self.timeframe_correlations[instrument][key1].get("significance", "none")
                    elif key2 in self.timeframe_correlations[instrument]:
                        corr = self.timeframe_correlations[instrument][key2].get("correlation", 0.0)
                        significance = self.timeframe_correlations[instrument][key2].get("significance", "none")
                    else:
                        corr = 0.0
                        significance = "none"
                    
                    # Apply minimum significance filter
                    if abs(corr) >= min_value:
                        matrix[tf1][tf2] = corr
                    else:
                        matrix[tf1][tf2] = 0.0  # Below threshold
            
            # Extract significant pairs for easy reference
            significant_pairs = []
            for (tf1, tf2), data in self.timeframe_correlations[instrument].items():
                corr = data.get("correlation", 0.0)
                if abs(corr) >= min_value:
                    significant_pairs.append({
                        "timeframe1": tf1,
                        "timeframe2": tf2,
                        "correlation": corr,
                        "significance": data.get("significance", "none"),
                        "sample_size": data.get("sample_size", 0)
                    })
            
            # Sort by correlation strength
            significant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "instrument": instrument,
                "status": "success",
                "timeframes": ordered_timeframes,
                "matrix": matrix,
                "significant_pairs": significant_pairs,
                "min_significance": min_significance,
                "query_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting correlation matrix: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to get correlation matrix: {str(e)}")
    
    async def get_multi_timeframe_insights(
        self,
        instrument: str,
        lookback_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive insights from multi-timeframe feedback analysis.
        
        Args:
            instrument: The instrument to analyze
            lookback_days: Number of days to look back (optional)
            
        Returns:
            Dict[str, Any]: Multi-timeframe insights
        """
        try:
            if instrument not in self.recent_feedback:
                return {"instrument": instrument, "status": "no_data"}
            
            # Set default lookback period
            if not lookback_days:
                lookback_days = self.config["default_lookback_days"]
            
            # Get all timeframes for this instrument
            timeframes = list(self.recent_feedback[instrument].keys())
            if not timeframes:
                return {"instrument": instrument, "status": "no_timeframes"}
            
            # Ensure we have correlations
            if instrument not in self.timeframe_correlations or not self.timeframe_correlations[instrument]:
                await self.analyze_cross_timeframe_correlations(instrument)
            
            # Get feedback statistics by timeframe
            feedback_stats = {}
            lookback_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            for tf in timeframes:
                # Filter to lookback period
                filtered_feedback = []
                for fb in self.recent_feedback[instrument][tf]:
                    try:
                        fb_time = datetime.fromisoformat(fb.timestamp)
                        if fb_time >= lookback_date:
                            filtered_feedback.append(fb)
                    except:
                        continue
                
                # Calculate statistics
                error_magnitudes = [
                    fb.error_magnitude for fb in filtered_feedback 
                    if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None
                ]
                
                stats = {
                    "count": len(filtered_feedback),
                    "recent_count": len(error_magnitudes)
                }
                
                if error_magnitudes:
                    stats["avg_error"] = sum(error_magnitudes) / len(error_magnitudes)
                    stats["min_error"] = min(error_magnitudes)
                    stats["max_error"] = max(error_magnitudes)
                    
                    if len(error_magnitudes) > 1:
                        mean = stats["avg_error"]
                        variance = sum((x - mean) ** 2 for x in error_magnitudes) / len(error_magnitudes)
                        stats["std_dev"] = variance ** 0.5
                
                feedback_stats[tf] = stats
            
            # Get correlation insights
            correlation_insights = self._extract_correlation_insights(instrument)
            
            # Generate temporal insights and recommendations
            temporal_insights = self._generate_temporal_insights(
                instrument, feedback_stats, correlation_insights
            )
            
            # Combine all insights
            insights = {
                "instrument": instrument,
                "status": "success",
                "lookback_days": lookback_days,
                "timeframes_analyzed": timeframes,
                "feedback_statistics": feedback_stats,
                "correlation_insights": correlation_insights,
                "temporal_insights": temporal_insights,
                "recommendations": self._generate_recommendations(
                    instrument, feedback_stats, correlation_insights, temporal_insights
                ),
                "analysis_time": datetime.utcnow().isoformat()
            }
            
            # Publish insights if available
            if self.event_publisher:
                await self.event_publisher.publish(
                    "feedback.timeframe.insights",
                    {
                        "instrument": instrument,
                        "timestamp": datetime.utcnow().isoformat(),
                        "timeframes_analyzed": timeframes,
                        "has_recommendations": len(insights["recommendations"]) > 0
                    }
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting multi-timeframe insights: {str(e)}", exc_info=True)
            raise TimeframeFeedbackError(f"Failed to get multi-timeframe insights: {str(e)}")
    
    def _extract_correlation_insights(self, instrument: str) -> Dict[str, Any]:
        """
        Extract insights from correlation data.
        
        Args:
            instrument: The instrument
            
        Returns:
            Dict[str, Any]: Correlation insights
        """
        insights = {
            "strong_correlations": [],
            "weak_correlations": [],
            "negative_correlations": [],
            "most_aligned_timeframes": [],
            "least_aligned_timeframes": []
        }
        
        if instrument not in self.timeframe_correlations:
            return insights
        
        # Process correlation data
        correlations = []
        for (tf1, tf2), data in self.timeframe_correlations[instrument].items():
            corr = data.get("correlation", 0.0)
            significance = data.get("significance", "none")
            
            correlation_info = {
                "timeframe1": tf1,
                "timeframe2": tf2,
                "correlation": corr,
                "significance": significance,
                "sample_size": data.get("sample_size", 0)
            }
            
            correlations.append(correlation_info)
            
            # Categorize correlations
            if significance == "strong" and corr > 0:
                insights["strong_correlations"].append(correlation_info)
            elif significance == "weak":
                insights["weak_correlations"].append(correlation_info)
            elif corr < 0:
                insights["negative_correlations"].append(correlation_info)
        
        # Find most/least aligned timeframes
        # First, create a score for each timeframe based on its correlations
        alignment_scores = defaultdict(float)
        alignment_counts = defaultdict(int)
        
        for info in correlations:
            tf1 = info["timeframe1"]
            tf2 = info["timeframe2"]
            corr = info["correlation"]
            
            alignment_scores[tf1] += corr
            alignment_scores[tf2] += corr
            
            alignment_counts[tf1] += 1
            alignment_counts[tf2] += 1
        
        # Calculate average alignment score
        avg_scores = {}
        for tf, score in alignment_scores.items():
            if alignment_counts[tf] > 0:
                avg_scores[tf] = score / alignment_counts[tf]
        
        # Sort by score
        if avg_scores:
            sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Most aligned (highest positive correlation)
            for tf, score in sorted_scores[:3]:  # Top 3
                if score > 0:
                    insights["most_aligned_timeframes"].append({
                        "timeframe": tf,
                        "avg_correlation": score,
                        "correlation_count": alignment_counts[tf]
                    })
            
            # Least aligned (lowest/negative correlation)
            for tf, score in sorted_scores[-3:]:  # Bottom 3
                insights["least_aligned_timeframes"].append({
                    "timeframe": tf,
                    "avg_correlation": score,
                    "correlation_count": alignment_counts[tf]
                })
        
        return insights
    
    def _generate_temporal_insights(
        self,
        instrument: str,
        feedback_stats: Dict[str, Dict[str, Any]],
        correlation_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate insights about temporal relationships between timeframes.
        
        Args:
            instrument: The instrument
            feedback_stats: Statistics about feedback by timeframe
            correlation_insights: Insights from correlation analysis
            
        Returns:
            Dict[str, Any]: Temporal insights
        """
        insights = {
            "error_patterns": [],
            "leading_timeframes": [],
            "lagging_timeframes": [],
            "most_accurate_timeframes": [],
            "least_accurate_timeframes": []
        }
        
        # Find most/least accurate timeframes
        error_by_timeframe = []
        for tf, stats in feedback_stats.items():
            if "avg_error" in stats:
                error_by_timeframe.append((tf, stats["avg_error"], stats["recent_count"]))
        
        if error_by_timeframe:
            # Sort by error (ascending)
            error_by_timeframe.sort(key=lambda x: x[1])
            
            # Most accurate (lowest error)
            for tf, error, count in error_by_timeframe[:3]:  # Top 3
                insights["most_accurate_timeframes"].append({
                    "timeframe": tf,
                    "avg_error": error,
                    "sample_count": count
                })
            
            # Least accurate (highest error)
            for tf, error, count in error_by_timeframe[-3:]:  # Bottom 3
                insights["least_accurate_timeframes"].append({
                    "timeframe": tf,
                    "avg_error": error,
                    "sample_count": count
                })
            
            # Look for patterns in errors across timeframes
            # Check if errors increase with timeframe duration
            # This is a simplistic check - would need more sophisticated analysis in practice
            
            # Try to order timeframes by duration
            ordered_timeframes = []
            for tf in self.common_timeframes:
                if any(t[0] == tf for t in error_by_timeframe):
                    ordered_timeframes.append(tf)
            
            if len(ordered_timeframes) >= 3:
                errors = []
                for tf in ordered_timeframes:
                    for t, error, _ in error_by_timeframe:
                        if t == tf:
                            errors.append(error)
                            break
                
                # Check for increasing/decreasing pattern
                if all(errors[i] <= errors[i+1] for i in range(len(errors)-1)):
                    insights["error_patterns"].append({
                        "pattern": "increasing",
                        "description": "Prediction error increases with timeframe duration",
                        "timeframes": ordered_timeframes,
                        "errors": errors
                    })
                elif all(errors[i] >= errors[i+1] for i in range(len(errors)-1)):
                    insights["error_patterns"].append({
                        "pattern": "decreasing",
                        "description": "Prediction error decreases with timeframe duration",
                        "timeframes": ordered_timeframes,
                        "errors": errors
                    })
        
        # Identify leading/lagging timeframes from correlations
        # This is a placeholder - real analysis would require temporal lead/lag analysis
        # For now, we'll use the strong correlations as a proxy
        for corr in correlation_insights["strong_correlations"]:
            tf1 = corr["timeframe1"]
            tf2 = corr["timeframe2"]
            
            # Check if we can determine which is leading
            # In this simple version, we'll assume shorter timeframes tend to lead
            # A real implementation would look at actual temporal relationships
            tf1_index = self.common_timeframes.index(tf1) if tf1 in self.common_timeframes else -1
            tf2_index = self.common_timeframes.index(tf2) if tf2 in self.common_timeframes else -1
            
            if tf1_index >= 0 and tf2_index >= 0:
                if tf1_index < tf2_index:
                    # tf1 is shorter timeframe (leading indicator)
                    insights["leading_timeframes"].append({
                        "timeframe": tf1,
                        "leads": tf2,
                        "correlation": corr["correlation"]
                    })
                    insights["lagging_timeframes"].append({
                        "timeframe": tf2,
                        "lags": tf1,
                        "correlation": corr["correlation"]
                    })
                else:
                    # tf2 is shorter timeframe (leading indicator)
                    insights["leading_timeframes"].append({
                        "timeframe": tf2,
                        "leads": tf1,
                        "correlation": corr["correlation"]
                    })
                    insights["lagging_timeframes"].append({
                        "timeframe": tf1,
                        "lags": tf2,
                        "correlation": corr["correlation"]
                    })
        
        return insights
    
    def _generate_recommendations(
        self,
        instrument: str,
        feedback_stats: Dict[str, Dict[str, Any]],
        correlation_insights: Dict[str, Any],
        temporal_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on multi-timeframe analysis.
        
        Args:
            instrument: The instrument
            feedback_stats: Statistics about feedback by timeframe
            correlation_insights: Insights from correlation analysis
            temporal_insights: Insights from temporal analysis
            
        Returns:
            List[Dict[str, Any]]: Recommendations
        """
        recommendations = []
        
        # Recommendation for strongly correlated timeframes
        if correlation_insights["strong_correlations"]:
            pairs = [f"{c['timeframe1']}-{c['timeframe2']}" for c in correlation_insights["strong_correlations"][:3]]
            recommendations.append({
                "type": "correlation",
                "priority": "medium",
                "description": f"Consider using {'and'.join(pairs)} together as they show strong correlation",
                "rationale": "Strongly correlated timeframes can reinforce each other's signals"
            })
        
        # Recommendation for leading timeframes
        if temporal_insights["leading_timeframes"]:
            leading_tfs = set(item["timeframe"] for item in temporal_insights["leading_timeframes"])
            if leading_tfs:
                recommendations.append({
                    "type": "leading_indicator",
                    "priority": "high",
                    "description": f"Use {', '.join(leading_tfs)} as leading indicators",
                    "rationale": "These timeframes show predictive correlation with larger timeframes"
                })
        
        # Recommendation for most accurate timeframes
        if temporal_insights["most_accurate_timeframes"]:
            accurate_tfs = [item["timeframe"] for item in temporal_insights["most_accurate_timeframes"]]
            if accurate_tfs:
                recommendations.append({
                    "type": "accuracy",
                    "priority": "high",
                    "description": f"Prioritize {', '.join(accurate_tfs)} for most accurate predictions",
                    "rationale": "These timeframes show lowest prediction errors"
                })
        
        # Recommendation for error patterns
        if temporal_insights["error_patterns"]:
            for pattern in temporal_insights["error_patterns"]:
                if pattern["pattern"] == "increasing":
                    recommendations.append({
                        "type": "error_pattern",
                        "priority": "medium",
                        "description": "Adjust confidence levels downward for longer timeframes",
                        "rationale": "Prediction errors consistently increase with timeframe duration"
                    })
                elif pattern["pattern"] == "decreasing":
                    recommendations.append({
                        "type": "error_pattern",
                        "priority": "medium",
                        "description": "Consider focusing on longer timeframes for better accuracy",
                        "rationale": "Prediction errors consistently decrease with timeframe duration"
                    })
        
        # Recommendation for multi-timeframe approach
        has_multiple_timeframes = len(feedback_stats) > 1
        if has_multiple_timeframes:
            recommendations.append({
                "type": "multi_timeframe",
                "priority": "medium",
                "description": "Implement weighted voting across timeframes for robust predictions",
                "rationale": "Combining multiple timeframes can reduce individual prediction errors"
            })
        
        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations
