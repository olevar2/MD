"""
Specialized service for handling multi-timeframe prediction feedback.

This service provides mechanisms for collecting, correlating, and processing
feedback related to predictions across multiple timeframes, enabling the
adaptive layer to adjust prediction strategies based on temporal patterns.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from core_foundations.models.feedback import TimeframeFeedback, FeedbackPriority, FeedbackCategory, FeedbackSource, FeedbackStatus
from core_foundations.interfaces.model_trainer import IFeedbackRepository

logger = logging.getLogger(__name__)


class TimeframeFeedbackService:
    """
    Service responsible for specialized handling of multi-timeframe prediction feedback.
    
    This service enables:
    1. Collection of correlated feedback across multiple timeframes
    2. Analysis of temporal relationships between timeframe predictions
    3. Generation of specialized feedback for model retraining
    """
    
    def __init__(
        self, 
        feedback_repository: IFeedbackRepository,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the TimeframeFeedbackService.
        
        Args:
            feedback_repository: Repository for storing and retrieving feedback
            config: Configuration settings for the service
        """
        self.feedback_repository = feedback_repository
        self.config = config or {}
        self.correlation_threshold = self.config.get('correlation_threshold', 0.6)
        self.significance_threshold = self.config.get('significance_threshold', 0.7)
        logger.info("TimeframeFeedbackService initialized with correlation threshold: %.2f", 
                   self.correlation_threshold)
    
    def submit_timeframe_feedback(
        self,
        model_id: str,
        timeframe: str,
        prediction_error: float,
        actual_value: float,
        predicted_value: float,
        prediction_timestamp: datetime,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Submit feedback for a specific timeframe prediction.
        
        Args:
            model_id: Identifier of the model that made the prediction
            timeframe: The timeframe of this prediction (e.g., "1h", "4h", "1d")
            prediction_error: The calculated error between prediction and actual
            actual_value: The actual observed value
            predicted_value: The value that was predicted
            prediction_timestamp: When the prediction was made
            metadata: Additional metadata about the prediction context
            
        Returns:
            The ID of the created feedback item
        """
        # Calculate statistical significance based on error magnitude
        # (In a real implementation, this would use a more sophisticated approach)
        statistical_significance = self._calculate_significance(prediction_error, timeframe)
        
        # Determine priority based on error magnitude and significance
        priority = self._determine_priority(prediction_error, statistical_significance)
        
        # Create the timeframe feedback object
        feedback = TimeframeFeedback(
            timeframe=timeframe,
            related_timeframes=[],  # Will be populated later by correlation analysis
            temporal_correlation_data={},  # Will be populated later
            model_id=model_id,
            category=FeedbackCategory.TIMEFRAME_ADJUSTMENT,
            priority=priority,
            source=FeedbackSource.PERFORMANCE_METRICS,
            status=FeedbackStatus.NEW,
            statistical_significance=statistical_significance,
            content={
                "prediction_error": prediction_error,
                "actual_value": actual_value,
                "predicted_value": predicted_value,
                "prediction_timestamp": prediction_timestamp.isoformat()
            },
            metadata=metadata or {}
        )
        
        # Store the feedback and get the ID
        feedback_id = self._store_feedback(feedback)
        
        # Trigger asynchronous correlation analysis if configured
        if self.config.get('auto_correlate', True):
            self._schedule_correlation_analysis(feedback_id, model_id, timeframe)
            
        return feedback_id
    
    def correlate_timeframes(
        self,
        model_id: str,
        primary_timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Analyze correlations between different timeframe predictions.
        
        This method examines feedback from different timeframes within the
        specified time period to identify temporal patterns and correlations.
        
        Args:
            model_id: Identifier of the model to analyze
            primary_timeframe: The main timeframe to correlate with others
            start_time: Start time of the analysis period
            end_time: End time of the analysis period
            
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info("Starting timeframe correlation analysis for model %s (primary: %s)",
                   model_id, primary_timeframe)
        
        # Fetch all feedback for this model within the time period
        feedback_items = self._fetch_timeframe_feedback(model_id, start_time, end_time)
        
        if not feedback_items:
            logger.warning("No feedback found for correlation analysis")
            return {"status": "no_data", "correlations": {}}
            
        # Group feedback by timeframe
        timeframe_groups = self._group_by_timeframe(feedback_items)
        
        # Calculate correlations between timeframes
        correlations = self._calculate_timeframe_correlations(
            timeframe_groups, primary_timeframe)
        
        # Identify strongly correlated timeframes
        correlated_timeframes = [
            tf for tf, corr in correlations.items() 
            if abs(corr) >= self.correlation_threshold and tf != primary_timeframe
        ]
        
        # Update the original feedback items with correlation data
        self._update_feedback_with_correlations(
            feedback_items, primary_timeframe, correlated_timeframes, correlations)
        
        return {
            "status": "success",
            "correlations": correlations,
            "strongly_correlated_timeframes": correlated_timeframes,
            "analysis_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
    
    def generate_timeframe_adjustment_feedback(
        self,
        model_id: str,
        correlation_analysis: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate consolidated feedback for timeframe adjustments based on correlation analysis.
        
        This creates a higher-level feedback item that suggests timeframe
        adjustments based on the correlation patterns detected.
        
        Args:
            model_id: Identifier of the model
            correlation_analysis: Results from the correlate_timeframes method
            
        Returns:
            ID of the generated feedback item, or None if no adjustments needed
        """
        if correlation_analysis["status"] != "success":
            logger.warning("Cannot generate adjustment feedback: correlation analysis not successful")
            return None
            
        correlations = correlation_analysis["correlations"]
        if not correlations:
            return None
            
        # Identify the most strongly correlated timeframes (positive and negative)
        strongest_positive = max(correlations.items(), key=lambda x: x[1] if x[1] > 0 else -1)
        strongest_negative = min(correlations.items(), key=lambda x: x[1] if x[1] < 0 else 1)
        
        # Build adjustment recommendations
        adjustments = []
        
        if strongest_positive[1] >= self.correlation_threshold:
            adjustments.append({
                "timeframe_pair": strongest_positive[0],
                "correlation": strongest_positive[1],
                "recommendation": "Consider merging prediction models or using transfer learning"
            })
            
        if strongest_negative[1] <= -self.correlation_threshold:
            adjustments.append({
                "timeframe_pair": strongest_negative[0],
                "correlation": strongest_negative[1],
                "recommendation": "Examine inverse relationship for predictive opportunities"
            })
            
        if not adjustments:
            logger.info("No significant timeframe adjustments identified")
            return None
            
        # Create a feedback item with the adjustments
        feedback = TimeframeFeedback(
            timeframe="multiple",  # Indicates this feedback applies to multiple timeframes
            related_timeframes=list(correlations.keys()),
            temporal_correlation_data=correlations,
            model_id=model_id,
            category=FeedbackCategory.TIMEFRAME_ADJUSTMENT,
            priority=FeedbackPriority.HIGH,  # Timeframe adjustments are strategically important
            source=FeedbackSource.SYSTEM_AUTO,
            status=FeedbackStatus.VALIDATED,  # This is already validated by correlation analysis
            statistical_significance=max([abs(corr) for corr in correlations.values()]) if correlations else 0.5,
            content={
                "adjustments": adjustments,
                "analysis_period": correlation_analysis["analysis_period"]
            },
            metadata={
                "auto_generated": True,
                "correlation_threshold_used": self.correlation_threshold
            }
        )
        
        # Store the feedback
        feedback_id = self._store_feedback(feedback)
        logger.info("Generated timeframe adjustment feedback with ID %s", feedback_id)
        
        return feedback_id
    
    def _calculate_significance(self, prediction_error: float, timeframe: str) -> float:
        """
        Calculate statistical significance of a prediction error.
        
        Args:
            prediction_error: The prediction error value
            timeframe: The timeframe of the prediction
            
        Returns:
            A significance score between 0 and 1
        """
        # In a real implementation, this would use statistical methods
        # based on historical error distribution for this timeframe
        
        # For now, use a simple heuristic based on error magnitude and timeframe
        # (assuming longer timeframes should be more accurate)
        timeframe_factor = self._get_timeframe_factor(timeframe)
        
        # Scale error relative to timeframe expectations
        scaled_error = abs(prediction_error) * timeframe_factor
        
        # Convert to a significance score (higher error = higher significance)
        # using a sigmoid-like function to bound between 0 and 1
        significance = min(1.0, scaled_error / (1.0 + scaled_error))
        
        return significance
    
    def _get_timeframe_factor(self, timeframe: str) -> float:
        """
        Get a scaling factor for a timeframe.
        
        Longer timeframes generally have higher expectations for accuracy,
        so their errors are considered more significant.
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")
            
        Returns:
            A scaling factor for significance calculation
        """
        # Extract the numeric part and unit from the timeframe string
        # This is a simplified implementation
        if timeframe.endswith('m'):
            # Minutes
            value = float(timeframe[:-1])
            return value / 60.0  # Normalize to hours
        elif timeframe.endswith('h'):
            # Hours
            return float(timeframe[:-1])
        elif timeframe.endswith('d'):
            # Days
            return float(timeframe[:-1]) * 24.0  # Convert to hours
        elif timeframe.endswith('w'):
            # Weeks
            return float(timeframe[:-1]) * 24.0 * 7.0  # Convert to hours
        else:
            # Default case
            return 1.0
    
    def _determine_priority(self, prediction_error: float, significance: float) -> FeedbackPriority:
        """
        Determine the priority of feedback based on error and significance.
        
        Args:
            prediction_error: The prediction error value
            significance: The calculated significance score
            
        Returns:
            A FeedbackPriority enum value
        """
        # Combine error magnitude and significance
        weighted_score = abs(prediction_error) * significance
        
        # Apply thresholds from config or use defaults
        critical_threshold = self.config.get('critical_error_threshold', 0.8)
        high_threshold = self.config.get('high_error_threshold', 0.6)
        medium_threshold = self.config.get('medium_error_threshold', 0.3)
        
        if weighted_score >= critical_threshold:
            return FeedbackPriority.CRITICAL
        elif weighted_score >= high_threshold:
            return FeedbackPriority.HIGH
        elif weighted_score >= medium_threshold:
            return FeedbackPriority.MEDIUM
        else:
            return FeedbackPriority.LOW
    
    def _store_feedback(self, feedback: TimeframeFeedback) -> str:
        """
        Store a feedback item in the repository.
        
        Args:
            feedback: The TimeframeFeedback object to store
            
        Returns:
            The ID of the stored feedback item
        """
        # In a real implementation, this would call the repository's store method
        # For now, we'll just return the feedback ID that was already generated
        logger.debug("Storing timeframe feedback for model %s, timeframe %s", 
                    feedback.model_id, feedback.timeframe)
        
        # Placeholder implementation
        # In reality, the repository would handle persistence
        return feedback.feedback_id
    
    def _schedule_correlation_analysis(self, feedback_id: str, model_id: str, timeframe: str):
        """
        Schedule an asynchronous correlation analysis.
        
        Args:
            feedback_id: ID of the just-stored feedback
            model_id: Model identifier
            timeframe: Timeframe of the feedback
        """
        # In a real implementation, this would enqueue a task to a worker
        # or use async processing to perform correlation analysis
        logger.debug("Scheduling correlation analysis for feedback %s", feedback_id)
        
        # Placeholder - in a real system, this would be async
        # self.worker_queue.enqueue('correlate_timeframes', 
        #                          model_id, timeframe, 
        #                          datetime.utcnow() - timedelta(days=7), 
        #                          datetime.utcnow())
    
    def _fetch_timeframe_feedback(
        self, 
        model_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[TimeframeFeedback]:
        """
        Fetch timeframe feedback within a time period.
        
        Args:
            model_id: Model identifier
            start_time: Start of time period
            end_time: End of time period
            
        Returns:
            List of TimeframeFeedback items
        """
        # In a real implementation, this would query the repository
        # For now, return an empty list as placeholder
        logger.debug("Fetching timeframe feedback for model %s between %s and %s", 
                    model_id, start_time, end_time)
        
        # Placeholder - in a real implementation this would fetch from repository
        return []
    
    def _group_by_timeframe(self, feedback_items: List[TimeframeFeedback]) -> Dict[str, List[TimeframeFeedback]]:
        """
        Group feedback items by timeframe.
        
        Args:
            feedback_items: List of feedback items to group
            
        Returns:
            Dictionary mapping timeframes to lists of feedback items
        """
        groups = {}
        for item in feedback_items:
            if item.timeframe not in groups:
                groups[item.timeframe] = []
            groups[item.timeframe].append(item)
        return groups
    
    def _calculate_timeframe_correlations(
        self, 
        timeframe_groups: Dict[str, List[TimeframeFeedback]],
        primary_timeframe: str
    ) -> Dict[str, float]:
        """
        Calculate correlations between timeframes.
        
        Args:
            timeframe_groups: Feedback items grouped by timeframe
            primary_timeframe: The primary timeframe to correlate against
            
        Returns:
            Dictionary mapping timeframe pairs to correlation coefficients
        """
        correlations = {}
        
        if primary_timeframe not in timeframe_groups:
            logger.warning("Primary timeframe %s not found in feedback", primary_timeframe)
            return correlations
            
        primary_errors = self._extract_prediction_errors(timeframe_groups[primary_timeframe])
        
        for tf, items in timeframe_groups.items():
            if tf == primary_timeframe:
                continue
                
            # Extract prediction errors for this timeframe
            errors = self._extract_prediction_errors(items)
            
            # Calculate correlation coefficient
            correlation = self._correlation_coefficient(primary_errors, errors)
            
            # Store the result
            correlations[f"{primary_timeframe}_{tf}"] = correlation
        
        return correlations
    
    def _extract_prediction_errors(self, feedback_items: List[TimeframeFeedback]) -> List[float]:
        """
        Extract prediction errors from feedback items.
        
        Args:
            feedback_items: List of feedback items
            
        Returns:
            List of prediction error values
        """
        errors = []
        for item in feedback_items:
            if 'prediction_error' in item.content:
                errors.append(item.content['prediction_error'])
        return errors
    
    def _correlation_coefficient(self, x: List[float], y: List[float]) -> float:
        """
        Calculate the Pearson correlation coefficient between two series.
        
        Args:
            x: First series of values
            y: Second series of values
            
        Returns:
            Correlation coefficient between -1 and 1
        """
        # Simple implementation of Pearson correlation
        # In a real application, you would use numpy or scipy
        if len(x) != len(y) or len(x) == 0:
            return 0.0
            
        # Calculate means
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        # Calculate covariance and variances
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        variance_x = sum((val - mean_x) ** 2 for val in x)
        variance_y = sum((val - mean_y) ** 2 for val in y)
        
        # Calculate correlation coefficient
        if variance_x == 0 or variance_y == 0:
            return 0.0
        else:
            return covariance / ((variance_x * variance_y) ** 0.5)
    
    def _update_feedback_with_correlations(
        self,
        feedback_items: List[TimeframeFeedback],
        primary_timeframe: str,
        correlated_timeframes: List[str],
        correlations: Dict[str, float]
    ):
        """
        Update feedback items with correlation information.
        
        Args:
            feedback_items: List of feedback items to update
            primary_timeframe: Primary timeframe
            correlated_timeframes: List of correlated timeframe identifiers
            correlations: Dictionary of correlation coefficients
        """
        # Find feedback items for the primary timeframe
        primary_items = [item for item in feedback_items if item.timeframe == primary_timeframe]
        
        for item in primary_items:
            # Extract the timeframe from the correlation key (format: "tf1_tf2")
            related = [key.split('_')[1] for key in correlations.keys()]
            
            # Update the item's related timeframes and correlation data
            item.related_timeframes = related
            item.temporal_correlation_data = correlations
            
            # Store the updated item
            self._store_feedback(item)
