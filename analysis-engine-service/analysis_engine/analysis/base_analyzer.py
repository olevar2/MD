"""
Base Analyzer Module

This module provides the foundation for all technical analysis components
with standardized interfaces, effectiveness logging, and performance monitoring.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import traceback
import uuid
from abc import ABC, abstractmethod

# --- Add imports for event publishing ---
from analysis_engine.config import settings
from analysis_engine.events.publisher import EventPublisher
from analysis_engine.events.schemas import AnalysisCompletionEvent, AnalysisCompletionPayload
# --- End Add imports ---

from analysis_engine.models.analysis_result import AnalysisResult
from analysis_engine.learning_from_mistakes.effectiveness_logger import EffectivenessLogger

logger = logging.getLogger(__name__)

# --- Initialize Event Publisher (consider dependency injection later) ---
# Note: Creating instance here for simplicity. In a larger app, use DI.
try:
    event_publisher = EventPublisher()
except Exception as e:
    logger.error(f"Failed to initialize EventPublisher in BaseAnalyzer: {e}", exc_info=True)
    event_publisher = None
# --- End Initialize Event Publisher ---

class BaseAnalyzer(ABC):
    """
    Abstract base class for all analysis components

    This class defines the standard interface for all analyzers and provides
    common functionality such as effectiveness logging and performance monitoring.
    """

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the base analyzer

        Args:
            name: Name identifier for the analyzer
            parameters: Configuration parameters for the analyzer
        """
        self.name = name
        self.parameters = parameters or {}
        self.effectiveness_logger = EffectivenessLogger(analyzer_name=name)
        self.last_execution_time = None
        self.execution_history = []
        self.max_history_size = 100
        self._initialize_performance_metrics()

    def _initialize_performance_metrics(self):
        """Initialize performance monitoring metrics"""
        self.performance_metrics = {
            "execution_count": 0,
            "total_execution_time": 0,
            "average_execution_time": 0,
            "max_execution_time": 0,
            "min_execution_time": float('inf'),
            "error_count": 0,
            "last_execution_timestamp": None
        }

    def _start_performance_timer(self) -> float:
        """Start the performance timer"""
        return time.perf_counter()

    def _stop_performance_timer(self, start_time: float) -> float:
        """
        Stop the performance timer and update metrics

        Args:
            start_time: Start time from _start_performance_timer

        Returns:
            Execution time in seconds
        """
        execution_time = time.perf_counter() - start_time

        # Update performance metrics
        self.performance_metrics["execution_count"] += 1
        self.performance_metrics["total_execution_time"] += execution_time
        self.performance_metrics["average_execution_time"] = (
            self.performance_metrics["total_execution_time"] /
            self.performance_metrics["execution_count"]
        )

        if execution_time > self.performance_metrics["max_execution_time"]:
            self.performance_metrics["max_execution_time"] = execution_time

        if execution_time < self.performance_metrics["min_execution_time"]:
            self.performance_metrics["min_execution_time"] = execution_time

        self.performance_metrics["last_execution_timestamp"] = datetime.now().isoformat()

        # Track execution history
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time
        })

        # Limit history size
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]

        return execution_time

    @abstractmethod
    async def analyze(self, data: Any) -> AnalysisResult:
        """
        Perform analysis on provided data

        Args:
            data: Data to analyze

        Returns:
            AnalysisResult containing analysis results
        """
        pass

    async def execute(self, data: Any) -> AnalysisResult:
        """
        Execute analysis with performance monitoring

        Args:
            data: Data to analyze

        Returns:
            AnalysisResult containing analysis results
        """
        execution_id = str(uuid.uuid4())
        start_time = self._start_performance_timer()

        try:
            logger.info(f"Starting analysis: {self.name} (ID: {execution_id})")
            result = await self.analyze(data)
            execution_time = self._stop_performance_timer(start_time)
            logger.info(
                f"Analysis complete: {self.name} (ID: {execution_id}), "
                f"execution time: {execution_time:.4f}s"
            )

            # --- Publish Analysis Completion Event (Success) ---
            if event_publisher and result:
                try:
                    payload = AnalysisCompletionPayload(
                        analysis_id=execution_id,
                        symbol=result.metadata.get("symbol", "unknown"), # Assuming symbol is in metadata
                        timeframe=result.metadata.get("timeframe", "unknown"), # Assuming timeframe is in metadata
                        status="completed",
                        results_summary=result.get_summary(), # Assuming AnalysisResult has a summary method
                        error_message=None
                    )
                    event = AnalysisCompletionEvent(payload=payload)
                    event_publisher.publish(topic=settings.KAFKA_ANALYSIS_TOPIC, event=event)
                except Exception as pub_exc:
                    logger.error(f"Failed to publish AnalysisCompletionEvent (Success) for {execution_id}: {pub_exc}", exc_info=True)
            # --- End Publish Event ---

            # Add performance metadata to result
            if result and hasattr(result, 'metadata'):
                result.metadata.update({
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "analyzer_name": self.name,
                    "timestamp": datetime.now().isoformat()
                })

            return result

        except Exception as e:
            execution_time = self._stop_performance_timer(start_time)
            self.performance_metrics["error_count"] += 1

            logger.error(
                f"Error in analyzer {self.name} (ID: {execution_id}), "
                f"execution time: {execution_time:.4f}s: {str(e)}"
            )
            logger.debug(f"Error details: {traceback.format_exc()}")

            # --- Publish Analysis Completion Event (Error) ---
            if event_publisher:
                try:
                    # Attempt to get symbol/timeframe if available in input data or context
                    # This part might need adjustment based on how 'data' is structured
                    symbol = getattr(data, 'symbol', 'unknown') if hasattr(data, 'symbol') else 'unknown'
                    timeframe = getattr(data, 'timeframe', 'unknown') if hasattr(data, 'timeframe') else 'unknown'

                    payload = AnalysisCompletionPayload(
                        analysis_id=execution_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        status="failed",
                        results_summary={"error": str(e)},
                        error_message=str(e)
                    )
                    event = AnalysisCompletionEvent(payload=payload)
                    event_publisher.publish(topic=settings.KAFKA_ANALYSIS_TOPIC, event=event)
                except Exception as pub_exc:
                    logger.error(f"Failed to publish AnalysisCompletionEvent (Error) for {execution_id}: {pub_exc}", exc_info=True)
            # --- End Publish Event ---

            # Create error result
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={
                    "error": f"Error in analysis: {str(e)}",
                    "execution_id": execution_id
                },
                is_valid=False,
                metadata={
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )

    async def log_effectiveness(
        self,
        analysis_result: AnalysisResult,
        actual_outcome: Dict[str, Any],
        timeframe: str = None,
        instrument: str = None
    ) -> str:
        """
        Log the effectiveness of an analysis result

        Args:
            analysis_result: The analysis result to evaluate
            actual_outcome: The actual market outcome
            timeframe: Optional timeframe of the analysis
            instrument: Optional instrument being analyzed

        Returns:
            ID of the effectiveness log entry
        """
        if not analysis_result or not actual_outcome:
            logger.warning(f"Cannot log effectiveness, missing result or outcome")
            return None

        # Extract prediction from analysis result
        prediction = await self._extract_prediction(analysis_result)

        # Compare prediction with actual outcome
        accuracy = await self._calculate_accuracy(prediction, actual_outcome)

        # Log the effectiveness
        log_id = self.effectiveness_logger.log(
            prediction=prediction,
            actual=actual_outcome,
            accuracy=accuracy,
            timeframe=timeframe,
            instrument=instrument,
            result_id=analysis_result.metadata.get("execution_id", None) if hasattr(analysis_result, "metadata") else None,
            parameters=self.parameters
        )

        return log_id

    async def _extract_prediction(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """
        Extract prediction from analysis result

        Each analyzer subclass should override this method to extract
        relevant prediction information based on its specific result structure

        Args:
            analysis_result: Analysis result to extract prediction from

        Returns:
            Dictionary containing prediction information
        """
        # Default implementation - override in subclasses
        # Return the entire result data as the prediction
        return analysis_result.result_data if hasattr(analysis_result, "result_data") else {}

    async def _calculate_accuracy(self, prediction: Dict[str, Any], actual: Dict[str, Any]) -> float:
        """
        Calculate accuracy of prediction compared to actual outcome

        Each analyzer subclass should override this method to implement
        appropriate accuracy calculation based on its specific predictions

        Args:
            prediction: Prediction dictionary
            actual: Actual outcome dictionary

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        # Default implementation - override in subclasses
        return 0.5  # Default to neutral accuracy

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this analyzer

        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics

    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics"""
        self._initialize_performance_metrics()
        self.execution_history = []
