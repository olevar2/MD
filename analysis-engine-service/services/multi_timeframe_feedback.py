"""
Multi-Timeframe Feedback Service

Handles prediction feedback across different timeframes, performs correlation analysis,
detects leading indicators, and generates recommendations for optimal timeframe usage.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
from core_foundations.models.feedback import TradeFeedback
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_schema import Event, EventType
from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.feedback_exceptions import FeedbackProcessingError, TimeframeFeedbackError
logger = get_logger(__name__)
SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MultiTimeframeFeedbackService:
    """
    Service for handling and analyzing prediction feedback across multiple timeframes.

    Responsibilities include:
    - Processing and storing feedback for various timeframes.
    - Calculating correlations between timeframe prediction errors.
    - Identifying leading timeframes based on lagged correlations.
    - Generating recommendations for optimal timeframe selection.
    - Calculating weighted prediction scores across timeframes.
    """

    def __init__(self, event_publisher: EventPublisher, config: Optional[
        Dict[str, Any]]=None):
        """Initialize the MultiTimeframeFeedbackService.

        Args:
            feedback_repository: Repository for storing and retrieving feedback data.
            event_publisher: Publisher for sending events (e.g., Kafka).
            config: Configuration dictionary for service parameters.
        """
        self.event_publisher = event_publisher
        self.config = config or {}
        self._set_default_config()
        self.recent_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        logger.info('MultiTimeframeFeedbackService initialized.')

    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {'max_data_points_per_timeframe': 2000,
            'correlation_min_periods': 50, 'lead_lag_max_shift': 10,
            'significant_correlation_threshold': 0.5,
            'leading_indicator_threshold': 0.6,
            'analysis_lookback_period_hours': 24 * 7,
            'publish_event_on_correlation': True, 'timeframe_weights': {
            '1m': 0.1, '5m': 0.15, '15m': 0.2, '1h': 0.25, '4h': 0.15, '1d':
            0.15}}
        for key, value in defaults.items():
            self.config.setdefault(key, value)
        total_weight = sum(self.config['timeframe_weights'].get(tf, 0) for
            tf in SUPPORTED_TIMEFRAMES)
        if total_weight > 0:
            self.config['timeframe_weights'] = {tf: (w / total_weight) for 
                tf, w in self.config['timeframe_weights'].items() if tf in
                SUPPORTED_TIMEFRAMES}
        else:
            self.config['timeframe_weights'] = {tf: (1.0 / len(
                SUPPORTED_TIMEFRAMES)) for tf in SUPPORTED_TIMEFRAMES}

    async def register_event_handlers(self):
        """Register handlers for relevant Kafka events.
        This might be called externally or during service startup.
        Example: Listen for new prediction feedback events.
        """
        logger.info('Event handlers registered (if applicable).')

    @async_with_exception_handling
    async def handle_feedback_event(self, event_data: Dict[str, Any]):
        """Process incoming feedback events from the event bus.
        """
        try:
            feedback = TradeFeedback()
            timeframe = event_data.get('timeframe')
            if timeframe not in SUPPORTED_TIMEFRAMES:
                logger.warning(
                    f'Received feedback for unsupported timeframe: {timeframe}. Skipping.'
                    )
                return
            await self.process_feedback(feedback)
        except Exception as e:
            logger.error(f'Error handling feedback event: {e}', exc_info=True)

    @with_database_resilience('process_feedback')
    @async_with_exception_handling
    async def process_feedback(self, feedback: TradeFeedback):
        """Process and store a single piece of trade feedback.

        Args:
            feedback: The TradeFeedback object.
        """
        try:
            instrument = getattr(feedback, 'instrument', None)
            timeframe = getattr(feedback, 'metadata', {}).get('timeframe')
            timestamp = getattr(feedback, 'timestamp', None)
            error = getattr(feedback, 'error_magnitude', None)
            if not all([instrument, timeframe, timestamp, error is not None]):
                logger.warning(
                    f'Incomplete feedback received: {feedback.id}. Skipping.')
                return
            if timeframe not in SUPPORTED_TIMEFRAMES:
                logger.warning(
                    f"Feedback for unsupported timeframe '{timeframe}' received: {feedback.id}. Skipping."
                    )
                return
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            if instrument not in self.recent_data:
                self.recent_data[instrument] = {tf: pd.DataFrame(columns=[
                    'timestamp', 'error']).set_index('timestamp') for tf in
                    SUPPORTED_TIMEFRAMES}
            new_data = pd.DataFrame([{'timestamp': timestamp, 'error': error}]
                ).set_index('timestamp')
            self.recent_data[instrument][timeframe] = pd.concat([self.
                recent_data[instrument][timeframe], new_data])
            df = self.recent_data[instrument][timeframe]
            if len(df) > self.config['max_data_points_per_timeframe']:
                self.recent_data[instrument][timeframe] = df.iloc[-self.
                    config['max_data_points_per_timeframe']:]
            logger.debug(
                f'Processed feedback {feedback.id} for {instrument} on {timeframe}.'
                )
        except Exception as e:
            logger.error(
                f"Error processing feedback {getattr(feedback, 'id', 'N/A')}: {e}"
                , exc_info=True)
            raise FeedbackProcessingError(f'Failed to process feedback: {e}')

    @with_analysis_resilience('analyze_correlations')
    @async_with_exception_handling
    async def analyze_correlations(self, instrument: str) ->Optional[pd.
        DataFrame]:
        """Perform correlation analysis between prediction errors of different timeframes
        for a given instrument using the most recent data.

        Args:
            instrument: The instrument symbol (e.g., 'EUR_USD').

        Returns:
            A pandas DataFrame representing the correlation matrix, or None if analysis fails.
        """
        logger.info(f'Starting correlation analysis for {instrument}...')
        if instrument not in self.recent_data:
            logger.warning(
                f'No recent data found for instrument {instrument} to analyze correlations.'
                )
            return None
        try:
            instrument_data = self.recent_data[instrument]
            error_series = {}
            min_common_timestamp = None
            max_common_timestamp = None
            for tf in SUPPORTED_TIMEFRAMES:
                df = instrument_data.get(tf)
                if df is not None and not df.empty and len(df) >= self.config[
                    'correlation_min_periods']:
                    df = df.sort_index()
                    error_series[tf] = df['error']
                    current_min = df.index.min()
                    current_max = df.index.max()
                    if (min_common_timestamp is None or current_min >
                        min_common_timestamp):
                        min_common_timestamp = current_min
                    if (max_common_timestamp is None or current_max <
                        max_common_timestamp):
                        max_common_timestamp = current_max
                else:
                    logger.debug(
                        f'Skipping timeframe {tf} for {instrument} due to insufficient data ({len(df) if df is not None else 0} points).'
                        )
            if len(error_series) < 2:
                logger.warning(
                    f'Insufficient timeframes with enough data for correlation analysis on {instrument}.'
                    )
                return None
            aligned_data = {}
            for tf, series in error_series.items():
                series_filtered = series[(series.index >=
                    min_common_timestamp) & (series.index <=
                    max_common_timestamp)]
                if len(series_filtered) >= self.config[
                    'correlation_min_periods']:
                    aligned_data[tf] = series_filtered
                else:
                    logger.debug(
                        f'Timeframe {tf} for {instrument} has insufficient data in the common range.'
                        )
            if len(aligned_data) < 2:
                logger.warning(
                    f'Insufficient timeframes after aligning to common time range for {instrument}.'
                    )
                return None
            combined_df = pd.DataFrame(aligned_data).ffill().dropna()
            if len(combined_df) < self.config['correlation_min_periods']:
                logger.warning(
                    f'Insufficient overlapping data points ({len(combined_df)}) after alignment for {instrument}.'
                    )
                return None
            correlation_matrix = combined_df.corr(method='pearson')
            if instrument not in self.analysis_results:
                self.analysis_results[instrument] = {}
            self.analysis_results[instrument]['correlations'
                ] = correlation_matrix
            logger.info(
                f'Successfully calculated correlation matrix for {instrument}.'
                )
            if self.config['publish_event_on_correlation']:
                await self._publish_analysis_event(instrument,
                    'correlation_analysis_completed', {'correlation_matrix':
                    correlation_matrix.to_dict()})
            return correlation_matrix
        except Exception as e:
            logger.error(
                f'Error during correlation analysis for {instrument}: {e}',
                exc_info=True)
            raise TimeframeFeedbackError(
                f'Correlation analysis failed for {instrument}: {e}')

    @async_with_exception_handling
    async def detect_leading_timeframes(self, instrument: str) ->Optional[Dict
        [str, Dict[str, float]]]:
        """Detect which timeframes act as leading indicators for others based on lagged correlations.

        Args:
            instrument: The instrument symbol.

        Returns:
            A dictionary where keys are timeframes and values are dicts of
            {leading_timeframe: max_correlation_value} indicating the best lead, or None.
        """
        logger.info(f'Starting leading timeframe detection for {instrument}...'
            )
        if instrument not in self.recent_data:
            logger.warning(
                f'No recent data for {instrument} to detect leading timeframes.'
                )
            return None
        try:
            instrument_data = self.recent_data[instrument]
            prepared_series = {}
            for tf in SUPPORTED_TIMEFRAMES:
                df = instrument_data.get(tf)
                if df is not None and not df.empty and len(df) >= self.config[
                    'correlation_min_periods']:
                    prepared_series[tf] = df['error'].sort_index()
            if len(prepared_series) < 2:
                logger.warning(
                    f'Insufficient timeframes with data for lead/lag analysis on {instrument}.'
                    )
                return None
            lead_lag_results = {}
            max_shift = self.config['lead_lag_max_shift']
            timeframes = list(prepared_series.keys())
            for i, target_tf in enumerate(timeframes):
                target_series = prepared_series[target_tf]
                best_lead = {'timeframe': None, 'correlation': -1.0, 'lag': 0}
                for j, potential_lead_tf in enumerate(timeframes):
                    if i == j:
                        continue
                    lead_series = prepared_series[potential_lead_tf]
                    max_corr = -1.0
                    best_lag = 0
                    for lag in range(1, max_shift + 1):
                        shifted_lead = lead_series.shift(lag)
                        combined = pd.DataFrame({'target': target_series,
                            'shifted_lead': shifted_lead}).dropna()
                        if len(combined) >= self.config[
                            'correlation_min_periods']:
                            corr = combined['target'].corr(combined[
                                'shifted_lead'])
                            if corr > max_corr:
                                max_corr = corr
                                best_lag = lag
                    if max_corr > self.config['leading_indicator_threshold'
                        ] and max_corr > best_lead['correlation']:
                        best_lead = {'timeframe': potential_lead_tf,
                            'correlation': max_corr, 'lag': best_lag}
                if best_lead['timeframe']:
                    lead_lag_results[target_tf] = {'leading_timeframe':
                        best_lead['timeframe'], 'max_correlation':
                        best_lead['correlation'], 'optimal_lag': best_lead[
                        'lag']}
                    logger.debug(
                        f"Detected {best_lead['timeframe']} as potential lead for {target_tf} (Corr: {best_lead['correlation']:.3f}, Lag: {best_lead['lag']}) for {instrument}."
                        )
            if instrument not in self.analysis_results:
                self.analysis_results[instrument] = {}
            self.analysis_results[instrument]['lead_lag'] = lead_lag_results
            logger.info(
                f'Successfully completed leading timeframe detection for {instrument}.'
                )
            await self._publish_analysis_event(instrument,
                'lead_lag_analysis_completed', {'lead_lag_results':
                lead_lag_results})
            return lead_lag_results
        except Exception as e:
            logger.error(
                f'Error during leading timeframe detection for {instrument}: {e}'
                , exc_info=True)
            raise TimeframeFeedbackError(
                f'Leading timeframe detection failed for {instrument}: {e}')

    async def generate_timeframe_recommendations(self, instrument: str) ->Dict[
        str, Any]:
        """Generate recommendations for optimal timeframe usage based on analysis results.

        Args:
            instrument: The instrument symbol.

        Returns:
            A dictionary containing recommendations.
        """
        logger.info(f'Generating timeframe recommendations for {instrument}...'
            )
        recommendations = {'instrument': instrument, 'timestamp': datetime.
            utcnow().isoformat(), 'optimal_timeframes': [],
            'leading_indicators': {}, 'warnings': [], 'notes': []}
        if instrument not in self.analysis_results:
            recommendations['warnings'].append(
                'Analysis results not available. Run analysis first.')
            logger.warning(
                f'Cannot generate recommendations for {instrument}: Analysis results missing.'
                )
            return recommendations
        analysis = self.analysis_results[instrument]
        correlations = analysis.get('correlations')
        lead_lag = analysis.get('lead_lag', {})
        if correlations is not None:
            avg_abs_corr = correlations.abs().mean(axis=1)
            low_corr_tfs = avg_abs_corr[avg_abs_corr < 1.0 / len(
                correlations.columns) + 0.1].index.tolist()
            if low_corr_tfs:
                recommendations['notes'].append(
                    f'Timeframes {low_corr_tfs} show lower average correlation, potentially offering unique insights.'
                    )
                recommendations['optimal_timeframes'].extend(tf for tf in
                    low_corr_tfs if tf not in recommendations[
                    'optimal_timeframes'])
        lead_counts = pd.Series({tf_info['leading_timeframe']: (0) for tf,
            tf_info in lead_lag.items()})
        for tf, tf_info in lead_lag.items():
            lead_counts[tf_info['leading_timeframe']] += 1
        strong_leads = lead_counts[lead_counts >= 1].index.tolist()
        if strong_leads:
            recommendations['notes'].append(
                f'Timeframes {strong_leads} frequently act as leading indicators.'
                )
            recommendations['leading_indicators'] = {tf: lead_lag[target_tf
                ] for target_tf, tf_info in lead_lag.items() if (tf :=
                tf_info['leading_timeframe']) in strong_leads}
            recommendations['optimal_timeframes'].extend(tf for tf in
                strong_leads if tf not in recommendations['optimal_timeframes']
                )
        sorted_weights = sorted(self.config['timeframe_weights'].items(),
            key=lambda item: item[1], reverse=True)
        top_weighted_tfs = [tf for tf, w in sorted_weights[:2]]
        recommendations['notes'].append(
            f'Timeframes {top_weighted_tfs} have the highest configured weights.'
            )
        recommendations['optimal_timeframes'].extend(tf for tf in
            top_weighted_tfs if tf not in recommendations['optimal_timeframes']
            )
        if instrument in self.recent_data:
            available_tfs = [tf for tf, df in self.recent_data[instrument].
                items() if df is not None and not df.empty]
            recommendations['notes'].append(
                f'Sufficient recent data available for timeframes: {available_tfs}'
                )
        else:
            recommendations['warnings'].append(
                'No recent data found for the instrument.')
        recommendations['optimal_timeframes'] = sorted(list(set(
            recommendations['optimal_timeframes'])), key=lambda tf:
            SUPPORTED_TIMEFRAMES.index(tf))
        if not recommendations['optimal_timeframes']:
            recommendations['notes'].append(
                'No strong indicators for optimal timeframes found based on current analysis. Consider default short/medium term.'
                )
            recommendations['optimal_timeframes'] = ['5m', '1h']
        logger.info(
            f"Generated recommendations for {instrument}: Optimal={recommendations['optimal_timeframes']}"
            )
        await self._publish_analysis_event(instrument,
            'recommendations_generated', {'recommendations': recommendations})
        return recommendations

    @with_analysis_resilience('calculate_weighted_score')
    async def calculate_weighted_score(self, instrument: str,
        prediction_scores: Dict[str, float]) ->Optional[float]:
        """
        Calculate a single weighted prediction score across multiple timeframes.

        Args:
            instrument: The instrument symbol.
            prediction_scores: A dictionary mapping timeframes to their individual
                               prediction scores (e.g., accuracy, confidence).

        Returns:
            The calculated weighted score, or None if calculation is not possible.
        """
        logger.debug(
            f'Calculating weighted score for {instrument} with scores: {prediction_scores}'
            )
        weighted_score = 0.0
        total_weight_used = 0.0
        timeframe_weights = self.config['timeframe_weights']
        for timeframe, score in prediction_scores.items():
            if timeframe in timeframe_weights:
                weight = timeframe_weights[timeframe]
                weighted_score += score * weight
                total_weight_used += weight
            else:
                logger.warning(
                    f'No weight configured for timeframe {timeframe} in weighted score calculation for {instrument}.'
                    )
        if total_weight_used > 0 and not np.isclose(total_weight_used, 1.0):
            logger.debug(
                f'Normalizing weighted score for {instrument}. Total weight used: {total_weight_used:.3f}'
                )
            weighted_score /= total_weight_used
        elif total_weight_used == 0:
            logger.warning(
                f'Could not calculate weighted score for {instrument}: No valid timeframes with weights found in input scores.'
                )
            return None
        logger.info(
            f'Calculated weighted score for {instrument}: {weighted_score:.4f}'
            )
        return weighted_score

    @async_with_exception_handling
    async def _publish_analysis_event(self, instrument: str, event_subtype:
        str, data: Dict[str, Any]):
        """Helper to publish analysis-related events."""
        if not self.event_publisher:
            return
        event_type = EventType.FEEDBACK_ANALYSIS
        event_payload = {'instrument': instrument, 'analysis_type':
            event_subtype, 'timestamp': datetime.utcnow().isoformat(), **data}
        try:
            await self.event_publisher.publish(topic=
                f'feedback.analysis.{event_subtype.lower()}', event_data=
                event_payload)
            logger.debug(f"Published event '{event_subtype}' for {instrument}."
                )
        except Exception as e:
            logger.error(
                f'Failed to publish event {event_subtype} for {instrument}: {e}'
                , exc_info=True)
