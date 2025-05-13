"""
Parameter feedback module.

This module provides functionality for...
"""

import asyncio
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ParameterFeedbackTracker:
    """
    Tracks how parameter changes affect strategy performance, records outcomes,
    performs statistical analysis, and generates adjustment recommendations.
    """

    def __init__(self, db_repository: Any, event_bus: Any):
        """
        Initializes the ParameterFeedbackTracker.

        Args:
            db_repository: The repository for storing and retrieving parameter feedback data.
            event_bus: The event bus for publishing and subscribing to events.
        """
        self.db_repository = db_repository
        self.event_bus = event_bus

    @async_with_exception_handling
    async def track_parameter_change(self, strategy_id: str,
        parameter_changes: Dict[str, Any], outcome: Dict[str, Any],
        timestamp: pd.Timestamp=None):
        """
        Records a parameter change and its associated outcome.

        Args:
            strategy_id: The ID of the strategy.
            parameter_changes: A dictionary representing the changed parameters and their new values.
            outcome: A dictionary representing the performance outcome after the change (e.g., {'profit': 100, 'sharpe': 1.2}).
            timestamp: Optional timestamp for the record. Defaults to now.
        """
        try:
            logger.info(
                f'Tracking parameter change for strategy {strategy_id}: {parameter_changes} -> {outcome}'
                )
            record_timestamp = timestamp or pd.Timestamp.utcnow()
            await self.db_repository.store_parameter_change(strategy_id=
                strategy_id, changes=parameter_changes, outcome=outcome,
                timestamp=record_timestamp)
            logger.info(
                f'Parameter change tracked successfully for strategy {strategy_id}.'
                )
        except Exception as e:
            logger.error(
                f'Error tracking parameter change for strategy {strategy_id}: {e}'
                , exc_info=True)

    @with_analysis_resilience('analyze_parameter_effectiveness')
    @async_with_exception_handling
    async def analyze_parameter_effectiveness(self, strategy_id: str,
        parameters: List[str], performance_metric: str='profit') ->Dict[str,
        Dict[str, float]]:
        """
        Performs statistical analysis to determine the effectiveness of specific parameters
        on a given performance metric.

        Args:
            strategy_id: The ID of the strategy.
            parameters: A list of parameter names to analyze.
            performance_metric: The key in the 'outcome' dictionary to use for effectiveness analysis (e.g., 'profit', 'sharpe').

        Returns:
            A dictionary mapping parameter names to their effectiveness analysis results
            (e.g., {'correlation': 0.7, 'p_value': 0.02}).
            Returns empty dict if analysis cannot be performed.
        """
        try:
            logger.info(
                f"Analyzing effectiveness of parameters {parameters} for strategy {strategy_id} using metric '{performance_metric}'"
                )
            history = await self.db_repository.get_parameter_history(
                strategy_id, parameters)
            if not history or len(history) < 5:
                logger.warning(
                    f'Insufficient data (<5 records) to analyze parameter effectiveness for strategy {strategy_id}.'
                    )
                return {}
            data = []
            for record in history:
                row = {}
                for param in parameters:
                    row[param] = record.get('changes', {}).get(param)
                row[performance_metric] = record.get('outcome', {}).get(
                    performance_metric)
                if row[performance_metric] is not None:
                    data.append(row)
            if len(data) < 5:
                logger.warning(
                    f"Insufficient data (<5 records) with metric '{performance_metric}' for strategy {strategy_id}."
                    )
                return {}
            df = pd.DataFrame(data)
            effectiveness_results = self._calculate_statistical_significance(df
                , parameters, performance_metric)
            logger.info(
                f'Parameter effectiveness analysis complete for strategy {strategy_id}: {effectiveness_results}'
                )
            return effectiveness_results
        except Exception as e:
            logger.error(
                f'Error analyzing parameter effectiveness for strategy {strategy_id}: {e}'
                , exc_info=True)
            return {}

    @async_with_exception_handling
    async def generate_parameter_recommendations(self, strategy_id: str,
        performance_metric: str='profit', significance_threshold: float=0.05
        ) ->Dict[str, str]:
        """
        Generates recommendations for parameter adjustments based on historical effectiveness.

        Args:
            strategy_id: The ID of the strategy.
            performance_metric: The performance metric used for analysis.
            significance_threshold: The p-value threshold to consider a result statistically significant.

        Returns:
            A dictionary containing recommended parameter adjustments (e.g., {'param1': 'increase', 'param2': 'decrease'}).
        """
        try:
            logger.info(
                f"Generating parameter recommendations for strategy {strategy_id} using metric '{performance_metric}'"
                )
            try:
                all_parameters = (await self.db_repository.
                    get_strategy_parameters(strategy_id))
                if not all_parameters:
                    logger.warning(
                        f'No parameters found for strategy {strategy_id}. Cannot generate recommendations.'
                        )
                    return {}
            except AttributeError:
                logger.error(
                    f"Database repository does not have 'get_strategy_parameters' method. Cannot determine parameters for {strategy_id}."
                    )
                history = await self.db_repository.get_parameter_history(
                    strategy_id)
                if not history:
                    return {}
                all_params_set = set()
                for record in history:
                    all_params_set.update(record.get('changes', {}).keys())
                all_parameters = list(all_params_set)
                if not all_parameters:
                    return {}
            effectiveness_results = await self.analyze_parameter_effectiveness(
                strategy_id, all_parameters, performance_metric)
            if not effectiveness_results:
                logger.warning(
                    f'No effectiveness results available for strategy {strategy_id}. Cannot generate recommendations.'
                    )
                return {}
            recommendations = self._generate_recommendations(
                effectiveness_results, significance_threshold)
            logger.info(
                f'Generated parameter recommendations for strategy {strategy_id}: {recommendations}'
                )
            return recommendations
        except Exception as e:
            logger.error(
                f'Error generating parameter recommendations for strategy {strategy_id}: {e}'
                , exc_info=True)
            return {}

    @with_exception_handling
    def _calculate_statistical_significance(self, df: pd.DataFrame,
        parameters: List[str], performance_metric: str) ->Dict[str, Dict[
        str, float]]:
        """
        Calculates the statistical significance (correlation and p-value) between parameters and performance.

        Args:
            df: DataFrame containing parameter values and the performance metric.
            parameters: List of parameter columns to analyze.
            performance_metric: The column name of the performance metric.

        Returns:
            A dictionary mapping parameter names to their analysis results {'correlation': float, 'p_value': float}.
        """
        results = {}
        if performance_metric not in df.columns:
            logger.error(
                f"Performance metric '{performance_metric}' not found in DataFrame."
                )
            return {}
        target = df[performance_metric].dropna()
        if len(target) < 3:
            return {}
        for param in parameters:
            if param not in df.columns:
                logger.warning(
                    f"Parameter '{param}' not found in DataFrame for analysis."
                    )
                continue
            param_data = df[param].loc[target.index].dropna()
            aligned_target = target.loc[param_data.index]
            if len(param_data) < 3 or len(aligned_target) < 3:
                logger.warning(
                    f"Insufficient aligned data points (<3) for parameter '{param}' and metric '{performance_metric}'."
                    )
                continue
            try:
                param_data = pd.to_numeric(param_data)
                aligned_target = pd.to_numeric(aligned_target)
            except ValueError:
                logger.warning(
                    f"Could not convert parameter '{param}' or metric '{performance_metric}' to numeric. Skipping correlation."
                    )
                continue
            if param_data.nunique() <= 1 or aligned_target.nunique() <= 1:
                logger.warning(
                    f"Parameter '{param}' or metric '{performance_metric}' has no variance. Skipping correlation."
                    )
                continue
            try:
                correlation, p_value = stats.pearsonr(param_data,
                    aligned_target)
                if np.isnan(correlation) or np.isnan(p_value):
                    logger.warning(
                        f"NaN result for correlation between '{param}' and '{performance_metric}'. Skipping."
                        )
                    continue
                results[param] = {'correlation': round(correlation, 4),
                    'p_value': round(p_value, 4)}
            except Exception as e:
                logger.error(f'Error calculating correlation for {param}: {e}',
                    exc_info=True)
        return results

    def _generate_recommendations(self, effectiveness_results: Dict[str,
        Dict[str, float]], significance_threshold: float) ->Dict[str, str]:
        """
        Generates adjustment recommendations based on statistically significant correlations.

        Args:
            effectiveness_results: The output from _calculate_statistical_significance.
            significance_threshold: The p-value threshold.

        Returns:
            A dictionary of recommendations (e.g., {'param': 'increase'}).
        """
        recommendations = {}
        for param, results in effectiveness_results.items():
            correlation = results.get('correlation')
            p_value = results.get('p_value')
            if (correlation is not None and p_value is not None and p_value <
                significance_threshold):
                if correlation > 0.1:
                    recommendations[param] = 'increase'
                elif correlation < -0.1:
                    recommendations[param] = 'decrease'
        return recommendations


async def main():
    """
    Main.
    
    """



    class MockRepo:
    """
    MockRepo class.
    
    Attributes:
        Add attributes here
    """

        _history = []
        _params = ['param1', 'param2', 'param3']

        async def store_parameter_change(self, strategy_id: str, changes:
            Dict, outcome: Dict, timestamp: pd.Timestamp):
    """
    Store parameter change.
    
    Args:
        strategy_id: Description of strategy_id
        changes: Description of changes
        outcome: Description of outcome
        timestamp: Description of timestamp
    
    """

            print(
                f'Storing change for {strategy_id}: {changes} -> {outcome} at {timestamp}'
                )
            self._history.append({'strategy_id': strategy_id, 'changes':
                changes, 'outcome': outcome, 'timestamp': timestamp})

        @with_resilience('get_parameter_history')
        async def get_parameter_history(self, sid, params=None):
    """
    Get parameter history.
    
    Args:
        sid: Description of sid
        params: Description of params
    
    """

            print(f'Getting history for {sid}, params: {params}')
            return [r for r in self._history if r['strategy_id'] == sid]

        @with_resilience('get_strategy_parameters')
        async def get_strategy_parameters(self, sid):
    """
    Get strategy parameters.
    
    Args:
        sid: Description of sid
    
    """

            print(f'Getting params for {sid}')
            return self._params


    class MockEventBus:
    """
    MockEventBus class.
    
    Attributes:
        Add attributes here
    """


        async def publish(self, topic, data):
    """
    Publish.
    
    Args:
        topic: Description of topic
        data: Description of data
    
    """

            print(f'Publishing to {topic}: {data}')
    repo = MockRepo()
    bus = MockEventBus()
    tracker = ParameterFeedbackTracker(db_repository=repo, event_bus=bus)
    await tracker.track_parameter_change('strat1', {'param1': 10, 'param2':
        0.5}, {'profit': 100, 'sharpe': 1.2})
    await asyncio.sleep(0.1)
    await tracker.track_parameter_change('strat1', {'param1': 12, 'param2':
        0.5}, {'profit': 110, 'sharpe': 1.3})
    await asyncio.sleep(0.1)
    await tracker.track_parameter_change('strat1', {'param1': 12, 'param2':
        0.4}, {'profit': 115, 'sharpe': 1.4})
    await asyncio.sleep(0.1)
    await tracker.track_parameter_change('strat1', {'param1': 9, 'param2': 
        0.4}, {'profit': 95, 'sharpe': 1.1})
    await asyncio.sleep(0.1)
    await tracker.track_parameter_change('strat1', {'param1': 11, 'param2':
        0.6}, {'profit': 105, 'sharpe': 1.25})
    analysis = await tracker.analyze_parameter_effectiveness('strat1', [
        'param1', 'param2'], performance_metric='profit')
    print(f'Analysis Results: {analysis}')
    recs = await tracker.generate_parameter_recommendations('strat1',
        performance_metric='profit', significance_threshold=0.1)
    print(f'Recommendations: {recs}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
