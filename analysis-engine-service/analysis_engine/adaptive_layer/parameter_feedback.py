import asyncio
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats # Import for statistical analysis

# Placeholder for actual repository and event bus types
# from analysis_engine.repositories.parameter_feedback_repository import ParameterFeedbackRepositoryBase, ParameterChangeRecord
# from analysis_engine.integration.event_bus import EventBusBase

logger = logging.getLogger(__name__)

class ParameterFeedbackTracker:
    """
    Tracks how parameter changes affect strategy performance, records outcomes,
    performs statistical analysis, and generates adjustment recommendations.
    """

    # def __init__(self, db_repository: ParameterFeedbackRepositoryBase, event_bus: EventBusBase):
    def __init__(self, db_repository: Any, event_bus: Any):
        """
        Initializes the ParameterFeedbackTracker.

        Args:
            db_repository: The repository for storing and retrieving parameter feedback data.
            event_bus: The event bus for publishing and subscribing to events.
        """
        self.db_repository = db_repository
        self.event_bus = event_bus
        # Subscription logic is likely handled externally (e.g., in feedback_router.py)
        # based on the initial prompt's integration points.

    async def track_parameter_change(self, strategy_id: str, parameter_changes: Dict[str, Any], outcome: Dict[str, Any], timestamp: pd.Timestamp = None):
        """
        Records a parameter change and its associated outcome.

        Args:
            strategy_id: The ID of the strategy.
            parameter_changes: A dictionary representing the changed parameters and their new values.
            outcome: A dictionary representing the performance outcome after the change (e.g., {'profit': 100, 'sharpe': 1.2}).
            timestamp: Optional timestamp for the record. Defaults to now.
        """
        try:
            logger.info(f"Tracking parameter change for strategy {strategy_id}: {parameter_changes} -> {outcome}")
            record_timestamp = timestamp or pd.Timestamp.utcnow()
            # Store the parameter change and outcome in the database
            await self.db_repository.store_parameter_change(
                strategy_id=strategy_id,
                changes=parameter_changes,
                outcome=outcome,
                timestamp=record_timestamp
            )

            # Optionally, trigger analysis immediately or batch it
            # Consider if analysis should be triggered here or by a separate process/schedule
            # await self.analyze_parameter_effectiveness(strategy_id, list(parameter_changes.keys()))

            logger.info(f"Parameter change tracked successfully for strategy {strategy_id}.")
            # Optionally publish an event about the tracked change
            # await self.event_bus.publish("parameter_change_tracked", {"strategy_id": strategy_id, "changes": parameter_changes})

        except Exception as e:
            logger.error(f"Error tracking parameter change for strategy {strategy_id}: {e}", exc_info=True)

    async def analyze_parameter_effectiveness(self, strategy_id: str, parameters: List[str], performance_metric: str = 'profit') -> Dict[str, Dict[str, float]]:
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
            logger.info(f"Analyzing effectiveness of parameters {parameters} for strategy {strategy_id} using metric '{performance_metric}'")
            # Retrieve historical parameter changes and outcomes from the database
            # Assuming db_repository returns a list of records (e.g., ParameterChangeRecord dataclass instances or dicts)
            # Each record should have 'changes' (dict), 'outcome' (dict), and 'timestamp'
            history = await self.db_repository.get_parameter_history(strategy_id, parameters)

            if not history or len(history) < 5: # Need sufficient data for meaningful analysis
                 logger.warning(f"Insufficient data (<5 records) to analyze parameter effectiveness for strategy {strategy_id}.")
                 return {}

            # Prepare data for analysis
            data = []
            for record in history:
                row = {}
                # Flatten parameter changes into the row
                for param in parameters:
                    # Handle cases where a parameter might not be in every 'changes' dict
                    row[param] = record.get('changes', {}).get(param)
                # Add the performance metric from the outcome
                row[performance_metric] = record.get('outcome', {}).get(performance_metric)
                # Add timestamp if needed for time-series analysis (optional)
                # row['timestamp'] = record.get('timestamp')
                if row[performance_metric] is not None: # Only include records with the target metric
                    data.append(row)

            if len(data) < 5: # Check again after filtering for the metric
                logger.warning(f"Insufficient data (<5 records) with metric '{performance_metric}' for strategy {strategy_id}.")
                return {}

            df = pd.DataFrame(data)

            # Implement statistical analysis
            effectiveness_results = self._calculate_statistical_significance(df, parameters, performance_metric)

            logger.info(f"Parameter effectiveness analysis complete for strategy {strategy_id}: {effectiveness_results}")
            return effectiveness_results
        except Exception as e:
            logger.error(f"Error analyzing parameter effectiveness for strategy {strategy_id}: {e}", exc_info=True)
            return {}

    async def generate_parameter_recommendations(self, strategy_id: str, performance_metric: str = 'profit', significance_threshold: float = 0.05) -> Dict[str, str]:
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
            logger.info(f"Generating parameter recommendations for strategy {strategy_id} using metric '{performance_metric}'")
            # Get all parameters associated with the strategy (assuming a method exists)
            try:
                all_parameters = await self.db_repository.get_strategy_parameters(strategy_id)
                if not all_parameters:
                    logger.warning(f"No parameters found for strategy {strategy_id}. Cannot generate recommendations.")
                    return {}
            except AttributeError:
                 logger.error(f"Database repository does not have 'get_strategy_parameters' method. Cannot determine parameters for {strategy_id}.")
                 # Fallback: try to infer parameters from history, less reliable
                 history = await self.db_repository.get_parameter_history(strategy_id)
                 if not history: return {}
                 all_params_set = set()
                 for record in history:
                     all_params_set.update(record.get('changes', {}).keys())
                 all_parameters = list(all_params_set)
                 if not all_parameters: return {}


            # Analyze effectiveness for all parameters
            effectiveness_results = await self.analyze_parameter_effectiveness(strategy_id, all_parameters, performance_metric)

            if not effectiveness_results:
                logger.warning(f"No effectiveness results available for strategy {strategy_id}. Cannot generate recommendations.")
                return {}

            # Implement logic to generate recommendations based on significant results
            recommendations = self._generate_recommendations(effectiveness_results, significance_threshold)

            # TODO: Implement checks to avoid overfitting (e.g., cross-validation if modeling, hold-out set testing, regularization)
            # The current approach based on simple correlation is less prone to overfitting than complex models,
            # but recommendations should still be treated cautiously and validated.

            logger.info(f"Generated parameter recommendations for strategy {strategy_id}: {recommendations}")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating parameter recommendations for strategy {strategy_id}: {e}", exc_info=True)
            return {}

    def _calculate_statistical_significance(self, df: pd.DataFrame, parameters: List[str], performance_metric: str) -> Dict[str, Dict[str, float]]:
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
            logger.error(f"Performance metric '{performance_metric}' not found in DataFrame.")
            return {}

        target = df[performance_metric].dropna()
        if len(target) < 3: # Need at least 3 points for correlation
             return {}

        for param in parameters:
            if param not in df.columns:
                logger.warning(f"Parameter '{param}' not found in DataFrame for analysis.")
                continue

            # Align parameter data with available performance data
            param_data = df[param].loc[target.index].dropna()
            aligned_target = target.loc[param_data.index]

            if len(param_data) < 3 or len(aligned_target) < 3: # Check length after alignment and dropna
                logger.warning(f"Insufficient aligned data points (<3) for parameter '{param}' and metric '{performance_metric}'.")
                continue

            # Ensure data is numeric for correlation
            try:
                param_data = pd.to_numeric(param_data)
                aligned_target = pd.to_numeric(aligned_target)
            except ValueError:
                logger.warning(f"Could not convert parameter '{param}' or metric '{performance_metric}' to numeric. Skipping correlation.")
                continue

            # Check for constant series (no variance)
            if param_data.nunique() <= 1 or aligned_target.nunique() <= 1:
                 logger.warning(f"Parameter '{param}' or metric '{performance_metric}' has no variance. Skipping correlation.")
                 continue


            # Calculate Pearson correlation coefficient and p-value
            try:
                correlation, p_value = stats.pearsonr(param_data, aligned_target)
                if np.isnan(correlation) or np.isnan(p_value):
                     logger.warning(f"NaN result for correlation between '{param}' and '{performance_metric}'. Skipping.")
                     continue
                results[param] = {"correlation": round(correlation, 4), "p_value": round(p_value, 4)}
            except Exception as e:
                 logger.error(f"Error calculating correlation for {param}: {e}", exc_info=True)


        return results

    def _generate_recommendations(self, effectiveness_results: Dict[str, Dict[str, float]], significance_threshold: float) -> Dict[str, str]:
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

            if correlation is not None and p_value is not None and p_value < significance_threshold:
                if correlation > 0.1: # Threshold for positive correlation strength
                    recommendations[param] = "increase"
                elif correlation < -0.1: # Threshold for negative correlation strength
                    recommendations[param] = "decrease"
                # else: correlation is too weak, even if significant

        return recommendations

# Example usage (remains mostly the same, added type hints and metric)
async def main():
    class MockRepo:
        _history = []
        _params = ["param1", "param2", "param3"]
        async def store_parameter_change(self, strategy_id: str, changes: Dict, outcome: Dict, timestamp: pd.Timestamp):
            print(f"Storing change for {strategy_id}: {changes} -> {outcome} at {timestamp}")
            self._history.append({"strategy_id": strategy_id, "changes": changes, "outcome": outcome, "timestamp": timestamp})
        async def get_parameter_history(self, sid, params=None):
            print(f"Getting history for {sid}, params: {params}")
            # Simulate filtering if params are provided (basic example)
            # In reality, the DB query would handle filtering
            return [r for r in self._history if r['strategy_id'] == sid]
        async def get_strategy_parameters(self, sid):
             print(f"Getting params for {sid}")
             return self._params

    class MockEventBus:
        async def publish(self, topic, data): print(f"Publishing to {topic}: {data}")

    repo = MockRepo()
    bus = MockEventBus()
    tracker = ParameterFeedbackTracker(db_repository=repo, event_bus=bus)

    # Add some sample data
    await tracker.track_parameter_change("strat1", {"param1": 10, "param2": 0.5}, {"profit": 100, "sharpe": 1.2})
    await asyncio.sleep(0.1) # Ensure timestamps differ slightly
    await tracker.track_parameter_change("strat1", {"param1": 12, "param2": 0.5}, {"profit": 110, "sharpe": 1.3})
    await asyncio.sleep(0.1)
    await tracker.track_parameter_change("strat1", {"param1": 12, "param2": 0.4}, {"profit": 115, "sharpe": 1.4})
    await asyncio.sleep(0.1)
    await tracker.track_parameter_change("strat1", {"param1": 9, "param2": 0.4}, {"profit": 95, "sharpe": 1.1})
    await asyncio.sleep(0.1)
    await tracker.track_parameter_change("strat1", {"param1": 11, "param2": 0.6}, {"profit": 105, "sharpe": 1.25})


    # Analyze and recommend
    analysis = await tracker.analyze_parameter_effectiveness("strat1", ["param1", "param2"], performance_metric='profit')
    print(f"Analysis Results: {analysis}")

    recs = await tracker.generate_parameter_recommendations("strat1", performance_metric='profit', significance_threshold=0.1) # Looser threshold for example
    print(f"Recommendations: {recs}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
