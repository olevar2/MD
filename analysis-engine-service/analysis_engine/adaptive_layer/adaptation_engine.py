"""
Adaptation Engine

Responsible for executing adaptation actions based on decisions made by the
FeedbackLoop or other adaptive processes.

This could involve:
- Triggering model retraining pipelines.
- Calling APIs to update strategy parameters in the execution engine.
- Initiating the deployment of mutated or newly optimized strategies.
- Adjusting resource allocation for specific models or strategies.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
import tempfile
import os
from core_foundations.utils.logger import get_logger
from core_foundations.resilience.circuit_breaker import CircuitBreakerState
from core_foundations.exceptions.client_exceptions import MLClientConnectionError, MLJobSubmissionError, ExecutionEngineConnectionError, StrategyDeploymentError, BacktestError
from ..clients.ml_pipeline_client import MLPipelineClient
from ..clients.execution_engine_client import ExecutionEngineClient
from ..adapters.strategy_execution_adapter import StrategyExecutorAdapter
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AdaptationEngine:
    """
    Executes concrete adaptation steps.
    """

    def __init__(self, ml_pipeline_client: MLPipelineClient=None,
        execution_engine_client: ExecutionEngineClient=None, config:
        Optional[Dict[str, Any]]=None):
        """
        Initializes the AdaptationEngine.

        Args:
            ml_pipeline_client: Client for ML pipeline operations
            execution_engine_client: Client for strategy execution engine operations
            config: Configuration parameters for the engine and its clients.
        """
        self.config = config or {}
        self.ml_client = ml_pipeline_client
        self.exec_client = execution_engine_client
        self.backtest_required = self.config_manager.get('backtest_required', True)
        self.backtest_min_sharpe = self.config_manager.get('backtest_min_sharpe', 1.0)
        self.backtest_max_drawdown = self.config.get('backtest_max_drawdown',
            15.0)
        self.backtest_min_win_rate = self.config.get('backtest_min_win_rate',
            0.5)
        logger.info('AdaptationEngine initialized.')

    @async_with_exception_handling
    async def trigger_model_retraining(self, model_id: str,
        retraining_params: Optional[Dict[str, Any]]=None):
        """
        Initiates a retraining job for a specific ML model.

        Args:
            model_id: The identifier of the model to retrain.
            retraining_params: Optional dictionary of parameters for the retraining job.

        Returns:
            str: Job ID for the retraining job if successful, None otherwise

        Raises:
            MLJobSubmissionError: If there was an error submitting the job
        """
        logger.info(
            f'Received request to trigger retraining for model: {model_id}')
        if not self.ml_client:
            logger.error(
                f'Cannot trigger retraining for model {model_id}: ML client not initialized'
                )
            return None
        try:
            job_id = await self.ml_client.start_retraining_job(model_id,
                retraining_params)
            if job_id:
                logger.info(
                    f'Successfully triggered retraining job for model {model_id}. Job ID: {job_id}'
                    )
                return job_id
            else:
                logger.error(
                    f'Failed to get job ID for model retraining: {model_id}')
                return None
        except MLClientConnectionError as e:
            logger.error(
                f'Connection error while triggering retraining for model {model_id}: {e}'
                , exc_info=True)
            return None
        except MLJobSubmissionError as e:
            logger.error(f'Job submission error for model {model_id}: {e}',
                exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f'Unexpected error while triggering retraining for model {model_id}: {e}'
                , exc_info=True)
            return None

    @with_resilience('update_strategy_parameter')
    @async_with_exception_handling
    async def update_strategy_parameter(self, strategy_id: str,
        parameter_name: str, new_value: Any):
        """
        Updates a specific parameter for a running strategy.

        Args:
            strategy_id: The identifier of the strategy to update.
            parameter_name: The name of the parameter to change.
            new_value: The new value for the parameter.
        """
        logger.info(
            f"Received request to update parameter '{parameter_name}' for strategy: {strategy_id} to {new_value}"
            )
        try:
            logger.warning(
                f'Placeholder: Updating parameter {parameter_name} for strategy {strategy_id} to {new_value}'
                )
            await asyncio.sleep(0.1)
            return True
        except Exception as e:
            logger.error(
                f'Failed to update parameter for strategy {strategy_id}: {e}',
                exc_info=True)
            return False

    @async_with_exception_handling
    async def run_backtest(self, strategy_id: str, config: Dict[str, Any]
        ) ->Dict[str, Any]:
        """
        Run backtest for a strategy configuration using the real backtesting framework.

        Args:
            strategy_id: The identifier of the strategy to backtest
            config: The strategy configuration to test

        Returns:
            Dict[str, Any]: Backtest results including metrics
        """
        logger.info(f'Running backtest for strategy {strategy_id}')
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix
                ='.json') as tmp:
                if 'id' not in config:
                    config['id'] = strategy_id
                json.dump(config, tmp)
                config_path = tmp.name
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime(
                '%Y-%m-%d')
            if 'backtest' in config:
                backtest_config = config['backtest']
                if 'lookback_days' in backtest_config:
                    start_date = (datetime.now() - timedelta(days=
                        backtest_config['lookback_days'])).strftime('%Y-%m-%d')
                if 'start_date' in backtest_config:
                    start_date = backtest_config['start_date']
                if 'end_date' in backtest_config:
                    end_date = backtest_config['end_date']
            assets = config_manager.get('instruments', None)
            logger.info(
                f'Running backtest for strategy {strategy_id} from {start_date} to {end_date}'
                )
            strategy_executor = StrategyExecutorAdapter()
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d'
                ) if isinstance(start_date, str) else start_date
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d'
                ) if isinstance(end_date, str) else end_date
            result = await strategy_executor.backtest_strategy(strategy_id=
                strategy_id, symbol=assets[0] if assets and isinstance(
                assets, list) else 'EURUSD', timeframe='1h', start_date=
                start_date_obj, end_date=end_date_obj, parameters=config)
            try:
                os.unlink(config_path)
            except Exception as e:
                logger.warning(
                    f'Failed to remove temporary config file {config_path}: {e}'
                    )
            if result['success']:
                logger.info(
                    f'Backtest completed successfully for strategy {strategy_id}'
                    )
                logger.info(
                    f"Key metrics - Sharpe: {result.get('metrics', {}).get('sharpe_ratio', 'N/A')}, Max Drawdown: {result.get('metrics', {}).get('max_drawdown', 'N/A')}%, Win Rate: {result.get('metrics', {}).get('win_rate', 'N/A')}"
                    )
            else:
                logger.warning(
                    f"Backtest failed for strategy {strategy_id}: {result.get('error', 'Unknown error')}"
                    )
            return result
        except Exception as e:
            logger.error(
                f'Failed to run backtest for strategy {strategy_id}: {e}',
                exc_info=True)
            return {'success': False, 'error': str(e)}

    def _validate_backtest_results(self, results: Dict[str, Any]) ->Tuple[
        bool, List[str]]:
        """
        Validate backtest results against minimum thresholds.

        Args:
            results: Backtest results to validate

        Returns:
            Tuple of (passed, reasons) where passed is bool and reasons is list of
            failure or warning messages
        """
        passed = True
        reasons = []
        if not results.get('success', False):
            return False, [
                f"Backtest failed to complete: {results.get('error', 'Unknown error')}"
                ]
        metrics = results.get('metrics', {})
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < self.backtest_min_sharpe:
            passed = False
            reasons.append(
                f'Sharpe ratio too low: {sharpe:.2f} (minimum: {self.backtest_min_sharpe:.2f})'
                )
        drawdown = metrics.get('max_drawdown', 100)
        if drawdown > self.backtest_max_drawdown:
            passed = False
            reasons.append(
                f'Max drawdown too high: {drawdown:.2f}% (maximum: {self.backtest_max_drawdown:.2f}%)'
                )
        win_rate = metrics.get('win_rate', 0)
        if win_rate < self.backtest_min_win_rate:
            passed = False
            reasons.append(
                f'Win rate too low: {win_rate:.2f} (minimum: {self.backtest_min_win_rate:.2f})'
                )
        return passed, reasons

    @async_with_exception_handling
    async def deploy_strategy_update(self, strategy_id: str, new_config:
        Dict[str, Any]):
        """
        Deploys an updated configuration or version of a strategy after validating with backtest.

        Args:
            strategy_id: The identifier of the strategy being updated.
            new_config: The new strategy configuration dictionary.

        Returns:
            Dict[str, Any]: Deployment information or None if deployment failed

        Raises:
            StrategyDeploymentError: If deployment fails
        """
        logger.info(
            f'Received request to deploy update for strategy: {strategy_id}')
        if not self.exec_client:
            error_msg = (
                f'Cannot deploy strategy {strategy_id}: Execution Engine client not initialized'
                )
            logger.error(error_msg)
            raise StrategyDeploymentError(error_msg)
        try:
            if self.backtest_required:
                logger.info(
                    f'Running backtest for strategy {strategy_id} before deployment'
                    )
                backtest_results = await self.run_backtest(strategy_id,
                    new_config)
                backtest_passed, reasons = self._validate_backtest_results(
                    backtest_results)
                if not backtest_passed:
                    reasons_str = '; '.join(reasons)
                    logger.warning(
                        f'Strategy {strategy_id} failed backtest validation: {reasons_str}'
                        )
                    return {'success': False, 'status': 'rejected',
                        'reason':
                        f'Failed backtest validation: {reasons_str}',
                        'backtest_results': backtest_results}
                logger.info(
                    f'Strategy {strategy_id} passed backtest validation')
            deployment_result = await self.exec_client.deploy_strategy(
                strategy_id, new_config)
            if deployment_result and deployment_result.get('success', False):
                deployment_id = deployment_result.get('deployment_id',
                    'unknown')
                logger.info(
                    f'Successfully deployed strategy {strategy_id}. Deployment ID: {deployment_id}'
                    )
                if (self.backtest_required and 'backtest_results' not in
                    deployment_result):
                    deployment_result['backtest_results'] = backtest_results
                return deployment_result
            else:
                error_msg = (
                    f"Deployment failed: {deployment_result.get('message', 'Unknown error')}"
                    )
                logger.error(
                    f'Failed to deploy strategy {strategy_id}: {error_msg}')
                return None
        except ExecutionEngineConnectionError as e:
            logger.error(
                f'Connection error while deploying strategy {strategy_id}: {e}'
                , exc_info=True)
            raise StrategyDeploymentError(f'Connection error: {e}')
        except StrategyDeploymentError as e:
            logger.error(f'Deployment error for strategy {strategy_id}: {e}',
                exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f'Unexpected error while deploying strategy {strategy_id}: {e}'
                , exc_info=True)
            raise StrategyDeploymentError(f'Unexpected error: {e}')


import asyncio
