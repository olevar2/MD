"""
ExperimentTracker Module

This module provides a class for tracking ML experiments using MLflow.
It includes functionality for experiment creation, logging metrics and parameters,
artifact storage, and experiment comparison.
"""
import os
import json
import pandas as pd
import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime

from ..config.mlflow_config import mlflow_settings

class ExperimentTracker:
    """
    A class to track ML experiments using MLflow.
    
    This class provides methods to create experiments, log parameters and metrics,
    store artifacts, and compare experiment runs.
    
    Attributes:
        client: MLflow client instance
        experiment_name: Name of the current experiment
        experiment_id: ID of the current experiment
        run_id: ID of the current run if active
        active_run: Currently active MLflow run object if any
    """
    
    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize the ExperimentTracker with MLflow settings.
        
        Args:
            experiment_name: Optional name for the experiment.
                            Defaults to value in MLflowSettings.
        """
        # Initialize MLflow settings
        mlflow.set_tracking_uri(mlflow_settings.mlflow_tracking_uri)
        
        if mlflow_settings.registry_uri:
            mlflow.set_registry_uri(mlflow_settings.registry_uri)
            
        # Set S3 endpoint if provided (for artifact storage)
        if mlflow_settings.s3_endpoint_url:
            os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow_settings.s3_endpoint_url
            if mlflow_settings.s3_ignore_tls:
                os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
        
        # Create MLflow client
        self.client = MlflowClient()
        
        # Set experiment name and create experiment if needed
        self.experiment_name = experiment_name or mlflow_settings.default_experiment_name
        self._create_experiment_if_not_exists()
        
        # Initialize run tracking variables
        self.run_id = None
        self.active_run = None
        
        logging.info(f"ExperimentTracker initialized with experiment: {self.experiment_name}")
    
    def _create_experiment_if_not_exists(self) -> None:
        """Create the experiment if it doesn't already exist."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            artifact_location = mlflow_settings.mlflow_artifact_location
            self.experiment_id = mlflow.create_experiment(
                name=self.experiment_name, 
                artifact_location=artifact_location,
                tags=mlflow_settings.experiment_tags
            )
            logging.info(f"Created new experiment: {self.experiment_name}, ID: {self.experiment_id}")
        else:
            self.experiment_id = experiment.experiment_id
            logging.info(f"Using existing experiment: {self.experiment_name}, ID: {self.experiment_id}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run within the experiment.
        
        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags for the run
            
        Returns:
            str: ID of the created run
        """
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
        self.run_id = self.active_run.info.run_id
        
        # Add default tags and timestamp information
        current_time = datetime.now().isoformat()
        default_tags = {
            "start_time": current_time,
            "mlflow.source.type": "FOREX_TRADING_PLATFORM"
        }
        
        # Combine default tags with any provided tags
        all_tags = {**default_tags, **(tags or {})}
        for key, value in all_tags.items():
            mlflow.set_tag(key, value)
        
        logging.info(f"Started MLflow run: {run_name or 'unnamed'}, Run ID: {self.run_id}")
        return self.run_id
    
    def end_run(self, status: str = RunStatus.FINISHED) -> None:
        """
        End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED, etc.)
        """
        if self.active_run:
            # Add end time tag
            mlflow.set_tag("end_time", datetime.now().isoformat())
            
            # End the run with specified status
            mlflow.end_run(status=status)
            logging.info(f"Ended MLflow run: {self.run_id} with status: {status}")
            self.active_run = None
            self.run_id = None
        else:
            logging.warning("No active run to end")
    
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter to the current run.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters to the current run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        mlflow.log_params(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric to the current run.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step value (for tracking metrics over iterations)
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics to the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step value (for tracking metrics over iterations)
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str) -> None:
        """
        Log an artifact (file) to the current run.
        
        Args:
            local_path: Path to the local file to log
        """
        mlflow.log_artifact(local_path)
        logging.info(f"Logged artifact: {local_path}")
    
    def log_figure(self, fig, artifact_name: str) -> None:
        """
        Log a matplotlib or plotly figure to the current run.
        
        Args:
            fig: Figure object to log
            artifact_name: Name for the saved figure
        """
        # Ensure artifact_name has an extension
        if not artifact_name.endswith(('.png', '.pdf', '.svg', '.html')):
            artifact_name += '.png'
        
        # Create a temporary directory for the figure
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / artifact_name
            
            # Handle different figure types
            import matplotlib.pyplot as plt
            try:
                # Check if it's a matplotlib figure
                if 'matplotlib' in str(type(fig)):
                    fig.savefig(temp_path)
                # Check if it's a plotly figure
                elif 'plotly' in str(type(fig)):
                    fig.write_html(str(temp_path))
                else:
                    logging.warning(f"Unsupported figure type: {type(fig)}")
                    return
                
                # Log the saved figure as an artifact
                self.log_artifact(str(temp_path))
            except Exception as e:
                logging.error(f"Error saving figure: {e}")
    
    def log_model(
        self, 
        model_object: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        conda_env: Optional[Union[Dict[str, Any], str]] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
    ) -> None:
        """
        Log a model to the current run and optionally register it.
        
        Args:
            model_object: Trained model object
            artifact_path: Path within the MLflow run artifacts where model is saved
            registered_model_name: Optional name for registering the model in the Model Registry
            conda_env: Optional Conda environment for the model
            signature: Optional model signature specifying the schema of model's inputs and outputs
            input_example: Optional input example for the model
        """
        try:
            # Check model type to use appropriate MLflow flavor
            if 'sklearn' in str(type(model_object)):
                mlflow.sklearn.log_model(
                    sk_model=model_object,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    conda_env=conda_env,
                    signature=signature,
                    input_example=input_example,
                )
            elif 'xgboost' in str(type(model_object)):
                mlflow.xgboost.log_model(
                    xgb_model=model_object,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    conda_env=conda_env,
                    signature=signature,
                    input_example=input_example,
                )
            elif 'lightgbm' in str(type(model_object)):
                mlflow.lightgbm.log_model(
                    lgb_model=model_object,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    conda_env=conda_env,
                    signature=signature,
                    input_example=input_example,
                )
            elif 'keras' in str(type(model_object)) or 'tensorflow' in str(type(model_object)):
                mlflow.tensorflow.log_model(
                    model=model_object,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    conda_env=conda_env,
                    signature=signature,
                    input_example=input_example,
                )
            elif 'torch' in str(type(model_object)):
                mlflow.pytorch.log_model(
                    pytorch_model=model_object,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    conda_env=conda_env,
                    signature=signature,
                    input_example=input_example,
                )
            else:
                # Use generic pyfunc for other model types
                mlflow.pyfunc.log_model(
                    python_model=model_object,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    conda_env=conda_env,
                    signature=signature,
                    input_example=input_example,
                )
            
            logging.info(f"Logged model to artifact path: {artifact_path}")
            if registered_model_name:
                logging.info(f"Registered model with name: {registered_model_name}")
        
        except Exception as e:
            logging.error(f"Error logging model: {str(e)}")
            raise
    
    def compare_runs(self, run_ids: List[str], metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple runs based on their metrics and parameters.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Optional list of specific metrics to compare.
                    If None, all metrics are compared.
        
        Returns:
            pd.DataFrame: DataFrame with run metrics and parameters for comparison
        """
        results = []
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                
                # Extract run name and status
                run_name = run.data.tags.get("mlflow.runName", "unnamed")
                run_status = run.info.status
                
                # Extract metrics of interest
                run_metrics = {}
                for key, value in run.data.metrics.items():
                    if metrics is None or key in metrics:
                        run_metrics[f"metric_{key}"] = value
                
                # Extract parameters (limit to important ones if there are too many)
                run_params = {}
                for key, value in run.data.params.items():
                    run_params[f"param_{key}"] = value
                
                # Combine all information
                run_info = {
                    "run_id": run_id,
                    "run_name": run_name,
                    "status": run_status,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0),
                }
                
                if run.info.end_time:
                    run_info["end_time"] = datetime.fromtimestamp(run.info.end_time / 1000.0)
                    run_info["duration_seconds"] = (run.info.end_time - run.info.start_time) / 1000.0
                
                # Combine all information
                run_data = {**run_info, **run_metrics, **run_params}
                results.append(run_data)
            
            except Exception as e:
                logging.error(f"Error retrieving run {run_id}: {str(e)}")
        
        # Convert to DataFrame and return
        if results:
            return pd.DataFrame(results)
        else:
            logging.warning("No valid runs found for comparison")
            return pd.DataFrame()
    
    def get_best_run(
        self, 
        metric_name: str, 
        max_results: int = 10,
        ascending: bool = False
    ) -> Tuple[str, float]:
        """
        Get the run with the best metric value.
        
        Args:
            metric_name: Name of the metric to use for comparison
            max_results: Maximum number of runs to consider
            ascending: If True, the smallest value is best. If False, the largest value is best.
        
        Returns:
            Tuple[str, float]: Tuple of (run_id, metric_value) for the best run
        """
        # Get runs for the experiment
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=max_results
        )
        
        if runs.empty:
            logging.warning(f"No runs found with metric: {metric_name}")
            return None, None
        
        # Get the best run (first row due to ordering)
        best_run_id = runs.iloc[0]["run_id"]
        best_metric_value = runs.iloc[0][f"metrics.{metric_name}"]
        
        return best_run_id, best_metric_value
    
    def get_experiment_history(self) -> pd.DataFrame:
        """
        Get all runs for the current experiment with their metrics and parameters.
        
        Returns:
            pd.DataFrame: DataFrame with run information
        """
        return mlflow.search_runs(experiment_ids=[self.experiment_id])