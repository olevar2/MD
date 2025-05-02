"""
Training Pipeline Orchestrator Module

This module provides functionality to orchestrate the entire model training pipeline,
from data preparation to model training, evaluation, and registration.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ..services.experiment_tracker import ExperimentTracker
from ..services.dataset_preparation import DatasetPreparation


class TrainingPipelineOrchestrator:
    """
    A class for orchestrating machine learning training pipelines.
    
    This class provides methods to coordinate the end-to-end process of
    preparing data, training models, evaluating performance, and logging
    results to MLflow.
    
    Attributes:
        experiment_tracker: ExperimentTracker instance for tracking experiments
        dataset_preparation: DatasetPreparation instance for preparing datasets
        logger: Logger instance
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        feature_client=None
    ):
        """
        Initialize the TrainingPipelineOrchestrator.
        
        Args:
            experiment_name: Optional name for the experiment.
                            If None, uses default from MLflowSettings.
            feature_client: Optional FeatureStoreClient instance.
                           If None, a default client will be created.
        """
        self.experiment_tracker = ExperimentTracker(experiment_name=experiment_name)
        self.dataset_preparation = DatasetPreparation(feature_client=feature_client)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("TrainingPipelineOrchestrator initialized")
    
    def train_model(
        self,
        model_config: Dict[str, Any],
        run_name: Optional[str] = None,
        register_model: bool = False,
        model_name: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute a complete model training pipeline from data preparation to evaluation.
        
        Args:
            model_config: Dictionary with configuration for model training
                          Must contain:
                          - 'data_config': Configuration for data preparation
                          - 'model_type': Type of model to train
                          - 'model_params': Parameters for the model
                          - 'evaluation': Configuration for model evaluation
            run_name: Optional name for the MLflow run
            register_model: Whether to register the model in the MLflow registry
            model_name: Name to use when registering the model
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Trained model and evaluation results
        """
        # Start MLflow run
        if not run_name:
            run_name = f"model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_tracker.start_run(run_name=run_name)
        self.logger.info(f"Starting training pipeline with run name: {run_name}")
        
        try:
            # Log model configuration
            self.experiment_tracker.log_params({
                f"config.{key}": json.dumps(value) if isinstance(value, dict) else value
                for key, value in model_config.items()
                if key != 'data_config'  # Exclude data_config as it might be too large
            })
            
            # Extract configurations
            data_config = model_config.get('data_config', {})
            model_type = model_config.get('model_type')
            model_params = model_config.get('model_params', {})
            evaluation_config = model_config.get('evaluation', {})
            
            if not model_type:
                raise ValueError("Model type must be specified in model_config")
            
            # Prepare dataset
            self.logger.info("Preparing dataset for model training")
            dataset = self.dataset_preparation.prepare_dataset_for_model(**data_config)
            
            # Log dataset information
            features_count = len(dataset['X_train'].columns)
            train_samples = len(dataset['X_train'])
            val_samples = len(dataset['X_val'])
            test_samples = len(dataset['X_test'])
            
            self.experiment_tracker.log_params({
                'features_count': features_count,
                'train_samples': train_samples,
                'val_samples': val_samples,
                'test_samples': test_samples,
                'target_column': dataset['target_column'],
                'symbol': data_config.get('symbol'),
                'timeframe': data_config.get('timeframe')
            })
            
            # Train model
            self.logger.info(f"Training model of type {model_type}")
            model, train_history = self._train_model_by_type(
                model_type=model_type,
                model_params=model_params,
                dataset=dataset
            )
            
            # Log training history if available
            if train_history:
                for metric_name, values in train_history.items():
                    if isinstance(values, list):
                        for i, value in enumerate(values):
                            self.experiment_tracker.log_metric(f'train_{metric_name}', value, step=i)
            
            # Evaluate model
            self.logger.info("Evaluating model performance")
            evaluation_results = self._evaluate_model(
                model=model,
                dataset=dataset,
                evaluation_config=evaluation_config
            )
            
            # Log evaluation metrics
            for metric_name, value in evaluation_results.items():
                if isinstance(value, (int, float)):
                    self.experiment_tracker.log_metric(metric_name, value)
            
            # Create and log feature importance if applicable
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                self._log_feature_importance(model, dataset['X_train'].columns)
            
            # Create and log confusion matrix if classification task
            if 'confusion_matrix' in evaluation_results:
                self._log_confusion_matrix(evaluation_results['confusion_matrix'], evaluation_results['classes'])
            
            # Save and log model
            self.logger.info("Saving model artifacts")
            model_path = self._save_model(model, model_type, dataset)
            
            # Register model if requested
            if register_model and model_name:
                self.logger.info(f"Registering model as: {model_name}")
                self.experiment_tracker.log_model(
                    model_object=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
            
            self.logger.info("Training pipeline completed successfully")
            self.experiment_tracker.end_run()
            
            return model, evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            self.experiment_tracker.end_run(status="FAILED")
            raise
    
    def _train_model_by_type(
        self,
        model_type: str,
        model_params: Dict[str, Any],
        dataset: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model based on the specified type and parameters.
        
        Args:
            model_type: Type of model to train
            model_params: Parameters for the model
            dataset: Dataset from DatasetPreparation
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Trained model and training history
        """
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        
        training_history = {}
        model = None
        
        # Handle different model types
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # Determine if classification or regression
            if y_train.dtype == 'int64' or y_train.nunique() < 10:
                model = RandomForestClassifier(**model_params)
            else:
                model = RandomForestRegressor(**model_params)
            
            # Train the model
            model.fit(X_train, y_train)
            
        elif model_type == 'xgboost':
            import xgboost as xgb
            
            # Determine if classification or regression
            if y_train.dtype == 'int64' or y_train.nunique() < 10:
                objective = model_params.pop('objective', 'binary:logistic')
                if y_train.nunique() > 2:
                    objective = 'multi:softprob'
                    model_params['num_class'] = y_train.nunique()
                
                model = xgb.XGBClassifier(objective=objective, **model_params)
            else:
                model = xgb.XGBRegressor(**model_params)
            
            # Train with early stopping if validation data provided
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_metric = model_params.pop('eval_metric', None)
            
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric=eval_metric,
                early_stopping_rounds=model_params.pop('early_stopping_rounds', 50),
                verbose=model_params.pop('verbose', True)
            )
            
            # Extract training history
            if hasattr(model, 'evals_result'):
                training_history = model.evals_result()
            
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            
            # Determine if classification or regression
            if y_train.dtype == 'int64' or y_train.nunique() < 10:
                objective = model_params.pop('objective', 'binary')
                if y_train.nunique() > 2:
                    objective = 'multiclass'
                    model_params['num_class'] = y_train.nunique()
                
                model = lgb.LGBMClassifier(objective=objective, **model_params)
            else:
                model = lgb.LGBMRegressor(**model_params)
            
            # Train with early stopping if validation data provided
            eval_set = [(X_val, y_val)]
            eval_metric = model_params.pop('eval_metric', None)
            
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric=eval_metric,
                early_stopping_rounds=model_params.pop('early_stopping_rounds', 50),
                verbose=model_params.pop('verbose', True)
            )
            
            # Extract training history
            if hasattr(model, 'evals_result_'):
                training_history = model.evals_result_
            
        elif model_type == 'catboost':
            from catboost import CatBoostClassifier, CatBoostRegressor
            
            # Determine if classification or regression
            if y_train.dtype == 'int64' or y_train.nunique() < 10:
                model = CatBoostClassifier(**model_params)
            else:
                model = CatBoostRegressor(**model_params)
            
            # Train the model
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=model_params.pop('verbose', True)
            )
            
            # Extract training history
            if hasattr(model, 'evals_result_'):
                training_history = model.evals_result_
            
        elif model_type == 'linear':
            from sklearn.linear_model import LogisticRegression, LinearRegression
            
            # Determine if classification or regression
            if y_train.dtype == 'int64' or y_train.nunique() < 10:
                model = LogisticRegression(**model_params)
            else:
                model = LinearRegression(**model_params)
            
            # Train the model
            model.fit(X_train, y_train)
            
        elif model_type == 'svm':
            from sklearn.svm import SVC, SVR
            
            # Determine if classification or regression
            if y_train.dtype == 'int64' or y_train.nunique() < 10:
                model = SVC(probability=True, **model_params)
            else:
                model = SVR(**model_params)
            
            # Train the model
            model.fit(X_train, y_train)
            
        elif model_type == 'nn_tf':
            # Implementation would depend on TensorFlow specifics
            # This is a placeholder for a TensorFlow implementation
            self.logger.warning("TensorFlow model implementation is more complex and project-specific")
            self.logger.info("For a complete TensorFlow implementation, please implement a specialized trainer")
            
            # A basic skeleton would look like:
            # import tensorflow as tf
            # model = tf.keras.Sequential([
            #     tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            #     tf.keras.layers.Dropout(0.2),
            #     tf.keras.layers.Dense(64, activation='relu'),
            #     tf.keras.layers.Dense(1, activation='sigmoid' if y_train.nunique() <= 2 else 'softmax')
            # ])
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
            # training_history = history.history
            
            # Return None for the model since we're not implementing it here
            model = None
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model, training_history
    
    def _evaluate_model(
        self,
        model: Any,
        dataset: Dict[str, Any],
        evaluation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate model performance using validation and test sets.
        
        Args:
            model: Trained model
            dataset: Dataset dictionary from DatasetPreparation
            evaluation_config: Configuration for evaluation
            
        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics
        """
        results = {}
        
        # Extract datasets
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        
        # Determine if classification or regression task
        is_classification = y_train.dtype == 'int64' or y_train.nunique() < 10
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Get probabilities if it's a classification task and model supports it
        if is_classification and hasattr(model, 'predict_proba'):
            train_proba = model.predict_proba(X_train)
            val_proba = model.predict_proba(X_val)
            test_proba = model.predict_proba(X_test)
        else:
            train_proba = val_proba = test_proba = None
        
        # Calculate metrics
        if is_classification:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, confusion_matrix, classification_report
            )
            
            # Basic classification metrics
            for dataset_name, y_true, y_pred, y_proba in [
                ('train', y_train, train_pred, train_proba),
                ('val', y_val, val_pred, val_proba),
                ('test', y_test, test_pred, test_proba)
            ]:
                results[f'{dataset_name}_accuracy'] = accuracy_score(y_true, y_pred)
                
                # Binary classification metrics
                if y_true.nunique() == 2:
                    results[f'{dataset_name}_precision'] = precision_score(y_true, y_pred)
                    results[f'{dataset_name}_recall'] = recall_score(y_true, y_pred)
                    results[f'{dataset_name}_f1'] = f1_score(y_true, y_pred)
                    
                    # ROC AUC if probabilities are available
                    if y_proba is not None:
                        results[f'{dataset_name}_roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                
                # Multi-class classification metrics
                else:
                    results[f'{dataset_name}_precision_macro'] = precision_score(y_true, y_pred, average='macro')
                    results[f'{dataset_name}_recall_macro'] = recall_score(y_true, y_pred, average='macro')
                    results[f'{dataset_name}_f1_macro'] = f1_score(y_true, y_pred, average='macro')
                    
                    results[f'{dataset_name}_precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
                    results[f'{dataset_name}_recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
                    results[f'{dataset_name}_f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            
            # Store confusion matrix for later visualization
            results['confusion_matrix'] = confusion_matrix(y_test, test_pred)
            results['classes'] = sorted(y_train.unique())
            
            # Store classification report
            report = classification_report(y_test, test_pred, output_dict=True)
            results['classification_report'] = report
            
        else:
            # Regression metrics
            from sklearn.metrics import (
                mean_squared_error, mean_absolute_error, r2_score,
                mean_absolute_percentage_error
            )
            
            for dataset_name, y_true, y_pred in [
                ('train', y_train, train_pred),
                ('val', y_val, val_pred),
                ('test', y_test, test_pred)
            ]:
                results[f'{dataset_name}_mse'] = mean_squared_error(y_true, y_pred)
                results[f'{dataset_name}_rmse'] = mean_squared_error(y_true, y_pred, squared=False)
                results[f'{dataset_name}_mae'] = mean_absolute_error(y_true, y_pred)
                results[f'{dataset_name}_r2'] = r2_score(y_true, y_pred)
                
                try:
                    results[f'{dataset_name}_mape'] = mean_absolute_percentage_error(y_true, y_pred)
                except:
                    # MAPE might fail if y_true contains zeros
                    pass
        
        # Calculate custom forex-specific metrics if configured
        if evaluation_config.get('custom_forex_metrics', False):
            results.update(self._calculate_custom_forex_metrics(
                y_test=y_test,
                y_pred=test_pred,
                y_proba=test_proba if is_classification else None,
                is_classification=is_classification
            ))
        
        # Store additional information
        results['model_type'] = type(model).__name__
        results['feature_count'] = X_train.shape[1]
        
        return results
    
    def _log_feature_importance(self, model: Any, feature_names: List[str]) -> None:
        """
        Create and log feature importance visualization.
        
        Args:
            model: Trained model with feature_importances_ or coef_ attribute
            feature_names: Names of features
        """
        plt.figure(figsize=(12, 8))
        
        # Extract feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            self.logger.warning("Model doesn't have feature_importances_ or coef_ attribute")
            return
        
        # Create DataFrame for easier sorting and plotting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Plot top 20 features (or all if less than 20)
        n_features = min(20, len(feature_importance_df))
        top_features_df = feature_importance_df.head(n_features)
        
        sns.barplot(x='Importance', y='Feature', data=top_features_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Log to MLflow
        self.experiment_tracker.log_figure(plt, "feature_importance.png")
        plt.close()
        
        # Also log feature importance as a JSON artifact
        with open('feature_importance.json', 'w') as f:
            json.dump(feature_importance_df.to_dict('records'), f)
        
        self.experiment_tracker.log_artifact('feature_importance.json')
        os.remove('feature_importance.json')
    
    def _log_confusion_matrix(self, confusion_matrix_data: np.ndarray, class_names: List) -> None:
        """
        Create and log confusion matrix visualization.
        
        Args:
            confusion_matrix_data: Confusion matrix data
            class_names: Names of classes
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            confusion_matrix_data,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Log to MLflow
        self.experiment_tracker.log_figure(plt, "confusion_matrix.png")
        plt.close()
    
    def _calculate_custom_forex_metrics(
        self,
        y_test: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Calculate custom metrics for forex prediction evaluation.
        
        Args:
            y_test: True target values
            y_pred: Predicted target values
            y_proba: Predicted probabilities (for classification)
            is_classification: Whether the task is classification
            
        Returns:
            Dict[str, float]: Dictionary with custom forex metrics
        """
        results = {}
        
        # For binary classification (direction prediction)
        if is_classification and len(set(y_test)) == 2:
            # Calculate win rate (accuracy with different name)
            win_rate = (y_test == y_pred).mean()
            results['forex_win_rate'] = win_rate
            
            # Calculate profit factor (sum of gains / sum of losses)
            # This is a simplified version assuming equal position sizes
            correct_predictions = y_test == y_pred
            incorrect_predictions = ~correct_predictions
            
            profit_factor = correct_predictions.sum() / max(1, incorrect_predictions.sum())
            results['forex_profit_factor'] = profit_factor
            
            # Calculate expected payoff
            # (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            # Simplified version assuming equal wins and losses
            expected_payoff = (win_rate * 1) - ((1 - win_rate) * 1)
            results['forex_expected_payoff'] = expected_payoff
            
            # Calculate confidence-weighted metrics if probabilities are available
            if y_proba is not None:
                # Get probability of predicted class
                pred_proba = np.max(y_proba, axis=1)
                high_conf_mask = pred_proba >= 0.7  # Threshold for high confidence
                
                if high_conf_mask.any():
                    high_conf_win_rate = (y_test[high_conf_mask] == y_pred[high_conf_mask]).mean()
                    results['forex_high_conf_win_rate'] = high_conf_win_rate
                    
                    # Count of high confidence predictions
                    results['forex_high_conf_count'] = high_conf_mask.sum()
                    results['forex_high_conf_percentage'] = high_conf_mask.mean() * 100
        
        # For regression (price prediction)
        else:
            # Calculate direction accuracy for regression models
            direction_correct = ((y_test > 0) == (y_pred > 0)).mean()
            results['forex_direction_accuracy'] = direction_correct
            
            # Mean absolute percentage error normalized by average true range
            # This would require ATR data which we don't have here
            # This is just a placeholder for a project-specific implementation
            results['forex_normalized_error'] = 0.0
            
        return results
    
    def _save_model(
        self,
        model: Any,
        model_type: str,
        dataset: Dict[str, Any]
    ) -> str:
        """
        Save the trained model and related artifacts.
        
        Args:
            model: Trained model
            model_type: Type of model
            dataset: Dataset dictionary containing preprocessing params
            
        Returns:
            str: Path to saved model
        """
        # Create a temporary directory to save model artifacts
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Save the model
            if model_type in ['random_forest', 'linear', 'svm']:
                model_path = model_dir / 'model.joblib'
                joblib.dump(model, model_path)
                
            elif model_type in ['xgboost', 'lightgbm', 'catboost']:
                model_path = model_dir / 'model.json'
                if hasattr(model, 'save_model'):
                    model.save_model(str(model_path))
                else:
                    # Fallback to joblib if the model doesn't have save_model
                    model_path = model_dir / 'model.joblib'
                    joblib.dump(model, model_path)
            
            elif model_type == 'nn_tf':
                # This would be implemented for TensorFlow models
                model_path = model_dir / 'model.h5'
                # model.save(model_path)  # Commented since we aren't implementing TF models here
                model_path = None
            
            else:
                # Generic fallback
                model_path = model_dir / 'model.joblib'
                joblib.dump(model, model_path)
            
            # Save preprocessing parameters
            if 'preprocessing_params' in dataset:
                preproc_path = model_dir / 'preprocessing.joblib'
                joblib.dump(dataset['preprocessing_params'], preproc_path)
                
                # Log to MLflow
                if model_path:
                    self.experiment_tracker.log_artifact(str(model_path))
                self.experiment_tracker.log_artifact(str(preproc_path))
            
            # Save feature names
            if 'X_train' in dataset:
                feature_names_path = model_dir / 'feature_names.json'
                with open(feature_names_path, 'w') as f:
                    json.dump(list(dataset['X_train'].columns), f)
                
                self.experiment_tracker.log_artifact(str(feature_names_path))
            
            return str(model_path) if model_path else None
    
    def compare_runs(self, metrics: Optional[List[str]] = None, max_runs: int = 5) -> pd.DataFrame:
        """
        Compare the latest experiment runs based on specified metrics.
        
        Args:
            metrics: List of metrics to compare. If None, will use standard metrics.
            max_runs: Maximum number of runs to compare
            
        Returns:
            pd.DataFrame: DataFrame with run comparisons
        """
        # Get experiment history
        experiment_history = self.experiment_tracker.get_experiment_history()
        
        # If no metrics specified, use some defaults
        if metrics is None:
            # Check column names to determine if we have classification or regression metrics
            columns = experiment_history.columns
            
            if any('accuracy' in col for col in columns):
                metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
            elif any('mse' in col for col in columns):
                metrics = ['test_mse', 'test_rmse', 'test_mae', 'test_r2']
            else:
                # Generic fallback
                metrics = [col for col in columns if col.startswith('test_') and not col.startswith('test_params')]
        
        # Filter relevant columns for the comparison
        run_columns = ['run_id', 'run_name', 'status', 'start_time', 'end_time']
        metric_columns = [f'metrics.{metric}' for metric in metrics]
        param_columns = [col for col in experiment_history.columns if 'params' in col]
        
        # Select columns that exist in the dataframe
        existing_columns = run_columns + [col for col in metric_columns if col in experiment_history.columns]
        
        # Add some key parameters if they exist
        key_params = ['params.model_type', 'params.config.model_params', 'params.features_count', 'params.symbol', 'params.timeframe']
        existing_params = [col for col in key_params if col in experiment_history.columns]
        
        # Filter and sort
        comparison_df = experiment_history[existing_columns + existing_params].sort_values('start_time', ascending=False).head(max_runs)
        
        # Rename columns to make them more user-friendly
        rename_dict = {f'metrics.{metric}': metric for metric in metrics}
        rename_dict.update({param: param.replace('params.', '') for param in existing_params})
        
        comparison_df = comparison_df.rename(columns=rename_dict)
        
        return comparison_df
    
    def get_best_model_run(self, metric: str, ascending: bool = False) -> Dict[str, Any]:
        """
        Get information about the best model run based on a metric.
        
        Args:
            metric: Metric to use for determining the best run
            ascending: If True, lower values are better
            
        Returns:
            Dict[str, Any]: Information about the best run
        """
        # Get the best run ID
        best_run_id, best_metric_value = self.experiment_tracker.get_best_run(
            metric_name=metric,
            ascending=ascending
        )
        
        if best_run_id is None:
            self.logger.warning(f"No runs found with metric: {metric}")
            return {}
        
        # Get all runs
        runs_df = self.experiment_tracker.get_experiment_history()
        
        # Filter for the best run
        best_run_df = runs_df[runs_df['run_id'] == best_run_id]
        
        if best_run_df.empty:
            self.logger.warning(f"Run details not found for run_id: {best_run_id}")
            return {'run_id': best_run_id, 'metric_value': best_metric_value}
        
        # Extract run information
        best_run_info = best_run_df.iloc[0].to_dict()
        
        # Format information for readability
        result = {
            'run_id': best_run_id,
            'run_name': best_run_info.get('run_name', 'Unknown'),
            'metric_name': metric,
            'metric_value': best_metric_value,
            'start_time': best_run_info.get('start_time', None),
            'params': {k.replace('params.', ''): v for k, v in best_run_info.items() if k.startswith('params.')}
        }
        
        return result