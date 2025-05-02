"""
Model Evaluation and Feedback System

This module provides tools for evaluating ML model performance and creating feedback
loops that improve both models and indicator configurations over time.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
import os
import time
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, classification_report
)

# Local imports
from analysis_engine.analysis.indicator_interface import indicator_registry
from analysis_engine.analysis.ml_integration import ModelPrediction, PredictionType

# Configure logging
logger = logging.getLogger(__name__)


class EvaluationMetricType(Enum):
    """Types of evaluation metrics"""
    CLASSIFICATION = "classification"  # For classification models
    REGRESSION = "regression"          # For regression models
    RANKING = "ranking"                # For ranking models
    CUSTOM = "custom"                  # Custom metrics


@dataclass
class PredictionEvaluation:
    """Evaluation of a prediction against actual outcome"""
    prediction_id: str
    model_id: str
    timestamp: datetime
    instrument: str
    timeframe: str
    prediction_type: PredictionType
    predicted_values: Dict[str, Any]
    actual_values: Dict[str, Any]
    metrics: Dict[str, float]
    is_correct: bool = False
    error_margin: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "prediction_type": self.prediction_type.name,
            "predicted_values": self.predicted_values,
            "actual_values": self.actual_values,
            "metrics": self.metrics,
            "is_correct": self.is_correct,
            "error_margin": self.error_margin
        }


@dataclass
class IndicatorFeedback:
    """Feedback on indicator performance for improving configurations"""
    indicator_name: str
    timestamp: datetime
    instrument: str
    timeframe: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    suggested_parameters: Dict[str, Any]
    improvement_potential: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "indicator_name": self.indicator_name,
            "timestamp": self.timestamp.isoformat(),
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "parameters": self.parameters,
            "performance_metrics": self.performance_metrics,
            "suggested_parameters": self.suggested_parameters,
            "improvement_potential": self.improvement_potential,
            "confidence": self.confidence
        }


class ModelEvaluator:
    """Evaluates model predictions against actual outcomes"""
    
    def __init__(self):
        """Initialize model evaluator"""
        self.evaluations = []
        self.metrics_history = {}  # model_id -> list of metric dicts
    
    def evaluate_prediction(self, 
                          prediction: ModelPrediction, 
                          actual_data: pd.DataFrame) -> PredictionEvaluation:
        """
        Evaluate a prediction against actual outcome data
        
        Args:
            prediction: The prediction to evaluate
            actual_data: Actual market data for the prediction period
            
        Returns:
            Evaluation of the prediction
        """
        # Extract actual outcome based on prediction type
        actual_values = {}
        is_correct = False
        error_margin = 0.0
        metrics = {}
        
        # Get data point at prediction expiration (or latest available)
        prediction_horizon_seconds = prediction.horizon.value
        target_timestamp = prediction.timestamp + timedelta(seconds=prediction_horizon_seconds)
        
        # Find closest data point to target timestamp
        if isinstance(actual_data.index, pd.DatetimeIndex):
            # For datetime index
            closest_idx = actual_data.index.get_indexer([target_timestamp], method='nearest')[0]
        else:
            # For numeric index, find closest by timestamp
            actual_data['_timestamp_diff'] = abs(pd.to_datetime(actual_data['timestamp']) - target_timestamp)
            closest_idx = actual_data['_timestamp_diff'].idxmin()
        
        # Special handling based on prediction type
        if prediction.prediction_type == PredictionType.PRICE_DIRECTION:
            # Get starting price (at prediction time)
            start_price = None
            
            if prediction.timestamp in actual_data.index:
                start_price = actual_data.loc[prediction.timestamp, 'close']
            else:
                # Find closest starting point
                if isinstance(actual_data.index, pd.DatetimeIndex):
                    start_idx = actual_data.index.get_indexer([prediction.timestamp], method='nearest')[0]
                    start_price = actual_data.iloc[start_idx]['close']
                else:
                    actual_data['_start_diff'] = abs(pd.to_datetime(actual_data['timestamp']) - prediction.timestamp)
                    start_idx = actual_data['_start_diff'].idxmin()
                    start_price = actual_data.iloc[start_idx]['close']
            
            # Get ending price (at prediction horizon)
            end_price = actual_data.iloc[closest_idx]['close']
            
            # Calculate actual direction
            if start_price is not None:
                price_change = end_price - start_price
                actual_direction = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
                actual_values = {
                    'direction': actual_direction,
                    'price_change': price_change,
                    'price_change_pct': (price_change / start_price) if start_price != 0 else 0
                }
                
                # Check if prediction was correct
                predicted_direction = prediction.values.get('direction', 0)
                is_correct = predicted_direction == actual_direction
                
                # Set metrics
                metrics['direction_accuracy'] = 1.0 if is_correct else 0.0
                
                # If both are non-zero, check if signs match
                if predicted_direction != 0 and actual_direction != 0:
                    metrics['sign_accuracy'] = 1.0 if (predicted_direction * actual_direction > 0) else 0.0
        
        elif prediction.prediction_type == PredictionType.PRICE_TARGET:
            # For price target predictions
            predicted_price = prediction.values.get('value', 0.0)
            actual_price = actual_data.iloc[closest_idx]['close']
            
            actual_values = {
                'price': actual_price
            }
            
            # Calculate error
            error = abs(predicted_price - actual_price)
            error_pct = error / actual_price if actual_price != 0 else float('inf')
            
            # Set metrics
            metrics['mean_absolute_error'] = error
            metrics['mean_percentage_error'] = error_pct
            
            # Define correct prediction threshold (e.g., within 0.5%)
            threshold = 0.005
            is_correct = error_pct <= threshold
            
            error_margin = error_pct
            
        elif prediction.prediction_type == PredictionType.VOLATILITY:
            # For volatility predictions
            predicted_volatility = prediction.values.get('value', 0.0)
            
            # Calculate actual volatility as high-low range over the period
            period_data = actual_data.loc[prediction.timestamp:target_timestamp]
            if not period_data.empty:
                actual_volatility = (period_data['high'].max() - period_data['low'].min()) / period_data['close'].mean()
            else:
                actual_volatility = 0.0
                
            actual_values = {
                'volatility': actual_volatility
            }
            
            # Calculate error
            error = abs(predicted_volatility - actual_volatility)
            
            # Set metrics
            metrics['mean_absolute_error'] = error
            metrics['squared_error'] = error ** 2
            
            # Define correct prediction threshold
            threshold = 0.1  # 10% error margin
            is_correct = error <= threshold * actual_volatility
            
            error_margin = error / actual_volatility if actual_volatility > 0 else float('inf')
        
        # Add common metrics for all prediction types
        metrics['timestamp_diff'] = abs((target_timestamp - actual_data.index[closest_idx]).total_seconds()) \
            if isinstance(actual_data.index, pd.DatetimeIndex) else None
        
        # Create evaluation object
        evaluation = PredictionEvaluation(
            prediction_id=prediction.id,
            model_id=prediction.model_id,
            timestamp=datetime.now(),
            instrument=prediction.instrument,
            timeframe=prediction.timeframe,
            prediction_type=prediction.prediction_type,
            predicted_values=prediction.values,
            actual_values=actual_values,
            metrics=metrics,
            is_correct=is_correct,
            error_margin=error_margin
        )
        
        # Store evaluation
        self.evaluations.append(evaluation)
        
        # Update metrics history
        if prediction.model_id not in self.metrics_history:
            self.metrics_history[prediction.model_id] = []
        
        self.metrics_history[prediction.model_id].append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'is_correct': is_correct
        })
        
        return evaluation
    
    def evaluate_predictions_batch(self, 
                                predictions: List[ModelPrediction], 
                                actual_data: pd.DataFrame) -> List[PredictionEvaluation]:
        """
        Evaluate multiple predictions against actual data
        
        Args:
            predictions: List of predictions to evaluate
            actual_data: Actual market data covering the prediction periods
            
        Returns:
            List of prediction evaluations
        """
        evaluations = []
        
        for prediction in predictions:
            try:
                evaluation = self.evaluate_prediction(prediction, actual_data)
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error evaluating prediction {prediction.id}: {str(e)}")
                continue
        
        return evaluations
    
    def calculate_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Calculate overall performance metrics for a model
        
        Args:
            model_id: ID of the model to evaluate
            
        Returns:
            Dictionary with performance metrics
        """
        # Get all evaluations for this model
        model_evals = [e for e in self.evaluations if e.model_id == model_id]
        
        if not model_evals:
            return {'error': 'No evaluations found for model'}
            
        # Get prediction type from first evaluation
        pred_type = model_evals[0].prediction_type
        
        # Calculate metrics based on prediction type
        metrics = {}
        
        # Basic metrics for all types
        metrics['evaluation_count'] = len(model_evals)
        metrics['last_evaluated'] = max(e.timestamp for e in model_evals)
        
        if pred_type == PredictionType.PRICE_DIRECTION:
            # Classification metrics
            correct_count = sum(1 for e in model_evals if e.is_correct)
            metrics['accuracy'] = correct_count / len(model_evals) if model_evals else 0.0
            
            # Get predicted and actual directions for more metrics
            y_true = []
            y_pred = []
            
            for eval in model_evals:
                actual_dir = eval.actual_values.get('direction')
                pred_dir = eval.predicted_values.get('direction')
                
                if actual_dir is not None and pred_dir is not None:
                    y_true.append(actual_dir)
                    y_pred.append(pred_dir)
            
            if y_true and y_pred:
                # Convert to binary for some metrics
                y_true_binary = [1 if d > 0 else 0 for d in y_true]
                y_pred_binary = [1 if d > 0 else 0 for d in y_pred]
                
                try:
                    metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                    metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                    metrics['f1_score'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                    
                    # Special handling for three-class direction prediction
                    # Convert to -1, 0, 1 classes
                    unique_classes = set(y_true + y_pred)
                    if len(unique_classes) > 2:
                        metrics['multi_accuracy'] = accuracy_score(y_true, y_pred)
                except Exception as e:
                    logger.warning(f"Error calculating classification metrics: {str(e)}")
        
        elif pred_type in (PredictionType.PRICE_TARGET, PredictionType.VOLATILITY):
            # Regression metrics
            y_true = []
            y_pred = []
            
            for eval in model_evals:
                if pred_type == PredictionType.PRICE_TARGET:
                    actual_val = eval.actual_values.get('price')
                    pred_val = eval.predicted_values.get('value')
                else:  # Volatility
                    actual_val = eval.actual_values.get('volatility')
                    pred_val = eval.predicted_values.get('value')
                
                if actual_val is not None and pred_val is not None:
                    y_true.append(actual_val)
                    y_pred.append(pred_val)
            
            if y_true and y_pred:
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['r2'] = r2_score(y_true, y_pred)
                
                # Calculate mean absolute percentage error
                metrics['mape'] = np.mean([abs((a - p) / a) if a != 0 else float('inf') 
                                          for a, p in zip(y_true, y_pred) if a != 0])
                
                # Remove inf values
                if np.isinf(metrics['mape']):
                    metrics['mape'] = None
        
        # Time-based analysis
        timestamps = [e.timestamp for e in model_evals]
        if timestamps:
            metrics['time_span'] = (max(timestamps) - min(timestamps)).total_seconds()
            
            # Performance over time
            recent_count = 10  # Use most recent 10 evaluations
            if len(model_evals) >= recent_count * 2:
                recent_evals = sorted(model_evals, key=lambda e: e.timestamp, reverse=True)[:recent_count]
                older_evals = sorted(model_evals, key=lambda e: e.timestamp, reverse=True)[recent_count:recent_count*2]
                
                # Compare recent vs older accuracy
                recent_accuracy = sum(1 for e in recent_evals if e.is_correct) / len(recent_evals)
                older_accuracy = sum(1 for e in older_evals if e.is_correct) / len(older_evals)
                
                metrics['recent_accuracy'] = recent_accuracy
                metrics['older_accuracy'] = older_accuracy
                metrics['accuracy_trend'] = recent_accuracy - older_accuracy
        
        return metrics
    
    def save_evaluations_to_file(self, filepath: str) -> bool:
        """
        Save evaluations to a JSON file
        
        Args:
            filepath: Path to save the evaluations
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert evaluations to dictionaries
            evaluations_data = [e.to_dict() for e in self.evaluations]
            
            with open(filepath, 'w') as f:
                json.dump(evaluations_data, f, indent=4)
                
            logger.info(f"Saved {len(evaluations_data)} evaluations to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving evaluations to {filepath}: {str(e)}")
            return False
    
    def load_evaluations_from_file(self, filepath: str) -> int:
        """
        Load evaluations from a JSON file
        
        Args:
            filepath: Path to load the evaluations from
            
        Returns:
            Number of evaluations loaded
        """
        try:
            with open(filepath, 'r') as f:
                evaluations_data = json.load(f)
            
            loaded_count = 0
            for eval_data in evaluations_data:
                try:
                    # Create PredictionEvaluation
                    evaluation = PredictionEvaluation(
                        prediction_id=eval_data['prediction_id'],
                        model_id=eval_data['model_id'],
                        timestamp=datetime.fromisoformat(eval_data['timestamp']),
                        instrument=eval_data['instrument'],
                        timeframe=eval_data['timeframe'],
                        prediction_type=PredictionType[eval_data['prediction_type']],
                        predicted_values=eval_data['predicted_values'],
                        actual_values=eval_data['actual_values'],
                        metrics=eval_data['metrics'],
                        is_correct=eval_data['is_correct'],
                        error_margin=eval_data['error_margin']
                    )
                    
                    self.evaluations.append(evaluation)
                    loaded_count += 1
                    
                    # Update metrics history
                    if evaluation.model_id not in self.metrics_history:
                        self.metrics_history[evaluation.model_id] = []
                        
                    self.metrics_history[evaluation.model_id].append({
                        'timestamp': evaluation.timestamp,
                        'metrics': evaluation.metrics,
                        'is_correct': evaluation.is_correct
                    })
                    
                except Exception as e:
                    logger.warning(f"Error loading evaluation: {str(e)}")
                    continue
            
            logger.info(f"Loaded {loaded_count} evaluations from {filepath}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading evaluations from {filepath}: {str(e)}")
            return 0


class IndicatorFeedbackSystem:
    """
    System for generating feedback on indicator performance 
    and suggesting parameter optimizations
    """
    
    def __init__(self):
        """Initialize the indicator feedback system"""
        self.feedback_history = []
    
    def generate_feedback_for_indicator(self, 
                                     indicator_name: str, 
                                     data: pd.DataFrame,
                                     instrument: str,
                                     timeframe: str,
                                     current_params: Dict[str, Any],
                                     target_column: str = 'close',
                                     horizons: List[int] = None) -> IndicatorFeedback:
        """
        Generate feedback for an indicator based on performance
        
        Args:
            indicator_name: Name of the indicator
            data: Market data
            instrument: Instrument name
            timeframe: Timeframe
            current_params: Current indicator parameters
            target_column: Target column for optimization
            horizons: List of forward horizons to test (in periods)
            
        Returns:
            Feedback with suggested parameter improvements
        """
        # Use default horizons if not provided
        if horizons is None:
            horizons = [1, 5, 10]
            
        try:
            # Calculate the indicator with current parameters
            result = indicator_registry.calculate_indicator(
                indicator_name, data, **current_params
            )
            
            # Extract the main indicator column
            # Assuming first non-input column is the main indicator output
            input_columns = set(data.columns)
            indicator_columns = [col for col in result.data.columns if col not in input_columns]
            
            if not indicator_columns:
                raise ValueError(f"No output columns found for indicator {indicator_name}")
                
            # First indicator column
            main_indicator_column = indicator_columns[0]
            
            # Calculate performance metrics
            performance_metrics = {}
            
            # Correlation with future price changes
            for horizon in horizons:
                future_returns = data[target_column].pct_change(horizon).shift(-horizon)
                correlation = result.data[main_indicator_column].corr(future_returns)
                performance_metrics[f'correlation_h{horizon}'] = correlation
            
            # Average correlation across horizons
            performance_metrics['avg_correlation'] = sum(
                performance_metrics[f'correlation_h{h}'] for h in horizons
            ) / len(horizons)
            
            # Generate suggested parameters
            # (In a real system, this would use optimization techniques)
            suggested_params = self._suggest_improved_parameters(
                indicator_name, current_params, performance_metrics
            )
            
            # Calculate improvement potential
            improvement_potential = 0.1  # Placeholder value
            
            # Create feedback object
            feedback = IndicatorFeedback(
                indicator_name=indicator_name,
                timestamp=datetime.now(),
                instrument=instrument,
                timeframe=timeframe,
                parameters=current_params,
                performance_metrics=performance_metrics,
                suggested_parameters=suggested_params,
                improvement_potential=improvement_potential,
                confidence=0.7  # Placeholder confidence value
            )
            
            # Store feedback
            self.feedback_history.append(feedback)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating feedback for {indicator_name}: {str(e)}")
            
            # Return basic feedback with error
            return IndicatorFeedback(
                indicator_name=indicator_name,
                timestamp=datetime.now(),
                instrument=instrument,
                timeframe=timeframe,
                parameters=current_params,
                performance_metrics={"error": str(e)},
                suggested_parameters=current_params,
                improvement_potential=0.0,
                confidence=0.0
            )
    
    def _suggest_improved_parameters(self, 
                                  indicator_name: str, 
                                  current_params: Dict[str, Any],
                                  metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Suggest improved parameters for an indicator
        
        Args:
            indicator_name: Name of the indicator
            current_params: Current parameters
            metrics: Performance metrics
            
        Returns:
            Dictionary of suggested parameters
        """
        # This is a placeholder implementation
        # In a real system, this would use parameter optimization techniques
        
        # For now, just copy current parameters
        suggested = current_params.copy()
        
        # Example adjustments based on indicator type
        if indicator_name == "RSI":
            # Adjust period based on correlation
            avg_corr = metrics.get('avg_correlation', 0)
            
            current_period = suggested.get('period', 14)
            
            # If correlation is negative, slightly increase period
            if avg_corr < -0.1:
                suggested['period'] = min(current_period + 2, 30)
            # If correlation is very low, slightly decrease period
            elif avg_corr < 0.05:
                suggested['period'] = max(current_period - 2, 5)
                
        elif indicator_name == "MACD":
            # Adjust fast/slow periods based on correlation
            avg_corr = metrics.get('avg_correlation', 0)
            
            fast_period = suggested.get('fastperiod', 12)
            slow_period = suggested.get('slowperiod', 26)
            
            # If correlation is low, try to increase the difference between periods
            if abs(avg_corr) < 0.05:
                suggested['fastperiod'] = max(fast_period - 1, 8)
                suggested['slowperiod'] = min(slow_period + 2, 35)
                
        elif indicator_name == "BollingerBands":
            # Adjust stddev based on correlation
            avg_corr = metrics.get('avg_correlation', 0)
            
            # If correlation is low, try different stddev
            if abs(avg_corr) < 0.1:
                current_stddev = suggested.get('stddev', 2.0)
                if current_stddev == 2.0:
                    suggested['stddev'] = 2.5
                else:
                    suggested['stddev'] = 2.0
        
        # Otherwise, return unchanged parameters
        return suggested
    
    def bulk_generate_feedback(self, 
                             indicators: List[Dict[str, Any]],
                             data: pd.DataFrame,
                             instrument: str,
                             timeframe: str) -> List[IndicatorFeedback]:
        """
        Generate feedback for multiple indicators
        
        Args:
            indicators: List of dictionaries with indicator name and parameters
            data: Market data
            instrument: Instrument name
            timeframe: Timeframe
            
        Returns:
            List of feedback objects
        """
        feedback_list = []
        
        for ind_info in indicators:
            indicator_name = ind_info.get('name')
            parameters = ind_info.get('parameters', {})
            
            if indicator_name:
                try:
                    feedback = self.generate_feedback_for_indicator(
                        indicator_name, data, instrument, timeframe, parameters
                    )
                    feedback_list.append(feedback)
                except Exception as e:
                    logger.error(f"Error generating feedback for {indicator_name}: {str(e)}")
                    continue
        
        return feedback_list
    
    def save_feedback_to_file(self, filepath: str) -> bool:
        """
        Save feedback history to a JSON file
        
        Args:
            filepath: Path to save the feedback
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert feedback to dictionaries
            feedback_data = [f.to_dict() for f in self.feedback_history]
            
            with open(filepath, 'w') as f:
                json.dump(feedback_data, f, indent=4)
                
            logger.info(f"Saved {len(feedback_data)} feedback items to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback to {filepath}: {str(e)}")
            return False
    
    def load_feedback_from_file(self, filepath: str) -> int:
        """
        Load feedback history from a JSON file
        
        Args:
            filepath: Path to load the feedback from
            
        Returns:
            Number of feedback items loaded
        """
        try:
            with open(filepath, 'r') as f:
                feedback_data = json.load(f)
            
            loaded_count = 0
            for fb_data in feedback_data:
                try:
                    # Create IndicatorFeedback
                    feedback = IndicatorFeedback(
                        indicator_name=fb_data['indicator_name'],
                        timestamp=datetime.fromisoformat(fb_data['timestamp']),
                        instrument=fb_data['instrument'],
                        timeframe=fb_data['timeframe'],
                        parameters=fb_data['parameters'],
                        performance_metrics=fb_data['performance_metrics'],
                        suggested_parameters=fb_data['suggested_parameters'],
                        improvement_potential=fb_data['improvement_potential'],
                        confidence=fb_data['confidence']
                    )
                    
                    self.feedback_history.append(feedback)
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error loading feedback: {str(e)}")
                    continue
            
            logger.info(f"Loaded {loaded_count} feedback items from {filepath}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading feedback from {filepath}: {str(e)}")
            return 0


class IndicatorModelIntegrator:
    """
    Integrates indicators with models by continuously improving both
    components based on feedback loops
    """
    
    def __init__(self):
        """Initialize the integrator"""
        self.evaluator = ModelEvaluator()
        self.feedback_system = IndicatorFeedbackSystem()
    
    def apply_feedback_loop(self,
                          model_id: str,
                          indicator_params: Dict[str, Dict[str, Any]],
                          data: pd.DataFrame,
                          instrument: str,
                          timeframe: str) -> Dict[str, Any]:
        """
        Apply a feedback loop to improve indicators based on model performance
        
        Args:
            model_id: ID of the model being used
            indicator_params: Current indicator parameters (name -> params)
            data: Market data
            instrument: Instrument name
            timeframe: Timeframe
            
        Returns:
            Dictionary with feedback results and suggested improvements
        """
        # Get model performance metrics
        model_metrics = self.evaluator.calculate_model_performance(model_id)
        
        if 'error' in model_metrics:
            logger.warning(f"Cannot apply feedback loop: {model_metrics['error']}")
            return {'error': model_metrics['error']}
        
        # Prepare indicator list for feedback
        indicators = []
        for name, params in indicator_params.items():
            indicators.append({
                'name': name,
                'parameters': params
            })
        
        # Generate feedback for each indicator
        feedback_list = self.feedback_system.bulk_generate_feedback(
            indicators, data, instrument, timeframe
        )
        
        # Extract suggested parameters
        suggested_params = {}
        for feedback in feedback_list:
            suggested_params[feedback.indicator_name] = feedback.suggested_parameters
        
        # Calculate overall improvement potential
        avg_improvement = sum(f.improvement_potential for f in feedback_list) / len(feedback_list) \
            if feedback_list else 0.0
        
        return {
            'model_metrics': model_metrics,
            'model_id': model_id,
            'feedback_count': len(feedback_list),
            'suggested_parameters': suggested_params,
            'improvement_potential': avg_improvement,
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_models_with_indicators(self,
                                     models: List[str],
                                     indicator_configs: List[Dict[str, Any]],
                                     data: pd.DataFrame,
                                     instrument: str,
                                     timeframe: str,
                                     test_period: int = 100) -> Dict[str, Any]:
        """
        Evaluate multiple models with different indicator configurations
        
        Args:
            models: List of model IDs
            indicator_configs: List of indicator configurations
            data: Market data
            instrument: Instrument name
            timeframe: Timeframe
            test_period: Number of periods to test
            
        Returns:
            Evaluation results
        """
        # This would be a complex implementation in a real system
        # For now, return a placeholder result
        return {
            'message': 'Evaluation not implemented in this version',
            'models_count': len(models),
            'indicator_configs_count': len(indicator_configs),
            'test_period': test_period
        }


# Create global instances for easy import
model_evaluator = ModelEvaluator()
indicator_feedback = IndicatorFeedbackSystem()
ml_integrator = IndicatorModelIntegrator()
