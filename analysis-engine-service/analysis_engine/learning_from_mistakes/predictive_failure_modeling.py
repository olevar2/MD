"""
Predictive Failure Modeling

This module implements predictive modeling capabilities for identifying potential
strategy failure conditions before they occur.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import uuid

from analysis_engine.learning_from_mistakes.error_pattern_recognition import ErrorPatternRecognitionSystem, ErrorPattern
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

class PredictiveFailureModel:
    """
    Predictive model for identifying potential strategy failure conditions before they occur.
    
    Key capabilities:
    - Train models to predict strategy failures based on historical error data
    - Generate failure risk scores for current market conditions
    - Provide explainable predictions with feature importance
    - Continuously update models with new data
    """
    
    def __init__(
        self,
        error_pattern_system: ErrorPatternRecognitionSystem,
        model_dir: str = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the PredictiveFailureModel.
        
        Args:
            error_pattern_system: The error pattern recognition system
            model_dir: Directory to store trained models
            config: Configuration parameters for the model
        """
        self.error_pattern_system = error_pattern_system
        self.model_dir = model_dir or os.path.join(os.getcwd(), "models", "failure_prediction")
        self.config = config or {}
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model configuration
        self.min_training_samples = self.config.get('min_training_samples', 100)
        self.feature_importance_threshold = self.config.get('feature_importance_threshold', 0.02)
        self.prediction_threshold = self.config.get('prediction_threshold', 0.7)
        
        # Store models by strategy and timeframe
        self.models = {}
        self.feature_importances = {}
        self.model_performance = {}
        self.feature_columns = {}
        self.scalers = {}
        
        logger.info("PredictiveFailureModel initialized")

    def prepare_training_data(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        successful_trades: List[Dict[str, Any]],
        failed_trades: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from successful and failed trades.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            successful_trades: List of successful trade records
            failed_trades: List of failed trade records
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Features DataFrame and target labels
        """
        # Combine successful and failed trades
        all_trades = []
        
        for trade in successful_trades:
            if 'market_conditions' in trade:
                record = trade['market_conditions'].copy()
                record['outcome'] = 0  # Success
                all_trades.append(record)
                
        for trade in failed_trades:
            if 'market_conditions' in trade:
                record = trade['market_conditions'].copy()
                record['outcome'] = 1  # Failure
                all_trades.append(record)
        
        if not all_trades:
            raise ValueError("No valid trade data with market conditions found")
            
        # Convert to DataFrame
        df = pd.DataFrame(all_trades)
        
        # Check if we have enough data
        if len(df) < self.min_training_samples:
            raise ValueError(
                f"Insufficient training data: {len(df)} samples (need {self.min_training_samples})"
            )
            
        # Handle common preprocessing
        df = self._preprocess_features(df)
        
        # Extract target and features
        y = df['outcome'].values
        X = df.drop('outcome', axis=1)
        
        # Store feature columns for future predictions
        key = f"{strategy_id}:{instrument}:{timeframe}"
        self.feature_columns[key] = X.columns.tolist()
        
        logger.info(
            "Prepared training data for %s on %s (%s): %d features, %d samples, %.1f%% failures",
            strategy_id, instrument, timeframe, X.shape[1], X.shape[0], 100 * y.mean()
        )
        
        return X, y
        
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for model training.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Handle categorical features
        for col in df.columns:
            if df[col].dtype == 'object':
                # For string/categorical columns, create dummy variables
                if col != 'outcome':  # Don't convert the target
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(col, axis=1, inplace=True)
        
        # Handle special time-based features
        if 'timestamp' in df.columns:
            # Extract useful time components
            timestamps = pd.to_datetime(df['timestamp'])
            df['hour_of_day'] = timestamps.dt.hour
            df['day_of_week'] = timestamps.dt.dayofweek
            df['day_of_month'] = timestamps.dt.day
            df['month'] = timestamps.dt.month
            df.drop('timestamp', axis=1, inplace=True)
            
        # Fill missing values with median for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                
        # Drop any remaining columns with missing values
        df = df.dropna(axis=1, how='any')
        
        return df
        
    def train_model(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train a predictive model for strategy failures.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            X: Feature DataFrame
            y: Target labels (1 for failure, 0 for success)
            
        Returns:
            Dict[str, Any]: Model performance metrics
        """
        key = f"{strategy_id}:{instrument}:{timeframe}"
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[key] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Train model (use GradientBoosting for better handling of imbalanced data)
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=15,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        threshold = self.prediction_threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label=1
        )
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store model and metrics
        self.models[key] = model
        
        # Extract feature importances
        feature_importances = {}
        for i, feature in enumerate(X.columns):
            importance = model.feature_importances_[i]
            if importance >= self.feature_importance_threshold:
                feature_importances[feature] = importance
                
        # Sort by importance
        self.feature_importances[key] = dict(
            sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
        )
        
        # Save model to disk
        model_filename = f"failure_model_{strategy_id}_{instrument}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_columns': X.columns.tolist(),
            'feature_importances': self.feature_importances[key],
            'training_date': datetime.now(),
            'threshold': threshold
        }
        
        try:
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
        
        # Store performance metrics
        performance = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'threshold': threshold,
            'training_samples': len(X),
            'failure_rate': y.mean(),
            'training_date': datetime.now(),
            'model_path': model_path
        }
        
        self.model_performance[key] = performance
        
        logger.info(
            "Trained failure prediction model for %s on %s (%s): AUC=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f",
            strategy_id, instrument, timeframe, auc, precision, recall, f1
        )
        
        return performance
        
    def predict_failure_risk(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict failure risk for given market conditions.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            market_conditions: Current market conditions
            
        Returns:
            Dict[str, Any]: Prediction results with risk score and contributing factors
        """
        key = f"{strategy_id}:{instrument}:{timeframe}"
        
        # Check if we have a trained model
        if key not in self.models:
            raise ValueError(f"No trained model found for {strategy_id} on {instrument} ({timeframe})")
            
        model = self.models[key]
        feature_columns = self.feature_columns[key]
        scaler = self.scalers[key]
        
        # Prepare input data
        input_data = market_conditions.copy()
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess
        input_df = self._preprocess_features(input_df)
        
        # Ensure all expected columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing columns
        
        # Select only the features used during training
        input_df = input_df[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        failure_probability = model.predict_proba(input_scaled)[0, 1]
        is_high_risk = failure_probability >= self.prediction_threshold
        
        # Get contributing factors (feature importance * feature value)
        contributing_factors = []
        
        # Calculate normalized feature values
        scaled_values = input_scaled[0]
        feature_contributions = {}
        
        for i, feature in enumerate(feature_columns):
            if feature in self.feature_importances[key]:
                importance = self.feature_importances[key][feature]
                scaled_value = scaled_values[i]
                
                # Only consider features that contribute to positive prediction
                contribution = importance * scaled_value
                feature_contributions[feature] = contribution
        
        # Sort by absolute contribution and get top contributors
        top_contributors = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]  # Top 5 contributors
        
        for feature, contribution in top_contributors:
            contributing_factors.append({
                'feature': feature,
                'importance': self.feature_importances[key][feature],
                'value': input_df[feature].values[0],
                'contribution': contribution,
                'direction': 'increases risk' if contribution > 0 else 'decreases risk'
            })
        
        # Generate textual insight about the prediction
        if is_high_risk:
            primary_factor = contributing_factors[0]['feature'] if contributing_factors else "unknown factors"
            insight = f"High failure risk due primarily to {primary_factor}"
        else:
            insight = "Low failure risk for current conditions"
        
        result = {
            'strategy_id': strategy_id,
            'instrument': instrument,
            'timeframe': timeframe,
            'failure_probability': failure_probability,
            'is_high_risk': is_high_risk,
            'prediction_threshold': self.prediction_threshold,
            'contributing_factors': contributing_factors,
            'insight': insight,
            'timestamp': datetime.now()
        }
        
        logger.info(
            "Predicted failure risk for %s on %s (%s): %.1f%% (%s)",
            strategy_id, instrument, timeframe, 
            100 * failure_probability,
            "HIGH RISK" if is_high_risk else "low risk"
        )
        
        return result
        
    def load_model(self, strategy_id: str, instrument: str, timeframe: str, model_path: str = None) -> bool:
        """
        Load a previously trained model from disk.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            model_path: Path to the model file (will search for latest if None)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        key = f"{strategy_id}:{instrument}:{timeframe}"
        
        try:
            # If no specific path, find the latest model
            if model_path is None:
                pattern = f"failure_model_{strategy_id}_{instrument}_{timeframe}_*.pkl"
                model_files = [f for f in os.listdir(self.model_dir) if f.startswith(f"failure_model_{strategy_id}_{instrument}_{timeframe}_")]
                
                if not model_files:
                    logger.warning(f"No model files found matching {pattern}")
                    return False
                    
                # Sort by creation time (newest first)
                model_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.model_dir, f)), reverse=True)
                model_path = os.path.join(self.model_dir, model_files[0])
            
            # Load model data
            model_data = joblib.load(model_path)
            
            # Extract components
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            feature_importances = model_data.get('feature_importances', {})
            threshold = model_data.get('threshold', self.prediction_threshold)
            
            # Store in memory
            self.models[key] = model
            self.scalers[key] = scaler
            self.feature_columns[key] = feature_columns
            self.feature_importances[key] = feature_importances
            
            # Set performance metrics placeholder
            self.model_performance[key] = {
                'threshold': threshold,
                'model_path': model_path,
                'loading_date': datetime.now()
            }
            
            logger.info(
                "Loaded failure prediction model for %s on %s (%s) from %s",
                strategy_id, instrument, timeframe, model_path
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to load model for %s on %s (%s): %s",
                strategy_id, instrument, timeframe, str(e)
            )
            return False
            
    def get_model_performance(self, strategy_id: str, instrument: str, timeframe: str) -> Dict[str, Any]:
        """
        Get performance metrics for a trained model.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            
        Returns:
            Dict[str, Any]: Model performance metrics
        """
        key = f"{strategy_id}:{instrument}:{timeframe}"
        
        if key not in self.model_performance:
            return {'status': 'no_model_found'}
            
        return self.model_performance[key]
        
    def get_feature_importances(self, strategy_id: str, instrument: str, timeframe: str) -> Dict[str, float]:
        """
        Get feature importances for a trained model.
        
        Args:
            strategy_id: Identifier for the strategy
            instrument: The trading instrument
            timeframe: The timeframe for analysis
            
        Returns:
            Dict[str, float]: Feature importances
        """
        key = f"{strategy_id}:{instrument}:{timeframe}"
        
        if key not in self.feature_importances:
            return {}
            
        return self.feature_importances[key]
