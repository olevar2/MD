"""
Machine Learning Interfaces Module

This module provides interfaces for machine learning components used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field


class ModelType(str, Enum):
    """Types of machine learning models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    REINFORCEMENT = "reinforcement"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"


class ModelFramework(str, Enum):
    """Machine learning frameworks."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    STABLE_BASELINES = "stable_baselines"
    CUSTOM = "custom"


class IMLModelProvider(ABC):
    """Interface for ML model providers."""

    @abstractmethod
    async def get_model_prediction(
        self,
        model_id: str,
        features: Dict[str, Any],
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get prediction from a machine learning model.

        Args:
            model_id: ID of the model to use
            features: Input features for prediction
            version: Optional model version

        Returns:
            Dictionary with prediction results
        """
        pass

    @abstractmethod
    async def get_model_metadata(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metadata about a machine learning model.

        Args:
            model_id: ID of the model
            version: Optional model version

        Returns:
            Dictionary with model metadata
        """
        pass

    @abstractmethod
    async def get_feature_importance(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get feature importance for a model.

        Args:
            model_id: ID of the model
            version: Optional model version

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass


class IRLModelTrainer(ABC):
    """Interface for reinforcement learning model trainers."""

    @abstractmethod
    async def train_model(
        self,
        model_config: Dict[str, Any],
        environment_config: Dict[str, Any],
        training_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a reinforcement learning model.

        Args:
            model_config: Configuration for the model
            environment_config: Configuration for the environment
            training_params: Parameters for training

        Returns:
            Dictionary with training results
        """
        pass

    @abstractmethod
    async def evaluate_model(
        self,
        model_id: str,
        environment_config: Dict[str, Any],
        evaluation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a reinforcement learning model.

        Args:
            model_id: ID of the model to evaluate
            environment_config: Configuration for the environment
            evaluation_params: Parameters for evaluation

        Returns:
            Dictionary with evaluation results
        """
        pass

    @abstractmethod
    async def optimize_hyperparameters(
        self,
        model_type: str,
        environment_config: Dict[str, Any],
        hyperparameter_space: Dict[str, Any],
        optimization_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a reinforcement learning model.

        Args:
            model_type: Type of model to optimize
            environment_config: Configuration for the environment
            hyperparameter_space: Space of hyperparameters to search
            optimization_params: Parameters for optimization

        Returns:
            Dictionary with optimization results
        """
        pass


@dataclass
class ModelConfiguration:
    """Model configuration data"""
    model_id: str
    name: str
    version: str
    source: str
    model_type: str
    features: List[str]
    target: str
    metadata: Optional[Dict[str, Any]] = None
    api_config: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None


@dataclass
class ModelPrediction:
    """Model prediction data"""
    model_id: str
    timestamp: datetime
    prediction: Any
    confidence: float
    features_used: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class IMLModelConnector(ABC):
    """Interface for ML model connector functionality"""

    @abstractmethod
    async def get_market_analysis(
        self,
        symbol: str,
        timeframe: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get market analysis for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            user_preferences: Optional user preferences

        Returns:
            Market analysis data
        """
        pass

    @abstractmethod
    async def get_price_prediction(
        self,
        symbol: str,
        timeframe: str,
        horizon: str = "short_term",
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get price prediction for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for prediction
            horizon: Prediction horizon
            user_preferences: Optional user preferences

        Returns:
            Price prediction data
        """
        pass

    @abstractmethod
    async def get_trading_recommendation(
        self,
        symbol: str,
        timeframe: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get trading recommendation for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for recommendation
            user_preferences: Optional user preferences

        Returns:
            Trading recommendation data
        """
        pass

    @abstractmethod
    async def get_sentiment_analysis(
        self,
        symbol: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis for a symbol.

        Args:
            symbol: Trading symbol
            lookback_days: Number of days to look back

        Returns:
            Sentiment analysis data
        """
        pass


class IExplanationGenerator(ABC):
    """Interface for explanation generator functionality"""

    @abstractmethod
    async def generate_explanation(
        self,
        model_type: str,
        prediction: Dict[str, Any],
        inputs: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for a model prediction.

        Args:
            model_type: Type of model
            prediction: Model prediction
            inputs: Model inputs
            user_preferences: Optional user preferences

        Returns:
            Explanation data
        """
        pass

    @abstractmethod
    async def get_feature_importance(
        self,
        model_type: str,
        model_id: str,
        prediction: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get feature importance for a model prediction.

        Args:
            model_type: Type of model
            model_id: Model identifier
            prediction: Model prediction
            inputs: Model inputs

        Returns:
            Feature importance data
        """
        pass


class IUserPreferenceManager(ABC):
    """Interface for user preference manager functionality"""

    @abstractmethod
    async def get_user_preferences(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get user preferences.

        Args:
            user_id: User identifier

        Returns:
            User preferences
        """
        pass

    @abstractmethod
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Update user preferences.

        Args:
            user_id: User identifier
            preferences: User preferences to update

        Returns:
            Success flag
        """
        pass

    @abstractmethod
    async def detect_preferences_from_message(
        self,
        user_id: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Detect user preferences from a message.

        Args:
            user_id: User identifier
            message: User message

        Returns:
            Detected preferences
        """
        pass


class IAnalysisEngineClient(ABC):
    """Interface for analysis engine client functionality"""

    @abstractmethod
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get market data for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            bars: Number of bars to retrieve
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            Market data
        """
        pass

    @abstractmethod
    async def get_technical_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicators: List[Dict[str, Any]],
        bars: int = 100
    ) -> Dict[str, Any]:
        """
        Get technical indicators for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for indicators
            indicators: List of indicator configurations
            bars: Number of bars to calculate indicators for

        Returns:
            Technical indicator data
        """
        pass

    @abstractmethod
    async def get_market_regime(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get market regime for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for regime detection

        Returns:
            Market regime data
        """
        pass

    @abstractmethod
    async def get_support_resistance_levels(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get support and resistance levels for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for level detection

        Returns:
            Support and resistance levels
        """
        pass
