"""
Example of Using the New Feature Extraction Framework

This module demonstrates how to use the consolidated feature extraction framework
from the Analysis Engine service within the ML Integration Service.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from core.feature_extraction import (
    FeatureExtractor, FeatureDefinition, FeatureType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features_for_model(
    data: pd.DataFrame,
    task_type: str = "direction",
) -> pd.DataFrame:
    """
    Extract features for a machine learning model using the legacy framework.

    Args:
        data: DataFrame with market data and indicators
        task_type: Type of prediction task ('direction', 'volatility', 'price_level')

    Returns:
        DataFrame with extracted features
    """
    # Create feature extractor (now always returns legacy)
    feature_extractor = FeatureExtractor.create() # Removed use_analysis_engine argument

    # Get standard feature set for the task
    # The check for get_standard_feature_set is no longer needed as we always get legacy
    # if hasattr(feature_extractor, "get_standard_feature_set"):
    #     # Using Analysis Engine client
    #     feature_definitions = feature_extractor.get_standard_feature_set(task_type)
    #     logger.info(f"Using standard feature set from Analysis Engine for task: {task_type}")
    # else:
    # Using legacy implementation - define features manually
    feature_definitions = get_legacy_standard_features(task_type)
    logger.info(f"Using legacy feature definitions for task: {task_type}")

    # Extract features
    features_df = feature_extractor.extract_features(
        data=data,
        feature_definitions=feature_definitions,
        fit_scalers=True
    )

    logger.info(f"Extracted {len(features_df.columns)} features")

    return features_df


def get_legacy_standard_features(task_type: str) -> List[FeatureDefinition]:
    """
    Get standard feature definitions for a prediction task (legacy implementation)

    Args:
        task_type: Type of prediction task ('direction', 'volatility', 'price_level')

    Returns:
        List of feature definitions
    """
    if task_type == "direction":
        return [
            # Price features
            FeatureDefinition(
                name="price_normalized",
                source_columns=["close"],
                feature_type=FeatureType.NORMALIZED
            ),

            # RSI features
            FeatureDefinition(
                name="rsi_normalized",
                source_columns=["rsi_14"],
                feature_type=FeatureType.NORMALIZED,
                params={"min_value": 0, "max_value": 100}
            ),

            # MACD features
            FeatureDefinition(
                name="macd_normalized",
                source_columns=["macd_12_26"],
                feature_type=FeatureType.NORMALIZED
            ),
            FeatureDefinition(
                name="macd_hist_normalized",
                source_columns=["macd_hist_12_26_9"],
                feature_type=FeatureType.NORMALIZED
            ),

            # Moving average features
            FeatureDefinition(
                name="ma_crossover",
                source_columns=["sma_10", "sma_50"],
                feature_type=FeatureType.CROSSOVER,
                params={"method": "binary"}
            ),
        ]

    elif task_type == "volatility":
        return [
            # ATR features
            FeatureDefinition(
                name="atr_normalized",
                source_columns=["atr_14"],
                feature_type=FeatureType.NORMALIZED
            ),

            # Bollinger Band features
            FeatureDefinition(
                name="bb_width",
                source_columns=["bb_upper_20_2", "bb_lower_20_2"],
                feature_type=FeatureType.VOLATILITY,
                params={"method": "bollinger_width"}
            ),
        ]

    elif task_type == "price_level":
        return [
            # Price distribution
            FeatureDefinition(
                name="price_zscore",
                source_columns=["close"],
                feature_type=FeatureType.RELATIVE,
                params={"method": "z_score", "window": 50}
            ),

            # Bollinger Band position
            FeatureDefinition(
                name="bb_position",
                source_columns=["close", "bb_upper_20_2", "bb_lower_20_2"],
                feature_type=FeatureType.COMPOSITE,
                params={"method": "custom"}
            ),
        ]

    else:
        logger.warning(f"Unknown task type: {task_type}. Using direction features as default.")
        return get_legacy_standard_features("direction")


def main():
    """
    Example usage - Will now always use the legacy framework
    """
    # Create sample data (replace with actual data loading)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1H')
    data = pd.DataFrame(index=dates)
    data['close'] = 100 + np.cumsum(np.random.randn(200) * 0.5)
    # Add dummy indicator columns that legacy features might expect
    data['rsi_14'] = np.random.rand(200) * 100
    data['sma_10'] = data['close'].rolling(10).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['atr_14'] = np.random.rand(200) * 2
    data['bb_upper_20_2'] = data['close'] + data['atr_14']
    data['bb_lower_20_2'] = data['close'] - data['atr_14']
    data['macd_12_26'] = np.random.randn(200)
    data['macd_hist_12_26_9'] = np.random.randn(200)
    data = data.dropna()

    logger.info("Extracting features using legacy framework...")
    features = extract_features_for_model(data, task_type="direction") # Removed use_analysis_engine argument
    print("\nDirection Features (Legacy):")
    print(features.head())
    print(f"Shape: {features.shape}")

    # Example for volatility
    # features_vol = extract_features_for_model(data, task_type="volatility", use_analysis_engine=False)
    # print("\nVolatility Features (Legacy):")
    # print(features_vol.head())


if __name__ == "__main__":
    main()
