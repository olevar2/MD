#!/usr/bin/env python
"""
Example script demonstrating how to use the Historical Data Management service for ML model training.

This script shows how to create and use ML datasets for training predictive models.
"""

import asyncio
import datetime
import logging
import os
from typing import Dict, List, Any, Tuple

import httpx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API URL
API_URL = "http://localhost:8000"


async def create_ml_dataset(
    symbols: List[str],
    timeframes: List[str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    features: List[str],
    target: str,
    transformations: List[Dict[str, Any]],
    validation_split: float = 0.2,
    test_split: float = 0.1
) -> pd.DataFrame:
    """
    Create a dataset for machine learning.
    
    Args:
        symbols: List of trading symbols
        timeframes: List of timeframes
        start_date: Start date
        end_date: End date
        features: List of features to include
        target: Target variable
        transformations: List of transformations to apply
        validation_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        DataFrame with the dataset
    """
    # Create dataset config
    config = {
        "dataset_id": f"ml_example_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "name": "ML Example Dataset",
        "symbols": symbols,
        "timeframes": timeframes,
        "start_timestamp": start_date.isoformat(),
        "end_timestamp": end_date.isoformat(),
        "features": features,
        "target": target,
        "transformations": transformations,
        "validation_split": validation_split,
        "test_split": test_split
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/historical/ml-dataset",
            json=config,
            params={"format": "json"}
        )
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df


def prepare_data(
    df: pd.DataFrame,
    features: List[str],
    target: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for ML model training.
    
    Args:
        df: DataFrame with the dataset
        features: List of features to use
        target: Target variable
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Drop rows with missing values
    df = df.dropna()
    
    # Split data
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    
    # Extract features and target
    X_train = train_df[features].values
    y_train = train_df[target].values
    
    X_val = val_df[features].values
    y_val = val_df[target].values
    
    X_test = test_df[features].values
    y_test = test_df[target].values
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Trained model
    """
    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_accuracy = model.score(X_val, y_val)
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        
    Returns:
        Dictionary with evaluation results
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = model.score(X_test, y_test)
    
    # Get feature importances
    feature_importances = dict(zip(feature_names, model.feature_importances_))
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "feature_importances": feature_importances,
        "classification_report": report,
        "confusion_matrix": cm
    }


def plot_results(results: Dict[str, Any], feature_names: List[str]) -> None:
    """
    Plot evaluation results.
    
    Args:
        results: Dictionary with evaluation results
        feature_names: Names of features
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot feature importances
    importances = pd.Series(results["feature_importances"]).sort_values(ascending=False)
    importances.plot(kind="bar", ax=ax1)
    ax1.set_title("Feature Importances")
    ax1.set_ylabel("Importance")
    ax1.set_xlabel("Feature")
    
    # Plot confusion matrix
    sns.heatmap(
        results["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax2,
        xticklabels=["Down", "Up"],
        yticklabels=["Down", "Up"]
    )
    ax2.set_title("Confusion Matrix")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")
    
    plt.tight_layout()
    plt.savefig("ml_results.png")
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    report = results["classification_report"]
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision (Class 1): {report['1']['precision']:.4f}")
    print(f"Recall (Class 1): {report['1']['recall']:.4f}")
    print(f"F1-score (Class 1): {report['1']['f1-score']:.4f}")


async def main() -> None:
    """Main entry point."""
    logger.info("Starting ML example")
    
    # Set parameters
    symbols = ["EURUSD"]
    timeframes = ["1h"]
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=60)
    
    # Define features and transformations
    features = ["open", "high", "low", "close", "volume", "sma_10", "sma_20", "rsi_14"]
    target = "target_direction_1"
    
    transformations = [
        {
            "type": "add_technical_indicator",
            "indicator": "sma",
            "params": {"period": 10}
        },
        {
            "type": "add_technical_indicator",
            "indicator": "sma",
            "params": {"period": 20}
        },
        {
            "type": "add_technical_indicator",
            "indicator": "rsi",
            "params": {"period": 14}
        },
        {
            "type": "add_target",
            "target_type": "direction",
            "periods": 1
        }
    ]
    
    # Create ML dataset
    logger.info("Creating ML dataset")
    df = await create_ml_dataset(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        features=features,
        target=target,
        transformations=transformations,
        validation_split=0.2,
        test_split=0.1
    )
    
    if df.empty:
        logger.error("Failed to create ML dataset")
        return
    
    logger.info(f"Created dataset with {len(df)} records")
    
    # Prepare data
    logger.info("Preparing data")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df, features, target)
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    logger.info("Training model")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    logger.info("Evaluating model")
    results = evaluate_model(model, X_test, y_test, features)
    
    logger.info(f"Test accuracy: {results['accuracy']:.4f}")
    
    # Plot results
    logger.info("Plotting results")
    plot_results(results, features)
    
    logger.info("ML example complete")


if __name__ == "__main__":
    asyncio.run(main())
