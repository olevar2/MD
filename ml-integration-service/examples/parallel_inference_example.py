"""
Example usage of the parallel inference framework.

This module demonstrates how to use the parallel inference framework
for various ML inference tasks, including single model inference,
multi-model inference, and batch inference.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pandas as pd
import numpy as np

from ml_integration_service.parallel import (
    ModelInferenceSpec,
    ParallelInferenceProcessor,
)

from data_pipeline_service.parallel import (
    ParallelizationMethod,
    TaskPriority,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Example functions for inference

def run_model_inference(features: pd.DataFrame, model: ModelInferenceSpec) -> Dict[str, Any]:
    """
    Example function to run inference for a single model.
    
    Args:
        features: Input features
        model: Model specification
        
    Returns:
        Inference result
    """
    logger.info(f"Running inference for model: {model.model_id}")
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Generate some random predictions based on model type
    if 'classifier' in model.model_id:
        # Classification model
        classes = ['bullish', 'neutral', 'bearish']
        class_idx = np.random.randint(0, len(classes))
        probabilities = np.random.dirichlet(np.ones(len(classes)))
        
        return {
            "model_id": model.model_id,
            "version": model.version,
            "prediction": classes[class_idx],
            "probabilities": {cls: float(prob) for cls, prob in zip(classes, probabilities)},
            "confidence": float(probabilities[class_idx]),
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Regression model
        prediction = float(np.random.normal(0, 1))
        
        return {
            "model_id": model.model_id,
            "version": model.version,
            "prediction": prediction,
            "confidence": float(np.random.uniform(0.7, 0.99)),
            "timestamp": datetime.now().isoformat()
        }


def run_batch_inference(features_batch: List[pd.DataFrame], model: ModelInferenceSpec) -> List[Dict[str, Any]]:
    """
    Example function to run batch inference for a single model.
    
    Args:
        features_batch: List of feature DataFrames
        model: Model specification
        
    Returns:
        List of inference results
    """
    logger.info(f"Running batch inference for model: {model.model_id} with batch size {len(features_batch)}")
    
    # Simulate processing time
    time.sleep(0.8)
    
    # Generate predictions for each item in the batch
    results = []
    for i, features in enumerate(features_batch):
        # Similar logic to single inference, but for a batch
        if 'classifier' in model.model_id:
            # Classification model
            classes = ['bullish', 'neutral', 'bearish']
            class_idx = np.random.randint(0, len(classes))
            probabilities = np.random.dirichlet(np.ones(len(classes)))
            
            results.append({
                "batch_index": i,
                "model_id": model.model_id,
                "version": model.version,
                "prediction": classes[class_idx],
                "probabilities": {cls: float(prob) for cls, prob in zip(classes, probabilities)},
                "confidence": float(probabilities[class_idx]),
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Regression model
            prediction = float(np.random.normal(0, 1))
            
            results.append({
                "batch_index": i,
                "model_id": model.model_id,
                "version": model.version,
                "prediction": prediction,
                "confidence": float(np.random.uniform(0.7, 0.99)),
                "timestamp": datetime.now().isoformat()
            })
    
    return results


async def example_single_model_inference():
    """Example of single model inference."""
    logger.info("Running single model inference example")
    
    # Create processor
    processor = ParallelInferenceProcessor()
    
    # Create sample features
    features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'feature4': np.random.normal(0, 1, 100),
        'feature5': np.random.normal(0, 1, 100)
    })
    
    # Define models
    models = [
        ModelInferenceSpec(
            model_id='trend_classifier',
            version='1.0.0',
            params={'threshold': 0.5},
            priority=TaskPriority.HIGH
        )
    ]
    
    # Run inference
    results = await processor.run_inference(
        features=features,
        models=models,
        inference_func=run_model_inference
    )
    
    # Print results
    logger.info(f"Inference results for {len(results)} models")
    for model_id, result in results.items():
        if 'probabilities' in result:
            logger.info(f"  {model_id}: prediction={result['prediction']}, confidence={result['confidence']:.2f}")
            logger.info(f"    probabilities: {result['probabilities']}")
        else:
            logger.info(f"  {model_id}: prediction={result['prediction']:.4f}, confidence={result['confidence']:.2f}")


async def example_multi_model_inference():
    """Example of multi-model inference."""
    logger.info("Running multi-model inference example")
    
    # Create processor
    processor = ParallelInferenceProcessor()
    
    # Create sample features
    features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'feature4': np.random.normal(0, 1, 100),
        'feature5': np.random.normal(0, 1, 100)
    })
    
    # Define models
    models = [
        ModelInferenceSpec(
            model_id='trend_classifier',
            version='1.0.0',
            params={'threshold': 0.5},
            priority=TaskPriority.HIGH
        ),
        ModelInferenceSpec(
            model_id='price_predictor',
            version='2.1.0',
            params={'horizon': 10},
            priority=TaskPriority.MEDIUM
        ),
        ModelInferenceSpec(
            model_id='volatility_predictor',
            version='1.5.0',
            params={'window': 20},
            priority=TaskPriority.LOW
        )
    ]
    
    # Run inference
    results = await processor.run_inference(
        features=features,
        models=models,
        inference_func=run_model_inference
    )
    
    # Print results
    logger.info(f"Inference results for {len(results)} models")
    for model_id, result in results.items():
        if 'probabilities' in result:
            logger.info(f"  {model_id}: prediction={result['prediction']}, confidence={result['confidence']:.2f}")
        else:
            logger.info(f"  {model_id}: prediction={result['prediction']:.4f}, confidence={result['confidence']:.2f}")


async def example_batch_inference():
    """Example of batch inference."""
    logger.info("Running batch inference example")
    
    # Create processor
    processor = ParallelInferenceProcessor()
    
    # Create sample features batch
    features_batch = []
    for _ in range(5):
        features = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'feature4': np.random.normal(0, 1, 100),
            'feature5': np.random.normal(0, 1, 100)
        })
        features_batch.append(features)
    
    # Define model
    model = ModelInferenceSpec(
        model_id='trend_classifier',
        version='1.0.0',
        params={'threshold': 0.5},
        priority=TaskPriority.HIGH
    )
    
    # Run batch inference
    results = await processor.run_batch_inference(
        features_batch=features_batch,
        model=model,
        inference_func=run_batch_inference
    )
    
    # Print results
    logger.info(f"Batch inference results for {len(results)} samples")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        logger.info(f"  Sample {result['batch_index']}: prediction={result['prediction']}, confidence={result['confidence']:.2f}")
    
    if len(results) > 3:
        logger.info(f"  ... and {len(results) - 3} more")


async def example_multi_model_batch_inference():
    """Example of multi-model batch inference."""
    logger.info("Running multi-model batch inference example")
    
    # Create processor
    processor = ParallelInferenceProcessor()
    
    # Create sample features batch
    features_batch = []
    for _ in range(5):
        features = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'feature4': np.random.normal(0, 1, 100),
            'feature5': np.random.normal(0, 1, 100)
        })
        features_batch.append(features)
    
    # Define models
    models = [
        ModelInferenceSpec(
            model_id='trend_classifier',
            version='1.0.0',
            params={'threshold': 0.5},
            priority=TaskPriority.HIGH
        ),
        ModelInferenceSpec(
            model_id='price_predictor',
            version='2.1.0',
            params={'horizon': 10},
            priority=TaskPriority.MEDIUM
        )
    ]
    
    # Run multi-model batch inference
    results = await processor.run_multi_model_batch_inference(
        features_batch=features_batch,
        models=models,
        inference_func=run_batch_inference
    )
    
    # Print results
    logger.info(f"Multi-model batch inference results for {len(results)} models")
    for model_id, model_results in results.items():
        logger.info(f"  Model {model_id}: {len(model_results)} batch results")
        for i, result in enumerate(model_results[:2]):  # Show first 2 results per model
            logger.info(f"    Sample {result['batch_index']}: prediction={result['prediction']}, confidence={result['confidence']:.2f}")
        
        if len(model_results) > 2:
            logger.info(f"    ... and {len(model_results) - 2} more")


async def main():
    """Run all examples."""
    await example_single_model_inference()
    print()
    
    await example_multi_model_inference()
    print()
    
    await example_batch_inference()
    print()
    
    await example_multi_model_batch_inference()


if __name__ == "__main__":
    asyncio.run(main())
