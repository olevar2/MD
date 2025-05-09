"""
ML Optimization Example

This script demonstrates how to use the ML optimization tools to improve
performance of machine learning models and pipelines.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path

# Add parent directory to path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_workbench_service.optimization import (
    ModelInferenceOptimizer,
    FeatureEngineeringOptimizer,
    ModelTrainingOptimizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples=1000, n_features=20):
    """Create sample data for demonstration."""
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1) > 0.5
    return X, y.astype(np.int32)

def create_sample_model_tensorflow():
    """Create a sample TensorFlow model for demonstration."""
    try:
        import tensorflow as tf
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except ImportError:
        logger.warning("TensorFlow not available. Skipping TensorFlow model creation.")
        return None

def create_sample_model_pytorch():
    """Create a sample PyTorch model for demonstration."""
    try:
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(20, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                return self.layers(x)
        
        return SimpleModel()
    except ImportError:
        logger.warning("PyTorch not available. Skipping PyTorch model creation.")
        return None

def feature_mean(data):
    """Sample feature function: compute mean of each feature."""
    return data.mean(axis=0)

def feature_std(data):
    """Sample feature function: compute standard deviation of each feature."""
    return data.std(axis=0)

def feature_quantiles(data):
    """Sample feature function: compute quantiles of each feature."""
    return np.percentile(data, [25, 50, 75], axis=0)

def feature_correlations(data):
    """Sample feature function: compute correlation matrix."""
    return np.corrcoef(data, rowvar=False)

def demonstrate_inference_optimization():
    """Demonstrate model inference optimization."""
    logger.info("Demonstrating model inference optimization...")
    
    # Create sample data and model
    X, y = create_sample_data()
    
    # Try TensorFlow first
    tf_model = create_sample_model_tensorflow()
    if tf_model is not None:
        # Train the model briefly
        tf_model.fit(X, y, epochs=1, verbose=0)
        
        # Save the model
        model_dir = Path("./optimization_output/models")
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / "sample_tf_model"
        tf_model.save(model_path)
        
        # Initialize optimizer
        optimizer = ModelInferenceOptimizer(
            model_path=str(model_path),
            framework="tensorflow",
            device="cpu"
        )
        
        # Benchmark baseline performance
        baseline_metrics = optimizer.benchmark_baseline(
            input_data=X,
            batch_sizes=[1, 32, 64]
        )
        logger.info(f"Baseline metrics: {baseline_metrics}")
        
        # Apply quantization
        quantized_model, quantization_metadata = optimizer.quantize_model(
            quantization_type="float16"
        )
        logger.info(f"Quantization metadata: {quantization_metadata}")
        
        # Apply operator fusion
        fused_model, fusion_metadata = optimizer.apply_operator_fusion()
        logger.info(f"Fusion metadata: {fusion_metadata}")
        
        # Configure batch inference
        batch_config = optimizer.configure_batch_inference(
            optimal_batch_size=32
        )
        logger.info(f"Batch inference config: {batch_config}")
        
        # Benchmark optimized performance
        optimized_metrics = optimizer.benchmark_optimized(
            input_data=X,
            batch_sizes=[1, 32, 64]
        )
        logger.info(f"Optimized metrics: {optimized_metrics}")
        
        return
    
    # Fall back to PyTorch if TensorFlow is not available
    pt_model = create_sample_model_pytorch()
    if pt_model is not None:
        # Save the model
        try:
            import torch
            model_dir = Path("./optimization_output/models")
            model_dir.mkdir(exist_ok=True, parents=True)
            model_path = model_dir / "sample_pt_model.pt"
            torch.save(pt_model, model_path)
            
            # Initialize optimizer
            optimizer = ModelInferenceOptimizer(
                model_path=str(model_path),
                framework="pytorch",
                device="cpu"
            )
            
            # Convert input data to PyTorch tensor
            input_data = torch.tensor(X, dtype=torch.float32)
            
            # Benchmark baseline performance
            baseline_metrics = optimizer.benchmark_baseline(
                input_data=input_data,
                batch_sizes=[1, 32, 64]
            )
            logger.info(f"Baseline metrics: {baseline_metrics}")
            
            # Apply operator fusion
            fused_model, fusion_metadata = optimizer.apply_operator_fusion()
            logger.info(f"Fusion metadata: {fusion_metadata}")
            
            # Configure batch inference
            batch_config = optimizer.configure_batch_inference(
                optimal_batch_size=32
            )
            logger.info(f"Batch inference config: {batch_config}")
            
            # Benchmark optimized performance
            optimized_metrics = optimizer.benchmark_optimized(
                input_data=input_data,
                batch_sizes=[1, 32, 64]
            )
            logger.info(f"Optimized metrics: {optimized_metrics}")
        except Exception as e:
            logger.error(f"Error in PyTorch optimization: {str(e)}")

def demonstrate_feature_optimization():
    """Demonstrate feature engineering optimization."""
    logger.info("Demonstrating feature engineering optimization...")
    
    # Create sample data
    X, _ = create_sample_data(n_samples=10000)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Initialize optimizer
    optimizer = FeatureEngineeringOptimizer(
        cache_dir="./optimization_output/feature_cache",
        n_jobs=4
    )
    
    # Define feature functions
    feature_funcs = [
        feature_mean,
        feature_std,
        feature_quantiles,
        feature_correlations
    ]
    
    # Benchmark feature pipeline
    benchmark_results = optimizer.benchmark_feature_pipeline(
        data=df,
        feature_funcs=feature_funcs,
        use_cache=True,
        use_parallel=True,
        n_runs=3
    )
    logger.info(f"Benchmark results: {benchmark_results['overall']}")
    
    # Compute features in parallel with caching
    features, metadata = optimizer.parallel_feature_computation(
        data=df,
        feature_funcs=feature_funcs,
        use_cache=True
    )
    logger.info(f"Parallel computation metadata: {metadata}")
    
    # Demonstrate incremental computation
    new_data = pd.DataFrame(
        np.random.randn(1000, X.shape[1]),
        columns=[f"feature_{i}" for i in range(X.shape[1])]
    )
    
    incremental_features, inc_metadata = optimizer.incremental_feature_computation(
        previous_data=df,
        previous_features=features,
        new_data=new_data,
        feature_funcs=feature_funcs
    )
    logger.info(f"Incremental computation metadata: {inc_metadata}")

def demonstrate_training_optimization():
    """Demonstrate model training optimization."""
    logger.info("Demonstrating model training optimization...")
    
    # Create sample data
    X, y = create_sample_data()
    
    # Try TensorFlow first
    tf_model = create_sample_model_tensorflow()
    if tf_model is not None:
        # Create TensorFlow dataset
        try:
            import tensorflow as tf
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            
            # Initialize optimizer
            optimizer = ModelTrainingOptimizer(
                model=tf_model,
                framework="tensorflow",
                device="cpu",
                output_dir="./optimization_output/training"
            )
            
            # Configure mixed precision
            mp_config = optimizer.configure_mixed_precision(
                enabled=True,
                precision="float16"
            )
            logger.info(f"Mixed precision config: {mp_config}")
            
            # Configure gradient accumulation
            ga_config = optimizer.configure_gradient_accumulation(
                accumulation_steps=4
            )
            logger.info(f"Gradient accumulation config: {ga_config}")
            
            # Benchmark training
            benchmark_results = optimizer.benchmark_training(
                train_dataset=dataset,
                batch_size=32,
                num_epochs=1,
                mixed_precision=True,
                gradient_accumulation_steps=4
            )
            logger.info(f"Training benchmark results: {benchmark_results}")
            
            return
        except Exception as e:
            logger.error(f"Error in TensorFlow training optimization: {str(e)}")
    
    # Fall back to PyTorch if TensorFlow is not available
    pt_model = create_sample_model_pytorch()
    if pt_model is not None:
        try:
            import torch
            from torch.utils.data import TensorDataset
            
            # Create PyTorch dataset
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Initialize optimizer
            optimizer = ModelTrainingOptimizer(
                model=pt_model,
                framework="pytorch",
                device="cpu",
                output_dir="./optimization_output/training"
            )
            
            # Configure mixed precision
            mp_config = optimizer.configure_mixed_precision(
                enabled=True,
                precision="float16"
            )
            logger.info(f"Mixed precision config: {mp_config}")
            
            # Configure gradient accumulation
            ga_config = optimizer.configure_gradient_accumulation(
                accumulation_steps=4
            )
            logger.info(f"Gradient accumulation config: {ga_config}")
            
            # Benchmark training
            benchmark_results = optimizer.benchmark_training(
                train_dataset=dataset,
                batch_size=32,
                num_epochs=1,
                mixed_precision=True,
                gradient_accumulation_steps=4
            )
            logger.info(f"Training benchmark results: {benchmark_results}")
        except Exception as e:
            logger.error(f"Error in PyTorch training optimization: {str(e)}")

def main():
    """Main function to run all demonstrations."""
    # Create output directory
    output_dir = Path("./optimization_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run demonstrations
    demonstrate_inference_optimization()
    demonstrate_feature_optimization()
    demonstrate_training_optimization()

if __name__ == "__main__":
    main()
