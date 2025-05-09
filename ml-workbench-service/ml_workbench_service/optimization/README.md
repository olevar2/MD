# ML Optimization Package

This package provides tools for optimizing machine learning models and pipelines in the Forex Trading Platform.

## Overview

The ML Optimization package includes three main components:

1. **Model Inference Optimizer**: Optimizes ML model inference performance through techniques like quantization, operator fusion, and batch inference.
2. **Feature Engineering Optimizer**: Optimizes feature engineering pipelines through techniques like caching, incremental computation, and parallel processing.
3. **Model Training Optimizer**: Optimizes ML model training performance through techniques like mixed precision, gradient accumulation, and distributed training.

## Installation

The package is part of the ML Workbench Service and is installed automatically with it.

## Usage

### Model Inference Optimization

```python
from ml_workbench_service.optimization import ModelInferenceOptimizer

# Initialize optimizer
optimizer = ModelInferenceOptimizer(
    model_path="path/to/model",
    framework="tensorflow",  # "tensorflow", "pytorch", "onnx"
    device="cpu"  # "cpu", "gpu"
)

# Benchmark baseline performance
baseline_metrics = optimizer.benchmark_baseline(
    input_data=X,
    batch_sizes=[1, 32, 64]
)

# Apply quantization
quantized_model, metadata = optimizer.quantize_model(
    quantization_type="int8"  # "int8", "float16", "dynamic"
)

# Apply operator fusion
fused_model, metadata = optimizer.apply_operator_fusion()

# Configure batch inference
batch_config = optimizer.configure_batch_inference(
    optimal_batch_size=32
)

# Benchmark optimized performance
optimized_metrics = optimizer.benchmark_optimized(
    input_data=X,
    batch_sizes=[1, 32, 64]
)
```

### Feature Engineering Optimization

```python
from ml_workbench_service.optimization import FeatureEngineeringOptimizer

# Initialize optimizer
optimizer = FeatureEngineeringOptimizer(
    cache_dir="./feature_cache",
    n_jobs=4  # Number of parallel jobs
)

# Compute features with caching
features, metadata = optimizer.cached_feature_computation(
    data=df,
    feature_func=compute_feature
)

# Compute multiple features in parallel
features, metadata = optimizer.parallel_feature_computation(
    data=df,
    feature_funcs=[feature_func1, feature_func2, feature_func3],
    use_cache=True
)

# Compute features incrementally for new data
incremental_features, metadata = optimizer.incremental_feature_computation(
    previous_data=df,
    previous_features=features,
    new_data=new_df,
    feature_funcs=[feature_func1, feature_func2, feature_func3]
)

# Benchmark feature pipeline
benchmark_results = optimizer.benchmark_feature_pipeline(
    data=df,
    feature_funcs=[feature_func1, feature_func2, feature_func3],
    use_cache=True,
    use_parallel=True
)
```

### Model Training Optimization

```python
from ml_workbench_service.optimization import ModelTrainingOptimizer

# Initialize optimizer
optimizer = ModelTrainingOptimizer(
    model=model,
    framework="tensorflow",  # "tensorflow", "pytorch"
    device="auto"  # "auto", "cpu", "gpu", "tpu"
)

# Configure mixed precision training
mp_config = optimizer.configure_mixed_precision(
    enabled=True,
    precision="float16"  # "float16", "bfloat16"
)

# Configure gradient accumulation
ga_config = optimizer.configure_gradient_accumulation(
    accumulation_steps=4
)

# Configure distributed training
dist_config = optimizer.configure_distributed_training(
    strategy="mirrored"  # "mirrored", "multi_worker", "parameter_server", "tpu"
)

# Benchmark training performance
benchmark_results = optimizer.benchmark_training(
    train_dataset=dataset,
    batch_size=32,
    num_epochs=1,
    mixed_precision=True,
    gradient_accumulation_steps=4,
    distributed=True
)
```

## Command-Line Interface

The package also provides a command-line interface for optimizing models and pipelines:

```bash
# Optimize model inference
python -m ml_workbench_service.optimization.cli inference \
    --model-path path/to/model \
    --framework tensorflow \
    --quantize \
    --fusion \
    --batch-inference

# Optimize feature engineering
python -m ml_workbench_service.optimization.cli features \
    --data-path path/to/data.csv \
    --feature-functions path/to/feature_functions.py \
    --parallel \
    --benchmark

# Optimize model training
python -m ml_workbench_service.optimization.cli training \
    --model-path path/to/model \
    --data-path path/to/data \
    --framework tensorflow \
    --mixed-precision \
    --gradient-accumulation \
    --accumulation-steps 4
```

## Examples

See the `examples/ml_optimization_example.py` script for complete examples of using the optimization tools.

## Supported Frameworks

- **TensorFlow**: Full support for inference, training, and feature optimization
- **PyTorch**: Full support for inference, training, and feature optimization
- **ONNX**: Support for inference optimization only

## Performance Considerations

- **Model Quantization**: Can reduce model size by 2-4x and improve inference speed by 1.5-3x
- **Operator Fusion**: Can improve inference speed by 1.2-2x
- **Batch Inference**: Can improve throughput by 2-10x depending on the model
- **Feature Caching**: Can reduce feature computation time by 10-100x for repeated computations
- **Parallel Feature Computation**: Can reduce computation time by 2-8x depending on the number of cores
- **Mixed Precision Training**: Can reduce memory usage by 2x and improve training speed by 1.5-3x
- **Gradient Accumulation**: Enables training with larger effective batch sizes
- **Distributed Training**: Can reduce training time by N times with N workers

## Limitations

- Quantization may reduce model accuracy, especially for int8 quantization
- Not all models support operator fusion
- Feature caching requires sufficient disk space
- Mixed precision training may not be supported on all hardware
- Distributed training requires appropriate infrastructure

## Contributing

Contributions to the ML Optimization package are welcome. Please follow the standard contribution guidelines for the Forex Trading Platform.
