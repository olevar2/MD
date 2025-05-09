"""
Hardware-Specific Optimization Example

This script demonstrates how to use the Hardware-Specific Optimizer to optimize
ML models for different hardware platforms like GPUs, TPUs, FPGAs, and CPUs.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_workbench_service.optimization import (
    ModelInferenceOptimizer,
    HardwareSpecificOptimizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_model_tensorflow():
    """Create a sample TensorFlow model for demonstration."""
    try:
        import tensorflow as tf

        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train the model briefly
        X = np.random.randn(1000, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=(1000, 1))
        model.fit(X, y, epochs=1, verbose=0)

        # Save the model
        model_dir = Path("./hardware_optimization_output/models")
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / "sample_tf_model"
        model.save(model_path)

        return model, str(model_path)
    except ImportError:
        logger.warning("TensorFlow not available. Skipping TensorFlow model creation.")
        return None, None

# Define PyTorch model class at module level
try:
    import torch
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.layers(x)

        # Add serialization methods
        def __getstate__(self):
            return self.state_dict()

        def __setstate__(self, state):
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.load_state_dict(state)

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

def create_sample_model_pytorch():
    """Create a sample PyTorch model for demonstration."""
    if not PYTORCH_AVAILABLE:
        logger.warning("PyTorch not available. Skipping PyTorch model creation.")
        return None, None

    try:
        # Create model
        model = SimpleModel()

        # Save the model
        model_dir = Path("./hardware_optimization_output/models")
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / "sample_pt_model.pt"
        torch.save(model, model_path)

        return model, str(model_path)
    except Exception as e:
        logger.error(f"Error creating PyTorch model: {str(e)}")
        return None, None

def demonstrate_gpu_optimization():
    """Demonstrate GPU optimization."""
    logger.info("Demonstrating GPU optimization")

    # Create sample models
    tf_model, tf_model_path = create_sample_model_tensorflow()
    pt_model, pt_model_path = create_sample_model_pytorch()

    # Use TensorFlow model if available, otherwise use PyTorch
    if tf_model is not None:
        model = tf_model
        model_path = tf_model_path
        framework = "tensorflow"
    elif pt_model is not None:
        model = pt_model
        model_path = pt_model_path
        framework = "pytorch"
    else:
        logger.error("No ML framework available. Please install TensorFlow or PyTorch.")
        return

    # Initialize optimizer
    optimizer = HardwareSpecificOptimizer(
        model_object=model,
        model_path=model_path,
        framework=framework,
        output_dir="./hardware_optimization_output",
        model_name="sample_model"
    )

    # Check available hardware
    logger.info("Checking available hardware")
    for hw_type, hw_info in optimizer.available_hardware.items():
        if hw_info["available"]:
            logger.info(f"{hw_type.upper()} is available: {hw_info['info']}")
        else:
            logger.info(f"{hw_type.upper()} is not available")

    # Optimize for GPU if available
    if optimizer.available_hardware["gpu"]["available"]:
        logger.info("Optimizing for GPU")

        # Create sample input
        sample_input = np.random.randn(1, 10).astype(np.float32)
        if framework == "pytorch":
            import torch
            sample_input = torch.tensor(sample_input)

        # Optimize for GPU
        gpu_results = optimizer.optimize_for_gpu(
            use_tensorrt=True,
            use_cuda_graphs=True,
            precision="fp16",
            sample_input=sample_input
        )

        # Print results
        logger.info(f"GPU optimization completed with {len(gpu_results['optimizations'])} optimizations")
        for opt in gpu_results["optimizations"]:
            logger.info(f"  - {opt['type']}")
    else:
        logger.info("GPU not available for optimization")

def demonstrate_cpu_optimization():
    """Demonstrate CPU optimization."""
    logger.info("Demonstrating CPU optimization")

    # Create sample models
    tf_model, tf_model_path = create_sample_model_tensorflow()
    pt_model, pt_model_path = create_sample_model_pytorch()

    # Use TensorFlow model if available, otherwise use PyTorch
    if tf_model is not None:
        model = tf_model
        model_path = tf_model_path
        framework = "tensorflow"
    elif pt_model is not None:
        model = pt_model
        model_path = pt_model_path
        framework = "pytorch"
    else:
        logger.error("No ML framework available. Please install TensorFlow or PyTorch.")
        return

    # Initialize optimizer
    optimizer = HardwareSpecificOptimizer(
        model_object=model,
        model_path=model_path,
        framework=framework,
        output_dir="./hardware_optimization_output",
        model_name="sample_model"
    )

    # Create sample input
    sample_input = np.random.randn(1, 10).astype(np.float32)
    if framework == "pytorch":
        import torch
        sample_input = torch.tensor(sample_input)

    # Optimize for CPU
    logger.info("Optimizing for CPU")
    cpu_results = optimizer.optimize_for_cpu(
        precision="fp32",
        use_mkl=True,
        use_onednn=True,
        num_threads=4,
        sample_input=sample_input
    )

    # Print results
    logger.info(f"CPU optimization completed with {len(cpu_results['optimizations'])} optimizations")
    for opt in cpu_results["optimizations"]:
        logger.info(f"  - {opt['type']}")

def main():
    """Main function to run all demonstrations."""
    # Create output directory
    output_dir = Path("./hardware_optimization_output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Run demonstrations
    demonstrate_gpu_optimization()
    demonstrate_cpu_optimization()

    # Note about TPU and FPGA optimizations
    logger.info("TPU optimization requires access to TPU hardware")
    logger.info("FPGA optimization requires OpenVINO and FPGA hardware")

if __name__ == "__main__":
    main()
