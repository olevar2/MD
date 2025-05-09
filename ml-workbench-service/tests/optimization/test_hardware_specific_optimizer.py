"""
Tests for the HardwareSpecificOptimizer class.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Import the HardwareSpecificOptimizer class
try:
    from ml_workbench_service.optimization.hardware_specific_optimizer import HardwareSpecificOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Check if PyTorch is available
try:
    import torch
    PYTORCH_AVAILABLE = True

    # Define PyTorch model class at module level
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid()
            )

        def forward(self, x):
            return self.layers(x)

        # Add serialization methods
        def __getstate__(self):
            return self.state_dict()

        def __setstate__(self, state):
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid()
            )
            self.load_state_dict(state)
except ImportError:
    PYTORCH_AVAILABLE = False

@unittest.skipIf(not OPTIMIZER_AVAILABLE, "HardwareSpecificOptimizer not available")
class TestHardwareSpecificOptimizer(unittest.TestCase):
    """Test cases for the HardwareSpecificOptimizer class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Create sample data
        self.sample_data = np.random.randn(100, 10).astype(np.float32)

        # Create models if frameworks are available
        self.tf_model = None
        self.tf_model_path = None
        self.pt_model = None
        self.pt_model_path = None

        if TENSORFLOW_AVAILABLE:
            self.tf_model, self.tf_model_path = self._create_tensorflow_model()

        if PYTORCH_AVAILABLE:
            self.pt_model, self.pt_model_path = self._create_pytorch_model()

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def _create_tensorflow_model(self):
        """Create a simple TensorFlow model for testing."""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train the model briefly
        y = np.random.randint(0, 2, size=(100, 1))
        model.fit(self.sample_data, y, epochs=1, verbose=0)

        # Save the model
        model_path = os.path.join(self.test_dir, "tf_model")
        model.save(model_path)

        return model, model_path

    def _create_pytorch_model(self):
        """Create a simple PyTorch model for testing."""
        # Create a simple model
        model = SimpleModel()

        # Save the model
        model_path = os.path.join(self.test_dir, "pt_model.pt")
        torch.save(model, model_path)

        return model, model_path

    def test_hardware_detection(self):
        """Test hardware detection."""
        # Initialize optimizer with TensorFlow model if available
        if TENSORFLOW_AVAILABLE:
            optimizer = HardwareSpecificOptimizer(
                model_object=self.tf_model,
                model_path=self.tf_model_path,
                framework="tensorflow",
                output_dir=self.test_dir
            )
        elif PYTORCH_AVAILABLE:
            optimizer = HardwareSpecificOptimizer(
                model_object=self.pt_model,
                model_path=self.pt_model_path,
                framework="pytorch",
                output_dir=self.test_dir
            )
        else:
            self.skipTest("No ML framework available")

        # Verify hardware detection
        self.assertIn("cpu", optimizer.available_hardware)
        self.assertIn("gpu", optimizer.available_hardware)
        self.assertIn("tpu", optimizer.available_hardware)
        self.assertIn("fpga", optimizer.available_hardware)

        # CPU should always be available
        self.assertTrue(optimizer.available_hardware["cpu"]["available"])
        self.assertIn("cores", optimizer.available_hardware["cpu"]["info"])

    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_cpu_optimization(self):
        """Test CPU optimization for TensorFlow models."""
        # Initialize optimizer
        optimizer = HardwareSpecificOptimizer(
            model_object=self.tf_model,
            model_path=self.tf_model_path,
            framework="tensorflow",
            output_dir=self.test_dir
        )

        # Optimize for CPU
        results = optimizer.optimize_for_cpu(
            precision="fp32",
            use_mkl=True,
            use_onednn=True,
            num_threads=2,
            sample_input=self.sample_data
        )

        # Verify results
        self.assertEqual(results["model_name"], "model")
        self.assertEqual(results["framework"], "tensorflow")
        self.assertEqual(results["target_hardware"], "cpu")
        self.assertEqual(results["precision"], "fp32")
        self.assertIn("optimizations", results)

        # Verify optimization results were saved
        self.assertIn("cpu", optimizer.optimization_results)

    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_pytorch_cpu_optimization(self):
        """Test CPU optimization for PyTorch models."""
        # Initialize optimizer
        optimizer = HardwareSpecificOptimizer(
            model_object=self.pt_model,
            model_path=self.pt_model_path,
            framework="pytorch",
            output_dir=self.test_dir
        )

        # Create sample input
        sample_input = torch.tensor(self.sample_data)

        # Optimize for CPU
        results = optimizer.optimize_for_cpu(
            precision="fp32",
            use_mkl=True,
            use_onednn=True,
            num_threads=2,
            sample_input=sample_input
        )

        # Verify results
        self.assertEqual(results["model_name"], "model")
        self.assertEqual(results["framework"], "pytorch")
        self.assertEqual(results["target_hardware"], "cpu")
        self.assertEqual(results["precision"], "fp32")
        self.assertIn("optimizations", results)

        # Verify optimization results were saved
        self.assertIn("cpu", optimizer.optimization_results)

    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_gpu_optimization(self):
        """Test GPU optimization for TensorFlow models."""
        # Initialize optimizer
        optimizer = HardwareSpecificOptimizer(
            model_object=self.tf_model,
            model_path=self.tf_model_path,
            framework="tensorflow",
            output_dir=self.test_dir
        )

        # Skip test if GPU is not available
        if not optimizer.available_hardware["gpu"]["available"]:
            self.skipTest("GPU not available")

        # Optimize for GPU
        results = optimizer.optimize_for_gpu(
            use_tensorrt=True,
            use_cuda_graphs=True,
            precision="fp16",
            sample_input=self.sample_data
        )

        # Verify results
        self.assertEqual(results["model_name"], "model")
        self.assertEqual(results["framework"], "tensorflow")
        self.assertEqual(results["target_hardware"], "gpu")
        self.assertEqual(results["precision"], "fp16")
        self.assertIn("optimizations", results)

        # Verify optimization results were saved
        self.assertIn("gpu", optimizer.optimization_results)

    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_pytorch_gpu_optimization(self):
        """Test GPU optimization for PyTorch models."""
        # Initialize optimizer
        optimizer = HardwareSpecificOptimizer(
            model_object=self.pt_model,
            model_path=self.pt_model_path,
            framework="pytorch",
            output_dir=self.test_dir
        )

        # Skip test if GPU is not available
        if not optimizer.available_hardware["gpu"]["available"]:
            self.skipTest("GPU not available")

        # Create sample input
        sample_input = torch.tensor(self.sample_data)

        # Optimize for GPU
        results = optimizer.optimize_for_gpu(
            use_tensorrt=True,
            use_cuda_graphs=True,
            precision="fp16",
            sample_input=sample_input
        )

        # Verify results
        self.assertEqual(results["model_name"], "model")
        self.assertEqual(results["framework"], "pytorch")
        self.assertEqual(results["target_hardware"], "gpu")
        self.assertEqual(results["precision"], "fp16")
        self.assertIn("optimizations", results)

        # Verify optimization results were saved
        self.assertIn("gpu", optimizer.optimization_results)

    def test_convert_to_onnx(self):
        """Test conversion to ONNX format."""
        # Skip test if no framework is available
        if not TENSORFLOW_AVAILABLE and not PYTORCH_AVAILABLE:
            self.skipTest("No ML framework available")

        # Skip test if ONNX is not available
        try:
            import onnx
            import onnxruntime
        except ImportError:
            self.skipTest("ONNX or ONNX Runtime not available")

        # Initialize optimizer with TensorFlow model if available
        if TENSORFLOW_AVAILABLE:
            try:
                import tf2onnx
            except ImportError:
                self.skipTest("tf2onnx not available")

            optimizer = HardwareSpecificOptimizer(
                model_object=self.tf_model,
                model_path=self.tf_model_path,
                framework="tensorflow",
                output_dir=self.test_dir
            )
            sample_input = self.sample_data
        else:
            optimizer = HardwareSpecificOptimizer(
                model_object=self.pt_model,
                model_path=self.pt_model_path,
                framework="pytorch",
                output_dir=self.test_dir
            )
            sample_input = torch.tensor(self.sample_data)

        # Convert to ONNX
        onnx_path = optimizer._convert_to_onnx(sample_input)

        # Verify conversion
        self.assertIsNotNone(onnx_path)
        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(str(onnx_path).endswith(".onnx"))

if __name__ == '__main__':
    unittest.main()
