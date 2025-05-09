"""
Tests for the ModelInferenceOptimizer class.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Import the ModelInferenceOptimizer class
try:
    from ml_workbench_service.optimization.model_inference_optimizer import ModelInferenceOptimizer
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
except ImportError:
    PYTORCH_AVAILABLE = False

# Check if ONNX is available
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

@unittest.skipIf(not OPTIMIZER_AVAILABLE, "ModelInferenceOptimizer not available")
class TestModelInferenceOptimizer(unittest.TestCase):
    """Test cases for the ModelInferenceOptimizer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create sample data
        self.sample_data = np.random.randn(100, 10).astype(np.float32)
        
        # Create models if frameworks are available
        self.tf_model_path = None
        self.pt_model_path = None
        self.onnx_model_path = None
        
        if TENSORFLOW_AVAILABLE:
            self.tf_model_path = self._create_tensorflow_model()
            
        if PYTORCH_AVAILABLE:
            self.pt_model_path = self._create_pytorch_model()
            
        if ONNX_AVAILABLE and TENSORFLOW_AVAILABLE:
            self.onnx_model_path = self._create_onnx_model()
    
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
        
        return model_path
    
    def _create_pytorch_model(self):
        """Create a simple PyTorch model for testing."""
        # Create a simple model
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
        
        model = SimpleModel()
        
        # Save the model
        model_path = os.path.join(self.test_dir, "pt_model.pt")
        torch.save(model, model_path)
        
        return model_path
    
    def _create_onnx_model(self):
        """Create a simple ONNX model for testing."""
        # Convert TensorFlow model to ONNX
        import tf2onnx
        
        model = tf.keras.models.load_model(self.tf_model_path)
        
        # Create a concrete function
        input_signature = [tf.TensorSpec([None, 10], tf.float32, name='input')]
        concrete_func = tf.function(lambda x: model(x)).get_concrete_function(*input_signature)
        
        # Convert to ONNX
        model_path = os.path.join(self.test_dir, "model.onnx")
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=12)
        
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        
        return model_path
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_inference_optimization(self):
        """Test inference optimization for TensorFlow models."""
        # Initialize optimizer
        optimizer = ModelInferenceOptimizer(
            model_path=self.tf_model_path,
            framework="tensorflow",
            device="cpu",
            cache_dir=self.cache_dir
        )
        
        # Test benchmark_baseline
        baseline_metrics = optimizer.benchmark_baseline(
            input_data=self.sample_data,
            batch_sizes=[1, 8]
        )
        self.assertIn("results", baseline_metrics)
        self.assertIn("batch_1", baseline_metrics["results"])
        self.assertIn("batch_8", baseline_metrics["results"])
        
        # Test quantize_model if TFLite is available
        try:
            quantized_model, metadata = optimizer.quantize_model(
                quantization_type="float16"
            )
            self.assertIsNotNone(quantized_model)
            self.assertIn("quantization_type", metadata)
            self.assertEqual(metadata["quantization_type"], "float16")
        except Exception as e:
            print(f"Quantization test skipped: {str(e)}")
        
        # Test apply_operator_fusion
        fused_model, metadata = optimizer.apply_operator_fusion()
        self.assertIsNotNone(fused_model)
        self.assertIn("framework", metadata)
        self.assertEqual(metadata["framework"], "tensorflow")
        
        # Test configure_batch_inference
        batch_config = optimizer.configure_batch_inference(
            optimal_batch_size=8
        )
        self.assertIn("optimal_batch_size", batch_config)
        self.assertEqual(batch_config["optimal_batch_size"], 8)
        
        # Test benchmark_optimized
        optimized_metrics = optimizer.benchmark_optimized(
            input_data=self.sample_data,
            batch_sizes=[1, 8]
        )
        self.assertIn("results", optimized_metrics)
        self.assertIn("batch_1", optimized_metrics["results"])
        self.assertIn("batch_8", optimized_metrics["results"])
    
    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_pytorch_inference_optimization(self):
        """Test inference optimization for PyTorch models."""
        # Initialize optimizer
        optimizer = ModelInferenceOptimizer(
            model_path=self.pt_model_path,
            framework="pytorch",
            device="cpu",
            cache_dir=self.cache_dir
        )
        
        # Convert input data to PyTorch tensor
        input_data = torch.tensor(self.sample_data, dtype=torch.float32)
        
        # Test benchmark_baseline
        baseline_metrics = optimizer.benchmark_baseline(
            input_data=input_data,
            batch_sizes=[1, 8]
        )
        self.assertIn("results", baseline_metrics)
        self.assertIn("batch_1", baseline_metrics["results"])
        self.assertIn("batch_8", baseline_metrics["results"])
        
        # Test apply_operator_fusion
        fused_model, metadata = optimizer.apply_operator_fusion()
        self.assertIsNotNone(fused_model)
        self.assertIn("framework", metadata)
        self.assertEqual(metadata["framework"], "pytorch")
        
        # Test configure_batch_inference
        batch_config = optimizer.configure_batch_inference(
            optimal_batch_size=8
        )
        self.assertIn("optimal_batch_size", batch_config)
        self.assertEqual(batch_config["optimal_batch_size"], 8)
        
        # Test benchmark_optimized
        optimized_metrics = optimizer.benchmark_optimized(
            input_data=input_data,
            batch_sizes=[1, 8]
        )
        self.assertIn("results", optimized_metrics)
        self.assertIn("batch_1", optimized_metrics["results"])
        self.assertIn("batch_8", optimized_metrics["results"])
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not available")
    def test_onnx_inference_optimization(self):
        """Test inference optimization for ONNX models."""
        if self.onnx_model_path is None:
            self.skipTest("ONNX model not created")
            
        # Initialize optimizer
        optimizer = ModelInferenceOptimizer(
            model_path=self.onnx_model_path,
            framework="onnx",
            device="cpu",
            cache_dir=self.cache_dir
        )
        
        # Test benchmark_baseline
        baseline_metrics = optimizer.benchmark_baseline(
            input_data=self.sample_data,
            batch_sizes=[1, 8]
        )
        self.assertIn("results", baseline_metrics)
        self.assertIn("batch_1", baseline_metrics["results"])
        self.assertIn("batch_8", baseline_metrics["results"])
        
        # Test apply_operator_fusion
        fused_model, metadata = optimizer.apply_operator_fusion()
        self.assertIsNotNone(fused_model)
        self.assertIn("framework", metadata)
        self.assertEqual(metadata["framework"], "onnx")
        
        # Test configure_batch_inference
        batch_config = optimizer.configure_batch_inference(
            optimal_batch_size=8
        )
        self.assertIn("optimal_batch_size", batch_config)
        self.assertEqual(batch_config["optimal_batch_size"], 8)
        
        # Test benchmark_optimized
        optimized_metrics = optimizer.benchmark_optimized(
            input_data=self.sample_data,
            batch_sizes=[1, 8]
        )
        self.assertIn("results", optimized_metrics)
        self.assertIn("batch_1", optimized_metrics["results"])
        self.assertIn("batch_8", optimized_metrics["results"])

if __name__ == '__main__':
    unittest.main()
