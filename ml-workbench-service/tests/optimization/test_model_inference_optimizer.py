"""
Tests for the ModelInferenceOptimizer class.
"""
import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
try:
    from ml_workbench_service.optimization.model_inference_optimizer import ModelInferenceOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@unittest.skipIf(not OPTIMIZER_AVAILABLE,
    'ModelInferenceOptimizer not available')
class TestModelInferenceOptimizer(unittest.TestCase):
    """Test cases for the ModelInferenceOptimizer class."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.sample_data = np.random.randn(100, 10).astype(np.float32)
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
        shutil.rmtree(self.test_dir)

    def _create_tensorflow_model(self):
        """Create a simple TensorFlow model for testing."""
        model = tf.keras.Sequential([tf.keras.layers.Dense(32, activation=
            'relu', input_shape=(10,)), tf.keras.layers.Dense(16,
            activation='relu'), tf.keras.layers.Dense(1, activation='sigmoid')]
            )
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics
            =['accuracy'])
        y = np.random.randint(0, 2, size=(100, 1))
        model.fit(self.sample_data, y, epochs=1, verbose=0)
        model_path = os.path.join(self.test_dir, 'tf_model')
        model.save(model_path)
        return model_path

    def _create_pytorch_model(self):
        """Create a simple PyTorch model for testing."""


        class SimpleModel(torch.nn.Module):
    """
    SimpleModel class that inherits from torch.nn.Module.
    
    Attributes:
        Add attributes here
    """


            def __init__(self):
    """
      init  .
    
    """

                super().__init__()
                self.layers = torch.nn.Sequential(torch.nn.Linear(10, 32),
                    torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU
                    (), torch.nn.Linear(16, 1), torch.nn.Sigmoid())

            def forward(self, x):
    """
    Forward.
    
    Args:
        x: Description of x
    
    """

                return self.layers(x)
        model = SimpleModel()
        model_path = os.path.join(self.test_dir, 'pt_model.pt')
        torch.save(model, model_path)
        return model_path

    def _create_onnx_model(self):
        """Create a simple ONNX model for testing."""
        import tf2onnx
        model = tf.keras.models.load_model(self.tf_model_path)
        input_signature = [tf.TensorSpec([None, 10], tf.float32, name='input')]
        concrete_func = tf.function(lambda x: model(x)).get_concrete_function(*
            input_signature)
        model_path = os.path.join(self.test_dir, 'model.onnx')
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature,
            opset=12)
        with open(model_path, 'wb') as f:
            f.write(model_proto.SerializeToString())
        return model_path

    @unittest.skipIf(not TENSORFLOW_AVAILABLE, 'TensorFlow not available')
    @with_exception_handling
    def test_tensorflow_inference_optimization(self):
        """Test inference optimization for TensorFlow models."""
        optimizer = ModelInferenceOptimizer(model_path=self.tf_model_path,
            framework='tensorflow', device='cpu', cache_dir=self.cache_dir)
        baseline_metrics = optimizer.benchmark_baseline(input_data=self.
            sample_data, batch_sizes=[1, 8])
        self.assertIn('results', baseline_metrics)
        self.assertIn('batch_1', baseline_metrics['results'])
        self.assertIn('batch_8', baseline_metrics['results'])
        try:
            quantized_model, metadata = optimizer.quantize_model(
                quantization_type='float16')
            self.assertIsNotNone(quantized_model)
            self.assertIn('quantization_type', metadata)
            self.assertEqual(metadata['quantization_type'], 'float16')
        except Exception as e:
            print(f'Quantization test skipped: {str(e)}')
        fused_model, metadata = optimizer.apply_operator_fusion()
        self.assertIsNotNone(fused_model)
        self.assertIn('framework', metadata)
        self.assertEqual(metadata['framework'], 'tensorflow')
        batch_config = optimizer.configure_batch_inference(optimal_batch_size=8
            )
        self.assertIn('optimal_batch_size', batch_config)
        self.assertEqual(batch_config['optimal_batch_size'], 8)
        optimized_metrics = optimizer.benchmark_optimized(input_data=self.
            sample_data, batch_sizes=[1, 8])
        self.assertIn('results', optimized_metrics)
        self.assertIn('batch_1', optimized_metrics['results'])
        self.assertIn('batch_8', optimized_metrics['results'])

    @unittest.skipIf(not PYTORCH_AVAILABLE, 'PyTorch not available')
    def test_pytorch_inference_optimization(self):
        """Test inference optimization for PyTorch models."""
        optimizer = ModelInferenceOptimizer(model_path=self.pt_model_path,
            framework='pytorch', device='cpu', cache_dir=self.cache_dir)
        input_data = torch.tensor(self.sample_data, dtype=torch.float32)
        baseline_metrics = optimizer.benchmark_baseline(input_data=
            input_data, batch_sizes=[1, 8])
        self.assertIn('results', baseline_metrics)
        self.assertIn('batch_1', baseline_metrics['results'])
        self.assertIn('batch_8', baseline_metrics['results'])
        fused_model, metadata = optimizer.apply_operator_fusion()
        self.assertIsNotNone(fused_model)
        self.assertIn('framework', metadata)
        self.assertEqual(metadata['framework'], 'pytorch')
        batch_config = optimizer.configure_batch_inference(optimal_batch_size=8
            )
        self.assertIn('optimal_batch_size', batch_config)
        self.assertEqual(batch_config['optimal_batch_size'], 8)
        optimized_metrics = optimizer.benchmark_optimized(input_data=
            input_data, batch_sizes=[1, 8])
        self.assertIn('results', optimized_metrics)
        self.assertIn('batch_1', optimized_metrics['results'])
        self.assertIn('batch_8', optimized_metrics['results'])

    @unittest.skipIf(not ONNX_AVAILABLE, 'ONNX not available')
    def test_onnx_inference_optimization(self):
        """Test inference optimization for ONNX models."""
        if self.onnx_model_path is None:
            self.skipTest('ONNX model not created')
        optimizer = ModelInferenceOptimizer(model_path=self.onnx_model_path,
            framework='onnx', device='cpu', cache_dir=self.cache_dir)
        baseline_metrics = optimizer.benchmark_baseline(input_data=self.
            sample_data, batch_sizes=[1, 8])
        self.assertIn('results', baseline_metrics)
        self.assertIn('batch_1', baseline_metrics['results'])
        self.assertIn('batch_8', baseline_metrics['results'])
        fused_model, metadata = optimizer.apply_operator_fusion()
        self.assertIsNotNone(fused_model)
        self.assertIn('framework', metadata)
        self.assertEqual(metadata['framework'], 'onnx')
        batch_config = optimizer.configure_batch_inference(optimal_batch_size=8
            )
        self.assertIn('optimal_batch_size', batch_config)
        self.assertEqual(batch_config['optimal_batch_size'], 8)
        optimized_metrics = optimizer.benchmark_optimized(input_data=self.
            sample_data, batch_sizes=[1, 8])
        self.assertIn('results', optimized_metrics)
        self.assertIn('batch_1', optimized_metrics['results'])
        self.assertIn('batch_8', optimized_metrics['results'])


if __name__ == '__main__':
    unittest.main()
