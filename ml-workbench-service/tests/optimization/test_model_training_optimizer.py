"""
Tests for the ModelTrainingOptimizer class.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Import the ModelTrainingOptimizer class
try:
    from ml_workbench_service.optimization.model_training_optimizer import ModelTrainingOptimizer
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

@unittest.skipIf(not OPTIMIZER_AVAILABLE, "ModelTrainingOptimizer not available")
class TestModelTrainingOptimizer(unittest.TestCase):
    """Test cases for the ModelTrainingOptimizer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(1000, 10).astype(np.float32)
        self.y = (np.random.randn(1000) > 0).astype(np.int32)
        
        # Create models if frameworks are available
        self.tf_model = None
        self.pt_model = None
        
        if TENSORFLOW_AVAILABLE:
            self.tf_model = self._create_tensorflow_model()
            
        if PYTORCH_AVAILABLE:
            self.pt_model = self._create_pytorch_model()
    
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
        
        return model
    
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
        
        return SimpleModel()
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_mixed_precision(self):
        """Test mixed precision configuration for TensorFlow."""
        # Initialize optimizer
        optimizer = ModelTrainingOptimizer(
            model=self.tf_model,
            framework="tensorflow",
            device="cpu",
            output_dir=self.test_dir
        )
        
        # Configure mixed precision
        config = optimizer.configure_mixed_precision(
            enabled=True,
            precision="float16"
        )
        
        # Verify configuration
        self.assertTrue(config["enabled"])
        self.assertEqual(config["precision"], "float16")
        self.assertIn("framework_specific", config)
        self.assertIn("tensorflow_policy", config["framework_specific"])
        self.assertEqual(config["framework_specific"]["tensorflow_policy"], "float16")
        
        # Verify global policy is set
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, "mixed_float16")
    
    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_pytorch_mixed_precision(self):
        """Test mixed precision configuration for PyTorch."""
        # Initialize optimizer
        optimizer = ModelTrainingOptimizer(
            model=self.pt_model,
            framework="pytorch",
            device="cpu",
            output_dir=self.test_dir
        )
        
        # Configure mixed precision
        config = optimizer.configure_mixed_precision(
            enabled=True,
            precision="float16"
        )
        
        # Verify configuration
        self.assertTrue(config["enabled"])
        self.assertEqual(config["precision"], "float16")
        self.assertIn("framework_specific", config)
        self.assertIn("pytorch_scaler", config["framework_specific"])
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation configuration."""
        # Initialize optimizer
        optimizer = ModelTrainingOptimizer(
            framework="tensorflow",
            device="cpu",
            output_dir=self.test_dir
        )
        
        # Configure gradient accumulation with steps
        config1 = optimizer.configure_gradient_accumulation(
            accumulation_steps=4
        )
        
        # Verify configuration
        self.assertEqual(config1["accumulation_steps"], 4)
        self.assertEqual(config1["effective_batch_multiplier"], 4)
        
        # Configure gradient accumulation with effective batch size
        config2 = optimizer.configure_gradient_accumulation(
            effective_batch_size=128,
            base_batch_size=32
        )
        
        # Verify configuration
        self.assertEqual(config2["accumulation_steps"], 4)
        self.assertEqual(config2["effective_batch_multiplier"], 4)
        self.assertEqual(config2["base_batch_size"], 32)
        self.assertEqual(config2["effective_batch_size"], 128)
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_distributed_training(self):
        """Test distributed training configuration for TensorFlow."""
        # Initialize optimizer
        optimizer = ModelTrainingOptimizer(
            model=self.tf_model,
            framework="tensorflow",
            device="cpu",
            output_dir=self.test_dir
        )
        
        # Configure distributed training
        config = optimizer.configure_distributed_training(
            strategy="mirrored"
        )
        
        # Verify configuration
        self.assertEqual(config["strategy"], "mirrored")
        self.assertEqual(config["framework"], "tensorflow")
        self.assertIn("tf_strategy", config)
    
    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_pytorch_distributed_training(self):
        """Test distributed training configuration for PyTorch."""
        # Initialize optimizer
        optimizer = ModelTrainingOptimizer(
            model=self.pt_model,
            framework="pytorch",
            device="cpu",
            output_dir=self.test_dir
        )
        
        # Configure distributed training
        config = optimizer.configure_distributed_training(
            strategy="data_parallel"
        )
        
        # Verify configuration
        self.assertEqual(config["strategy"], "data_parallel")
        self.assertEqual(config["framework"], "pytorch")
        self.assertIn("backend", config)
        self.assertIn("distributed_type", config)
        self.assertEqual(config["distributed_type"], "DataParallel")
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_benchmark_training(self):
        """Test training benchmarking for TensorFlow."""
        # Initialize optimizer
        optimizer = ModelTrainingOptimizer(
            model=self.tf_model,
            framework="tensorflow",
            device="cpu",
            output_dir=self.test_dir
        )
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        dataset = dataset.batch(32)
        
        # Benchmark training
        results = optimizer.benchmark_training(
            train_dataset=dataset,
            batch_size=32,
            num_epochs=1,
            steps_per_epoch=10
        )
        
        # Verify results
        self.assertIn("total_time_seconds", results)
        self.assertIn("samples_per_second", results)
        self.assertIn("framework", results)
        self.assertEqual(results["framework"], "tensorflow")
        self.assertIn("device", results)
        self.assertIn("framework_metrics", results)
        self.assertIn("avg_batch_time", results["framework_metrics"])
        self.assertIn("avg_epoch_time", results["framework_metrics"])
    
    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_pytorch_benchmark_training(self):
        """Test training benchmarking for PyTorch."""
        # Initialize optimizer
        optimizer = ModelTrainingOptimizer(
            model=self.pt_model,
            framework="pytorch",
            device="cpu",
            output_dir=self.test_dir
        )
        
        # Create PyTorch dataset
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        # Benchmark training
        results = optimizer.benchmark_training(
            train_dataset=dataset,
            batch_size=32,
            num_epochs=1,
            steps_per_epoch=10
        )
        
        # Verify results
        self.assertIn("total_time_seconds", results)
        self.assertIn("samples_per_second", results)
        self.assertIn("framework", results)
        self.assertEqual(results["framework"], "pytorch")
        self.assertIn("device", results)
        self.assertIn("framework_metrics", results)
        self.assertIn("avg_batch_time", results["framework_metrics"])
        self.assertIn("avg_epoch_time", results["framework_metrics"])
        self.assertIn("final_loss", results["framework_metrics"])

if __name__ == '__main__':
    unittest.main()
