"""
Tests for the ModelServingOptimizer class.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Import the ModelServingOptimizer class
try:
    from ml_workbench_service.optimization.model_serving_optimizer import ModelServingOptimizer
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

@unittest.skipIf(not OPTIMIZER_AVAILABLE, "ModelServingOptimizer not available")
class TestModelServingOptimizer(unittest.TestCase):
    """Test cases for the ModelServingOptimizer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = np.random.randn(100, 10).astype(np.float32)
        
        # Create models if frameworks are available
        self.tf_model_path = None
        self.pt_model_path = None
        
        if TENSORFLOW_AVAILABLE:
            self.tf_model_path = self._create_tensorflow_model()
            
        if PYTORCH_AVAILABLE:
            self.pt_model_path = self._create_pytorch_model()
    
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
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_serving_optimization(self):
        """Test serving optimization for TensorFlow models."""
        # Initialize optimizer
        optimizer = ModelServingOptimizer(
            model_path=self.tf_model_path,
            framework="tensorflow",
            serving_dir=os.path.join(self.test_dir, "serving"),
            model_name="test_model",
            version="v1"
        )
        
        # Test prepare_model_for_serving
        serving_metadata = optimizer.prepare_model_for_serving(
            optimization_level="performance",
            target_device="cpu"
        )
        
        self.assertIn("serving_path", serving_metadata)
        self.assertEqual(serving_metadata["model_name"], "test_model")
        self.assertEqual(serving_metadata["version"], "v1")
        
        # Test deploy_model
        deployment_status = optimizer.deploy_model(
            deployment_type="blue_green",
            traffic_percentage=100.0
        )
        
        self.assertEqual(deployment_status["status"], "deployed")
        self.assertEqual(deployment_status["deployment_type"], "blue_green")
        self.assertEqual(deployment_status["traffic_percentage"], 100.0)
        
        # Test monitor_serving_performance
        performance_metrics = optimizer.monitor_serving_performance(
            duration_seconds=5,
            metrics_interval_seconds=1,
            simulated_load=True,
            simulated_qps=10.0
        )
        
        self.assertIn("summary", performance_metrics)
        self.assertIn("avg_latency_ms", performance_metrics["summary"])
        self.assertIn("avg_qps", performance_metrics["summary"])
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_ab_testing(self):
        """Test A/B testing functionality."""
        # Initialize optimizer
        optimizer = ModelServingOptimizer(
            model_path=self.tf_model_path,
            framework="tensorflow",
            serving_dir=os.path.join(self.test_dir, "serving"),
            model_name="test_model",
            version="v1"
        )
        
        # Prepare model for serving
        optimizer.prepare_model_for_serving()
        
        # Set up A/B testing
        ab_config = optimizer.setup_ab_testing(
            variant_b_model_path=self.tf_model_path,  # Use same model for testing
            traffic_split=0.5,
            test_duration_hours=24
        )
        
        self.assertEqual(ab_config["model_name"], "test_model")
        self.assertEqual(ab_config["traffic_split"], 0.5)
        self.assertEqual(ab_config["status"], "configured")
        
        # Simulate A/B test results
        ab_results = optimizer.simulate_ab_test_results(
            duration_minutes=1,
            update_interval_seconds=1
        )
        
        self.assertIn("summary", ab_results)
        self.assertIn("winner", ab_results["summary"])
        self.assertIn("metrics_comparison", ab_results["summary"])
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_auto_scaling(self):
        """Test auto-scaling functionality."""
        # Initialize optimizer
        optimizer = ModelServingOptimizer(
            model_path=self.tf_model_path,
            framework="tensorflow",
            serving_dir=os.path.join(self.test_dir, "serving"),
            model_name="test_model",
            version="v1"
        )
        
        # Prepare model for serving
        optimizer.prepare_model_for_serving()
        
        # Configure auto-scaling
        scaling_config = optimizer.configure_auto_scaling(
            min_replicas=1,
            max_replicas=5,
            target_cpu_utilization=70,
            target_memory_utilization=80
        )
        
        self.assertEqual(scaling_config["model_name"], "test_model")
        self.assertEqual(scaling_config["min_replicas"], 1)
        self.assertEqual(scaling_config["max_replicas"], 5)
        
        # Simulate auto-scaling
        scaling_simulation = optimizer.simulate_auto_scaling(
            duration_minutes=1,
            update_interval_seconds=1,
            load_pattern="spike"
        )
        
        self.assertIn("summary", scaling_simulation)
        self.assertIn("avg_replicas", scaling_simulation["summary"])
        self.assertIn("scaling_events", scaling_simulation)

if __name__ == '__main__':
    unittest.main()
