"""
Tests for the MLProfilingMonitor class.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import json
from pathlib import Path

# Import the MLProfilingMonitor class
try:
    """
    try class.
    
    Attributes:
        Add attributes here
    """

    from core.ml_profiling_monitor import MLProfilingMonitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

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

@unittest.skipIf(not MONITOR_AVAILABLE, "MLProfilingMonitor not available")
class TestMLProfilingMonitor(unittest.TestCase):
    """Test cases for the MLProfilingMonitor class."""
    
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
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                    torch.nn.Sigmoid()
                )
                
            def forward(self, x):
    """
    Forward.
    
    Args:
        x: Description of x
    
    """

                return self.layers(x)
        
        model = SimpleModel()
        
        # Save the model
        model_path = os.path.join(self.test_dir, "pt_model.pt")
        torch.save(model, model_path)
        
        return model, model_path
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_profiling(self):
        """Test profiling for TensorFlow models."""
        # Initialize monitor
        monitor = MLProfilingMonitor(
            model_object=self.tf_model,
            model_path=self.tf_model_path,
            framework="tensorflow",
            output_dir=self.test_dir,
            model_name="test_model"
        )
        
        # Profile model
        results = monitor.profile_model(
            input_data=self.sample_data,
            batch_sizes=[1, 8],
            warmup_runs=2,
            profile_runs=5,
            profile_memory=True,
            profile_cpu=True,
            export_trace=True
        )
        
        # Verify results
        self.assertIn("summary", results)
        self.assertIn("optimal_batch_size", results["summary"])
        self.assertIn("max_throughput", results["summary"])
        self.assertIn("min_latency", results["summary"])
        
        self.assertIn("batch_sizes", results)
        self.assertIn("1", results["batch_sizes"])
        self.assertIn("8", results["batch_sizes"])
        
        # Verify batch results
        for batch_size in ["1", "8"]:
            batch_results = results["batch_sizes"][batch_size]
            self.assertIn("latency_ms", batch_results)
            self.assertIn("throughput", batch_results)
            
            # Verify latency metrics
            self.assertIn("mean", batch_results["latency_ms"])
            self.assertIn("p95", batch_results["latency_ms"])
            
            # Verify throughput metrics
            self.assertIn("samples_per_second", batch_results["throughput"])
            self.assertIn("batches_per_second", batch_results["throughput"])
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_tensorflow_inference_recording(self):
        """Test inference recording for TensorFlow models."""
        # Initialize monitor
        monitor = MLProfilingMonitor(
            model_object=self.tf_model,
            model_path=self.tf_model_path,
            framework="tensorflow",
            output_dir=self.test_dir,
            model_name="test_model"
        )
        
        # Record inference
        result, latency = monitor.record_inference(
            input_data=self.sample_data,
            batch_size=None,
            record_metrics=True
        )
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertGreater(latency, 0)
        
        # Verify metrics
        self.assertEqual(monitor.metrics["inference_count"], 1)
        self.assertEqual(len(monitor.metrics["inference_latency_ms"]), 1)
        self.assertEqual(len(monitor.metrics["batch_sizes"]), 1)
        self.assertEqual(len(monitor.metrics["timestamp"]), 1)
        self.assertEqual(monitor.metrics["error_count"], 0)
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_dashboard_generation(self):
        """Test Grafana dashboard generation."""
        # Initialize monitor
        monitor = MLProfilingMonitor(
            model_object=self.tf_model,
            model_path=self.tf_model_path,
            framework="tensorflow",
            output_dir=self.test_dir,
            model_name="test_model"
        )
        
        # Generate dashboard
        dashboard_path = os.path.join(self.test_dir, "dashboard.json")
        dashboard = monitor.generate_grafana_dashboard(
            dashboard_title="Test Dashboard",
            prometheus_datasource="Prometheus",
            dashboard_path=dashboard_path
        )
        
        # Verify dashboard
        self.assertIn("panels", dashboard)
        self.assertGreater(len(dashboard["panels"]), 0)
        
        # Verify dashboard file
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Load dashboard file
        with open(dashboard_path, "r") as f:
            loaded_dashboard = json.load(f)
            
        # Verify loaded dashboard
        self.assertEqual(loaded_dashboard["title"], "Test Dashboard")
        self.assertIn("panels", loaded_dashboard)
        self.assertGreater(len(loaded_dashboard["panels"]), 0)
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_alert_configuration(self):
        """Test alert configuration."""
        # Initialize monitor
        monitor = MLProfilingMonitor(
            model_object=self.tf_model,
            model_path=self.tf_model_path,
            framework="tensorflow",
            output_dir=self.test_dir,
            model_name="test_model"
        )
        
        # Configure alerts
        alert_path = os.path.join(self.test_dir, "alerts.json")
        alerts = monitor.configure_alerts(
            latency_threshold_ms=50.0,
            error_rate_threshold=0.01,
            memory_threshold_mb=500.0,
            cpu_threshold_percent=70.0,
            alert_config_path=alert_path
        )
        
        # Verify alerts
        self.assertIn("groups", alerts)
        self.assertGreater(len(alerts["groups"]), 0)
        self.assertGreater(len(alerts["groups"][0]["rules"]), 0)
        
        # Verify alert file
        self.assertTrue(os.path.exists(alert_path))
        
        # Load alert file
        with open(alert_path, "r") as f:
            loaded_alerts = json.load(f)
            
        # Verify loaded alerts
        self.assertIn("groups", loaded_alerts)
        self.assertGreater(len(loaded_alerts["groups"]), 0)
        self.assertGreater(len(loaded_alerts["groups"][0]["rules"]), 0)

if __name__ == '__main__':
    unittest.main()
