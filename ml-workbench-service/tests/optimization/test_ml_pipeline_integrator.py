"""
Tests for the MLPipelineIntegrator class.
"""

import unittest
import os
import tempfile
import shutil
import json
from pathlib import Path

# Import the MLPipelineIntegrator class
try:
    """
    try class.
    
    Attributes:
        Add attributes here
    """

    from ml_workbench_service.optimization.ml_pipeline_integrator import MLPipelineIntegrator
    INTEGRATOR_AVAILABLE = True
except ImportError:
    INTEGRATOR_AVAILABLE = False

@unittest.skipIf(not INTEGRATOR_AVAILABLE, "MLPipelineIntegrator not available")
class TestMLPipelineIntegrator(unittest.TestCase):
    """Test cases for the MLPipelineIntegrator class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a sample project structure
        self.project_dir = os.path.join(self.test_dir, "sample_project")
        self._create_sample_project()
        
        # Initialize integrator
        self.integrator = MLPipelineIntegrator(
            project_root=self.project_dir,
            output_dir=self.output_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def _create_sample_project(self):
        """Create a sample project structure for testing."""
        # Create project directory
        os.makedirs(self.project_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(self.project_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "training"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "serving"), exist_ok=True)
        
        # Create sample model file
        model_file = os.path.join(self.project_dir, "models", "test_model.py")
        with open(model_file, "w") as f:
            f.write("""
import tensorflow as tf

class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        return self.dense(inputs)
""")
        
        # Create sample feature file
        feature_file = os.path.join(self.project_dir, "features", "test_features.py")
        with open(feature_file, "w") as f:
            f.write("""
import pandas as pd

def compute_features(data):
    data['feature1'] = data['value'] * 2
    data['feature2'] = data['value'].rolling(window=3).mean()
    return data
""")
        
        # Create sample training file
        training_file = os.path.join(self.project_dir, "training", "test_train.py")
        with open(training_file, "w") as f:
            f.write("""
import tensorflow as tf
from models.test_model import TestModel

def train_model(data, epochs=10):
    model = TestModel()
    model.compile(optimizer='adam', loss='mse')
    model.fit(data['features'], data['targets'], epochs=epochs)
    return model
""")
        
        # Create sample serving file
        serving_file = os.path.join(self.project_dir, "serving", "test_serve.py")
        with open(serving_file, "w") as f:
            f.write("""
import tensorflow as tf
from models.test_model import TestModel

def serve_model(model_path=None):
    if model_path:
        model = tf.keras.models.load_model(model_path)
    else:
        model = TestModel()
    return model

def predict(model, data):
    return model(data)
""")
    
    def test_discover_ml_components(self):
        """Test discovering ML components."""
        # Discover components
        discovered = self.integrator.discover_ml_components()
        
        # Verify discovery
        self.assertIn("models", discovered)
        self.assertIn("feature_pipelines", discovered)
        self.assertIn("training_pipelines", discovered)
        self.assertIn("serving_endpoints", discovered)
        
        # Check if specific components were discovered
        self.assertIn("TestModel", discovered["models"])
        self.assertIn("compute_features", discovered["feature_pipelines"])
        self.assertIn("train_model", discovered["training_pipelines"])
        self.assertIn("serve_model", discovered["serving_endpoints"])
        self.assertIn("predict", discovered["serving_endpoints"])
        
        # Check if discovery was saved
        discovery_path = os.path.join(self.output_dir, "discovered_components.json")
        self.assertTrue(os.path.exists(discovery_path))
        
        # Load discovery file
        with open(discovery_path, "r") as f:
            saved_discovery = json.load(f)
            
        # Verify saved discovery
        self.assertIn("models", saved_discovery)
        self.assertIn("TestModel", saved_discovery["models"])
    
    def test_create_automated_optimization_pipeline(self):
        """Test creating an automated optimization pipeline."""
        # Discover components first
        self.integrator.discover_ml_components()
        
        # Create pipeline
        pipeline_path = self.integrator.create_automated_optimization_pipeline(
            output_path="test_pipeline.py",
            schedule="daily"
        )
        
        # Verify pipeline was created
        self.assertTrue(os.path.exists(pipeline_path))
        
        # Check pipeline content
        with open(pipeline_path, "r") as f:
            content = f.read()
            
        # Verify key components in pipeline
        self.assertIn("MLPipelineIntegrator", content)
        self.assertIn("discover_ml_components", content)
        self.assertIn("optimize_model", content)
        self.assertIn("optimize_feature_pipeline", content)
        self.assertIn("optimize_training_pipeline", content)
        self.assertIn("optimize_serving_endpoint", content)
    
    def test_generate_optimization_report(self):
        """Test generating an optimization report."""
        # Generate report
        report = self.integrator.generate_optimization_report()
        
        # Verify report structure
        self.assertIn("timestamp", report)
        self.assertIn("summary", report)
        self.assertIn("models", report)
        self.assertIn("feature_pipelines", report)
        self.assertIn("training_pipelines", report)
        self.assertIn("serving_endpoints", report)
        
        # Verify report was saved
        report_path = os.path.join(self.output_dir, "optimization_report.json")
        self.assertTrue(os.path.exists(report_path))

if __name__ == '__main__':
    unittest.main()
