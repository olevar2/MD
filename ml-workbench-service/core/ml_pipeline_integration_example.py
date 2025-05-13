"""
ML Pipeline Integration Example

This script demonstrates how to use the ML Pipeline Integrator to discover
and optimize ML components in the forex trading platform.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add parent directory to path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_workbench_service.optimization import (
    ModelInferenceOptimizer,
    FeatureEngineeringOptimizer,
    ModelTrainingOptimizer,
    ModelServingOptimizer,
    MLPipelineIntegrator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_project():
    """Create a sample project structure with ML components for demonstration."""
    logger.info("Creating sample project structure")
    
    # Create project directory
    project_dir = Path("./sample_forex_project")
    project_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    (project_dir / "models").mkdir(exist_ok=True)
    (project_dir / "features").mkdir(exist_ok=True)
    (project_dir / "training").mkdir(exist_ok=True)
    (project_dir / "serving").mkdir(exist_ok=True)
    
    # Create sample model file
    model_file = project_dir / "models" / "price_predictor_model.py"
    with open(model_file, "w") as f:
        f.write("""
import tensorflow as tf

class PricePredictorModel(tf.keras.Model):
    """
    PricePredictorModel class that inherits from tf.keras.Model.
    
    Attributes:
        Add attributes here
    """

    def __init__(self):
    """
      init  .
    
    """

        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
        
def create_model():
    return PricePredictorModel()
""")
    
    # Create sample feature pipeline file
    feature_file = project_dir / "features" / "price_features.py"
    with open(feature_file, "w") as f:
        f.write("""
import pandas as pd
import numpy as np

def compute_technical_features(data):
    """
    Compute technical features.
    
    Args:
        data: Description of data
    
    """

    # Calculate moving averages
    data['ma_5'] = data['close'].rolling(window=5).mean()
    data['ma_20'] = data['close'].rolling(window=20).mean()
    
    # Calculate RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    return data
""")
    
    # Create sample training pipeline file
    training_file = project_dir / "training" / "train_model.py"
    with open(training_file, "w") as f:
        f.write("""
import tensorflow as tf
import numpy as np
import pandas as pd
from models.price_predictor_model import create_model
from features.price_features import compute_technical_features

def train_price_predictor(data, epochs=10, batch_size=32):
    """
    Train price predictor.
    
    Args:
        data: Description of data
        epochs: Description of epochs
        batch_size: Description of batch_size
    
    """

    # Prepare features
    data = compute_technical_features(data)
    data = data.dropna()
    
    # Create features and targets
    features = data[['ma_5', 'ma_20', 'rsi', 'macd', 'macd_signal']].values
    targets = data['close'].shift(-1).values[:-1]
    features = features[:-1]
    
    # Split data
    split_idx = int(len(features) * 0.8)
    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]
    
    # Create and train model
    model = create_model()
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(
        train_features, train_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_features, val_targets)
    )
    
    return model, history
""")
    
    # Create sample serving file
    serving_file = project_dir / "serving" / "model_server.py"
    with open(serving_file, "w") as f:
        f.write("""
import tensorflow as tf
import numpy as np
import pandas as pd
from models.price_predictor_model import create_model
from features.price_features import compute_technical_features

class PricePredictorService:
    """
    PricePredictorService class.
    
    Attributes:
        Add attributes here
    """

    def __init__(self, model_path=None):
    """
      init  .
    
    Args:
        model_path: Description of model_path
    
    """

        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = create_model()
            
    def predict(self, data):
    """
    Predict.
    
    Args:
        data: Description of data
    
    """

        # Prepare features
        data = compute_technical_features(data)
        data = data.dropna()
        
        # Extract features
        features = data[['ma_5', 'ma_20', 'rsi', 'macd', 'macd_signal']].values
        
        # Make predictions
        predictions = self.model.predict(features)
        
        return predictions
        
def serve_model(model_path=None):
    service = PricePredictorService(model_path)
    return service
""")
    
    logger.info(f"Sample project created at {project_dir}")
    return project_dir

def demonstrate_pipeline_integration():
    """Demonstrate ML pipeline integration."""
    logger.info("Demonstrating ML pipeline integration")
    
    # Create sample project
    project_dir = create_sample_project()
    
    # Initialize pipeline integrator
    integrator = MLPipelineIntegrator(
        project_root=str(project_dir),
        output_dir="./pipeline_optimization_output"
    )
    
    # Discover ML components
    logger.info("Discovering ML components")
    discovered = integrator.discover_ml_components()
    
    # Print discovered components
    for component_type, components in discovered.items():
        logger.info(f"Discovered {len(components)} {component_type}:")
        for name, details in components.items():
            logger.info(f"  - {name} ({details['framework']})")
    
    # Create automated optimization pipeline
    logger.info("Creating automated optimization pipeline")
    pipeline_path = integrator.create_automated_optimization_pipeline(
        output_path="automated_optimization_pipeline.py",
        schedule="daily"
    )
    
    logger.info(f"Automated optimization pipeline created at {pipeline_path}")
    
    # Generate optimization report
    logger.info("Generating optimization report")
    report = integrator.generate_optimization_report()
    
    logger.info("ML pipeline integration demonstration completed")

if __name__ == "__main__":
    demonstrate_pipeline_integration()
