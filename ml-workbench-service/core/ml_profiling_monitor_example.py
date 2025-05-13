"""
ML Profiling and Monitoring Example

This script demonstrates how to use the ML Profiling Monitor to profile
and monitor ML models in the forex trading platform.
"""
import os
import sys
import logging
import numpy as np
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_workbench_service.optimization import ModelInferenceOptimizer, ModelTrainingOptimizer, ModelServingOptimizer
from core.ml_profiling_monitor import MLProfilingMonitor
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def create_sample_model_tensorflow():
    """Create a sample TensorFlow model for demonstration."""
    try:
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation=
            'relu', input_shape=(10,)), tf.keras.layers.Dense(32,
            activation='relu'), tf.keras.layers.Dense(1, activation='sigmoid')]
            )
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics
            =['accuracy'])
        X = np.random.randn(1000, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=(1000, 1))
        model.fit(X, y, epochs=1, verbose=0)
        model_dir = Path('./profiling_output/models')
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / 'sample_tf_model'
        model.save(model_path)
        return model, str(model_path)
    except ImportError:
        logger.warning(
            'TensorFlow not available. Skipping TensorFlow model creation.')
        return None, None


try:
    import torch
    import torch.nn as nn


    class SimpleModel(nn.Module):
    """
    SimpleModel class that inherits from nn.Module.
    
    Attributes:
        Add attributes here
    """


        def __init__(self):
    """
      init  .
    
    """

            super().__init__()
            self.layers = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.
                Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        def forward(self, x):
    """
    Forward.
    
    Args:
        x: Description of x
    
    """

            return self.layers(x)

        def __getstate__(self):
    """
      getstate  .
    
    """

            return self.state_dict()

        def __setstate__(self, state):
    """
      setstate  .
    
    Args:
        state: Description of state
    
    """

            self.layers = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.
                Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            self.load_state_dict(state)
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


@with_exception_handling
def create_sample_model_pytorch():
    """Create a sample PyTorch model for demonstration."""
    if not PYTORCH_AVAILABLE:
        logger.warning(
            'PyTorch not available. Skipping PyTorch model creation.')
        return None, None
    try:
        model = SimpleModel()
        model_dir = Path('./profiling_output/models')
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / 'sample_pt_model.pt'
        torch.save(model, model_path)
        return model, str(model_path)
    except Exception as e:
        logger.error(f'Error creating PyTorch model: {str(e)}')
        return None, None


def demonstrate_profiling():
    """Demonstrate model profiling."""
    logger.info('Demonstrating model profiling')
    tf_model, tf_model_path = create_sample_model_tensorflow()
    pt_model, pt_model_path = create_sample_model_pytorch()
    if tf_model is not None:
        model = tf_model
        model_path = tf_model_path
        framework = 'tensorflow'
    elif pt_model is not None:
        model = pt_model
        model_path = pt_model_path
        framework = 'pytorch'
    else:
        logger.error(
            'No ML framework available. Please install TensorFlow or PyTorch.')
        return
    monitor = MLProfilingMonitor(model_object=model, model_path=model_path,
        framework=framework, output_dir='./profiling_output', model_name=
        'sample_model', prometheus_port=8000)
    input_data = np.random.randn(100, 10).astype(np.float32)
    if framework == 'pytorch':
        import torch
        input_data = torch.tensor(input_data)
    logger.info('Profiling model')
    profiling_results = monitor.profile_model(input_data=input_data,
        batch_sizes=[1, 8, 32, 64], warmup_runs=5, profile_runs=20,
        profile_memory=True, profile_cpu=True, export_trace=True)
    logger.info(
        f"Optimal batch size: {profiling_results['summary']['optimal_batch_size']}"
        )
    logger.info(
        f"Max throughput: {profiling_results['summary']['max_throughput']:.2f} samples/second"
        )
    logger.info(
        f"Min latency: {profiling_results['summary']['min_latency']:.2f} ms")
    logger.info('Generating Grafana dashboard')
    dashboard = monitor.generate_grafana_dashboard(dashboard_title=
        'Sample Model Performance', prometheus_datasource='Prometheus',
        dashboard_path='./profiling_output/sample_model_dashboard.json')
    logger.info('Configuring alerts')
    alerts = monitor.configure_alerts(latency_threshold_ms=50.0,
        error_rate_threshold=0.01, memory_threshold_mb=500.0,
        cpu_threshold_percent=70.0, alert_config_path=
        './profiling_output/sample_model_alerts.json')
    logger.info('Model profiling demonstration completed')


@with_exception_handling
def demonstrate_monitoring():
    """Demonstrate model monitoring."""
    logger.info('Demonstrating model monitoring')
    tf_model, tf_model_path = create_sample_model_tensorflow()
    pt_model, pt_model_path = create_sample_model_pytorch()
    if tf_model is not None:
        model = tf_model
        model_path = tf_model_path
        framework = 'tensorflow'
    elif pt_model is not None:
        model = pt_model
        model_path = pt_model_path
        framework = 'pytorch'
    else:
        logger.error(
            'No ML framework available. Please install TensorFlow or PyTorch.')
        return
    monitor = MLProfilingMonitor(model_object=model, model_path=model_path,
        framework=framework, output_dir='./profiling_output', model_name=
        'sample_model', prometheus_port=8000)
    try:
        logger.info('Starting Prometheus exporter')
        monitor.start_prometheus_exporter()
        input_data = np.random.randn(32, 10).astype(np.float32)
        if framework == 'pytorch':
            import torch
            input_data = torch.tensor(input_data)
        logger.info('Running inference and recording metrics')
        for i in range(20):
            result, latency = monitor.record_inference(input_data=
                input_data, batch_size=32, record_metrics=True)
            logger.info(f'Inference {i + 1}/20: Latency = {latency:.2f} ms')
            time.sleep(0.5)
        logger.info(f"Total inferences: {monitor.metrics['inference_count']}")
        logger.info(
            f"Average latency: {np.mean(monitor.metrics['inference_latency_ms']):.2f} ms"
            )
        logger.info(f"Error count: {monitor.metrics['error_count']}")
        logger.info(
            'Metrics are being exported to Prometheus. Press Ctrl+C to stop.')
        logger.info(
            f'Prometheus metrics available at: http://localhost:{monitor.prometheus_port}/metrics'
            )
        time.sleep(10)
    finally:
        logger.info('Stopping Prometheus exporter')
        monitor.stop_prometheus_exporter()
    logger.info('Model monitoring demonstration completed')


def main():
    """Main function to run all demonstrations."""
    output_dir = Path('./profiling_output')
    output_dir.mkdir(exist_ok=True, parents=True)
    demonstrate_profiling()
    demonstrate_monitoring()


if __name__ == '__main__':
    main()
