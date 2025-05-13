"""
Model Inference Optimizer

This module provides tools for optimizing ML model inference performance through
techniques like quantization, operator fusion, and batch inference.

It includes:
- Model quantization for reduced memory footprint and faster inference
- Operator fusion for optimizing computational graphs
- Batch inference for improved throughput
- Performance benchmarking tools
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import json
import uuid
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
logger = logging.getLogger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelInferenceOptimizer:
    """
    Optimizes ML models for inference performance.

    This class provides methods for:
    - Quantizing models to reduce memory and improve speed
    - Applying operator fusion to optimize computational graphs
    - Configuring batch inference for improved throughput
    - Benchmarking performance before and after optimization
    """

    def __init__(self, model_path: Optional[str]=None, model_object:
        Optional[Any]=None, framework: str='tensorflow', optimization_level:
        str='balanced', device: str='cpu', cache_dir: str=
        './optimization_cache'):
        """
        Initialize the model inference optimizer.

        Args:
            model_path: Path to the model file
            model_object: Model object (alternative to model_path)
            framework: ML framework the model belongs to
            optimization_level: Optimization priority (speed, balanced, memory)
            device: Target device for optimized model
            cache_dir: Directory to cache optimized models
        """
        self.model_path = model_path
        self.model_object = model_object
        self.framework = framework.lower()
        self.optimization_level = optimization_level
        self.device = device.lower()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._validate_framework()
        if model_path and not model_object:
            self.model_object = self._load_model(model_path)
        self.baseline_metrics = {}
        self.optimized_metrics = {}

    def _validate_framework(self):
        """Validate that the requested framework is available."""
        if self.framework == 'tensorflow' and not TENSORFLOW_AVAILABLE:
            raise ImportError(
                'TensorFlow is not available. Please install it to use this framework.'
                )
        elif self.framework == 'pytorch' and not PYTORCH_AVAILABLE:
            raise ImportError(
                'PyTorch is not available. Please install it to use this framework.'
                )
        elif self.framework == 'onnx' and not ONNX_AVAILABLE:
            raise ImportError(
                'ONNX Runtime is not available. Please install it to use this framework.'
                )

    @with_exception_handling
    def _load_model(self, model_path: str) ->Any:
        """Load a model from the specified path."""
        logger.info(f'Loading {self.framework} model from {model_path}')
        try:
            if self.framework == 'tensorflow':
                return tf.saved_model.load(model_path)
            elif self.framework == 'pytorch':
                return torch.load(model_path)
            elif self.framework == 'onnx':
                return onnx.load(model_path)
            else:
                raise ValueError(f'Unsupported framework: {self.framework}')
        except Exception as e:
            logger.error(f'Error loading model: {str(e)}')
            raise

    def benchmark_baseline(self, input_data: Union[np.ndarray, pd.DataFrame,
        List], batch_sizes: List[int]=[1, 8, 16, 32, 64], num_runs: int=10,
        warmup_runs: int=3) ->Dict[str, Any]:
        """
        Benchmark the baseline model performance.

        Args:
            input_data: Sample input data for benchmarking
            batch_sizes: List of batch sizes to test
            num_runs: Number of inference runs to average over
            warmup_runs: Number of warmup runs before timing

        Returns:
            Dictionary with benchmark results
        """
        logger.info(
            f'Benchmarking baseline model performance with {num_runs} runs')
        results = {}
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        for batch_size in batch_sizes:
            batch_results = self._run_inference_benchmark(self.model_object,
                input_data, batch_size, num_runs, warmup_runs)
            results[f'batch_{batch_size}'] = batch_results
        self.baseline_metrics = {'timestamp': datetime.now().isoformat(),
            'results': results}
        return self.baseline_metrics

    def _run_inference_benchmark(self, model: Any, input_data: Any,
        batch_size: int, num_runs: int, warmup_runs: int) ->Dict[str, float]:
        """Run inference benchmark for a specific batch size."""
        if len(input_data) < batch_size:
            repeats = batch_size // len(input_data) + 1
            input_data = np.tile(input_data, (repeats, 1))[:batch_size]
        batched_input = input_data[:batch_size]
        for _ in range(warmup_runs):
            self._run_inference(model, batched_input)
        latencies = []
        start_time = time.time()
        for _ in range(num_runs):
            run_start = time.time()
            _ = self._run_inference(model, batched_input)
            latencies.append((time.time() - run_start) * 1000)
        total_time = time.time() - start_time
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = num_runs * batch_size / total_time
        return {'avg_latency_ms': avg_latency, 'p95_latency_ms':
            p95_latency, 'p99_latency_ms': p99_latency,
            'throughput_samples_per_sec': throughput, 'batch_size': batch_size}

    def _run_inference(self, model: Any, input_data: Any) ->Any:
        """Run inference with the given model and input data."""
        if self.framework == 'tensorflow':
            if hasattr(model, 'predict'):
                return model.predict(input_data)
            elif hasattr(model, '__call__'):
                return model(input_data)
            else:
                raise ValueError(
                    "TensorFlow model doesn't have predict or __call__ method")
        elif self.framework == 'pytorch':
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.tensor(input_data, dtype=torch.float32
                        )
                    if self.device == 'gpu' and torch.cuda.is_available():
                        input_tensor = input_tensor.cuda()
                    return model(input_tensor)
                else:
                    return model(input_data)
        elif self.framework == 'onnx':
            session = ort.InferenceSession(model.SerializeToString())
            input_name = session.get_inputs()[0].name
            return session.run(None, {input_name: input_data})

    @with_exception_handling
    def quantize_model(self, quantization_type: str='int8',
        calibration_data: Optional[Any]=None, optimize_for_device: bool=True
        ) ->Tuple[Any, Dict[str, Any]]:
        """
        Quantize the model to reduce size and improve inference speed.

        Args:
            quantization_type: Type of quantization to apply
            calibration_data: Representative dataset for calibration (required for some types)
            optimize_for_device: Whether to optimize for the target device

        Returns:
            Tuple of (quantized model, quantization metadata)
        """
        logger.info(f'Quantizing model with {quantization_type} quantization')
        if self.model_object is None:
            raise ValueError(
                'No model loaded. Please provide a model_path or model_object.'
                )
        quantized_id = str(uuid.uuid4())[:8]
        quantized_path = (self.cache_dir /
            f'quantized_{self.framework}_{quantization_type}_{quantized_id}')
        start_time = time.time()
        try:
            if self.framework == 'tensorflow':
                quantized_model = self._quantize_tensorflow(self.
                    model_object, quantization_type, calibration_data, str(
                    quantized_path))
            elif self.framework == 'pytorch':
                quantized_model = self._quantize_pytorch(self.model_object,
                    quantization_type, calibration_data)
            elif self.framework == 'onnx':
                quantized_model = self._quantize_onnx(self.model_object,
                    quantization_type, calibration_data, str(quantized_path))
            else:
                raise ValueError(f'Unsupported framework: {self.framework}')
            quantization_time = time.time() - start_time
            original_size = self._get_model_size(self.model_object)
            quantized_size = self._get_model_size(quantized_model)
            size_reduction = (1 - quantized_size / original_size
                ) * 100 if original_size > 0 else 0
            metadata = {'quantization_type': quantization_type, 'framework':
                self.framework, 'original_size_bytes': original_size,
                'quantized_size_bytes': quantized_size,
                'size_reduction_percent': size_reduction,
                'quantization_time_seconds': quantization_time, 'timestamp':
                datetime.now().isoformat(), 'quantized_model_path': str(
                quantized_path) if quantized_path.exists() else None}
            logger.info(
                f'Model quantized successfully. Size reduction: {size_reduction:.2f}%'
                )
            self.model_object = quantized_model
            return quantized_model, metadata
        except Exception as e:
            logger.error(f'Error during quantization: {str(e)}')
            raise

    def _quantize_tensorflow(self, model: Any, quantization_type: str,
        calibration_data: Optional[Any], output_path: str) ->Any:
        """Quantize a TensorFlow model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError('TensorFlow is not available')
        if quantization_type == 'int8':
            converter = tf.lite.TFLiteConverter.from_saved_model(self.
                model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if calibration_data is not None:

                def representative_dataset():
    """
    Representative dataset.
    
    """

                    for i in range(min(10, len(calibration_data))):
                        yield [calibration_data[i:i + 1].astype(np.float32)]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.
                    TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            quantized_tflite = converter.convert()
            with open(f'{output_path}.tflite', 'wb') as f:
                f.write(quantized_tflite)
            interpreter = tf.lite.Interpreter(model_content=quantized_tflite)
            interpreter.allocate_tensors()
            return interpreter
        elif quantization_type == 'float16':
            converter = tf.lite.TFLiteConverter.from_saved_model(self.
                model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            quantized_tflite = converter.convert()
            with open(f'{output_path}.tflite', 'wb') as f:
                f.write(quantized_tflite)
            interpreter = tf.lite.Interpreter(model_content=quantized_tflite)
            interpreter.allocate_tensors()
            return interpreter
        else:
            return model

    def _quantize_pytorch(self, model: Any, quantization_type: str,
        calibration_data: Optional[Any]) ->Any:
        """Quantize a PyTorch model."""
        if not PYTORCH_AVAILABLE:
            raise ImportError('PyTorch is not available')
        model.eval()
        if quantization_type == 'int8':
            if calibration_data is not None:
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm'
                    )
                torch.quantization.prepare(model, inplace=True)
                with torch.no_grad():
                    for data_batch in calibration_data:
                        if isinstance(data_batch, np.ndarray):
                            data_batch = torch.from_numpy(data_batch).float()
                        model(data_batch)
                torch.quantization.convert(model, inplace=True)
            else:
                model = torch.quantization.quantize_dynamic(model, {torch.
                    nn.Linear}, dtype=torch.qint8)
        elif quantization_type == 'float16':
            model = model.half()
        return model

    def _quantize_onnx(self, model: Any, quantization_type: str,
        calibration_data: Optional[Any], output_path: str) ->Any:
        """Quantize an ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError('ONNX Runtime is not available')
        if isinstance(model, onnx.ModelProto):
            onnx_path = f'{output_path}_original.onnx'
            onnx.save(model, onnx_path)
        else:
            onnx_path = self.model_path
        quantized_path = f'{output_path}.onnx'
        if quantization_type == 'int8':
            from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
            if calibration_data is not None:
                quantize_static(onnx_path, quantized_path, calibration_data if
                    hasattr(calibration_data, '__iter__') else [
                    calibration_data])
            else:
                quantize_dynamic(onnx_path, quantized_path, weight_type=
                    QuantType.QInt8)
        quantized_model = onnx.load(quantized_path)
        return quantized_model

    @with_exception_handling
    def _get_model_size(self, model: Any) ->int:
        """Get the size of a model in bytes."""
        if self.framework == 'tensorflow':
            if hasattr(model, 'get_concrete_function'):
                try:
                    concrete_func = model.get_concrete_function()
                    return concrete_func.graph.as_graph_def().ByteSize()
                except:
                    return 0
            elif hasattr(model, '_model_size'):
                return model._model_size
            else:
                return 0
        elif self.framework == 'pytorch':
            import io
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            return buffer.getbuffer().nbytes
        elif self.framework == 'onnx':
            if hasattr(model, 'SerializeToString'):
                return len(model.SerializeToString())
            else:
                return 0
        else:
            return 0

    @with_exception_handling
    def apply_operator_fusion(self) ->Tuple[Any, Dict[str, Any]]:
        """
        Apply operator fusion to optimize the computational graph.

        Operator fusion combines multiple operations into a single optimized
        operation to reduce memory transfers and improve performance.

        Returns:
            Tuple of (optimized model, fusion metadata)
        """
        logger.info(f'Applying operator fusion for {self.framework}')
        if self.model_object is None:
            raise ValueError(
                'No model loaded. Please provide a model_path or model_object.'
                )
        start_time = time.time()
        try:
            if self.framework == 'tensorflow':
                optimized_model = self._apply_tensorflow_fusion(self.
                    model_object)
            elif self.framework == 'pytorch':
                optimized_model = self._apply_pytorch_fusion(self.model_object)
            elif self.framework == 'onnx':
                optimized_model = self._apply_onnx_fusion(self.model_object)
            else:
                raise ValueError(f'Unsupported framework: {self.framework}')
            fusion_time = time.time() - start_time
            metadata = {'framework': self.framework, 'fusion_time_seconds':
                fusion_time, 'timestamp': datetime.now().isoformat()}
            logger.info(
                f'Operator fusion applied successfully in {fusion_time:.2f} seconds'
                )
            self.model_object = optimized_model
            return optimized_model, metadata
        except Exception as e:
            logger.error(f'Error during operator fusion: {str(e)}')
            raise

    @with_exception_handling
    def _apply_tensorflow_fusion(self, model: Any) ->Any:
        """Apply operator fusion to a TensorFlow model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError('TensorFlow is not available')
        try:
            if hasattr(model, 'get_concrete_function'):
                concrete_func = model.get_concrete_function()
                from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
                frozen_func = convert_variables_to_constants_v2(concrete_func)
                config = tf.compat.v1.ConfigProto()
                rewriter_config = config.graph_options.rewrite_options
                rewriter_config.optimizers.append('pruning')
                rewriter_config.optimizers.append('constfold')
                rewriter_config.optimizers.append('layout')
                rewriter_config.optimizers.append('remapping')
                rewriter_config.optimizers.append('arithmetic')
                rewriter_config.optimizers.append('dependency')
                meta_graph = tf.compat.v1.MetaGraphDef()
                meta_graph.graph_def.CopyFrom(frozen_func.graph.as_graph_def())
                from tensorflow.python.grappler import tf_optimizer
                optimized_graph = tf_optimizer.OptimizeGraph(config, meta_graph
                    )
                return model
            elif hasattr(model, 'get_signature_list'):
                return model
            else:
                logger.warning(
                    'Unknown TensorFlow model format, returning original model'
                    )
                return model
        except Exception as e:
            logger.error(f'Error in TensorFlow fusion: {str(e)}')
            return model

    @with_exception_handling
    def _apply_pytorch_fusion(self, model: Any) ->Any:
        """Apply operator fusion to a PyTorch model."""
        if not PYTORCH_AVAILABLE:
            raise ImportError('PyTorch is not available')
        try:
            model.eval()
            with torch.no_grad():
                scripted_model = torch.jit.script(model)
                optimized_model = torch.jit.optimize_for_inference(
                    scripted_model)
            return optimized_model
        except Exception as e:
            logger.error(f'Error in PyTorch fusion: {str(e)}')
            return model

    @with_exception_handling
    def _apply_onnx_fusion(self, model: Any) ->Any:
        """Apply operator fusion to an ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError('ONNX Runtime is not available')
        try:
            import onnxruntime as ort
            temp_path = str(self.cache_dir / f'temp_model_{uuid.uuid4()}.onnx')
            onnx.save(model, temp_path)
            opt_options = ort.SessionOptions()
            opt_options.graph_optimization_level = (ort.
                GraphOptimizationLevel.ORT_ENABLE_ALL)
            optimized_model_path = str(self.cache_dir /
                f'optimized_model_{uuid.uuid4()}.onnx')
            try:
                from onnxruntime.transformers import optimizer
                opt = optimizer.optimize_model(temp_path, model_type='bert',
                    num_heads=12, hidden_size=768)
                opt.save_model_to_file(optimized_model_path)
            except Exception as fusion_error:
                logger.warning(
                    f'Transformer-specific optimization failed: {str(fusion_error)}'
                    )
                session = ort.InferenceSession(temp_path, opt_options)
                onnx.save(model, optimized_model_path)
            optimized_model = onnx.load(optimized_model_path)
            try:
                os.remove(temp_path)
            except:
                pass
            return optimized_model
        except Exception as e:
            logger.error(f'Error in ONNX fusion: {str(e)}')
            return model

    def configure_batch_inference(self, optimal_batch_size: Optional[int]=
        None, max_batch_size: int=64, dynamic_batching: bool=True,
        timeout_ms: int=100) ->Dict[str, Any]:
        """
        Configure the model for efficient batch inference.

        Args:
            optimal_batch_size: Optimal batch size (if None, will be determined automatically)
            max_batch_size: Maximum batch size to consider
            dynamic_batching: Whether to use dynamic batching
            timeout_ms: Maximum wait time for dynamic batching

        Returns:
            Dictionary with batch inference configuration
        """
        logger.info('Configuring batch inference settings')
        if self.model_object is None:
            raise ValueError(
                'No model loaded. Please provide a model_path or model_object.'
                )
        if optimal_batch_size is None:
            optimal_batch_size = self._find_optimal_batch_size(max_batch_size)
        if self.framework == 'tensorflow':
            config = self._configure_tensorflow_batching(self.model_object,
                optimal_batch_size, dynamic_batching, timeout_ms)
        elif self.framework == 'pytorch':
            config = self._configure_pytorch_batching(self.model_object,
                optimal_batch_size, dynamic_batching, timeout_ms)
        elif self.framework == 'onnx':
            config = self._configure_onnx_batching(self.model_object,
                optimal_batch_size, dynamic_batching, timeout_ms)
        else:
            raise ValueError(f'Unsupported framework: {self.framework}')
        logger.info(
            f'Batch inference configured with optimal batch size: {optimal_batch_size}'
            )
        return config

    def _find_optimal_batch_size(self, max_batch_size: int) ->int:
        """Find the optimal batch size for inference."""
        logger.info(f'Finding optimal batch size (max: {max_batch_size})')
        input_shape = self._get_model_input_shape()
        if not input_shape:
            logger.warning(
                'Could not determine input shape, using default batch size of 16'
                )
            return 16
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        batch_sizes = [b for b in batch_sizes if b <= max_batch_size]
        throughputs = []
        for batch_size in batch_sizes:
            test_data = np.random.random((batch_size,) + input_shape[1:]
                ).astype(np.float32)
            start_time = time.time()
            num_runs = 10
            for _ in range(num_runs):
                _ = self._run_inference(self.model_object, test_data)
            elapsed = time.time() - start_time
            throughput = num_runs * batch_size / elapsed
            throughputs.append(throughput)
            logger.debug(
                f'Batch size {batch_size}: {throughput:.2f} samples/sec')
        best_idx = throughputs.index(max(throughputs))
        optimal_batch_size = batch_sizes[best_idx]
        logger.info(f'Optimal batch size determined: {optimal_batch_size}')
        return optimal_batch_size

    @with_exception_handling
    def _get_model_input_shape(self) ->Optional[Tuple]:
        """Get the input shape of the model."""
        try:
            if self.framework == 'tensorflow':
                if hasattr(self.model_object, 'inputs'):
                    return tuple(self.model_object.inputs[0].shape)
                elif hasattr(self.model_object, 'get_input_details'):
                    input_details = self.model_object.get_input_details()
                    return tuple(input_details[0]['shape'])
            elif self.framework == 'pytorch':
                return 1, 3, 224, 224
            elif self.framework == 'onnx':
                inputs = self.model_object.graph.input
                if inputs:
                    shape = []
                    for dim in inputs[0].type.tensor_type.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        else:
                            shape.append(1)
                    return tuple(shape)
            return None
        except Exception as e:
            logger.warning(f'Error getting model input shape: {str(e)}')
            return None

    def _configure_tensorflow_batching(self, model: Any, batch_size: int,
        dynamic_batching: bool, timeout_ms: int) ->Dict[str, Any]:
        """Configure TensorFlow model for batch inference."""
        config = {'framework': 'tensorflow', 'optimal_batch_size':
            batch_size, 'dynamic_batching': dynamic_batching, 'timeout_ms':
            timeout_ms}
        return config

    def _configure_pytorch_batching(self, model: Any, batch_size: int,
        dynamic_batching: bool, timeout_ms: int) ->Dict[str, Any]:
        """Configure PyTorch model for batch inference."""
        config = {'framework': 'pytorch', 'optimal_batch_size': batch_size,
            'dynamic_batching': dynamic_batching, 'timeout_ms': timeout_ms}
        return config

    def _configure_onnx_batching(self, model: Any, batch_size: int,
        dynamic_batching: bool, timeout_ms: int) ->Dict[str, Any]:
        """Configure ONNX model for batch inference."""
        config = {'framework': 'onnx', 'optimal_batch_size': batch_size,
            'dynamic_batching': dynamic_batching, 'timeout_ms': timeout_ms}
        return config

    def benchmark_optimized(self, input_data: Union[np.ndarray, pd.
        DataFrame, List], batch_sizes: List[int]=None, num_runs: int=10,
        warmup_runs: int=3) ->Dict[str, Any]:
        """
        Benchmark the optimized model performance.

        Args:
            input_data: Sample input data for benchmarking
            batch_sizes: List of batch sizes to test (if None, uses the same as baseline)
            num_runs: Number of inference runs to average over
            warmup_runs: Number of warmup runs before timing

        Returns:
            Dictionary with benchmark results and comparison to baseline
        """
        logger.info(
            f'Benchmarking optimized model performance with {num_runs} runs')
        if not self.baseline_metrics:
            logger.warning(
                'No baseline metrics available. Run benchmark_baseline first.')
        if batch_sizes is None and self.baseline_metrics:
            batch_sizes = [int(k.split('_')[1]) for k in self.
                baseline_metrics.get('results', {}).keys()]
        elif batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64]
        results = {}
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        for batch_size in batch_sizes:
            batch_results = self._run_inference_benchmark(self.model_object,
                input_data, batch_size, num_runs, warmup_runs)
            results[f'batch_{batch_size}'] = batch_results
        self.optimized_metrics = {'timestamp': datetime.now().isoformat(),
            'results': results}
        if self.baseline_metrics and 'results' in self.baseline_metrics:
            comparison = self._compare_benchmark_results(self.
                baseline_metrics['results'], results)
            self.optimized_metrics['comparison'] = comparison
        return self.optimized_metrics

    def _compare_benchmark_results(self, baseline_results: Dict[str, Dict[
        str, float]], optimized_results: Dict[str, Dict[str, float]]) ->Dict[
        str, Dict[str, float]]:
        """Compare baseline and optimized benchmark results."""
        comparison = {}
        for batch_key in baseline_results:
            if batch_key in optimized_results:
                baseline = baseline_results[batch_key]
                optimized = optimized_results[batch_key]
                batch_comparison = {}
                for metric in ['avg_latency_ms', 'p95_latency_ms',
                    'p99_latency_ms', 'throughput_samples_per_sec']:
                    if metric in baseline and metric in optimized:
                        if 'latency' in metric:
                            improvement = (baseline[metric] - optimized[metric]
                                ) / baseline[metric] * 100
                        else:
                            improvement = (optimized[metric] - baseline[metric]
                                ) / baseline[metric] * 100
                        batch_comparison[f'{metric}_improvement_pct'
                            ] = improvement
                comparison[batch_key] = batch_comparison
        return comparison

    @with_exception_handling
    def _apply_tensorflow_fusion(self, model: Any) ->Any:
        """Apply operator fusion to a TensorFlow model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError('TensorFlow is not available')
        try:
            if hasattr(model, 'get_concrete_function'):
                concrete_func = model.get_concrete_function()
                from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
                frozen_func = convert_variables_to_constants_v2(concrete_func)
                config = tf.compat.v1.ConfigProto()
                rewriter_config = config.graph_options.rewrite_options
                rewriter_config.optimizers.append('pruning')
                rewriter_config.optimizers.append('constfold')
                rewriter_config.optimizers.append('layout')
                rewriter_config.optimizers.append('remapping')
                rewriter_config.optimizers.append('arithmetic')
                rewriter_config.optimizers.append('dependency')
                meta_graph = tf.compat.v1.MetaGraphDef()
                meta_graph.graph_def.CopyFrom(frozen_func.graph.as_graph_def())
                from tensorflow.python.grappler import tf_optimizer
                optimized_graph = tf_optimizer.OptimizeGraph(config, meta_graph
                    )
                return model
            elif hasattr(model, 'get_signature_list'):
                return model
            else:
                logger.warning(
                    'Unknown TensorFlow model format, returning original model'
                    )
                return model
        except Exception as e:
            logger.error(f'Error in TensorFlow fusion: {str(e)}')
            return model

    @with_exception_handling
    def _apply_pytorch_fusion(self, model: Any) ->Any:
        """Apply operator fusion to a PyTorch model."""
        if not PYTORCH_AVAILABLE:
            raise ImportError('PyTorch is not available')
        try:
            model.eval()
            with torch.no_grad():
                scripted_model = torch.jit.script(model)
                optimized_model = torch.jit.optimize_for_inference(
                    scripted_model)
            return optimized_model
        except Exception as e:
            logger.error(f'Error in PyTorch fusion: {str(e)}')
            return model

    @with_exception_handling
    def _apply_onnx_fusion(self, model: Any) ->Any:
        """Apply operator fusion to an ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError('ONNX Runtime is not available')
        try:
            import onnxruntime as ort
            temp_path = str(self.cache_dir / f'temp_model_{uuid.uuid4()}.onnx')
            onnx.save(model, temp_path)
            opt_options = ort.SessionOptions()
            opt_options.graph_optimization_level = (ort.
                GraphOptimizationLevel.ORT_ENABLE_ALL)
            optimized_model_path = str(self.cache_dir /
                f'optimized_model_{uuid.uuid4()}.onnx')
            try:
                from onnxruntime.transformers import optimizer
                opt = optimizer.optimize_model(temp_path, model_type='bert',
                    num_heads=12, hidden_size=768)
                opt.save_model_to_file(optimized_model_path)
            except Exception as fusion_error:
                logger.warning(
                    f'Transformer-specific optimization failed: {str(fusion_error)}'
                    )
                session = ort.InferenceSession(temp_path, opt_options)
                onnx.save(model, optimized_model_path)
            optimized_model = onnx.load(optimized_model_path)
            try:
                os.remove(temp_path)
            except:
                pass
            return optimized_model
        except Exception as e:
            logger.error(f'Error in ONNX fusion: {str(e)}')
            return model

    def configure_batch_inference(self, optimal_batch_size: Optional[int]=
        None, max_batch_size: int=64, dynamic_batching: bool=True,
        timeout_ms: int=100) ->Dict[str, Any]:
        """
        Configure the model for efficient batch inference.

        Args:
            optimal_batch_size: Optimal batch size (if None, will be determined automatically)
            max_batch_size: Maximum batch size to consider
            dynamic_batching: Whether to use dynamic batching
            timeout_ms: Maximum wait time for dynamic batching

        Returns:
            Dictionary with batch inference configuration
        """
        logger.info('Configuring batch inference settings')
        if self.model_object is None:
            raise ValueError(
                'No model loaded. Please provide a model_path or model_object.'
                )
        if optimal_batch_size is None:
            optimal_batch_size = self._find_optimal_batch_size(max_batch_size)
        if self.framework == 'tensorflow':
            config = self._configure_tensorflow_batching(self.model_object,
                optimal_batch_size, dynamic_batching, timeout_ms)
        elif self.framework == 'pytorch':
            config = self._configure_pytorch_batching(self.model_object,
                optimal_batch_size, dynamic_batching, timeout_ms)
        elif self.framework == 'onnx':
            config = self._configure_onnx_batching(self.model_object,
                optimal_batch_size, dynamic_batching, timeout_ms)
        else:
            raise ValueError(f'Unsupported framework: {self.framework}')
        logger.info(
            f'Batch inference configured with optimal batch size: {optimal_batch_size}'
            )
        return config

    def _find_optimal_batch_size(self, max_batch_size: int) ->int:
        """Find the optimal batch size for inference."""
        logger.info(f'Finding optimal batch size (max: {max_batch_size})')
        input_shape = self._get_model_input_shape()
        if not input_shape:
            logger.warning(
                'Could not determine input shape, using default batch size of 16'
                )
            return 16
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        batch_sizes = [b for b in batch_sizes if b <= max_batch_size]
        throughputs = []
        for batch_size in batch_sizes:
            test_data = np.random.random((batch_size,) + input_shape[1:]
                ).astype(np.float32)
            start_time = time.time()
            num_runs = 10
            for _ in range(num_runs):
                _ = self._run_inference(self.model_object, test_data)
            elapsed = time.time() - start_time
            throughput = num_runs * batch_size / elapsed
            throughputs.append(throughput)
            logger.debug(
                f'Batch size {batch_size}: {throughput:.2f} samples/sec')
        best_idx = throughputs.index(max(throughputs))
        optimal_batch_size = batch_sizes[best_idx]
        logger.info(f'Optimal batch size determined: {optimal_batch_size}')
        return optimal_batch_size

    @with_exception_handling
    def _get_model_input_shape(self) ->Optional[Tuple]:
        """Get the input shape of the model."""
        try:
            if self.framework == 'tensorflow':
                if hasattr(self.model_object, 'inputs'):
                    return tuple(self.model_object.inputs[0].shape)
                elif hasattr(self.model_object, 'get_input_details'):
                    input_details = self.model_object.get_input_details()
                    return tuple(input_details[0]['shape'])
            elif self.framework == 'pytorch':
                return 1, 3, 224, 224
            elif self.framework == 'onnx':
                inputs = self.model_object.graph.input
                if inputs:
                    shape = []
                    for dim in inputs[0].type.tensor_type.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        else:
                            shape.append(1)
                    return tuple(shape)
            return None
        except Exception as e:
            logger.warning(f'Error getting model input shape: {str(e)}')
            return None

    def _configure_tensorflow_batching(self, model: Any, batch_size: int,
        dynamic_batching: bool, timeout_ms: int) ->Dict[str, Any]:
        """Configure TensorFlow model for batch inference."""
        config = {'framework': 'tensorflow', 'optimal_batch_size':
            batch_size, 'dynamic_batching': dynamic_batching, 'timeout_ms':
            timeout_ms}
        return config

    def _configure_pytorch_batching(self, model: Any, batch_size: int,
        dynamic_batching: bool, timeout_ms: int) ->Dict[str, Any]:
        """Configure PyTorch model for batch inference."""
        config = {'framework': 'pytorch', 'optimal_batch_size': batch_size,
            'dynamic_batching': dynamic_batching, 'timeout_ms': timeout_ms}
        return config

    def _configure_onnx_batching(self, model: Any, batch_size: int,
        dynamic_batching: bool, timeout_ms: int) ->Dict[str, Any]:
        """Configure ONNX model for batch inference."""
        config = {'framework': 'onnx', 'optimal_batch_size': batch_size,
            'dynamic_batching': dynamic_batching, 'timeout_ms': timeout_ms}
        return config

    def benchmark_optimized(self, input_data: Union[np.ndarray, pd.
        DataFrame, List], batch_sizes: List[int]=None, num_runs: int=10,
        warmup_runs: int=3) ->Dict[str, Any]:
        """
        Benchmark the optimized model performance.

        Args:
            input_data: Sample input data for benchmarking
            batch_sizes: List of batch sizes to test (if None, uses the same as baseline)
            num_runs: Number of inference runs to average over
            warmup_runs: Number of warmup runs before timing

        Returns:
            Dictionary with benchmark results and comparison to baseline
        """
        logger.info(
            f'Benchmarking optimized model performance with {num_runs} runs')
        if not self.baseline_metrics:
            logger.warning(
                'No baseline metrics available. Run benchmark_baseline first.')
        if batch_sizes is None and self.baseline_metrics:
            batch_sizes = [int(k.split('_')[1]) for k in self.
                baseline_metrics.get('results', {}).keys()]
        elif batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64]
        results = {}
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        for batch_size in batch_sizes:
            batch_results = self._run_inference_benchmark(self.model_object,
                input_data, batch_size, num_runs, warmup_runs)
            results[f'batch_{batch_size}'] = batch_results
        self.optimized_metrics = {'timestamp': datetime.now().isoformat(),
            'results': results}
        if self.baseline_metrics and 'results' in self.baseline_metrics:
            comparison = self._compare_benchmark_results(self.
                baseline_metrics['results'], results)
            self.optimized_metrics['comparison'] = comparison
        return self.optimized_metrics

    def _compare_benchmark_results(self, baseline_results: Dict[str, Dict[
        str, float]], optimized_results: Dict[str, Dict[str, float]]) ->Dict[
        str, Dict[str, float]]:
        """Compare baseline and optimized benchmark results."""
        comparison = {}
        for batch_key in baseline_results:
            if batch_key in optimized_results:
                baseline = baseline_results[batch_key]
                optimized = optimized_results[batch_key]
                batch_comparison = {}
                for metric in ['avg_latency_ms', 'p95_latency_ms',
                    'p99_latency_ms', 'throughput_samples_per_sec']:
                    if metric in baseline and metric in optimized:
                        if 'latency' in metric:
                            improvement = (baseline[metric] - optimized[metric]
                                ) / baseline[metric] * 100
                        else:
                            improvement = (optimized[metric] - baseline[metric]
                                ) / baseline[metric] * 100
                        batch_comparison[f'{metric}_improvement_pct'
                            ] = improvement
                comparison[batch_key] = batch_comparison
        return comparison
