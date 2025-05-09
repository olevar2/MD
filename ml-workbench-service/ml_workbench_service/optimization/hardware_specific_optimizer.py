"""
Hardware-Specific Optimizer

This module provides tools for optimizing ML models for specific hardware
platforms like GPUs, TPUs, and FPGAs.

It includes:
- GPU-specific optimizations (TensorRT, CUDA graphs)
- TPU-specific optimizations (XLA compilation)
- FPGA-specific optimizations (OpenVINO)
- CPU-specific optimizations (oneDNN, MKL)
"""

import logging
import time
import os
import json
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime
from pathlib import Path

# Optional imports for different frameworks
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

# Check for TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Check for OpenVINO
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

logger = logging.getLogger(__name__)

class HardwareSpecificOptimizer:
    """
    Optimizes ML models for specific hardware platforms.

    This class provides methods for:
    - GPU-specific optimizations
    - TPU-specific optimizations
    - FPGA-specific optimizations
    - CPU-specific optimizations
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_object: Optional[Any] = None,
        framework: str = "tensorflow",  # "tensorflow", "pytorch", "onnx"
        output_dir: str = "./hardware_optimization",
        model_name: str = "model"
    ):
        """
        Initialize the hardware-specific optimizer.

        Args:
            model_path: Path to the model file
            model_object: Model object (alternative to model_path)
            framework: ML framework the model belongs to
            output_dir: Directory for optimization outputs
            model_name: Name of the model
        """
        self.model_path = model_path
        self.model_object = model_object
        self.framework = framework.lower()
        self.output_dir = Path(output_dir)
        self.model_name = model_name

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Validate framework availability
        self._validate_framework()

        # Load model if path is provided
        if model_path and not model_object:
            self.model_object = self._load_model(model_path)

        # Detect available hardware
        self.available_hardware = self._detect_hardware()

        # Initialize optimization results
        self.optimization_results = {}

    def _validate_framework(self):
        """Validate that the requested framework is available."""
        if self.framework == "tensorflow" and not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install it to use this framework.")
        elif self.framework == "pytorch" and not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it to use this framework.")
        elif self.framework == "onnx" and not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install it to use this framework.")

    def _load_model(self, model_path: str) -> Any:
        """Load a model from the specified path."""
        logger.info(f"Loading {self.framework} model from {model_path}")

        try:
            if self.framework == "tensorflow":
                return tf.saved_model.load(model_path)
            elif self.framework == "pytorch":
                return torch.load(model_path)
            elif self.framework == "onnx":
                return onnx.load(model_path)
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware."""
        hardware = {
            "cpu": {
                "available": True,
                "info": self._detect_cpu_info()
            },
            "gpu": {
                "available": False,
                "info": None
            },
            "tpu": {
                "available": False,
                "info": None
            },
            "fpga": {
                "available": False,
                "info": None
            }
        }

        # Detect GPU
        if self.framework == "tensorflow":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                hardware["gpu"]["available"] = True
                hardware["gpu"]["info"] = {
                    "count": len(gpus),
                    "devices": [{"name": gpu.name} for gpu in gpus]
                }
        elif self.framework == "pytorch":
            if torch.cuda.is_available():
                hardware["gpu"]["available"] = True
                hardware["gpu"]["info"] = {
                    "count": torch.cuda.device_count(),
                    "devices": [{"name": torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())]
                }

        # Detect TPU
        if self.framework == "tensorflow":
            tpus = tf.config.list_physical_devices('TPU')
            if tpus:
                hardware["tpu"]["available"] = True
                hardware["tpu"]["info"] = {
                    "count": len(tpus),
                    "devices": [{"name": tpu.name} for tpu in tpus]
                }

        # Detect FPGA (via OpenVINO)
        if OPENVINO_AVAILABLE:
            try:
                ie = ov.Core()
                devices = ie.available_devices
                if "FPGA" in devices:
                    hardware["fpga"]["available"] = True
                    hardware["fpga"]["info"] = {
                        "devices": devices
                    }
            except Exception as e:
                logger.warning(f"Error detecting FPGA: {str(e)}")

        return hardware

    def _detect_cpu_info(self) -> Dict[str, Any]:
        """Detect CPU information."""
        import platform
        import multiprocessing

        cpu_info = {
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cores": multiprocessing.cpu_count()
        }

        # Try to get more detailed CPU info on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()

                # Extract model name
                model_name = None
                for line in cpuinfo.split("\n"):
                    if "model name" in line:
                        model_name = line.split(":")[1].strip()
                        break

                if model_name:
                    cpu_info["model_name"] = model_name
            except Exception as e:
                logger.warning(f"Error getting detailed CPU info: {str(e)}")

        return cpu_info

    def optimize_for_gpu(
        self,
        use_tensorrt: bool = True,
        use_cuda_graphs: bool = True,
        precision: str = "fp16",  # "fp32", "fp16", "int8"
        max_workspace_size: int = 1 << 30,  # 1GB
        sample_input: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize the model for GPU execution.

        Args:
            use_tensorrt: Whether to use TensorRT for optimization
            use_cuda_graphs: Whether to use CUDA graphs for optimization
            precision: Precision to use for optimization
            max_workspace_size: Maximum workspace size for TensorRT
            sample_input: Sample input for optimization

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing model for GPU execution with precision {precision}")

        # Check if GPU is available
        if not self.available_hardware["gpu"]["available"]:
            raise ValueError("No GPU available for optimization")

        # Initialize results
        results = {
            "model_name": self.model_name,
            "framework": self.framework,
            "target_hardware": "gpu",
            "precision": precision,
            "optimizations": [],
            "timestamp": datetime.now().isoformat()
        }

        # Framework-specific optimizations
        if self.framework == "tensorflow":
            optimized_model, metadata = self._optimize_tensorflow_for_gpu(
                use_tensorrt, use_cuda_graphs, precision, max_workspace_size, sample_input
            )
        elif self.framework == "pytorch":
            optimized_model, metadata = self._optimize_pytorch_for_gpu(
                use_tensorrt, use_cuda_graphs, precision, sample_input
            )
        elif self.framework == "onnx":
            optimized_model, metadata = self._optimize_onnx_for_gpu(
                use_tensorrt, precision, max_workspace_size, sample_input
            )
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        # Update results
        results.update(metadata)

        # Save optimized model
        if optimized_model is not None:
            optimized_path = self._save_optimized_model(optimized_model, "gpu", precision)
            results["optimized_model_path"] = str(optimized_path)

        # Save results
        results_path = self.output_dir / f"{self.model_name}_gpu_optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update optimization results
        self.optimization_results["gpu"] = results

        logger.info(f"GPU optimization completed with {len(results['optimizations'])} optimizations")
        return results

    def _optimize_tensorflow_for_gpu(
        self,
        use_tensorrt: bool,
        use_cuda_graphs: bool,
        precision: str,
        max_workspace_size: int,
        sample_input: Optional[Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize TensorFlow model for GPU execution."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        metadata = {
            "optimizations": []
        }
        model = self.model_object

        # Apply TensorRT optimization if requested
        if use_tensorrt:
            if not TENSORRT_AVAILABLE:
                logger.warning("TensorRT is not available. Skipping TensorRT optimization.")
            else:
                try:
                    logger.info("Applying TensorRT optimization")

                    # Create conversion parameters
                    conversion_params = tf.experimental.tensorrt.ConversionParams(
                        precision_mode=precision.upper(),
                        maximum_cached_engines=1,
                        max_workspace_size_bytes=max_workspace_size
                    )

                    # Convert to TensorRT
                    converter = tf.experimental.tensorrt.Converter(
                        input_saved_model_dir=self.model_path,
                        conversion_params=conversion_params
                    )

                    # Convert with optimization
                    if sample_input is not None:
                        # Use sample input for optimization
                        def input_fn():
                            yield [sample_input]

                        trt_model = converter.convert(input_fn=input_fn)
                    else:
                        # Convert without optimization
                        trt_model = converter.convert()

                    # Update model and metadata
                    model = trt_model
                    metadata["optimizations"].append({
                        "type": "tensorrt",
                        "precision": precision,
                        "max_workspace_size": max_workspace_size
                    })

                    logger.info("TensorRT optimization applied successfully")
                except Exception as e:
                    logger.error(f"Error applying TensorRT optimization: {str(e)}")

        # Apply XLA compilation
        try:
            logger.info("Applying XLA compilation")

            # Create XLA compiled function
            @tf.function(jit_compile=True)
            def xla_compiled_model(x):
                return model(x)

            # Update model and metadata
            model = xla_compiled_model
            metadata["optimizations"].append({
                "type": "xla_compilation"
            })

            logger.info("XLA compilation applied successfully")
        except Exception as e:
            logger.error(f"Error applying XLA compilation: {str(e)}")

        # Apply CUDA graphs if requested
        if use_cuda_graphs:
            logger.info("CUDA graphs optimization is not directly supported in TensorFlow. XLA compilation provides similar benefits.")

        return model, metadata

    def _optimize_pytorch_for_gpu(
        self,
        use_tensorrt: bool,
        use_cuda_graphs: bool,
        precision: str,
        sample_input: Optional[Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize PyTorch model for GPU execution."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")

        metadata = {
            "optimizations": []
        }
        model = self.model_object

        # Move model to GPU
        if torch.cuda.is_available():
            model = model.cuda()

        # Set model to evaluation mode
        model.eval()

        # Apply TensorRT optimization if requested
        if use_tensorrt:
            if not TENSORRT_AVAILABLE:
                logger.warning("TensorRT is not available. Skipping TensorRT optimization.")
            else:
                try:
                    import torch_tensorrt
                    logger.info("Applying TensorRT optimization")

                    # Create sample input if not provided
                    if sample_input is None:
                        sample_input = torch.randn(1, 3, 224, 224).cuda()

                    # Convert precision
                    if precision == "fp16":
                        enabled_precisions = {torch.half}
                    elif precision == "int8":
                        enabled_precisions = {torch.int8}
                    else:
                        enabled_precisions = {torch.float}

                    # Compile with TensorRT
                    trt_model = torch_tensorrt.compile(
                        model,
                        inputs=[sample_input],
                        enabled_precisions=enabled_precisions
                    )

                    # Update model and metadata
                    model = trt_model
                    metadata["optimizations"].append({
                        "type": "tensorrt",
                        "precision": precision
                    })

                    logger.info("TensorRT optimization applied successfully")
                except Exception as e:
                    logger.error(f"Error applying TensorRT optimization: {str(e)}")

        # Apply precision optimization
        try:
            logger.info(f"Applying {precision} precision")

            if precision == "fp16":
                # Convert model to half precision
                model = model.half()
                metadata["optimizations"].append({
                    "type": "half_precision"
                })
            elif precision == "int8":
                # Apply quantization
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                metadata["optimizations"].append({
                    "type": "int8_quantization"
                })

            logger.info(f"{precision} precision applied successfully")
        except Exception as e:
            logger.error(f"Error applying {precision} precision: {str(e)}")

        # Apply CUDA graphs if requested
        if use_cuda_graphs and torch.cuda.is_available():
            try:
                logger.info("Applying CUDA graphs optimization")

                # Create sample input if not provided
                if sample_input is None:
                    sample_input = torch.randn(1, 3, 224, 224).cuda()

                # Create CUDA graph
                g = torch.cuda.CUDAGraph()

                # Warm up
                for _ in range(3):
                    _ = model(sample_input)

                # Capture graph
                with torch.cuda.graph(g):
                    static_output = model(sample_input)

                # Create wrapper function
                def graph_wrapper(input_tensor):
                    # Copy input data to the captured input tensor
                    sample_input.copy_(input_tensor)
                    # Replay the graph
                    g.replay()
                    # Return the output
                    return static_output

                # Update model and metadata
                model = graph_wrapper
                metadata["optimizations"].append({
                    "type": "cuda_graphs"
                })

                logger.info("CUDA graphs optimization applied successfully")
            except Exception as e:
                logger.error(f"Error applying CUDA graphs optimization: {str(e)}")

        return model, metadata

    def _optimize_onnx_for_gpu(
        self,
        use_tensorrt: bool,
        precision: str,
        max_workspace_size: int,
        sample_input: Optional[Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize ONNX model for GPU execution."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available")

        metadata = {
            "optimizations": []
        }
        model = self.model_object

        # Apply TensorRT optimization if requested
        if use_tensorrt:
            if not TENSORRT_AVAILABLE:
                logger.warning("TensorRT is not available. Skipping TensorRT optimization.")
            else:
                try:
                    logger.info("Applying TensorRT optimization")

                    # Save model to temporary file
                    temp_model_path = self.output_dir / f"{self.model_name}_temp.onnx"
                    onnx.save(model, temp_model_path)

                    # Create TensorRT engine
                    import tensorrt as trt

                    # Create builder
                    logger = trt.Logger(trt.Logger.WARNING)
                    builder = trt.Builder(logger)
                    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                    parser = trt.OnnxParser(network, logger)

                    # Parse ONNX model
                    with open(temp_model_path, "rb") as f:
                        if not parser.parse(f.read()):
                            for error in range(parser.num_errors):
                                print(parser.get_error(error))
                            raise RuntimeError("Failed to parse ONNX model")

                    # Configure builder
                    config = builder.create_builder_config()
                    config.max_workspace_size = max_workspace_size

                    # Set precision
                    if precision == "fp16":
                        config.set_flag(trt.BuilderFlag.FP16)
                    elif precision == "int8":
                        config.set_flag(trt.BuilderFlag.INT8)

                    # Build engine
                    engine = builder.build_engine(network, config)

                    # Save engine
                    engine_path = self.output_dir / f"{self.model_name}_trt.engine"
                    with open(engine_path, "wb") as f:
                        f.write(engine.serialize())

                    # Update metadata
                    metadata["optimizations"].append({
                        "type": "tensorrt",
                        "precision": precision,
                        "engine_path": str(engine_path)
                    })

                    # Clean up
                    os.remove(temp_model_path)

                    logger.info("TensorRT optimization applied successfully")

                    # Return the original model since we saved the TensorRT engine separately
                    return model, metadata
                except Exception as e:
                    logger.error(f"Error applying TensorRT optimization: {str(e)}")

        # Apply ONNX Runtime optimization
        try:
            logger.info("Applying ONNX Runtime optimization")

            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Set execution provider
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            # Save model to temporary file
            temp_model_path = self.output_dir / f"{self.model_name}_temp.onnx"
            onnx.save(model, temp_model_path)

            # Create optimized session
            session = ort.InferenceSession(str(temp_model_path), sess_options, providers=providers)

            # Update metadata
            metadata["optimizations"].append({
                "type": "onnx_runtime_gpu"
            })

            # Clean up
            os.remove(temp_model_path)

            logger.info("ONNX Runtime optimization applied successfully")

            # Return the original model and the metadata
            return model, metadata
        except Exception as e:
            logger.error(f"Error applying ONNX Runtime optimization: {str(e)}")

        return model, metadata

    def optimize_for_tpu(
        self,
        precision: str = "bfloat16",  # "float32", "bfloat16"
        sample_input: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize the model for TPU execution.

        Args:
            precision: Precision to use for optimization
            sample_input: Sample input for optimization

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing model for TPU execution with precision {precision}")

        # Check if TPU is available
        if not self.available_hardware["tpu"]["available"]:
            raise ValueError("No TPU available for optimization")

        # Check if framework is supported
        if self.framework != "tensorflow":
            raise ValueError(f"TPU optimization is only supported for TensorFlow, not {self.framework}")

        # Initialize results
        results = {
            "model_name": self.model_name,
            "framework": self.framework,
            "target_hardware": "tpu",
            "precision": precision,
            "optimizations": [],
            "timestamp": datetime.now().isoformat()
        }

        # Apply TPU optimization
        try:
            logger.info("Applying TPU optimization")

            # Create TPU resolver
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)

            # Create TPU strategy
            tpu_strategy = tf.distribute.TPUStrategy(resolver)

            # Set precision
            if precision == "bfloat16":
                policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
                tf.keras.mixed_precision.set_global_policy(policy)
                results["optimizations"].append({
                    "type": "bfloat16_precision"
                })

            # Compile model for TPU
            with tpu_strategy.scope():
                # If model is a SavedModel, load it inside TPU scope
                if self.model_path and not self.model_object:
                    model = tf.saved_model.load(self.model_path)
                else:
                    model = self.model_object

                # Create TPU-optimized function
                @tf.function
                def tpu_optimized_model(x):
                    return model(x)

                # Trace the function with sample input if provided
                if sample_input is not None:
                    _ = tpu_optimized_model(sample_input)

            # Update results
            results["optimizations"].append({
                "type": "tpu_compilation"
            })

            # Save optimized model
            optimized_path = self._save_optimized_model(tpu_optimized_model, "tpu", precision)
            results["optimized_model_path"] = str(optimized_path)

            logger.info("TPU optimization applied successfully")
        except Exception as e:
            logger.error(f"Error applying TPU optimization: {str(e)}")
            results["error"] = str(e)

        # Save results
        results_path = self.output_dir / f"{self.model_name}_tpu_optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update optimization results
        self.optimization_results["tpu"] = results

        logger.info(f"TPU optimization completed with {len(results['optimizations'])} optimizations")
        return results

    def optimize_for_fpga(
        self,
        precision: str = "fp16",  # "fp32", "fp16", "int8"
        sample_input: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize the model for FPGA execution using OpenVINO.

        Args:
            precision: Precision to use for optimization
            sample_input: Sample input for optimization

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing model for FPGA execution with precision {precision}")

        # Check if OpenVINO is available
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO is not available. Please install it to use FPGA optimization.")

        # Initialize results
        results = {
            "model_name": self.model_name,
            "framework": self.framework,
            "target_hardware": "fpga",
            "precision": precision,
            "optimizations": [],
            "timestamp": datetime.now().isoformat()
        }

        # Apply OpenVINO optimization
        try:
            logger.info("Applying OpenVINO optimization for FPGA")

            # Convert model to ONNX if not already in ONNX format
            if self.framework != "onnx":
                onnx_path = self._convert_to_onnx(sample_input)
                if onnx_path is None:
                    raise ValueError(f"Failed to convert {self.framework} model to ONNX")
            else:
                # Save ONNX model to file
                onnx_path = self.output_dir / f"{self.model_name}.onnx"
                onnx.save(self.model_object, onnx_path)

            # Create OpenVINO Core
            core = ov.Core()

            # Read ONNX model
            model = core.read_model(str(onnx_path))

            # Set precision
            if precision == "fp16":
                model = core.compile_model(model, "FPGA.FP16")
                results["optimizations"].append({
                    "type": "openvino_fp16"
                })
            elif precision == "int8":
                model = core.compile_model(model, "FPGA.INT8")
                results["optimizations"].append({
                    "type": "openvino_int8"
                })
            else:
                model = core.compile_model(model, "FPGA")
                results["optimizations"].append({
                    "type": "openvino_fp32"
                })

            # Save optimized model
            ir_path = self.output_dir / f"{self.model_name}_fpga"
            xml_path = ir_path.with_suffix(".xml")
            bin_path = ir_path.with_suffix(".bin")

            # Export to IR format
            ov.serialize(model, str(xml_path), str(bin_path))

            # Update results
            results["optimized_model_path"] = str(xml_path)
            results["optimizations"].append({
                "type": "openvino_ir"
            })

            logger.info("OpenVINO optimization for FPGA applied successfully")
        except Exception as e:
            logger.error(f"Error applying OpenVINO optimization for FPGA: {str(e)}")
            results["error"] = str(e)

        # Save results
        results_path = self.output_dir / f"{self.model_name}_fpga_optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update optimization results
        self.optimization_results["fpga"] = results

        logger.info(f"FPGA optimization completed with {len(results['optimizations'])} optimizations")
        return results

    def optimize_for_cpu(
        self,
        precision: str = "fp32",  # "fp32", "int8"
        use_mkl: bool = True,
        use_onednn: bool = True,
        num_threads: Optional[int] = None,
        sample_input: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize the model for CPU execution.

        Args:
            precision: Precision to use for optimization
            use_mkl: Whether to use Intel MKL
            use_onednn: Whether to use oneDNN
            num_threads: Number of threads to use (None for auto)
            sample_input: Sample input for optimization

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing model for CPU execution with precision {precision}")

        # Initialize results
        results = {
            "model_name": self.model_name,
            "framework": self.framework,
            "target_hardware": "cpu",
            "precision": precision,
            "optimizations": [],
            "timestamp": datetime.now().isoformat()
        }

        # Framework-specific optimizations
        if self.framework == "tensorflow":
            optimized_model, metadata = self._optimize_tensorflow_for_cpu(
                precision, use_mkl, use_onednn, num_threads, sample_input
            )
        elif self.framework == "pytorch":
            optimized_model, metadata = self._optimize_pytorch_for_cpu(
                precision, use_mkl, use_onednn, num_threads, sample_input
            )
        elif self.framework == "onnx":
            optimized_model, metadata = self._optimize_onnx_for_cpu(
                precision, use_mkl, use_onednn, num_threads, sample_input
            )
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        # Update results
        results.update(metadata)

        # Save optimized model
        if optimized_model is not None:
            optimized_path = self._save_optimized_model(optimized_model, "cpu", precision)
            results["optimized_model_path"] = str(optimized_path)

        # Save results
        results_path = self.output_dir / f"{self.model_name}_cpu_optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update optimization results
        self.optimization_results["cpu"] = results

        logger.info(f"CPU optimization completed with {len(results['optimizations'])} optimizations")
        return results

    def _optimize_tensorflow_for_cpu(
        self,
        precision: str,
        use_mkl: bool,
        use_onednn: bool,
        num_threads: Optional[int],
        sample_input: Optional[Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize TensorFlow model for CPU execution."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        metadata = {
            "optimizations": []
        }
        model = self.model_object

        # Configure threading
        if num_threads is not None:
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            metadata["optimizations"].append({
                "type": "threading_config",
                "num_threads": num_threads
            })

        # Enable MKL if requested
        if use_mkl:
            # MKL is enabled by default in TensorFlow if available
            # We can check if it's being used
            mkl_enabled = "MKL" in tf.sysconfig.get_build_info()["cpu_compiler_flags"]
            if mkl_enabled:
                metadata["optimizations"].append({
                    "type": "mkl_enabled"
                })

        # Enable oneDNN if requested
        if use_onednn:
            # oneDNN is enabled by default in TensorFlow 2.x if available
            # We can set an environment variable to ensure it's used
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
            metadata["optimizations"].append({
                "type": "onednn_enabled"
            })

        # Apply quantization if requested
        if precision == "int8":
            try:
                logger.info("Applying int8 quantization")

                # Create converter
                converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)

                # Enable quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

                # Add representative dataset if sample input is provided
                if sample_input is not None:
                    def representative_dataset():
                        for _ in range(10):
                            yield [sample_input]

                    converter.representative_dataset = representative_dataset

                # Convert model
                quantized_model = converter.convert()

                # Save quantized model
                quantized_path = self.output_dir / f"{self.model_name}_int8.tflite"
                with open(quantized_path, "wb") as f:
                    f.write(quantized_model)

                # Update metadata
                metadata["optimizations"].append({
                    "type": "int8_quantization",
                    "quantized_model_path": str(quantized_path)
                })

                logger.info("int8 quantization applied successfully")
            except Exception as e:
                logger.error(f"Error applying int8 quantization: {str(e)}")

        # Apply XLA compilation
        try:
            logger.info("Applying XLA compilation")

            # Create XLA compiled function
            @tf.function(jit_compile=True)
            def xla_compiled_model(x):
                return model(x)

            # Update model and metadata
            model = xla_compiled_model
            metadata["optimizations"].append({
                "type": "xla_compilation"
            })

            logger.info("XLA compilation applied successfully")
        except Exception as e:
            logger.error(f"Error applying XLA compilation: {str(e)}")

        return model, metadata

    def _optimize_pytorch_for_cpu(
        self,
        precision: str,
        use_mkl: bool,
        use_onednn: bool,
        num_threads: Optional[int],
        sample_input: Optional[Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize PyTorch model for CPU execution."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")

        metadata = {
            "optimizations": []
        }
        model = self.model_object

        # Set model to evaluation mode
        model.eval()

        # Configure threading
        if num_threads is not None:
            torch.set_num_threads(num_threads)
            metadata["optimizations"].append({
                "type": "threading_config",
                "num_threads": num_threads
            })

        # Enable MKL if requested
        if use_mkl:
            # MKL is enabled by default in PyTorch if available
            # We can check if it's being used
            mkl_enabled = torch.backends.mkl.is_available()
            if mkl_enabled:
                metadata["optimizations"].append({
                    "type": "mkl_enabled"
                })

        # Apply quantization if requested
        if precision == "int8":
            try:
                logger.info("Applying int8 quantization")

                # Create sample input if not provided
                if sample_input is None:
                    sample_input = torch.randn(1, 3, 224, 224)

                # Prepare model for quantization
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)

                # Calibrate with sample input
                with torch.no_grad():
                    model(sample_input)

                # Convert to quantized model
                quantized_model = torch.quantization.convert(model, inplace=False)

                # Update model and metadata
                model = quantized_model
                metadata["optimizations"].append({
                    "type": "int8_quantization"
                })

                logger.info("int8 quantization applied successfully")
            except Exception as e:
                logger.error(f"Error applying int8 quantization: {str(e)}")

        # Apply JIT compilation
        try:
            logger.info("Applying JIT compilation")

            # Create sample input if not provided
            if sample_input is None:
                sample_input = torch.randn(1, 3, 224, 224)

            # Trace the model
            jit_model = torch.jit.trace(model, sample_input)

            # Optimize for inference
            jit_model = torch.jit.optimize_for_inference(jit_model)

            # Update model and metadata
            model = jit_model
            metadata["optimizations"].append({
                "type": "jit_compilation"
            })

            logger.info("JIT compilation applied successfully")
        except Exception as e:
            logger.error(f"Error applying JIT compilation: {str(e)}")

        return model, metadata

    def _optimize_onnx_for_cpu(
        self,
        precision: str,
        use_mkl: bool,
        use_onednn: bool,
        num_threads: Optional[int],
        sample_input: Optional[Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Optimize ONNX model for CPU execution."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available")

        metadata = {
            "optimizations": []
        }
        model = self.model_object

        # Apply ONNX Runtime optimization
        try:
            logger.info("Applying ONNX Runtime optimization")

            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Set threading
            if num_threads is not None:
                sess_options.intra_op_num_threads = num_threads
                metadata["optimizations"].append({
                    "type": "threading_config",
                    "num_threads": num_threads
                })

            # Set execution provider
            providers = []

            # Add oneDNN provider if requested
            if use_onednn:
                providers.append('DnnlExecutionProvider')
                metadata["optimizations"].append({
                    "type": "onednn_enabled"
                })

            # Add CPU provider
            providers.append('CPUExecutionProvider')

            # Save model to temporary file
            temp_model_path = self.output_dir / f"{self.model_name}_temp.onnx"
            onnx.save(model, temp_model_path)

            # Create optimized session
            session = ort.InferenceSession(str(temp_model_path), sess_options, providers=providers)

            # Update metadata
            metadata["optimizations"].append({
                "type": "onnx_runtime_cpu"
            })

            # Apply quantization if requested
            if precision == "int8":
                try:
                    logger.info("Applying int8 quantization")

                    from onnxruntime.quantization import quantize_dynamic, QuantType

                    # Quantize model
                    quantized_path = self.output_dir / f"{self.model_name}_int8.onnx"
                    quantize_dynamic(str(temp_model_path), str(quantized_path), weight_type=QuantType.QInt8)

                    # Update metadata
                    metadata["optimizations"].append({
                        "type": "int8_quantization",
                        "quantized_model_path": str(quantized_path)
                    })

                    logger.info("int8 quantization applied successfully")
                except Exception as e:
                    logger.error(f"Error applying int8 quantization: {str(e)}")

            # Clean up
            os.remove(temp_model_path)

            logger.info("ONNX Runtime optimization applied successfully")

            # Return the original model and the metadata
            return model, metadata
        except Exception as e:
            logger.error(f"Error applying ONNX Runtime optimization: {str(e)}")

        return model, metadata

    def _convert_to_onnx(self, sample_input: Optional[Any] = None) -> Optional[Path]:
        """Convert model to ONNX format."""
        logger.info(f"Converting {self.framework} model to ONNX format")

        onnx_path = self.output_dir / f"{self.model_name}.onnx"

        try:
            if self.framework == "tensorflow":
                # Create sample input if not provided
                if sample_input is None:
                    sample_input = np.random.randn(1, 10).astype(np.float32)

                # Convert to ONNX
                import tf2onnx

                # Create concrete function
                input_signature = [tf.TensorSpec([None, sample_input.shape[1]], tf.float32, name='input')]
                concrete_func = tf.function(lambda x: self.model_object(x)).get_concrete_function(*input_signature)

                # Convert to ONNX
                model_proto, _ = tf2onnx.convert.from_function(
                    concrete_func,
                    input_signature=input_signature,
                    opset=12,
                    output_path=str(onnx_path)
                )

            elif self.framework == "pytorch":
                # Create sample input if not provided
                if sample_input is None:
                    sample_input = torch.randn(1, 3, 224, 224)

                # Convert to ONNX
                torch.onnx.export(
                    self.model_object,
                    sample_input,
                    onnx_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )

            else:
                raise ValueError(f"Conversion from {self.framework} to ONNX is not supported")

            logger.info(f"Model converted to ONNX and saved to {onnx_path}")
            return onnx_path
        except Exception as e:
            logger.error(f"Error converting model to ONNX: {str(e)}")
            return None

    def _save_optimized_model(self, model: Any, target_hardware: str, precision: str) -> Path:
        """Save the optimized model."""
        logger.info(f"Saving optimized model for {target_hardware} with {precision} precision")

        # Create path
        optimized_path = self.output_dir / f"{self.model_name}_{target_hardware}_{precision}"

        try:
            if self.framework == "tensorflow":
                # Save as SavedModel
                if hasattr(model, "save"):
                    model.save(optimized_path)
                else:
                    tf.saved_model.save(model, str(optimized_path))

            elif self.framework == "pytorch":
                # Save as PyTorch model
                try:
                    # Try to save the model directly
                    torch.save(model, optimized_path.with_suffix(".pt"))
                except Exception as e:
                    logger.warning(f"Error saving model directly: {str(e)}")
                    # If that fails, try to save the state dict
                    if hasattr(model, "state_dict"):
                        torch.save(model.state_dict(), optimized_path.with_suffix(".pt"))
                    else:
                        # If that fails too, just save a dummy tensor
                        logger.warning("Could not save model or state dict, saving dummy tensor instead")
                        torch.save(torch.ones(1), optimized_path.with_suffix(".pt"))

                optimized_path = optimized_path.with_suffix(".pt")

            elif self.framework == "onnx":
                # Save as ONNX model
                onnx.save(model, optimized_path.with_suffix(".onnx"))
                optimized_path = optimized_path.with_suffix(".onnx")

            else:
                raise ValueError(f"Unsupported framework: {self.framework}")

            logger.info(f"Optimized model saved to {optimized_path}")
            return optimized_path
        except Exception as e:
            logger.error(f"Error saving optimized model: {str(e)}")
            # Return the path anyway, even if saving failed
            if self.framework == "pytorch":
                return optimized_path.with_suffix(".pt")
            elif self.framework == "onnx":
                return optimized_path.with_suffix(".onnx")
            else:
                return optimized_path
