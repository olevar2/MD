"""
ML Pipeline Integrator

This module provides tools for integrating optimization techniques with
existing ML pipelines in the forex trading platform.

It includes:
- Discovery of existing ML models and pipelines
- Integration of optimization techniques
- Validation of optimized pipelines
- Deployment of optimized pipelines
"""

import logging
import time
import os
import json
import shutil
import importlib
import inspect
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from ml_workbench_service.optimization.model_inference_optimizer import ModelInferenceOptimizer
from ml_workbench_service.optimization.feature_engineering_optimizer import FeatureEngineeringOptimizer
from ml_workbench_service.optimization.model_training_optimizer import ModelTrainingOptimizer
from ml_workbench_service.optimization.model_serving_optimizer import ModelServingOptimizer

logger = logging.getLogger(__name__)

class MLPipelineIntegrator:
    """
    Integrates optimization techniques with existing ML pipelines.

    This class provides methods for:
    - Discovering existing ML models and pipelines
    - Integrating optimization techniques
    - Validating optimized pipelines
    - Deploying optimized pipelines
    """

    def __init__(
        self,
        project_root: str,
        output_dir: str = "./ml_pipeline_optimization",
        config_file: Optional[str] = None
    ):
        """
        Initialize the ML pipeline integrator.

        Args:
            project_root: Root directory of the project
            output_dir: Directory for optimization outputs
            config_file: Path to configuration file
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load configuration if provided
        self.config = {}
        if config_file:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.config = json.load(f)

        # Initialize discovered components
        self.discovered_models = {}
        self.discovered_feature_pipelines = {}
        self.discovered_training_pipelines = {}
        self.discovered_serving_endpoints = {}

        # Initialize optimization results
        self.optimization_results = {
            "models": {},
            "feature_pipelines": {},
            "training_pipelines": {},
            "serving_endpoints": {}
        }

    def discover_ml_components(
        self,
        scan_patterns: Optional[Dict[str, List[str]]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Discover ML components in the project.

        Args:
            scan_patterns: Dictionary of component types to file patterns
            exclude_patterns: List of patterns to exclude

        Returns:
            Dictionary of discovered components
        """
        logger.info(f"Discovering ML components in {self.project_root}")

        # Default scan patterns if not provided
        if scan_patterns is None:
            scan_patterns = {
                "models": ["**/models/*.py", "**/model/*.py", "**/*model*.py"],
                "feature_pipelines": ["**/features/*.py", "**/feature/*.py", "**/*feature*.py"],
                "training_pipelines": ["**/train/*.py", "**/training/*.py", "**/*train*.py"],
                "serving_endpoints": ["**/serve/*.py", "**/serving/*.py", "**/*serve*.py"]
            }

        # Default exclude patterns if not provided
        if exclude_patterns is None:
            exclude_patterns = ["**/test/*", "**/tests/*", "**/__pycache__/*"]

        # Discover components
        for component_type, patterns in scan_patterns.items():
            logger.info(f"Scanning for {component_type}...")
            discovered = self._scan_for_components(patterns, exclude_patterns)

            if component_type == "models":
                self.discovered_models = discovered
            elif component_type == "feature_pipelines":
                self.discovered_feature_pipelines = discovered
            elif component_type == "training_pipelines":
                self.discovered_training_pipelines = discovered
            elif component_type == "serving_endpoints":
                self.discovered_serving_endpoints = discovered

            logger.info(f"Discovered {len(discovered)} {component_type}")

        # Combine all discoveries
        all_discoveries = {
            "models": self.discovered_models,
            "feature_pipelines": self.discovered_feature_pipelines,
            "training_pipelines": self.discovered_training_pipelines,
            "serving_endpoints": self.discovered_serving_endpoints
        }

        # Save discoveries
        discoveries_path = self.output_dir / "discovered_components.json"
        with open(discoveries_path, "w") as f:
            # Convert Path objects to strings for JSON serialization
            serializable_discoveries = {}
            for comp_type, comps in all_discoveries.items():
                serializable_discoveries[comp_type] = {}
                for name, details in comps.items():
                    serializable_details = {}
                    for k, v in details.items():
                        if isinstance(v, Path):
                            serializable_details[k] = str(v)
                        else:
                            serializable_details[k] = v
                    serializable_discoveries[comp_type][name] = serializable_details

            json.dump(serializable_discoveries, f, indent=2)

        return all_discoveries

    def _scan_for_components(
        self,
        patterns: List[str],
        exclude_patterns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Scan for components matching the given patterns."""
        import glob

        discovered = {}

        # Process each pattern
        for pattern in patterns:
            # Get absolute pattern
            abs_pattern = str(self.project_root / pattern)

            # Find matching files
            for file_path in glob.glob(abs_pattern, recursive=True):
                # Check if file should be excluded
                if any(re.match(exclude, file_path) for exclude in exclude_patterns):
                    continue

                # Analyze file
                file_info = self._analyze_python_file(file_path)

                # Add to discoveries if relevant
                if file_info:
                    for name, details in file_info.items():
                        discovered[name] = details

        return discovered

    def _analyze_python_file(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """Analyze a Python file for ML components."""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Look for ML-related imports
            ml_imports = []
            if re.search(r"import\s+tensorflow|from\s+tensorflow", content):
                ml_imports.append("tensorflow")
            if re.search(r"import\s+torch|from\s+torch", content):
                ml_imports.append("pytorch")
            if re.search(r"import\s+sklearn|from\s+sklearn", content):
                ml_imports.append("scikit-learn")
            if re.search(r"import\s+xgboost|from\s+xgboost", content):
                ml_imports.append("xgboost")
            if re.search(r"import\s+lightgbm|from\s+lightgbm", content):
                ml_imports.append("lightgbm")

            # Skip if no ML imports found
            if not ml_imports:
                return {}

            # Extract classes and functions
            classes = re.findall(r"class\s+(\w+)\s*\(", content)
            functions = re.findall(r"def\s+(\w+)\s*\(", content)

            # Categorize components
            components = {}

            # Check for model classes
            model_keywords = ["Model", "Estimator", "Classifier", "Regressor", "Network"]
            for cls in classes:
                if any(keyword in cls for keyword in model_keywords):
                    components[cls] = {
                        "type": "model",
                        "framework": ml_imports[0] if ml_imports else "unknown",
                        "file_path": Path(file_path),
                        "name": cls
                    }

            # Check for feature pipeline classes/functions
            feature_keywords = ["Feature", "Preprocess", "Transform", "Encoder", "Embedding"]
            for item in classes + functions:
                if any(keyword in item for keyword in feature_keywords):
                    components[item] = {
                        "type": "feature_pipeline",
                        "framework": ml_imports[0] if ml_imports else "unknown",
                        "file_path": Path(file_path),
                        "name": item
                    }

            # Check for training functions
            train_keywords = ["train", "fit", "learn", "optimize"]
            for func in functions:
                if any(keyword in func.lower() for keyword in train_keywords):
                    components[func] = {
                        "type": "training_pipeline",
                        "framework": ml_imports[0] if ml_imports else "unknown",
                        "file_path": Path(file_path),
                        "name": func
                    }

            # Check for serving functions
            serve_keywords = ["serve", "predict", "inference", "deploy"]
            for func in functions:
                if any(keyword in func.lower() for keyword in serve_keywords):
                    components[func] = {
                        "type": "serving_endpoint",
                        "framework": ml_imports[0] if ml_imports else "unknown",
                        "file_path": Path(file_path),
                        "name": func
                    }

            return components

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {}

    def optimize_model(
        self,
        model_name: str,
        optimization_techniques: List[str] = ["quantization", "operator_fusion", "batch_inference"],
        target_device: str = "cpu",
        validation_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize a discovered model.

        Args:
            model_name: Name of the model to optimize
            optimization_techniques: List of optimization techniques to apply
            target_device: Target device for optimization
            validation_data: Data for validating the optimized model

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing model {model_name}")

        # Check if model exists
        if model_name not in self.discovered_models:
            raise ValueError(f"Model {model_name} not found in discovered models")

        model_info = self.discovered_models[model_name]
        model_path = model_info["file_path"]

        # Load model
        model_object = self._load_model(model_path, model_name)

        # Initialize optimizer
        optimizer = ModelInferenceOptimizer(
            model_object=model_object,
            framework=model_info["framework"],
            device=target_device,
            cache_dir=str(self.output_dir / "model_cache")
        )

        # Apply optimizations
        results = {
            "model_name": model_name,
            "framework": model_info["framework"],
            "original_path": str(model_path),
            "optimizations": [],
            "timestamp": datetime.now().isoformat()
        }

        # Benchmark baseline if validation data provided
        if validation_data is not None:
            baseline_metrics = optimizer.benchmark_baseline(
                input_data=validation_data,
                batch_sizes=[1, 8, 32]
            )
            results["baseline_metrics"] = baseline_metrics

        # Apply quantization if requested
        if "quantization" in optimization_techniques:
            try:
                quantized_model, metadata = optimizer.quantize_model(
                    quantization_type="int8" if target_device == "cpu" else "float16"
                )
                results["optimizations"].append({
                    "type": "quantization",
                    "metadata": metadata
                })
                logger.info(f"Quantization applied: {metadata.get('size_reduction_percent', 0):.2f}% size reduction")
            except Exception as e:
                logger.error(f"Error during quantization: {str(e)}")

        # Apply operator fusion if requested
        if "operator_fusion" in optimization_techniques:
            try:
                fused_model, metadata = optimizer.apply_operator_fusion()
                results["optimizations"].append({
                    "type": "operator_fusion",
                    "metadata": metadata
                })
                logger.info(f"Operator fusion applied in {metadata.get('fusion_time_seconds', 0):.2f} seconds")
            except Exception as e:
                logger.error(f"Error during operator fusion: {str(e)}")

        # Configure batch inference if requested
        if "batch_inference" in optimization_techniques:
            try:
                batch_config = optimizer.configure_batch_inference(
                    optimal_batch_size=32
                )
                results["optimizations"].append({
                    "type": "batch_inference",
                    "config": batch_config
                })
                logger.info(f"Batch inference configured with batch size {batch_config.get('optimal_batch_size', 32)}")
            except Exception as e:
                logger.error(f"Error configuring batch inference: {str(e)}")

        # Benchmark optimized model if validation data provided
        if validation_data is not None:
            optimized_metrics = optimizer.benchmark_optimized(
                input_data=validation_data,
                batch_sizes=[1, 8, 32]
            )
            results["optimized_metrics"] = optimized_metrics

            # Log improvement
            if "comparison" in optimized_metrics:
                batch_key = "batch_32"
                if batch_key in optimized_metrics["comparison"]:
                    comparison = optimized_metrics["comparison"][batch_key]
                    throughput_improvement = comparison.get("throughput_samples_per_sec_improvement_pct", 0)
                    latency_improvement = comparison.get("avg_latency_ms_improvement_pct", 0)
                    logger.info(f"Performance improvement: {throughput_improvement:.2f}% throughput, {latency_improvement:.2f}% latency")

        # Save optimized model
        optimized_dir = self.output_dir / "optimized_models" / model_name
        optimized_dir.mkdir(exist_ok=True, parents=True)

        # Save results
        results_path = optimized_dir / "optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update optimization results
        self.optimization_results["models"][model_name] = results

        logger.info(f"Model {model_name} optimization completed")
        return results

    def _load_model(self, model_path: Path, model_name: str) -> Any:
        """Load a model from a Python file."""
        try:
            # Get module path relative to project root
            rel_path = model_path.relative_to(self.project_root)
            module_path = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")

            # Import module
            module = importlib.import_module(module_path)

            # Get model class or function
            if hasattr(module, model_name):
                model_class_or_func = getattr(module, model_name)

                # Check if it's a class
                if inspect.isclass(model_class_or_func):
                    # Instantiate class
                    model = model_class_or_func()
                else:
                    # Assume it's a function that returns a model
                    model = model_class_or_func()

                return model
            else:
                raise ValueError(f"Model {model_name} not found in module {module_path}")

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def optimize_feature_pipeline(
        self,
        pipeline_name: str,
        optimization_techniques: List[str] = ["caching", "parallel", "incremental"],
        sample_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize a discovered feature pipeline.

        Args:
            pipeline_name: Name of the feature pipeline to optimize
            optimization_techniques: List of optimization techniques to apply
            sample_data: Sample data for testing the pipeline

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing feature pipeline {pipeline_name}")

        # Check if pipeline exists
        if pipeline_name not in self.discovered_feature_pipelines:
            raise ValueError(f"Feature pipeline {pipeline_name} not found in discovered pipelines")

        pipeline_info = self.discovered_feature_pipelines[pipeline_name]
        pipeline_path = pipeline_info["file_path"]

        # Load pipeline
        pipeline_func = self._load_feature_pipeline(pipeline_path, pipeline_name)

        # Initialize optimizer
        optimizer = FeatureEngineeringOptimizer(
            cache_dir=str(self.output_dir / "feature_cache"),
            max_cache_size_mb=1024,
            n_jobs=multiprocessing.cpu_count()
        )

        # Apply optimizations
        results = {
            "pipeline_name": pipeline_name,
            "framework": pipeline_info["framework"],
            "original_path": str(pipeline_path),
            "optimizations": [],
            "timestamp": datetime.now().isoformat()
        }

        # Benchmark baseline if sample data provided
        if sample_data is not None:
            baseline_metrics = optimizer.benchmark_feature_pipeline(
                data=sample_data,
                feature_funcs=[pipeline_func],
                use_cache=False,
                use_parallel=False,
                n_runs=3
            )
            results["baseline_metrics"] = baseline_metrics

        # Apply caching if requested
        if "caching" in optimization_techniques:
            try:
                # Test caching with sample data
                if sample_data is not None:
                    cached_result, metadata = optimizer.cached_feature_computation(
                        data=sample_data,
                        feature_func=pipeline_func
                    )
                    results["optimizations"].append({
                        "type": "caching",
                        "metadata": metadata
                    })
                    logger.info(f"Caching applied: {metadata.get('computation_time', 0):.2f}s computation time")
            except Exception as e:
                logger.error(f"Error applying caching: {str(e)}")

        # Apply parallel processing if requested
        if "parallel" in optimization_techniques:
            try:
                # Test parallel processing with sample data
                if sample_data is not None:
                    parallel_results, metadata = optimizer.parallel_feature_computation(
                        data=sample_data,
                        feature_funcs=[pipeline_func],
                        use_cache=True
                    )
                    results["optimizations"].append({
                        "type": "parallel",
                        "metadata": metadata
                    })
                    logger.info(f"Parallel processing applied: {metadata.get('total_time', 0):.2f}s total time")
            except Exception as e:
                logger.error(f"Error applying parallel processing: {str(e)}")

        # Apply incremental computation if requested
        if "incremental" in optimization_techniques and sample_data is not None:
            try:
                # Split sample data for incremental testing
                if isinstance(sample_data, pd.DataFrame):
                    prev_data = sample_data.iloc[:int(len(sample_data) * 0.8)]
                    new_data = sample_data.iloc[int(len(sample_data) * 0.8):]

                    # Compute initial features
                    initial_features, _ = optimizer.parallel_feature_computation(
                        data=prev_data,
                        feature_funcs=[pipeline_func]
                    )

                    # Test incremental computation
                    incremental_results, metadata = optimizer.incremental_feature_computation(
                        previous_data=prev_data,
                        previous_features=initial_features,
                        new_data=new_data,
                        feature_funcs=[pipeline_func]
                    )

                    results["optimizations"].append({
                        "type": "incremental",
                        "metadata": metadata
                    })
                    logger.info(f"Incremental computation applied: {metadata.get('total_time', 0):.2f}s total time")
            except Exception as e:
                logger.error(f"Error applying incremental computation: {str(e)}")

        # Benchmark optimized pipeline if sample data provided
        if sample_data is not None:
            optimized_metrics = optimizer.benchmark_feature_pipeline(
                data=sample_data,
                feature_funcs=[pipeline_func],
                use_cache=True,
                use_parallel=True,
                n_runs=3
            )
            results["optimized_metrics"] = optimized_metrics

            # Calculate improvement
            if "baseline_metrics" in results:
                baseline_time = results["baseline_metrics"]["overall"]["avg_time"]
                optimized_time = optimized_metrics["overall"]["avg_time"]

                if baseline_time > 0:
                    improvement_pct = ((baseline_time - optimized_time) / baseline_time) * 100
                    results["improvement_pct"] = improvement_pct
                    logger.info(f"Performance improvement: {improvement_pct:.2f}%")

        # Save optimized pipeline
        optimized_dir = self.output_dir / "optimized_feature_pipelines" / pipeline_name
        optimized_dir.mkdir(exist_ok=True, parents=True)

        # Save results
        results_path = optimized_dir / "optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update optimization results
        self.optimization_results["feature_pipelines"][pipeline_name] = results

        logger.info(f"Feature pipeline {pipeline_name} optimization completed")
        return results

    def _load_feature_pipeline(self, pipeline_path: Path, pipeline_name: str) -> Callable:
        """Load a feature pipeline from a Python file."""
        try:
            # Get module path relative to project root
            rel_path = pipeline_path.relative_to(self.project_root)
            module_path = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")

            # Import module
            module = importlib.import_module(module_path)

            # Get pipeline function or class
            if hasattr(module, pipeline_name):
                pipeline = getattr(module, pipeline_name)

                # Check if it's a class
                if inspect.isclass(pipeline):
                    # Instantiate class and get its __call__ method
                    instance = pipeline()
                    return instance.__call__
                else:
                    # Assume it's a function
                    return pipeline
            else:
                raise ValueError(f"Pipeline {pipeline_name} not found in module {module_path}")

        except Exception as e:
            logger.error(f"Error loading feature pipeline {pipeline_name}: {str(e)}")
            raise

    def optimize_training_pipeline(
        self,
        pipeline_name: str,
        optimization_techniques: List[str] = ["mixed_precision", "gradient_accumulation", "distributed"],
        model_object: Optional[Any] = None,
        sample_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize a discovered training pipeline.

        Args:
            pipeline_name: Name of the training pipeline to optimize
            optimization_techniques: List of optimization techniques to apply
            model_object: Model object to use for optimization
            sample_data: Sample data for testing the pipeline

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing training pipeline {pipeline_name}")

        # Check if pipeline exists
        if pipeline_name not in self.discovered_training_pipelines:
            raise ValueError(f"Training pipeline {pipeline_name} not found in discovered pipelines")

        pipeline_info = self.discovered_training_pipelines[pipeline_name]
        pipeline_path = pipeline_info["file_path"]

        # Load pipeline
        pipeline_func = self._load_training_pipeline(pipeline_path, pipeline_name)

        # Determine framework
        framework = pipeline_info["framework"]

        # Initialize optimizer
        optimizer = ModelTrainingOptimizer(
            model=model_object,
            framework=framework,
            device="auto",
            output_dir=str(self.output_dir / "training_optimization")
        )

        # Apply optimizations
        results = {
            "pipeline_name": pipeline_name,
            "framework": framework,
            "original_path": str(pipeline_path),
            "optimizations": [],
            "timestamp": datetime.now().isoformat()
        }

        # Apply mixed precision if requested
        if "mixed_precision" in optimization_techniques:
            try:
                mp_config = optimizer.configure_mixed_precision(
                    enabled=True,
                    precision="float16"
                )
                results["optimizations"].append({
                    "type": "mixed_precision",
                    "config": mp_config
                })
                logger.info(f"Mixed precision configured with {mp_config.get('precision', 'float16')} precision")
            except Exception as e:
                logger.error(f"Error configuring mixed precision: {str(e)}")

        # Apply gradient accumulation if requested
        if "gradient_accumulation" in optimization_techniques:
            try:
                ga_config = optimizer.configure_gradient_accumulation(
                    accumulation_steps=4
                )
                results["optimizations"].append({
                    "type": "gradient_accumulation",
                    "config": ga_config
                })
                logger.info(f"Gradient accumulation configured with {ga_config.get('accumulation_steps', 4)} steps")
            except Exception as e:
                logger.error(f"Error configuring gradient accumulation: {str(e)}")

        # Apply distributed training if requested
        if "distributed" in optimization_techniques:
            try:
                dist_config = optimizer.configure_distributed_training(
                    strategy="mirrored" if framework == "tensorflow" else "data_parallel"
                )
                results["optimizations"].append({
                    "type": "distributed_training",
                    "config": dist_config
                })
                logger.info(f"Distributed training configured with {dist_config.get('strategy', 'default')} strategy")
            except Exception as e:
                logger.error(f"Error configuring distributed training: {str(e)}")

        # Benchmark training if model and data provided
        if model_object is not None and sample_data is not None:
            try:
                benchmark_results = optimizer.benchmark_training(
                    model=model_object,
                    train_dataset=sample_data,
                    batch_size=32,
                    num_epochs=1,
                    mixed_precision="mixed_precision" in optimization_techniques,
                    gradient_accumulation_steps=4 if "gradient_accumulation" in optimization_techniques else 1,
                    distributed="distributed" in optimization_techniques
                )
                results["benchmark_results"] = benchmark_results

                logger.info(f"Training benchmark completed in {benchmark_results.get('total_time_seconds', 0):.2f} seconds")
                logger.info(f"Performance: {benchmark_results.get('samples_per_second', 0):.2f} samples/second")
            except Exception as e:
                logger.error(f"Error during training benchmark: {str(e)}")

        # Save optimized pipeline
        optimized_dir = self.output_dir / "optimized_training_pipelines" / pipeline_name
        optimized_dir.mkdir(exist_ok=True, parents=True)

        # Save results
        results_path = optimized_dir / "optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update optimization results
        self.optimization_results["training_pipelines"][pipeline_name] = results

        logger.info(f"Training pipeline {pipeline_name} optimization completed")
        return results

    def _load_training_pipeline(self, pipeline_path: Path, pipeline_name: str) -> Callable:
        """Load a training pipeline from a Python file."""
        try:
            # Get module path relative to project root
            rel_path = pipeline_path.relative_to(self.project_root)
            module_path = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")

            # Import module
            module = importlib.import_module(module_path)

            # Get pipeline function
            if hasattr(module, pipeline_name):
                return getattr(module, pipeline_name)
            else:
                raise ValueError(f"Pipeline {pipeline_name} not found in module {module_path}")

        except Exception as e:
            logger.error(f"Error loading training pipeline {pipeline_name}: {str(e)}")
            raise

    def optimize_serving_endpoint(
        self,
        endpoint_name: str,
        optimization_techniques: List[str] = ["deployment_strategy", "auto_scaling", "ab_testing"],
        model_object: Optional[Any] = None,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize a discovered serving endpoint.

        Args:
            endpoint_name: Name of the serving endpoint to optimize
            optimization_techniques: List of optimization techniques to apply
            model_object: Model object to use for optimization
            model_path: Path to the model file

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing serving endpoint {endpoint_name}")

        # Check if endpoint exists
        if endpoint_name not in self.discovered_serving_endpoints:
            raise ValueError(f"Serving endpoint {endpoint_name} not found in discovered endpoints")

        endpoint_info = self.discovered_serving_endpoints[endpoint_name]
        endpoint_path = endpoint_info["file_path"]

        # Determine framework
        framework = endpoint_info["framework"]

        # Initialize optimizer
        optimizer = ModelServingOptimizer(
            model_object=model_object,
            model_path=model_path,
            framework=framework,
            serving_dir=str(self.output_dir / "serving_optimization"),
            model_name=endpoint_name
        )

        # Apply optimizations
        results = {
            "endpoint_name": endpoint_name,
            "framework": framework,
            "original_path": str(endpoint_path),
            "optimizations": [],
            "timestamp": datetime.now().isoformat()
        }

        # Prepare model for serving
        try:
            serving_metadata = optimizer.prepare_model_for_serving(
                optimization_level="performance",
                target_device="cpu"
            )
            results["serving_metadata"] = serving_metadata
            logger.info(f"Model prepared for serving with format {serving_metadata.get('format', 'unknown')}")
        except Exception as e:
            logger.error(f"Error preparing model for serving: {str(e)}")

        # Apply deployment strategy if requested
        if "deployment_strategy" in optimization_techniques:
            try:
                deployment_type = "blue_green"  # Could be "rolling", "blue_green", "canary", "shadow"
                deployment_status = optimizer.deploy_model(
                    deployment_type=deployment_type,
                    traffic_percentage=100.0
                )
                results["optimizations"].append({
                    "type": "deployment_strategy",
                    "strategy": deployment_type,
                    "status": deployment_status
                })
                logger.info(f"Deployment strategy applied: {deployment_type}")
            except Exception as e:
                logger.error(f"Error applying deployment strategy: {str(e)}")

        # Apply auto-scaling if requested
        if "auto_scaling" in optimization_techniques:
            try:
                scaling_config = optimizer.configure_auto_scaling(
                    min_replicas=1,
                    max_replicas=5,
                    target_cpu_utilization=70,
                    target_memory_utilization=80
                )

                # Simulate auto-scaling behavior
                scaling_simulation = optimizer.simulate_auto_scaling(
                    duration_minutes=2,
                    update_interval_seconds=5,
                    load_pattern="spike"
                )

                results["optimizations"].append({
                    "type": "auto_scaling",
                    "config": scaling_config,
                    "simulation": {
                        "avg_replicas": scaling_simulation["summary"]["avg_replicas"],
                        "max_replicas": scaling_simulation["summary"]["max_replicas"],
                        "num_scale_events": len(scaling_simulation["scaling_events"])
                    }
                })
                logger.info(f"Auto-scaling configured with {scaling_config['min_replicas']}-{scaling_config['max_replicas']} replicas")
            except Exception as e:
                logger.error(f"Error configuring auto-scaling: {str(e)}")

        # Apply A/B testing if requested
        if "ab_testing" in optimization_techniques and model_path:
            try:
                # For simulation, we'll use the same model as both A and B
                ab_config = optimizer.setup_ab_testing(
                    variant_b_model_path=model_path,
                    traffic_split=0.5,
                    test_duration_hours=24
                )

                # Simulate A/B test results
                ab_results = optimizer.simulate_ab_test_results(
                    duration_minutes=1,
                    update_interval_seconds=5
                )

                results["optimizations"].append({
                    "type": "ab_testing",
                    "config": {
                        "traffic_split": ab_config["traffic_split"],
                        "test_duration_hours": ab_config["test_duration_hours"]
                    },
                    "simulation": {
                        "winner": ab_results["summary"]["winner"],
                        "metrics_comparison": ab_results["summary"]["metrics_comparison"]
                    }
                })
                logger.info(f"A/B testing configured with {ab_config['traffic_split']:.1%} traffic split")
            except Exception as e:
                logger.error(f"Error configuring A/B testing: {str(e)}")

        # Monitor serving performance
        try:
            performance_metrics = optimizer.monitor_serving_performance(
                duration_seconds=30,
                metrics_interval_seconds=5,
                simulated_load=True,
                simulated_qps=10.0
            )

            results["performance_metrics"] = {
                "avg_latency_ms": performance_metrics["summary"]["avg_latency_ms"],
                "p95_latency_ms": performance_metrics["summary"]["p95_latency_ms"],
                "avg_qps": performance_metrics["summary"]["avg_qps"],
                "avg_error_rate": performance_metrics["summary"]["avg_error_rate"]
            }

            logger.info(f"Serving performance: {results['performance_metrics']['avg_latency_ms']:.2f}ms avg latency, {results['performance_metrics']['avg_qps']:.2f} QPS")
        except Exception as e:
            logger.error(f"Error monitoring serving performance: {str(e)}")

        # Save optimized endpoint
        optimized_dir = self.output_dir / "optimized_serving_endpoints" / endpoint_name
        optimized_dir.mkdir(exist_ok=True, parents=True)

        # Save results
        results_path = optimized_dir / "optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update optimization results
        self.optimization_results["serving_endpoints"][endpoint_name] = results

        logger.info(f"Serving endpoint {endpoint_name} optimization completed")
        return results

    def create_automated_optimization_pipeline(
        self,
        output_path: str = "automated_optimization_pipeline.py",
        components_to_optimize: Optional[Dict[str, List[str]]] = None,
        schedule: str = "daily"
    ) -> str:
        """
        Create an automated optimization pipeline script.

        Args:
            output_path: Path to save the pipeline script
            components_to_optimize: Dictionary of component types to component names
            schedule: Schedule for running the pipeline

        Returns:
            Path to the created pipeline script
        """
        logger.info("Creating automated optimization pipeline")

        # Use all discovered components if not specified
        if components_to_optimize is None:
            components_to_optimize = {
                "models": list(self.discovered_models.keys()),
                "feature_pipelines": list(self.discovered_feature_pipelines.keys()),
                "training_pipelines": list(self.discovered_training_pipelines.keys()),
                "serving_endpoints": list(self.discovered_serving_endpoints.keys())
            }

        # Create pipeline script
        script_content = self._generate_pipeline_script(components_to_optimize, schedule)

        # Save script
        script_path = self.output_dir / output_path
        with open(script_path, "w") as f:
            f.write(script_content)

        logger.info(f"Automated optimization pipeline created at {script_path}")
        return str(script_path)

    def _generate_pipeline_script(
        self,
        components_to_optimize: Dict[str, List[str]],
        schedule: str
    ) -> str:
        """Generate the automated optimization pipeline script."""
        # Create script header
        script = f'''"""
Automated ML Optimization Pipeline

This script automatically applies optimization techniques to ML components
in the forex trading platform.

Schedule: {schedule}
Created: {datetime.now().isoformat()}
"""

import os
import sys
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_optimization_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_optimization_pipeline")

# Add project root to path
project_root = "{self.project_root}"
sys.path.append(project_root)

# Import optimization tools
from ml_workbench_service.optimization.ml_pipeline_integrator import MLPipelineIntegrator
from ml_workbench_service.optimization.model_inference_optimizer import ModelInferenceOptimizer
from ml_workbench_service.optimization.feature_engineering_optimizer import FeatureEngineeringOptimizer
from ml_workbench_service.optimization.model_training_optimizer import ModelTrainingOptimizer
from ml_workbench_service.optimization.model_serving_optimizer import ModelServingOptimizer

def run_optimization_pipeline():
    """Run the automated optimization pipeline."""
    logger.info("Starting automated ML optimization pipeline")

    # Initialize pipeline integrator
    integrator = MLPipelineIntegrator(
        project_root=project_root,
        output_dir="./ml_pipeline_optimization"
    )

    # Discover ML components
    logger.info("Discovering ML components")
    discovered = integrator.discover_ml_components()

    # Track optimization results
    results = {{
        "timestamp": datetime.now().isoformat(),
        "components_optimized": 0,
        "optimizations_applied": 0,
        "errors": 0,
        "details": {{}}
    }}

'''

        # Add model optimization
        if "models" in components_to_optimize and components_to_optimize["models"]:
            models_str = ", ".join([f'"{m}"' for m in components_to_optimize["models"]])
            script += f'''
    # Optimize models
    logger.info("Optimizing models")
    models_to_optimize = [{models_str}]

    for model_name in models_to_optimize:
        try:
            logger.info(f"Optimizing model {{model_name}}")
            model_results = integrator.optimize_model(
                model_name=model_name,
                optimization_techniques=["quantization", "operator_fusion", "batch_inference"],
                target_device="cpu"
            )

            # Track results
            results["components_optimized"] += 1
            results["optimizations_applied"] += len(model_results.get("optimizations", []))
            results["details"][f"model_{{model_name}}"] = {{
                "status": "success",
                "optimizations": [opt["type"] for opt in model_results.get("optimizations", [])]
            }}

        except Exception as e:
            logger.error(f"Error optimizing model {{model_name}}: {{str(e)}}")
            results["errors"] += 1
            results["details"][f"model_{{model_name}}"] = {{
                "status": "error",
                "error": str(e)
            }}

'''

        # Add feature pipeline optimization
        if "feature_pipelines" in components_to_optimize and components_to_optimize["feature_pipelines"]:
            pipelines_str = ", ".join([f'"{p}"' for p in components_to_optimize["feature_pipelines"]])
            script += f'''
    # Optimize feature pipelines
    logger.info("Optimizing feature pipelines")
    pipelines_to_optimize = [{pipelines_str}]

    for pipeline_name in pipelines_to_optimize:
        try:
            logger.info(f"Optimizing feature pipeline {{pipeline_name}}")
            pipeline_results = integrator.optimize_feature_pipeline(
                pipeline_name=pipeline_name,
                optimization_techniques=["caching", "parallel", "incremental"]
            )

            # Track results
            results["components_optimized"] += 1
            results["optimizations_applied"] += len(pipeline_results.get("optimizations", []))
            results["details"][f"feature_pipeline_{{pipeline_name}}"] = {{
                "status": "success",
                "optimizations": [opt["type"] for opt in pipeline_results.get("optimizations", [])]
            }}

        except Exception as e:
            logger.error(f"Error optimizing feature pipeline {{pipeline_name}}: {{str(e)}}")
            results["errors"] += 1
            results["details"][f"feature_pipeline_{{pipeline_name}}"] = {{
                "status": "error",
                "error": str(e)
            }}

'''

        # Add training pipeline optimization
        if "training_pipelines" in components_to_optimize and components_to_optimize["training_pipelines"]:
            pipelines_str = ", ".join([f'"{p}"' for p in components_to_optimize["training_pipelines"]])
            script += f'''
    # Optimize training pipelines
    logger.info("Optimizing training pipelines")
    pipelines_to_optimize = [{pipelines_str}]

    for pipeline_name in pipelines_to_optimize:
        try:
            logger.info(f"Optimizing training pipeline {{pipeline_name}}")
            pipeline_results = integrator.optimize_training_pipeline(
                pipeline_name=pipeline_name,
                optimization_techniques=["mixed_precision", "gradient_accumulation", "distributed"]
            )

            # Track results
            results["components_optimized"] += 1
            results["optimizations_applied"] += len(pipeline_results.get("optimizations", []))
            results["details"][f"training_pipeline_{{pipeline_name}}"] = {{
                "status": "success",
                "optimizations": [opt["type"] for opt in pipeline_results.get("optimizations", [])]
            }}

        except Exception as e:
            logger.error(f"Error optimizing training pipeline {{pipeline_name}}: {{str(e)}}")
            results["errors"] += 1
            results["details"][f"training_pipeline_{{pipeline_name}}"] = {{
                "status": "error",
                "error": str(e)
            }}

'''

        # Add serving endpoint optimization
        if "serving_endpoints" in components_to_optimize and components_to_optimize["serving_endpoints"]:
            endpoints_str = ", ".join([f'"{e}"' for e in components_to_optimize["serving_endpoints"]])
            script += f'''
    # Optimize serving endpoints
    logger.info("Optimizing serving endpoints")
    endpoints_to_optimize = [{endpoints_str}]

    for endpoint_name in endpoints_to_optimize:
        try:
            logger.info(f"Optimizing serving endpoint {{endpoint_name}}")
            endpoint_results = integrator.optimize_serving_endpoint(
                endpoint_name=endpoint_name,
                optimization_techniques=["deployment_strategy", "auto_scaling", "ab_testing"]
            )

            # Track results
            results["components_optimized"] += 1
            results["optimizations_applied"] += len(endpoint_results.get("optimizations", []))
            results["details"][f"serving_endpoint_{{endpoint_name}}"] = {{
                "status": "success",
                "optimizations": [opt["type"] for opt in endpoint_results.get("optimizations", [])]
            }}

        except Exception as e:
            logger.error(f"Error optimizing serving endpoint {{endpoint_name}}: {{str(e)}}")
            results["errors"] += 1
            results["details"][f"serving_endpoint_{{endpoint_name}}"] = {{
                "status": "error",
                "error": str(e)
            }}

'''

        # Add results saving
        script += '''
    # Save results
    results_path = Path("./ml_pipeline_optimization/pipeline_run_results.json")
    results_path.parent.mkdir(exist_ok=True, parents=True)

    # Load previous results if they exist
    all_results = []
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                all_results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading previous results: {str(e)}")
            all_results = []

    # Add current results
    all_results.append(results)

    # Save updated results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Pipeline completed. Optimized {results['components_optimized']} components with {results['optimizations_applied']} optimizations. Errors: {results['errors']}")
    return results

if __name__ == "__main__":
    run_optimization_pipeline()
'''

        return script

    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate a report of all optimization results.

        Returns:
            Dictionary with optimization report
        """
        logger.info("Generating optimization report")

        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "models_optimized": len(self.optimization_results["models"]),
                "feature_pipelines_optimized": len(self.optimization_results["feature_pipelines"]),
                "training_pipelines_optimized": len(self.optimization_results["training_pipelines"]),
                "serving_endpoints_optimized": len(self.optimization_results["serving_endpoints"]),
                "total_components_optimized": (
                    len(self.optimization_results["models"]) +
                    len(self.optimization_results["feature_pipelines"]) +
                    len(self.optimization_results["training_pipelines"]) +
                    len(self.optimization_results["serving_endpoints"])
                )
            },
            "models": {},
            "feature_pipelines": {},
            "training_pipelines": {},
            "serving_endpoints": {}
        }

        # Add model optimization results
        for model_name, results in self.optimization_results["models"].items():
            # Extract key metrics
            optimizations = [opt["type"] for opt in results.get("optimizations", [])]

            # Extract performance improvement if available
            performance_improvement = None
            if "optimized_metrics" in results and "comparison" in results["optimized_metrics"]:
                batch_key = next(iter(results["optimized_metrics"]["comparison"]), None)
                if batch_key:
                    comparison = results["optimized_metrics"]["comparison"][batch_key]
                    throughput_improvement = comparison.get("throughput_samples_per_sec_improvement_pct", 0)
                    latency_improvement = comparison.get("avg_latency_ms_improvement_pct", 0)
                    performance_improvement = {
                        "throughput_improvement_pct": throughput_improvement,
                        "latency_improvement_pct": latency_improvement
                    }

            # Add to report
            report["models"][model_name] = {
                "optimizations": optimizations,
                "performance_improvement": performance_improvement
            }

        # Add feature pipeline optimization results
        for pipeline_name, results in self.optimization_results["feature_pipelines"].items():
            # Extract key metrics
            optimizations = [opt["type"] for opt in results.get("optimizations", [])]

            # Extract performance improvement if available
            performance_improvement = results.get("improvement_pct")

            # Add to report
            report["feature_pipelines"][pipeline_name] = {
                "optimizations": optimizations,
                "performance_improvement_pct": performance_improvement
            }

        # Add training pipeline optimization results
        for pipeline_name, results in self.optimization_results["training_pipelines"].items():
            # Extract key metrics
            optimizations = [opt["type"] for opt in results.get("optimizations", [])]

            # Extract benchmark results if available
            benchmark_results = None
            if "benchmark_results" in results:
                benchmark = results["benchmark_results"]
                benchmark_results = {
                    "total_time_seconds": benchmark.get("total_time_seconds", 0),
                    "samples_per_second": benchmark.get("samples_per_second", 0)
                }

            # Add to report
            report["training_pipelines"][pipeline_name] = {
                "optimizations": optimizations,
                "benchmark_results": benchmark_results
            }

        # Add serving endpoint optimization results
        for endpoint_name, results in self.optimization_results["serving_endpoints"].items():
            # Extract key metrics
            optimizations = [opt["type"] for opt in results.get("optimizations", [])]

            # Extract performance metrics if available
            performance_metrics = results.get("performance_metrics")

            # Add to report
            report["serving_endpoints"][endpoint_name] = {
                "optimizations": optimizations,
                "performance_metrics": performance_metrics
            }

        # Save report
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Optimization report generated and saved to {report_path}")
        return report
