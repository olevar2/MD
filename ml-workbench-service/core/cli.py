"""
ML Optimization CLI

Command-line interface for optimizing machine learning models and pipelines.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from core.model_inference_optimizer import ModelInferenceOptimizer
from core.feature_engineering_optimizer import FeatureEngineeringOptimizer
from core.model_training_optimizer import ModelTrainingOptimizer
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

def setup_parser() ->argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(description='ML Optimization CLI')
    subparsers = parser.add_subparsers(dest='command', help=
        'Optimization command')
    inference_parser = subparsers.add_parser('inference', help=
        'Optimize model inference')
    inference_parser.add_argument('--model-path', required=True, help=
        'Path to the model file')
    inference_parser.add_argument('--framework', choices=['tensorflow',
        'pytorch', 'onnx'], default='tensorflow', help='ML framework')
    inference_parser.add_argument('--optimization-level', choices=['speed',
        'balanced', 'memory'], default='balanced', help='Optimization priority'
        )
    inference_parser.add_argument('--device', choices=['cpu', 'gpu'],
        default='cpu', help='Target device')
    inference_parser.add_argument('--quantize', action='store_true', help=
        'Apply quantization')
    inference_parser.add_argument('--quantization-type', choices=['int8',
        'float16', 'dynamic'], default='int8', help='Quantization type')
    inference_parser.add_argument('--fusion', action='store_true', help=
        'Apply operator fusion')
    inference_parser.add_argument('--batch-inference', action='store_true',
        help='Configure batch inference')
    inference_parser.add_argument('--batch-size', type=int, default=32,
        help='Batch size for inference')
    inference_parser.add_argument('--input-data', help=
        'Path to sample input data for benchmarking')
    inference_parser.add_argument('--output-dir', default=
        './optimization_output', help='Directory to save optimization results')
    feature_parser = subparsers.add_parser('features', help=
        'Optimize feature engineering')
    feature_parser.add_argument('--data-path', required=True, help=
        'Path to input data')
    feature_parser.add_argument('--feature-functions', required=True, help=
        'Python module containing feature functions')
    feature_parser.add_argument('--cache-dir', default='./feature_cache',
        help='Directory to cache computed features')
    feature_parser.add_argument('--max-cache-size', type=int, default=1024,
        help='Maximum cache size in MB')
    feature_parser.add_argument('--parallel', action='store_true', help=
        'Use parallel feature computation')
    feature_parser.add_argument('--n-jobs', type=int, default=-1, help=
        'Number of parallel jobs (-1 for all cores)')
    feature_parser.add_argument('--benchmark', action='store_true', help=
        'Benchmark feature pipeline performance')
    feature_parser.add_argument('--output-dir', default=
        './optimization_output', help='Directory to save optimization results')
    training_parser = subparsers.add_parser('training', help=
        'Optimize model training')
    training_parser.add_argument('--model-path', required=True, help=
        'Path to the model file')
    training_parser.add_argument('--data-path', required=True, help=
        'Path to training data')
    training_parser.add_argument('--framework', choices=['tensorflow',
        'pytorch'], default='tensorflow', help='ML framework')
    training_parser.add_argument('--device', choices=['auto', 'cpu', 'gpu',
        'tpu'], default='auto', help='Target device')
    training_parser.add_argument('--mixed-precision', action='store_true',
        help='Use mixed precision training')
    training_parser.add_argument('--precision', choices=['float16',
        'bfloat16'], default='float16', help='Precision format')
    training_parser.add_argument('--gradient-accumulation', action=
        'store_true', help='Use gradient accumulation')
    training_parser.add_argument('--accumulation-steps', type=int, default=
        1, help='Number of gradient accumulation steps')
    training_parser.add_argument('--distributed', action='store_true', help
        ='Use distributed training')
    training_parser.add_argument('--strategy', default='mirrored', help=
        'Distributed training strategy')
    training_parser.add_argument('--batch-size', type=int, default=32, help
        ='Batch size for training')
    training_parser.add_argument('--epochs', type=int, default=1, help=
        'Number of training epochs')
    training_parser.add_argument('--output-dir', default=
        './optimization_output', help='Directory to save optimization results')
    return parser


@with_exception_handling
def optimize_inference(args: argparse.Namespace) ->Dict[str, Any]:
    """Run model inference optimization."""
    logger.info(f'Optimizing model inference for {args.model_path}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    optimizer = ModelInferenceOptimizer(model_path=args.model_path,
        framework=args.framework, optimization_level=args.
        optimization_level, device=args.device, cache_dir=str(output_dir /
        'cache'))
    results = {'model_path': args.model_path, 'framework': args.framework,
        'device': args.device, 'optimizations': []}
    input_data = None
    if args.input_data:
        try:
            import numpy as np
            input_data = np.load(args.input_data)
            baseline_metrics = optimizer.benchmark_baseline(input_data=
                input_data, batch_sizes=[1, args.batch_size])
            results['baseline_metrics'] = baseline_metrics
        except Exception as e:
            logger.error(f'Error loading input data: {str(e)}')
    if args.quantize:
        try:
            quantized_model, quantization_metadata = optimizer.quantize_model(
                quantization_type=args.quantization_type)
            results['optimizations'].append({'type': 'quantization',
                'metadata': quantization_metadata})
            logger.info(
                f"Quantization applied: {quantization_metadata.get('size_reduction_percent', 0):.2f}% size reduction"
                )
        except Exception as e:
            logger.error(f'Error during quantization: {str(e)}')
    if args.fusion:
        try:
            fused_model, fusion_metadata = optimizer.apply_operator_fusion()
            results['optimizations'].append({'type': 'operator_fusion',
                'metadata': fusion_metadata})
            logger.info(
                f"Operator fusion applied in {fusion_metadata.get('fusion_time_seconds', 0):.2f} seconds"
                )
        except Exception as e:
            logger.error(f'Error during operator fusion: {str(e)}')
    if args.batch_inference:
        try:
            batch_config = optimizer.configure_batch_inference(
                optimal_batch_size=args.batch_size)
            results['optimizations'].append({'type': 'batch_inference',
                'config': batch_config})
            logger.info(
                f"Batch inference configured with batch size {batch_config_manager.get('optimal_batch_size', args.batch_size)}"
                )
        except Exception as e:
            logger.error(f'Error configuring batch inference: {str(e)}')
    if input_data is not None:
        try:
            optimized_metrics = optimizer.benchmark_optimized(input_data=
                input_data, batch_sizes=[1, args.batch_size])
            results['optimized_metrics'] = optimized_metrics
            if 'comparison' in optimized_metrics:
                batch_key = f'batch_{args.batch_size}'
                if batch_key in optimized_metrics['comparison']:
                    comparison = optimized_metrics['comparison'][batch_key]
                    throughput_improvement = comparison.get(
                        'throughput_samples_per_sec_improvement_pct', 0)
                    latency_improvement = comparison.get(
                        'avg_latency_ms_improvement_pct', 0)
                    logger.info(
                        f'Performance improvement: {throughput_improvement:.2f}% throughput, {latency_improvement:.2f}% latency'
                        )
        except Exception as e:
            logger.error(f'Error during optimized benchmarking: {str(e)}')
    results_path = output_dir / 'inference_optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Optimization results saved to {results_path}')
    return results


@with_exception_handling
def optimize_features(args: argparse.Namespace) ->Dict[str, Any]:
    """Run feature engineering optimization."""
    logger.info(f'Optimizing feature engineering for {args.data_path}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('feature_module',
            args.feature_functions)
        feature_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_module)
        feature_funcs = []
        for name in dir(feature_module):
            if name.startswith('feature_') and callable(getattr(
                feature_module, name)):
                feature_funcs.append(getattr(feature_module, name))
        if not feature_funcs:
            logger.warning(
                f'No feature functions found in {args.feature_functions}')
            return {'error': 'No feature functions found'}
        logger.info(f'Found {len(feature_funcs)} feature functions')
    except Exception as e:
        logger.error(f'Error loading feature functions: {str(e)}')
        return {'error': str(e)}
    try:
        import pandas as pd
        if args.data_path.endswith('.csv'):
            data = pd.read_csv(args.data_path)
        elif args.data_path.endswith('.parquet'):
            data = pd.read_parquet(args.data_path)
        elif args.data_path.endswith('.json'):
            data = pd.read_json(args.data_path)
        else:
            logger.error(f'Unsupported data format: {args.data_path}')
            return {'error': 'Unsupported data format'}
        logger.info(f'Loaded data with shape {data.shape}')
    except Exception as e:
        logger.error(f'Error loading data: {str(e)}')
        return {'error': str(e)}
    optimizer = FeatureEngineeringOptimizer(cache_dir=args.cache_dir,
        max_cache_size_mb=args.max_cache_size, n_jobs=args.n_jobs)
    results = {'data_path': args.data_path, 'data_shape': data.shape,
        'n_feature_functions': len(feature_funcs), 'optimizations': []}
    if args.benchmark:
        try:
            benchmark_results = optimizer.benchmark_feature_pipeline(data=
                data, feature_funcs=feature_funcs, use_cache=True,
                use_parallel=args.parallel, n_runs=3)
            results['benchmark'] = benchmark_results
            avg_time = benchmark_results['overall']['avg_time']
            logger.info(
                f'Benchmark completed: {avg_time:.2f} seconds average time')
            for func_name, metrics in benchmark_results['per_feature'].items():
                if 'avg_time' in metrics:
                    logger.info(
                        f"  {func_name}: {metrics['avg_time']:.4f}s ({metrics.get('pct_of_total', 0):.1f}% of total)"
                        )
        except Exception as e:
            logger.error(f'Error during benchmarking: {str(e)}')
    if args.parallel:
        try:
            import time
            start_time = time.time()
            features, metadata = optimizer.parallel_feature_computation(data
                =data, feature_funcs=feature_funcs, use_cache=True)
            elapsed = time.time() - start_time
            results['optimizations'].append({'type': 'parallel_computation',
                'metadata': metadata})
            success_rate = metadata.get('success_rate', 0) * 100
            total_time = metadata.get('total_time', 0)
            logger.info(
                f'Parallel computation completed: {success_rate:.1f}% success rate, {total_time:.2f} seconds'
                )
            if 'cache_hits' in metadata:
                cache_hits = sum(1 for hit in metadata['cache_hits'].values
                    () if hit)
                cache_hit_rate = cache_hits / len(metadata['cache_hits']
                    ) if metadata['cache_hits'] else 0
                logger.info(f'Cache hit rate: {cache_hit_rate:.1%}')
        except Exception as e:
            logger.error(f'Error during parallel computation: {str(e)}')
    results_path = output_dir / 'feature_optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Optimization results saved to {results_path}')
    return results


@with_exception_handling
def optimize_training(args: argparse.Namespace) ->Dict[str, Any]:
    """Run model training optimization."""
    logger.info(f'Optimizing model training for {args.model_path}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    model = None
    try:
        if args.framework == 'tensorflow':
            import tensorflow as tf
            model = tf.keras.models.load_model(args.model_path)
        elif args.framework == 'pytorch':
            import torch
            model = torch.load(args.model_path)
        else:
            logger.error(f'Unsupported framework: {args.framework}')
            return {'error': 'Unsupported framework'}
        logger.info(f'Loaded model from {args.model_path}')
    except Exception as e:
        logger.error(f'Error loading model: {str(e)}')
        return {'error': str(e)}
    train_dataset = None
    try:
        if args.framework == 'tensorflow':
            import tensorflow as tf
            if args.data_path.endswith('.tfrecord'):
                train_dataset = tf.data.TFRecordDataset(args.data_path)
            else:
                import numpy as np
                data = np.load(args.data_path)
                if isinstance(data, dict):
                    train_dataset = tf.data.Dataset.from_tensor_slices((
                        data['x'], data['y']))
                else:
                    train_dataset = tf.data.Dataset.from_tensor_slices(data)
        elif args.framework == 'pytorch':
            import torch
            if args.data_path.endswith('.pt'):
                train_dataset = torch.load(args.data_path)
            else:
                import numpy as np
                data = np.load(args.data_path)


                class NumpyDataset(torch.utils.data.Dataset):
    """
    NumpyDataset class that inherits from torch.utils.data.Dataset.
    
    Attributes:
        Add attributes here
    """


                    def __init__(self, data):
    """
      init  .
    
    Args:
        data: Description of data
    
    """

                        if isinstance(data, dict):
                            self.x = torch.from_numpy(data['x'])
                            self.y = torch.from_numpy(data['y'])
                            self.has_labels = True
                        else:
                            self.x = torch.from_numpy(data)
                            self.has_labels = False

                    def __len__(self):
    """
      len  .
    
    """

                        return len(self.x)

                    def __getitem__(self, idx):
    """
      getitem  .
    
    Args:
        idx: Description of idx
    
    """

                        if self.has_labels:
                            return self.x[idx], self.y[idx]
                        else:
                            return self.x[idx]
                train_dataset = NumpyDataset(data)
        logger.info(f'Loaded training data from {args.data_path}')
    except Exception as e:
        logger.error(f'Error loading training data: {str(e)}')
        return {'error': str(e)}
    optimizer = ModelTrainingOptimizer(model=model, framework=args.
        framework, device=args.device, output_dir=str(output_dir))
    results = {'model_path': args.model_path, 'data_path': args.data_path,
        'framework': args.framework, 'device': optimizer.device,
        'optimizations': []}
    if args.mixed_precision:
        try:
            mp_config = optimizer.configure_mixed_precision(enabled=True,
                precision=args.precision)
            results['optimizations'].append({'type': 'mixed_precision',
                'config': mp_config})
            logger.info(
                f'Mixed precision configured with {args.precision} precision')
        except Exception as e:
            logger.error(f'Error configuring mixed precision: {str(e)}')
    if args.gradient_accumulation:
        try:
            ga_config = optimizer.configure_gradient_accumulation(
                accumulation_steps=args.accumulation_steps)
            results['optimizations'].append({'type':
                'gradient_accumulation', 'config': ga_config})
            logger.info(
                f'Gradient accumulation configured with {args.accumulation_steps} steps'
                )
        except Exception as e:
            logger.error(f'Error configuring gradient accumulation: {str(e)}')
    if args.distributed:
        try:
            dist_config = optimizer.configure_distributed_training(strategy
                =args.strategy)
            results['optimizations'].append({'type': 'distributed_training',
                'config': dist_config})
            logger.info(
                f'Distributed training configured with {args.strategy} strategy'
                )
        except Exception as e:
            logger.error(f'Error configuring distributed training: {str(e)}')
    try:
        benchmark_results = optimizer.benchmark_training(model=model,
            train_dataset=train_dataset, batch_size=args.batch_size,
            num_epochs=args.epochs, mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.accumulation_steps if args.
            gradient_accumulation else 1, distributed=args.distributed)
        results['benchmark'] = benchmark_results
        total_time = benchmark_results.get('total_time_seconds', 0)
        samples_per_second = benchmark_results.get('samples_per_second', 0)
        logger.info(f'Training benchmark completed in {total_time:.2f} seconds'
            )
        logger.info(f'Performance: {samples_per_second:.2f} samples/second')
    except Exception as e:
        logger.error(f'Error during training benchmark: {str(e)}')
    results_path = output_dir / 'training_optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Optimization results saved to {results_path}')
    return results


def main():
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    if args.command == 'inference':
        optimize_inference(args)
    elif args.command == 'features':
        optimize_features(args)
    elif args.command == 'training':
        optimize_training(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
