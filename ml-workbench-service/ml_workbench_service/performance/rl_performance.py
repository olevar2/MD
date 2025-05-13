"""
Performance testing and optimization for reinforcement learning models.

This module provides tools for benchmarking, profiling, and optimizing RL models
for production deployment in the forex trading platform.
"""
import os
import gc
import time
import logging
import psutil
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
import json
import matplotlib.pyplot as plt
import threading
from collections import deque
import torch.quantization
import torch.nn.utils.prune as prune
from copy import deepcopy
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    from stable_baselines3.common import vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class InferenceBenchmark:
    """Class for benchmarking RL model inference performance."""

    def __init__(self, model, environment):
        """
        Initialize the inference benchmark.
        
        Args:
            model: The RL model to benchmark
            environment: The environment to use for testing
        """
        self.model = model
        self.environment = environment
        self.latency_stats = {}
        self.memory_stats = {}

    def run_inference_benchmark(self, num_episodes: int=10,
        steps_per_episode: int=100, warmup_episodes: int=2) ->Dict[str,
        Dict[str, float]]:
        """
        Run a comprehensive inference benchmark.
        
        Args:
            num_episodes: Number of episodes to benchmark
            steps_per_episode: Steps to run per episode
            warmup_episodes: Number of warmup episodes
            
        Returns:
            Dictionary of benchmark results
        """
        latencies = []
        memory_usages = []
        batch_sizes = [1, 4, 8, 16, 32]
        process = psutil.Process(os.getpid())
        logger.info(f'Running {warmup_episodes} warmup episodes')
        self._run_episode(steps_per_episode, warmup=True)
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        starting_memory = process.memory_info().rss / 1024 / 1024
        logger.info(
            f'Running {num_episodes} benchmark episodes with {steps_per_episode} steps each'
            )
        total_latency = 0
        total_inference_calls = 0
        peak_memory = starting_memory
        for episode in range(num_episodes):
            episode_stats = self._run_episode(steps_per_episode)
            latencies.extend(episode_stats['step_latencies'])
            total_latency += episode_stats['total_latency']
            total_inference_calls += episode_stats['inference_calls']
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usages.append(current_memory)
            peak_memory = max(peak_memory, current_memory)
            avg_step_latency = episode_stats['total_latency'] / max(1,
                episode_stats['inference_calls'])
            logger.info(
                f'Episode {episode + 1}/{num_episodes}: Avg step latency: {avg_step_latency * 1000:.2f}ms, Memory: {current_memory:.1f}MB'
                )
        latencies = sorted(latencies)
        avg_latency = sum(latencies) / max(1, len(latencies))
        p50_latency = latencies[int(len(latencies) * 0.5)] if latencies else 0
        p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0
        p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0
        avg_memory = sum(memory_usages) / max(1, len(memory_usages))
        memory_change = memory_usages[-1
            ] - starting_memory if memory_usages else 0
        self.latency_stats = {'average_ms': avg_latency * 1000, 'p50_ms': 
            p50_latency * 1000, 'p95_ms': p95_latency * 1000, 'p99_ms': 
            p99_latency * 1000, 'total_latency_s': total_latency,
            'inference_calls': total_inference_calls, 'throughput': 
            total_inference_calls / max(0.001, total_latency)}
        self.memory_stats = {'start_mb': starting_memory, 'end_mb': 
            memory_usages[-1] if memory_usages else starting_memory,
            'peak_mb': peak_memory, 'average_mb': avg_memory, 'change_mb':
            memory_change}
        batch_perf = {}
        if hasattr(self.model, 'predict_batch') or hasattr(self.model,
            'predict'
            ) and 'batch_size' in self.model.predict.__code__.co_varnames:
            batch_perf = self._benchmark_batch_sizes(batch_sizes)
        results = {'latency': self.latency_stats, 'memory': self.
            memory_stats, 'batch_performance': batch_perf, 'timestamp':
            datetime.now().isoformat(), 'model_type': str(type(self.model)),
            'environment_type': str(type(self.environment))}
        return results

    def _run_episode(self, steps_per_episode: int, warmup: bool=False) ->Dict[
        str, Any]:
        """Run a single episode and collect performance stats."""
        step_latencies = []
        start_time = time.time()
        inference_calls = 0
        observation = self.environment.reset()
        for step in range(steps_per_episode):
            inference_start = time.time()
            action, _states = self.model.predict(observation, deterministic
                =True)
            inference_time = time.time() - inference_start
            if not warmup:
                step_latencies.append(inference_time)
            observation, reward, done, info = self.environment.step(action)
            inference_calls += 1
            if done:
                observation = self.environment.reset()
        total_time = time.time() - start_time
        return {'step_latencies': step_latencies, 'total_latency':
            total_time, 'inference_calls': inference_calls}

    @with_exception_handling
    def _benchmark_batch_sizes(self, batch_sizes: List[int]) ->Dict[str,
        List[float]]:
        """Benchmark inference with different batch sizes."""
        batch_results = {'batch_size': [], 'latency_ms': [], 'throughput': []}
        observation = self.environment.reset()
        if isinstance(observation, np.ndarray):
            sample_shape = observation.shape
        else:
            logger.warning(
                'Cannot benchmark batch sizes with non-array observations')
            return batch_results
        for batch_size in batch_sizes:
            if len(sample_shape) == 1:
                batch = np.tile(observation, (batch_size, 1))
            else:
                batch = np.stack([observation] * batch_size)
            num_runs = 50
            latencies = []
            for _ in range(num_runs):
                start_time = time.time()
                try:
                    if hasattr(self.model, 'predict_batch'):
                        self.model.predict_batch(batch)
                    else:
                        self.model.predict(batch)
                except Exception as e:
                    logger.warning(f'Batch size {batch_size} failed: {e}')
                    break
                latency = time.time() - start_time
                latencies.append(latency)
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                throughput = batch_size / avg_latency
                batch_results['batch_size'].append(batch_size)
                batch_results['latency_ms'].append(avg_latency * 1000)
                batch_results['throughput'].append(throughput)
                logger.info(
                    f'Batch size {batch_size}: {avg_latency * 1000:.2f}ms, {throughput:.1f} obs/sec'
                    )
        return batch_results

    def generate_report(self, filename: Optional[str]=None) ->Dict[str, Any]:
        """
        Generate a performance report.
        
        Args:
            filename: Optional file to save the report to
            
        Returns:
            Dictionary with the report data
        """
        if not self.latency_stats:
            logger.warning(
                'No benchmark data available, run the benchmark first')
            return {}
        report = {'summary': {'average_inference_latency_ms': self.
            latency_stats.get('average_ms', 0), 'p95_latency_ms': self.
            latency_stats.get('p95_ms', 0), 'p99_latency_ms': self.
            latency_stats.get('p99_ms', 0), 'inference_throughput': self.
            latency_stats.get('throughput', 0), 'peak_memory_mb': self.
            memory_stats.get('peak_mb', 0), 'memory_growth_mb': self.
            memory_stats.get('change_mb', 0)}, 'requirements_check': {
            'meets_latency_requirement': self.latency_stats.get('p95_ms', 
            100) < 50, 'meets_throughput_requirement': self.latency_stats.
            get('throughput', 0) > 20}, 'detail': {'latency': self.
            latency_stats, 'memory': self.memory_stats}, 'timestamp':
            datetime.now().isoformat()}
        if filename:
            os.makedirs(os.path.dirname(os.path.abspath(filename)),
                exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f'Performance report saved to {filename}')
        return report


class TrainingBenchmark:
    """Class for benchmarking RL model training performance."""

    def __init__(self, model_cls, model_kwargs, env_cls, env_kwargs,
        metrics_dir: Optional[str]=None):
        """
        Initialize the training benchmark.
        
        Args:
            model_cls: RL model class constructor
            model_kwargs: Arguments for the model constructor
            env_cls: Environment class constructor
            env_kwargs: Arguments for the environment constructor
            metrics_dir: Directory to save metrics
        """
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        self.env_cls = env_cls
        self.env_kwargs = env_kwargs or {}
        self.metrics_dir = metrics_dir
        self.env = None
        self.model = None
        self._create_env_and_model()
        self.training_stats = {}

    @with_exception_handling
    def _create_env_and_model(self):
        """Create the environment and model for benchmarking."""
        try:
            self.env = self.env_cls(**self.env_kwargs)
            self.model = self.model_cls(self.env, **self.model_kwargs)
            logger.info(
                f'Created model {self.model_cls.__name__} and environment {self.env_cls.__name__}'
                )
        except Exception as e:
            logger.error(f'Error creating model or environment: {e}')
            raise

    @with_exception_handling
    def run_training_benchmark(self, total_timesteps: int=10000,
        log_interval: int=1000) ->Dict[str, Dict[str, float]]:
        """
        Run a training benchmark to measure training performance.
        
        Args:
            total_timesteps: Total timesteps to train for
            log_interval: How often to log stats
            
        Returns:
            Dictionary of benchmark results
        """
        if not SB3_AVAILABLE:
            logger.warning(
                'Stable Baselines 3 is not available. Cannot run training benchmark.'
                )
            return {}


        class BenchmarkCallback(BaseCallback):
    """
    BenchmarkCallback class that inherits from BaseCallback.
    
    Attributes:
        Add attributes here
    """


            def __init__(self, log_interval: int, verbose=0):
    """
      init  .
    
    Args:
        log_interval: Description of log_interval
        verbose: Description of verbose
    
    """

                super(BenchmarkCallback, self).__init__(verbose)
                self.log_interval = log_interval
                self.total_steps = total_timesteps
                self.throughputs = []
                self.timestamps = []
                self.step_timestamps = []
                self.gpu_utilizations = []
                self.gpu_memory_usages = []
                self.cpu_utilizations = []
                self.memory_usages = []
                self.last_time = time.time()
                self.last_timesteps = 0
                self.process = psutil.Process(os.getpid())

            @with_exception_handling
            def _on_step(self):
    """
     on step.
    
    """

                self.step_timestamps.append(time.time())
                if self.num_timesteps % self.log_interval == 0:
                    current_time = time.time()
                    elapsed = current_time - self.last_time
                    steps_taken = self.num_timesteps - self.last_timesteps
                    if elapsed > 0:
                        throughput = steps_taken / elapsed
                        self.throughputs.append(throughput)
                        self.timestamps.append(current_time)
                        self.cpu_utilizations.append(self.process.cpu_percent()
                            )
                        self.memory_usages.append(self.process.memory_info(
                            ).rss / 1024 / 1024)
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            try:
                                self.gpu_utilizations.append(torch.cuda.
                                    utilization())
                                self.gpu_memory_usages.append(torch.cuda.
                                    memory_allocated() / 1024 / 1024)
                            except:
                                self.gpu_utilizations.append(0)
                                self.gpu_memory_usages.append(0)
                        logger.info(
                            f'Step {self.num_timesteps}/{self.total_steps} ({self.num_timesteps / self.total_steps * 100:.1f}%) | Throughput: {throughput:.2f} steps/sec'
                            )
                    self.last_time = current_time
                    self.last_timesteps = self.num_timesteps
                return True
        logger.info(
            f'Starting training benchmark for {total_timesteps} timesteps')
        start_time = time.time()
        try:
            benchmark_callback = BenchmarkCallback(log_interval)
            self.model.learn(total_timesteps=total_timesteps, callback=
                benchmark_callback)
            training_time = time.time() - start_time
            avg_throughput = total_timesteps / max(0.001, training_time)
            avg_cpu = np.mean(benchmark_callback.cpu_utilizations
                ) if benchmark_callback.cpu_utilizations else 0
            avg_memory = np.mean(benchmark_callback.memory_usages
                ) if benchmark_callback.memory_usages else 0
            peak_memory = max(benchmark_callback.memory_usages
                ) if benchmark_callback.memory_usages else 0
            avg_gpu = np.mean(benchmark_callback.gpu_utilizations
                ) if benchmark_callback.gpu_utilizations else 0
            avg_gpu_memory = np.mean(benchmark_callback.gpu_memory_usages
                ) if benchmark_callback.gpu_memory_usages else 0
            self.training_stats = {'total_time_seconds': training_time,
                'total_timesteps': total_timesteps, 'average_throughput':
                avg_throughput, 'throughput_history': benchmark_callback.
                throughputs, 'timestamps': [(t - start_time) for t in
                benchmark_callback.timestamps], 'cpu_utilization_percent':
                avg_cpu, 'memory_usage_mb': {'average': avg_memory, 'peak':
                peak_memory}, 'gpu_stats': {'utilization_percent': avg_gpu,
                'memory_mb': avg_gpu_memory} if TORCH_AVAILABLE and torch.
                cuda.is_available() else {}, 'step_times': np.diff(
                benchmark_callback.step_timestamps).tolist() if len(
                benchmark_callback.step_timestamps) > 1 else []}
            logger.info(
                f'Training benchmark completed in {training_time:.2f} seconds')
            logger.info(f'Average throughput: {avg_throughput:.2f} steps/sec')
            logger.info(
                f'CPU utilization: {avg_cpu:.1f}%, Memory usage: {avg_memory:.1f}MB'
                )
            if TORCH_AVAILABLE and torch.cuda.is_available():
                logger.info(
                    f'GPU utilization: {avg_gpu:.1f}%, GPU memory: {avg_gpu_memory:.1f}MB'
                    )
            if self.metrics_dir:
                os.makedirs(self.metrics_dir, exist_ok=True)
                metrics_file = os.path.join(self.metrics_dir,
                    f'training_benchmark_{int(time.time())}.json')
                with open(metrics_file, 'w') as f:
                    json.dump(self.training_stats, f, indent=2)
                logger.info(f'Training metrics saved to {metrics_file}')
            return self.training_stats
        except Exception as e:
            logger.error(f'Error during training benchmark: {e}')
            raise
        finally:
            if self.env:
                try:
                    self.env.close()
                except:
                    pass
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_training_report(self, filename: Optional[str]=None) ->Dict[
        str, Any]:
        """
        Generate a training performance report.
        
        Args:
            filename: Optional file to save the report to
            
        Returns:
            Dictionary with the report data
        """
        if not self.training_stats:
            logger.warning(
                'No training benchmark data available, run the benchmark first'
                )
            return {}
        report = {'summary': {'total_training_time_seconds': self.
            training_stats.get('total_time_seconds', 0),
            'average_throughput': self.training_stats.get(
            'average_throughput', 0), 'cpu_utilization_percent': self.
            training_stats.get('cpu_utilization_percent', 0),
            'peak_memory_mb': self.training_stats.get('memory_usage_mb', {}
            ).get('peak', 0)}, 'requirements_check': {
            'meets_throughput_requirement': self.training_stats.get(
            'average_throughput', 0) > 100, 'meets_memory_requirement': 
            self.training_stats.get('memory_usage_mb', {}).get('peak', 0) <
            4000}, 'detail': self.training_stats, 'timestamp': datetime.now
            ().isoformat()}
        if filename:
            os.makedirs(os.path.dirname(os.path.abspath(filename)),
                exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f'Training report saved to {filename}')
        return report


class OptimizationTechniques:
    """Class implementing various optimization techniques for RL models."""

    @staticmethod
    @with_exception_handling
    def optimize_environment(env, vectorized: bool=True, num_envs: int=4):
        """
        Optimize environment for better performance.
        
        Args:
            env: Original environment
            vectorized: Whether to use vectorized environments
            num_envs: Number of environments to run in parallel
            
        Returns:
            Optimized environment
        """
        if not SB3_AVAILABLE:
            logger.warning(
                'Stable Baselines 3 is not available. Cannot optimize environment.'
                )
            return env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        if vectorized:
            try:
                if hasattr(env, 'num_envs'):
                    logger.info(
                        f'Environment is already vectorized with {env.num_envs} envs'
                        )
                    return env
                if num_envs > 1:
                    try:

                        def make_env():
    """
    Make env.
    
    """

                            return type(env)(**getattr(env, 'init_kwargs', {}))
                        vec_env = SubprocVecEnv([make_env for _ in range(
                            num_envs)])
                        logger.info(
                            f'Created SubprocVecEnv with {num_envs} parallel environments'
                            )
                        return vec_env
                    except Exception as e:
                        logger.warning(f'Failed to create SubprocVecEnv: {e}')
                        logger.info('Falling back to DummyVecEnv')
                        vec_env = DummyVecEnv([lambda : env])
                        return vec_env
                else:
                    vec_env = DummyVecEnv([lambda : env])
                    return vec_env
            except Exception as e:
                logger.warning(f'Failed to vectorize environment: {e}')
        return env

    @staticmethod
    def optimize_policy_network(policy, batch_norm: bool=True, dropout_rate:
        float=0.0):
        """
        Optimize the policy network architecture for better performance.
        
        Args:
            policy: The policy network to optimize
            batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate if using dropout
            
        Returns:
            Optimized policy
        """
        if not TORCH_AVAILABLE:
            logger.warning(
                'PyTorch is not available. Cannot optimize policy network.')
            return policy
        logger.info(
            'Policy network optimization is model-specific. Returning original policy.'
            )
        return policy

    @staticmethod
    @with_exception_handling
    def quantize_model(model: torch.nn.Module, quantization_type: str=
        'dynamic', backend: str='qnnpack', eval_fn: Optional[Callable]=None,
        calib_data: Optional[Any]=None):
        """
        Quantize the model to reduce memory usage and improve inference speed.

        Args:
            model: The PyTorch model (nn.Module) to quantize. Assumes model is in eval mode.
            quantization_type: Type of quantization ('dynamic', 'static', 'qat').
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM).
            eval_fn: Optional evaluation function (model, data) -> accuracy/metric for static/qat.
            calib_data: Optional calibration data loader/iterator for static/qat.

        Returns:
            Quantized model or original model if quantization fails.
        """
        if not TORCH_AVAILABLE:
            logger.warning('PyTorch is not available. Cannot quantize model.')
            return model
        if not isinstance(model, torch.nn.Module):
            logger.warning(
                f'Quantization requires a torch.nn.Module. Got {type(model)}. Skipping.'
                )
            return model
        try:
            original_device = next(model.parameters()).device
            model.cpu().eval()
            quantized_model = deepcopy(model)
            logger.info(
                f'Attempting {quantization_type} quantization with backend {backend}...'
                )
            if quantization_type == 'dynamic':
                quantized_model = torch.quantization.quantize_dynamic(
                    quantized_model, {torch.nn.Linear}, dtype=torch.qint8)
                logger.info('Dynamic quantization applied.')
            elif quantization_type in ['static', 'qat']:
                if eval_fn is None or calib_data is None:
                    logger.warning(
                        "Static/QAT quantization requires 'eval_fn' and 'calib_data'. Skipping."
                        )
                    return model
                quantized_model.qconfig = (torch.quantization.
                    get_default_qconfig(backend))
                logger.info('Preparing model for static quantization...')
                torch.quantization.prepare(quantized_model, inplace=True)
                logger.info('Calibrating model with provided data...')
                with torch.no_grad():
                    for data_batch in calib_data:
                        if isinstance(data_batch, (list, tuple)):
                            inputs = data_batch[0].cpu()
                        elif isinstance(data_batch, dict):
                            inputs = data_batch.get('observations',
                                data_batch.get('input')).cpu()
                        else:
                            inputs = data_batch.cpu()
                        quantized_model(inputs)
                logger.info('Calibration complete.')
                logger.info('Converting model to quantized version...')
                torch.quantization.convert(quantized_model, inplace=True)
                logger.info('Static quantization applied.')
                logger.info('Evaluating accuracy of static quantized model...')
                accuracy = eval_fn(quantized_model, calib_data)
                logger.info(f'Static quantized model accuracy: {accuracy}')
            else:
                logger.warning(
                    f'Unsupported quantization type: {quantization_type}. Returning original model.'
                    )
                return model
            logger.info('Quantization process completed.')
            return quantized_model
        except Exception as e:
            logger.error(f'Failed to quantize model: {e}', exc_info=True)
            return model

    @staticmethod
    @with_exception_handling
    def prune_model(model: torch.nn.Module, pruning_method: str=
        'l1_unstructured', amount: float=0.3, module_types=(torch.nn.Linear,
        torch.nn.Conv2d), eval_fn: Optional[Callable]=None, eval_data:
        Optional[Any]=None):
        """
        Apply pruning to the model to reduce size and potentially speed up inference.

        Args:
            model: The PyTorch model (nn.Module) to prune.
            pruning_method: Pruning method ('l1_unstructured', 'random_unstructured', 'global_unstructured').
            amount: Fraction of connections to prune (0.0 to 1.0) or number of connections for structured.
            module_types: Tuple of module types to apply pruning to.
            eval_fn: Optional evaluation function (model, data) -> accuracy/metric.
            eval_data: Optional data loader/iterator for evaluation.


        Returns:
            Pruned model (note: pruning is applied in-place, but the mask needs to be made permanent).
        """
        if not TORCH_AVAILABLE:
            logger.warning('PyTorch is not available. Cannot prune model.')
            return model
        if not isinstance(model, torch.nn.Module):
            logger.warning(
                f'Pruning requires a torch.nn.Module. Got {type(model)}. Skipping.'
                )
            return model
        try:
            logger.info(
                f'Applying {pruning_method} pruning with amount {amount}...')
            original_device = next(model.parameters()).device
            model.cpu().eval()
            parameters_to_prune = []
            for module in model.modules():
                if isinstance(module, module_types):
                    parameters_to_prune.append((module, 'weight'))
            if not parameters_to_prune:
                logger.warning(
                    f'No modules of types {module_types} found for pruning.')
                return model
            if pruning_method == 'l1_unstructured':
                prune.global_unstructured(parameters_to_prune,
                    pruning_method=prune.L1Unstructured, amount=amount)
            elif pruning_method == 'random_unstructured':
                prune.global_unstructured(parameters_to_prune,
                    pruning_method=prune.RandomUnstructured, amount=amount)
            else:
                logger.warning(
                    f'Unsupported pruning method: {pruning_method}. Skipping.')
                return model
            logger.info('Making pruning permanent...')
            for module, name in parameters_to_prune:
                if prune.is_pruned(module):
                    prune.remove(module, name)
            sparsity = OptimizationTechniques.calculate_sparsity(model)
            logger.info(f'Pruning applied. Global sparsity: {sparsity:.2%}')
            if eval_fn and eval_data:
                logger.info('Evaluating accuracy of pruned model...')
                accuracy = eval_fn(model, eval_data)
                logger.info(f'Pruned model accuracy: {accuracy}')
            model.to(original_device)
            return model
        except Exception as e:
            logger.error(f'Failed to prune model: {e}', exc_info=True)
            return model

    @staticmethod
    def calculate_sparsity(model: torch.nn.Module) ->float:
        """Calculate the global sparsity of the model."""
        total_params = 0
        zero_params = 0
        for module in model.modules():
            if hasattr(module, 'weight') and isinstance(module.weight,
                torch.Tensor):
                total_params += module.weight.nelement()
                zero_params += torch.sum(module.weight == 0).item()
            if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor
                ):
                total_params += module.bias.nelement()
                zero_params += torch.sum(module.bias == 0).item()
        return zero_params / total_params if total_params > 0 else 0.0

    @staticmethod
    def benchmark_optimized_model(original_model, optimized_model,
        environment, eval_fn: Optional[Callable]=None, eval_data: Optional[
        Any]=None, num_episodes: int=10, steps_per_episode: int=100):
        """
        Benchmark an optimized model (quantized/pruned) against the original.

        Args:
            original_model: The original, unoptimized model.
            optimized_model: The optimized model (quantized or pruned).
            environment: The environment for inference benchmarking.
            eval_fn: Optional evaluation function (model, data) -> accuracy/metric.
            eval_data: Optional data loader/iterator for evaluation.
            num_episodes: Number of episodes for inference benchmark.
            steps_per_episode: Steps per episode for inference benchmark.

        Returns:
            Dictionary containing comparison results.
        """
        results = {'original': {}, 'optimized': {}, 'comparison': {}}
        logger.info('Benchmarking ORIGINAL model...')
        orig_benchmark = InferenceBenchmark(original_model, environment)
        orig_perf = orig_benchmark.run_inference_benchmark(num_episodes,
            steps_per_episode)
        results['original']['performance'] = orig_perf
        if eval_fn and eval_data:
            results['original']['accuracy'] = eval_fn(original_model, eval_data
                )
        logger.info('Benchmarking OPTIMIZED model...')
        opt_benchmark = InferenceBenchmark(optimized_model, environment)
        opt_perf = opt_benchmark.run_inference_benchmark(num_episodes,
            steps_per_episode)
        results['optimized']['performance'] = opt_perf
        if eval_fn and eval_data:
            results['optimized']['accuracy'] = eval_fn(optimized_model,
                eval_data)
        orig_latency = orig_perf.get('latency', {}).get('average_ms', 0)
        opt_latency = opt_perf.get('latency', {}).get('average_ms', 0)
        orig_throughput = orig_perf.get('latency', {}).get('throughput', 0)
        opt_throughput = opt_perf.get('latency', {}).get('throughput', 0)
        orig_peak_mem = orig_perf.get('memory', {}).get('peak_mb', 0)
        opt_peak_mem = opt_perf.get('memory', {}).get('peak_mb', 0)
        results['comparison']['latency_reduction_percent'] = (1 - 
            opt_latency / orig_latency) * 100 if orig_latency else 0
        results['comparison']['throughput_increase_percent'] = (
            opt_throughput / orig_throughput - 1
            ) * 100 if orig_throughput else 0
        results['comparison']['peak_memory_reduction_percent'] = (1 - 
            opt_peak_mem / orig_peak_mem) * 100 if orig_peak_mem else 0
        if 'accuracy' in results['original'] and 'accuracy' in results[
            'optimized']:
            orig_acc = results['original']['accuracy']
            opt_acc = results['optimized']['accuracy']
            if isinstance(orig_acc, dict) and isinstance(opt_acc, dict):
                primary_metric = next(iter(orig_acc))
                results['comparison']['accuracy_change'] = opt_acc.get(
                    primary_metric, 0) - orig_acc.get(primary_metric, 0)
            elif isinstance(orig_acc, (int, float)) and isinstance(opt_acc,
                (int, float)):
                results['comparison']['accuracy_change'] = opt_acc - orig_acc
            else:
                results['comparison']['accuracy_change'
                    ] = 'N/A (Incompatible formats)'
        logger.info(f'Comparison Summary:')
        logger.info(
            f"  Latency Reduction: {results['comparison']['latency_reduction_percent']:.2f}%"
            )
        logger.info(
            f"  Throughput Increase: {results['comparison']['throughput_increase_percent']:.2f}%"
            )
        logger.info(
            f"  Peak Memory Reduction: {results['comparison']['peak_memory_reduction_percent']:.2f}%"
            )
        if 'accuracy_change' in results['comparison']:
            logger.info(
                f"  Accuracy Change: {results['comparison']['accuracy_change']}"
                )
        return results

    @staticmethod
    def optimize_batch_inference(model, batch_size: int=16):
        """
        Optimize the model for efficient batch inference.
        
        Args:
            model: The model to optimize
            batch_size: Target batch size for optimization
            
        Returns:
            Optimized model for batch inference
        """
        if not hasattr(model, 'predict_batch'):
            original_predict = model.predict

            def predict_batch(self, observations, deterministic=False):
    """
    Predict batch.
    
    Args:
        observations: Description of observations
        deterministic: Description of deterministic
    
    """

                if isinstance(observations, np.ndarray) and len(observations
                    .shape) > 1:
                    batch_size = observations.shape[0]
                    results = [original_predict(observations[i:i + 1],
                        deterministic=deterministic) for i in range(batch_size)
                        ]
                    actions = np.vstack([r[0] for r in results])
                    states = [r[1] for r in results] if results[0][1
                        ] is not None else None
                    return actions, states
                else:
                    return original_predict(observations, deterministic=
                        deterministic)
            model.predict_batch = predict_batch.__get__(model)
            logger.info(
                f'Added batch prediction method optimized for batch size {batch_size}'
                )
        return model


class PerformanceTester:
    """Class for comprehensive performance testing of RL models."""

    def __init__(self, model=None, model_cls=None, model_kwargs=None,
        environment=None, env_cls=None, env_kwargs=None, output_dir:
        Optional[str]=None):
        """
        Initialize the performance tester.
        
        Args:
            model: Pre-initialized model (or will be created if model_cls is provided)
            model_cls: Model class constructor
            model_kwargs: Arguments for the model constructor
            environment: Pre-initialized environment (or will be created if env_cls is provided)
            env_cls: Environment class constructor
            env_kwargs: Arguments for the environment constructor
            output_dir: Directory to save performance reports
        """
        self.model = model
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        self.environment = environment
        self.env_cls = env_cls
        self.env_kwargs = env_kwargs or {}
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if self.model is None and self.model_cls is not None:
            if self.environment is None and self.env_cls is not None:
                self.environment = self.env_cls(**self.env_kwargs)
            if self.environment is not None:
                self.model = self.model_cls(self.environment, **self.
                    model_kwargs)
        self.inference_benchmark = None
        self.training_benchmark = None

    def run_comprehensive_test(self) ->Dict[str, Any]:
        """
        Run a comprehensive performance test including inference and training.
        
        Returns:
            Dictionary with all test results
        """
        results = {'timestamp': datetime.now().isoformat(), 'system_info':
            self._get_system_info()}
        if self.model is not None and self.environment is not None:
            logger.info('Running inference benchmark...')
            self.inference_benchmark = InferenceBenchmark(self.model, self.
                environment)
            inference_results = (self.inference_benchmark.
                run_inference_benchmark())
            results['inference'] = inference_results
            if self.output_dir:
                report_path = os.path.join(self.output_dir,
                    'inference_report.json')
                self.inference_benchmark.generate_report(report_path)
        if self.model_cls is not None and (self.env_cls is not None or self
            .environment is not None):
            logger.info('Running training benchmark...')
            self.training_benchmark = TrainingBenchmark(self.model_cls,
                self.model_kwargs, self.env_cls or type(self.environment),
                self.env_kwargs, metrics_dir=self.output_dir)
            training_results = self.training_benchmark.run_training_benchmark()
            results['training'] = training_results
            if self.output_dir:
                report_path = os.path.join(self.output_dir,
                    'training_report.json')
                self.training_benchmark.generate_training_report(report_path)
        results['optimization'] = self._run_optimization_demo()
        if self.output_dir:
            report_path = os.path.join(self.output_dir,
                'comprehensive_performance_report.json')
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(
                f'Comprehensive performance report saved to {report_path}')
        return results

    @with_exception_handling
    def _get_system_info(self) ->Dict[str, Any]:
        """Get system hardware and software information."""
        info = {'cpu': {'count': os.cpu_count(), 'brand': 'Unknown'},
            'memory': {'total_gb': psutil.virtual_memory().total / 1024 ** 3}}
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        info['cpu']['brand'] = line.split(':')[1].strip()
                        break
        except:
            pass
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info['gpu'] = {'count': torch.cuda.device_count(), 'name':
                torch.cuda.get_device_name(0), 'memory_gb': torch.cuda.
                get_device_properties(0).total_memory / 1024 ** 3}
        return info

    @with_exception_handling
    def _run_optimization_demo(self) ->Dict[str, Any]:
        """Run a demonstration of optimization techniques."""
        if self.model is None or self.environment is None:
            return {'status': 'skipped', 'reason':
                'Model or environment not available'}
        results = {}
        try:
            logger.info('Testing vectorized environment optimization...')
            start_time = time.time()
            original_env = self.environment
            optimized_env = OptimizationTechniques.optimize_environment(
                original_env, num_envs=4)
            original_time = 0
            optimized_time = 0
            if hasattr(original_env, 'reset'):
                obs = original_env.reset()
                start = time.time()
                for i in range(100):
                    action = np.zeros(original_env.action_space.shape)
                    obs, _, _, _ = original_env.step(action)
                original_time = time.time() - start
            if hasattr(optimized_env, 'reset'):
                obs = optimized_env.reset()
                start = time.time()
                for i in range(100):
                    if hasattr(optimized_env, 'num_envs'):
                        action = np.zeros((optimized_env.num_envs,) +
                            original_env.action_space.shape)
                    else:
                        action = np.zeros(original_env.action_space.shape)
                    obs, _, _, _ = optimized_env.step(action)
                optimized_time = time.time() - start
            speedup = original_time / max(0.001, optimized_time)
            results['vectorized_env'] = {'original_time_ms': original_time *
                1000, 'optimized_time_ms': optimized_time * 1000, 'speedup':
                speedup, 'status': 'success' if speedup > 1 else
                'no improvement'}
            logger.info(f'Environment optimization: {speedup:.2f}x speedup')
        except Exception as e:
            logger.error(f'Error testing environment optimization: {e}')
            results['vectorized_env'] = {'status': 'error', 'error': str(e)}
        try:
            logger.info('Testing batch inference optimization...')
            optimized_model = OptimizationTechniques.optimize_batch_inference(
                self.model)
            batch_sizes = [1, 4, 16, 32]
            batch_results = {'batch_size': [], 'latency_ms': [],
                'throughput': []}
            obs = self.environment.reset()
            obs_shape = obs.shape if isinstance(obs, np.ndarray) else (1,)
            for batch_size in batch_sizes:
                if len(obs_shape) == 1:
                    batch = np.zeros((batch_size, *obs_shape))
                else:
                    batch = np.zeros((batch_size, *obs_shape))
                try:
                    start = time.time()
                    for i in range(10):
                        optimized_model.predict_batch(batch)
                    latency = (time.time() - start) / 10
                    throughput = batch_size / latency
                    batch_results['batch_size'].append(batch_size)
                    batch_results['latency_ms'].append(latency * 1000)
                    batch_results['throughput'].append(throughput)
                except Exception as e:
                    logger.warning(
                        f'Error testing batch size {batch_size}: {e}')
            results['batch_inference'] = batch_results
        except Exception as e:
            logger.error(f'Error testing batch inference optimization: {e}')
            results['batch_inference'] = {'status': 'error', 'error': str(e)}
        return results

    @with_exception_handling
    def cleanup(self):
        """Clean up resources."""
        if self.environment:
            try:
                self.environment.close()
            except:
                pass
        self.environment = None
        self.model = None
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    import gymnasium as gym
    from stable_baselines3 import PPO, A2C
    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env)
    tester = PerformanceTester(model=model, environment=env, model_cls=PPO,
        model_kwargs={'policy': 'MlpPolicy'}, env_cls=gym.make, env_kwargs=
        {'id': 'CartPole-v1'}, output_dir='./performance_reports')
    results = tester.run_comprehensive_test()
    tester.cleanup()
