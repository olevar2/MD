"""
Model Training Optimizer

This module provides tools for optimizing ML model training performance through
techniques like mixed precision, gradient accumulation, and distributed training.

It includes:
- Mixed precision training
- Gradient accumulation
- Distributed training configuration
- Training performance benchmarking
"""

import logging
import time
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
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

logger = logging.getLogger(__name__)

class ModelTrainingOptimizer:
    """
    Optimizes ML model training performance.

    This class provides methods for:
    - Configuring mixed precision training
    - Setting up gradient accumulation
    - Configuring distributed training
    - Benchmarking training performance
    """

    def __init__(
        self,
        model: Any = None,
        framework: str = "tensorflow",  # "tensorflow", "pytorch"
        device: str = "auto",  # "auto", "cpu", "gpu", "tpu"
        output_dir: str = "./training_output"
    ):
        """
        Initialize the model training optimizer.

        Args:
            model: Model object to optimize
            framework: ML framework the model belongs to
            device: Target device for training
            output_dir: Directory to save training outputs
        """
        self.model = model
        self.framework = framework.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Determine device
        self.device = self._determine_device(device)

        # Validate framework availability
        self._validate_framework()

        # Performance tracking
        self.baseline_metrics = {}
        self.optimized_metrics = {}

    def _validate_framework(self):
        """Validate that the requested framework is available."""
        if self.framework == "tensorflow" and not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install it to use this framework.")
        elif self.framework == "pytorch" and not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it to use this framework.")

    def _determine_device(self, device: str) -> str:
        """Determine the appropriate device for training."""
        if device.lower() == "auto":
            # Auto-detect available devices
            if self.framework == "tensorflow":
                if tf.config.list_physical_devices('GPU'):
                    return "gpu"
                elif tf.config.list_physical_devices('TPU'):
                    return "tpu"
                else:
                    return "cpu"
            elif self.framework == "pytorch":
                if torch.cuda.is_available():
                    return "gpu"
                else:
                    return "cpu"
            else:
                return "cpu"
        else:
            return device.lower()

    def configure_mixed_precision(
        self,
        enabled: bool = True,
        precision: str = "float16",  # "float16", "bfloat16"
        dynamic_loss_scaling: bool = True
    ) -> Dict[str, Any]:
        """
        Configure mixed precision training.

        Args:
            enabled: Whether to enable mixed precision
            precision: Precision format to use
            dynamic_loss_scaling: Whether to use dynamic loss scaling

        Returns:
            Dictionary with mixed precision configuration
        """
        logger.info(f"Configuring mixed precision training (enabled={enabled}, precision={precision})")

        if not enabled:
            return {"enabled": False}

        if self.device == "cpu" and precision == "float16":
            logger.warning("float16 precision not well supported on CPU. Consider using bfloat16 or disabling mixed precision.")

        config = {
            "enabled": enabled,
            "precision": precision,
            "dynamic_loss_scaling": dynamic_loss_scaling
        }

        try:
            if self.framework == "tensorflow":
                self._configure_tensorflow_mixed_precision(precision)
                config["framework_specific"] = {"tensorflow_policy": precision}
            elif self.framework == "pytorch":
                scaler = self._configure_pytorch_mixed_precision(precision, dynamic_loss_scaling)
                config["framework_specific"] = {"pytorch_scaler": scaler is not None}

            logger.info("Mixed precision configured successfully")

        except Exception as e:
            logger.error(f"Error configuring mixed precision: {str(e)}")
            config["error"] = str(e)
            config["enabled"] = False

        return config

    def _configure_tensorflow_mixed_precision(self, precision: str):
        """Configure mixed precision for TensorFlow."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        if precision == "float16":
            policy = 'mixed_float16'
        elif precision == "bfloat16":
            policy = 'mixed_bfloat16'
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        tf.keras.mixed_precision.set_global_policy(policy)

    def _configure_pytorch_mixed_precision(self, precision: str, dynamic_loss_scaling: bool):
        """Configure mixed precision for PyTorch."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")

        if precision == "float16":
            dtype = torch.float16
        elif precision == "bfloat16":
            if not hasattr(torch, 'bfloat16'):
                raise ValueError("bfloat16 not supported in this PyTorch version")
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        # For PyTorch, we return a GradScaler for float16 precision
        if precision == "float16" and dynamic_loss_scaling:
            return torch.cuda.amp.GradScaler()
        else:
            return None

    def configure_gradient_accumulation(
        self,
        accumulation_steps: int = 1,
        effective_batch_size: Optional[int] = None,
        base_batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Configure gradient accumulation for training.

        Args:
            accumulation_steps: Number of steps to accumulate gradients
            effective_batch_size: Target effective batch size
            base_batch_size: Base batch size per step

        Returns:
            Dictionary with gradient accumulation configuration
        """
        # If effective and base batch sizes are provided, calculate accumulation steps
        if effective_batch_size is not None and base_batch_size is not None:
            accumulation_steps = max(1, effective_batch_size // base_batch_size)
            if effective_batch_size % base_batch_size != 0:
                logger.warning(
                    f"Effective batch size {effective_batch_size} is not divisible by "
                    f"base batch size {base_batch_size}. Using {accumulation_steps} accumulation steps "
                    f"for an effective batch size of {accumulation_steps * base_batch_size}."
                )

        logger.info(f"Configuring gradient accumulation with {accumulation_steps} steps")

        config = {
            "accumulation_steps": accumulation_steps,
            "effective_batch_multiplier": accumulation_steps
        }

        if base_batch_size:
            config["base_batch_size"] = base_batch_size
            config["effective_batch_size"] = base_batch_size * accumulation_steps

        return config

    def configure_distributed_training(
        self,
        strategy: str = "mirrored",  # "mirrored", "multi_worker", "parameter_server", "tpu"
        num_workers: int = None,
        communication_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Configure distributed training.

        Args:
            strategy: Distributed training strategy
            num_workers: Number of workers (None for auto-detect)
            communication_options: Additional communication options

        Returns:
            Dictionary with distributed training configuration
        """
        logger.info(f"Configuring distributed training with strategy: {strategy}")

        if communication_options is None:
            communication_options = {}

        config = {
            "strategy": strategy,
            "framework": self.framework
        }

        try:
            if self.framework == "tensorflow":
                dist_strategy = self._configure_tensorflow_distributed(strategy, num_workers, communication_options)
                config["tf_strategy"] = str(dist_strategy.__class__.__name__)

            elif self.framework == "pytorch":
                dist_config = self._configure_pytorch_distributed(strategy, num_workers, communication_options)
                config.update(dist_config)

            logger.info("Distributed training configured successfully")

        except Exception as e:
            logger.error(f"Error configuring distributed training: {str(e)}")
            config["error"] = str(e)

        return config

    def _configure_tensorflow_distributed(
        self,
        strategy: str,
        num_workers: Optional[int],
        communication_options: Dict[str, Any]
    ) -> Any:
        """Configure distributed training for TensorFlow."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        if strategy == "mirrored":
            # Multi-GPU on a single machine
            return tf.distribute.MirroredStrategy()

        elif strategy == "multi_worker":
            # Multi-worker distributed training
            communication = communication_options.get("communication", "auto")
            if communication == "auto":
                return tf.distribute.MultiWorkerMirroredStrategy()
            else:
                return tf.distribute.MultiWorkerMirroredStrategy(
                    communication=getattr(tf.distribute.experimental.CollectiveCommunication, communication.upper())
                )

        elif strategy == "parameter_server":
            # Parameter server strategy
            cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
            return tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

        elif strategy == "tpu":
            # TPU strategy
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.TPUStrategy(resolver)

        else:
            raise ValueError(f"Unsupported TensorFlow distribution strategy: {strategy}")

    def _configure_pytorch_distributed(
        self,
        strategy: str,
        num_workers: Optional[int],
        communication_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure distributed training for PyTorch."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")

        config = {"backend": communication_options.get("backend", "nccl")}

        if strategy == "data_parallel":
            # DataParallel (single machine, multiple GPUs)
            if num_workers is None:
                num_workers = torch.cuda.device_count()
            config["num_gpus"] = num_workers
            config["distributed_type"] = "DataParallel"

        elif strategy == "distributed_data_parallel":
            # DistributedDataParallel
            if "init_method" in communication_options:
                config["init_method"] = communication_options["init_method"]
            else:
                config["init_method"] = "env://"

            if "rank" in communication_options:
                config["rank"] = communication_options["rank"]
            if "world_size" in communication_options:
                config["world_size"] = communication_options["world_size"]

            config["distributed_type"] = "DistributedDataParallel"

        else:
            raise ValueError(f"Unsupported PyTorch distribution strategy: {strategy}")

        return config

    def benchmark_training(
        self,
        model: Any = None,
        train_dataset: Any = None,
        batch_size: int = 32,
        num_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        optimizer: Any = None,
        loss_function: Any = None,
        mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        distributed: bool = False,
        callbacks: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Benchmark model training performance.

        Args:
            model: Model to benchmark (uses self.model if None)
            train_dataset: Training dataset
            batch_size: Batch size
            num_epochs: Number of epochs
            steps_per_epoch: Number of steps per epoch (if None, uses full dataset)
            optimizer: Optimizer to use
            loss_function: Loss function to use
            mixed_precision: Whether to use mixed precision
            gradient_accumulation_steps: Number of gradient accumulation steps
            distributed: Whether to use distributed training
            callbacks: List of callbacks to use during training

        Returns:
            Dictionary with benchmark results
        """
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model provided for benchmarking")

        if train_dataset is None:
            raise ValueError("No training dataset provided for benchmarking")

        logger.info(f"Benchmarking training performance for {num_epochs} epochs with batch size {batch_size}")

        # Configure optimizations
        if mixed_precision:
            mp_config = self.configure_mixed_precision(enabled=True)

        if gradient_accumulation_steps > 1:
            ga_config = self.configure_gradient_accumulation(accumulation_steps=gradient_accumulation_steps)

        if distributed:
            dist_config = self.configure_distributed_training()

        # Framework-specific benchmarking
        start_time = time.time()

        try:
            if self.framework == "tensorflow":
                metrics = self._benchmark_tensorflow_training(
                    model, train_dataset, batch_size, num_epochs, steps_per_epoch,
                    optimizer, loss_function, mixed_precision, gradient_accumulation_steps,
                    distributed, callbacks
                )
            elif self.framework == "pytorch":
                metrics = self._benchmark_pytorch_training(
                    model, train_dataset, batch_size, num_epochs, steps_per_epoch,
                    optimizer, loss_function, mixed_precision, gradient_accumulation_steps,
                    distributed, callbacks
                )
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")

        except Exception as e:
            logger.error(f"Error during training benchmark: {str(e)}")
            metrics = {"error": str(e)}

        total_time = time.time() - start_time

        # Combine metrics
        results = {
            "total_time_seconds": total_time,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "samples_per_second": metrics.get("samples_per_second", 0),
            "framework": self.framework,
            "device": self.device,
            "mixed_precision": mixed_precision,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "distributed": distributed,
            "timestamp": datetime.now().isoformat()
        }

        # Add framework-specific metrics
        if "framework_metrics" in metrics:
            results["framework_metrics"] = metrics["framework_metrics"]

        logger.info(f"Training benchmark completed in {total_time:.2f} seconds")
        return results

    def _benchmark_tensorflow_training(
        self,
        model: Any,
        train_dataset: Any,
        batch_size: int,
        num_epochs: int,
        steps_per_epoch: Optional[int],
        optimizer: Any,
        loss_function: Any,
        mixed_precision: bool,
        gradient_accumulation_steps: int,
        distributed: bool,
        callbacks: List[Any]
    ) -> Dict[str, Any]:
        """Benchmark TensorFlow model training."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        # Prepare dataset
        if not isinstance(train_dataset, tf.data.Dataset):
            # Convert to TensorFlow dataset
            if hasattr(train_dataset, "to_tensorflow"):
                # For datasets with conversion method
                train_dataset = train_dataset.to_tensorflow()
            else:
                # Try to convert numpy arrays or similar
                try:
                    if isinstance(train_dataset, tuple) and len(train_dataset) == 2:
                        # Assuming (x, y) format
                        x, y = train_dataset
                        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
                    else:
                        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
                except Exception as e:
                    raise ValueError(f"Could not convert dataset to TensorFlow format: {str(e)}")

        # Apply batching if not already batched
        if not hasattr(train_dataset, "_batch_size") or train_dataset._batch_size != batch_size:
            train_dataset = train_dataset.batch(batch_size)

        # Apply prefetching for better performance
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()

        # Create loss function if not provided
        if loss_function is None:
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        # Create callbacks if not provided
        if callbacks is None:
            callbacks = []

        # Add timing callback
        class TimingCallback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.batch_times = []
                self.epoch_times = []

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                self.epoch_times.append(time.time() - self.epoch_start_time)

            def on_train_batch_begin(self, batch, logs=None):
                self.batch_start_time = time.time()

            def on_train_batch_end(self, batch, logs=None):
                self.batch_times.append(time.time() - self.batch_start_time)

        timing_callback = TimingCallback()
        callbacks.append(timing_callback)

        # Compile model if not compiled
        if not hasattr(model, "optimizer") or model.optimizer is None:
            model.compile(optimizer=optimizer, loss=loss_function)

        # Train model
        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1
        )

        # Calculate metrics
        batch_times = timing_callback.batch_times
        epoch_times = timing_callback.epoch_times

        num_samples = batch_size * (steps_per_epoch or len(train_dataset))
        samples_per_second = num_samples / np.mean(epoch_times) if epoch_times else 0

        metrics = {
            "samples_per_second": samples_per_second,
            "framework_metrics": {
                "avg_batch_time": np.mean(batch_times) if batch_times else 0,
                "avg_epoch_time": np.mean(epoch_times) if epoch_times else 0,
                "history": {k: v[-1] for k, v in history.history.items()}
            }
        }

        return metrics

    def _benchmark_pytorch_training(
        self,
        model: Any,
        train_dataset: Any,
        batch_size: int,
        num_epochs: int,
        steps_per_epoch: Optional[int],
        optimizer: Any,
        loss_function: Any,
        mixed_precision: bool,
        gradient_accumulation_steps: int,
        distributed: bool,
        callbacks: List[Any]
    ) -> Dict[str, Any]:
        """Benchmark PyTorch model training."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")

        # Prepare dataset and dataloader
        if not hasattr(train_dataset, "__getitem__") or not hasattr(train_dataset, "__len__"):
            raise ValueError("PyTorch training requires a dataset with __getitem__ and __len__ methods")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == "gpu" else False
        )

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())

        # Create loss function if not provided
        if loss_function is None:
            loss_function = torch.nn.CrossEntropyLoss()

        # Set up mixed precision if enabled
        scaler = torch.cuda.amp.GradScaler() if mixed_precision and self.device == "gpu" else None

        # Move model to device
        if self.device == "gpu":
            model = model.cuda()

        # Set model to training mode
        model.train()

        # Training loop
        batch_times = []
        epoch_times = []
        losses = []

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0

            for i, data in enumerate(train_loader):
                if steps_per_epoch is not None and i >= steps_per_epoch:
                    break

                batch_start = time.time()

                # Get inputs and targets
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    inputs, targets = data[0], data[1]
                else:
                    inputs, targets = data, None

                # Move data to device
                if self.device == "gpu":
                    inputs = inputs.cuda()
                    if targets is not None:
                        targets = targets.cuda()

                # Zero gradients
                if i % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()

                # Forward pass with mixed precision if enabled
                if mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = loss_function(outputs, targets) if targets is not None else outputs.mean()

                    # Scale loss and backward pass
                    scaler.scale(loss).backward()

                    # Step optimizer if gradient accumulation complete
                    if (i + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    # Standard forward and backward pass
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets) if targets is not None else outputs.mean()

                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    # Step optimizer if gradient accumulation complete
                    if (i + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()

                # Record metrics
                epoch_loss += loss.item()
                batch_times.append(time.time() - batch_start)

                if (i + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # Record epoch metrics
            epoch_times.append(time.time() - epoch_start)
            losses.append(epoch_loss / (steps_per_epoch or len(train_loader)))

            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_times[-1]:.2f}s, Loss: {losses[-1]:.4f}")

        # Calculate metrics
        num_samples = batch_size * (steps_per_epoch or len(train_loader))
        samples_per_second = num_samples / np.mean(epoch_times) if epoch_times else 0

        metrics = {
            "samples_per_second": samples_per_second,
            "framework_metrics": {
                "avg_batch_time": np.mean(batch_times) if batch_times else 0,
                "avg_epoch_time": np.mean(epoch_times) if epoch_times else 0,
                "final_loss": losses[-1] if losses else 0,
                "losses": losses
            }
        }

        return metrics