"""
Strategy Mutation Framework

This module implements a genetic algorithm-inspired mutation framework that allows
strategies to evolve based on feedback from the trading system.
"""

import asyncio
import random
import uuid
import logging
import copy
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher

from analysis_engine.adaptive_layer.parameter_statistical_validator import ParameterStatisticalValidator

logger = get_logger(__name__)


class StrategyMutationService:
    """
    Service responsible for mutating strategies based on feedback.
    
    This service applies genetic algorithm principles to evolve trading strategies
    based on feedback data, creating new strategy versions with modified parameters
    to improve performance.
    """
    
    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        strategy_repository: Any = None,  # Would be a proper repository interface
        parameter_validator: Optional[ParameterStatisticalValidator] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the strategy mutation service.
        
        Args:
            event_publisher: Event publisher for emitting mutation events
            strategy_repository: Repository for storing and retrieving strategies
            parameter_validator: Validator for parameter effectiveness
            config: Configuration dictionary
        """
        self.event_publisher = event_publisher
        self.strategy_repository = strategy_repository
        self.parameter_validator = parameter_validator or ParameterStatisticalValidator(config)
        self.config = config or {}
        
        # Mutation parameters (with defaults)
        self.mutation_rate = self.config.get("mutation_rate", 0.3)
        self.mutation_magnitude = self.config.get("mutation_magnitude", 0.2)
        self.initial_explore_generations = self.config.get("initial_explore_generations", 3)
        self.max_generations_without_improvement = self.config.get("max_generations_without_improvement", 5)
        self.min_samples_for_evaluation = self.config.get("min_samples_for_evaluation", 10)
        self.performance_window_days = self.config.get("performance_window_days", 14)
        self.generation_limit = self.config.get("generation_limit", 20)
        self.preserve_best_ratio = self.config.get("preserve_best_ratio", 0.3)
        
        # Storage for version tracking
        self.strategy_versions = {}  # strategy_id -> Dict of versions
        self.version_performance = {}  # version_id -> performance metrics
        self.version_lineage = {}  # version_id -> parent_id
        self.version_generations = {}  # version_id -> generation number
        self.active_versions = {}  # strategy_id -> version_id
        self.last_mutation_time = {}  # strategy_id -> datetime
        
        # Mutation schedules and tasks
        self.mutation_schedules = {}  # strategy_id -> next mutation time
        self._mutation_tasks = {}  # strategy_id -> asyncio task
        
        logger.info("StrategyMutationService initialized with mutation rate: %.2f, magnitude: %.2f",
                   self.mutation_rate, self.mutation_magnitude)
    
    async def start(self):
        """Start the mutation service background tasks."""
        # Load existing strategy versions and performance data
        if self.strategy_repository:
            await self._load_existing_data()
            
        logger.info("StrategyMutationService started")
    
    async def stop(self):
        """Stop the mutation service background tasks."""
        # Cancel all scheduled mutation tasks
        for strategy_id, task in self._mutation_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._mutation_tasks.clear()
        logger.info("StrategyMutationService stopped")
    
    async def register_strategy(
        self, 
        strategy_id: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new strategy for mutation tracking.
        
        Args:
            strategy_id: Strategy identifier
            parameters: Initial strategy parameters
            metadata: Additional strategy metadata
            
        Returns:
            Version ID of the registered strategy
        """
        # Generate initial version ID
        version_id = f"{strategy_id}_v1"
        generation = 1
        
        # Store initial version
        self.strategy_versions.setdefault(strategy_id, {})
        self.strategy_versions[strategy_id][version_id] = {
            "parameters": copy.deepcopy(parameters),
            "metadata": copy.deepcopy(metadata or {}),
            "creation_time": datetime.utcnow(),
            "is_active": True
        }
        
        # Set as active version
        self.active_versions[strategy_id] = version_id
        
        # Initialize version tracking
        self.version_lineage[version_id] = None  # No parent (first generation)
        self.version_generations[version_id] = generation
        self.version_performance[version_id] = []
        
        # Initialize last mutation time
        self.last_mutation_time[strategy_id] = datetime.utcnow()
        
        # Publish event if publisher available
        if self.event_publisher:
            await self.event_publisher.publish(
                "strategy.registered",
                {
                    "strategy_id": strategy_id,
                    "version_id": version_id,
                    "generation": generation,
                    "parameter_count": len(parameters),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        logger.info("Registered strategy %s with initial version %s", strategy_id, version_id)
        return version_id
    
    async def record_version_performance(
        self,
        version_id: str,
        performance_metrics: Dict[str, Any],
        market_regime: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Record performance metrics for a specific strategy version.
        
        Args:
            version_id: Version identifier
            performance_metrics: Dictionary of performance metrics
            market_regime: Optional market regime context
            timestamp: Optional timestamp (ISO format string)
            
        Returns:
            Boolean indicating success
        """
        if version_id not in self.version_performance:
            logger.warning("Cannot record performance for unknown version: %s", version_id)
            return False
            
        # Add timestamp if not provided
        if not timestamp:
            timestamp = datetime.utcnow().isoformat()
            
        # Create performance record
        performance_record = {
            "metrics": performance_metrics,
            "market_regime": market_regime,
            "timestamp": timestamp
        }
        
        # Add to version performance history
        self.version_performance[version_id].append(performance_record)
        
        # Keep history within window limit
        if len(self.version_performance[version_id]) > 100:  # arbitrary limit to prevent unbounded growth
            self.version_performance[version_id] = self.version_performance[version_id][-100:]
            
        # Extract strategy ID from version ID
        strategy_id = version_id.split('_v')[0] if '_v' in version_id else None
        
        # If this is for the active version, check if we should schedule mutation
        if strategy_id and self.active_versions.get(strategy_id) == version_id:
            await self._check_mutation_schedule(strategy_id)
            
        return True
    
    async def mutate_strategy(
        self,
        strategy_id: str,
        force: bool = False,
        specific_parameters: Optional[List[str]] = None,
        market_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a mutated version of a strategy based on feedback.
        
        Args:
            strategy_id: Strategy identifier
            force: Force mutation even if conditions aren't optimal
            specific_parameters: Optional list of specific parameters to mutate
            market_regime: Optional target market regime
            
        Returns:
            Dict with mutation results including new version ID
        """
        if strategy_id not in self.strategy_versions:
            return {
                "success": False,
                "reason": f"Strategy {strategy_id} not found",
                "strategy_id": strategy_id
            }
            
        # Get active version
        active_version_id = self.active_versions.get(strategy_id)
        if not active_version_id or active_version_id not in self.strategy_versions[strategy_id]:
            return {
                "success": False,
                "reason": f"No active version found for strategy {strategy_id}",
                "strategy_id": strategy_id
            }
            
        # Get active version data
        active_version = self.strategy_versions[strategy_id][active_version_id]
        
        # Check if we have enough performance data (skip if forced)
        if not force:
            version_performance = self.version_performance.get(active_version_id, [])
            recent_performance = self._get_recent_performance(version_performance)
            if len(recent_performance) < self.min_samples_for_evaluation:
                return {
                    "success": False,
                    "reason": f"Insufficient performance samples ({len(recent_performance)}/{self.min_samples_for_evaluation})",
                    "strategy_id": strategy_id,
                    "version_id": active_version_id
                }
        
        # Get current parameters
        current_parameters = active_version["parameters"]
        
        # Create mutation
        parent_generation = self.version_generations.get(active_version_id, 1)
        new_generation = parent_generation + 1
        
        # Check if we've hit generation limit
        if new_generation > self.generation_limit and not force:
            return {
                "success": False,
                "reason": f"Generation limit reached ({new_generation}/{self.generation_limit})",
                "strategy_id": strategy_id,
                "version_id": active_version_id,
                "generation": parent_generation
            }
        
        # Create new version ID
        new_version_id = f"{strategy_id}_v{new_generation}"
        mutation_id = str(uuid.uuid4())
        
        # Perform parameter mutation
        new_parameters, param_changes = self._mutate_parameters(
            current_parameters,
            specific_parameters=specific_parameters,
            market_regime=market_regime
        )
        
        # Create new version entry
        metadata = copy.deepcopy(active_version.get("metadata", {}))
        metadata.update({
            "parent_version": active_version_id,
            "mutation_id": mutation_id,
            "market_regime": market_regime,
            "forced_mutation": force,
            "specific_parameters": specific_parameters
        })
        
        # Store the new version
        self.strategy_versions[strategy_id][new_version_id] = {
            "parameters": new_parameters,
            "metadata": metadata,
            "creation_time": datetime.utcnow(),
            "is_active": True  # New version starts as active
        }
        
        # Update version tracking
        self.version_lineage[new_version_id] = active_version_id
        self.version_generations[new_version_id] = new_generation
        self.version_performance[new_version_id] = []
        
        # De-activate previous version
        self.strategy_versions[strategy_id][active_version_id]["is_active"] = False
        
        # Update active version
        self.active_versions[strategy_id] = new_version_id
        
        # Update last mutation time
        self.last_mutation_time[strategy_id] = datetime.utcnow()
        
        # Publish mutation event if publisher available
        if self.event_publisher:
            await self.event_publisher.publish(
                "strategy.mutated",
                {
                    "strategy_id": strategy_id,
                    "mutation_id": mutation_id,
                    "parent_version": active_version_id,
                    "new_version": new_version_id,
                    "generation": new_generation,
                    "parameter_changes": param_changes,
                    "market_regime": market_regime,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        logger.info("Created mutation %s for strategy %s: version %s -> %s with %d parameter changes",
                   mutation_id, strategy_id, active_version_id, new_version_id, len(param_changes))
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "mutation_id": mutation_id,
            "parent_version": active_version_id,
            "new_version": new_version_id,
            "generation": new_generation,
            "parameter_changes": param_changes,
            "change_count": len(param_changes)
        }
    
    async def evaluate_and_select_best_version(
        self,
        strategy_id: str,
        market_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate and select the best performing version of a strategy.
        
        Args:
            strategy_id: Strategy identifier
            market_regime: Optional market regime to consider
            
        Returns:
            Dict with evaluation results
        """
        if strategy_id not in self.strategy_versions:
            return {
                "success": False,
                "reason": f"Strategy {strategy_id} not found",
                "strategy_id": strategy_id
            }
            
        # Get all versions for this strategy
        versions = self.strategy_versions[strategy_id]
        if not versions:
            return {
                "success": False,
                "reason": f"No versions found for strategy {strategy_id}",
                "strategy_id": strategy_id
            }
            
        # Collect performance data for each version
        evaluated_versions = []
        for version_id, version_data in versions.items():
            # Only evaluate recent versions (last 3 generations)
            generation = self.version_generations.get(version_id, 1)
            current_generation = max(self.version_generations.values()) if self.version_generations else 1
            
            if current_generation - generation > 3 and len(versions) > 3:
                continue
                
            perf_data = self.version_performance.get(version_id, [])
            
            # Filter by market regime if specified
            if market_regime:
                perf_data = [p for p in perf_data if p.get("market_regime") == market_regime]
                
            # Get recent performance
            recent_perf = self._get_recent_performance(perf_data)
            
            # Need minimum samples to evaluate
            if len(recent_perf) < self.min_samples_for_evaluation:
                continue
                
            # Calculate aggregate metrics
            try:
                agg_metrics = self._calculate_aggregate_metrics(recent_perf)
                
                # Add to evaluated versions
                evaluated_versions.append({
                    "version_id": version_id,
                    "generation": generation,
                    "metrics": agg_metrics,
                    "sample_count": len(recent_perf),
                    "is_active": version_id == self.active_versions.get(strategy_id)
                })
            except Exception as e:
                logger.error("Error calculating metrics for version %s: %s", version_id, str(e))
        
        # If no versions have enough data, return early
        if not evaluated_versions:
            return {
                "success": False,
                "reason": "Insufficient performance data for evaluation",
                "strategy_id": strategy_id,
                "versions_checked": len(versions)
            }
            
        # Find best version based on fitness function
        best_version = self._select_best_version(evaluated_versions)
        best_version_id = best_version["version_id"]
        
        # Check if best version is already active
        current_active = self.active_versions.get(strategy_id)
        if best_version_id == current_active:
            return {
                "success": True,
                "action": "no_change",
                "reason": "Best version already active",
                "strategy_id": strategy_id,
                "best_version": best_version_id,
                "versions_evaluated": len(evaluated_versions)
            }
            
        # Activate the best version
        if current_active and current_active in self.strategy_versions[strategy_id]:
            self.strategy_versions[strategy_id][current_active]["is_active"] = False
            
        self.strategy_versions[strategy_id][best_version_id]["is_active"] = True
        self.active_versions[strategy_id] = best_version_id
        
        # Publish version change event if publisher available
        if self.event_publisher:
            await self.event_publisher.publish(
                "strategy.version_changed",
                {
                    "strategy_id": strategy_id,
                    "previous_version": current_active,
                    "new_version": best_version_id,
                    "reason": "performance_evaluation",
                    "market_regime": market_regime,
                    "performance_metrics": best_version["metrics"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        logger.info("Changed active version for strategy %s: %s -> %s based on performance evaluation",
                   strategy_id, current_active, best_version_id)
        
        return {
            "success": True,
            "action": "version_changed",
            "strategy_id": strategy_id,
            "previous_version": current_active,
            "best_version": best_version_id,
            "generation": best_version["generation"],
            "performance_metrics": best_version["metrics"],
            "versions_evaluated": len(evaluated_versions)
        }
    
    async def get_version_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of version history entries
        """
        if strategy_id not in self.strategy_versions:
            return []
            
        versions = self.strategy_versions[strategy_id]
        active_version = self.active_versions.get(strategy_id)
        
        history = []
        for version_id, version_data in versions.items():
            perf_data = self.version_performance.get(version_id, [])
            recent_perf = self._get_recent_performance(perf_data)
            
            # Calculate aggregate metrics if we have enough data
            metrics = {}
            if len(recent_perf) >= 5:
                try:
                    metrics = self._calculate_aggregate_metrics(recent_perf)
                except Exception:
                    pass
                    
            history.append({
                "version_id": version_id,
                "parent_id": self.version_lineage.get(version_id),
                "generation": self.version_generations.get(version_id, 1),
                "active": version_id == active_version,
                "creation_timestamp": version_data.get("creation_time", datetime.utcnow()).isoformat(),
                "sample_count": len(recent_perf),
                "metrics": metrics,
                "metadata": version_data.get("metadata", {})
            })
            
        # Sort by generation
        history.sort(key=lambda x: x["generation"])
        return history
    
    async def get_mutation_effectiveness(self, strategy_id: str) -> Dict[str, Any]:
        """
        Analyze the effectiveness of mutations for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict with mutation effectiveness metrics
        """
        if strategy_id not in self.strategy_versions:
            return {
                "strategy_id": strategy_id,
                "error": "Strategy not found"
            }
            
        # Get version history
        versions = await self.get_version_history(strategy_id)
        if len(versions) <= 1:
            return {
                "strategy_id": strategy_id,
                "effectiveness": 0,
                "reason": "Insufficient versions for analysis",
                "version_count": len(versions)
            }
            
        # Count successful mutations (where a child outperforms its parent)
        successful_mutations = 0
        total_evaluated = 0
        
        # Create version lookup for faster access
        version_lookup = {v["version_id"]: v for v in versions}
        
        for version in versions:
            parent_id = version["parent_id"]
            
            # Skip first version (no parent)
            if not parent_id:
                continue
                
            # If both have metrics, compare them
            if parent_id in version_lookup and "metrics" in version and "metrics" in version_lookup[parent_id]:
                parent_metrics = version_lookup[parent_id]["metrics"]
                child_metrics = version["metrics"]
                
                # Skip if metrics are empty
                if not parent_metrics or not child_metrics:
                    continue
                    
                # Compare using fitness function
                parent_fitness = self._calculate_fitness(parent_metrics)
                child_fitness = self._calculate_fitness(child_metrics)
                
                if child_fitness > parent_fitness:
                    successful_mutations += 1
                    
                total_evaluated += 1
                
        # Calculate success rate
        success_rate = successful_mutations / total_evaluated if total_evaluated > 0 else 0
        
        # Find best generation vs first generation improvement
        first_version = next((v for v in versions if v["generation"] == 1), None)
        best_version = max(versions, key=lambda v: self._calculate_fitness(v.get("metrics", {})) 
                                               if v.get("metrics") else 0)
        
        improvement = 0
        if first_version and best_version and first_version != best_version:
            first_metrics = first_version.get("metrics", {})
            best_metrics = best_version.get("metrics", {})
            
            if first_metrics and best_metrics:
                first_fitness = self._calculate_fitness(first_metrics)
                best_fitness = self._calculate_fitness(best_metrics)
                
                if first_fitness > 0:
                    improvement = (best_fitness - first_fitness) / first_fitness
        
        return {
            "strategy_id": strategy_id,
            "version_count": len(versions),
            "generations": max(v["generation"] for v in versions),
            "mutations_evaluated": total_evaluated,
            "successful_mutations": successful_mutations,
            "success_rate": success_rate,
            "overall_improvement": improvement,
            "best_version": best_version["version_id"] if best_version else None,
            "best_generation": best_version["generation"] if best_version else None
        }
    
    def _mutate_parameters(
        self,
        parameters: Dict[str, Any],
        specific_parameters: Optional[List[str]] = None,
        market_regime: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Create mutated parameters based on current parameters.
        
        Args:
            parameters: Current parameter values
            specific_parameters: Optional list of specific parameters to mutate
            market_regime: Optional market regime context
            
        Returns:
            Tuple of (mutated parameters dict, list of parameter changes)
        """
        # Make a copy of the parameters
        new_parameters = copy.deepcopy(parameters)
        param_changes = []
        
        # Determine which parameters to mutate
        param_keys = list(parameters.keys())
        if specific_parameters:
            # Filter to only include parameters that exist
            param_keys = [k for k in specific_parameters if k in parameters]
            
        # Select parameters to mutate based on mutation rate
        if not specific_parameters:
            param_count = max(1, int(len(param_keys) * self.mutation_rate))
            params_to_mutate = random.sample(param_keys, param_count)
        else:
            params_to_mutate = param_keys
            
        # Mutate each selected parameter
        for param_name in params_to_mutate:
            original_value = parameters[param_name]
            
            # Skip non-numeric parameters
            if not isinstance(original_value, (int, float)):
                continue
                
            # Apply mutation based on parameter type
            if isinstance(original_value, int):
                # Integer mutation - ensure we change by at least 1
                mutation_size = max(1, int(abs(original_value) * self.mutation_magnitude))
                mutation_direction = random.choice([-1, 1])
                new_value = original_value + (mutation_direction * mutation_size)
                
                # Ensure integers stay integers
                new_parameters[param_name] = int(new_value)
                
            else:
                # Float mutation
                mutation_size = abs(original_value) * self.mutation_magnitude
                mutation_direction = random.uniform(-1, 1)
                new_value = original_value + (mutation_direction * mutation_size)
                
                # Round to reasonable precision to avoid floating point issues
                new_parameters[param_name] = round(new_value, 6)
                
            # Record the change
            param_changes.append({
                "parameter": param_name,
                "old_value": original_value,
                "new_value": new_parameters[param_name],
                "percent_change": ((new_parameters[param_name] - original_value) / original_value) 
                                   if original_value != 0 else float('inf')
            })
            
        return new_parameters, param_changes
    
    def _calculate_aggregate_metrics(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate metrics from a list of performance records.
        
        Args:
            performance_data: List of performance data records
            
        Returns:
            Dict of aggregate metric values
        """
        if not performance_data:
            return {}
            
        # Extract metrics from performance records
        all_metrics = [record["metrics"] for record in performance_data if "metrics" in record]
        if not all_metrics:
            return {}
            
        # Initialize aggregated metrics
        agg_metrics = {}
        
        # Calculate aggregates for each metric
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            # Skip non-numeric metrics
            if not all(isinstance(m.get(key), (int, float)) for m in all_metrics if key in m):
                continue
                
            # Extract values for this metric
            values = [m[key] for m in all_metrics if key in m]
            if not values:
                continue
                
            # Calculate basic statistics
            agg_metrics[key] = sum(values) / len(values)  # Mean
            agg_metrics[f"{key}_min"] = min(values)
            agg_metrics[f"{key}_max"] = max(values)
            
            # Calculate standard deviation if we have enough values
            if len(values) > 1:
                mean = agg_metrics[key]
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                agg_metrics[f"{key}_std"] = variance ** 0.5
        
        # Special metrics for trading strategies
        if "profit_loss" in agg_metrics:
            profit_loss_values = [m["profit_loss"] for m in all_metrics if "profit_loss" in m]
            
            # Win rate
            if "is_win" in all_metrics[0]:
                win_values = [m["is_win"] for m in all_metrics if "is_win" in m]
                agg_metrics["win_rate"] = sum(win_values) / len(win_values) if win_values else 0
                
            # Calculate drawdown if we have drawdown values
            if all("max_drawdown" in m for m in all_metrics):
                drawdown_values = [m["max_drawdown"] for m in all_metrics]
                agg_metrics["avg_max_drawdown"] = sum(drawdown_values) / len(drawdown_values)
                agg_metrics["worst_drawdown"] = max(drawdown_values)
                
            # Profit factor
            if all("gross_profit" in m and "gross_loss" in m for m in all_metrics):
                total_profit = sum(m["gross_profit"] for m in all_metrics)
                total_loss = sum(abs(m["gross_loss"]) for m in all_metrics)
                agg_metrics["profit_factor"] = total_profit / total_loss if total_loss != 0 else float('inf')
                
            # Risk-adjusted return - approximate Sharpe ratio
            if len(profit_loss_values) > 1:
                mean_return = agg_metrics["profit_loss"]
                std_dev = agg_metrics.get("profit_loss_std", 0)
                if std_dev > 0:
                    agg_metrics["sharpe_ratio"] = mean_return / std_dev
        
        return agg_metrics
    
    def _calculate_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall fitness score from metrics.
        
        Args:
            metrics: Dict of metric values
            
        Returns:
            Fitness score (higher is better)
        """
        if not metrics:
            return 0.0
            
        # Get fitness metric weights from config
        weights = self.config.get("fitness_metrics", {
            "profit_loss": 0.4,
            "win_rate": 0.2,
            "profit_factor": 0.2,
            "max_drawdown": -0.1,  # Negative weight since lower is better
            "sharpe_ratio": 0.1
        })
        
        fitness = 0.0
        
        # Apply weights to metrics
        for metric, weight in weights.items():
            if metric in metrics:
                # For drawdown, lower is better
                if metric == "max_drawdown" or metric == "avg_max_drawdown":
                    # Convert to positive impact (lower drawdown = higher score)
                    # A drawdown of 0 is perfect, 1 (100%) is worst
                    fitness += weight * (1.0 - metrics[metric])
                else:
                    fitness += weight * metrics[metric]
                    
        return fitness
    
    def _select_best_version(self, evaluated_versions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best version based on performance metrics.
        
        Args:
            evaluated_versions: List of versions with metrics
            
        Returns:
            The best version dict
        """
        if not evaluated_versions:
            return {}
            
        # Calculate fitness score for each version
        for version in evaluated_versions:
            version["fitness"] = self._calculate_fitness(version["metrics"])
            
        # Sort by fitness (descending)
        evaluated_versions.sort(key=lambda v: v["fitness"], reverse=True)
        
        return evaluated_versions[0]
    
    def _get_recent_performance(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get recent performance data within the configured window.
        
        Args:
            performance_data: List of all performance records
            
        Returns:
            Filtered list of recent performance records
        """
        if not performance_data:
            return []
            
        # Calculate cutoff time
        cutoff = datetime.utcnow() - timedelta(days=self.performance_window_days)
        
        # Filter to recent records
        recent = []
        for record in performance_data:
            # Get timestamp from record
            timestamp_str = record.get("timestamp")
            if not timestamp_str:
                continue
                
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp >= cutoff:
                    recent.append(record)
            except (ValueError, TypeError):
                # Skip records with invalid timestamps
                continue
                
        return recent
    
    async def _check_mutation_schedule(self, strategy_id: str) -> None:
        """
        Check if a strategy should be scheduled for mutation.
        
        Args:
            strategy_id: Strategy identifier
        """
        # If already scheduled, do nothing
        if strategy_id in self._mutation_tasks and not self._mutation_tasks[strategy_id].done():
            return
            
        # Check if it's time to mutate
        last_mutation = self.last_mutation_time.get(strategy_id)
        if not last_mutation:
            return
            
        # Calculate time since last mutation
        time_since_mutation = (datetime.utcnow() - last_mutation).total_seconds() / 86400  # days
        
        # Get performance data for active version
        active_version = self.active_versions.get(strategy_id)
        if not active_version:
            return
            
        performance_data = self.version_performance.get(active_version, [])
        recent_performance = self._get_recent_performance(performance_data)
        
        # Determine if we should schedule mutation
        should_schedule = False
        schedule_delay = 1.0  # default 1 day
        
        # If we have enough data and it's been long enough
        min_days_between_mutations = self.config.get("min_days_between_mutations", 1)
        
        if time_since_mutation >= min_days_between_mutations:
            if len(recent_performance) >= self.min_samples_for_evaluation:
                # Calculate performance trend
                metrics = self._calculate_aggregate_metrics(recent_performance)
                current_fitness = self._calculate_fitness(metrics)
                
                # Get previous generation
                generation = self.version_generations.get(active_version, 1)
                if generation > 1:
                    # See if we can find the parent version
                    parent_version = self.version_lineage.get(active_version)
                    if parent_version and parent_version in self.version_performance:
                        parent_perf = self._get_recent_performance(self.version_performance[parent_version])
                        if parent_perf:
                            parent_metrics = self._calculate_aggregate_metrics(parent_perf)
                            parent_fitness = self._calculate_fitness(parent_metrics)
                            
                            # If performance is worse than parent, schedule mutation sooner
                            if current_fitness < parent_fitness:
                                should_schedule = True
                                schedule_delay = 0.5  # Half a day
                                
                # If in early generations, mutate more frequently
                if generation <= self.initial_explore_generations:
                    should_schedule = True
                    schedule_delay = 0.5
                    
                # Otherwise schedule based on regular interval
                elif time_since_mutation >= self.config.get("days_between_mutations", 7):
                    should_schedule = True
        
        # Schedule task if needed
        if should_schedule:
            # Calculate when to run the mutation
            mutation_time = datetime.utcnow() + timedelta(days=schedule_delay)
            
            # Calculate seconds until mutation
            seconds_until_mutation = (mutation_time - datetime.utcnow()).total_seconds()
            seconds_until_mutation = max(60, seconds_until_mutation)  # At least 1 minute
            
            # Schedule the mutation
            task = asyncio.create_task(self._scheduled_mutation_task(strategy_id, seconds_until_mutation))
            self._mutation_tasks[strategy_id] = task
            
            logger.info("Scheduled mutation for strategy %s in %.1f hours", 
                       strategy_id, seconds_until_mutation / 3600)
    
    async def _scheduled_mutation_task(self, strategy_id: str, delay_seconds: float) -> None:
        """
        Task that runs after a delay to perform a scheduled mutation.
        
        Args:
            strategy_id: Strategy identifier
            delay_seconds: Seconds to delay before mutation
        """
        try:
            # Wait for the scheduled time
            await asyncio.sleep(delay_seconds)
            
            # Perform mutation
            logger.info("Running scheduled mutation for strategy %s", strategy_id)
            await self.mutate_strategy(strategy_id)
            
        except asyncio.CancelledError:
            # Task was cancelled, that's OK
            logger.debug("Scheduled mutation for strategy %s was cancelled", strategy_id)
            
        except Exception as e:
            logger.error("Error in scheduled mutation for strategy %s: %s", 
                        strategy_id, str(e))
    
    async def _load_existing_data(self) -> None:
        """Load existing strategy versions and performance data from repository."""
        if not self.strategy_repository:
            return
            
        try:
            # This would call repository methods to load data
            # For now, it's a placeholder
            logger.info("Would load existing strategy data from repository")
        except Exception as e:
            logger.error("Error loading existing strategy data: %s", str(e))
