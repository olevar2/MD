"""
Auto-Optimization Framework - Core Module

This module provides the core functionality for the Auto-Optimization Framework,
including parameter tuning, hyperparameter search, and multi-objective optimization.
"""
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import concurrent.futures
import json
import random
import math
from enum import Enum
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

class OptimizationAlgorithm(str, Enum):
    """Supported optimization algorithms."""
    GRID_SEARCH = 'grid_search'
    RANDOM_SEARCH = 'random_search'
    BAYESIAN = 'bayesian'
    GENETIC = 'genetic'
    PARTICLE_SWARM = 'particle_swarm'
    SIMULATED_ANNEALING = 'simulated_annealing'


class OptimizationObjective(str, Enum):
    """Supported optimization objectives."""
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'


class ParameterType(str, Enum):
    """Types of parameters that can be optimized."""
    CONTINUOUS = 'continuous'
    INTEGER = 'integer'
    CATEGORICAL = 'categorical'
    BOOLEAN = 'boolean'


class ParameterSpace(BaseModel):
    """Definition of a parameter space for optimization."""
    name: str
    type: ParameterType
    range: Union[List[Union[float, int, str, bool]], Tuple[float, float],
        Tuple[int, int]]
    scale: Optional[str] = None
    step: Optional[float] = None

    def sample(self) ->Any:
        """Sample a random value from the parameter space."""
        if self.type == ParameterType.CONTINUOUS:
            if isinstance(self.range, (list, tuple)) and len(self.range) == 2:
                min_val, max_val = self.range
                if self.scale == 'log':
                    return math.exp(random.uniform(math.log(min_val), math.
                        log(max_val)))
                else:
                    return random.uniform(min_val, max_val)
        elif self.type == ParameterType.INTEGER:
            if isinstance(self.range, (list, tuple)) and len(self.range) == 2:
                min_val, max_val = self.range
                if self.step:
                    options = list(range(min_val, max_val + 1, int(self.step)))
                    return random.choice(options)
                else:
                    return random.randint(min_val, max_val)
        elif self.type == ParameterType.CATEGORICAL:
            if isinstance(self.range, list):
                return random.choice(self.range)
        elif self.type == ParameterType.BOOLEAN:
            return random.choice([True, False])
        return self.range[0] if isinstance(self.range, list
            ) and self.range else None


class OptimizationConfiguration(BaseModel):
    """Configuration for an optimization run."""
    parameter_space: List[ParameterSpace]
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE
    max_iterations: int = 100
    population_size: int = 30
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    convergence_threshold: float = 0.001
    max_stagnant_iterations: int = 10
    random_state: Optional[int] = None
    n_jobs: int = 1
    timeout: Optional[int] = None
    objectives: Optional[List[Dict[str, Any]]] = None
    pareto_front_size: int = 10
    max_drawdown: Optional[float] = None
    max_volatility: Optional[float] = None
    min_sharpe_ratio: Optional[float] = None
    max_var: Optional[float] = None
    max_cvar: Optional[float] = None
    custom_constraints: List[Dict[str, Any]] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Result of evaluating a single parameter set."""
    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    metrics: Dict[str, float] = Field(default_factory=dict)
    constraints_violated: bool = False
    violated_constraints: List[str] = Field(default_factory=list)
    evaluation_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    """Result of an optimization run."""
    optimization_id: str = Field(default_factory=lambda : str(uuid.uuid4()))
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0
    algorithm: str
    best_parameters: Dict[str, Any]
    best_objectives: Dict[str, float]
    best_metrics: Dict[str, float] = Field(default_factory=dict)
    convergence_history: List[Dict[str, Any]] = Field(default_factory=list)
    all_evaluations: List[EvaluationResult] = Field(default_factory=list)
    pareto_front: Optional[List[EvaluationResult]] = None
    status: str = 'completed'
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Optimizer:
    """
    Base class for optimization algorithms.
    """

    def __init__(self, config: OptimizationConfiguration,
        evaluation_function: Callable):
        """
        Initialize the optimizer.
        
        Args:
            config: Optimization configuration
            evaluation_function: Function that evaluates a parameter set and returns objectives
        """
        self.config = config
        self.evaluate = evaluation_function
        self.best_parameters = None
        self.best_objectives = None
        self.best_metrics = {}
        self.convergence_history = []
        self.all_evaluations = []
        self.pareto_front = []
        if self.config.random_state is not None:
            random.seed(self.config.random_state)
            np.random.seed(self.config.random_state)

    @with_exception_handling
    def optimize(self) ->OptimizationResult:
        """
        Run the optimization algorithm.
        
        Returns:
            OptimizationResult: The result of the optimization
        """
        start_time = datetime.utcnow()
        try:
            if self.config.algorithm == OptimizationAlgorithm.GRID_SEARCH:
                self._run_grid_search()
            elif self.config.algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
                self._run_random_search()
            elif self.config.algorithm == OptimizationAlgorithm.GENETIC:
                self._run_genetic_algorithm()
            elif self.config.algorithm == OptimizationAlgorithm.BAYESIAN:
                self._run_bayesian_optimization()
            elif self.config.algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                self._run_particle_swarm()
            elif self.config.algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
                self._run_simulated_annealing()
            else:
                raise ValueError(
                    f'Unsupported optimization algorithm: {self.config.algorithm}'
                    )
            status = 'completed'
            error_message = None
        except Exception as e:
            logger.error(f'Optimization failed: {str(e)}')
            status = 'failed'
            error_message = str(e)
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        result = OptimizationResult(algorithm=self.config.algorithm,
            best_parameters=self.best_parameters or {}, best_objectives=
            self.best_objectives or {}, best_metrics=self.best_metrics or {
            }, convergence_history=self.convergence_history,
            all_evaluations=self.all_evaluations, pareto_front=self.
            pareto_front if self.pareto_front else None, end_time=end_time,
            duration_seconds=duration, status=status, error_message=
            error_message)
        return result

    def _sample_random_parameters(self) ->Dict[str, Any]:
        """Sample random parameters from the parameter space."""
        params = {}
        for param in self.config.parameter_space:
            params[param.name] = param.sample()
        return params

    def _run_grid_search(self):
        """Run grid search optimization."""
        logger.info('Running grid search optimization...')
        grid_values = {}
        for param in self.config.parameter_space:
            if param.type == ParameterType.CONTINUOUS:
                min_val, max_val = param.range
                if param.step:
                    grid_values[param.name] = np.arange(min_val, max_val +
                        param.step, param.step).tolist()
                else:
                    grid_values[param.name] = np.linspace(min_val, max_val, 10
                        ).tolist()
            elif param.type == ParameterType.INTEGER:
                min_val, max_val = param.range
                if param.step:
                    grid_values[param.name] = list(range(min_val, max_val +
                        1, int(param.step)))
                else:
                    grid_values[param.name] = list(range(min_val, max_val + 1))
            elif param.type == ParameterType.CATEGORICAL or param.type == ParameterType.BOOLEAN:
                grid_values[param.name] = param.range
        self._evaluate_grid_combinations(grid_values)

    def _evaluate_grid_combinations(self, grid_values, current_params=None,
        param_names=None, current_idx=0):
        """Recursively evaluate all combinations of grid values."""
        if current_params is None:
            current_params = {}
        if param_names is None:
            param_names = list(grid_values.keys())
        if current_idx == len(param_names):
            self._evaluate_and_update(current_params)
            return
        param_name = param_names[current_idx]
        for value in grid_values[param_name]:
            current_params[param_name] = value
            self._evaluate_grid_combinations(grid_values, current_params,
                param_names, current_idx + 1)

    def _run_random_search(self):
        """Run random search optimization."""
        logger.info('Running random search optimization...')
        for i in range(self.config.max_iterations):
            params = self._sample_random_parameters()
            self._evaluate_and_update(params)
            if self._should_stop():
                logger.info(f'Early stopping at iteration {i + 1}')
                break

    def _run_genetic_algorithm(self):
        """Run genetic algorithm optimization."""
        logger.info('Running genetic algorithm optimization...')
        population = []
        for _ in range(self.config.population_size):
            params = self._sample_random_parameters()
            population.append(params)
        for generation in range(self.config.max_iterations):
            logger.debug(
                f'Generation {generation + 1}/{self.config.max_iterations}')
            evaluation_results = []
            for params in population:
                result = self._evaluate_and_update(params)
                evaluation_results.append((params, result))
            if self._should_stop():
                logger.info(f'Early stopping at generation {generation + 1}')
                break
            parents = self._select_parents(evaluation_results)
            next_population = []
            while len(next_population) < self.config.population_size:
                parent1, parent2 = random.sample(parents, 2)
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                next_population.append(child1)
                next_population.append(child2)
            population = next_population[:self.config.population_size]

    def _select_parents(self, evaluation_results):
        """Select parents using tournament selection."""
        if self.config.objective == OptimizationObjective.MAXIMIZE:
            evaluation_results.sort(key=lambda x: list(x[1].objectives.
                values())[0], reverse=True)
        else:
            evaluation_results.sort(key=lambda x: list(x[1].objectives.
                values())[0])
        parents = [params for params, _ in evaluation_results[:self.config.
            population_size // 2]]
        return parents

    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        child1, child2 = {}, {}
        for param in self.config.parameter_space:
            if random.random() < 0.5:
                child1[param.name] = parent1[param.name]
                child2[param.name] = parent2[param.name]
            else:
                child1[param.name] = parent2[param.name]
                child2[param.name] = parent1[param.name]
        return child1, child2

    def _mutate(self, params):
        """Mutate parameters based on mutation rate."""
        mutated = params.copy()
        for param in self.config.parameter_space:
            if random.random() < self.config.mutation_rate:
                for param_def in self.config.parameter_space:
                    if param_def.name == param:
                        mutated[param] = param_def.sample()
                        break
        return mutated

    def _run_bayesian_optimization(self):
        """Run Bayesian optimization."""
        logger.info(
            'Bayesian optimization not fully implemented yet, using random search...'
            )
        self._run_random_search()

    def _run_particle_swarm(self):
        """Run particle swarm optimization."""
        logger.info(
            'Particle swarm optimization not fully implemented yet, using random search...'
            )
        self._run_random_search()

    def _run_simulated_annealing(self):
        """Run simulated annealing optimization."""
        logger.info(
            'Simulated annealing not fully implemented yet, using random search...'
            )
        self._run_random_search()

    @with_exception_handling
    def _evaluate_and_update(self, params):
        """Evaluate parameters and update best result."""
        try:
            start_time = datetime.utcnow()
            result = self.evaluate(params)
            end_time = datetime.utcnow()
            evaluation_time = (end_time - start_time).total_seconds()
            if not isinstance(result, EvaluationResult):
                if isinstance(result, dict):
                    objectives = {}
                    metrics = {}
                    if self.config.objectives:
                        for obj_config in self.config.objectives:
                            key = obj_config['name']
                            if key in result:
                                objectives[key] = result[key]
                    else:
                        first_key = next(iter(result))
                        objectives = {first_key: result[first_key]}
                        metrics = {k: v for k, v in result.items() if k !=
                            first_key}
                    constraints_violated = False
                    violated_constraints = []
                    if (self.config.max_drawdown is not None and 
                        'max_drawdown' in metrics):
                        if metrics['max_drawdown'] > self.config.max_drawdown:
                            constraints_violated = True
                            violated_constraints.append(
                                f"max_drawdown: {metrics['max_drawdown']} > {self.config.max_drawdown}"
                                )
                    if (self.config.max_volatility is not None and 
                        'volatility' in metrics):
                        if metrics['volatility'] > self.config.max_volatility:
                            constraints_violated = True
                            violated_constraints.append(
                                f"volatility: {metrics['volatility']} > {self.config.max_volatility}"
                                )
                    if (self.config.min_sharpe_ratio is not None and 
                        'sharpe_ratio' in metrics):
                        if metrics['sharpe_ratio'
                            ] < self.config.min_sharpe_ratio:
                            constraints_violated = True
                            violated_constraints.append(
                                f"sharpe_ratio: {metrics['sharpe_ratio']} < {self.config.min_sharpe_ratio}"
                                )
                    result = EvaluationResult(parameters=params, objectives
                        =objectives, metrics=metrics, constraints_violated=
                        constraints_violated, violated_constraints=
                        violated_constraints, evaluation_time=evaluation_time)
            self.all_evaluations.append(result)
            if result.constraints_violated:
                return result
            if self._is_better(result):
                self.best_parameters = params.copy()
                self.best_objectives = result.objectives.copy()
                self.best_metrics = result.metrics.copy()
            self._update_convergence_history()
            if self.config.objectives and len(self.config.objectives) > 1:
                self._update_pareto_front(result)
            return result
        except Exception as e:
            logger.error(f'Error evaluating parameters {params}: {str(e)}')
            return EvaluationResult(parameters=params, objectives={},
                constraints_violated=True, violated_constraints=[
                f'Evaluation error: {str(e)}'], evaluation_time=0)

    def _is_better(self, result):
        """Check if the current result is better than the best so far."""
        if self.best_objectives is None:
            return True
        obj_key = next(iter(result.objectives))
        current_val = result.objectives[obj_key]
        best_val = self.best_objectives[obj_key]
        if self.config.objective == OptimizationObjective.MAXIMIZE:
            return current_val > best_val
        else:
            return current_val < best_val

    def _update_convergence_history(self):
        """Update convergence history with current best value."""
        if not self.best_objectives:
            return
        obj_key = next(iter(self.best_objectives))
        best_val = self.best_objectives[obj_key]
        self.convergence_history.append({'iteration': len(self.
            convergence_history) + 1, 'objective': obj_key, 'value':
            best_val, 'parameters': self.best_parameters})

    def _should_stop(self):
        """Check if optimization should stop early."""
        if len(self.convergence_history
            ) <= self.config.max_stagnant_iterations:
            return False
        recent_history = self.convergence_history[-self.config.
            max_stagnant_iterations:]
        first_val = recent_history[0]['value']
        for entry in recent_history[1:]:
            rel_change = abs(entry['value'] - first_val) / (abs(first_val) if
                abs(first_val) > 1e-10 else 1)
            if rel_change > self.config.convergence_threshold:
                return False
        return True

    def _update_pareto_front(self, result):
        """Update the Pareto front for multi-objective optimization."""
        if result.constraints_violated:
            return
        if not self.pareto_front:
            self.pareto_front.append(result)
            return
        dominated = False
        solutions_to_remove = []
        for idx, existing in enumerate(self.pareto_front):
            if self._dominates(result, existing):
                solutions_to_remove.append(idx)
            elif self._dominates(existing, result):
                dominated = True
                break
        if not dominated:
            for idx in sorted(solutions_to_remove, reverse=True):
                self.pareto_front.pop(idx)
            self.pareto_front.append(result)
            if len(self.pareto_front) > self.config.pareto_front_size:
                self._sort_pareto_front()
                self.pareto_front = self.pareto_front[:self.config.
                    pareto_front_size]

    def _dominates(self, solution1, solution2):
        """Check if solution1 dominates solution2 (is better in all objectives)."""
        better_in_at_least_one = False
        for obj_key in solution1.objectives:
            if obj_key not in solution2.objectives:
                continue
            val1 = solution1.objectives[obj_key]
            val2 = solution2.objectives[obj_key]
            if self._get_objective_direction(obj_key
                ) == OptimizationObjective.MAXIMIZE:
                if val1 < val2:
                    return False
                if val1 > val2:
                    better_in_at_least_one = True
            else:
                if val1 > val2:
                    return False
                if val1 < val2:
                    better_in_at_least_one = True
        return better_in_at_least_one

    def _get_objective_direction(self, obj_key):
        """Get optimization direction for an objective."""
        if self.config.objectives:
            for obj_config in self.config.objectives:
                if obj_config['name'] == obj_key:
                    return obj_config_manager.get('direction', self.config.objective)
        return self.config.objective

    def _sort_pareto_front(self):
        """Sort Pareto front solutions."""
        if not self.pareto_front:
            return
        objectives = set()
        min_vals = {}
        max_vals = {}
        for solution in self.pareto_front:
            for obj_key, val in solution.objectives.items():
                objectives.add(obj_key)
                if obj_key not in min_vals or val < min_vals[obj_key]:
                    min_vals[obj_key] = val
                if obj_key not in max_vals or val > max_vals[obj_key]:
                    max_vals[obj_key] = val

        def get_score(solution):
    """
    Get score.
    
    Args:
        solution: Description of solution
    
    """

            score = 0
            for obj_key in objectives:
                if obj_key in solution.objectives:
                    val = solution.objectives[obj_key]
                    min_val = min_vals[obj_key]
                    max_val = max_vals[obj_key]
                    if max_val == min_val:
                        normalized = 0.5
                    else:
                        normalized = (val - min_val) / (max_val - min_val)
                    if self._get_objective_direction(obj_key
                        ) == OptimizationObjective.MINIMIZE:
                        normalized = 1 - normalized
                    score += normalized
            return score
        self.pareto_front.sort(key=get_score, reverse=True)


class HyperParameterOptimizer(Optimizer):
    """
    Specialized optimizer for ML model hyperparameter tuning.
    """

    @with_exception_handling
    def __init__(self, config: OptimizationConfiguration,
        model_trainer_function: Callable, model_evaluator_function: Callable):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            config: Optimization configuration
            model_trainer_function: Function that trains a model with given hyperparameters
            model_evaluator_function: Function that evaluates a trained model
        """
        self.train_model = model_trainer_function
        self.evaluate_model = model_evaluator_function

        @with_exception_handling
        def evaluation_function(params):
    """
    Evaluation function.
    
    Args:
        params: Description of params
    
    """

            try:
                model = self.train_model(params)
                metrics = self.evaluate_model(model, params)
                return metrics
            except Exception as e:
                logger.error(f'Error in hyperparameter evaluation: {str(e)}')
                return {'error': str(e)}
        super().__init__(config, evaluation_function)


class MultiObjectiveOptimizer(Optimizer):
    """
    Specialized optimizer for multi-objective optimization with risk constraints.
    """

    def __init__(self, config: OptimizationConfiguration,
        evaluation_function: Callable):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            config: Optimization configuration
            evaluation_function: Function that evaluates a parameter set and returns objectives
        """
        super().__init__(config, evaluation_function)
        if not config.objectives or len(config.objectives) < 2:
            raise ValueError(
                'Multi-objective optimization requires at least two objectives'
                )


class OptimizationService:
    """
    Service for managing optimization jobs.
    """

    def __init__(self):
        """Initialize the optimization service."""
        self.optimizations = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def create_optimizer(self, config: OptimizationConfiguration,
        evaluation_function: Callable) ->Optimizer:
        """
        Create an appropriate optimizer based on configuration.
        
        Args:
            config: Optimization configuration
            evaluation_function: Function to evaluate parameters
            
        Returns:
            Optimizer: The created optimizer
        """
        if config.objectives and len(config.objectives) > 1:
            return MultiObjectiveOptimizer(config, evaluation_function)
        else:
            return Optimizer(config, evaluation_function)

    def create_hyperparameter_optimizer(self, config:
        OptimizationConfiguration, train_function: Callable,
        evaluate_function: Callable) ->HyperParameterOptimizer:
        """
        Create a hyperparameter optimizer.
        
        Args:
            config: Optimization configuration
            train_function: Function to train model
            evaluate_function: Function to evaluate model
            
        Returns:
            HyperParameterOptimizer: The created optimizer
        """
        return HyperParameterOptimizer(config, train_function,
            evaluate_function)

    def run_optimization(self, config: OptimizationConfiguration,
        evaluation_function: Callable) ->str:
        """
        Run optimization asynchronously.
        
        Args:
            config: Optimization configuration
            evaluation_function: Function to evaluate parameters
            
        Returns:
            str: Optimization ID
        """
        optimizer = self.create_optimizer(config, evaluation_function)
        optimization_id = str(uuid.uuid4())
        future = self.executor.submit(optimizer.optimize)
        self.optimizations[optimization_id] = {'future': future, 'config':
            config, 'start_time': datetime.utcnow(), 'status': 'running'}
        return optimization_id

    @with_exception_handling
    def get_optimization_result(self, optimization_id: str) ->Dict[str, Any]:
        """
        Get the result of an optimization job.
        
        Args:
            optimization_id: ID of the optimization job
            
        Returns:
            dict: Optimization status and result if available
        """
        if optimization_id not in self.optimizations:
            return {'status': 'not_found', 'error':
                'Optimization job not found'}
        job = self.optimizations[optimization_id]
        if job['status'] == 'running':
            if job['future'].done():
                try:
                    result = job['future'].result()
                    job['status'] = 'completed'
                    job['result'] = result
                    return {'status': 'completed', 'result': result.dict()}
                except Exception as e:
                    job['status'] = 'failed'
                    job['error'] = str(e)
                    return {'status': 'failed', 'error': str(e)}
            else:
                elapsed = (datetime.utcnow() - job['start_time']
                    ).total_seconds()
                progress = min(0.99, elapsed / (job['config'].
                    max_iterations * 2))
                return {'status': 'running', 'progress': progress}
        elif job['status'] == 'completed':
            return {'status': 'completed', 'result': job['result'].dict()}
        else:
            return {'status': job['status'], 'error': job.get('error',
                'Unknown error')}

    def stop_optimization(self, optimization_id: str) ->Dict[str, Any]:
        """
        Stop an ongoing optimization job.
        
        Args:
            optimization_id: ID of the optimization job
            
        Returns:
            dict: Status of the operation
        """
        if optimization_id not in self.optimizations:
            return {'status': 'not_found', 'error':
                'Optimization job not found'}
        job = self.optimizations[optimization_id]
        if job['status'] == 'running':
            job['future'].cancel()
            job['status'] = 'stopped'
            return {'status': 'stopped', 'message': 'Optimization job stopped'}
        else:
            return {'status': job['status'], 'message':
                f"Job already {job['status']}, cannot stop"}
