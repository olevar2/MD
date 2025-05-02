"""
Implementation Shortfall Algorithm.

This module implements an Implementation Shortfall algorithm that minimizes
the difference between the decision price and the final execution price
for large orders by adapting execution based on market conditions.
"""

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from ..interfaces.broker_adapter_interface import (
    BrokerAdapterInterface,
    OrderRequest,
    ExecutionReport,
    OrderStatus,
)
from .base_algorithm import BaseExecutionAlgorithm, ExecutionResult


class MarketImpactModel:
    """
    Model for estimating market impact of orders.
    
    Used by the Implementation Shortfall algorithm to estimate price impact
    and optimize execution strategy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market impact model.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default parameters
        self.alpha = self.config.get('alpha', 0.1)  # Base impact factor
        self.beta = self.config.get('beta', 0.3)  # Order size exponent
        self.gamma = self.config.get('gamma', 0.5)  # Volatility factor
        
    def estimate_impact(self, 
                      order_size: float, 
                      avg_daily_volume: float,
                      volatility: float,
                      spread: float) -> float:
        """
        Estimate the market impact of an order.
        
        Args:
            order_size: Size of the order
            avg_daily_volume: Average daily volume for the instrument
            volatility: Current volatility (e.g., standard deviation of returns)
            spread: Current bid-ask spread
            
        Returns:
            Estimated price impact in percentage
        """
        # Calculate relative order size
        relative_size = order_size / avg_daily_volume if avg_daily_volume > 0 else 0.1
        
        # Square root model with volatility adjustment
        impact = self.alpha * spread + self.gamma * volatility * (relative_size ** self.beta)
        
        return impact
    
    def optimal_participation_rate(self, 
                                 order_size: float, 
                                 avg_daily_volume: float,
                                 volatility: float,
                                 urgency: float) -> float:
        """
        Calculate the optimal participation rate.
        
        Args:
            order_size: Size of the order
            avg_daily_volume: Average daily volume for the instrument
            volatility: Current volatility
            urgency: Urgency factor (0-1)
            
        Returns:
            Optimal participation rate (0-1)
        """
        # Base participation rate based on order size
        base_rate = min(0.3, (order_size / avg_daily_volume) ** 0.5) if avg_daily_volume > 0 else 0.1
        
        # Adjust for volatility (higher volatility -> lower participation)
        vol_adjustment = 1.0 / (1.0 + volatility)
        
        # Adjust for urgency (higher urgency -> higher participation)
        urgency_adjustment = 1.0 + urgency
        
        # Calculate final rate
        rate = base_rate * vol_adjustment * urgency_adjustment
        
        # Ensure rate is within bounds
        return max(0.01, min(0.5, rate))


class ImplementationShortfallAlgorithm(BaseExecutionAlgorithm):
    """
    Implementation Shortfall algorithm.
    
    This algorithm minimizes the difference between the decision price and
    the final execution price for large orders by adapting execution based
    on market conditions.
    """
    
    def __init__(self, 
                 broker_adapters: Dict[str, BrokerAdapterInterface],
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Implementation Shortfall algorithm.
        
        Args:
            broker_adapters: Dictionary of broker adapters by name
            logger: Logger instance
            config: Algorithm-specific configuration
        """
        super().__init__(broker_adapters, logger, config)
        
        # Default configuration
        self.default_broker = self.config.get('default_broker')
        self.max_duration_minutes = self.config.get('max_duration_minutes', 120)  # Default: 2 hours
        self.min_duration_minutes = self.config.get('min_duration_minutes', 10)  # Default: 10 minutes
        self.min_slice_size = self.config.get('min_slice_size', 1000)  # Minimum size per slice
        self.urgency = self.config.get('urgency', 0.5)  # Urgency factor (0-1)
        self.market_data_service = self.config.get('market_data_service')
        
        # Create market impact model
        self.impact_model = MarketImpactModel(self.config.get('impact_model', {}))
        
        # Execution state
        self.original_order = None
        self.execution_task = None
        self.child_orders = {}  # order_id -> order details
        self.execution_reports = {}  # order_id -> execution report
        self.is_executing = False
        self.is_cancelled = False
        self.start_time = None
        self.end_time = None
        self.next_slice_time = None
        self.slices_executed = 0
        self.slices_total = 0
        self.market_conditions = {}
        self.decision_price = None
        self.initial_market_impact = 0.0
    
    async def execute(self, order: OrderRequest) -> ExecutionResult:
        """
        Execute the Implementation Shortfall algorithm for the given order.
        
        Args:
            order: The order to execute
            
        Returns:
            ExecutionResult with details of the execution
        """
        if self.is_executing:
            self.logger.warning("Implementation Shortfall algorithm is already executing an order")
            return ExecutionResult(
                algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id,
                status='FAILED',
                metrics={'error': 'Algorithm already executing an order'}
            )
        
        self.is_executing = True
        self.original_order = order
        self.child_orders = {}
        self.execution_reports = {}
        self.is_cancelled = False
        self.start_time = datetime.utcnow()
        
        try:
            # Get current market conditions
            self.market_conditions = await self._get_market_conditions(order.instrument)
            
            if not self.market_conditions:
                self.logger.error(f"Failed to get market conditions for {order.instrument}")
                return ExecutionResult(
                    algorithm_id=self.algorithm_id,
                    original_order_id=order.client_order_id,
                    status='FAILED',
                    metrics={'error': 'Failed to get market conditions'}
                )
            
            # Store decision price
            self.decision_price = order.price or self.market_conditions.get('price')
            
            # Estimate initial market impact
            self.initial_market_impact = self.impact_model.estimate_impact(
                order_size=order.quantity,
                avg_daily_volume=self.market_conditions.get('avg_daily_volume', 1000000),
                volatility=self.market_conditions.get('volatility', 0.001),
                spread=self.market_conditions.get('spread', 0.0001)
            )
            
            # Calculate optimal execution parameters
            participation_rate = self.impact_model.optimal_participation_rate(
                order_size=order.quantity,
                avg_daily_volume=self.market_conditions.get('avg_daily_volume', 1000000),
                volatility=self.market_conditions.get('volatility', 0.001),
                urgency=self.urgency
            )
            
            # Calculate execution duration based on participation rate
            expected_volume_per_minute = self.market_conditions.get('avg_daily_volume', 1000000) / (24 * 60)
            ideal_duration_minutes = order.quantity / (expected_volume_per_minute * participation_rate)
            
            # Constrain duration to configured limits
            duration_minutes = max(
                self.min_duration_minutes,
                min(self.max_duration_minutes, ideal_duration_minutes)
            )
            
            # Calculate number of slices
            self.slices_total = max(2, min(24, int(duration_minutes / 5)))  # One slice every 5 minutes
            
            # Calculate end time
            self.end_time = self.start_time + timedelta(minutes=duration_minutes)
            self.next_slice_time = self.start_time
            
            # Calculate slice interval
            slice_interval = timedelta(minutes=duration_minutes / self.slices_total)
            
            # Trigger started callbacks
            self._trigger_callbacks('started', {
                'algorithm_id': self.algorithm_id,
                'order_id': order.client_order_id,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'slices': self.slices_total,
                'slice_interval_seconds': slice_interval.total_seconds(),
                'participation_rate': participation_rate,
                'initial_market_impact': self.initial_market_impact,
                'decision_price': self.decision_price
            })
            
            # Start the execution task
            self.execution_task = asyncio.create_task(self._execute_slices(order, slice_interval))
            
            # Wait for the execution task to complete
            result = await self.execution_task
            return result
            
        except asyncio.CancelledError:
            self.logger.info("Implementation Shortfall execution task was cancelled")
            
            # Create execution result for cancelled execution
            return ExecutionResult(
                algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id,
                status='CANCELLED',
                execution_reports=list(self.execution_reports.values()),
                metrics={
                    'slices_executed': self.slices_executed,
                    'slices_total': self.slices_total,
                    'execution_time_minutes': (datetime.utcnow() - self.start_time).total_seconds() / 60,
                    'implementation_shortfall': self._calculate_implementation_shortfall()
                }
            )
        except Exception as e:
            self.logger.error(f"Error executing Implementation Shortfall algorithm: {str(e)}")
            
            # Trigger failed callbacks
            self._trigger_callbacks('failed', {
                'algorithm_id': self.algorithm_id,
                'order_id': order.client_order_id,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return ExecutionResult(
                algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id,
                status='FAILED',
                execution_reports=list(self.execution_reports.values()),
                metrics={'error': str(e)}
            )
        finally:
            self.is_executing = False
    
    async def cancel(self) -> bool:
        """
        Cancel the current execution.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.is_executing:
            return True  # Nothing to cancel
        
        self.is_cancelled = True
        
        # Cancel the execution task
        if self.execution_task and not self.execution_task.done():
            self.execution_task.cancel()
            
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        
        # Trigger cancelled callbacks
        self._trigger_callbacks('cancelled', {
            'algorithm_id': self.algorithm_id,
            'order_id': self.original_order.client_order_id if self.original_order else None,
            'slices_executed': self.slices_executed,
            'slices_total': self.slices_total,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the execution.
        
        Returns:
            Dictionary with the current status
        """
        now = datetime.utcnow()
        total_quantity = self.original_order.quantity if self.original_order else 0
        filled_quantity = sum(
            report.filled_quantity 
            for report in self.execution_reports.values()
            if hasattr(report, 'filled_quantity')
        )
        
        time_elapsed = (now - self.start_time).total_seconds() if self.start_time else 0
        time_total = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        time_percentage = (time_elapsed / time_total * 100) if time_total > 0 else 0
        
        return {
            'algorithm_id': self.algorithm_id,
            'is_executing': self.is_executing,
            'is_cancelled': self.is_cancelled,
            'original_order_id': self.original_order.client_order_id if self.original_order else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'next_slice_time': self.next_slice_time.isoformat() if self.next_slice_time else None,
            'slices_executed': self.slices_executed,
            'slices_total': self.slices_total,
            'slices_percentage': (self.slices_executed / self.slices_total * 100) if self.slices_total > 0 else 0,
            'time_elapsed_seconds': time_elapsed,
            'time_total_seconds': time_total,
            'time_percentage': time_percentage,
            'total_quantity': total_quantity,
            'filled_quantity': filled_quantity,
            'quantity_percentage': (filled_quantity / total_quantity * 100) if total_quantity > 0 else 0,
            'child_orders': len(self.child_orders),
            'execution_reports': len(self.execution_reports),
            'implementation_shortfall': self._calculate_implementation_shortfall()
        }
    
    async def _get_market_conditions(self, instrument: str) -> Dict[str, Any]:
        """
        Get current market conditions for an instrument.
        
        Args:
            instrument: The instrument to get market conditions for
            
        Returns:
            Dictionary with market conditions
        """
        # Default market conditions
        conditions = {
            'price': None,
            'spread': 0.0001,  # 1 pip for major forex pairs
            'volatility': 0.001,  # 0.1% volatility
            'avg_daily_volume': 1000000,  # 1M units
            'market_regime': 'normal'
        }
        
        # If market data service is available, use it to get market conditions
        if self.market_data_service:
            try:
                if hasattr(self.market_data_service, 'get_market_conditions'):
                    market_data = await self.market_data_service.get_market_conditions(instrument)
                    if market_data:
                        conditions.update(market_data)
                else:
                    # Get individual metrics if available
                    if hasattr(self.market_data_service, 'get_price'):
                        conditions['price'] = await self.market_data_service.get_price(instrument)
                    
                    if hasattr(self.market_data_service, 'get_spread'):
                        conditions['spread'] = await self.market_data_service.get_spread(instrument)
                    
                    if hasattr(self.market_data_service, 'get_volatility'):
                        conditions['volatility'] = await self.market_data_service.get_volatility(instrument)
                    
                    if hasattr(self.market_data_service, 'get_avg_daily_volume'):
                        conditions['avg_daily_volume'] = await self.market_data_service.get_avg_daily_volume(instrument)
                    
                    if hasattr(self.market_data_service, 'get_market_regime'):
                        conditions['market_regime'] = await self.market_data_service.get_market_regime(instrument)
            except Exception as e:
                self.logger.error(f"Error getting market conditions: {str(e)}")
        
        # If price is still None, try to get it from broker adapters
        if conditions['price'] is None:
            for broker_name, adapter in self.broker_adapters.items():
                try:
                    if hasattr(adapter, 'get_price') and callable(adapter.get_price):
                        price = await adapter.get_price(instrument)
                        if price is not None:
                            conditions['price'] = price
                            break
                except Exception as e:
                    self.logger.error(f"Error getting price from broker {broker_name}: {str(e)}")
        
        return conditions
    
    def _calculate_slice_sizes(self, order: OrderRequest) -> List[float]:
        """
        Calculate sizes for all slices based on market impact model.
        
        Args:
            order: The original order
            
        Returns:
            List of slice sizes
        """
        total_quantity = order.quantity
        slice_sizes = []
        
        # Get market conditions
        volatility = self.market_conditions.get('volatility', 0.001)
        market_regime = self.market_conditions.get('market_regime', 'normal')
        
        # Adjust slice distribution based on market regime
        if market_regime == 'volatile':
            # In volatile markets, front-load the execution
            # Use a decreasing exponential distribution
            factor = 0.8  # Decay factor
            weights = [factor ** i for i in range(self.slices_total)]
        elif market_regime == 'trending':
            # In trending markets, use a more uniform distribution
            # with slight front-loading
            weights = [1.0 - (i * 0.5 / self.slices_total) for i in range(self.slices_total)]
        else:
            # In normal markets, use a balanced distribution
            # with slightly larger initial and final slices (U-shaped)
            weights = []
            for i in range(self.slices_total):
                # Position in [0, 1] range
                x = i / (self.slices_total - 1) if self.slices_total > 1 else 0.5
                # U-shaped function
                weight = 0.8 + 0.4 * ((x - 0.5) ** 2)
                weights.append(weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate slice sizes
        remaining = total_quantity
        for i in range(self.slices_total):
            # Last slice gets all remaining quantity
            if i == self.slices_total - 1:
                size = remaining
            else:
                # Calculate size based on weight
                size = total_quantity * weights[i]
                # Ensure minimum slice size
                size = max(self.min_slice_size, size)
                # Ensure we don't exceed remaining quantity
                size = min(size, remaining * 0.9)  # Leave 10% buffer
            
            slice_sizes.append(round(size, 2))
            remaining -= size
        
        return slice_sizes
    
    def _calculate_implementation_shortfall(self) -> Dict[str, Any]:
        """
        Calculate the implementation shortfall metrics.
        
        Returns:
            Dictionary with implementation shortfall metrics
        """
        if not self.decision_price or not self.execution_reports:
            return {
                'shortfall_pips': 0,
                'shortfall_percentage': 0,
                'shortfall_cost': 0
            }
        
        # Calculate volume-weighted average execution price
        total_value = 0.0
        total_quantity = 0.0
        
        for report in self.execution_reports.values():
            if hasattr(report, 'executed_price') and hasattr(report, 'filled_quantity'):
                if report.executed_price and report.filled_quantity > 0:
                    total_value += report.executed_price * report.filled_quantity
                    total_quantity += report.filled_quantity
        
        if total_quantity <= 0:
            return {
                'shortfall_pips': 0,
                'shortfall_percentage': 0,
                'shortfall_cost': 0
            }
        
        # Calculate average execution price
        avg_execution_price = total_value / total_quantity
        
        # Calculate shortfall
        if self.original_order.direction.value.upper() == 'BUY':
            # For buy orders, shortfall is execution price - decision price
            shortfall = avg_execution_price - self.decision_price
        else:
            # For sell orders, shortfall is decision price - execution price
            shortfall = self.decision_price - avg_execution_price
        
        # Calculate shortfall in pips (assuming 4 decimal places for forex)
        shortfall_pips = shortfall * 10000
        
        # Calculate shortfall as percentage
        shortfall_percentage = (shortfall / self.decision_price) * 100 if self.decision_price > 0 else 0
        
        # Calculate shortfall cost
        shortfall_cost = shortfall * total_quantity
        
        return {
            'decision_price': self.decision_price,
            'avg_execution_price': avg_execution_price,
            'shortfall_pips': shortfall_pips,
            'shortfall_percentage': shortfall_percentage,
            'shortfall_cost': shortfall_cost,
            'initial_market_impact': self.initial_market_impact,
            'total_quantity': total_quantity
        }
    
    async def _execute_slices(self, 
                            order: OrderRequest, 
                            slice_interval: timedelta) -> ExecutionResult:
        """
        Execute all slices of the order over time.
        
        Args:
            order: The original order
            slice_interval: Time interval between slices
            
        Returns:
            ExecutionResult with details of the execution
        """
        # Calculate slice sizes
        slice_sizes = self._calculate_slice_sizes(order)
        
        for slice_index in range(self.slices_total):
            if self.is_cancelled:
                self.logger.info("Implementation Shortfall execution cancelled")
                break
            
            # Calculate time until next slice
            now = datetime.utcnow()
            self.next_slice_time = self.start_time + slice_interval * slice_index
            
            if now < self.next_slice_time:
                wait_time = (self.next_slice_time - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Check if market conditions have changed significantly
            if slice_index > 0 and slice_index % 3 == 0:  # Check every 3 slices
                new_conditions = await self._get_market_conditions(order.instrument)
                
                # Check for significant changes
                if self._is_significant_market_change(new_conditions):
                    self.logger.info("Significant market change detected, recalculating slice sizes")
                    
                    # Update market conditions
                    self.market_conditions = new_conditions
                    
                    # Recalculate remaining slice sizes
                    remaining_quantity = order.quantity - sum(
                        self.child_orders[order_id].get('quantity', 0)
                        for order_id in self.child_orders
                    )
                    
                    # Create a temporary order with remaining quantity
                    temp_order = OrderRequest(
                        instrument=order.instrument,
                        order_type=order.order_type,
                        direction=order.direction,
                        quantity=remaining_quantity,
                        price=order.price,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        client_order_id=f"{order.client_order_id}_temp"
                    )
                    
                    # Recalculate slice sizes for remaining slices
                    new_sizes = self._calculate_slice_sizes(temp_order)
                    
                    # Update slice sizes for remaining slices
                    remaining_slices = self.slices_total - slice_index
                    if remaining_slices > 0 and len(new_sizes) >= remaining_slices:
                        slice_sizes[slice_index:] = new_sizes[:remaining_slices]
            
            # Execute this slice
            try:
                slice_size = slice_sizes[slice_index]
                
                # Place the child order
                execution_report = await self._place_slice_order(order, slice_index, slice_size)
                
                # Store the execution report
                if execution_report:
                    self.execution_reports[execution_report.client_order_id] = execution_report
                
                # Update state
                self.slices_executed += 1
                
                # Trigger progress callbacks
                self._trigger_callbacks('progress', {
                    'algorithm_id': self.algorithm_id,
                    'order_id': order.client_order_id,
                    'slice_index': slice_index,
                    'slices_total': self.slices_total,
                    'slice_size': slice_size,
                    'timestamp': datetime.utcnow().isoformat(),
                    'implementation_shortfall': self._calculate_implementation_shortfall()
                })
                
            except Exception as e:
                self.logger.error(f"Error executing slice {slice_index}: {str(e)}")
        
        # Determine overall status
        if self.is_cancelled:
            status = 'CANCELLED'
        elif self.slices_executed == 0:
            status = 'FAILED'
        elif self.slices_executed < self.slices_total:
            status = 'PARTIAL'
        else:
            status = 'COMPLETED'
        
        # Calculate execution metrics
        execution_time = (datetime.utcnow() - self.start_time).total_seconds() / 60
        implementation_shortfall = self._calculate_implementation_shortfall()
        
        # Create execution result
        result = ExecutionResult(
            algorithm_id=self.algorithm_id,
            original_order_id=order.client_order_id,
            status=status,
            execution_reports=list(self.execution_reports.values()),
            metrics={
                'slices_executed': self.slices_executed,
                'slices_total': self.slices_total,
                'execution_time_minutes': execution_time,
                'implementation_shortfall': implementation_shortfall
            }
        )
        
        # Trigger completed callbacks
        self._trigger_callbacks('completed', result.to_dict())
        
        return result
    
    def _is_significant_market_change(self, new_conditions: Dict[str, Any]) -> bool:
        """
        Check if there has been a significant change in market conditions.
        
        Args:
            new_conditions: New market conditions
            
        Returns:
            True if there has been a significant change, False otherwise
        """
        # Check for significant changes in key metrics
        
        # Volatility change
        old_volatility = self.market_conditions.get('volatility', 0.001)
        new_volatility = new_conditions.get('volatility', 0.001)
        
        if new_volatility > old_volatility * 1.5:
            # Volatility increased by 50% or more
            return True
        
        # Spread change
        old_spread = self.market_conditions.get('spread', 0.0001)
        new_spread = new_conditions.get('spread', 0.0001)
        
        if new_spread > old_spread * 2:
            # Spread doubled or more
            return True
        
        # Market regime change
        old_regime = self.market_conditions.get('market_regime', 'normal')
        new_regime = new_conditions.get('market_regime', 'normal')
        
        if old_regime != new_regime:
            # Market regime changed
            return True
        
        # Price change
        old_price = self.market_conditions.get('price')
        new_price = new_conditions.get('price')
        
        if old_price and new_price:
            price_change_pct = abs(new_price - old_price) / old_price
            
            if price_change_pct > 0.005:  # 0.5% price change
                # Significant price change
                return True
        
        return False
    
    async def _place_slice_order(self, 
                               parent_order: OrderRequest, 
                               slice_index: int, 
                               quantity: float) -> Optional[ExecutionReport]:
        """
        Place an order for a single slice.
        
        Args:
            parent_order: The original parent order
            slice_index: Index of the current slice (0-based)
            quantity: Quantity for this slice
            
        Returns:
            Execution report for the slice order, or None if failed
        """
        # Determine which broker to use
        broker_name = self.default_broker
        if not broker_name or broker_name not in self.broker_adapters:
            # Use the first available broker if default not specified or invalid
            if not self.broker_adapters:
                self.logger.error("No broker adapters available")
                return None
            broker_name = next(iter(self.broker_adapters))
        
        adapter = self.broker_adapters[broker_name]
        
        # Create child order
        child_order_id = f"{parent_order.client_order_id}_IS_{slice_index}_{uuid.uuid4().hex[:8]}"
        
        child_order = OrderRequest(
            instrument=parent_order.instrument,
            order_type=parent_order.order_type,
            direction=parent_order.direction,
            quantity=quantity,
            price=parent_order.price,
            stop_loss=parent_order.stop_loss,
            take_profit=parent_order.take_profit,
            client_order_id=child_order_id
        )
        
        # Store child order details
        self.child_orders[child_order_id] = {
            'broker_name': broker_name,
            'quantity': quantity,
            'slice_index': slice_index,
            'parent_order_id': parent_order.client_order_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Place the order
        self.logger.info(
            f"Placing Implementation Shortfall slice {slice_index + 1}/{self.slices_total} order {child_order_id}: "
            f"{child_order.instrument} {child_order.direction.value} {child_order.quantity}"
        )
        
        try:
            execution_report = await adapter.place_order(child_order)
            return execution_report
        except Exception as e:
            self.logger.error(f"Error placing Implementation Shortfall slice order: {str(e)}")
            return None
