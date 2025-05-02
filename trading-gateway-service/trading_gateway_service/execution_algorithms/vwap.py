"""
Volume-Weighted Average Price (VWAP) Algorithm.

This module implements a VWAP algorithm that executes orders based on
historical volume profiles to achieve a price close to the volume-weighted
average price.
"""

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from ..interfaces.broker_adapter_interface import (
    BrokerAdapterInterface,
    OrderRequest,
    ExecutionReport,
    OrderStatus,
)
from .base_algorithm import BaseExecutionAlgorithm, ExecutionResult


class VWAPAlgorithm(BaseExecutionAlgorithm):
    """
    Volume-Weighted Average Price (VWAP) algorithm.
    
    This algorithm executes orders based on historical volume profiles
    to achieve a price close to the volume-weighted average price.
    """
    
    def __init__(self, 
                 broker_adapters: Dict[str, BrokerAdapterInterface],
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VWAP algorithm.
        
        Args:
            broker_adapters: Dictionary of broker adapters by name
            logger: Logger instance
            config: Algorithm-specific configuration
        """
        super().__init__(broker_adapters, logger, config)
        
        # Default configuration
        self.default_broker = self.config.get('default_broker')
        self.duration_minutes = self.config.get('duration_minutes', 60)  # Default: 1 hour
        self.num_slices = self.config.get('num_slices', 12)  # Default: 12 slices
        self.min_slice_size = self.config.get('min_slice_size', 1000)  # Minimum size per slice
        self.volume_profile_source = self.config.get('volume_profile_source', 'historical')
        self.market_data_service = self.config.get('market_data_service')
        
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
        self.volume_profile = None
    
    async def execute(self, order: OrderRequest) -> ExecutionResult:
        """
        Execute the VWAP algorithm for the given order.
        
        Args:
            order: The order to execute
            
        Returns:
            ExecutionResult with details of the execution
        """
        if self.is_executing:
            self.logger.warning("VWAP algorithm is already executing an order")
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
        self.end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        self.next_slice_time = self.start_time
        
        try:
            # Get volume profile for the instrument
            self.volume_profile = await self._get_volume_profile(order.instrument)
            
            if not self.volume_profile:
                self.logger.error(f"Failed to get volume profile for {order.instrument}")
                return ExecutionResult(
                    algorithm_id=self.algorithm_id,
                    original_order_id=order.client_order_id,
                    status='FAILED',
                    metrics={'error': 'Failed to get volume profile'}
                )
            
            # Calculate slice details
            self.slices_total = self.num_slices
            slice_interval = timedelta(minutes=self.duration_minutes / self.slices_total)
            
            # Trigger started callbacks
            self._trigger_callbacks('started', {
                'algorithm_id': self.algorithm_id,
                'order_id': order.client_order_id,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'slices': self.slices_total,
                'slice_interval_seconds': slice_interval.total_seconds()
            })
            
            # Start the execution task
            self.execution_task = asyncio.create_task(self._execute_slices(order, slice_interval))
            
            # Wait for the execution task to complete
            result = await self.execution_task
            return result
            
        except asyncio.CancelledError:
            self.logger.info("VWAP execution task was cancelled")
            
            # Create execution result for cancelled execution
            return ExecutionResult(
                algorithm_id=self.algorithm_id,
                original_order_id=order.client_order_id,
                status='CANCELLED',
                execution_reports=list(self.execution_reports.values()),
                metrics={
                    'slices_executed': self.slices_executed,
                    'slices_total': self.slices_total,
                    'duration_minutes': self.duration_minutes,
                    'execution_time_minutes': (datetime.utcnow() - self.start_time).total_seconds() / 60
                }
            )
        except Exception as e:
            self.logger.error(f"Error executing VWAP algorithm: {str(e)}")
            
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
            'execution_reports': len(self.execution_reports)
        }
    
    async def _get_volume_profile(self, instrument: str) -> List[float]:
        """
        Get the volume profile for an instrument.
        
        Args:
            instrument: The instrument to get the volume profile for
            
        Returns:
            List of volume percentages for each slice
        """
        try:
            if self.volume_profile_source == 'historical':
                # Use historical volume profile
                return await self._get_historical_volume_profile(instrument)
            elif self.volume_profile_source == 'predicted':
                # Use predicted volume profile
                return await self._get_predicted_volume_profile(instrument)
            elif self.volume_profile_source == 'realtime':
                # Use real-time volume profile
                return await self._get_realtime_volume_profile(instrument)
            else:
                # Use default volume profile
                return self._get_default_volume_profile()
        except Exception as e:
            self.logger.error(f"Error getting volume profile: {str(e)}")
            # Fall back to default profile
            return self._get_default_volume_profile()
    
    async def _get_historical_volume_profile(self, instrument: str) -> List[float]:
        """
        Get historical volume profile for an instrument.
        
        Args:
            instrument: The instrument to get the volume profile for
            
        Returns:
            List of volume percentages for each slice
        """
        # If market data service is available, use it to get historical volume
        if self.market_data_service and hasattr(self.market_data_service, 'get_historical_volume'):
            try:
                # Get current time components
                now = datetime.utcnow()
                current_hour = now.hour
                
                # Get historical volume for the same time period on previous days
                volume_data = await self.market_data_service.get_historical_volume(
                    instrument=instrument,
                    period='1d',  # Daily data
                    lookback_days=20,  # Look back 20 days
                    hour_of_day=current_hour
                )
                
                if volume_data and len(volume_data) > 0:
                    # Calculate average volume profile
                    total_periods = len(volume_data)
                    volume_profile = []
                    
                    # Divide each day's data into the required number of slices
                    for day_data in volume_data:
                        day_slices = self._divide_into_slices(day_data, self.slices_total)
                        
                        if not volume_profile:
                            volume_profile = day_slices
                        else:
                            # Add to running total
                            volume_profile = [a + b for a, b in zip(volume_profile, day_slices)]
                    
                    # Calculate average
                    volume_profile = [v / total_periods for v in volume_profile]
                    
                    # Normalize to percentages
                    total_volume = sum(volume_profile)
                    if total_volume > 0:
                        return [v / total_volume for v in volume_profile]
            except Exception as e:
                self.logger.error(f"Error getting historical volume profile: {str(e)}")
        
        # Fall back to default profile
        return self._get_default_volume_profile()
    
    async def _get_predicted_volume_profile(self, instrument: str) -> List[float]:
        """
        Get predicted volume profile for an instrument.
        
        Args:
            instrument: The instrument to get the volume profile for
            
        Returns:
            List of volume percentages for each slice
        """
        # If market data service is available, use it to get predicted volume
        if self.market_data_service and hasattr(self.market_data_service, 'get_predicted_volume'):
            try:
                # Get predicted volume for the execution period
                predicted_volume = await self.market_data_service.get_predicted_volume(
                    instrument=instrument,
                    start_time=self.start_time,
                    end_time=self.end_time,
                    num_slices=self.slices_total
                )
                
                if predicted_volume and len(predicted_volume) == self.slices_total:
                    # Normalize to percentages
                    total_volume = sum(predicted_volume)
                    if total_volume > 0:
                        return [v / total_volume for v in predicted_volume]
            except Exception as e:
                self.logger.error(f"Error getting predicted volume profile: {str(e)}")
        
        # Fall back to default profile
        return self._get_default_volume_profile()
    
    async def _get_realtime_volume_profile(self, instrument: str) -> List[float]:
        """
        Get real-time volume profile for an instrument.
        
        Args:
            instrument: The instrument to get the volume profile for
            
        Returns:
            List of volume percentages for each slice
        """
        # If market data service is available, use it to get real-time volume
        if self.market_data_service and hasattr(self.market_data_service, 'get_realtime_volume'):
            try:
                # Get real-time volume data
                realtime_volume = await self.market_data_service.get_realtime_volume(
                    instrument=instrument,
                    lookback_minutes=60  # Look back 1 hour
                )
                
                if realtime_volume and len(realtime_volume) > 0:
                    # Divide into slices
                    volume_profile = self._divide_into_slices(realtime_volume, self.slices_total)
                    
                    # Normalize to percentages
                    total_volume = sum(volume_profile)
                    if total_volume > 0:
                        return [v / total_volume for v in volume_profile]
            except Exception as e:
                self.logger.error(f"Error getting real-time volume profile: {str(e)}")
        
        # Fall back to default profile
        return self._get_default_volume_profile()
    
    def _get_default_volume_profile(self) -> List[float]:
        """
        Get a default volume profile.
        
        Returns:
            List of volume percentages for each slice
        """
        # Create a U-shaped volume profile (common in forex markets)
        # Higher volume at the beginning and end of the period
        profile = []
        
        for i in range(self.slices_total):
            # Calculate position in the range [0, 1]
            x = i / (self.slices_total - 1) if self.slices_total > 1 else 0.5
            
            # U-shaped function: y = a * (x - 0.5)^2 + b
            # Where a and b are chosen to make the profile sum to 1
            y = 2 * (x - 0.5) ** 2 + 0.5
            
            profile.append(y)
        
        # Normalize to make sum = 1
        total = sum(profile)
        return [v / total for v in profile]
    
    def _divide_into_slices(self, data: List[float], num_slices: int) -> List[float]:
        """
        Divide a data series into the specified number of slices.
        
        Args:
            data: The data to divide
            num_slices: Number of slices to create
            
        Returns:
            List of slice values
        """
        if not data:
            return [0] * num_slices
        
        # Convert to numpy array for easier manipulation
        data_array = np.array(data)
        
        # Calculate slice size
        slice_size = len(data_array) / num_slices
        
        # Create slices
        slices = []
        for i in range(num_slices):
            start_idx = int(i * slice_size)
            end_idx = int((i + 1) * slice_size) if i < num_slices - 1 else len(data_array)
            
            # Sum the values in this slice
            slice_sum = np.sum(data_array[start_idx:end_idx])
            slices.append(float(slice_sum))
        
        return slices
    
    def _calculate_slice_size(self, 
                            order: OrderRequest, 
                            slice_index: int) -> float:
        """
        Calculate the size for a specific slice based on the volume profile.
        
        Args:
            order: The original order
            slice_index: Index of the current slice (0-based)
            
        Returns:
            Size for the slice
        """
        # Get the volume percentage for this slice
        volume_percentage = self.volume_profile[slice_index]
        
        # Calculate the slice size
        slice_size = order.quantity * volume_percentage
        
        # Ensure minimum slice size
        slice_size = max(self.min_slice_size, slice_size)
        
        # Ensure we don't exceed the remaining quantity
        remaining = order.quantity - sum(
            self.child_orders[order_id].get('quantity', 0)
            for order_id in self.child_orders
        )
        
        # Last slice gets all remaining quantity
        if slice_index == self.slices_total - 1:
            slice_size = remaining
        else:
            # Ensure we leave enough for remaining slices
            slice_size = min(slice_size, remaining * 0.9)  # Leave 10% buffer
        
        # Round to 2 decimal places
        return round(slice_size, 2)
    
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
        for slice_index in range(self.slices_total):
            if self.is_cancelled:
                self.logger.info("VWAP execution cancelled")
                break
            
            # Calculate time until next slice
            now = datetime.utcnow()
            self.next_slice_time = self.start_time + slice_interval * slice_index
            
            if now < self.next_slice_time:
                wait_time = (self.next_slice_time - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Execute this slice
            try:
                slice_size = self._calculate_slice_size(order, slice_index)
                
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
                    'volume_percentage': self.volume_profile[slice_index] if self.volume_profile else 0,
                    'timestamp': datetime.utcnow().isoformat()
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
        
        # Create execution result
        result = ExecutionResult(
            algorithm_id=self.algorithm_id,
            original_order_id=order.client_order_id,
            status=status,
            execution_reports=list(self.execution_reports.values()),
            metrics={
                'slices_executed': self.slices_executed,
                'slices_total': self.slices_total,
                'duration_minutes': self.duration_minutes,
                'execution_time_minutes': execution_time,
                'volume_profile': self.volume_profile
            }
        )
        
        # Trigger completed callbacks
        self._trigger_callbacks('completed', result.to_dict())
        
        return result
    
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
        child_order_id = f"{parent_order.client_order_id}_VWAP_{slice_index}_{uuid.uuid4().hex[:8]}"
        
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
            f"Placing VWAP slice {slice_index + 1}/{self.slices_total} order {child_order_id}: "
            f"{child_order.instrument} {child_order.direction.value} {child_order.quantity}"
        )
        
        try:
            execution_report = await adapter.place_order(child_order)
            return execution_report
        except Exception as e:
            self.logger.error(f"Error placing VWAP slice order: {str(e)}")
            return None
