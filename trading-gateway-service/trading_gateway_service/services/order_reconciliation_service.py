"""
Order Reconciliation Service.

This service maintains consistent state between internal order management systems
and external broker systems, detecting and resolving discrepancies.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import time
from enum import Enum

from trading_gateway_service.interfaces.broker_adapter import (
    BrokerAdapter, OrderRequest, OrderType, OrderDirection, OrderStatus,
    ExecutionReport
)

logger = logging.getLogger(__name__)

class ReconciliationType(Enum):
    """Types of reconciliation operations."""
    ORDERS = "orders"
    POSITIONS = "positions"
    ACCOUNT = "account"


class ReconciliationStatus(Enum):
    """Status of a reconciliation operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class OrderReconciliationService:
    """
    Service to maintain consistent state between internal systems and broker.
    
    Key responsibilities:
    - Periodic reconciliation between internal order state and broker
    - Detection of missing or inconsistent orders
    - Resolution of discrepancies through predefined strategies
    - Detailed logging of reconciliation operations
    - Exception handling for broker communication issues
    """
    
    def __init__(self, 
                broker_adapter: BrokerAdapter,
                internal_order_store: Any,
                reconciliation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reconciliation service.
        
        Args:
            broker_adapter: The broker adapter to interface with
            internal_order_store: Internal order management system
            reconciliation_config: Configuration parameters for reconciliation
        """
        self.broker_adapter = broker_adapter
        self.internal_order_store = internal_order_store
        self.config = reconciliation_config or {}
        
        # Configure reconciliation intervals
        self.order_reconciliation_interval = self.config.get("order_reconciliation_interval_sec", 60)
        self.position_reconciliation_interval = self.config.get("position_reconciliation_interval_sec", 300)
        self.account_reconciliation_interval = self.config.get("account_reconciliation_interval_sec", 600)
        
        # State variables
        self.running = False
        self.reconciliation_tasks = {}
        self.ongoing_reconciliations = {}
        self.last_reconciliation_times = {
            ReconciliationType.ORDERS: datetime.now() - timedelta(days=1),
            ReconciliationType.POSITIONS: datetime.now() - timedelta(days=1),
            ReconciliationType.ACCOUNT: datetime.now() - timedelta(days=1),
        }
        
        # Reconciliation statistics
        self.reconciliation_stats = {
            ReconciliationType.ORDERS: {
                "total": 0,
                "discrepancies": 0,
                "resolved": 0,
                "failed": 0
            },
            ReconciliationType.POSITIONS: {
                "total": 0,
                "discrepancies": 0,
                "resolved": 0,
                "failed": 0
            },
            ReconciliationType.ACCOUNT: {
                "total": 0,
                "discrepancies": 0,
                "resolved": 0,
                "failed": 0
            }
        }
        
        logger.info("OrderReconciliationService initialized")
    
    async def start(self) -> None:
        """Start the reconciliation service."""
        if self.running:
            logger.warning("Reconciliation service already running")
            return
            
        logger.info("Starting reconciliation service")
        self.running = True
        
        # Start the reconciliation loops
        self.reconciliation_tasks[ReconciliationType.ORDERS] = asyncio.create_task(
            self._reconciliation_loop(ReconciliationType.ORDERS, self.order_reconciliation_interval)
        )
        self.reconciliation_tasks[ReconciliationType.POSITIONS] = asyncio.create_task(
            self._reconciliation_loop(ReconciliationType.POSITIONS, self.position_reconciliation_interval)
        )
        self.reconciliation_tasks[ReconciliationType.ACCOUNT] = asyncio.create_task(
            self._reconciliation_loop(ReconciliationType.ACCOUNT, self.account_reconciliation_interval)
        )
    
    async def stop(self) -> None:
        """Stop the reconciliation service."""
        if not self.running:
            logger.warning("Reconciliation service not running")
            return
            
        logger.info("Stopping reconciliation service")
        self.running = False
        
        # Cancel all tasks
        for task_type, task in self.reconciliation_tasks.items():
            logger.info(f"Cancelling {task_type.value} reconciliation task")
            task.cancel()
            
        # Wait for tasks to complete
        for task_type, task in self.reconciliation_tasks.items():
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"{task_type.value} reconciliation task cancelled")
                
        self.reconciliation_tasks = {}
        logger.info("Reconciliation service stopped")
    
    async def _reconciliation_loop(self, reconciliation_type: ReconciliationType, interval_sec: int) -> None:
        """
        Run a reconciliation loop for a specific type at given intervals.
        
        Args:
            reconciliation_type: The type of reconciliation to perform
            interval_sec: The interval between reconciliations in seconds
        """
        logger.info(f"Starting {reconciliation_type.value} reconciliation loop")
        
        try:
            while self.running:
                try:
                    # Perform reconciliation
                    await self.perform_reconciliation(reconciliation_type)
                    
                    # Update last reconciliation time
                    self.last_reconciliation_times[reconciliation_type] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error in {reconciliation_type.value} reconciliation: {e}")
                    self.reconciliation_stats[reconciliation_type]["failed"] += 1
                
                # Sleep until next interval
                await asyncio.sleep(interval_sec)
                
        except asyncio.CancelledError:
            logger.info(f"{reconciliation_type.value} reconciliation loop cancelled")
            raise
    
    async def perform_reconciliation(self, reconciliation_type: ReconciliationType) -> Dict[str, Any]:
        """
        Perform a reconciliation of the specified type.
        
        Args:
            reconciliation_type: The type of reconciliation to perform
            
        Returns:
            Dictionary with reconciliation results
        """
        logger.info(f"Starting {reconciliation_type.value} reconciliation")
        
        # Update status
        reconciliation_id = f"{reconciliation_type.value}-{int(time.time())}"
        self.ongoing_reconciliations[reconciliation_id] = {
            "type": reconciliation_type,
            "status": ReconciliationStatus.IN_PROGRESS,
            "start_time": datetime.now(),
            "end_time": None,
            "discrepancies": [],
            "resolution_actions": []
        }
        
        # Increment total count
        self.reconciliation_stats[reconciliation_type]["total"] += 1
        
        try:
            if reconciliation_type == ReconciliationType.ORDERS:
                result = await self._reconcile_orders()
            elif reconciliation_type == ReconciliationType.POSITIONS:
                result = await self._reconcile_positions()
            elif reconciliation_type == ReconciliationType.ACCOUNT:
                result = await self._reconcile_account()
            else:
                raise ValueError(f"Unknown reconciliation type: {reconciliation_type}")
            
            # Update statistics
            if result["discrepancies"]:
                self.reconciliation_stats[reconciliation_type]["discrepancies"] += len(result["discrepancies"])
                if result["resolved"]:
                    self.reconciliation_stats[reconciliation_type]["resolved"] += len(result["resolved"])
            
            # Update status
            self.ongoing_reconciliations[reconciliation_id]["status"] = ReconciliationStatus.COMPLETED
            self.ongoing_reconciliations[reconciliation_id]["end_time"] = datetime.now()
            self.ongoing_reconciliations[reconciliation_id]["discrepancies"] = result["discrepancies"]
            self.ongoing_reconciliations[reconciliation_id]["resolution_actions"] = result["resolution_actions"]
            
            logger.info(f"Completed {reconciliation_type.value} reconciliation: {len(result['discrepancies'])} discrepancies, {len(result['resolved'])} resolved")
            
            return {
                "reconciliation_id": reconciliation_id,
                "type": reconciliation_type.value,
                "status": "completed",
                "discrepancies_count": len(result["discrepancies"]),
                "resolved_count": len(result["resolved"]),
                "details": result
            }
            
        except Exception as e:
            logger.error(f"Failed to complete {reconciliation_type.value} reconciliation: {e}")
            
            # Update status
            self.ongoing_reconciliations[reconciliation_id]["status"] = ReconciliationStatus.FAILED
            self.ongoing_reconciliations[reconciliation_id]["end_time"] = datetime.now()
            self.ongoing_reconciliations[reconciliation_id]["error"] = str(e)
            
            return {
                "reconciliation_id": reconciliation_id,
                "type": reconciliation_type.value,
                "status": "failed",
                "error": str(e)
            }
    
    async def _reconcile_orders(self) -> Dict[str, Any]:
        """
        Reconcile orders between internal system and broker.
        
        Returns:
            Dictionary with reconciliation results
        """
        # Get orders from both systems
        try:
            broker_orders = await self.broker_adapter.get_orders()
            internal_orders = await self.internal_order_store.get_orders()
            
            logger.info(f"Reconciling orders: {len(broker_orders)} broker orders, {len(internal_orders)} internal orders")
            
            # Create dictionaries for easy lookup
            broker_orders_dict = {order["order_id"]: order for order in broker_orders}
            internal_orders_dict = {order["order_id"]: order for order in internal_orders}
            
            # Find discrepancies
            discrepancies = []
            
            # 1. Orders in broker but not in internal system
            for order_id, broker_order in broker_orders_dict.items():
                if order_id not in internal_orders_dict:
                    discrepancies.append({
                        "type": "missing_internal",
                        "order_id": order_id,
                        "broker_order": broker_order
                    })
            
            # 2. Orders in internal system but not in broker
            for order_id, internal_order in internal_orders_dict.items():
                if order_id not in broker_orders_dict:
                    discrepancies.append({
                        "type": "missing_broker",
                        "order_id": order_id,
                        "internal_order": internal_order
                    })
            
            # 3. Orders in both systems but with different status or filled amounts
            for order_id in set(broker_orders_dict.keys()) & set(internal_orders_dict.keys()):
                broker_order = broker_orders_dict[order_id]
                internal_order = internal_orders_dict[order_id]
                
                # Check for status mismatch
                if broker_order["status"] != internal_order["status"]:
                    discrepancies.append({
                        "type": "status_mismatch",
                        "order_id": order_id,
                        "broker_status": broker_order["status"],
                        "internal_status": internal_order["status"],
                        "broker_order": broker_order,
                        "internal_order": internal_order
                    })
                
                # Check for filled amount mismatch
                if broker_order["filled_size"] != internal_order["filled_size"]:
                    discrepancies.append({
                        "type": "fill_mismatch",
                        "order_id": order_id,
                        "broker_filled": broker_order["filled_size"],
                        "internal_filled": internal_order["filled_size"],
                        "broker_order": broker_order,
                        "internal_order": internal_order
                    })
            
            # Resolve discrepancies
            resolved = await self._resolve_order_discrepancies(discrepancies)
            
            return {
                "discrepancies": discrepancies,
                "resolved": resolved,
                "resolution_actions": [r["action"] for r in resolved]
            }
            
        except Exception as e:
            logger.error(f"Error reconciling orders: {e}")
            raise
    
    async def _resolve_order_discrepancies(self, discrepancies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve order discrepancies using predefined strategies.
        
        Args:
            discrepancies: List of discrepancies to resolve
            
        Returns:
            List of resolved discrepancies with actions taken
        """
        resolved = []
        
        for discrepancy in discrepancies:
            try:
                discrepancy_type = discrepancy["type"]
                order_id = discrepancy["order_id"]
                
                if discrepancy_type == "missing_internal":
                    # Order exists in broker but not in internal system
                    # Strategy: Add to internal system
                    broker_order = discrepancy["broker_order"]
                    await self.internal_order_store.add_external_order(broker_order)
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "added_to_internal",
                        "success": True
                    })
                    
                    logger.info(f"Added broker order {order_id} to internal system")
                    
                elif discrepancy_type == "missing_broker":
                    # Order exists in internal system but not in broker
                    # Strategy: Mark as inconsistent or cancel in internal system depending on config
                    internal_order = discrepancy["internal_order"]
                    
                    if self.config.get("auto_cancel_missing_broker_orders", False):
                        await self.internal_order_store.cancel_order(order_id, "Order not found at broker")
                        
                        resolved.append({
                            "discrepancy": discrepancy,
                            "action": "cancelled_internal",
                            "success": True
                        })
                        
                        logger.info(f"Cancelled internal order {order_id} not found at broker")
                    else:
                        await self.internal_order_store.update_order_status(
                            order_id, "INCONSISTENT", "Order not found at broker"
                        )
                        
                        resolved.append({
                            "discrepancy": discrepancy,
                            "action": "marked_inconsistent",
                            "success": True
                        })
                        
                        logger.info(f"Marked internal order {order_id} as inconsistent")
                    
                elif discrepancy_type == "status_mismatch":
                    # Order status mismatch between systems
                    # Strategy: Update internal status to match broker
                    broker_status = discrepancy["broker_status"]
                    
                    await self.internal_order_store.update_order_status(
                        order_id, broker_status, "Updated via reconciliation"
                    )
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "updated_status",
                        "success": True
                    })
                    
                    logger.info(f"Updated internal order {order_id} status to {broker_status}")
                    
                elif discrepancy_type == "fill_mismatch":
                    # Order fill mismatch between systems
                    # Strategy: Update internal fill to match broker
                    broker_filled = discrepancy["broker_filled"]
                    broker_order = discrepancy["broker_order"]
                    
                    await self.internal_order_store.update_order_fill(
                        order_id, broker_filled, broker_order.get("average_fill_price", 0)
                    )
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "updated_fill",
                        "success": True
                    })
                    
                    logger.info(f"Updated internal order {order_id} fill to {broker_filled}")
                
            except Exception as e:
                logger.error(f"Failed to resolve discrepancy for order {discrepancy['order_id']}: {e}")
                resolved.append({
                    "discrepancy": discrepancy,
                    "action": "resolution_failed",
                    "success": False,
                    "error": str(e)
                })
        
        return resolved
    
    async def _reconcile_positions(self) -> Dict[str, Any]:
        """
        Reconcile positions between internal system and broker.
        
        Returns:
            Dictionary with reconciliation results
        """
        # Get positions from both systems
        try:
            broker_positions = await self.broker_adapter.get_positions()
            internal_positions = await self.internal_order_store.get_positions()
            
            logger.info(f"Reconciling positions: {len(broker_positions)} broker positions, {len(internal_positions)} internal positions")
            
            # Create dictionaries for easy lookup
            # We use instrument as the key since position IDs might not match between systems
            broker_positions_dict = {pos["instrument"]: pos for pos in broker_positions}
            internal_positions_dict = {pos["instrument"]: pos for pos in internal_positions}
            
            # Find discrepancies
            discrepancies = []
            
            # 1. Positions in broker but not in internal system
            for instrument, broker_position in broker_positions_dict.items():
                if instrument not in internal_positions_dict:
                    discrepancies.append({
                        "type": "missing_internal_position",
                        "instrument": instrument,
                        "broker_position": broker_position
                    })
            
            # 2. Positions in internal system but not in broker
            for instrument, internal_position in internal_positions_dict.items():
                if instrument not in broker_positions_dict:
                    discrepancies.append({
                        "type": "missing_broker_position",
                        "instrument": instrument,
                        "internal_position": internal_position
                    })
            
            # 3. Positions in both systems but with different size or direction
            for instrument in set(broker_positions_dict.keys()) & set(internal_positions_dict.keys()):
                broker_position = broker_positions_dict[instrument]
                internal_position = internal_positions_dict[instrument]
                
                # Check for direction mismatch
                if broker_position["direction"] != internal_position["direction"]:
                    discrepancies.append({
                        "type": "position_direction_mismatch",
                        "instrument": instrument,
                        "broker_direction": broker_position["direction"],
                        "internal_direction": internal_position["direction"],
                        "broker_position": broker_position,
                        "internal_position": internal_position
                    })
                
                # Check for size mismatch
                if abs(broker_position["size"] - internal_position["size"]) > 0.01:  # Small tolerance for floating point
                    discrepancies.append({
                        "type": "position_size_mismatch",
                        "instrument": instrument,
                        "broker_size": broker_position["size"],
                        "internal_size": internal_position["size"],
                        "broker_position": broker_position,
                        "internal_position": internal_position
                    })
            
            # Resolve discrepancies
            resolved = await self._resolve_position_discrepancies(discrepancies)
            
            return {
                "discrepancies": discrepancies,
                "resolved": resolved,
                "resolution_actions": [r["action"] for r in resolved]
            }
            
        except Exception as e:
            logger.error(f"Error reconciling positions: {e}")
            raise
    
    async def _resolve_position_discrepancies(self, discrepancies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve position discrepancies using predefined strategies.
        
        Args:
            discrepancies: List of discrepancies to resolve
            
        Returns:
            List of resolved discrepancies with actions taken
        """
        resolved = []
        
        for discrepancy in discrepancies:
            try:
                discrepancy_type = discrepancy["type"]
                instrument = discrepancy["instrument"]
                
                if discrepancy_type == "missing_internal_position":
                    # Position exists in broker but not in internal system
                    # Strategy: Add to internal system
                    broker_position = discrepancy["broker_position"]
                    await self.internal_order_store.add_external_position(broker_position)
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "added_to_internal_positions",
                        "success": True
                    })
                    
                    logger.info(f"Added broker position for {instrument} to internal system")
                    
                elif discrepancy_type == "missing_broker_position":
                    # Position exists in internal system but not in broker
                    # Strategy: Mark as inconsistent in internal system
                    internal_position = discrepancy["internal_position"]
                    
                    await self.internal_order_store.mark_position_inconsistent(
                        instrument, "Position not found at broker"
                    )
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "marked_position_inconsistent",
                        "success": True
                    })
                    
                    logger.info(f"Marked internal position for {instrument} as inconsistent")
                    
                elif discrepancy_type in ["position_direction_mismatch", "position_size_mismatch"]:
                    # Position mismatch between systems
                    # Strategy: Update internal position to match broker
                    broker_position = discrepancy["broker_position"]
                    
                    await self.internal_order_store.update_position(
                        instrument,
                        broker_position["size"],
                        broker_position["direction"],
                        broker_position["entry_price"],
                        "Updated via reconciliation"
                    )
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "updated_position",
                        "success": True
                    })
                    
                    logger.info(f"Updated internal position for {instrument} to match broker")
                
            except Exception as e:
                logger.error(f"Failed to resolve discrepancy for position {discrepancy['instrument']}: {e}")
                resolved.append({
                    "discrepancy": discrepancy,
                    "action": "resolution_failed",
                    "success": False,
                    "error": str(e)
                })
        
        return resolved
    
    async def _reconcile_account(self) -> Dict[str, Any]:
        """
        Reconcile account balances and metrics between internal system and broker.
        
        Returns:
            Dictionary with reconciliation results
        """
        # Get account information from both systems
        try:
            broker_account = await self.broker_adapter.get_account_info()
            internal_account = await self.internal_order_store.get_account_info()
            
            logger.info("Reconciling account information")
            
            # Find discrepancies
            discrepancies = []
            
            # Check balance mismatch
            if abs(broker_account["balance"] - internal_account["balance"]) > 0.01:
                discrepancies.append({
                    "type": "balance_mismatch",
                    "broker_balance": broker_account["balance"],
                    "internal_balance": internal_account["balance"],
                    "difference": broker_account["balance"] - internal_account["balance"]
                })
            
            # Check equity mismatch
            if abs(broker_account["equity"] - internal_account["equity"]) > 0.01:
                discrepancies.append({
                    "type": "equity_mismatch",
                    "broker_equity": broker_account["equity"],
                    "internal_equity": internal_account["equity"],
                    "difference": broker_account["equity"] - internal_account["equity"]
                })
            
            # Check margin used mismatch
            if abs(broker_account["margin_used"] - internal_account["margin_used"]) > 0.01:
                discrepancies.append({
                    "type": "margin_mismatch",
                    "broker_margin": broker_account["margin_used"],
                    "internal_margin": internal_account["margin_used"],
                    "difference": broker_account["margin_used"] - internal_account["margin_used"]
                })
            
            # Resolve discrepancies
            resolved = await self._resolve_account_discrepancies(discrepancies)
            
            return {
                "discrepancies": discrepancies,
                "resolved": resolved,
                "resolution_actions": [r["action"] for r in resolved]
            }
            
        except Exception as e:
            logger.error(f"Error reconciling account: {e}")
            raise
    
    async def _resolve_account_discrepancies(self, discrepancies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve account discrepancies using predefined strategies.
        
        Args:
            discrepancies: List of discrepancies to resolve
            
        Returns:
            List of resolved discrepancies with actions taken
        """
        resolved = []
        
        for discrepancy in discrepancies:
            try:
                discrepancy_type = discrepancy["type"]
                
                if discrepancy_type == "balance_mismatch":
                    # Balance mismatch between systems
                    # Strategy: Update internal balance to match broker
                    broker_balance = discrepancy["broker_balance"]
                    
                    await self.internal_order_store.update_account_balance(
                        broker_balance, "Updated via reconciliation"
                    )
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "updated_balance",
                        "success": True
                    })
                    
                    logger.info(f"Updated internal account balance to {broker_balance}")
                    
                elif discrepancy_type == "equity_mismatch":
                    # Equity mismatch between systems
                    # Note: This will likely be resolved by position reconciliation and balance update
                    # Just log for now
                    logger.info(f"Equity mismatch: {discrepancy['broker_equity']} (broker) vs {discrepancy['internal_equity']} (internal)")
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "equity_logged",
                        "success": True
                    })
                    
                elif discrepancy_type == "margin_mismatch":
                    # Margin mismatch between systems
                    # Note: This will likely be resolved by position reconciliation
                    # Just log for now
                    logger.info(f"Margin mismatch: {discrepancy['broker_margin']} (broker) vs {discrepancy['internal_margin']} (internal)")
                    
                    resolved.append({
                        "discrepancy": discrepancy,
                        "action": "margin_logged",
                        "success": True
                    })
                
            except Exception as e:
                logger.error(f"Failed to resolve account discrepancy of type {discrepancy_type}: {e}")
                resolved.append({
                    "discrepancy": discrepancy,
                    "action": "resolution_failed",
                    "success": False,
                    "error": str(e)
                })
        
        return resolved
    
    async def force_reconciliation(self, reconciliation_type: ReconciliationType) -> Dict[str, Any]:
        """
        Force an immediate reconciliation of the specified type.
        
        Args:
            reconciliation_type: The type of reconciliation to perform
            
        Returns:
            Dictionary with reconciliation results
        """
        logger.info(f"Forcing immediate {reconciliation_type.value} reconciliation")
        return await self.perform_reconciliation(reconciliation_type)
    
    def get_reconciliation_status(self, reconciliation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific reconciliation operation.
        
        Args:
            reconciliation_id: The ID of the reconciliation operation
            
        Returns:
            Dictionary with reconciliation status or None if not found
        """
        reconciliation = self.ongoing_reconciliations.get(reconciliation_id)
        if not reconciliation:
            return None
            
        return {
            "reconciliation_id": reconciliation_id,
            "type": reconciliation["type"].value,
            "status": reconciliation["status"].value,
            "start_time": reconciliation["start_time"],
            "end_time": reconciliation["end_time"],
            "discrepancies_count": len(reconciliation.get("discrepancies", [])),
            "resolution_actions_count": len(reconciliation.get("resolution_actions", []))
        }
    
    def get_reconciliation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all reconciliation operations.
        
        Returns:
            Dictionary with reconciliation statistics
        """
        return {
            "orders": self.reconciliation_stats[ReconciliationType.ORDERS],
            "positions": self.reconciliation_stats[ReconciliationType.POSITIONS],
            "account": self.reconciliation_stats[ReconciliationType.ACCOUNT],
            "last_reconciliation_times": {
                k.value: v.isoformat() for k, v in self.last_reconciliation_times.items()
            }
        }


# Example usage
async def main():
    # This is for illustration only - real usage would require actual implementations
    from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator
    
    class MockOrderStore:
        """Mock internal order store for testing."""
        
        async def get_orders(self):
            return []
            
        async def get_positions(self):
            return []
            
        async def get_account_info(self):
            return {"balance": 10000, "equity": 10000, "margin_used": 0}
            
        async def add_external_order(self, order):
            print(f"Adding external order: {order['order_id']}")
            
        async def update_order_status(self, order_id, status, reason):
            print(f"Updating order status: {order_id} -> {status}")
            
        async def update_order_fill(self, order_id, filled_size, avg_price):
            print(f"Updating order fill: {order_id} -> {filled_size} @ {avg_price}")
            
        async def add_external_position(self, position):
            print(f"Adding external position: {position['instrument']}")
            
        async def mark_position_inconsistent(self, instrument, reason):
            print(f"Marking position inconsistent: {instrument}")
            
        async def update_position(self, instrument, size, direction, entry_price, reason):
            print(f"Updating position: {instrument} -> {size} {direction}")
            
        async def update_account_balance(self, balance, reason):
            print(f"Updating account balance: {balance}")
    
    # Create components
    broker_adapter = ForexBrokerSimulator()
    internal_store = MockOrderStore()
    
    # Create reconciliation service
    reconciliation_service = OrderReconciliationService(
        broker_adapter=broker_adapter,
        internal_order_store=internal_store,
        reconciliation_config={
            "order_reconciliation_interval_sec": 10,
            "position_reconciliation_interval_sec": 20,
            "account_reconciliation_interval_sec": 30,
        }
    )
    
    # Connect to broker
    await broker_adapter.connect()
    
    # Start reconciliation service
    await reconciliation_service.start()
    
    # Let it run for a bit
    print("Reconciliation service running - waiting for 60 seconds")
    await asyncio.sleep(60)
    
    # Force a reconciliation
    result = await reconciliation_service.force_reconciliation(ReconciliationType.ORDERS)
    print(f"Forced reconciliation result: {result}")
    
    # Get statistics
    stats = reconciliation_service.get_reconciliation_statistics()
    print(f"Reconciliation statistics: {stats}")
    
    # Stop reconciliation service
    await reconciliation_service.stop()
    
    # Disconnect from broker
    await broker_adapter.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
