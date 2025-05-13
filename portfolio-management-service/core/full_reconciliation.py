"""
Full account reconciliation implementation.

This module provides functionality for full reconciliation of account data,
including positions and orders.
"""

from typing import Dict, Any
import uuid

from core.base import ReconciliationBase
from core.position_reconciliation import PositionReconciliation
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class FullReconciliation(ReconciliationBase):
    """
    Full reconciliation implementation including positions and orders.
    """
    
    async def perform_reconciliation(
        self,
        internal_data: Dict[str, Any],
        broker_data: Dict[str, Any],
        tolerance: float
    ) -> Dict[str, Any]:
        """
        Perform full reconciliation including orders.
        
        Args:
            internal_data: Internal account data
            broker_data: Broker account data
            tolerance: Tolerance percentage for discrepancies
            
        Returns:
            Dict[str, Any]: Reconciliation results
        """
        # First perform position reconciliation
        position_reconciliation = PositionReconciliation(
            account_repository=self.account_repository,
            portfolio_repository=self.portfolio_repository,
            trading_gateway_client=self.trading_gateway_client,
            event_publisher=self.event_publisher,
            reconciliation_repository=self.reconciliation_repository
        )
        
        position_result = await position_reconciliation.perform_reconciliation(
            internal_data, broker_data, tolerance
        )
        
        discrepancies = position_result["discrepancies"]
        
        # Compare order counts
        internal_orders = internal_data.get("orders", [])
        broker_orders = broker_data.get("orders", [])
        
        if len(internal_orders) != len(broker_orders):
            discrepancies.append({
                "discrepancy_id": str(uuid.uuid4()),
                "field": "order_count",
                "internal_value": len(internal_orders),
                "broker_value": len(broker_orders),
                "absolute_difference": len(broker_orders) - len(internal_orders),
                "percentage_difference": None,
                "status": "detected",
                "severity": "medium"
            })
        
        # Create dictionaries for easier comparison
        internal_order_dict = {o["order_id"]: o for o in internal_orders}
        broker_order_dict = {o["order_id"]: o for o in broker_orders}
        
        # Find missing orders
        internal_only_ids = set(internal_order_dict.keys()) - set(broker_order_dict.keys())
        broker_only_ids = set(broker_order_dict.keys()) - set(internal_order_dict.keys())
        
        # Add missing order discrepancies
        for order_id in internal_only_ids:
            order = internal_order_dict[order_id]
            discrepancies.append({
                "discrepancy_id": str(uuid.uuid4()),
                "field": f"missing_broker_order",
                "order_id": order_id,
                "instrument": order["instrument"],
                "internal_value": {
                    "type": order["type"],
                    "direction": order["direction"],
                    "size": order["size"],
                    "price": order["price"]
                },
                "broker_value": None,
                "status": "detected",
                "severity": "high"
            })
        
        for order_id in broker_only_ids:
            order = broker_order_dict[order_id]
            discrepancies.append({
                "discrepancy_id": str(uuid.uuid4()),
                "field": f"missing_internal_order",
                "order_id": order_id,
                "instrument": order["instrument"],
                "internal_value": None,
                "broker_value": {
                    "type": order["type"],
                    "direction": order["direction"],
                    "size": order["size"],
                    "price": order["price"]
                },
                "status": "detected",
                "severity": "high"
            })
        
        # Compare common orders
        common_order_ids = set(internal_order_dict.keys()) & set(broker_order_dict.keys())
        
        for order_id in common_order_ids:
            int_order = internal_order_dict[order_id]
            bro_order = broker_order_dict[order_id]
            
            # Compare order details
            for field in ["size", "price"]:
                int_val = int_order.get(field, 0)
                bro_val = bro_order.get(field, 0)
                
                # Skip if both are zero
                if int_val == 0 and bro_val == 0:
                    continue
                    
                # Calculate discrepancy percentage
                if int_val != 0:
                    disc_pct = abs((bro_val - int_val) / int_val) * 100
                else:
                    disc_pct = float('inf') if bro_val != 0 else 0
                
                # Check if discrepancy is above tolerance
                if disc_pct > tolerance * 100:
                    discrepancies.append({
                        "discrepancy_id": str(uuid.uuid4()),
                        "field": f"order_{field}",
                        "order_id": order_id,
                        "instrument": int_order["instrument"],
                        "internal_value": int_val,
                        "broker_value": bro_val,
                        "absolute_difference": bro_val - int_val,
                        "percentage_difference": disc_pct,
                        "status": "detected",
                        "severity": "medium" if disc_pct > 5 else "low"
                    })
            
            # Check for type and direction mismatches (non-numeric fields)
            for field in ["type", "direction"]:
                int_val = int_order.get(field)
                bro_val = bro_order.get(field)
                
                if int_val != bro_val:
                    discrepancies.append({
                        "discrepancy_id": str(uuid.uuid4()),
                        "field": f"order_{field}",
                        "order_id": order_id,
                        "instrument": int_order["instrument"],
                        "internal_value": int_val,
                        "broker_value": bro_val,
                        "absolute_difference": None,
                        "percentage_difference": None,
                        "status": "detected",
                        "severity": "high"
                    })
        
        return {
            "discrepancies": discrepancies,
            "matched_fields": position_result["matched_fields"] + len(common_order_ids),
            "reconciliation_level": "full"
        }