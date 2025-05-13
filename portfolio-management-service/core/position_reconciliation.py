"""
Position reconciliation implementation.

This module provides functionality for reconciliation of position data,
comparing positions between internal records and broker data.
"""

from typing import Dict, Any
import uuid

from core.base import ReconciliationBase
from core.basic_reconciliation import BasicReconciliation
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class PositionReconciliation(ReconciliationBase):
    """
    Position reconciliation implementation focusing on position data.
    """
    
    async def perform_reconciliation(
        self,
        internal_data: Dict[str, Any],
        broker_data: Dict[str, Any],
        tolerance: float
    ) -> Dict[str, Any]:
        """
        Perform reconciliation of positions.
        
        Args:
            internal_data: Internal account data
            broker_data: Broker account data
            tolerance: Tolerance percentage for discrepancies
            
        Returns:
            Dict[str, Any]: Reconciliation results
        """
        # First perform basic reconciliation
        basic_reconciliation = BasicReconciliation(
            account_repository=self.account_repository,
            portfolio_repository=self.portfolio_repository,
            trading_gateway_client=self.trading_gateway_client,
            event_publisher=self.event_publisher,
            reconciliation_repository=self.reconciliation_repository
        )
        
        basic_result = await basic_reconciliation.perform_reconciliation(
            internal_data, broker_data, tolerance
        )
        
        discrepancies = basic_result["discrepancies"]
        
        # Compare position counts
        internal_positions = internal_data.get("positions", [])
        broker_positions = broker_data.get("positions", [])
        
        if len(internal_positions) != len(broker_positions):
            discrepancies.append({
                "discrepancy_id": str(uuid.uuid4()),
                "field": "position_count",
                "internal_value": len(internal_positions),
                "broker_value": len(broker_positions),
                "absolute_difference": len(broker_positions) - len(internal_positions),
                "percentage_difference": None,
                "status": "detected",
                "severity": "high"
            })
        
        # Create dictionaries for easier comparison
        internal_pos_dict = {p["position_id"]: p for p in internal_positions}
        broker_pos_dict = {p["position_id"]: p for p in broker_positions}
        
        # Find missing positions
        internal_only_ids = set(internal_pos_dict.keys()) - set(broker_pos_dict.keys())
        broker_only_ids = set(broker_pos_dict.keys()) - set(internal_pos_dict.keys())
        
        # Add missing position discrepancies
        for pos_id in internal_only_ids:
            pos = internal_pos_dict[pos_id]
            discrepancies.append({
                "discrepancy_id": str(uuid.uuid4()),
                "field": f"missing_broker_position",
                "position_id": pos_id,
                "instrument": pos["instrument"],
                "internal_value": {
                    "direction": pos["direction"],
                    "size": pos["size"],
                    "unrealized_pnl": pos["unrealized_pnl"]
                },
                "broker_value": None,
                "status": "detected",
                "severity": "high"
            })
        
        for pos_id in broker_only_ids:
            pos = broker_pos_dict[pos_id]
            discrepancies.append({
                "discrepancy_id": str(uuid.uuid4()),
                "field": f"missing_internal_position",
                "position_id": pos_id,
                "instrument": pos["instrument"],
                "internal_value": None,
                "broker_value": {
                    "direction": pos["direction"],
                    "size": pos["size"],
                    "unrealized_pnl": pos["unrealized_pnl"]
                },
                "status": "detected",
                "severity": "high"
            })
        
        # Compare common positions
        common_pos_ids = set(internal_pos_dict.keys()) & set(broker_pos_dict.keys())
        
        for pos_id in common_pos_ids:
            int_pos = internal_pos_dict[pos_id]
            bro_pos = broker_pos_dict[pos_id]
            
            # Compare position details
            for field in ["size", "open_price", "unrealized_pnl"]:
                int_val = int_pos.get(field, 0)
                bro_val = bro_pos.get(field, 0)
                
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
                        "field": f"position_{field}",
                        "position_id": pos_id,
                        "instrument": int_pos["instrument"],
                        "internal_value": int_val,
                        "broker_value": bro_val,
                        "absolute_difference": bro_val - int_val,
                        "percentage_difference": disc_pct,
                        "status": "detected",
                        "severity": "medium" if disc_pct > 5 else "low"
                    })
        
        return {
            "discrepancies": discrepancies,
            "matched_fields": basic_result["matched_fields"] + len(common_pos_ids),
            "reconciliation_level": "positions"
        }