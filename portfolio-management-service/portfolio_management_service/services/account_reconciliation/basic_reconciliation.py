"""
Basic account reconciliation implementation.

This module provides functionality for basic reconciliation of account data,
focusing on account balance and equity.
"""

from typing import Dict, Any
import uuid

from portfolio_management_service.services.account_reconciliation.base import ReconciliationBase
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class BasicReconciliation(ReconciliationBase):
    """
    Basic reconciliation implementation focusing on account balance and equity.
    """
    
    async def perform_reconciliation(
        self,
        internal_data: Dict[str, Any],
        broker_data: Dict[str, Any],
        tolerance: float
    ) -> Dict[str, Any]:
        """
        Perform basic reconciliation of account balance and equity.
        
        Args:
            internal_data: Internal account data
            broker_data: Broker account data
            tolerance: Tolerance percentage for discrepancies
            
        Returns:
            Dict[str, Any]: Reconciliation results
        """
        discrepancies = []
        
        # Check basic account metrics
        fields_to_check = [
            "balance", "equity", "margin_used", "free_margin"
        ]
        
        for field in fields_to_check:
            internal_value = internal_data.get(field, 0)
            broker_value = broker_data.get(field, 0)
            
            # Skip if both are zero or missing
            if internal_value == 0 and broker_value == 0:
                continue
                
            # Calculate discrepancy percentage
            if internal_value != 0:
                discrepancy_pct = abs((broker_value - internal_value) / internal_value) * 100
            else:
                discrepancy_pct = float('inf') if broker_value != 0 else 0
            
            # Check if discrepancy is above tolerance
            if discrepancy_pct > tolerance * 100:
                discrepancies.append({
                    "discrepancy_id": str(uuid.uuid4()),
                    "field": field,
                    "internal_value": internal_value,
                    "broker_value": broker_value,
                    "absolute_difference": broker_value - internal_value,
                    "percentage_difference": discrepancy_pct,
                    "status": "detected",
                    "severity": "high" if discrepancy_pct > 5 else "medium" if discrepancy_pct > 1 else "low"
                })
        
        return {
            "discrepancies": discrepancies,
            "matched_fields": len(fields_to_check) - len(discrepancies),
            "reconciliation_level": "basic"
        }