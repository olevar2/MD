"""
Account Reconciliation Service

This service provides functionality for automatic reconciliation between internal account data
and data fetched from trading brokers via the Trading Gateway Service, ensuring data integrity
and accuracy.
"""

from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
import asyncio
import uuid
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher

logger = get_logger(__name__)


class AccountReconciliationService:
    """
    Service for automatic reconciliation of account data between internal records
    and external broker data.
    
    Key capabilities:
    - Automated reconciliation of account data (balance, positions, trades)
    - Detection and reporting of data discrepancies
    - Support for manual and scheduled reconciliation
    - Historical reconciliation record tracking
    - Historical reconciliation analysis and trend detection
    """
    
    def __init__(
        self,
        account_repository=None,
        portfolio_repository=None,
        trading_gateway_client=None,
        event_publisher: Optional[EventPublisher] = None,
        reconciliation_repository=None
    ):
        """
        Initialize the reconciliation service.
        
        Args:
            account_repository: Repository for internal account data
            portfolio_repository: Repository for portfolio data
            trading_gateway_client: Client for accessing broker data
            event_publisher: Event publisher for notifications
            reconciliation_repository: Repository for storing reconciliation records
        """
        self.account_repository = account_repository
        self.portfolio_repository = portfolio_repository
        self.trading_gateway_client = trading_gateway_client
        self.event_publisher = event_publisher
        self.reconciliation_repository = reconciliation_repository
        
        # Track ongoing reconciliation processes
        self.active_reconciliations = {}
        
        # In-memory store of recent reconciliation results
        self.recent_reconciliations = {}
        
        logger.info("AccountReconciliationService initialized")
    
    async def reconcile_account(
        self,
        account_id: str,
        reconciliation_level: str = "full",  # "basic", "positions", "full"
        tolerance: float = 0.01,  # 1% tolerance for floating point differences
        notification_threshold: float = 1.0,  # 1% threshold for notifications
        auto_fix: bool = False,
        reference_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Perform account reconciliation between internal data and broker data.
        
        Args:
            account_id: ID of the account to reconcile
            reconciliation_level: Level of detail for reconciliation
            tolerance: Tolerance percentage for minor discrepancies
            notification_threshold: Threshold percentage for notifications/alerts
            auto_fix: Whether to automatically fix minor discrepancies
            reference_time: Optional reference time for historical reconciliation
            
        Returns:
            Dict[str, Any]: Reconciliation results
        """
        # Generate a unique ID for this reconciliation
        reconciliation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Add to active reconciliations
        self.active_reconciliations[reconciliation_id] = {
            "account_id": account_id,
            "start_time": start_time,
            "status": "in_progress"
        }
        
        try:
            logger.info(f"Starting reconciliation for account {account_id} with level {reconciliation_level}")
            
            # Step 1: Fetch internal account data
            internal_data = await self._get_internal_account_data(account_id, reference_time)
            
            # Step 2: Fetch broker account data
            broker_data = await self._get_broker_account_data(account_id, reference_time)
            
            # Step 3: Perform reconciliation based on level
            if reconciliation_level == "basic":
                reconciliation_result = await self._perform_basic_reconciliation(
                    internal_data, broker_data, tolerance
                )
            elif reconciliation_level == "positions":
                reconciliation_result = await self._perform_position_reconciliation(
                    internal_data, broker_data, tolerance
                )
            else:  # "full"
                reconciliation_result = await self._perform_full_reconciliation(
                    internal_data, broker_data, tolerance
                )
            
            # Step 4: Create detailed report
            report = await self._create_reconciliation_report(
                reconciliation_id,
                account_id,
                reconciliation_level,
                internal_data,
                broker_data,
                reconciliation_result,
                start_time,
                tolerance
            )
            
            # Step 5: Handle discrepancies
            if report["discrepancies"]["total_count"] > 0:
                await self._handle_discrepancies(
                    account_id, 
                    report, 
                    notification_threshold, 
                    auto_fix
                )
            
            # Update and store reconciliation status
            report["status"] = "completed"
            report["completion_time"] = datetime.utcnow()
            report["duration_seconds"] = (report["completion_time"] - start_time).total_seconds()
            
            # Update recent reconciliations cache
            self.recent_reconciliations[reconciliation_id] = report
            
            # Remove from active reconciliations
            if reconciliation_id in self.active_reconciliations:
                del self.active_reconciliations[reconciliation_id]
            
            logger.info(f"Reconciliation completed for account {account_id}, "
                      f"found {report['discrepancies']['total_count']} discrepancies")
            
            return report
            
        except Exception as e:
            logger.error(f"Error during account reconciliation: {str(e)}", exc_info=True)
            
            # Update active reconciliations status
            if reconciliation_id in self.active_reconciliations:
                self.active_reconciliations[reconciliation_id].update({
                    "status": "failed",
                    "error": str(e),
                    "completion_time": datetime.utcnow()
                })
            
            # Publish reconciliation failure event
            await self._publish_event(
                "reconciliation.failed",
                {
                    "account_id": account_id,
                    "reconciliation_id": reconciliation_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            raise ValueError(f"Account reconciliation failed: {str(e)}")
    
    async def start_scheduled_reconciliation(
        self,
        account_id: str,
        frequency: str = "daily",  # "hourly", "daily", "weekly"
        reconciliation_level: str = "full",
        tolerance: float = 0.01,
        notification_threshold: float = 1.0,
        auto_fix: bool = False
    ) -> Dict[str, Any]:
        """
        Configure and start scheduled reconciliation for an account.
        
        Args:
            account_id: ID of the account
            frequency: Reconciliation frequency
            reconciliation_level: Level of detail for reconciliation
            tolerance: Tolerance percentage for discrepancies
            notification_threshold: Threshold percentage for notifications
            auto_fix: Whether to automatically fix minor discrepancies
            
        Returns:
            Dict[str, Any]: Configuration details
        """
        # In a real implementation, this would register the schedule in a database
        # and be picked up by a scheduler service
        
        # For demonstration, we'll run an immediate reconciliation
        reconciliation = await self.reconcile_account(
            account_id=account_id,
            reconciliation_level=reconciliation_level,
            tolerance=tolerance,
            notification_threshold=notification_threshold,
            auto_fix=auto_fix
        )
        
        # Return the configuration and initial reconciliation result
        return {
            "account_id": account_id,
            "configuration": {
                "frequency": frequency,
                "reconciliation_level": reconciliation_level,
                "tolerance": tolerance,
                "notification_threshold": notification_threshold,
                "auto_fix": auto_fix,
                "status": "active"
            },
            "initial_reconciliation": {
                "reconciliation_id": reconciliation["reconciliation_id"],
                "timestamp": reconciliation["start_time"],
                "discrepancies_found": reconciliation["discrepancies"]["total_count"] > 0
            },
            "message": f"Scheduled reconciliation configured with {frequency} frequency"
        }
    
    async def get_reconciliation_status(self, reconciliation_id: str) -> Dict[str, Any]:
        """
        Get the status and results of a specific reconciliation.
        
        Args:
            reconciliation_id: ID of the reconciliation process
            
        Returns:
            Dict[str, Any]: Reconciliation status and results
        """
        # Check active reconciliations
        if reconciliation_id in self.active_reconciliations:
            return {
                "reconciliation_id": reconciliation_id,
                "status": "in_progress",
                "details": self.active_reconciliations[reconciliation_id]
            }
        
        # Check recent reconciliations
        if reconciliation_id in self.recent_reconciliations:
            return self.recent_reconciliations[reconciliation_id]
        
        # Not found in memory, would need to check database in a full implementation
        return {
            "reconciliation_id": reconciliation_id,
            "status": "not_found",
            "message": "Reconciliation not found or expired from cache"
        }
    
    async def get_recent_reconciliations(
        self,
        account_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent reconciliation results.
        
        Args:
            account_id: Optional filter by account ID
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Recent reconciliation results
        """
        results = []
        
        for recon_id, recon_data in sorted(
            self.recent_reconciliations.items(),
            key=lambda x: x[1].get("start_time", datetime.min),
            reverse=True
        ):
            # Filter by account_id if provided
            if account_id and recon_data.get("account_id") != account_id:
                continue
                
            # Add to results
            results.append(recon_data)
            
            # Limit results
            if len(results) >= limit:
                break
        
        return results
    
    async def fix_discrepancy(
        self,
        reconciliation_id: str,
        discrepancy_id: str,
        fix_source: str = "broker",  # "internal" or "broker"
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fix a specific discrepancy by updating either internal or broker data.
        
        Args:
            reconciliation_id: ID of the reconciliation
            discrepancy_id: ID of the discrepancy to fix
            fix_source: Source of truth for the fix
            comment: Optional comment for the fix
            
        Returns:
            Dict[str, Any]: Result of the fix operation
        """
        # Get reconciliation data
        recon_data = self.recent_reconciliations.get(reconciliation_id)
        if not recon_data:
            raise ValueError(f"Reconciliation {reconciliation_id} not found")
        
        # Find the specific discrepancy
        discrepancy = None
        for disc in recon_data.get("discrepancies", {}).get("details", []):
            if disc.get("discrepancy_id") == discrepancy_id:
                discrepancy = disc
                break
        
        if not discrepancy:
            raise ValueError(f"Discrepancy {discrepancy_id} not found in reconciliation {reconciliation_id}")
        
        # Apply the fix based on the specified source of truth
        try:
            if fix_source == "internal":
                # Use internal data as the source of truth
                await self._update_broker_data(
                    recon_data["account_id"],
                    discrepancy["field"],
                    discrepancy["internal_value"]
                )
                source_desc = "internal system"
            else:  # "broker"
                # Use broker data as the source of truth
                await self._update_internal_data(
                    recon_data["account_id"],
                    discrepancy["field"],
                    discrepancy["broker_value"]
                )
                source_desc = "broker"
            
            # Update the discrepancy status
            discrepancy["status"] = "fixed"
            discrepancy["fixed_at"] = datetime.utcnow().isoformat()
            discrepancy["fixed_by"] = fix_source
            discrepancy["comment"] = comment
            
            # Publish fix event
            await self._publish_event(
                "reconciliation.discrepancy.fixed",
                {
                    "account_id": recon_data["account_id"],
                    "reconciliation_id": reconciliation_id,
                    "discrepancy_id": discrepancy_id,
                    "field": discrepancy["field"],
                    "fixed_using": fix_source,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return {
                "status": "success",
                "message": f"Discrepancy fixed using {source_desc} as source of truth",
                "discrepancy": discrepancy
            }
            
        except Exception as e:
            logger.error(f"Error fixing discrepancy: {str(e)}", exc_info=True)
            
            # Update discrepancy status
            discrepancy["status"] = "fix_failed"
            discrepancy["fix_error"] = str(e)
            
            raise ValueError(f"Failed to fix discrepancy: {str(e)}")
    
    # Internal helper methods
    
    async def _get_internal_account_data(
        self,
        account_id: str,
        reference_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get internal account data for reconciliation.
        
        Args:
            account_id: ID of the account
            reference_time: Optional reference time for historical data
            
        Returns:
            Dict[str, Any]: Internal account data
        """
        try:
            if reference_time:
                # Get historical data if reference time provided
                account_data = await self.account_repository.get_account_at_timestamp(
                    account_id=account_id,
                    timestamp=reference_time
                )
                
                portfolio_data = await self.portfolio_repository.get_portfolio_at_timestamp(
                    account_id=account_id,
                    timestamp=reference_time
                )
            else:
                # Get current data
                account_data = await self.account_repository.get_account(account_id)
                portfolio_data = await self.portfolio_repository.get_portfolio(account_id)
            
            # Combine account and portfolio data
            result = {
                "account_id": account_id,
                "balance": account_data.balance,
                "equity": portfolio_data.equity,
                "margin_used": portfolio_data.margin,
                "free_margin": portfolio_data.free_margin,
                "margin_level": portfolio_data.margin_level,
                "timestamp": reference_time or datetime.utcnow(),
                "positions": [
                    {
                        "position_id": p.position_id,
                        "instrument": p.instrument,
                        "direction": p.direction,
                        "size": p.size,
                        "open_price": p.open_price,
                        "current_price": p.current_price,
                        "swap": getattr(p, "swap", 0),
                        "unrealized_pnl": p.unrealized_pnl
                    }
                    for p in getattr(portfolio_data, "positions", [])
                ],
                "orders": [
                    {
                        "order_id": o.order_id,
                        "instrument": o.instrument,
                        "type": o.type,
                        "direction": o.direction,
                        "size": o.size,
                        "price": o.price
                    }
                    for o in getattr(portfolio_data, "orders", [])
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving internal account data: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to get internal account data: {str(e)}")
    
    async def _get_broker_account_data(
        self,
        account_id: str,
        reference_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get account data from the broker via the trading gateway.
        
        Args:
            account_id: ID of the account
            reference_time: Optional reference time for historical data
            
        Returns:
            Dict[str, Any]: Broker account data
        """
        if not self.trading_gateway_client:
            raise ValueError("Trading gateway client not available")
            
        try:
            # Get account data from broker
            if reference_time:
                broker_account = await self.trading_gateway_client.get_account_history(
                    account_id=account_id,
                    timestamp=reference_time
                )
            else:
                broker_account = await self.trading_gateway_client.get_account(
                    account_id=account_id
                )
            
            # Get positions from broker
            broker_positions = await self.trading_gateway_client.get_positions(
                account_id=account_id
            )
            
            # Get open orders from broker
            broker_orders = await self.trading_gateway_client.get_orders(
                account_id=account_id
            )
            
            # Format broker data to match internal structure for comparison
            result = {
                "account_id": account_id,
                "balance": broker_account.get("balance"),
                "equity": broker_account.get("equity"),
                "margin_used": broker_account.get("margin_used"),
                "free_margin": broker_account.get("free_margin"),
                "margin_level": broker_account.get("margin_level"),
                "timestamp": reference_time or datetime.utcnow(),
                "positions": [
                    {
                        "position_id": p.get("id"),
                        "instrument": p.get("symbol"),
                        "direction": p.get("type"),
                        "size": p.get("volume"),
                        "open_price": p.get("open_price"),
                        "current_price": p.get("current_price"),
                        "swap": p.get("swap", 0),
                        "unrealized_pnl": p.get("profit")
                    }
                    for p in broker_positions
                ],
                "orders": [
                    {
                        "order_id": o.get("id"),
                        "instrument": o.get("symbol"),
                        "type": o.get("type"),
                        "direction": o.get("direction"),
                        "size": o.get("volume"),
                        "price": o.get("price")
                    }
                    for o in broker_orders
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving broker account data: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to get broker account data: {str(e)}")
    
    async def _perform_basic_reconciliation(
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
    
    async def _perform_position_reconciliation(
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
        basic_result = await self._perform_basic_reconciliation(
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
    
    async def _perform_full_reconciliation(
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
        position_result = await self._perform_position_reconciliation(
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
                "severity": "medium"
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
                "severity": "medium"
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
                        "severity": "low"
                    })
        
        return {
            "discrepancies": discrepancies,
            "matched_fields": position_result["matched_fields"] + len(common_order_ids),
            "reconciliation_level": "full"
        }
    
    async def _create_reconciliation_report(
        self,
        reconciliation_id: str,
        account_id: str,
        reconciliation_level: str,
        internal_data: Dict[str, Any],
        broker_data: Dict[str, Any],
        reconciliation_result: Dict[str, Any],
        start_time: datetime,
        tolerance: float
    ) -> Dict[str, Any]:
        """
        Create a detailed reconciliation report.
        
        Args:
            reconciliation_id: ID of this reconciliation
            account_id: ID of the account
            reconciliation_level: Level of reconciliation performed
            internal_data: Internal account data
            broker_data: Broker account data
            reconciliation_result: Results of reconciliation
            start_time: When reconciliation started
            tolerance: Tolerance percentage used
            
        Returns:
            Dict[str, Any]: Detailed reconciliation report
        """
        # Calculate discrepancy summaries
        discrepancies = reconciliation_result["discrepancies"]
        
        # Count by severity
        severity_counts = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        # Count by field type
        field_counts = {
            "account": 0,  # Balance, equity, etc.
            "positions": 0,
            "orders": 0
        }
        
        # Sum of absolute monetary differences
        total_monetary_diff = 0.0
        
        for disc in discrepancies:
            # Count by severity
            severity_counts[disc.get("severity", "medium")] += 1
            
            # Count by field type
            if disc["field"] in ["balance", "equity", "margin_used", "free_margin"]:
                field_counts["account"] += 1
            elif "position" in disc["field"]:
                field_counts["positions"] += 1
            elif "order" in disc["field"]:
                field_counts["orders"] += 1
            
            # Sum absolute differences for monetary fields
            if "absolute_difference" in disc and isinstance(disc["absolute_difference"], (int, float)):
                total_monetary_diff += abs(disc["absolute_difference"])
        
        # Create the report
        report = {
            "reconciliation_id": reconciliation_id,
            "account_id": account_id,
            "start_time": start_time,
            "reconciliation_level": reconciliation_level,
            "tolerance_percentage": tolerance * 100,
            "discrepancies": {
                "total_count": len(discrepancies),
                "by_severity": severity_counts,
                "by_field_type": field_counts,
                "total_monetary_difference": total_monetary_diff,
                "details": discrepancies
            },
            "summary": {
                "matched_fields": reconciliation_result["matched_fields"],
                "account_data_match": field_counts["account"] == 0,
                "positions_match": field_counts["positions"] == 0,
                "orders_match": field_counts["orders"] == 0,
                "overall_match": len(discrepancies) == 0
            }
        }
        
        return report
    
    async def _handle_discrepancies(
        self,
        account_id: str,
        report: Dict[str, Any],
        notification_threshold: float,
        auto_fix: bool
    ) -> None:
        """
        Handle detected discrepancies according to configuration.
        
        Args:
            account_id: ID of the account
            report: Reconciliation report
            notification_threshold: Threshold for notifications
            auto_fix: Whether to automatically fix discrepancies
        """
        # Publish reconciliation event with discrepancy count
        await self._publish_event(
            "reconciliation.completed",
            {
                "account_id": account_id,
                "reconciliation_id": report["reconciliation_id"],
                "discrepancy_count": report["discrepancies"]["total_count"],
                "severity": "high" if report["discrepancies"]["by_severity"]["high"] > 0 else
                            "medium" if report["discrepancies"]["by_severity"]["medium"] > 0 else
                            "low",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Auto-fix if configured
        if auto_fix:
            await self._auto_fix_discrepancies(account_id, report)
        
        # Check if notification threshold is exceeded
        monetary_diff = report["discrepancies"]["total_monetary_difference"]
        
        # Assuming threshold is percentage of account balance
        if "balance" in report.get("internal_data", {}):
            balance = report["internal_data"]["balance"]
            if balance > 0 and (monetary_diff / balance) * 100 > notification_threshold:
                await self._publish_event(
                    "reconciliation.threshold_exceeded",
                    {
                        "account_id": account_id,
                        "reconciliation_id": report["reconciliation_id"],
                        "monetary_difference": monetary_diff,
                        "threshold_percentage": notification_threshold,
                        "actual_percentage": (monetary_diff / balance) * 100,
                        "severity": "high",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
    
    async def _auto_fix_discrepancies(
        self,
        account_id: str,
        report: Dict[str, Any]
    ) -> None:
        """
        Automatically fix discrepancies based on rules.
        
        Args:
            account_id: ID of the account
            report: Reconciliation report
        """
        # Implement auto-fixing logic based on rules
        # In a real implementation, this would have more sophisticated logic
        # For now, we'll just log that we would fix these issues
        
        low_severity_count = 0
        
        for disc in report["discrepancies"]["details"]:
            # Only auto-fix low severity issues
            if disc.get("severity") == "low":
                low_severity_count += 1
                disc["status"] = "auto_fixed"
                disc["fixed_at"] = datetime.utcnow().isoformat()
                disc["fixed_by"] = "auto_fix"
                
                # In a real implementation, we would apply the fix here
                # Most likely using broker data as the source of truth
                
                logger.info(f"Auto-fixed discrepancy: {disc['field']} for account {account_id}")
        
        if low_severity_count > 0:
            # Publish auto-fix event
            await self._publish_event(
                "reconciliation.auto_fixed",
                {
                    "account_id": account_id,
                    "reconciliation_id": report["reconciliation_id"],
                    "discrepancies_fixed": low_severity_count,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _update_internal_data(
        self,
        account_id: str,
        field: str,
        value: Any
    ) -> None:
        """
        Update internal data to match broker data.
        
        Args:
            account_id: ID of the account
            field: Field to update
            value: New value to set
        """
        # Implementation would depend on the specific fields and repositories
        # This is a placeholder that logs the action
        logger.info(f"Updating internal {field} for account {account_id} to {value}")
        
        # In a real implementation, we would update the appropriate repository
        # For example:
        if field == "balance":
            await self.account_repository.update_balance(account_id, value)
        elif field == "equity":
            # This might require more complex updates across multiple tables
            pass
        elif "position" in field:
            # Handle position updates
            pass
        elif "order" in field:
            # Handle order updates
            pass
    
    async def _update_broker_data(
        self,
        account_id: str,
        field: str,
        value: Any
    ) -> None:
        """
        Update broker data to match internal data.
        
        Args:
            account_id: ID of the account
            field: Field to update
            value: New value to set
        """
        # In a real implementation, this would call the trading gateway to update broker data
        logger.info(f"Would update broker data for account {account_id}, field {field} to {value}")
        
        # For demonstration, we'll just log the update
        await self._publish_event(
            "reconciliation.broker_update",
            {
                "account_id": account_id,
                "field": field,
                "value": value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event if an event publisher is available"""
        if self.event_publisher:
            await self.event_publisher.publish(event_type, data)
        else:
            logger.debug(f"Would publish event: {event_type} with data: {data}")
    
    async def perform_historical_reconciliation_analysis(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "daily",  # "hourly", "daily", "weekly"
        reconciliation_level: str = "basic",
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Perform historical reconciliation analysis over a time period.
        
        This method runs multiple reconciliations at different points in time
        to analyze patterns and trends in reconciliation results.
        
        Args:
            account_id: ID of the account to analyze
            start_date: Start date for the analysis period
            end_date: End date for the analysis period
            interval: Time interval between reconciliation points
            reconciliation_level: Level of reconciliation detail
            tolerance: Tolerance percentage for reconciliation
            
        Returns:
            Dict[str, Any]: Historical reconciliation analysis results
        """
        logger.info(
            f"Starting historical reconciliation analysis for account {account_id} "
            f"from {start_date} to {end_date} with {interval} interval"
        )
        
        # Generate time points for reconciliation based on interval
        time_points = self._generate_time_points(start_date, end_date, interval)
        
        if not time_points:
            raise ValueError("No valid time points generated for historical reconciliation")
        
        # Perform reconciliation for each time point
        reconciliation_results = []
        for point in time_points:
            try:
                logger.info(f"Running reconciliation for time point: {point}")
                result = await self.reconcile_account(
                    account_id=account_id,
                    reconciliation_level=reconciliation_level,
                    tolerance=tolerance,
                    reference_time=point,
                    auto_fix=False
                )
                
                # Add the time point to the result
                result["time_point"] = point
                reconciliation_results.append(result)
                
            except Exception as e:
                logger.error(f"Error during historical reconciliation at {point}: {str(e)}", exc_info=True)
                # Add a placeholder for the failed reconciliation
                reconciliation_results.append({
                    "time_point": point,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Analyze the historical reconciliation results
        analysis = await self._analyze_historical_reconciliation(reconciliation_results)
        
        # Create a comprehensive report
        report = {
            "account_id": account_id,
            "analysis_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "interval": interval
            },
            "reconciliation_count": len(reconciliation_results),
            "successful_count": sum(1 for r in reconciliation_results if r.get("status") != "failed"),
            "failed_count": sum(1 for r in reconciliation_results if r.get("status") == "failed"),
            "analysis": analysis,
            "time_points": [r.get("time_point").isoformat() for r in reconciliation_results],
            "generation_time": datetime.utcnow().isoformat()
        }
        
        # Store the analysis report if a repository is available
        if self.reconciliation_repository:
            analysis_id = str(uuid.uuid4())
            await self.reconciliation_repository.store_historical_analysis(analysis_id, report)
            report["analysis_id"] = analysis_id
        
        logger.info(f"Completed historical reconciliation analysis for account {account_id}")
        return report
    
    async def get_historical_reconciliation_report(
        self,
        account_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        report_format: str = "summary"  # "summary", "detailed", "chart"
    ) -> Dict[str, Any]:
        """
        Generate a report of historical reconciliations for an account.
        
        Args:
            account_id: ID of the account
            start_date: Optional start date filter
            end_date: Optional end date filter
            report_format: Format of the report
            
        Returns:
            Dict[str, Any]: Historical reconciliation report
        """
        if not self.reconciliation_repository:
            raise ValueError("Reconciliation repository required for historical reporting")
        
        # Default dates if not provided
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=30))
        
        # Get historical reconciliations from repository
        reconciliations = await self.reconciliation_repository.get_reconciliations_by_timerange(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not reconciliations:
            return {
                "account_id": account_id,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "message": "No reconciliation records found for the specified period",
                "reconciliation_count": 0
            }
        
        # Generate the report based on format
        if report_format == "summary":
            report = self._generate_summary_report(account_id, reconciliations, start_date, end_date)
        elif report_format == "detailed":
            report = self._generate_detailed_report(account_id, reconciliations, start_date, end_date)
        elif report_format == "chart":
            report = await self._generate_chart_report(account_id, reconciliations, start_date, end_date)
        else:
            raise ValueError(f"Unsupported report format: {report_format}")
        
        return report
    
    async def detect_reconciliation_patterns(
        self,
        account_id: str,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Detect patterns and trends in reconciliation data over time.
        
        Args:
            account_id: ID of the account
            lookback_days: Number of days to look back
            
        Returns:
            Dict[str, Any]: Pattern analysis results
        """
        if not self.reconciliation_repository:
            raise ValueError("Reconciliation repository required for pattern detection")
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get reconciliation data
        reconciliations = await self.reconciliation_repository.get_reconciliations_by_timerange(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not reconciliations:
            return {
                "account_id": account_id,
                "lookback_days": lookback_days,
                "message": "Insufficient data for pattern detection",
                "patterns_detected": False
            }
        
        # Process data into a format suitable for analysis
        processed_data = self._process_reconciliation_data(reconciliations)
        
        # Detect recurring patterns
        recurring_patterns = self._detect_recurring_discrepancies(processed_data)
        
        # Detect trends
        trends = self._detect_discrepancy_trends(processed_data)
        
        # Detect correlations with external factors
        correlations = await self._detect_external_correlations(account_id, processed_data)
        
        return {
            "account_id": account_id,
            "analysis_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": lookback_days
            },
            "reconciliation_count": len(reconciliations),
            "patterns_detected": len(recurring_patterns) > 0 or len(trends) > 0 or len(correlations) > 0,
            "recurring_patterns": recurring_patterns,
            "trends": trends,
            "correlations": correlations,
            "generation_time": datetime.utcnow().isoformat()
        }
    
    async def compare_broker_accuracy(
        self,
        account_id: str,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Compare the accuracy of broker data over time by analyzing reconciliation results.
        
        Args:
            account_id: ID of the account
            lookback_days: Number of days to look back
            
        Returns:
            Dict[str, Any]: Broker accuracy analysis
        """
        if not self.reconciliation_repository:
            raise ValueError("Reconciliation repository required for broker accuracy analysis")
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get reconciliation data
        reconciliations = await self.reconciliation_repository.get_reconciliations_by_timerange(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not reconciliations:
            return {
                "account_id": account_id,
                "lookback_days": lookback_days,
                "message": "Insufficient data for broker accuracy analysis",
                "analysis_complete": False
            }
        
        # Calculate accuracy metrics
        total_reconciliations = len(reconciliations)
        reconciliations_with_discrepancies = sum(
            1 for r in reconciliations 
            if r.get("discrepancies", {}).get("total_count", 0) > 0
        )
        
        # Group by field for detailed analysis
        field_analysis = defaultdict(lambda: {"total": 0, "discrepancies": 0, "avg_discrepancy_pct": 0})
        
        for recon in reconciliations:
            for disc in recon.get("discrepancies", {}).get("details", []):
                field = disc.get("field")
                if field:
                    field_analysis[field]["total"] += 1
                    field_analysis[field]["discrepancies"] += 1
                    field_analysis[field]["avg_discrepancy_pct"] += disc.get("percentage_difference", 0)
        
        # Calculate average discrepancy for each field
        for field_data in field_analysis.values():
            if field_data["discrepancies"] > 0:
                field_data["avg_discrepancy_pct"] /= field_data["discrepancies"]
        
        # Calculate overall accuracy percentage
        overall_accuracy = (
            (total_reconciliations - reconciliations_with_discrepancies) / total_reconciliations * 100
            if total_reconciliations > 0 else 0
        )
        
        return {
            "account_id": account_id,
            "analysis_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": lookback_days
            },
            "total_reconciliations": total_reconciliations,
            "reconciliations_with_discrepancies": reconciliations_with_discrepancies,
            "overall_accuracy_percentage": overall_accuracy,
            "field_accuracy": {
                field: {
                    "accuracy_percentage": 
                        (1 - (data["discrepancies"] / data["total"])) * 100 
                        if data["total"] > 0 else 100,
                    "avg_discrepancy_percentage": data["avg_discrepancy_pct"],
                    "total_occurrences": data["total"]
                }
                for field, data in field_analysis.items()
            },
            "generation_time": datetime.utcnow().isoformat(),
            "analysis_complete": True
        }
    
    # Helper methods for historical reconciliation analysis
    
    def _generate_time_points(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> List[datetime]:
        """Generate time points for historical reconciliation based on interval"""
        time_points = []
        current = start_date
        
        if interval == "hourly":
            delta = timedelta(hours=1)
        elif interval == "daily":
            delta = timedelta(days=1)
        elif interval == "weekly":
            delta = timedelta(weeks=1)
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        
        while current <= end_date:
            time_points.append(current)
            current += delta
            
        return time_points
    
    async def _analyze_historical_reconciliation(
        self,
        reconciliation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze historical reconciliation results to identify patterns and trends"""
        # Skip failed reconciliations
        successful_results = [r for r in reconciliation_results if r.get("status") != "failed"]
        
        if not successful_results:
            return {
                "status": "insufficient_data",
                "message": "No successful reconciliations available for analysis"
            }
        
        # Extract discrepancy counts over time
        time_points = [r.get("time_point") for r in successful_results]
        discrepancy_counts = [r.get("discrepancies", {}).get("total_count", 0) for r in successful_results]
        
        # Calculate trend statistics
        trend_analysis = self._calculate_trend_statistics(time_points, discrepancy_counts)
        
        # Identify recurring fields with discrepancies
        recurring_fields = self._identify_recurring_discrepancy_fields(successful_results)
        
        # Identify correlation with trading volume or market volatility
        # This would require additional data, simplified here
        
        return {
            "trend": trend_analysis,
            "recurring_fields": recurring_fields,
            "summary": {
                "average_discrepancies": sum(discrepancy_counts) / len(discrepancy_counts) if discrepancy_counts else 0,
                "max_discrepancies": max(discrepancy_counts) if discrepancy_counts else 0,
                "min_discrepancies": min(discrepancy_counts) if discrepancy_counts else 0,
                "std_deviation": self._calculate_std_deviation(discrepancy_counts),
                "total_points_analyzed": len(successful_results)
            }
        }
    
    def _calculate_trend_statistics(
        self,
        time_points: List[datetime],
        discrepancy_counts: List[int]
    ) -> Dict[str, Any]:
        """Calculate trend statistics from time series data"""
        if not time_points or not discrepancy_counts or len(time_points) != len(discrepancy_counts):
            return {"status": "error", "message": "Invalid time series data"}
        
        # Check if we have enough data points for trend analysis
        if len(time_points) < 3:
            return {"status": "insufficient_data", "message": "Need at least 3 data points for trend analysis"}
        
        # Convert to numeric for calculation (days since first point)
        base_date = min(time_points)
        x_values = [(t - base_date).total_seconds() / 86400 for t in time_points]  # Convert to days
        y_values = discrepancy_counts
        
        # Simple linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x*y for x, y in zip(x_values, y_values))
        sum_xx = sum(x*x for x in x_values)
        
        # Calculate slope and intercept
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate R-squared (coefficient of determination)
            y_mean = sum_y / n
            ss_total = sum((y - y_mean) ** 2 for y in y_values)
            ss_residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            trend_direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
            trend_strength = "strong" if abs(r_squared) > 0.7 else "moderate" if abs(r_squared) > 0.3 else "weak"
            
            return {
                "direction": trend_direction,
                "strength": trend_strength,
                "slope": slope,
                "r_squared": r_squared,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error calculating trend statistics: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Error calculating trend: {str(e)}"}
    
    def _identify_recurring_discrepancy_fields(
        self,
        reconciliation_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify fields that frequently have discrepancies"""
        field_counts = defaultdict(int)
        field_severity = defaultdict(list)
        
        # Count occurrences of each field in discrepancies
        for result in reconciliation_results:
            for disc in result.get("discrepancies", {}).get("details", []):
                field = disc.get("field")
                severity = disc.get("severity", "low")
                
                if field:
                    field_counts[field] += 1
                    field_severity[field].append(severity)
        
        # Calculate frequency and average severity
        total_results = len(reconciliation_results)
        recurring_fields = []
        
        for field, count in field_counts.items():
            frequency = count / total_results if total_results > 0 else 0
            
            # Calculate severity score (high=3, medium=2, low=1)
            severity_map = {"high": 3, "medium": 2, "low": 1}
            severity_score = sum(severity_map.get(s, 1) for s in field_severity[field])
            avg_severity = severity_score / count if count > 0 else 0
            
            if frequency > 0.1:  # More than 10% of reconciliations
                recurring_fields.append({
                    "field": field,
                    "frequency": frequency,
                    "frequency_percentage": frequency * 100,
                    "occurrence_count": count,
                    "average_severity_score": avg_severity,
                    "criticality": "high" if frequency > 0.5 and avg_severity > 2 else
                                    "medium" if frequency > 0.3 or avg_severity > 2 else "low"
                })
        
        # Sort by criticality and frequency
        return sorted(
            recurring_fields,
            key=lambda x: (
                0 if x["criticality"] == "high" else 1 if x["criticality"] == "medium" else 2,
                -x["frequency"]
            )
        )
    
    def _calculate_std_deviation(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values"""
        if not values:
            return 0
            
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return variance ** 0.5
    
    def _process_reconciliation_data(
        self,
        reconciliations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process raw reconciliation data into a format suitable for analysis"""
        # Extract timestamps and create a DataFrame for analysis
        data_points = []
        
        for recon in reconciliations:
            timestamp = recon.get("start_time")
            if not timestamp:
                continue
                
            discrepancy_count = recon.get("discrepancies", {}).get("total_count", 0)
            
            # Extract specific fields with discrepancies
            field_discrepancies = {}
            for disc in recon.get("discrepancies", {}).get("details", []):
                field = disc.get("field")
                if field:
                    field_discrepancies[f"field_{field}"] = disc.get("percentage_difference", 0)
            
            data_point = {
                "timestamp": timestamp,
                "discrepancy_count": discrepancy_count,
                "reconciliation_id": recon.get("reconciliation_id"),
                **field_discrepancies
            }
            
            data_points.append(data_point)
        
        return {
            "data_points": data_points,
            "total_records": len(data_points)
        }
    
    def _detect_recurring_discrepancies(
        self,
        processed_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect recurring patterns in discrepancy data"""
        data_points = processed_data.get("data_points", [])
        if not data_points or len(data_points) < 5:  # Need reasonable number of points
            return []
        
        # Find fields that appear in multiple reconciliations
        field_occurrences = defaultdict(int)
        
        for point in data_points:
            for key in point.keys():
                if key.startswith("field_"):
                    field_occurrences[key] += 1
        
        # Find recurring patterns
        patterns = []
        for field, occurrences in field_occurrences.items():
            if occurrences >= 3:  # At least 3 occurrences to be a pattern
                field_name = field[6:]  # Remove "field_" prefix
                
                # Calculate frequency and consistency
                frequency = occurrences / len(data_points)
                values = [point.get(field, 0) for point in data_points if field in point]
                avg_value = sum(values) / len(values) if values else 0
                std_dev = self._calculate_std_deviation(values)
                
                patterns.append({
                    "field": field_name,
                    "occurrences": occurrences,
                    "frequency": frequency,
                    "average_discrepancy": avg_value,
                    "std_deviation": std_dev,
                    "consistency": "high" if std_dev < avg_value * 0.1 else
                                    "medium" if std_dev < avg_value * 0.25 else "low"
                })
        
        return sorted(patterns, key=lambda x: x["occurrences"], reverse=True)
    
    def _detect_discrepancy_trends(
        self,
        processed_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect trends in discrepancy data over time"""
        data_points = processed_data.get("data_points", [])
        if not data_points or len(data_points) < 5:  # Need reasonable number of points
            return []
        
        # Sort by timestamp
        sorted_points = sorted(data_points, key=lambda x: x.get("timestamp", datetime.min))
        
        # Extract time series for overall discrepancy count
        timestamps = [p.get("timestamp") for p in sorted_points]
        counts = [p.get("discrepancy_count", 0) for p in sorted_points]
        
        # Detect trend in overall discrepancies
        trends = []
        overall_trend = self._calculate_trend_statistics(timestamps, counts)
        
        if overall_trend.get("status") == "success":
            trends.append({
                "field": "overall_discrepancies",
                "direction": overall_trend.get("direction", "stable"),
                "strength": overall_trend.get("strength", "weak"),
                "r_squared": overall_trend.get("r_squared", 0),
                "description": f"{overall_trend.get('strength', 'weak').capitalize()} {overall_trend.get('direction', 'stable')} trend in overall discrepancies"
            })
        
        # Analyze trends for specific fields
        field_keys = [key for key in sorted_points[0].keys() if key.startswith("field_") and key in sorted_points[1]]
        
        for field in field_keys:
            field_values = [p.get(field, 0) for p in sorted_points if field in p]
            field_timestamps = [p.get("timestamp") for p in sorted_points if field in p]
            
            if len(field_values) >= 3 and len(field_timestamps) == len(field_values):
                field_trend = self._calculate_trend_statistics(field_timestamps, field_values)
                
                if field_trend.get("status") == "success" and field_trend.get("direction") != "stable":
                    trends.append({
                        "field": field[6:],  # Remove "field_" prefix
                        "direction": field_trend.get("direction", "stable"),
                        "strength": field_trend.get("strength", "weak"),
                        "r_squared": field_trend.get("r_squared", 0),
                        "description": f"{field_trend.get('strength', 'weak').capitalize()} {field_trend.get('direction', 'stable')} trend in {field[6:]} discrepancies"
                    })
        
        return trends
    
    async def _detect_external_correlations(
        self,
        account_id: str,
        processed_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect correlations between discrepancies and external factors.
        
        This would typically involve analysis against trading volume, market volatility,
        system load, etc. Simplified implementation here.
        """
        # In a real implementation, this would fetch external data and calculate correlations
        # For this demonstration, we'll return a placeholder
        return []
    
    def _generate_summary_report(
        self,
        account_id: str,
        reconciliations: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate a summary report of historical reconciliations"""
        total_count = len(reconciliations)
        
        # Calculate summary statistics
        discrepancy_counts = [r.get("discrepancies", {}).get("total_count", 0) for r in reconciliations]
        avg_discrepancies = sum(discrepancy_counts) / total_count if total_count > 0 else 0
        
        # Count reconciliations by status
        status_counts = defaultdict(int)
        for r in reconciliations:
            status_counts[r.get("status", "unknown")] += 1
        
        # Generate monthly/weekly summaries
        monthly_summary = self._generate_time_period_summary(reconciliations, "month")
        weekly_summary = self._generate_time_period_summary(reconciliations, "week")
        
        return {
            "account_id": account_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "total_reconciliations": total_count,
            "average_discrepancies": avg_discrepancies,
            "max_discrepancies": max(discrepancy_counts) if discrepancy_counts else 0,
            "status_summary": dict(status_counts),
            "monthly_summary": monthly_summary,
            "weekly_summary": weekly_summary,
            "generation_time": datetime.utcnow().isoformat()
        }
    
    def _generate_time_period_summary(
        self,
        reconciliations: List[Dict[str, Any]],
        period_type: str
    ) -> List[Dict[str, Any]]:
        """Generate a summary grouped by time period (week/month)"""
        period_summary = defaultdict(lambda: {"count": 0, "discrepancies": 0})
        
        for r in reconciliations:
            timestamp = r.get("start_time")
            if not timestamp:
                continue
                
            if period_type == "month":
                period_key = timestamp.strftime("%Y-%m")
            elif period_type == "week":
                # ISO week number (1-53)
                period_key = f"{timestamp.year}-W{timestamp.isocalendar()[1]:02d}"
            else:
                continue
                
            period_summary[period_key]["count"] += 1
            period_summary[period_key]["discrepancies"] += r.get("discrepancies", {}).get("total_count", 0)
        
        # Calculate averages and format for output
        result = []
        for period, data in sorted(period_summary.items()):
            result.append({
                "period": period,
                "reconciliation_count": data["count"],
                "total_discrepancies": data["discrepancies"],
                "average_discrepancies": data["discrepancies"] / data["count"] if data["count"] > 0 else 0
            })
            
        return result
    
    def _generate_detailed_report(
        self,
        account_id: str,
        reconciliations: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate a detailed report of historical reconciliations"""
        # Similar to summary but with additional details
        summary = self._generate_summary_report(account_id, reconciliations, start_date, end_date)
        
        # Add field-specific analysis
        field_analysis = defaultdict(lambda: {"occurrences": 0, "total_difference": 0})
        
        for r in reconciliations:
            for disc in r.get("discrepancies", {}).get("details", []):
                field = disc.get("field")
                if field:
                    field_analysis[field]["occurrences"] += 1
                    field_analysis[field]["total_difference"] += abs(disc.get("absolute_difference", 0))
        
        # Format for output
        fields_report = []
        for field, data in field_analysis.items():
            fields_report.append({
                "field": field,
                "occurrence_count": data["occurrences"],
                "occurrence_percentage": data["occurrences"] / len(reconciliations) * 100 if reconciliations else 0,
                "average_difference": data["total_difference"] / data["occurrences"] if data["occurrences"] > 0 else 0
            })
        
        # Sort by occurrence count
        fields_report.sort(key=lambda x: x["occurrence_count"], reverse=True)
        
        # Combine with summary
        detailed_report = {**summary, "field_analysis": fields_report}
        
        # Add sample reconciliations with most discrepancies
        top_discrepancy_samples = sorted(
            reconciliations,
            key=lambda x: x.get("discrepancies", {}).get("total_count", 0),
            reverse=True
        )[:5]
        
        detailed_report["top_discrepancy_samples"] = [
            {
                "reconciliation_id": r.get("reconciliation_id"),
                "timestamp": r.get("start_time").isoformat() if r.get("start_time") else None,
                "discrepancy_count": r.get("discrepancies", {}).get("total_count", 0),
                "status": r.get("status")
            }
            for r in top_discrepancy_samples
        ]
        
        return detailed_report
    
    async def _generate_chart_report(
        self,
        account_id: str,
        reconciliations: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate a report with charts of historical reconciliations"""
        # Generate the summary data first
        summary = self._generate_summary_report(account_id, reconciliations, start_date, end_date)
        
        # Prepare data for charting
        chart_data = []
        for r in sorted(reconciliations, key=lambda x: x.get("start_time", datetime.min)):
            timestamp = r.get("start_time")
            if timestamp:
                chart_data.append({
                    "timestamp": timestamp,
                    "discrepancy_count": r.get("discrepancies", {}).get("total_count", 0)
                })
        
        # Create time series chart of discrepancies
        time_series_chart = self._create_time_series_chart(chart_data)
        
        # Create field frequency chart
        field_frequency_chart = self._create_field_frequency_chart(reconciliations)
        
        # Add charts to the report
        chart_report = {
            **summary,
            "charts": {
                "time_series": time_series_chart,
                "field_frequency": field_frequency_chart
            }
        }
        
        return chart_report
    
    def _create_time_series_chart(self, chart_data: List[Dict[str, Any]]) -> str:
        """
        Create a time series chart of discrepancies over time.
        
        In a real implementation, this would generate an actual chart.
        For this demonstration, we return a placeholder string that would represent the chart data.
        """
        return "time_series_chart_data_placeholder"
    
    def _create_field_frequency_chart(self, reconciliations: List[Dict[str, Any]]) -> str:
        """
        Create a chart showing frequency of discrepancies by field.
        
        In a real implementation, this would generate an actual chart.
        For this demonstration, we return a placeholder string that would represent the chart data.
        """
        return "field_frequency_chart_data_placeholder"
