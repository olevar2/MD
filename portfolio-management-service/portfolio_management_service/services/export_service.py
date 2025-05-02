"""
Portfolio Export Service

This service provides functionality for exporting portfolio data in various formats
including CSV, JSON, and XLSX, with configurable options for customization.
"""

import csv
import json
import os
from datetime import datetime
from io import StringIO, BytesIO
from typing import Dict, List, Optional, Union, Any, BinaryIO

import pandas as pd
from fastapi import HTTPException

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioExportService:
    """
    Service for exporting portfolio data in various formats with customizable options.
    
    Supported formats:
    - CSV: Simple tabular data format
    - JSON: Structured data format
    - XLSX: Excel spreadsheet format (with optional multiple sheets)
    """
    
    def __init__(self, portfolio_repository=None, trade_repository=None, position_repository=None):
        """
        Initialize the export service with required repositories.
        
        Args:
            portfolio_repository: Repository for accessing portfolio data
            trade_repository: Repository for accessing trade history data
            position_repository: Repository for accessing position data
        """
        self.portfolio_repository = portfolio_repository
        self.trade_repository = trade_repository
        self.position_repository = position_repository
    
    async def export_portfolio_snapshot(
        self,
        account_id: str,
        export_format: str,
        timestamp: Optional[datetime] = None,
        include_positions: bool = True,
        include_orders: bool = True,
        include_trades: bool = False,
        formatting_options: Optional[Dict[str, Any]] = None
    ) -> Union[str, bytes]:
        """
        Export a portfolio snapshot in the specified format.
        
        Args:
            account_id: ID of the account to export
            export_format: Format for export ('csv', 'json', 'xlsx')
            timestamp: Timestamp for historical snapshot (None for current data)
            include_positions: Whether to include positions in the export
            include_orders: Whether to include open orders in the export
            include_trades: Whether to include recent trades in the export
            formatting_options: Optional dict with format-specific options
            
        Returns:
            Union[str, bytes]: The exported data
        """
        # Get portfolio data
        try:
            portfolio_data = await self._get_portfolio_data(
                account_id, 
                timestamp, 
                include_positions,
                include_orders,
                include_trades
            )
        except Exception as e:
            logger.error(f"Error retrieving portfolio data: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving portfolio data: {str(e)}")
        
        # Apply format-specific options
        options = formatting_options or {}
        
        # Format and return the data
        export_format = export_format.lower()
        if export_format == 'csv':
            return self._format_as_csv(portfolio_data, options)
        elif export_format == 'json':
            return self._format_as_json(portfolio_data, options)
        elif export_format == 'xlsx':
            return self._format_as_xlsx(portfolio_data, options)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    async def export_trade_history(
        self,
        account_id: str,
        export_format: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        instruments: Optional[List[str]] = None,
        group_by: Optional[str] = None,
        formatting_options: Optional[Dict[str, Any]] = None
    ) -> Union[str, bytes]:
        """
        Export trade history in the specified format.
        
        Args:
            account_id: ID of the account to export trades for
            export_format: Format for export ('csv', 'json', 'xlsx')
            start_date: Start date for filtering trades
            end_date: End date for filtering trades
            instruments: List of instruments to filter by
            group_by: Optional grouping field (e.g., 'instrument', 'strategy')
            formatting_options: Optional dict with format-specific options
            
        Returns:
            Union[str, bytes]: The exported data
        """
        # Get trade history
        try:
            trade_data = await self._get_trade_history(
                account_id,
                start_date,
                end_date,
                instruments
            )
        except Exception as e:
            logger.error(f"Error retrieving trade history: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving trade history: {str(e)}")
        
        # Apply grouping if specified
        if group_by and trade_data:
            grouped_data = self._group_data(trade_data, group_by)
        else:
            grouped_data = {"All Trades": trade_data}
        
        # Apply format-specific options
        options = formatting_options or {}
        
        # Format and return the data
        export_format = export_format.lower()
        if export_format == 'csv':
            return self._format_grouped_as_csv(grouped_data, options)
        elif export_format == 'json':
            return self._format_as_json(grouped_data, options)
        elif export_format == 'xlsx':
            return self._format_grouped_as_xlsx(grouped_data, options)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    async def export_performance_metrics(
        self,
        account_id: str,
        export_format: str,
        period: str = "all",  # "day", "week", "month", "year", "all"
        metrics: Optional[List[str]] = None,
        formatting_options: Optional[Dict[str, Any]] = None
    ) -> Union[str, bytes]:
        """
        Export performance metrics in the specified format.
        
        Args:
            account_id: ID of the account to export metrics for
            export_format: Format for export ('csv', 'json', 'xlsx')
            period: Time period for metrics calculation
            metrics: List of specific metrics to include
            formatting_options: Optional dict with format-specific options
            
        Returns:
            Union[str, bytes]: The exported data
        """
        # Define default metrics if none specified
        if metrics is None:
            metrics = [
                "total_pnl", "win_rate", "profit_factor", "sharpe_ratio",
                "max_drawdown", "avg_win", "avg_loss", "avg_hold_time"
            ]
        
        # Get performance metrics
        try:
            metrics_data = await self._get_performance_metrics(
                account_id,
                period,
                metrics
            )
        except Exception as e:
            logger.error(f"Error retrieving performance metrics: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving performance metrics: {str(e)}")
        
        # Apply format-specific options
        options = formatting_options or {}
        
        # Format and return the data
        export_format = export_format.lower()
        if export_format == 'csv':
            return self._format_as_csv(metrics_data, options)
        elif export_format == 'json':
            return self._format_as_json(metrics_data, options)
        elif export_format == 'xlsx':
            return self._format_as_xlsx(metrics_data, options)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    async def export_account_snapshots(
        self,
        account_id: str,
        export_format: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily",  # "hourly", "daily", "weekly", "monthly"
        include_positions: bool = False,
        formatting_options: Optional[Dict[str, Any]] = None
    ) -> Union[str, bytes]:
        """
        Export account snapshots over time in the specified format.
        
        Args:
            account_id: ID of the account to export
            export_format: Format for export ('csv', 'json', 'xlsx')
            start_date: Start date for snapshots
            end_date: End date for snapshots
            frequency: Frequency of snapshots
            include_positions: Whether to include positions in each snapshot
            formatting_options: Optional dict with format-specific options
            
        Returns:
            Union[str, bytes]: The exported data
        """
        # Get account snapshots
        try:
            snapshots_data = await self._get_account_snapshots(
                account_id,
                start_date,
                end_date,
                frequency,
                include_positions
            )
        except Exception as e:
            logger.error(f"Error retrieving account snapshots: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving account snapshots: {str(e)}")
        
        # Apply format-specific options
        options = formatting_options or {}
        
        # Format and return the data
        export_format = export_format.lower()
        if export_format == 'csv':
            return self._format_as_csv(snapshots_data, options)
        elif export_format == 'json':
            return self._format_as_json(snapshots_data, options)
        elif export_format == 'xlsx':
            return self._format_as_xlsx(snapshots_data, options)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    async def export_to_external_system(
        self,
        account_id: str,
        system_type: str,
        data_type: str = "full",  # "full", "trades", "positions", "summary"
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        formatting_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export portfolio data in a format compatible with external accounting systems.
        
        Args:
            account_id: ID of the account to export
            system_type: Type of external system (e.g., "quickbooks", "xero", "generic")
            data_type: Type of data to export
            start_date: Start date for filtering data
            end_date: End date for filtering data
            formatting_options: Optional dict with format-specific options
            
        Returns:
            Dict[str, Any]: Result of the export operation
        """
        # Apply system-specific mapping
        try:
            mapping_result = await self._apply_external_system_mapping(
                account_id,
                system_type,
                data_type,
                start_date,
                end_date,
                formatting_options or {}
            )
            
            return mapping_result
        except Exception as e:
            logger.error(f"Error during external system export: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during external system export: {str(e)}")
    
    # Helper methods for data retrieval
    
    async def _get_portfolio_data(
        self,
        account_id: str,
        timestamp: Optional[datetime],
        include_positions: bool,
        include_orders: bool,
        include_trades: bool
    ) -> Dict[str, Any]:
        """
        Retrieve portfolio data for export.
        
        Args:
            account_id: ID of the account
            timestamp: Point in time for historical data
            include_positions: Whether to include positions
            include_orders: Whether to include open orders
            include_trades: Whether to include recent trades
            
        Returns:
            Dict[str, Any]: Portfolio data
        """
        # Get portfolio summary
        if timestamp:
            portfolio = await self.portfolio_repository.get_portfolio_at_timestamp(account_id, timestamp)
        else:
            portfolio = await self.portfolio_repository.get_portfolio(account_id)
        
        # Create result dictionary with account details
        result = {
            "account": {
                "account_id": portfolio.account_id,
                "balance": portfolio.balance,
                "equity": portfolio.equity,
                "margin": portfolio.margin,
                "free_margin": portfolio.free_margin,
                "margin_level": portfolio.margin_level,
                "timestamp": portfolio.timestamp.isoformat()
            }
        }
        
        # Add positions if requested
        if include_positions and hasattr(portfolio, "positions"):
            result["positions"] = [
                {
                    "position_id": pos.position_id,
                    "instrument": pos.instrument,
                    "direction": pos.direction,
                    "size": pos.size,
                    "open_price": pos.open_price,
                    "current_price": pos.current_price,
                    "pnl": pos.unrealized_pnl,
                    "swap": pos.swap,
                    "open_time": pos.open_time.isoformat()
                }
                for pos in portfolio.positions
            ]
        
        # Add orders if requested
        if include_orders and hasattr(portfolio, "orders"):
            result["orders"] = [
                {
                    "order_id": order.order_id,
                    "instrument": order.instrument,
                    "type": order.type,
                    "direction": order.direction,
                    "size": order.size,
                    "price": order.price,
                    "stop_loss": order.stop_loss,
                    "take_profit": order.take_profit,
                    "create_time": order.create_time.isoformat()
                }
                for order in portfolio.orders
            ]
        
        # Add recent trades if requested
        if include_trades:
            # Get the 10 most recent closed trades
            trades = await self.trade_repository.get_recent_trades(
                account_id, 
                limit=10
            )
            
            result["recent_trades"] = [
                {
                    "trade_id": trade.trade_id,
                    "instrument": trade.instrument,
                    "direction": trade.direction,
                    "size": trade.size,
                    "open_price": trade.open_price,
                    "close_price": trade.close_price,
                    "pnl": trade.realized_pnl,
                    "open_time": trade.open_time.isoformat(),
                    "close_time": trade.close_time.isoformat() if trade.close_time else None
                }
                for trade in trades
            ]
        
        return result
    
    async def _get_trade_history(
        self,
        account_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        instruments: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trade history for export.
        
        Args:
            account_id: ID of the account
            start_date: Start date for filtering
            end_date: End date for filtering
            instruments: List of instruments to filter by
            
        Returns:
            List[Dict[str, Any]]: Trade history
        """
        # Get trades with filtering
        trades = await self.trade_repository.get_trades(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date,
            instruments=instruments
        )
        
        # Format trade data
        trade_data = [
            {
                "trade_id": trade.trade_id,
                "instrument": trade.instrument,
                "direction": trade.direction,
                "size": trade.size,
                "open_price": trade.open_price,
                "close_price": trade.close_price,
                "pnl": trade.realized_pnl,
                "open_time": trade.open_time.isoformat(),
                "close_time": trade.close_time.isoformat() if trade.close_time else None,
                "strategy": getattr(trade, "strategy_id", "Unknown"),
                "trade_tags": getattr(trade, "tags", [])
            }
            for trade in trades
        ]
        
        return trade_data
    
    async def _get_performance_metrics(
        self,
        account_id: str,
        period: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve performance metrics for export.
        
        Args:
            account_id: ID of the account
            period: Time period for metrics
            metrics: List of metrics to retrieve
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        # In a real implementation, this would call a performance analytics service
        # For now, return placeholder data
        
        # Mock data for demonstration
        result = {
            "account_id": account_id,
            "period": period,
            "calculation_time": datetime.utcnow().isoformat(),
            "metrics": {}
        }
        
        # Add requested metrics (mock data)
        if "total_pnl" in metrics:
            result["metrics"]["total_pnl"] = 1250.75
        
        if "win_rate" in metrics:
            result["metrics"]["win_rate"] = 0.65
        
        if "profit_factor" in metrics:
            result["metrics"]["profit_factor"] = 1.8
        
        if "sharpe_ratio" in metrics:
            result["metrics"]["sharpe_ratio"] = 1.2
        
        if "max_drawdown" in metrics:
            result["metrics"]["max_drawdown"] = -450.25
        
        if "avg_win" in metrics:
            result["metrics"]["avg_win"] = 125.5
        
        if "avg_loss" in metrics:
            result["metrics"]["avg_loss"] = -75.25
        
        if "avg_hold_time" in metrics:
            result["metrics"]["avg_hold_time"] = 3600  # seconds
        
        return result
    
    async def _get_account_snapshots(
        self,
        account_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: str,
        include_positions: bool
    ) -> List[Dict[str, Any]]:
        """
        Retrieve account snapshots for export.
        
        Args:
            account_id: ID of the account
            start_date: Start date for snapshots
            end_date: End date for snapshots
            frequency: Frequency of snapshots
            include_positions: Whether to include positions
            
        Returns:
            List[Dict[str, Any]]: Account snapshots
        """
        # In a real implementation, this would call the account_snapshot_service
        # For now, return placeholder data for demonstration
        from datetime import timedelta
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.utcnow()
        
        if not start_date:
            # Default to 1 month of data
            start_date = end_date - timedelta(days=30)
        
        # Determine snapshot interval based on frequency
        if frequency == "hourly":
            interval = timedelta(hours=1)
        elif frequency == "daily":
            interval = timedelta(days=1)
        elif frequency == "weekly":
            interval = timedelta(weeks=1)
        elif frequency == "monthly":
            # Approximate a month as 30 days for simplicity
            interval = timedelta(days=30)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Generate snapshots
        snapshots = []
        current_time = start_date
        
        while current_time <= end_date:
            # Create a snapshot for this point in time
            snapshot = {
                "account_id": account_id,
                "timestamp": current_time.isoformat(),
                "balance": 10000 + (current_time - start_date).days * 50,  # Mock increasing balance
                "equity": 10200 + (current_time - start_date).days * 48,   # Mock increasing equity
                "margin": 1000,
                "free_margin": 9200 + (current_time - start_date).days * 48,
                "margin_level": 1020 + (current_time - start_date).days * 4.8
            }
            
            # Add positions if requested (mock data)
            if include_positions:
                snapshot["positions"] = [
                    {
                        "instrument": "EUR/USD",
                        "direction": "BUY",
                        "size": 0.1,
                        "open_price": 1.1850,
                        "current_price": 1.1870 + (current_time - start_date).days * 0.001,
                        "pnl": 20 + (current_time - start_date).days
                    },
                    {
                        "instrument": "GBP/USD",
                        "direction": "SELL",
                        "size": 0.05,
                        "open_price": 1.3750,
                        "current_price": 1.3720 - (current_time - start_date).days * 0.0005,
                        "pnl": 15 + (current_time - start_date).days * 0.5
                    }
                ]
            
            snapshots.append(snapshot)
            current_time += interval
        
        return snapshots
    
    async def _apply_external_system_mapping(
        self,
        account_id: str,
        system_type: str,
        data_type: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map internal data structures to external system format.
        
        Args:
            account_id: ID of the account
            system_type: Type of external system
            data_type: Type of data to map
            start_date: Start date for data
            end_date: End date for data
            options: Format-specific options
            
        Returns:
            Dict[str, Any]: Mapping result
        """
        # Get the appropriate data based on data_type
        if data_type == "trades":
            data = await self._get_trade_history(account_id, start_date, end_date, None)
        elif data_type == "positions":
            portfolio = await self._get_portfolio_data(account_id, None, True, False, False)
            data = portfolio.get("positions", [])
        elif data_type == "summary":
            data = await self._get_performance_metrics(account_id, "custom", ["total_pnl", "win_rate"])
        else:  # "full"
            data = await self._get_portfolio_data(account_id, None, True, True, True)
        
        # Apply system-specific mapping
        if system_type.lower() == "quickbooks":
            return self._map_to_quickbooks_format(data, data_type, options)
        elif system_type.lower() == "xero":
            return self._map_to_xero_format(data, data_type, options)
        else:  # "generic"
            return self._map_to_generic_format(data, data_type, options)
    
    def _map_to_quickbooks_format(self, data: Any, data_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Map data to QuickBooks format."""
        # Implementation would depend on QuickBooks API requirements
        # This is a simplified example
        
        result = {
            "system": "QuickBooks",
            "format_version": "1.0",
            "export_time": datetime.utcnow().isoformat(),
            "data_type": data_type
        }
        
        if data_type == "trades":
            # Map trades to QuickBooks transactions
            result["transactions"] = []
            
            for trade in data:
                # Create a transaction entry for QuickBooks
                transaction = {
                    "TxnDate": trade.get("close_time", trade.get("open_time")),
                    "DocNumber": f"TRADE-{trade.get('trade_id')}",
                    "Amount": trade.get("pnl", 0),
                    "AccountRef": {
                        "name": f"Trading {trade.get('instrument')}",
                        "value": f"Trading-{trade.get('instrument')}"
                    },
                    "Description": f"{trade.get('direction')} {trade.get('size')} {trade.get('instrument')}"
                }
                result["transactions"].append(transaction)
        
        elif data_type == "summary":
            # Create a summary journal entry
            result["journal_entries"] = [{
                "TxnDate": datetime.utcnow().isoformat(),
                "DocNumber": f"SUM-{datetime.utcnow().strftime('%Y%m%d')}",
                "Description": "Trading Account Summary",
                "JournalEntryLineDetail": [
                    {
                        "AccountRef": {"name": "Trading Account", "value": "Trading"},
                        "Description": "Net Trading Profit/Loss",
                        "Amount": data.get("metrics", {}).get("total_pnl", 0),
                        "PostingType": "Credit" if data.get("metrics", {}).get("total_pnl", 0) > 0 else "Debit"
                    }
                ]
            }]
        
        # Add export stats
        result["stats"] = {
            "record_count": len(data) if isinstance(data, list) else 1,
            "ready_for_import": True
        }
        
        return result
    
    def _map_to_xero_format(self, data: Any, data_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Map data to Xero format."""
        # Implementation would depend on Xero API requirements
        # This is a simplified example
        
        result = {
            "system": "Xero",
            "format_version": "1.0",
            "export_time": datetime.utcnow().isoformat(),
            "data_type": data_type
        }
        
        if data_type == "trades":
            # Map trades to Xero bank transactions
            result["BankTransactions"] = []
            
            for trade in data:
                # Create a transaction entry for Xero
                is_positive = trade.get("pnl", 0) > 0
                transaction = {
                    "Date": trade.get("close_time", trade.get("open_time")),
                    "Reference": f"TRADE-{trade.get('trade_id')}",
                    "Type": "RECEIVE" if is_positive else "SPEND",
                    "LineItems": [
                        {
                            "Description": f"{trade.get('direction')} {trade.get('size')} {trade.get('instrument')}",
                            "Quantity": 1.0,
                            "UnitAmount": abs(trade.get("pnl", 0)),
                            "AccountCode": "4000" if is_positive else "5000"  # Example account codes
                        }
                    ]
                }
                result["BankTransactions"].append(transaction)
        
        # Add export stats
        result["stats"] = {
            "record_count": len(data) if isinstance(data, list) else 1,
            "ready_for_import": True
        }
        
        return result
    
    def _map_to_generic_format(self, data: Any, data_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Map data to a generic format suitable for most accounting systems."""
        
        result = {
            "system": "Generic",
            "format_version": "1.0",
            "export_time": datetime.utcnow().isoformat(),
            "data_type": data_type
        }
        
        # Simply transform the data into a more generic structure
        # The exact structure would depend on common accounting system expectations
        
        if data_type == "trades":
            result["entries"] = []
            
            for trade in data:
                entry = {
                    "date": trade.get("close_time", trade.get("open_time")),
                    "reference": f"TRADE-{trade.get('trade_id')}",
                    "description": f"{trade.get('direction')} {trade.get('size')} {trade.get('instrument')}",
                    "amount": trade.get("pnl", 0),
                    "category": "Trading Income" if trade.get("pnl", 0) > 0 else "Trading Expense",
                    "subcategory": trade.get('instrument', '')
                }
                result["entries"].append(entry)
        
        elif data_type == "positions":
            result["assets"] = []
            
            for position in data:
                asset = {
                    "date": datetime.utcnow().isoformat(),
                    "asset_id": position.get("position_id", ""),
                    "description": f"{position.get('direction')} {position.get('size')} {position.get('instrument')}",
                    "acquisition_value": position.get("size", 0) * position.get("open_price", 0),
                    "current_value": position.get("size", 0) * position.get("current_price", 0),
                    "unrealized_gain": position.get("pnl", 0)
                }
                result["assets"].append(asset)
        
        # Add export stats
        result["stats"] = {
            "record_count": len(data) if isinstance(data, list) else 1,
            "format": data_type
        }
        
        return result
    
    # Helper methods for data transformation and formatting
    
    def _group_data(self, data: List[Dict[str, Any]], group_by: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group data by the specified field.
        
        Args:
            data: List of data items
            group_by: Field to group by
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Grouped data
        """
        result = {}
        
        for item in data:
            group_value = item.get(group_by, "Unknown")
            
            if group_value not in result:
                result[group_value] = []
                
            result[group_value].append(item)
            
        return result
    
    def _format_as_csv(self, data: Any, options: Dict[str, Any]) -> str:
        """
        Format data as CSV.
        
        Args:
            data: Data to format
            options: Formatting options
            
        Returns:
            str: CSV formatted data
        """
        output = StringIO()
        
        # Flatten nested data if needed
        if isinstance(data, dict) and not all(isinstance(v, (str, int, float, bool, type(None))) for v in data.values()):
            # Complex dictionary - flatten it
            flattened_data = self._flatten_dict(data)
            
            # Write as CSV
            writer = csv.DictWriter(output, fieldnames=flattened_data.keys())
            writer.writeheader()
            writer.writerow(flattened_data)
        
        elif isinstance(data, list):
            # List of items (likely dictionaries)
            if not data:
                # Empty list
                output.write("No data")
            elif isinstance(data[0], dict):
                # Get field names from the first item
                field_names = list(data[0].keys())
                
                # Apply field filtering if specified in options
                if "fields" in options and isinstance(options["fields"], list):
                    field_names = [f for f in field_names if f in options["fields"]]
                
                writer = csv.DictWriter(output, fieldnames=field_names)
                writer.writeheader()
                
                for item in data:
                    # Filter to just the fields we want
                    filtered_item = {k: item[k] for k in field_names if k in item}
                    writer.writerow(filtered_item)
            else:
                # List of simple values
                writer = csv.writer(output)
                writer.writerow(["Value"])
                for item in data:
                    writer.writerow([item])
        else:
            # Simple value or unknown type
            output.write(str(data))
        
        return output.getvalue()
    
    def _format_grouped_as_csv(self, grouped_data: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        Format grouped data as CSV.
        
        Args:
            grouped_data: Grouped data to format
            options: Formatting options
            
        Returns:
            str: CSV formatted data
        """
        output = StringIO()
        
        for group_name, group_data in grouped_data.items():
            # Add group header
            output.write(f"# Group: {group_name}\n")
            
            # Format this group's data as CSV
            group_csv = self._format_as_csv(group_data, options)
            
            # Add to output
            output.write(group_csv)
            output.write("\n\n")  # Add separation between groups
        
        return output.getvalue()
    
    def _format_as_json(self, data: Any, options: Dict[str, Any]) -> str:
        """
        Format data as JSON.
        
        Args:
            data: Data to format
            options: Formatting options
            
        Returns:
            str: JSON formatted data
        """
        # Apply JSON-specific options
        indent = options.get("indent", 2)
        
        # Filter fields if specified
        if "fields" in options and isinstance(options["fields"], list) and isinstance(data, (dict, list)):
            if isinstance(data, dict):
                data = {k: v for k, v in data.items() if k in options["fields"]}
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                data = [{k: v for k, v in item.items() if k in options["fields"]} for item in data]
        
        # Convert to JSON
        return json.dumps(data, indent=indent)
    
    def _format_as_xlsx(self, data: Any, options: Dict[str, Any]) -> bytes:
        """
        Format data as XLSX spreadsheet.
        
        Args:
            data: Data to format
            options: Formatting options
            
        Returns:
            bytes: XLSX formatted data
        """
        output = BytesIO()
        
        # Create a pandas DataFrame from the data
        if isinstance(data, dict) and not all(isinstance(v, (str, int, float, bool, type(None))) for v in data.values()):
            # Complex dictionary - convert to DataFrame
            if "account" in data and isinstance(data["account"], dict):
                # Special case for portfolio data with account and positions
                sheets = {}
                
                # Create account sheet
                sheets["Account"] = pd.DataFrame([data["account"]])
                
                # Create positions sheet if present
                if "positions" in data and isinstance(data["positions"], list) and data["positions"]:
                    sheets["Positions"] = pd.DataFrame(data["positions"])
                
                # Create orders sheet if present
                if "orders" in data and isinstance(data["orders"], list) and data["orders"]:
                    sheets["Orders"] = pd.DataFrame(data["orders"])
                
                # Create trades sheet if present
                if "recent_trades" in data and isinstance(data["recent_trades"], list) and data["recent_trades"]:
                    sheets["Recent Trades"] = pd.DataFrame(data["recent_trades"])
                
                # Write multi-sheet Excel file
                with pd.ExcelWriter(output) as writer:
                    for sheet_name, df in sheets.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Regular dictionary - flatten it
                df = pd.DataFrame([self._flatten_dict(data)])
                df.to_excel(output, index=False)
        
        elif isinstance(data, list):
            if not data:
                # Empty list
                pd.DataFrame().to_excel(output, index=False)
            elif isinstance(data[0], dict):
                # List of dictionaries
                df = pd.DataFrame(data)
                
                # Apply field filtering if specified in options
                if "fields" in options and isinstance(options["fields"], list):
                    fields = [f for f in df.columns if f in options["fields"]]
                    df = df[fields]
                
                df.to_excel(output, index=False)
            else:
                # List of simple values
                pd.DataFrame({"Value": data}).to_excel(output, index=False)
        else:
            # Simple value or unknown type
            pd.DataFrame({"Value": [str(data)]}).to_excel(output, index=False)
        
        # Reset pointer to beginning of file
        output.seek(0)
        
        return output.getvalue()
    
    def _format_grouped_as_xlsx(self, grouped_data: Dict[str, Any], options: Dict[str, Any]) -> bytes:
        """
        Format grouped data as XLSX spreadsheet with multiple sheets.
        
        Args:
            grouped_data: Grouped data to format
            options: Formatting options
            
        Returns:
            bytes: XLSX formatted data
        """
        output = BytesIO()
        
        # Create Excel writer
        with pd.ExcelWriter(output) as writer:
            for group_name, group_data in grouped_data.items():
                # Filter sheet name for Excel compatibility
                sheet_name = self._sanitize_sheet_name(group_name)
                
                # Create DataFrame for this group
                if isinstance(group_data, list) and group_data and isinstance(group_data[0], dict):
                    # List of dictionaries
                    df = pd.DataFrame(group_data)
                    
                    # Apply field filtering if specified in options
                    if "fields" in options and isinstance(options["fields"], list):
                        fields = [f for f in df.columns if f in options["fields"]]
                        df = df[fields]
                elif isinstance(group_data, dict):
                    # Single dictionary
                    df = pd.DataFrame([group_data])
                else:
                    # Other types
                    df = pd.DataFrame({"Value": group_data if isinstance(group_data, list) else [str(group_data)]})
                
                # Write to sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Reset pointer to beginning of file
        output.seek(0)
        
        return output.getvalue()
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary into a single level.
        
        Args:
            d: Dictionary to flatten
            parent_key: Key from parent level
            sep: Separator for nested keys
            
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                # Recursive case - another dictionary
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                # List of dictionaries - convert to count and first few items
                items.append((f"{new_key}{sep}count", len(v)))
                for i, item in enumerate(v[:3]):  # Include only first 3 items
                    flattened_item = self._flatten_dict(item, f"{new_key}{sep}{i}", sep)
                    items.extend(flattened_item.items())
            else:
                # Base case - simple value
                items.append((new_key, v))
                
        return dict(items)
    
    def _sanitize_sheet_name(self, name: str) -> str:
        """
        Sanitize a name for use as an Excel sheet name.
        
        Excel sheet names have restrictions: 
        - Max 31 characters
        - Cannot contain: [ ] : * ? / \
        
        Args:
            name: Original name
            
        Returns:
            str: Sanitized name
        """
        # Replace invalid characters
        for char in ['[', ']', ':', '*', '?', '/', '\\']:
            name = name.replace(char, '_')
        
        # Truncate to 31 characters
        return name[:31]
