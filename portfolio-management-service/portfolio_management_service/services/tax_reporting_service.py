"""
Tax Reporting Service

This service provides tax reporting functionality for forex trading accounts,
including profit and loss statements, transaction summaries, and tax jurisdiction mapping.
"""

from datetime import datetime, date, timedelta
from enum import Enum
import calendar
from typing import Dict, List, Optional, Any, Union, Tuple
import csv
import json
import io
from decimal import Decimal
import pandas as pd
import uuid
import os

from core_foundations.utils.logger import get_logger
from portfolio_management_service.models.tax import TaxReport, TaxJurisdiction, TransactionType

logger = get_logger(__name__)


class TaxReportType(str, Enum):
    """Types of tax reports that can be generated"""
    TRANSACTION_SUMMARY = "transaction_summary"
    PROFIT_LOSS = "profit_loss"
    CAPITAL_GAINS = "capital_gains"
    INTEREST_INCOME = "interest_income"
    COMPLETE = "complete"


class TaxPeriod(str, Enum):
    """Common tax reporting periods"""
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


class HoldingPeriod(str, Enum):
    """Classification of holding periods for tax purposes"""
    SHORT_TERM = "short_term"  # Typically <= 1 year
    LONG_TERM = "long_term"    # Typically > 1 year
    OTHER = "other"            # Special cases


class TaxReportingService:
    """
    Service for generating tax reports and managing tax-related functionality.
    
    This service enables:
    - Generating transaction summaries for tax purposes
    - Creating profit/loss statements for specific periods
    - Supporting multiple tax jurisdictions with different requirements
    - Mapping transactions to appropriate tax categories
    """
    
    def __init__(self, 
                 trade_repository=None, 
                 account_repository=None,
                 transaction_repository=None,
                 tax_jurisdiction_repository=None,
                 portfolio_repository=None):
        """
        Initialize the tax reporting service with required repositories.
        
        Args:
            trade_repository: Repository for trade data
            account_repository: Repository for account data
            transaction_repository: Repository for transaction data
            tax_jurisdiction_repository: Repository for tax jurisdiction rules
            portfolio_repository: Repository for portfolio and position data
        """
        self.trade_repository = trade_repository
        self.account_repository = account_repository
        self.transaction_repository = transaction_repository
        self.tax_jurisdiction_repository = tax_jurisdiction_repository
        self.portfolio_repository = portfolio_repository
    
    async def generate_tax_report(
        self,
        account_id: str,
        jurisdiction: str,
        report_type: TaxReportType,
        year: int,
        period: TaxPeriod = TaxPeriod.YEAR,
        period_number: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        include_unrealized: bool = False
    ) -> TaxReport:
        """
        Generate a tax report for the specified parameters.
        
        Args:
            account_id: The account ID
            jurisdiction: Tax jurisdiction code (e.g., "US", "UK", "EU")
            report_type: Type of tax report to generate
            year: Tax year
            period: Reporting period type
            period_number: Period number (e.g., month number, quarter number)
            start_date: Custom start date (for custom periods)
            end_date: Custom end date (for custom periods)
            include_unrealized: Whether to include unrealized gains/losses
            
        Returns:
            The generated tax report
        """
        # Determine the date range for the report
        start_date, end_date = self._determine_date_range(
            year, period, period_number, start_date, end_date
        )
        
        # Get tax jurisdiction rules
        jurisdiction_rules = await self.tax_jurisdiction_repository.get_jurisdiction(jurisdiction)
        
        # Get transaction data based on date range
        transactions = await self.transaction_repository.get_transactions_by_date_range(
            account_id, start_date, end_date
        )
        
        # Get trades for the period
        trades = await self.trade_repository.get_trades_by_date_range(
            account_id, start_date, end_date
        )
        
        # Filter and categorize transactions according to jurisdiction rules
        categorized_transactions = self._categorize_transactions(
            transactions, trades, jurisdiction_rules
        )
        
        # Generate the appropriate report based on type
        if report_type == TaxReportType.TRANSACTION_SUMMARY:
            report_data = self._generate_transaction_summary(
                categorized_transactions, jurisdiction_rules
            )
        elif report_type == TaxReportType.PROFIT_LOSS:
            report_data = self._generate_profit_loss_report(
                categorized_transactions, trades, jurisdiction_rules
            )
        elif report_type == TaxReportType.CAPITAL_GAINS:
            report_data = self._generate_capital_gains_report(
                trades, jurisdiction_rules
            )
        elif report_type == TaxReportType.INTEREST_INCOME:
            report_data = self._generate_interest_income_report(
                categorized_transactions, jurisdiction_rules
            )
        elif report_type == TaxReportType.COMPLETE:
            report_data = self._generate_complete_tax_report(
                categorized_transactions, trades, jurisdiction_rules
            )
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        # Add unrealized gains/losses if requested
        if include_unrealized:
            unrealized_data = await self._calculate_unrealized_gains_losses(
                account_id, end_date, jurisdiction_rules
            )
            report_data["unrealized"] = unrealized_data
        
        # Create the report object
        report = TaxReport(
            account_id=account_id,
            jurisdiction=jurisdiction,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            generation_date=datetime.utcnow(),
            data=report_data
        )
        
        # Log report generation
        logger.info(
            f"Generated {report_type.value} tax report for account {account_id}, "
            f"jurisdiction {jurisdiction}, period {start_date} to {end_date}"
        )
        
        return report
    
    async def export_tax_report(
        self,
        report: TaxReport,
        format: str = "json",
        file_path: Optional[str] = None
    ) -> Union[str, bytes, None]:
        """
        Export a tax report in the specified format.
        
        Args:
            report: The tax report to export
            format: Export format ("json", "csv", "xlsx")
            file_path: Optional file path to save the report
            
        Returns:
            The exported report data as string/bytes or None if saved to file
        """
        report_data = report.data
        
        if format.lower() == "json":
            export_data = json.dumps(report_data, indent=2, default=self._json_serializer)
            content_type = "application/json"
        elif format.lower() == "csv":
            export_data = self._convert_to_csv(report_data, report.report_type)
            content_type = "text/csv"
        elif format.lower() == "xlsx":
            export_data = self._convert_to_excel(report_data, report.report_type)
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # If file path is provided, save to file
        if file_path:
            mode = "w" if format.lower() == "csv" or format.lower() == "json" else "wb"
            encoding = "utf-8" if format.lower() == "csv" or format.lower() == "json" else None
            
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, mode, encoding=encoding) as f:
                f.write(export_data)
            
            logger.info(f"Exported {report.report_type} report to {file_path}")
            return None
        
        # Otherwise, return the data
        return export_data
    
    def _convert_to_csv(self, report_data: Dict, report_type: TaxReportType) -> str:
        """Convert report data to CSV format"""
        output = io.StringIO()
        
        if report_type == TaxReportType.TRANSACTION_SUMMARY:
            writer = csv.writer(output)
            writer.writerow(["Category", "Count", "Total", "Currency"])
            
            for category, data in report_data.get("categories", {}).items():
                writer.writerow([
                    category, 
                    data.get("count", 0),
                    data.get("total", 0),
                    data.get("currency", "USD")
                ])
                
        elif report_type == TaxReportType.CAPITAL_GAINS:
            writer = csv.writer(output)
            writer.writerow([
                "Type", "Gains", "Losses", "Net", "Tax Rate", "Estimated Tax"
            ])
            
            writer.writerow([
                "Short Term",
                report_data.get("short_term_gains", 0),
                report_data.get("short_term_losses", 0),
                report_data.get("net_short_term", 0),
                report_data.get("short_term_tax_rate", "N/A"),
                report_data.get("estimated_short_term_tax", 0)
            ])
            
            writer.writerow([
                "Long Term",
                report_data.get("long_term_gains", 0),
                report_data.get("long_term_losses", 0),
                report_data.get("net_long_term", 0),
                report_data.get("long_term_tax_rate", "N/A"),
                report_data.get("estimated_long_term_tax", 0)
            ])
            
            writer.writerow([
                "Total",
                report_data.get("short_term_gains", 0) + report_data.get("long_term_gains", 0),
                report_data.get("short_term_losses", 0) + report_data.get("long_term_losses", 0),
                report_data.get("total_net_capital_gain", 0),
                "N/A",
                report_data.get("estimated_total_tax", 0)
            ])
            
        elif report_type == TaxReportType.COMPLETE:
            # For complete reports, create multiple CSV sections
            writer = csv.writer(output)
            
            # Summary section
            writer.writerow(["SUMMARY SECTION"])
            writer.writerow(["Total Taxable Amount", report_data.get("total_taxable_amount", 0)])
            writer.writerow(["Estimated Total Tax", report_data.get("estimated_total_tax", 0)])
            writer.writerow([])
            
            # Capital Gains section
            capital_gains = report_data.get("capital_gains", {})
            writer.writerow(["CAPITAL GAINS SECTION"])
            writer.writerow(["Net Short Term", capital_gains.get("net_short_term", 0)])
            writer.writerow(["Net Long Term", capital_gains.get("net_long_term", 0)])
            writer.writerow(["Total Net Capital Gain", capital_gains.get("total_net_capital_gain", 0)])
            writer.writerow([])
            
            # Interest Income section
            interest = report_data.get("interest_income", {})
            writer.writerow(["INTEREST INCOME SECTION"])
            writer.writerow(["Total Interest Income", interest.get("total_interest_income", 0)])
            writer.writerow([])
            
        else:
            # Generic CSV export for other report types
            flattened_data = self._flatten_dict(report_data)
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(flattened_data.keys())
            
            # Write values
            writer.writerow(flattened_data.values())
            
        return output.getvalue()
    
    def _convert_to_excel(self, report_data: Dict, report_type: TaxReportType) -> bytes:
        """Convert report data to Excel format"""
        if report_type == TaxReportType.TRANSACTION_SUMMARY:
            df_categories = pd.DataFrame([{
                "Category": category,
                "Count": data.get("count", 0),
                "Total": data.get("total", 0),
                "Currency": data.get("currency", "USD")
            } for category, data in report_data.get("categories", {}).items()])
            
            with io.BytesIO() as output:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_categories.to_excel(writer, sheet_name='Summary', index=False)
                return output.getvalue()
                
        elif report_type == TaxReportType.CAPITAL_GAINS:
            df = pd.DataFrame([
                {
                    "Type": "Short Term",
                    "Gains": report_data.get("short_term_gains", 0),
                    "Losses": report_data.get("short_term_losses", 0),
                    "Net": report_data.get("net_short_term", 0),
                    "Tax Rate": report_data.get("short_term_tax_rate", "N/A"),
                    "Estimated Tax": report_data.get("estimated_short_term_tax", 0)
                },
                {
                    "Type": "Long Term",
                    "Gains": report_data.get("long_term_gains", 0),
                    "Losses": report_data.get("long_term_losses", 0),
                    "Net": report_data.get("net_long_term", 0),
                    "Tax Rate": report_data.get("long_term_tax_rate", "N/A"),
                    "Estimated Tax": report_data.get("estimated_long_term_tax", 0)
                },
                {
                    "Type": "Total",
                    "Gains": report_data.get("short_term_gains", 0) + report_data.get("long_term_gains", 0),
                    "Losses": report_data.get("short_term_losses", 0) + report_data.get("long_term_losses", 0),
                    "Net": report_data.get("total_net_capital_gain", 0),
                    "Tax Rate": "N/A",
                    "Estimated Tax": report_data.get("estimated_total_tax", 0)
                }
            ])
            
            with io.BytesIO() as output:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Capital Gains', index=False)
                return output.getvalue()
                
        elif report_type == TaxReportType.COMPLETE:
            with io.BytesIO() as output:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Summary sheet
                    pd.DataFrame({
                        "Metric": ["Total Taxable Amount", "Estimated Total Tax"],
                        "Value": [
                            report_data.get("total_taxable_amount", 0),
                            report_data.get("estimated_total_tax", 0)
                        ]
                    }).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Capital Gains sheet
                    capital_gains = report_data.get("capital_gains", {})
                    pd.DataFrame([
                        {
                            "Type": "Short Term",
                            "Gains": capital_gains.get("short_term_gains", 0),
                            "Losses": capital_gains.get("short_term_losses", 0),
                            "Net": capital_gains.get("net_short_term", 0)
                        },
                        {
                            "Type": "Long Term",
                            "Gains": capital_gains.get("long_term_gains", 0),
                            "Losses": capital_gains.get("long_term_losses", 0),
                            "Net": capital_gains.get("net_long_term", 0)
                        }
                    ]).to_excel(writer, sheet_name='Capital Gains', index=False)
                    
                    # Interest Income sheet
                    interest = report_data.get("interest_income", {})
                    pd.DataFrame({
                        "Total Interest Income": [interest.get("total_interest_income", 0)]
                    }).to_excel(writer, sheet_name='Interest Income', index=False)
                    
                return output.getvalue()
        else:
            # Generic Excel export for other report types
            flattened_data = self._flatten_dict(report_data)
            df = pd.DataFrame([flattened_data])
            
            with io.BytesIO() as output:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Report', index=False)
                return output.getvalue()
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten a nested dictionary structure"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict) and not (isinstance(v, dict) and any(isinstance(i, dict) for i in v.values())):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
                
        return dict(items)
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    async def calculate_holding_period_classifications(
        self, 
        trades: List[Dict[str, Any]],
        jurisdiction_rules: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify trades by holding period according to jurisdiction rules.
        
        Args:
            trades: List of trades with entry/exit dates
            jurisdiction_rules: Rules for the tax jurisdiction
            
        Returns:
            Dictionary of trades classified by holding period
        """
        # Get the holding period threshold from jurisdiction rules (default to 365 days for 1 year)
        long_term_threshold_days = jurisdiction_rules.get("long_term_threshold_days", 365)
        
        classified_trades = {
            HoldingPeriod.SHORT_TERM.value: [],
            HoldingPeriod.LONG_TERM.value: [],
            HoldingPeriod.OTHER.value: []
        }
        
        for trade in trades:
            # Skip trades with missing dates
            if not trade.get("entry_date") or not trade.get("exit_date"):
                classified_trades[HoldingPeriod.OTHER.value].append(trade)
                continue
                
            # Calculate holding period in days
            entry_date = self._parse_date(trade["entry_date"])
            exit_date = self._parse_date(trade["exit_date"])
            
            if not entry_date or not exit_date:
                classified_trades[HoldingPeriod.OTHER.value].append(trade)
                continue
                
            holding_period_days = (exit_date - entry_date).days
            
            # Classify based on holding period
            if holding_period_days <= long_term_threshold_days:
                classified_trades[HoldingPeriod.SHORT_TERM.value].append(trade)
            else:
                classified_trades[HoldingPeriod.LONG_TERM.value].append(trade)
                
        return classified_trades
    
    def _parse_date(self, date_value: Any) -> Optional[date]:
        """Parse a date from various formats"""
        if isinstance(date_value, date):
            return date_value
        elif isinstance(date_value, datetime):
            return date_value.date()
        elif isinstance(date_value, str):
            try:
                return datetime.fromisoformat(date_value).date()
            except ValueError:
                try:
                    return datetime.strptime(date_value, "%Y-%m-%d").date()
                except ValueError:
                    return None
        return None
    
    async def detect_wash_sales(
        self,
        account_id: str,
        trades: List[Dict[str, Any]],
        jurisdiction_rules: Dict[str, Any],
        detection_window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Detect potential wash sales for tax reporting.
        
        A wash sale typically occurs when a security is sold at a loss and 
        repurchased within a short period (e.g., 30 days before or after the sale).
        
        Args:
            account_id: The account ID
            trades: List of trades to analyze
            jurisdiction_rules: Rules for the tax jurisdiction
            detection_window_days: Number of days to look for repurchases (default: 30)
            
        Returns:
            List of detected potential wash sales
        """
        # Check if wash sale rules apply for this jurisdiction
        if not jurisdiction_rules.get("apply_wash_sale_rules", True):
            return []
            
        # Window days can be overridden by jurisdiction rules
        window_days = jurisdiction_rules.get("wash_sale_window_days", detection_window_days)
        
        # Sort trades by instrument and exit date
        sorted_trades = sorted(
            trades, 
            key=lambda t: (t.get("instrument_id", ""), self._parse_date(t.get("exit_date")))
        )
        
        wash_sales = []
        instrument_trades = {}
        
        # Group trades by instrument
        for trade in sorted_trades:
            instrument = trade.get("instrument_id")
            if not instrument:
                continue
                
            if instrument not in instrument_trades:
                instrument_trades[instrument] = []
                
            instrument_trades[instrument].append(trade)
        
        # Check each instrument's trades for wash sale patterns
        for instrument, instrument_specific_trades in instrument_trades.items():
            for i, trade in enumerate(instrument_specific_trades):
                # Skip if not a loss trade
                if trade.get("realized_pl", 0) >= 0:
                    continue
                    
                exit_date = self._parse_date(trade.get("exit_date"))
                if not exit_date:
                    continue
                    
                # Look for purchases within the window
                wash_window_start = exit_date - timedelta(days=window_days)
                wash_window_end = exit_date + timedelta(days=window_days)
                
                for j, other_trade in enumerate(instrument_specific_trades):
                    if i == j:  # Skip comparing to self
                        continue
                        
                    entry_date = self._parse_date(other_trade.get("entry_date"))
                    if not entry_date:
                        continue
                        
                    # Check if this trade's entry date falls within the wash sale window
                    if wash_window_start <= entry_date <= wash_window_end:
                        wash_sales.append({
                            "loss_trade_id": trade.get("id"),
                            "repurchase_trade_id": other_trade.get("id"),
                            "instrument_id": instrument,
                            "loss_amount": trade.get("realized_pl"),
                            "loss_date": exit_date.isoformat(),
                            "repurchase_date": entry_date.isoformat(),
                            "days_between": (entry_date - exit_date).days,
                            "disallowed_loss": min(abs(trade.get("realized_pl", 0)), other_trade.get("volume", 0))
                        })
                        
                        # In some jurisdictions, only one repurchase needs to be identified
                        if jurisdiction_rules.get("single_wash_sale_match", False):
                            break
                            
        return wash_sales

    async def get_available_tax_jurisdictions(self) -> List[TaxJurisdiction]:
        """
        Get a list of available tax jurisdictions.
        
        Returns:
            List of available tax jurisdictions
        """
        return await self.tax_jurisdiction_repository.list_jurisdictions()
    
    async def get_jurisdiction_rules(self, jurisdiction: str) -> Dict[str, Any]:
        """
        Get detailed rules for a specific tax jurisdiction.
        
        Args:
            jurisdiction: Tax jurisdiction code
            
        Returns:
            Dictionary containing tax rules for the jurisdiction
        """
        return await self.tax_jurisdiction_repository.get_jurisdiction_rules(jurisdiction)
    
    async def map_account_to_jurisdiction(
        self,
        account_id: str,
        jurisdiction: str,
        tax_identifier: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Map an account to a specific tax jurisdiction.
        
        Args:
            account_id: The account ID
            jurisdiction: Tax jurisdiction code
            tax_identifier: Tax ID number for the jurisdiction
            additional_info: Additional tax-related information
        """
        mapping_info = {
            "jurisdiction": jurisdiction,
            "mapped_at": datetime.utcnow()
        }
        
        if tax_identifier:
            mapping_info["tax_identifier"] = tax_identifier
            
        if additional_info:
            mapping_info["additional_info"] = additional_info
        
        await self.account_repository.update_tax_information(
            account_id, mapping_info
        )
        
        logger.info(f"Mapped account {account_id} to tax jurisdiction {jurisdiction}")
    
    def _determine_date_range(
        self,
        year: int,
        period: TaxPeriod,
        period_number: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Tuple[date, date]:
        """
        Determine the start and end dates based on reporting period.
        
        Args:
            year: Tax year
            period: Reporting period type
            period_number: Period number (e.g., month number, quarter number)
            start_date: Custom start date
            end_date: Custom end date
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if period == TaxPeriod.CUSTOM:
            if not start_date or not end_date:
                raise ValueError("Custom period requires both start_date and end_date")
            return start_date, end_date
        
        if period == TaxPeriod.YEAR:
            return date(year, 1, 1), date(year, 12, 31)
        
        if period == TaxPeriod.MONTH:
            if not period_number or period_number < 1 or period_number > 12:
                raise ValueError("Valid month number (1-12) required for monthly reporting")
            
            # Get last day of the month
            last_day = calendar.monthrange(year, period_number)[1]
            return date(year, period_number, 1), date(year, period_number, last_day)
        
        if period == TaxPeriod.QUARTER:
            if not period_number or period_number < 1 or period_number > 4:
                raise ValueError("Valid quarter number (1-4) required for quarterly reporting")
            
            start_month = (period_number - 1) * 3 + 1
            end_month = start_month + 2
            
            # Get last day of the end month
            last_day = calendar.monthrange(year, end_month)[1]
            
            return date(year, start_month, 1), date(year, end_month, last_day)
        
        raise ValueError(f"Unsupported period type: {period}")
    
    def _categorize_transactions(
        self,
        transactions: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        jurisdiction_rules: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize transactions according to tax jurisdiction rules.
        
        Args:
            transactions: List of transactions
            trades: List of trades
            jurisdiction_rules: Rules for the tax jurisdiction
            
        Returns:
            Dictionary of categorized transactions
        """
        categories = {
            "trading_gains": [],
            "interest_income": [],
            "dividend_income": [],
            "fees_and_commissions": [],
            "other": []
        }
        
        for transaction in transactions:
            transaction_type = transaction.get("type")
            
            if transaction_type == TransactionType.REALIZED_PL:
                categories["trading_gains"].append(transaction)
            elif transaction_type == TransactionType.INTEREST:
                categories["interest_income"].append(transaction)
            elif transaction_type == TransactionType.DIVIDEND:
                categories["dividend_income"].append(transaction)
            elif transaction_type in (TransactionType.COMMISSION, TransactionType.FEE):
                categories["fees_and_commissions"].append(transaction)
            else:
                categories["other"].append(transaction)
        
        return categories
    
    def _generate_transaction_summary(
        self,
        categorized_transactions: Dict[str, List[Dict[str, Any]]],
        jurisdiction_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a transaction summary report.
        
        Args:
            categorized_transactions: Categorized transactions
            jurisdiction_rules: Rules for the tax jurisdiction
            
        Returns:
            Transaction summary report data
        """
        summary = {
            "total_transactions": sum(len(cat) for cat in categorized_transactions.values()),
            "categories": {}
        }
        
        # Summarize each category
        for category, transactions in categorized_transactions.items():
            if not transactions:
                continue
                
            category_sum = sum(t.get("amount", 0) for t in transactions)
            category_count = len(transactions)
            
            summary["categories"][category] = {
                "count": category_count,
                "total": category_sum,
                "currency": transactions[0].get("currency", "USD"),
                "transactions": transactions
            }
        
        # Add summary totals
        summary["total_trading_gains"] = summary.get("categories", {}).get("trading_gains", {}).get("total", 0)
        summary["total_interest_income"] = summary.get("categories", {}).get("interest_income", {}).get("total", 0)
        summary["total_dividend_income"] = summary.get("categories", {}).get("dividend_income", {}).get("total", 0)
        summary["total_fees_and_commissions"] = summary.get("categories", {}).get("fees_and_commissions", {}).get("total", 0)
        
        return summary
    
    def _generate_profit_loss_report(
        self,
        categorized_transactions: Dict[str, List[Dict[str, Any]]],
        trades: List[Dict[str, Any]],
        jurisdiction_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a profit and loss report.
        
        Args:
            categorized_transactions: Categorized transactions
            trades: List of trades
            jurisdiction_rules: Rules for the tax jurisdiction
            
        Returns:
            Profit and loss report data
        """
        # Group trades by instrument
        trades_by_instrument = {}
        for trade in trades:
            instrument = trade.get("instrument_id")
            if instrument not in trades_by_instrument:
                trades_by_instrument[instrument] = []
            trades_by_instrument[instrument].append(trade)
        
        # Calculate P&L by instrument
        pl_by_instrument = {}
        for instrument, instrument_trades in trades_by_instrument.items():
            total_pl = sum(t.get("realized_pl", 0) for t in instrument_trades)
            total_volume = sum(t.get("volume", 0) for t in instrument_trades)
            trade_count = len(instrument_trades)
            
            pl_by_instrument[instrument] = {
                "total_pl": total_pl,
                "trade_count": trade_count,
                "total_volume": total_volume,
                "average_pl_per_trade": total_pl / trade_count if trade_count > 0 else 0
            }
        
        # Create the profit/loss summary
        trading_gains = categorized_transactions.get("trading_gains", [])
        total_trading_gains = sum(t.get("amount", 0) for t in trading_gains)
        
        fees = categorized_transactions.get("fees_and_commissions", [])
        total_fees = sum(t.get("amount", 0) for t in fees)
        
        profit_loss = {
            "gross_trading_profit": total_trading_gains,
            "fees_and_commissions": total_fees,
            "net_trading_profit": total_trading_gains - total_fees,
            "by_instrument": pl_by_instrument,
            "winning_trades": sum(1 for t in trades if t.get("realized_pl", 0) > 0),
            "losing_trades": sum(1 for t in trades if t.get("realized_pl", 0) <= 0),
            "total_trades": len(trades)
        }
        
        return profit_loss
    
    def _generate_capital_gains_report(
        self,
        trades: List[Dict[str, Any]],
        jurisdiction_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a capital gains report.
        
        Args:
            trades: List of trades
            jurisdiction_rules: Rules for the tax jurisdiction
            
        Returns:
            Capital gains report data
        """
        # This would implement jurisdiction-specific capital gains calculations
        # For example, some jurisdictions have different rules for short vs long-term gains
        
        # For forex, typically all gains are short-term for most jurisdictions
        short_term_gains = sum(t.get("realized_pl", 0) for t in trades if t.get("realized_pl", 0) > 0)
        short_term_losses = sum(t.get("realized_pl", 0) for t in trades if t.get("realized_pl", 0) < 0)
        
        report = {
            "short_term_gains": short_term_gains,
            "short_term_losses": short_term_losses,
            "net_short_term": short_term_gains + short_term_losses,  # losses are negative
            "long_term_gains": 0,  # typically not applicable for forex
            "long_term_losses": 0,
            "net_long_term": 0,
            "total_net_capital_gain": short_term_gains + short_term_losses
        }
        
        # Add tax rate information if available in jurisdiction rules
        if "short_term_rate" in jurisdiction_rules:
            report["short_term_tax_rate"] = jurisdiction_rules["short_term_rate"]
            report["estimated_short_term_tax"] = report["net_short_term"] * jurisdiction_rules["short_term_rate"]
            report["estimated_total_tax"] = report["estimated_short_term_tax"]
        
        return report
    
    def _generate_interest_income_report(
        self,
        categorized_transactions: Dict[str, List[Dict[str, Any]]],
        jurisdiction_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an interest income report.
        
        Args:
            categorized_transactions: Categorized transactions
            jurisdiction_rules: Rules for the tax jurisdiction
            
        Returns:
            Interest income report data
        """
        interest_transactions = categorized_transactions.get("interest_income", [])
        total_interest = sum(t.get("amount", 0) for t in interest_transactions)
        
        report = {
            "total_interest_income": total_interest,
            "transactions": interest_transactions
        }
        
        # Add tax rate information if available
        if "interest_income_rate" in jurisdiction_rules:
            report["interest_income_tax_rate"] = jurisdiction_rules["interest_income_rate"]
            report["estimated_tax"] = total_interest * jurisdiction_rules["interest_income_rate"]
        
        return report
    
    def _generate_complete_tax_report(
        self,
        categorized_transactions: Dict[str, List[Dict[str, Any]]],
        trades: List[Dict[str, Any]],
        jurisdiction_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a complete tax report including all categories.
        
        Args:
            categorized_transactions: Categorized transactions
            trades: List of trades
            jurisdiction_rules: Rules for the tax jurisdiction
            
        Returns:
            Complete tax report data
        """
        # Generate all report sections
        transaction_summary = self._generate_transaction_summary(
            categorized_transactions, jurisdiction_rules
        )
        profit_loss = self._generate_profit_loss_report(
            categorized_transactions, trades, jurisdiction_rules
        )
        capital_gains = self._generate_capital_gains_report(
            trades, jurisdiction_rules
        )
        interest_income = self._generate_interest_income_report(
            categorized_transactions, jurisdiction_rules
        )
        
        # Combine into a complete report
        complete_report = {
            "transaction_summary": transaction_summary,
            "profit_loss": profit_loss,
            "capital_gains": capital_gains,
            "interest_income": interest_income,
            "total_taxable_amount": (
                capital_gains["total_net_capital_gain"] +
                interest_income["total_interest_income"]
            )
        }
        
        # Add estimated total tax if rates are available
        estimated_tax = 0
        if "estimated_total_tax" in capital_gains:
            estimated_tax += capital_gains["estimated_total_tax"]
        if "estimated_tax" in interest_income:
            estimated_tax += interest_income["estimated_tax"]
        
        complete_report["estimated_total_tax"] = estimated_tax
        
        return complete_report
    
    async def _calculate_unrealized_gains_losses(
        self,
        account_id: str,
        as_of_date: date,
        jurisdiction_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate unrealized gains and losses as of a specific date.
        
        Args:
            account_id: The account ID
            as_of_date: Date to calculate unrealized gains/losses
            jurisdiction_rules: Rules for the tax jurisdiction
            
        Returns:
            Dictionary with unrealized gains/losses data
        """
        # Get open positions as of the date
        positions = await self.portfolio_repository.get_open_positions(
            account_id, datetime.combine(as_of_date, datetime.min.time())
        )
        
        total_unrealized = sum(p.get("unrealized_pl", 0) for p in positions)
        
        unrealized_data = {
            "total_unrealized_pl": total_unrealized,
            "by_instrument": {},
            "positions": positions,
            "note": "Unrealized gains/losses are not taxable in most jurisdictions until realized."
        }
        
        # Calculate unrealized P&L by instrument
        for position in positions:
            instrument = position.get("instrument_id")
            unrealized_pl = position.get("unrealized_pl", 0)
            
            if instrument not in unrealized_data["by_instrument"]:
                unrealized_data["by_instrument"][instrument] = {
                    "total_unrealized_pl": 0,
                    "position_count": 0
                }
                
            unrealized_data["by_instrument"][instrument]["total_unrealized_pl"] += unrealized_pl
            unrealized_data["by_instrument"][instrument]["position_count"] += 1
        
        # Add tax implications based on jurisdiction rules
        if "unrealized_gains_taxable" in jurisdiction_rules:
            unrealized_data["taxable"] = jurisdiction_rules["unrealized_gains_taxable"]
            
            if jurisdiction_rules["unrealized_gains_taxable"]:
                unrealized_data["estimated_tax"] = (
                    unrealized_data["total_unrealized_pl"] * 
                    jurisdiction_rules.get("unrealized_gains_rate", 0)
                )
        
        return unrealized_data
