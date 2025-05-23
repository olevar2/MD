"""
Account Reconciliation Service Facade.

This module provides a facade for all account reconciliation functionality,
maintaining backward compatibility with the original monolithic implementation.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
import asyncio
import uuid
from collections import defaultdict
from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher
from core.base import ReconciliationBase
from core.basic_reconciliation import BasicReconciliation
from core.position_reconciliation import PositionReconciliation
from core.full_reconciliation import FullReconciliation
from core.historical_analysis import HistoricalAnalysis
from repositories.reporting import ReconciliationReporting
from core.discrepancy_handling import DiscrepancyHandling
logger = get_logger(__name__)


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

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

    def __init__(self, account_repository=None, portfolio_repository=None,
        trading_gateway_client=None, event_publisher: Optional[
        EventPublisher]=None, reconciliation_repository=None):
        """
        Initialize the reconciliation service.
        
        Args:
            account_repository: Repository for internal account data
            portfolio_repository: Repository for portfolio data
            trading_gateway_client: Client for accessing broker data
            event_publisher: Event publisher for notifications
            reconciliation_repository: Repository for storing reconciliation records
        """
        self.base = ReconciliationBase(account_repository=
            account_repository, portfolio_repository=portfolio_repository,
            trading_gateway_client=trading_gateway_client, event_publisher=
            event_publisher, reconciliation_repository=
            reconciliation_repository)
        self.basic_reconciliation = BasicReconciliation(account_repository=
            account_repository, portfolio_repository=portfolio_repository,
            trading_gateway_client=trading_gateway_client, event_publisher=
            event_publisher, reconciliation_repository=
            reconciliation_repository)
        self.position_reconciliation = PositionReconciliation(
            account_repository=account_repository, portfolio_repository=
            portfolio_repository, trading_gateway_client=
            trading_gateway_client, event_publisher=event_publisher,
            reconciliation_repository=reconciliation_repository)
        self.full_reconciliation = FullReconciliation(account_repository=
            account_repository, portfolio_repository=portfolio_repository,
            trading_gateway_client=trading_gateway_client, event_publisher=
            event_publisher, reconciliation_repository=
            reconciliation_repository)
        self.historical_analysis = HistoricalAnalysis(account_repository=
            account_repository, portfolio_repository=portfolio_repository,
            trading_gateway_client=trading_gateway_client, event_publisher=
            event_publisher, reconciliation_repository=
            reconciliation_repository)
        self.reporting = ReconciliationReporting(account_repository=
            account_repository, portfolio_repository=portfolio_repository,
            trading_gateway_client=trading_gateway_client, event_publisher=
            event_publisher, reconciliation_repository=
            reconciliation_repository)
        self.discrepancy_handling = DiscrepancyHandling(account_repository=
            account_repository, portfolio_repository=portfolio_repository,
            trading_gateway_client=trading_gateway_client, event_publisher=
            event_publisher, reconciliation_repository=
            reconciliation_repository)
        self.account_repository = account_repository
        self.portfolio_repository = portfolio_repository
        self.trading_gateway_client = trading_gateway_client
        self.event_publisher = event_publisher
        self.reconciliation_repository = reconciliation_repository
        self.active_reconciliations = {}
        self.recent_reconciliations = {}
        logger.info('AccountReconciliationService initialized')

    @async_with_exception_handling
    async def reconcile_account(self, account_id: str, reconciliation_level:
        str='full', tolerance: float=0.01, notification_threshold: float=
        1.0, auto_fix: bool=False, reference_time: Optional[datetime]=None
        ) ->Dict[str, Any]:
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
        reconciliation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        self.active_reconciliations[reconciliation_id] = {'account_id':
            account_id, 'start_time': start_time, 'status': 'in_progress'}
        try:
            logger.info(
                f'Starting reconciliation for account {account_id} with level {reconciliation_level}'
                )
            internal_data = await self.base._get_internal_account_data(
                account_id, reference_time)
            broker_data = await self.base._get_broker_account_data(account_id,
                reference_time)
            if reconciliation_level == 'basic':
                reconciliation_result = (await self.basic_reconciliation.
                    perform_reconciliation(internal_data, broker_data,
                    tolerance))
            elif reconciliation_level == 'positions':
                reconciliation_result = (await self.position_reconciliation
                    .perform_reconciliation(internal_data, broker_data,
                    tolerance))
            else:
                reconciliation_result = (await self.full_reconciliation.
                    perform_reconciliation(internal_data, broker_data,
                    tolerance))
            report = await self.base._create_reconciliation_report(
                reconciliation_id, account_id, reconciliation_level,
                internal_data, broker_data, reconciliation_result,
                start_time, tolerance)
            if report['discrepancies']['total_count'] > 0:
                await self.discrepancy_handling.handle_discrepancies(account_id
                    , report, notification_threshold, auto_fix)
            report['status'] = 'completed'
            report['completion_time'] = datetime.utcnow()
            report['duration_seconds'] = (report['completion_time'] -
                start_time).total_seconds()
            self.recent_reconciliations[reconciliation_id] = report
            if reconciliation_id in self.active_reconciliations:
                del self.active_reconciliations[reconciliation_id]
            logger.info(
                f"Reconciliation completed for account {account_id}, found {report['discrepancies']['total_count']} discrepancies"
                )
            return report
        except Exception as e:
            logger.error(f'Error during account reconciliation: {str(e)}',
                exc_info=True)
            if reconciliation_id in self.active_reconciliations:
                self.active_reconciliations[reconciliation_id].update({
                    'status': 'failed', 'error': str(e), 'completion_time':
                    datetime.utcnow()})
            await self.base._publish_event('reconciliation.failed', {
                'account_id': account_id, 'reconciliation_id':
                reconciliation_id, 'error': str(e), 'timestamp': datetime.
                utcnow().isoformat()})
            raise ValueError(f'Account reconciliation failed: {str(e)}')

    async def start_scheduled_reconciliation(self, account_id: str,
        frequency: str='daily', reconciliation_level: str='full', tolerance:
        float=0.01, notification_threshold: float=1.0, auto_fix: bool=False
        ) ->Dict[str, Any]:
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
        reconciliation = await self.reconcile_account(account_id=account_id,
            reconciliation_level=reconciliation_level, tolerance=tolerance,
            notification_threshold=notification_threshold, auto_fix=auto_fix)
        return {'account_id': account_id, 'configuration': {'frequency':
            frequency, 'reconciliation_level': reconciliation_level,
            'tolerance': tolerance, 'notification_threshold':
            notification_threshold, 'auto_fix': auto_fix, 'status':
            'active'}, 'initial_reconciliation': {'reconciliation_id':
            reconciliation['reconciliation_id'], 'timestamp':
            reconciliation['start_time'], 'discrepancies_found': 
            reconciliation['discrepancies']['total_count'] > 0}, 'message':
            f'Scheduled reconciliation configured with {frequency} frequency'}

    async def get_reconciliation_status(self, reconciliation_id: str) ->Dict[
        str, Any]:
        """
        Get the status and results of a specific reconciliation.
        
        Args:
            reconciliation_id: ID of the reconciliation process
            
        Returns:
            Dict[str, Any]: Reconciliation status and results
        """
        if reconciliation_id in self.active_reconciliations:
            return {'reconciliation_id': reconciliation_id, 'status':
                'in_progress', 'details': self.active_reconciliations[
                reconciliation_id]}
        if reconciliation_id in self.recent_reconciliations:
            return self.recent_reconciliations[reconciliation_id]
        return {'reconciliation_id': reconciliation_id, 'status':
            'not_found', 'message':
            'Reconciliation not found or expired from cache'}

    async def get_recent_reconciliations(self, account_id: Optional[str]=
        None, limit: int=10) ->List[Dict[str, Any]]:
        """
        Get recent reconciliation results.
        
        Args:
            account_id: Optional filter by account ID
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Recent reconciliation results
        """
        results = []
        for recon_id, recon_data in sorted(self.recent_reconciliations.
            items(), key=lambda x: x[1].get('start_time', datetime.min),
            reverse=True):
            if account_id and recon_data.get('account_id') != account_id:
                continue
            results.append(recon_data)
            if len(results) >= limit:
                break
        return results

    async def fix_discrepancy(self, reconciliation_id: str, discrepancy_id:
        str, fix_source: str='broker', comment: Optional[str]=None) ->Dict[
        str, Any]:
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
        self.discrepancy_handling.recent_reconciliations = (self.
            recent_reconciliations)
        return await self.discrepancy_handling.fix_discrepancy(
            reconciliation_id=reconciliation_id, discrepancy_id=
            discrepancy_id, fix_source=fix_source, comment=comment)

    @async_with_exception_handling
    async def perform_historical_reconciliation_analysis(self, account_id:
        str, start_date: datetime, end_date: datetime, interval: str=
        'daily', reconciliation_level: str='basic', tolerance: float=0.01
        ) ->Dict[str, Any]:
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
            f'Starting historical reconciliation analysis for account {account_id} from {start_date} to {end_date} with {interval} interval'
            )
        time_points = self.historical_analysis._generate_time_points(start_date
            , end_date, interval)
        if not time_points:
            raise ValueError(
                'No valid time points generated for historical reconciliation')
        reconciliation_results = []
        for point in time_points:
            try:
                logger.info(f'Running reconciliation for time point: {point}')
                result = await self.reconcile_account(account_id=account_id,
                    reconciliation_level=reconciliation_level, tolerance=
                    tolerance, reference_time=point, auto_fix=False)
                result['time_point'] = point
                reconciliation_results.append(result)
            except Exception as e:
                logger.error(
                    f'Error during historical reconciliation at {point}: {str(e)}'
                    , exc_info=True)
                reconciliation_results.append({'time_point': point,
                    'status': 'failed', 'error': str(e)})
        analysis = (await self.historical_analysis.
            analyze_historical_reconciliation(reconciliation_results))
        report = {'account_id': account_id, 'analysis_period': {'start':
            start_date.isoformat(), 'end': end_date.isoformat(), 'interval':
            interval}, 'reconciliation_count': len(reconciliation_results),
            'successful_count': sum(1 for r in reconciliation_results if r.
            get('status') != 'failed'), 'failed_count': sum(1 for r in
            reconciliation_results if r.get('status') == 'failed'),
            'analysis': analysis, 'time_points': [r.get('time_point').
            isoformat() for r in reconciliation_results], 'generation_time':
            datetime.utcnow().isoformat()}
        if self.reconciliation_repository:
            analysis_id = str(uuid.uuid4())
            await self.reconciliation_repository.store_historical_analysis(
                analysis_id, report)
            report['analysis_id'] = analysis_id
        logger.info(
            f'Completed historical reconciliation analysis for account {account_id}'
            )
        return report

    async def get_historical_reconciliation_report(self, account_id: str,
        start_date: Optional[datetime]=None, end_date: Optional[datetime]=
        None, report_format: str='summary') ->Dict[str, Any]:
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
            raise ValueError(
                'Reconciliation repository required for historical reporting')
        end_date = end_date or datetime.utcnow()
        start_date = start_date or end_date - timedelta(days=30)
        reconciliations = (await self.reconciliation_repository.
            get_reconciliations_by_timerange(account_id=account_id,
            start_date=start_date, end_date=end_date))
        if not reconciliations:
            return {'account_id': account_id, 'period': {'start':
                start_date.isoformat(), 'end': end_date.isoformat()},
                'message':
                'No reconciliation records found for the specified period',
                'reconciliation_count': 0}
        if report_format == 'summary':
            report = self.reporting.generate_summary_report(account_id,
                reconciliations, start_date, end_date)
        elif report_format == 'detailed':
            report = self.reporting.generate_detailed_report(account_id,
                reconciliations, start_date, end_date)
        elif report_format == 'chart':
            report = await self.reporting.generate_chart_report(account_id,
                reconciliations, start_date, end_date)
        else:
            raise ValueError(f'Unsupported report format: {report_format}')
        return report

    async def detect_reconciliation_patterns(self, account_id: str,
        lookback_days: int=90) ->Dict[str, Any]:
        """
        Detect patterns and trends in reconciliation data over time.
        
        Args:
            account_id: ID of the account
            lookback_days: Number of days to look back
            
        Returns:
            Dict[str, Any]: Pattern analysis results
        """
        if not self.reconciliation_repository:
            raise ValueError(
                'Reconciliation repository required for pattern detection')
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        reconciliations = (await self.reconciliation_repository.
            get_reconciliations_by_timerange(account_id=account_id,
            start_date=start_date, end_date=end_date))
        if not reconciliations:
            return {'account_id': account_id, 'lookback_days':
                lookback_days, 'message':
                'Insufficient data for pattern detection',
                'patterns_detected': False}
        processed_data = self.historical_analysis._process_reconciliation_data(
            reconciliations)
        recurring_patterns = (self.historical_analysis.
            _detect_recurring_discrepancies(processed_data))
        trends = self.historical_analysis._detect_discrepancy_trends(
            processed_data)
        correlations = (await self.historical_analysis.
            _detect_external_correlations(account_id, processed_data))
        return {'account_id': account_id, 'analysis_period': {'start':
            start_date.isoformat(), 'end': end_date.isoformat(), 'days':
            lookback_days}, 'reconciliation_count': len(reconciliations),
            'patterns_detected': len(recurring_patterns) > 0 or len(trends) >
            0 or len(correlations) > 0, 'recurring_patterns':
            recurring_patterns, 'trends': trends, 'correlations':
            correlations, 'generation_time': datetime.utcnow().isoformat()}

    async def compare_broker_accuracy(self, account_id: str, lookback_days:
        int=90) ->Dict[str, Any]:
        """
        Compare the accuracy of broker data over time by analyzing reconciliation results.
        
        Args:
            account_id: ID of the account
            lookback_days: Number of days to look back
            
        Returns:
            Dict[str, Any]: Broker accuracy analysis
        """
        if not self.reconciliation_repository:
            raise ValueError(
                'Reconciliation repository required for broker accuracy analysis'
                )
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        reconciliations = (await self.reconciliation_repository.
            get_reconciliations_by_timerange(account_id=account_id,
            start_date=start_date, end_date=end_date))
        if not reconciliations:
            return {'account_id': account_id, 'lookback_days':
                lookback_days, 'message':
                'Insufficient data for broker accuracy analysis',
                'analysis_complete': False}
        total_reconciliations = len(reconciliations)
        reconciliations_with_discrepancies = sum(1 for r in reconciliations if
            r.get('discrepancies', {}).get('total_count', 0) > 0)
        field_analysis = defaultdict(lambda : {'total': 0, 'discrepancies':
            0, 'avg_discrepancy_pct': 0})
        for recon in reconciliations:
            for disc in recon.get('discrepancies', {}).get('details', []):
                field = disc.get('field')
                if field:
                    field_analysis[field]['total'] += 1
                    field_analysis[field]['discrepancies'] += 1
                    field_analysis[field]['avg_discrepancy_pct'] += disc.get(
                        'percentage_difference', 0)
        for field_data in field_analysis.values():
            if field_data['discrepancies'] > 0:
                field_data['avg_discrepancy_pct'] /= field_data['discrepancies'
                    ]
        overall_accuracy = (total_reconciliations -
            reconciliations_with_discrepancies
            ) / total_reconciliations * 100 if total_reconciliations > 0 else 0
        return {'account_id': account_id, 'analysis_period': {'start':
            start_date.isoformat(), 'end': end_date.isoformat(), 'days':
            lookback_days}, 'total_reconciliations': total_reconciliations,
            'reconciliations_with_discrepancies':
            reconciliations_with_discrepancies,
            'overall_accuracy_percentage': overall_accuracy,
            'field_accuracy': {field: {'accuracy_percentage': (1 - data[
            'discrepancies'] / data['total']) * 100 if data['total'] > 0 else
            100, 'avg_discrepancy_percentage': data['avg_discrepancy_pct'],
            'total_occurrences': data['total']} for field, data in
            field_analysis.items()}, 'generation_time': datetime.utcnow().
            isoformat(), 'analysis_complete': True}
