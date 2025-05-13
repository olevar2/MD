"""
Discrepancy handling implementation.

This module provides functionality for handling reconciliation discrepancies,
including automatic fixing, notification, and manual resolution.
"""
from datetime import datetime
from typing import Dict, Any, Optional
from portfolio_management_service.services.account_reconciliation.base import ReconciliationBase
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from portfolio_management_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DiscrepancyHandling(ReconciliationBase):
    """
    Discrepancy handling implementation.
    """

    async def handle_discrepancies(self, account_id: str, report: Dict[str,
        Any], notification_threshold: float, auto_fix: bool) ->Dict[str, Any]:
        """
        Handle discrepancies based on severity and configuration.
        
        Args:
            account_id: ID of the account
            report: Reconciliation report
            notification_threshold: Threshold percentage for notifications
            auto_fix: Whether to automatically fix minor discrepancies
            
        Returns:
            Dict[str, Any]: Updated report with handling results
        """
        if report['discrepancies']['total_count'] == 0:
            return report
        total_discrepancy_pct = 0
        for disc in report['discrepancies']['details']:
            if 'percentage_difference' in disc and disc['percentage_difference'
                ] is not None:
                total_discrepancy_pct += disc['percentage_difference']
        if total_discrepancy_pct > notification_threshold * 100:
            await self._send_discrepancy_notification(account_id, report)
        if auto_fix:
            await self._auto_fix_discrepancies(account_id, report)
        return report

    async def _send_discrepancy_notification(self, account_id: str, report:
        Dict[str, Any]) ->None:
        """
        Send notification about reconciliation discrepancies.
        
        Args:
            account_id: ID of the account
            report: Reconciliation report
        """
        notification_data = {'account_id': account_id, 'reconciliation_id':
            report['reconciliation_id'], 'discrepancy_count': report[
            'discrepancies']['total_count'], 'severity_summary': report[
            'discrepancies']['by_severity'], 'field_summary': report[
            'discrepancies']['by_field_type'], 'timestamp': datetime.utcnow
            ().isoformat()}
        await self._publish_event('reconciliation.discrepancy.notification',
            notification_data)
        logger.info(
            f"Sent discrepancy notification for account {account_id} with {report['discrepancies']['total_count']} discrepancies"
            )

    async def _auto_fix_discrepancies(self, account_id: str, report: Dict[
        str, Any]) ->None:
        """
        Automatically fix low severity discrepancies.
        
        Args:
            account_id: ID of the account
            report: Reconciliation report
        """
        low_severity_count = 0
        for disc in report['discrepancies']['details']:
            if disc.get('severity') == 'low':
                low_severity_count += 1
                disc['status'] = 'auto_fixed'
                disc['fixed_at'] = datetime.utcnow().isoformat()
                disc['fixed_by'] = 'auto_fix'
                logger.info(
                    f"Auto-fixed discrepancy: {disc['field']} for account {account_id}"
                    )
        if low_severity_count > 0:
            await self._publish_event('reconciliation.auto_fixed', {
                'account_id': account_id, 'reconciliation_id': report[
                'reconciliation_id'], 'discrepancies_fixed':
                low_severity_count, 'timestamp': datetime.utcnow().isoformat()}
                )

    @async_with_exception_handling
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
        recon_data = self.recent_reconciliations.get(reconciliation_id)
        if not recon_data:
            raise ValueError(f'Reconciliation {reconciliation_id} not found')
        discrepancy = None
        for disc in recon_data.get('discrepancies', {}).get('details', []):
            if disc.get('discrepancy_id') == discrepancy_id:
                discrepancy = disc
                break
        if not discrepancy:
            raise ValueError(
                f'Discrepancy {discrepancy_id} not found in reconciliation {reconciliation_id}'
                )
        try:
            if fix_source == 'internal':
                await self._update_broker_data(recon_data['account_id'],
                    discrepancy['field'], discrepancy['internal_value'])
                source_desc = 'internal system'
            else:
                await self._update_internal_data(recon_data['account_id'],
                    discrepancy['field'], discrepancy['broker_value'])
                source_desc = 'broker'
            discrepancy['status'] = 'fixed'
            discrepancy['fixed_at'] = datetime.utcnow().isoformat()
            discrepancy['fixed_by'] = fix_source
            discrepancy['comment'] = comment
            await self._publish_event('reconciliation.discrepancy.fixed', {
                'account_id': recon_data['account_id'], 'reconciliation_id':
                reconciliation_id, 'discrepancy_id': discrepancy_id,
                'field': discrepancy['field'], 'fixed_using': fix_source,
                'timestamp': datetime.utcnow().isoformat()})
            return {'status': 'success', 'message':
                f'Discrepancy fixed using {source_desc} as source of truth',
                'discrepancy': discrepancy}
        except Exception as e:
            logger.error(f'Error fixing discrepancy: {str(e)}', exc_info=True)
            discrepancy['status'] = 'fix_failed'
            discrepancy['fix_error'] = str(e)
            raise ValueError(f'Failed to fix discrepancy: {str(e)}')
