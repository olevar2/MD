"""
Base classes and common functionality for account reconciliation.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
import uuid
from collections import defaultdict
from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher
logger = get_logger(__name__)


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ReconciliationBase:
    """
    Base class for all reconciliation operations.
    
    Provides common functionality for reconciliation operations including:
    - Event publishing
    - Discrepancy detection
    - Report generation
    """

    def __init__(self, account_repository=None, portfolio_repository=None,
        trading_gateway_client=None, event_publisher: Optional[
        EventPublisher]=None, reconciliation_repository=None):
        """
        Initialize the reconciliation base.
        
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
        logger.info('ReconciliationBase initialized')

    @async_with_exception_handling
    async def _publish_event(self, event_type: str, event_data: Dict[str, Any]
        ) ->None:
        """
        Publish an event using the event publisher.
        
        Args:
            event_type: Type of the event
            event_data: Event data
        """
        if self.event_publisher:
            try:
                await self.event_publisher.publish(event_type, event_data)
                logger.debug(f'Published event: {event_type}')
            except Exception as e:
                logger.error(f'Error publishing event {event_type}: {str(e)}',
                    exc_info=True)

    @async_with_exception_handling
    async def _get_internal_account_data(self, account_id: str,
        reference_time: Optional[datetime]=None) ->Dict[str, Any]:
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
                account_data = (await self.account_repository.
                    get_account_at_timestamp(account_id=account_id,
                    timestamp=reference_time))
                portfolio_data = (await self.portfolio_repository.
                    get_portfolio_at_timestamp(account_id=account_id,
                    timestamp=reference_time))
            else:
                account_data = await self.account_repository.get_account(
                    account_id)
                portfolio_data = await self.portfolio_repository.get_portfolio(
                    account_id)
            result = {'account_id': account_id, 'balance': account_data.
                balance, 'equity': portfolio_data.equity, 'margin_used':
                portfolio_data.margin, 'free_margin': portfolio_data.
                free_margin, 'margin_level': portfolio_data.margin_level,
                'timestamp': reference_time or datetime.utcnow(),
                'positions': [{'position_id': p.position_id, 'instrument':
                p.instrument, 'direction': p.direction, 'size': p.size,
                'open_price': p.open_price, 'current_price': p.
                current_price, 'swap': getattr(p, 'swap', 0),
                'unrealized_pnl': p.unrealized_pnl} for p in getattr(
                portfolio_data, 'positions', [])], 'orders': [{'order_id':
                o.order_id, 'instrument': o.instrument, 'type': o.type,
                'direction': o.direction, 'size': o.size, 'price': o.price} for
                o in getattr(portfolio_data, 'orders', [])]}
            return result
        except Exception as e:
            logger.error(f'Error retrieving internal account data: {str(e)}',
                exc_info=True)
            raise ValueError(f'Failed to get internal account data: {str(e)}')

    @async_with_exception_handling
    async def _get_broker_account_data(self, account_id: str,
        reference_time: Optional[datetime]=None) ->Dict[str, Any]:
        """
        Get account data from the broker via the trading gateway.
        
        Args:
            account_id: ID of the account
            reference_time: Optional reference time for historical data
            
        Returns:
            Dict[str, Any]: Broker account data
        """
        if not self.trading_gateway_client:
            raise ValueError('Trading gateway client not available')
        try:
            if reference_time:
                broker_account = (await self.trading_gateway_client.
                    get_account_history(account_id=account_id, timestamp=
                    reference_time))
            else:
                broker_account = await self.trading_gateway_client.get_account(
                    account_id=account_id)
            broker_positions = await self.trading_gateway_client.get_positions(
                account_id=account_id)
            broker_orders = await self.trading_gateway_client.get_orders(
                account_id=account_id)
            result = {'account_id': account_id, 'balance': broker_account.
                get('balance'), 'equity': broker_account.get('equity'),
                'margin_used': broker_account.get('margin_used'),
                'free_margin': broker_account.get('free_margin'),
                'margin_level': broker_account.get('margin_level'),
                'timestamp': reference_time or datetime.utcnow(),
                'positions': [{'position_id': p.get('id'), 'instrument': p.
                get('symbol'), 'direction': p.get('type'), 'size': p.get(
                'volume'), 'open_price': p.get('open_price'),
                'current_price': p.get('current_price'), 'swap': p.get(
                'swap', 0), 'unrealized_pnl': p.get('profit')} for p in
                broker_positions], 'orders': [{'order_id': o.get('id'),
                'instrument': o.get('symbol'), 'type': o.get('type'),
                'direction': o.get('direction'), 'size': o.get('volume'),
                'price': o.get('price')} for o in broker_orders]}
            return result
        except Exception as e:
            logger.error(f'Error retrieving broker account data: {str(e)}',
                exc_info=True)
            raise ValueError(f'Failed to get broker account data: {str(e)}')

    async def _create_reconciliation_report(self, reconciliation_id: str,
        account_id: str, reconciliation_level: str, internal_data: Dict[str,
        Any], broker_data: Dict[str, Any], reconciliation_result: Dict[str,
        Any], start_time: datetime, tolerance: float) ->Dict[str, Any]:
        """
        Create a detailed reconciliation report.
        
        Args:
            reconciliation_id: ID of the reconciliation
            account_id: ID of the account
            reconciliation_level: Level of reconciliation
            internal_data: Internal account data
            broker_data: Broker account data
            reconciliation_result: Result of the reconciliation
            start_time: Start time of the reconciliation
            tolerance: Tolerance percentage for discrepancies
            
        Returns:
            Dict[str, Any]: Detailed reconciliation report
        """
        severity_counts = defaultdict(int)
        field_counts = defaultdict(int)
        total_monetary_diff = 0.0
        for disc in reconciliation_result['discrepancies']:
            severity_counts[disc.get('severity', 'medium')] += 1
            if disc['field'] in ['balance', 'equity', 'margin_used',
                'free_margin']:
                field_counts['account'] += 1
            elif 'position' in disc['field']:
                field_counts['positions'] += 1
            elif 'order' in disc['field']:
                field_counts['orders'] += 1
            if 'absolute_difference' in disc and isinstance(disc[
                'absolute_difference'], (int, float)):
                total_monetary_diff += abs(disc['absolute_difference'])
        report = {'reconciliation_id': reconciliation_id, 'account_id':
            account_id, 'reconciliation_level': reconciliation_level,
            'start_time': start_time, 'tolerance_percentage': tolerance * 
            100, 'discrepancies': {'total_count': len(reconciliation_result
            ['discrepancies']), 'by_severity': dict(severity_counts),
            'by_field_type': dict(field_counts),
            'total_monetary_difference': total_monetary_diff, 'details':
            reconciliation_result['discrepancies']}, 'matched_fields':
            reconciliation_result['matched_fields'], 'status': 'completed'}
        return report

    async def _update_internal_data(self, account_id: str, field: str,
        value: Any) ->None:
        """
        Update internal data based on reconciliation.
        
        Args:
            account_id: ID of the account
            field: Field to update
            value: New value
        """
        logger.info(
            f'Updating internal data for account {account_id}, field {field} to {value}'
            )
        pass

    async def _update_broker_data(self, account_id: str, field: str, value: Any
        ) ->None:
        """
        Update broker data based on reconciliation.
        
        Args:
            account_id: ID of the account
            field: Field to update
            value: New value
        """
        logger.info(
            f'Updating broker data for account {account_id}, field {field} to {value}'
            )
        pass
