"""
Account Snapshot Service

This service provides functionality for creating and managing account snapshots with
time-series capabilities, allowing for historical analysis and tracking of account metrics.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from core_foundations.utils.logger import get_logger
from core.account import AccountSnapshot
from portfolio_management_service.models.performance import TimeSeriesMetric
logger = get_logger(__name__)


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AccountSnapshotService:
    """
    Service for creating and retrieving account snapshots with time-series capabilities.
    
    This service enables:
    - Taking point-in-time snapshots of account state
    - Retrieving historical account snapshots
    - Analyzing account metrics over time
    - Comparing account states across different time periods
    """

    def __init__(self, account_repository=None, portfolio_repository=None,
        snapshot_repository=None, metrics_repository=None):
        """
        Initialize the account snapshot service with required repositories.
        
        Args:
            account_repository: Repository for account data
            portfolio_repository: Repository for portfolio data
            snapshot_repository: Repository for storing and retrieving snapshots
            metrics_repository: Repository for time-series metrics
        """
        self.account_repository = account_repository
        self.portfolio_repository = portfolio_repository
        self.snapshot_repository = snapshot_repository
        self.metrics_repository = metrics_repository

    @async_with_exception_handling
    async def create_snapshot(self, account_id: str, timestamp: Optional[
        datetime]=None, include_positions: bool=True, include_orders: bool=
        False, tags: Optional[List[str]]=None) ->AccountSnapshot:
        """
        Create a point-in-time snapshot of an account's state.
        
        Args:
            account_id: ID of the account to snapshot
            timestamp: Optional timestamp for the snapshot (defaults to now)
            include_positions: Whether to include position details
            include_orders: Whether to include open orders
            tags: Optional tags for categorizing snapshots
            
        Returns:
            AccountSnapshot: The created account snapshot
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        try:
            portfolio = await self.portfolio_repository.get_portfolio(
                account_id)
        except Exception as e:
            logger.error(f'Error retrieving portfolio for snapshot: {str(e)}',
                exc_info=True)
            raise ValueError(f'Failed to create account snapshot: {str(e)}')
        snapshot = AccountSnapshot(account_id=account_id, snapshot_id=
            f'snap_{account_id}_{int(timestamp.timestamp())}', timestamp=
            timestamp, balance=portfolio.balance, equity=portfolio.equity,
            margin_used=portfolio.margin, free_margin=portfolio.free_margin,
            margin_level=portfolio.margin_level, positions=portfolio.
            positions if include_positions else None, orders=portfolio.
            orders if include_orders else None, tags=tags or [])
        if self.snapshot_repository:
            await self.snapshot_repository.store_snapshot(snapshot)
            if self.metrics_repository:
                await self._store_snapshot_metrics(snapshot)
        logger.info(f'Created account snapshot for {account_id} at {timestamp}'
            )
        return snapshot

    async def get_snapshot(self, snapshot_id: str) ->AccountSnapshot:
        """
        Retrieve a specific account snapshot by ID.
        
        Args:
            snapshot_id: ID of the snapshot to retrieve
            
        Returns:
            AccountSnapshot: The retrieved snapshot
        """
        if not self.snapshot_repository:
            raise ValueError('Snapshot repository not available')
        snapshot = await self.snapshot_repository.get_snapshot(snapshot_id)
        return snapshot

    async def get_snapshots(self, account_id: str, start_time: Optional[
        datetime]=None, end_time: Optional[datetime]=None, tags: Optional[
        List[str]]=None, limit: int=100, offset: int=0) ->List[AccountSnapshot
        ]:
        """
        Retrieve account snapshots with optional filtering.
        
        Args:
            account_id: ID of the account
            start_time: Optional start time filter
            end_time: Optional end time filter
            tags: Optional tags filter
            limit: Maximum number of snapshots to return
            offset: Number of snapshots to skip
            
        Returns:
            List[AccountSnapshot]: List of account snapshots
        """
        if not self.snapshot_repository:
            raise ValueError('Snapshot repository not available')
        snapshots = await self.snapshot_repository.get_snapshots(account_id
            =account_id, start_time=start_time, end_time=end_time, tags=
            tags, limit=limit, offset=offset)
        return snapshots

    async def get_nearest_snapshot(self, account_id: str, timestamp:
        datetime, max_time_diff: Optional[timedelta]=None) ->Optional[
        AccountSnapshot]:
        """
        Get the nearest snapshot to a specific point in time.
        
        Args:
            account_id: ID of the account
            timestamp: Target timestamp
            max_time_diff: Maximum allowed time difference
            
        Returns:
            Optional[AccountSnapshot]: The nearest snapshot, or None if none found
        """
        if not self.snapshot_repository:
            raise ValueError('Snapshot repository not available')
        if max_time_diff is None:
            max_time_diff = timedelta(hours=1)
        snapshot = await self.snapshot_repository.get_nearest_snapshot(
            account_id=account_id, timestamp=timestamp, max_time_diff=
            max_time_diff)
        return snapshot

    async def get_account_at_timestamp(self, account_id: str, timestamp:
        datetime) ->Dict[str, Any]:
        """
        Get account state at a specific timestamp.
        
        This will use the nearest snapshot if available, otherwise will try to
        reconstruct the account state from transactions.
        
        Args:
            account_id: ID of the account
            timestamp: Target timestamp
            
        Returns:
            Dict[str, Any]: Account state at the timestamp
        """
        snapshot = await self.get_nearest_snapshot(account_id=account_id,
            timestamp=timestamp, max_time_diff=timedelta(hours=1))
        if snapshot:
            return {'account_id': snapshot.account_id, 'timestamp':
                snapshot.timestamp, 'balance': snapshot.balance, 'equity':
                snapshot.equity, 'margin_used': snapshot.margin_used,
                'free_margin': snapshot.free_margin, 'margin_level':
                snapshot.margin_level, 'positions': snapshot.positions if
                hasattr(snapshot, 'positions') else [], 'source': 'snapshot'}
        else:
            logger.warning(
                f'No snapshot found for {account_id} at {timestamp}, reconstructing from transactions'
                )
            if self.account_repository:
                account_data = (await self.account_repository.
                    reconstruct_account_at_timestamp(account_id=account_id,
                    timestamp=timestamp))
                account_data['source'] = 'reconstructed'
                return account_data
            else:
                raise ValueError(
                    'Account repository not available for state reconstruction'
                    )

    async def compare_snapshots(self, snapshot1_id: str, snapshot2_id: str
        ) ->Dict[str, Any]:
        """
        Compare two account snapshots.
        
        Args:
            snapshot1_id: ID of the first snapshot
            snapshot2_id: ID of the second snapshot
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        if not self.snapshot_repository:
            raise ValueError('Snapshot repository not available')
        snapshot1 = await self.snapshot_repository.get_snapshot(snapshot1_id)
        snapshot2 = await self.snapshot_repository.get_snapshot(snapshot2_id)
        if not snapshot1 or not snapshot2:
            missing = []
            if not snapshot1:
                missing.append(snapshot1_id)
            if not snapshot2:
                missing.append(snapshot2_id)
            raise ValueError(f"Snapshot(s) not found: {', '.join(missing)}")
        if snapshot1.account_id != snapshot2.account_id:
            raise ValueError('Cannot compare snapshots from different accounts'
                )
        if snapshot1.timestamp > snapshot2.timestamp:
            snapshot1, snapshot2 = snapshot2, snapshot1
        time_diff_seconds = (snapshot2.timestamp - snapshot1.timestamp
            ).total_seconds()
        balance_change = snapshot2.balance - snapshot1.balance
        equity_change = snapshot2.equity - snapshot1.equity
        margin_used_change = snapshot2.margin_used - snapshot1.margin_used
        balance_pct_change = (balance_change / snapshot1.balance * 100 if
            snapshot1.balance else 0)
        equity_pct_change = (equity_change / snapshot1.equity * 100 if
            snapshot1.equity else 0)
        comparison = {'account_id': snapshot1.account_id, 'snapshot1': {
            'id': snapshot1.snapshot_id, 'timestamp': snapshot1.timestamp},
            'snapshot2': {'id': snapshot2.snapshot_id, 'timestamp':
            snapshot2.timestamp}, 'time_diff_seconds': time_diff_seconds,
            'changes': {'balance': {'absolute': balance_change,
            'percentage': balance_pct_change}, 'equity': {'absolute':
            equity_change, 'percentage': equity_pct_change}, 'margin_used':
            margin_used_change}}
        if hasattr(snapshot1, 'positions') and hasattr(snapshot2, 'positions'
            ) and snapshot1.positions and snapshot2.positions:
            comparison['position_changes'] = self._calculate_position_changes(
                snapshot1.positions, snapshot2.positions)
        return comparison

    async def create_scheduled_snapshots(self, account_id: str, frequency:
        str='daily', include_positions: bool=True, tags: Optional[List[str]
        ]=None) ->Dict[str, Any]:
        """
        Configure scheduled snapshots for an account.
        
        Args:
            account_id: ID of the account
            frequency: Snapshot frequency ('hourly', 'daily', 'weekly')
            include_positions: Whether to include positions in snapshots
            tags: Optional tags for the snapshots
            
        Returns:
            Dict[str, Any]: Configuration result
        """
        valid_frequencies = ['hourly', 'daily', 'weekly', 'monthly']
        if frequency not in valid_frequencies:
            raise ValueError(
                f'Invalid frequency: {frequency}. Must be one of {valid_frequencies}'
                )
        snapshot = await self.create_snapshot(account_id=account_id,
            include_positions=include_positions, tags=[frequency] + (tags or
            []))
        return {'account_id': account_id, 'configuration': {'frequency':
            frequency, 'include_positions': include_positions, 'tags': tags or
            []}, 'first_snapshot': {'snapshot_id': snapshot.snapshot_id,
            'timestamp': snapshot.timestamp}, 'message':
            f'Scheduled snapshots configured with {frequency} frequency'}

    async def get_metric_time_series(self, account_id: str, metric: str,
        start_time: Optional[datetime]=None, end_time: Optional[datetime]=
        None, interval: str='daily') ->List[TimeSeriesMetric]:
        """
        Retrieve a time series for a specific account metric.
        
        Args:
            account_id: ID of the account
            metric: Name of the metric (e.g., 'balance', 'equity', 'margin_level')
            start_time: Start time for the series (default: 30 days ago)
            end_time: End time for the series (default: now)
            interval: Data interval ('hourly', 'daily', 'weekly')
            
        Returns:
            List[TimeSeriesMetric]: Time series data for the metric
        """
        if not self.metrics_repository:
            raise ValueError('Metrics repository not available')
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        time_series = await self.metrics_repository.get_metric_time_series(
            account_id=account_id, metric_name=metric, start_time=
            start_time, end_time=end_time, interval=interval)
        return time_series

    async def delete_snapshot(self, snapshot_id: str) ->bool:
        """
        Delete a specific account snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to delete
            
        Returns:
            bool: True if deletion was successful
        """
        if not self.snapshot_repository:
            raise ValueError('Snapshot repository not available')
        result = await self.snapshot_repository.delete_snapshot(snapshot_id)
        return result

    async def _store_snapshot_metrics(self, snapshot: AccountSnapshot) ->None:
        """
        Store key metrics from a snapshot for time-series analysis.
        
        Args:
            snapshot: The account snapshot
        """
        if not self.metrics_repository:
            return
        metrics = [TimeSeriesMetric(account_id=snapshot.account_id,
            metric_name='balance', timestamp=snapshot.timestamp, value=
            snapshot.balance), TimeSeriesMetric(account_id=snapshot.
            account_id, metric_name='equity', timestamp=snapshot.timestamp,
            value=snapshot.equity), TimeSeriesMetric(account_id=snapshot.
            account_id, metric_name='margin_used', timestamp=snapshot.
            timestamp, value=snapshot.margin_used), TimeSeriesMetric(
            account_id=snapshot.account_id, metric_name='margin_level',
            timestamp=snapshot.timestamp, value=snapshot.margin_level)]
        if hasattr(snapshot, 'positions') and snapshot.positions:
            metrics.append(TimeSeriesMetric(account_id=snapshot.account_id,
                metric_name='positions_count', timestamp=snapshot.timestamp,
                value=len(snapshot.positions)))
        for metric in metrics:
            await self.metrics_repository.store_metric(metric)

    def _calculate_position_changes(self, positions1: List[Any], positions2:
        List[Any]) ->Dict[str, Any]:
        """
        Calculate changes between two sets of positions.
        
        Args:
            positions1: First set of positions
            positions2: Second set of positions
            
        Returns:
            Dict[str, Any]: Position changes
        """
        pos1_dict = {p.position_id: p for p in positions1}
        pos2_dict = {p.position_id: p for p in positions2}
        common_ids = set(pos1_dict.keys()) & set(pos2_dict.keys())
        closed_ids = set(pos1_dict.keys()) - set(pos2_dict.keys())
        new_ids = set(pos2_dict.keys()) - set(pos1_dict.keys())
        position_changes = {'changed': [], 'closed': [], 'opened': []}
        for pos_id in common_ids:
            pos1 = pos1_dict[pos_id]
            pos2 = pos2_dict[pos_id]
            price_change = pos2.current_price - pos1.current_price
            pnl_change = pos2.unrealized_pnl - pos1.unrealized_pnl
            position_changes['changed'].append({'position_id': pos_id,
                'instrument': pos1.instrument, 'price_change': price_change,
                'pnl_change': pnl_change})
        for pos_id in closed_ids:
            pos = pos1_dict[pos_id]
            position_changes['closed'].append({'position_id': pos_id,
                'instrument': pos.instrument, 'direction': pos.direction,
                'size': pos.size})
        for pos_id in new_ids:
            pos = pos2_dict[pos_id]
            position_changes['opened'].append({'position_id': pos_id,
                'instrument': pos.instrument, 'direction': pos.direction,
                'size': pos.size, 'open_price': pos.open_price})
        position_changes['summary'] = {'total_changed': len(
            position_changes['changed']), 'total_closed': len(
            position_changes['closed']), 'total_opened': len(
            position_changes['opened'])}
        return position_changes
