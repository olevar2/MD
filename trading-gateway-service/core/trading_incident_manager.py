"""
Trading Incident Manager for classifying and responding to trading incidents.

This component handles trading emergencies, classifies incidents by severity and type,
and provides structured responses to various trading-related incidents.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
import json
from core_foundations.utils.logger import get_logger
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class IncidentSeverity(Enum):
    """Severity levels for trading incidents."""
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'
    INFO = 'info'


class IncidentCategory(Enum):
    """Categories of trading incidents."""
    CONNECTIVITY = 'connectivity'
    DATA_QUALITY = 'data_quality'
    EXECUTION = 'execution'
    POSITION_MANAGEMENT = 'position'
    RISK = 'risk'
    SYSTEM = 'system'
    COMPLIANCE = 'compliance'
    SECURITY = 'security'


class IncidentStatus(Enum):
    """Status of an incident."""
    OPEN = 'open'
    INVESTIGATING = 'investigating'
    MITIGATING = 'mitigating'
    RESOLVED = 'resolved'
    CLOSED = 'closed'


class TradingIncidentManager:
    """
    Manages trading incidents, classification, and response procedures.
    
    This component identifies, classifies, and manages responses to trading incidents,
    providing a structured approach to handling emergencies in the trading system.
    """

    def __init__(self, config: Optional[Dict[str, Any]]=None):
        """
        Initialize the TradingIncidentManager.
        
        Args:
            config: Configuration parameters
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
        self._incidents: Dict[str, Dict[str, Any]] = {}
        self._incident_handlers: Dict[Tuple[IncidentCategory,
            IncidentSeverity], List[Callable]] = {}
        self.emergency_action_system = self.config.get(
            'emergency_action_system', None)
        self.notification_service = self.config.get('notification_service',
            None)
        self.runbooks = self._load_runbooks()
        self._register_default_handlers()
        self.logger.info('TradingIncidentManager initialized')

    async def report_incident(self, title: str, category: IncidentCategory,
        severity: IncidentSeverity, description: str, source: str, context:
        Optional[Dict[str, Any]]=None, auto_mitigate: bool=True) ->str:
        """
        Report a new trading incident.
        
        Args:
            title: Short descriptive title of the incident
            category: Category of the incident
            severity: Severity level of the incident
            description: Detailed description of the incident
            source: Source of the incident (service/component name)
            context: Additional context information
            auto_mitigate: Whether to automatically attempt mitigation
            
        Returns:
            ID of the created incident
        """
        incident_id = str(uuid.uuid4())
        timestamp = datetime.now()
        incident = {'id': incident_id, 'title': title, 'category': category
            .value, 'severity': severity.value, 'description': description,
            'source': source, 'context': context or {}, 'status':
            IncidentStatus.OPEN.value, 'created_at': timestamp.isoformat(),
            'updated_at': timestamp.isoformat(), 'timeline': [{'timestamp':
            timestamp.isoformat(), 'event': 'Incident created', 'details':
            f'Reported by {source}'}], 'actions_taken': [], 'resolution':
            None, 'post_mortem': None}
        self._incidents[incident_id] = incident
        self.logger.info(
            f'Incident reported: {incident_id} - {title} [{severity.value}/{category.value}]'
            )
        await self._handle_incident(incident_id, incident, auto_mitigate)
        return incident_id

    @with_broker_api_resilience('update_incident')
    async def update_incident(self, incident_id: str, status: Optional[
        IncidentStatus]=None, notes: Optional[str]=None, actions_taken:
        Optional[List[str]]=None, resolution: Optional[str]=None) ->bool:
        """
        Update an existing incident.
        
        Args:
            incident_id: ID of the incident to update
            status: New status of the incident
            notes: Additional notes to add to the timeline
            actions_taken: Actions taken to address the incident
            resolution: Resolution details if resolved
            
        Returns:
            True if successful, False if incident not found
        """
        if incident_id not in self._incidents:
            self.logger.warning(
                f'Attempted to update non-existent incident: {incident_id}')
            return False
        incident = self._incidents[incident_id]
        timestamp = datetime.now()
        incident['updated_at'] = timestamp.isoformat()
        if status:
            old_status = incident['status']
            incident['status'] = status.value
            incident['timeline'].append({'timestamp': timestamp.isoformat(),
                'event':
                f'Status changed from {old_status} to {status.value}',
                'details': notes if notes else 'Status updated'})
            self.logger.info(
                f'Incident {incident_id} status changed: {old_status} -> {status.value}'
                )
        elif notes:
            incident['timeline'].append({'timestamp': timestamp.isoformat(),
                'event': 'Note added', 'details': notes})
        if actions_taken:
            for action in actions_taken:
                if action not in incident['actions_taken']:
                    incident['actions_taken'].append(action)
            if not notes:
                incident['timeline'].append({'timestamp': timestamp.
                    isoformat(), 'event': 'Actions taken', 'details': ', '.
                    join(actions_taken)})
        if resolution:
            incident['resolution'] = resolution
            if not notes and not status:
                incident['timeline'].append({'timestamp': timestamp.
                    isoformat(), 'event': 'Resolution added', 'details':
                    resolution})
        return True

    async def resolve_incident(self, incident_id: str, resolution: str,
        actions_taken: Optional[List[str]]=None) ->bool:
        """
        Mark an incident as resolved.
        
        Args:
            incident_id: ID of the incident to resolve
            resolution: Description of how the incident was resolved
            actions_taken: Actions taken to resolve the incident
            
        Returns:
            True if successful, False if incident not found
        """
        return await self.update_incident(incident_id=incident_id, status=
            IncidentStatus.RESOLVED, notes=
            f'Incident resolved: {resolution}', actions_taken=actions_taken,
            resolution=resolution)

    async def close_incident(self, incident_id: str, post_mortem: Optional[
        Dict[str, Any]]=None) ->bool:
        """
        Close a resolved incident after review.
        
        Args:
            incident_id: ID of the incident to close
            post_mortem: Post-mortem analysis details
            
        Returns:
            True if successful, False if incident not found or not resolved
        """
        if incident_id not in self._incidents:
            self.logger.warning(
                f'Attempted to close non-existent incident: {incident_id}')
            return False
        incident = self._incidents[incident_id]
        if incident['status'] != IncidentStatus.RESOLVED.value:
            self.logger.warning(
                f'Attempted to close non-resolved incident: {incident_id}')
            return False
        if post_mortem:
            incident['post_mortem'] = post_mortem
        return await self.update_incident(incident_id=incident_id, status=
            IncidentStatus.CLOSED, notes='Incident closed after review')

    @with_broker_api_resilience('get_incident')
    async def get_incident(self, incident_id: str) ->Optional[Dict[str, Any]]:
        """
        Get details of a specific incident.
        
        Args:
            incident_id: ID of the incident
            
        Returns:
            Incident details or None if not found
        """
        return self._incidents.get(incident_id)

    @with_broker_api_resilience('get_incidents')
    async def get_incidents(self, status: Optional[IncidentStatus]=None,
        category: Optional[IncidentCategory]=None, severity: Optional[
        IncidentSeverity]=None, source: Optional[str]=None, limit: int=100,
        offset: int=0) ->List[Dict[str, Any]]:
        """
        Get incidents matching the specified filters.
        
        Args:
            status: Filter by incident status
            category: Filter by incident category
            severity: Filter by incident severity
            source: Filter by incident source
            limit: Maximum number of incidents to return
            offset: Offset for pagination
            
        Returns:
            List of incidents matching the filters
        """
        filtered_incidents = []
        for incident in self._incidents.values():
            if status and incident['status'] != status.value:
                continue
            if category and incident['category'] != category.value:
                continue
            if severity and incident['severity'] != severity.value:
                continue
            if source and incident['source'] != source:
                continue
            filtered_incidents.append(incident)
        filtered_incidents.sort(key=lambda x: x['created_at'], reverse=True)
        paginated_incidents = filtered_incidents[offset:offset + limit]
        return paginated_incidents

    @with_broker_api_resilience('get_runbook')
    async def get_runbook(self, category: IncidentCategory, severity:
        Optional[IncidentSeverity]=None) ->Optional[Dict[str, Any]]:
        """
        Get a runbook for a specific incident category and severity.
        
        Args:
            category: Incident category
            severity: Optional severity level (gets most severe if not specified)
            
        Returns:
            Runbook details or None if not found
        """
        if category.value not in self.runbooks:
            return None
        category_runbooks = self.runbooks[category.value]
        if severity:
            return category_runbooks.get(severity.value)
        else:
            for sev in [IncidentSeverity.CRITICAL.value, IncidentSeverity.
                HIGH.value, IncidentSeverity.MEDIUM.value, IncidentSeverity
                .LOW.value]:
                if sev in category_runbooks:
                    return category_runbooks[sev]
        return None

    def register_incident_handler(self, category: IncidentCategory,
        severity: Optional[IncidentSeverity]=None, handler: Callable[[str,
        Dict[str, Any]], None]=None) ->None:
        """
        Register a handler function for specific incident types.
        
        Args:
            category: Incident category to handle
            severity: Optional severity level (None for all severities)
            handler: Handler function that takes incident_id and incident dict
        """
        if not handler:
            self.logger.warning('Attempted to register None handler')
            return
        if severity is None:
            for sev in IncidentSeverity:
                key = category, sev
                if key not in self._incident_handlers:
                    self._incident_handlers[key] = []
                self._incident_handlers[key].append(handler)
        else:
            key = category, severity
            if key not in self._incident_handlers:
                self._incident_handlers[key] = []
            self._incident_handlers[key].append(handler)
        self.logger.debug(
            f"Registered handler for {category.value}/{severity.value if severity else 'all'}"
            )

    async def _handle_incident(self, incident_id: str, incident: Dict[str,
        Any], auto_mitigate: bool) ->None:
        """
        Handle a new incident by triggering notifications and handlers.
        
        Args:
            incident_id: ID of the incident
            incident: Incident details
            auto_mitigate: Whether to automatically attempt mitigation
        """
        await self.update_incident(incident_id=incident_id, status=
            IncidentStatus.INVESTIGATING)
        category = IncidentCategory(incident['category'])
        severity = IncidentSeverity(incident['severity'])
        await self._send_notifications(incident)
        runbook = await self.get_runbook(category, severity)
        await self._call_handlers(incident_id, incident)
        if auto_mitigate and self.emergency_action_system:
            await self._apply_automated_mitigation(incident_id, incident,
                runbook)

    @async_with_exception_handling
    async def _send_notifications(self, incident: Dict[str, Any]) ->None:
        """
        Send notifications about an incident.
        
        Args:
            incident: Incident details
        """
        if not self.notification_service:
            return
        try:
            severity = incident['severity']
            title = incident['title']
            category = incident['category']
            recipients = []
            if severity == IncidentSeverity.CRITICAL.value:
                recipients = ['engineering_team', 'operations_team',
                    'leadership']
            elif severity == IncidentSeverity.HIGH.value:
                recipients = ['engineering_team', 'operations_team']
            elif severity == IncidentSeverity.MEDIUM.value:
                recipients = ['operations_team']
            else:
                recipients = []
            if recipients:
                notification = {'title': f'Trading Incident: {title}',
                    'message':
                    f"""Category: {category}, Severity: {severity}
{incident['description']}"""
                    , 'recipients': recipients, 'urgency': 'high' if 
                    severity in [IncidentSeverity.CRITICAL.value,
                    IncidentSeverity.HIGH.value] else 'normal',
                    'incident_id': incident['id']}
                await self.notification_service.send_notification(notification)
        except Exception as e:
            self.logger.error(f'Error sending incident notification: {str(e)}')

    @async_with_exception_handling
    async def _call_handlers(self, incident_id: str, incident: Dict[str, Any]
        ) ->None:
        """
        Call registered handlers for an incident.
        
        Args:
            incident_id: ID of the incident
            incident: Incident details
        """
        category = IncidentCategory(incident['category'])
        severity = IncidentSeverity(incident['severity'])
        handlers = []
        key = category, severity
        if key in self._incident_handlers:
            handlers.extend(self._incident_handlers[key])
        for handler in handlers:
            try:
                await asyncio.create_task(handler(incident_id, incident))
            except Exception as e:
                self.logger.error(f'Error in incident handler: {str(e)}')

    @async_with_exception_handling
    async def _apply_automated_mitigation(self, incident_id: str, incident:
        Dict[str, Any], runbook: Optional[Dict[str, Any]]) ->None:
        """
        Apply automated mitigation actions based on runbook.
        
        Args:
            incident_id: ID of the incident
            incident: Incident details
            runbook: Runbook with mitigation steps
        """
        if not self.emergency_action_system or not runbook:
            return
        automated_actions = runbook.get('automated_actions', [])
        if not automated_actions:
            return
        await self.update_incident(incident_id=incident_id, status=
            IncidentStatus.MITIGATING, notes=
            'Applying automated mitigation actions')
        actions_taken = []
        for action in automated_actions:
            action_id = action.get('id')
            action_name = action.get('name')
            action_params = action.get('parameters', {})
            if not action_id:
                continue
            try:
                result = await self.emergency_action_system.execute_action(
                    action_id=action_id, parameters=action_params,
                    incident_id=incident_id)
                if result.get('success', False):
                    actions_taken.append(f'{action_name}: Success')
                else:
                    actions_taken.append(
                        f"{action_name}: Failed - {result.get('message', 'Unknown error')}"
                        )
            except Exception as e:
                self.logger.error(
                    f'Error applying automated mitigation action {action_name}: {str(e)}'
                    )
                actions_taken.append(f'{action_name}: Error - {str(e)}')
        if actions_taken:
            await self.update_incident(incident_id=incident_id,
                actions_taken=actions_taken, notes=
                f'Automated mitigation completed with {len(actions_taken)} actions'
                )

    def _register_default_handlers(self) ->None:
        """Register default handlers for common incident types."""
        self.register_incident_handler(category=IncidentCategory.
            DATA_QUALITY, handler=self._handle_data_quality_incident)
        self.register_incident_handler(category=IncidentCategory.
            CONNECTIVITY, handler=self._handle_connectivity_incident)
        self.register_incident_handler(category=IncidentCategory.EXECUTION,
            handler=self._handle_execution_incident)
        self.register_incident_handler(category=IncidentCategory.SYSTEM,
            severity=IncidentSeverity.CRITICAL, handler=self.
            _handle_critical_system_incident)

    async def _handle_data_quality_incident(self, incident_id: str,
        incident: Dict[str, Any]) ->None:
        """
        Handle data quality incidents.
        
        Args:
            incident_id: ID of the incident
            incident: Incident details
        """
        self.logger.info(
            f'Data quality incident handler triggered for incident {incident_id}'
            )

    async def _handle_connectivity_incident(self, incident_id: str,
        incident: Dict[str, Any]) ->None:
        """
        Handle connectivity incidents.
        
        Args:
            incident_id: ID of the incident
            incident: Incident details
        """
        self.logger.info(
            f'Connectivity incident handler triggered for incident {incident_id}'
            )

    async def _handle_execution_incident(self, incident_id: str, incident:
        Dict[str, Any]) ->None:
        """
        Handle execution incidents.
        
        Args:
            incident_id: ID of the incident
            incident: Incident details
        """
        self.logger.info(
            f'Execution incident handler triggered for incident {incident_id}')

    async def _handle_critical_system_incident(self, incident_id: str,
        incident: Dict[str, Any]) ->None:
        """
        Handle critical system incidents.
        
        Args:
            incident_id: ID of the incident
            incident: Incident details
        """
        self.logger.info(
            f'Critical system incident handler triggered for incident {incident_id}'
            )

    def _load_runbooks(self) ->Dict[str, Dict[str, Any]]:
        """
        Load incident runbooks from configuration.
        
        Returns:
            Dictionary of runbooks organized by category and severity
        """
        config_runbooks = self.config_manager.get('runbooks')
        if config_runbooks:
            return config_runbooks
        default_runbooks = {IncidentCategory.CONNECTIVITY.value: {
            IncidentSeverity.CRITICAL.value: {'title':
            'Critical Connectivity Loss', 'description':
            'Complete loss of connectivity to broker or critical services',
            'steps': ['Verify the status of network connections',
            'Check broker status on independent channels',
            'Implement emergency position reporting using backup channels',
            'Engage disaster recovery procedures if needed',
            'Notify all stakeholders of connectivity loss'],
            'automated_actions': [{'id': 'pause_all_trading', 'name':
            'Pause All Trading', 'parameters': {}}, {'id':
            'enable_circuit_breaker', 'name': 'Enable Circuit Breaker',
            'parameters': {'level': 'system'}}, {'id': 'attempt_reconnect',
            'name': 'Attempt Reconnection', 'parameters': {'max_attempts': 
            5}}], 'contacts': ['engineering_team', 'operations_team',
            'leadership']}, IncidentSeverity.HIGH.value: {'title':
            'Partial Connectivity Loss', 'description':
            'Intermittent connectivity issues with broker or services',
            'steps': ['Assess the impact on active orders and positions',
            'Switch to backup connections if available',
            'Throttle order submission rate',
            'Monitor connectivity recovery'], 'automated_actions': [{'id':
            'throttle_order_submission', 'name':
            'Throttle Order Submission', 'parameters': {'rate': 'low'}}, {
            'id': 'switch_to_backup', 'name': 'Switch to Backup Connection',
            'parameters': {}}], 'contacts': ['engineering_team',
            'operations_team']}}, IncidentCategory.DATA_QUALITY.value: {
            IncidentSeverity.CRITICAL.value: {'title':
            'Critical Data Quality Issue', 'description':
            'Severely incorrect or missing market data affecting multiple instruments'
            , 'steps': ['Identify affected instruments and time ranges',
            'Switch to alternative data sources if available',
            'Pause trading on affected instruments',
            'Verify positions and orders for inconsistencies',
            'Report data issues to provider'], 'automated_actions': [{'id':
            'pause_affected_instruments', 'name':
            'Pause Trading on Affected Instruments', 'parameters': {}}, {
            'id': 'switch_data_source', 'name':
            'Switch to Alternate Data Source', 'parameters': {}}],
            'contacts': ['engineering_team', 'operations_team']}},
            IncidentCategory.EXECUTION.value: {IncidentSeverity.HIGH.value:
            {'title': 'Order Execution Failures', 'description':
            'Multiple order execution failures or unexpected rejections',
            'steps': ['Review rejection reasons and patterns',
            'Verify account status with broker',
            'Check for regulatory trading halts',
            'Adjust order parameters if systematic issue detected',
            'Reconcile positions with broker'], 'automated_actions': [{'id':
            'reconcile_positions', 'name': 'Reconcile Positions',
            'parameters': {}}, {'id': 'adjust_order_parameters', 'name':
            'Adjust Order Parameters', 'parameters': {'slippage': 'higher'}
            }], 'contacts': ['engineering_team', 'operations_team']}},
            IncidentCategory.SYSTEM.value: {IncidentSeverity.CRITICAL.value:
            {'title': 'Critical System Failure', 'description':
            'Major system component failure affecting trading operations',
            'steps': ['Identify failed components and dependencies',
            'Engage disaster recovery procedures',
            'Activate backup systems if available',
            'Suspend new trading activities',
            'Verify all open positions and orders',
            'Prepare for potential manual intervention'],
            'automated_actions': [{'id': 'pause_all_trading', 'name':
            'Pause All Trading', 'parameters': {}}, {'id':
            'enable_circuit_breaker', 'name': 'Enable Circuit Breaker',
            'parameters': {'level': 'system'}}, {'id':
            'activate_backup_systems', 'name': 'Activate Backup Systems',
            'parameters': {}}], 'contacts': ['engineering_team',
            'operations_team', 'leadership']}}, IncidentCategory.RISK.value:
            {IncidentSeverity.CRITICAL.value: {'title':
            'Critical Risk Limit Breach', 'description':
            'Severe breach of risk limits requiring immediate action',
            'steps': ['Identify breached limits and affected positions',
            'Implement emergency position reduction measures',
            'Suspend new position opening in affected instruments',
            'Verify risk calculations and data accuracy',
            'Prepare detailed breach report'], 'automated_actions': [{'id':
            'close_risky_positions', 'name': 'Close High-Risk Positions',
            'parameters': {'risk_level': 'critical'}}, {'id':
            'disable_new_positions', 'name': 'Disable New Position Opening',
            'parameters': {}}], 'contacts': ['risk_team', 'operations_team',
            'leadership']}}}
        return default_runbooks
