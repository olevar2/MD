"""
UI Service: Customizable Dashboards Components

This module provides a framework for creating, saving, and loading customizable
dashboards for technical indicators and trading strategies.
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import uuid
logger = logging.getLogger(__name__)


from ui_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class WidgetType(Enum):
    """Types of dashboard widgets"""
    PRICE_CHART = auto()
    INDICATOR_CHART = auto()
    MULTI_INDICATOR = auto()
    CORRELATION_MATRIX = auto()
    HEATMAP = auto()
    STATISTICS = auto()
    PERFORMANCE_CHART = auto()
    SIGNAL_HISTORY = auto()
    ALERT_LIST = auto()
    WATCHLIST = auto()
    NEWS_FEED = auto()
    CALENDAR = auto()
    CUSTOM = auto()

    def __str__(self):
        """String representation of the widget type"""
        return self.name


@dataclass
class Widget:
    """Dashboard widget configuration"""
    id: str
    type: WidgetType
    title: str
    config: Dict[str, Any]
    position: Dict[str, int]

    def to_dict(self) ->Dict[str, Any]:
        """Convert widget to dictionary"""
        return {'id': self.id, 'type': str(self.type), 'title': self.title,
            'config': self.config, 'position': self.position}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'Widget':
        """Create widget from dictionary"""
        widget_type = WidgetType[data['type']] if isinstance(data['type'], str
            ) else data['type']
        return cls(id=data['id'], type=widget_type, title=data['title'],
            config=data['config'], position=data['position'])


@dataclass
class Dashboard:
    """Dashboard configuration"""
    id: str
    name: str
    description: str
    owner: str
    created_at: datetime
    updated_at: datetime
    is_public: bool = False
    tags: List[str] = field(default_factory=list)
    widgets: List[Widget] = field(default_factory=list)
    layout_config: Dict[str, Any] = field(default_factory=dict)

    def add_widget(self, widget: Widget) ->None:
        """Add a widget to the dashboard"""
        self.widgets.append(widget)
        self.updated_at = datetime.now()

    def remove_widget(self, widget_id: str) ->bool:
        """
        Remove a widget from the dashboard
        
        Returns:
            True if widget was removed, False if not found
        """
        initial_count = len(self.widgets)
        self.widgets = [w for w in self.widgets if w.id != widget_id]
        if len(self.widgets) < initial_count:
            self.updated_at = datetime.now()
            return True
        return False

    def update_widget(self, widget_id: str, updated_widget: Widget) ->bool:
        """
        Update a widget in the dashboard
        
        Returns:
            True if widget was updated, False if not found
        """
        for i, widget in enumerate(self.widgets):
            if widget.id == widget_id:
                updated_widget.id = widget_id
                self.widgets[i] = updated_widget
                self.updated_at = datetime.now()
                return True
        return False

    def update_layout(self, layout_config: Dict[str, Any]) ->None:
        """Update layout configuration"""
        self.layout_config.update(layout_config)
        self.updated_at = datetime.now()

    def to_dict(self) ->Dict[str, Any]:
        """Convert dashboard to dictionary"""
        return {'id': self.id, 'name': self.name, 'description': self.
            description, 'owner': self.owner, 'created_at': self.created_at
            .isoformat(), 'updated_at': self.updated_at.isoformat(),
            'is_public': self.is_public, 'tags': self.tags, 'widgets': [w.
            to_dict() for w in self.widgets], 'layout_config': self.
            layout_config}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'Dashboard':
        """Create dashboard from dictionary"""
        widgets = [Widget.from_dict(w) for w in data.get('widgets', [])]
        created_at = datetime.fromisoformat(data['created_at']) if isinstance(
            data['created_at'], str) else data['created_at']
        updated_at = datetime.fromisoformat(data['updated_at']) if isinstance(
            data['updated_at'], str) else data['updated_at']
        return cls(id=data['id'], name=data['name'], description=data[
            'description'], owner=data['owner'], created_at=created_at,
            updated_at=updated_at, is_public=data.get('is_public', False),
            tags=data.get('tags', []), widgets=widgets, layout_config=data.
            get('layout_config', {}))


class DashboardManager:
    """Manager for dashboard configurations"""

    def __init__(self, data_dir: str='./data/dashboards'):
        """
        Initialize the dashboard manager
        
        Args:
            data_dir: Directory to store dashboard data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.dashboards: Dict[str, Dashboard] = {}
        self._load_dashboards()

    def create_dashboard(self, name: str, description: str, owner: str,
        tags: List[str]=None) ->Dashboard:
        """
        Create a new dashboard
        
        Args:
            name: Dashboard name
            description: Dashboard description
            owner: Dashboard owner
            tags: Dashboard tags
            
        Returns:
            The created dashboard
        """
        dashboard_id = str(uuid.uuid4())
        dashboard = Dashboard(id=dashboard_id, name=name, description=
            description, owner=owner, created_at=datetime.now(), updated_at
            =datetime.now(), tags=tags or [])
        self.dashboards[dashboard_id] = dashboard
        self._save_dashboard(dashboard)
        logger.debug(f'Created dashboard {name} (ID: {dashboard_id})')
        return dashboard

    def get_dashboard(self, dashboard_id: str) ->Optional[Dashboard]:
        """
        Get a dashboard by ID
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            The dashboard or None if not found
        """
        return self.dashboards.get(dashboard_id)

    def list_dashboards(self, owner: Optional[str]=None, tags: Optional[
        List[str]]=None, include_public: bool=True) ->List[Dashboard]:
        """
        List dashboards with optional filtering
        
        Args:
            owner: Filter by owner
            tags: Filter by tags
            include_public: Whether to include public dashboards
            
        Returns:
            List of dashboards
        """
        results = list(self.dashboards.values())
        if owner is not None:
            if include_public:
                results = [d for d in results if d.owner == owner or d.
                    is_public]
            else:
                results = [d for d in results if d.owner == owner]
        if tags:
            tags_set = set(tags)
            results = [d for d in results if tags_set.issubset(set(d.tags))]
        results.sort(key=lambda d: d.updated_at, reverse=True)
        return results

    def save_dashboard(self, dashboard: Dashboard) ->bool:
        """
        Save a dashboard
        
        Args:
            dashboard: The dashboard to save
            
        Returns:
            True if saved, False if validation failed
        """
        if not dashboard.name or not dashboard.owner:
            return False
        dashboard.updated_at = datetime.now()
        self.dashboards[dashboard.id] = dashboard
        self._save_dashboard(dashboard)
        return True

    def delete_dashboard(self, dashboard_id: str) ->bool:
        """
        Delete a dashboard
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            True if deleted, False if not found
        """
        if dashboard_id not in self.dashboards:
            return False
        del self.dashboards[dashboard_id]
        path = self._dashboard_path(dashboard_id)
        if os.path.exists(path):
            os.remove(path)
        return True

    def share_dashboard(self, dashboard_id: str, is_public: bool) ->bool:
        """
        Set the public status of a dashboard
        
        Args:
            dashboard_id: Dashboard ID
            is_public: Whether the dashboard should be public
            
        Returns:
            True if updated, False if not found
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return False
        dashboard.is_public = is_public
        dashboard.updated_at = datetime.now()
        self._save_dashboard(dashboard)
        return True

    def clone_dashboard(self, dashboard_id: str, owner: str, new_name:
        Optional[str]=None) ->Optional[Dashboard]:
        """
        Clone a dashboard for another user
        
        Args:
            dashboard_id: ID of the dashboard to clone
            owner: New owner
            new_name: New name (if None, uses original name with " (Copy)" suffix)
            
        Returns:
            The cloned dashboard or None if the source dashboard was not found
        """
        source = self.get_dashboard(dashboard_id)
        if not source:
            return None
        if new_name is None:
            new_name = f'{source.name} (Copy)'
        cloned = self.create_dashboard(name=new_name, description=source.
            description, owner=owner, tags=source.tags.copy())
        for widget in source.widgets:
            new_widget = Widget(id=str(uuid.uuid4()), type=widget.type,
                title=widget.title, config=widget.config.copy(), position=
                widget.position.copy())
            cloned.add_widget(new_widget)
        cloned.layout_config = source.layout_config.copy()
        self._save_dashboard(cloned)
        return cloned

    def export_dashboard(self, dashboard_id: str) ->Optional[Dict[str, Any]]:
        """
        Export a dashboard to a portable format
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard data or None if not found
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return None
        return dashboard.to_dict()

    def import_dashboard(self, data: Dict[str, Any], owner: str) ->Dashboard:
        """
        Import a dashboard from exported data
        
        Args:
            data: Dashboard data
            owner: New owner
            
        Returns:
            The imported dashboard
        """
        dashboard = Dashboard.from_dict(data)
        dashboard.id = str(uuid.uuid4())
        dashboard.owner = owner
        dashboard.created_at = datetime.now()
        dashboard.updated_at = datetime.now()
        for widget in dashboard.widgets:
            widget.id = str(uuid.uuid4())
        self.dashboards[dashboard.id] = dashboard
        self._save_dashboard(dashboard)
        return dashboard

    def _dashboard_path(self, dashboard_id: str) ->str:
        """Get the path to a dashboard file"""
        return os.path.join(self.data_dir, f'dashboard_{dashboard_id}.json')

    def _save_dashboard(self, dashboard: Dashboard) ->None:
        """Save a dashboard to disk"""
        path = self._dashboard_path(dashboard.id)
        with open(path, 'w') as f:
            json.dump(dashboard.to_dict(), f, indent=2)

    @with_exception_handling
    def _load_dashboards(self) ->None:
        """Load dashboards from disk"""
        if not os.path.exists(self.data_dir):
            return
        files = [f for f in os.listdir(self.data_dir) if f.startswith(
            'dashboard_') and f.endswith('.json')]
        for file in files:
            path = os.path.join(self.data_dir, file)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                dashboard = Dashboard.from_dict(data)
                self.dashboards[dashboard.id] = dashboard
            except Exception as e:
                logger.error(f'Error loading dashboard from {path}: {str(e)}')
        logger.info(f'Loaded {len(self.dashboards)} dashboards')


class TemplateManager:
    """Manager for dashboard templates"""

    def __init__(self, template_dir: str='./data/templates'):
        """
        Initialize the template manager
        
        Args:
            template_dir: Directory to store templates
        """
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_templates()

    def add_template(self, name: str, description: str, dashboard_data:
        Dict[str, Any]) ->str:
        """
        Add a dashboard template
        
        Args:
            name: Template name
            description: Template description
            dashboard_data: Dashboard configuration
            
        Returns:
            Template ID
        """
        template_id = str(uuid.uuid4())
        template = {'id': template_id, 'name': name, 'description':
            description, 'created_at': datetime.now().isoformat(),
            'dashboard_data': dashboard_data}
        self.templates[template_id] = template
        self._save_template(template_id, template)
        logger.debug(f'Added template {name} (ID: {template_id})')
        return template_id

    def get_template(self, template_id: str) ->Optional[Dict[str, Any]]:
        """
        Get a template by ID
        
        Args:
            template_id: Template ID
            
        Returns:
            Template data or None if not found
        """
        return self.templates.get(template_id)

    def list_templates(self) ->List[Dict[str, Any]]:
        """
        List all templates
        
        Returns:
            List of templates
        """
        return sorted(self.templates.values(), key=lambda t: t['name'])

    def delete_template(self, template_id: str) ->bool:
        """
        Delete a template
        
        Args:
            template_id: Template ID
            
        Returns:
            True if deleted, False if not found
        """
        if template_id not in self.templates:
            return False
        del self.templates[template_id]
        path = self._template_path(template_id)
        if os.path.exists(path):
            os.remove(path)
        return True

    def _template_path(self, template_id: str) ->str:
        """Get the path to a template file"""
        return os.path.join(self.template_dir, f'template_{template_id}.json')

    def _save_template(self, template_id: str, template: Dict[str, Any]
        ) ->None:
        """Save a template to disk"""
        path = self._template_path(template_id)
        with open(path, 'w') as f:
            json.dump(template, f, indent=2)

    @with_exception_handling
    def _load_templates(self) ->None:
        """Load templates from disk"""
        if not os.path.exists(self.template_dir):
            return
        files = [f for f in os.listdir(self.template_dir) if f.startswith(
            'template_') and f.endswith('.json')]
        for file in files:
            path = os.path.join(self.template_dir, file)
            try:
                with open(path, 'r') as f:
                    template = json.load(f)
                template_id = template['id']
                self.templates[template_id] = template
            except Exception as e:
                logger.error(f'Error loading template from {path}: {str(e)}')
        logger.info(f'Loaded {len(self.templates)} templates')


class StrategyComparisonTool:
    """Tool for comparing trading strategies"""

    def __init__(self):
        """Initialize the strategy comparison tool"""
        pass

    def create_comparison_widget(self, strategies: List[Dict[str, Any]]
        ) ->Widget:
        """
        Create a widget for comparing strategies
        
        Args:
            strategies: List of strategies to compare
            
        Returns:
            Widget configuration
        """
        strategy_names = [s['name'] for s in strategies]
        widget = Widget(id=str(uuid.uuid4()), type=WidgetType.
            PERFORMANCE_CHART, title='Strategy Comparison', config={
            'strategies': strategies, 'metrics': ['win_rate',
            'profit_factor', 'sharpe_ratio', 'max_drawdown', 'expectancy'],
            'chart_type': 'radar'}, position={'x': 0, 'y': 0, 'w': 6, 'h': 4})
        return widget

    def generate_performance_metrics(self, signals: List[Dict[str, Any]]
        ) ->Dict[str, float]:
        """
        Generate performance metrics for a strategy
        
        Args:
            signals: List of trading signals
            
        Returns:
            Dictionary with performance metrics
        """
        win_signals = [s for s in signals if s.get('profit', 0) > 0]
        loss_signals = [s for s in signals if s.get('profit', 0) < 0]
        total_signals = len(signals)
        win_count = len(win_signals)
        if total_signals == 0:
            return {'win_rate': 0, 'profit_factor': 0, 'expectancy': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0}
        win_rate = win_count / total_signals if total_signals > 0 else 0
        total_profit = sum(s.get('profit', 0) for s in win_signals)
        total_loss = abs(sum(s.get('profit', 0) for s in loss_signals)
            ) if loss_signals else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        avg_win = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / len(loss_signals) if loss_signals else 0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        return {'win_rate': win_rate, 'profit_factor': profit_factor,
            'expectancy': expectancy, 'sharpe_ratio': 1.0, 'max_drawdown': 0.1}


def implement_customizable_dashboards():
    """
    Implements customizable analytical dashboards in the UI.
    - Develops save/load configuration mechanism.
    - Adds tools for strategy comparison.
    
    Returns:
        Tuple of (dashboard manager, template manager, strategy comparison tool)
    """
    dashboard_manager = DashboardManager(data_dir='./data/dashboards')
    template_manager = TemplateManager(template_dir='./data/templates')
    comparison_tool = StrategyComparisonTool()
    if not dashboard_manager.dashboards:
        example_dashboard = dashboard_manager.create_dashboard(name=
            'Trading Overview', description=
            'Overview of key markets and indicators', owner='system', tags=
            ['example', 'overview'])
        price_widget = Widget(id=str(uuid.uuid4()), type=WidgetType.
            PRICE_CHART, title='EUR/USD Price Chart', config={'instrument':
            'EUR/USD', 'timeframe': '1H', 'indicators': ['MA20', 'MA50',
            'RSI']}, position={'x': 0, 'y': 0, 'w': 8, 'h': 4})
        example_dashboard.add_widget(price_widget)
        indicator_widget = Widget(id=str(uuid.uuid4()), type=WidgetType.
            INDICATOR_CHART, title='RSI', config={'instrument': 'EUR/USD',
            'timeframe': '1H', 'indicator': 'RSI', 'params': {'period': 14},
            'overbought': 70, 'oversold': 30}, position={'x': 0, 'y': 4,
            'w': 4, 'h': 3})
        example_dashboard.add_widget(indicator_widget)
        correlation_widget = Widget(id=str(uuid.uuid4()), type=WidgetType.
            CORRELATION_MATRIX, title='Currency Correlation', config={
            'instruments': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD',
            'USD/CAD'], 'timeframe': '1D', 'period': 30}, position={'x': 4,
            'y': 4, 'w': 4, 'h': 3})
        example_dashboard.add_widget(correlation_widget)
        dashboard_manager.save_dashboard(example_dashboard)
        template_dashboard = {'name': 'Indicator Analysis', 'description':
            'Template for technical analysis with multiple indicators',
            'widgets': [{'type': 'PRICE_CHART', 'title': 'Price Chart',
            'config': {'indicators': ['MA20', 'MA50', 'Bollinger']},
            'position': {'x': 0, 'y': 0, 'w': 12, 'h': 4}}, {'type':
            'INDICATOR_CHART', 'title': 'RSI', 'config': {'indicator':
            'RSI', 'params': {'period': 14}}, 'position': {'x': 0, 'y': 4,
            'w': 6, 'h': 3}}, {'type': 'INDICATOR_CHART', 'title': 'MACD',
            'config': {'indicator': 'MACD'}, 'position': {'x': 6, 'y': 4,
            'w': 6, 'h': 3}}]}
        template_manager.add_template('Technical Analysis',
            'Multi-indicator technical analysis dashboard', template_dashboard)
    logger.info('Customizable dashboards initialized')
    return dashboard_manager, template_manager, comparison_tool
