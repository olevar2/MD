"""
Enhanced Deprecation Monitoring Module

This module provides advanced functionality to monitor usage of deprecated modules
and track migration progress with detailed analytics and reporting capabilities.
"""
import logging
import inspect
import os
import time
import json
import datetime
from typing import Dict, Set, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import threading
import atexit
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

@dataclass
class DeprecationUsage:
    """Record of a deprecated module usage with enhanced metadata."""
    module_name: str
    caller_file: str
    caller_line: int
    caller_function: str
    timestamp: float
    count: int = 1
    last_seen: str = field(default_factory=lambda : datetime.datetime.now()
        .isoformat())
    call_stack: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) ->Dict:
        """Convert to dictionary for serialization."""
        return {'module_name': self.module_name, 'caller_file': self.
            caller_file, 'caller_line': self.caller_line, 'caller_function':
            self.caller_function, 'timestamp': self.timestamp, 'count':
            self.count, 'last_seen': self.last_seen, 'call_stack': self.
            call_stack, 'context': self.context}

    def update(self) ->None:
        """Update usage record when called again."""
        self.count += 1
        self.timestamp = time.time()
        self.last_seen = datetime.datetime.now().isoformat()


@dataclass
class ModuleUsageStats:
    """Statistics for a deprecated module."""
    name: str
    total_usages: int = 0
    unique_locations: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    usage_trend: Dict[str, int] = field(default_factory=dict)
    top_callers: List[Dict[str, Any]] = field(default_factory=list)
    replacement: str = ''
    migration_guide: str = ''
    migration_tool: str = ''
    description: str = ''
    impact_level: str = 'medium'

    def to_dict(self) ->Dict:
        """Convert to dictionary for serialization."""
        return {'name': self.name, 'total_usages': self.total_usages,
            'unique_locations': self.unique_locations, 'first_seen': self.
            first_seen, 'last_seen': self.last_seen, 'usage_trend': self.
            usage_trend, 'top_callers': self.top_callers, 'replacement':
            self.replacement, 'migration_guide': self.migration_guide,
            'migration_tool': self.migration_tool, 'description': self.
            description, 'impact_level': self.impact_level}


class EnhancedDeprecationMonitor:
    """
    Enhanced monitor for deprecated modules with advanced analytics.
    
    This class provides comprehensive functionality to:
    1. Record and analyze usage of deprecated modules
    2. Generate detailed reports and visualizations
    3. Track migration progress over time
    4. Provide insights for migration planning
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EnhancedDeprecationMonitor, cls).__new__(
                    cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the enhanced deprecation monitor."""
        if self._initialized:
            return
        self._usages: Dict[str, DeprecationUsage] = {}
        self._module_stats: Dict[str, ModuleUsageStats] = {}
        self._lock = threading.Lock()
        self._config = self._load_config()
        report_dir = self._config_manager.get('monitoring', {}).get('report_directory',
            'logs/deprecation')
        self._report_dir = os.path.join(os.path.dirname(os.path.dirname(os.
            path.dirname(os.path.abspath(__file__)))), report_dir)
        os.makedirs(self._report_dir, exist_ok=True)
        self._report_path = os.path.join(self._report_dir,
            'deprecation_report.json')
        self._history_path = os.path.join(self._report_dir,
            'deprecation_history.json')
        self._load_data()
        self._initialize_module_stats()
        atexit.register(self.save_data)
        self._initialized = True
        logger.debug('Enhanced deprecation monitor initialized')

    @with_exception_handling
    def _load_config(self) ->Dict[str, Any]:
        """Load deprecation configuration."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.
            dirname(os.path.abspath(__file__)))), 'deprecation_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f'Failed to load deprecation config: {e}')
        return {'deprecation_date': '2023-12-31', 'modules': [],
            'monitoring': {'report_directory': 'logs/deprecation',
            'threshold_levels': {'high_usage': 100, 'medium_usage': 20,
            'low_usage': 1}}}

    def _initialize_module_stats(self) ->None:
        """Initialize module statistics from configuration."""
        for module in self._config_manager.get('modules', []):
            name = module.get('name')
            if name and name not in self._module_stats:
                self._module_stats[name] = ModuleUsageStats(name=name,
                    replacement=module.get('replacement', ''),
                    migration_guide=module.get('migration_guide', ''),
                    migration_tool=module.get('migration_tool', ''),
                    description=module.get('description', ''), impact_level
                    =module.get('impact_level', 'medium'))

    @with_exception_handling
    def _load_data(self) ->None:
        """Load existing usage data if available."""
        if os.path.exists(self._report_path):
            try:
                with open(self._report_path, 'r') as f:
                    report_data = json.load(f)
                    for module_name, module_data in report_data.get('modules',
                        {}).items():
                        self._module_stats[module_name] = ModuleUsageStats(name
                            =module_name, total_usages=module_data.get(
                            'total_usages', 0), unique_locations=
                            module_data.get('unique_locations', 0),
                            first_seen=module_data.get('first_seen'),
                            last_seen=module_data.get('last_seen'),
                            usage_trend=module_data.get('usage_trend', {}),
                            top_callers=module_data.get('top_callers', []),
                            replacement=module_data.get('replacement', ''),
                            migration_guide=module_data.get(
                            'migration_guide', ''), migration_tool=
                            module_data.get('migration_tool', ''),
                            description=module_data.get('description', ''),
                            impact_level=module_data.get('impact_level',
                            'medium'))
                    for module_data in report_data.get('modules', {}).values():
                        for usage in module_data.get('usages', []):
                            key = (
                                f"{usage.get('module_name')}:{usage.get('caller_file')}:{usage.get('caller_line')}:{usage.get('caller_function')}"
                                )
                            self._usages[key] = DeprecationUsage(module_name
                                =usage.get('module_name', ''), caller_file=
                                usage.get('caller_file', ''), caller_line=
                                usage.get('caller_line', 0),
                                caller_function=usage.get('caller_function',
                                ''), timestamp=usage.get('timestamp', 0),
                                count=usage.get('count', 1), last_seen=
                                usage.get('last_seen', ''), call_stack=
                                usage.get('call_stack', []), context=usage.
                                get('context', {}))
                logger.debug(
                    f'Loaded {len(self._usages)} deprecation usages from {self._report_path}'
                    )
            except Exception as e:
                logger.error(f'Failed to load deprecation report: {e}')
        if os.path.exists(self._history_path):
            try:
                with open(self._history_path, 'r') as f:
                    history_data = json.load(f)
                    for module_name, module_data in history_data.get('modules',
                        {}).items():
                        if module_name in self._module_stats:
                            self._module_stats[module_name
                                ].usage_trend = module_data.get('usage_trend',
                                {})
                logger.debug(
                    f'Loaded deprecation history from {self._history_path}')
            except Exception as e:
                logger.error(f'Failed to load deprecation history: {e}')

    @with_exception_handling
    def record_usage(self, module_name: str, context: Dict[str, Any]=None
        ) ->None:
        """
        Record usage of a deprecated module with enhanced context.
        
        Args:
            module_name: Name of the deprecated module
            context: Additional context information
        """
        frame = inspect.currentframe().f_back.f_back
        if not frame:
            return
        caller_file = frame.f_code.co_filename
        caller_line = frame.f_lineno
        caller_function = frame.f_code.co_name
        try:
            caller_file = os.path.relpath(caller_file)
        except ValueError:
            pass
        call_stack = []
        current_frame = frame
        for _ in range(5):
            if current_frame:
                frame_info = {'file': current_frame.f_code.co_filename,
                    'line': current_frame.f_lineno, 'function':
                    current_frame.f_code.co_name}
                try:
                    frame_info['file'] = os.path.relpath(frame_info['file'])
                except ValueError:
                    pass
                call_stack.append(
                    f"{frame_info['file']}:{frame_info['line']} in {frame_info['function']}"
                    )
                current_frame = current_frame.f_back
            else:
                break
        key = f'{module_name}:{caller_file}:{caller_line}:{caller_function}'
        with self._lock:
            if key in self._usages:
                self._usages[key].update()
            else:
                self._usages[key] = DeprecationUsage(module_name=
                    module_name, caller_file=caller_file, caller_line=
                    caller_line, caller_function=caller_function, timestamp
                    =time.time(), call_stack=call_stack, context=context or {})
            if module_name not in self._module_stats:
                module_config = next((m for m in self._config.get('modules',
                    []) if m.get('name') == module_name), {})
                self._module_stats[module_name] = ModuleUsageStats(name=
                    module_name, replacement=module_config.get(
                    'replacement', ''), migration_guide=module_config.get(
                    'migration_guide', ''), migration_tool=module_config.
                    get('migration_tool', ''), description=module_config.
                    get('description', ''), impact_level=module_config.get(
                    'impact_level', 'medium'))
            self._update_module_stats(module_name)
        logger.debug(
            f'Recorded usage of deprecated module {module_name} from {caller_file}:{caller_line}'
            )

    def _update_module_stats(self, module_name: str) ->None:
        """
        Update statistics for a module.
        
        Args:
            module_name: Name of the module to update
        """
        module_usages = [u for u in self._usages.values() if u.module_name ==
            module_name]
        self._module_stats[module_name].total_usages = sum(u.count for u in
            module_usages)
        self._module_stats[module_name].unique_locations = len(module_usages)
        if module_usages:
            timestamps = [u.timestamp for u in module_usages]
            first_seen = datetime.datetime.fromtimestamp(min(timestamps)
                ).isoformat()
            last_seen = datetime.datetime.fromtimestamp(max(timestamps)
                ).isoformat()
            self._module_stats[module_name].first_seen = first_seen
            self._module_stats[module_name].last_seen = last_seen
        today = datetime.date.today().isoformat()
        if today not in self._module_stats[module_name].usage_trend:
            self._module_stats[module_name].usage_trend[today] = 0
        self._module_stats[module_name].usage_trend[today] += 1
        top_callers = sorted(module_usages, key=lambda u: u.count, reverse=True
            )[:10]
        self._module_stats[module_name].top_callers = [{'file': u.
            caller_file, 'line': u.caller_line, 'function': u.
            caller_function, 'count': u.count, 'last_seen': u.last_seen} for
            u in top_callers]

    @with_resilience('get_usage_report')
    def get_usage_report(self) ->Dict[str, Any]:
        """
        Generate a comprehensive report on usage of deprecated modules.
        
        Returns:
            Dict: Detailed report data
        """
        with self._lock:
            deprecation_date = datetime.datetime.strptime(self._config.get(
                'deprecation_date', '2023-12-31'), '%Y-%m-%d').date()
            days_until_removal = (deprecation_date - datetime.date.today()
                ).days
            report = {'generated_at': datetime.datetime.now().isoformat(),
                'deprecation_date': self._config.get('deprecation_date',
                '2023-12-31'), 'days_until_removal': days_until_removal,
                'total_modules': len(self._module_stats),
                'total_unique_usages': len(self._usages),
                'total_usage_count': sum(u.count for u in self._usages.
                values()), 'modules': {}}
            for module_name, module_stats in self._module_stats.items():
                module_usages = [u for u in self._usages.values() if u.
                    module_name == module_name]
                report['modules'][module_name] = {'total_usages':
                    module_stats.total_usages, 'unique_locations':
                    module_stats.unique_locations, 'first_seen':
                    module_stats.first_seen, 'last_seen': module_stats.
                    last_seen, 'usage_trend': module_stats.usage_trend,
                    'top_callers': module_stats.top_callers, 'replacement':
                    module_stats.replacement, 'migration_guide':
                    module_stats.migration_guide, 'migration_tool':
                    module_stats.migration_tool, 'description':
                    module_stats.description, 'impact_level': module_stats.
                    impact_level, 'usages': [u.to_dict() for u in
                    module_usages]}
            return report

    @with_exception_handling
    def save_data(self) ->None:
        """Save usage data to files."""
        try:
            report = self.get_usage_report()
            with open(self._report_path, 'w') as f:
                json.dump(report, f, indent=2)
            history = {'generated_at': datetime.datetime.now().isoformat(),
                'modules': {}}
            for module_name, module_stats in self._module_stats.items():
                history['modules'][module_name] = {'usage_trend':
                    module_stats.usage_trend}
            with open(self._history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.debug(f'Saved deprecation data to {self._report_dir}')
        except Exception as e:
            logger.error(f'Failed to save deprecation data: {e}')

    def generate_charts(self, output_dir: str=None) ->Dict[str, str]:
        """
        Generate charts visualizing deprecation usage.
        
        Args:
            output_dir: Directory to save charts (defaults to report directory)
            
        Returns:
            Dict mapping chart names to file paths
        """
        if output_dir is None:
            output_dir = self._report_dir
        os.makedirs(output_dir, exist_ok=True)
        charts = {}
        charts['usage_by_module'] = self._generate_usage_by_module_chart(
            output_dir)
        charts['usage_trend'] = self._generate_trend_chart(output_dir)
        charts['migration_progress'] = self._generate_migration_progress_chart(
            output_dir)
        return charts

    def _generate_usage_by_module_chart(self, output_dir: str) ->str:
        """
        Generate a chart showing usage by module.
        
        Args:
            output_dir: Directory to save the chart
            
        Returns:
            Path to the generated chart
        """
        modules = []
        usages = []
        locations = []
        for module_name, module_stats in self._module_stats.items():
            modules.append(module_name)
            usages.append(module_stats.total_usages)
            locations.append(module_stats.unique_locations)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(modules))
        width = 0.35
        ax.bar(x - width / 2, usages, width, label='Total Usages')
        ax.bar(x + width / 2, locations, width, label='Unique Locations')
        ax.set_title('Deprecated Module Usage')
        ax.set_xlabel('Module')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(modules, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'usage_by_module.png')
        plt.savefig(chart_path)
        plt.close()
        return chart_path

    def _generate_trend_chart(self, output_dir: str) ->str:
        """
        Generate a chart showing usage trends over time.
        
        Args:
            output_dir: Directory to save the chart
            
        Returns:
            Path to the generated chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for module_name, module_stats in self._module_stats.items():
            dates = sorted(module_stats.usage_trend.keys())
            counts = [module_stats.usage_trend[date] for date in dates]
            ax.plot(dates, counts, marker='o', label=module_name)
        ax.set_title('Deprecated Module Usage Trend')
        ax.set_xlabel('Date')
        ax.set_ylabel('Usage Count')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'usage_trend.png')
        plt.savefig(chart_path)
        plt.close()
        return chart_path

    def _generate_migration_progress_chart(self, output_dir: str) ->str:
        """
        Generate a chart showing migration progress.
        
        Args:
            output_dir: Directory to save the chart
            
        Returns:
            Path to the generated chart
        """
        modules = []
        progress = []
        initial_counts = {}
        for module_name, module_stats in self._module_stats.items():
            trend = module_stats.usage_trend
            if trend:
                initial_counts[module_name] = max(trend.values()) * 10
            else:
                initial_counts[module_name] = module_stats.total_usages
        for module_name, module_stats in self._module_stats.items():
            modules.append(module_name)
            initial = initial_counts.get(module_name, 100)
            current = module_stats.unique_locations
            if initial > 0:
                progress_pct = max(0, min(100, 100 - current / initial * 100))
            else:
                progress_pct = 100
            progress.append(progress_pct)
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(modules))
        ax.barh(y_pos, progress, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(modules)
        ax.invert_yaxis()
        ax.set_xlabel('Migration Progress (%)')
        ax.set_title('Deprecated Module Migration Progress')
        for i, v in enumerate(progress):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'migration_progress.png')
        plt.savefig(chart_path)
        plt.close()
        return chart_path

    @with_analysis_resilience('analyze_imports')
    @with_exception_handling
    def analyze_imports(self) ->Dict[str, List[Tuple[str, int, str]]]:
        """
        Analyze import statements for deprecated modules in the codebase.
        
        Returns:
            Dict mapping module names to lists of (file_path, line_number, line_content)
        """
        results = {}
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.
            abspath(__file__))))
        for module in self._config_manager.get('modules', []):
            module_name = module.get('name')
            if not module_name:
                continue
            patterns = [re.compile(
                f'from\\s+{re.escape(module_name)}\\s+import'), re.compile(
                f'import\\s+{re.escape(module_name)}')]
            module_results = []
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if not filename.endswith('.py'):
                        continue
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for i, line in enumerate(lines):
                                for pattern in patterns:
                                    if pattern.search(line):
                                        rel_path = os.path.relpath(file_path,
                                            root_dir)
                                        module_results.append((rel_path, i + 1,
                                            line.strip()))
                                        break
                    except UnicodeDecodeError:
                        continue
            results[module_name] = module_results
        return results

    @with_resilience('get_migration_status')
    def get_migration_status(self) ->Dict[str, Any]:
        """
        Get the current migration status.
        
        Returns:
            Dict with migration status information
        """
        deprecation_date = datetime.datetime.strptime(self._config.get(
            'deprecation_date', '2023-12-31'), '%Y-%m-%d').date()
        days_until_removal = (deprecation_date - datetime.date.today()).days
        total_initial = 0
        total_current = 0
        initial_counts = {}
        for module_name, module_stats in self._module_stats.items():
            trend = module_stats.usage_trend
            if trend:
                initial_counts[module_name] = max(trend.values()) * 10
            else:
                initial_counts[module_name] = module_stats.total_usages
        module_progress = {}
        for module_name, module_stats in self._module_stats.items():
            initial = initial_counts.get(module_name, 100)
            current = module_stats.unique_locations
            total_initial += initial
            total_current += current
            if initial > 0:
                progress_pct = max(0, min(100, 100 - current / initial * 100))
            else:
                progress_pct = 100
            module_progress[module_name] = {'initial_estimate': initial,
                'current_usages': current, 'progress_percentage': progress_pct}
        if total_initial > 0:
            overall_progress = max(0, min(100, 100 - total_current /
                total_initial * 100))
        else:
            overall_progress = 100
        if days_until_removal <= 0:
            status = 'PAST DUE'
        elif overall_progress >= 90:
            status = 'ON TRACK'
        elif overall_progress >= 50:
            status = 'IN PROGRESS'
        else:
            status = 'AT RISK'
        return {'deprecation_date': self._config.get('deprecation_date',
            '2023-12-31'), 'days_until_removal': days_until_removal,
            'overall_progress': overall_progress, 'status': status,
            'module_progress': module_progress, 'total_modules': len(self.
            _module_stats), 'total_unique_usages': len(self._usages),
            'total_usage_count': sum(u.count for u in self._usages.values())}


_monitor = EnhancedDeprecationMonitor()


def record_usage(module_name: str, context: Dict[str, Any]=None) ->None:
    """
    Record usage of a deprecated module with enhanced context.
    
    Args:
        module_name: Name of the deprecated module
        context: Additional context information
    """
    _monitor.record_usage(module_name, context)


def get_usage_report() ->Dict[str, Any]:
    """
    Generate a comprehensive report on usage of deprecated modules.
    
    Returns:
        Dict: Detailed report data
    """
    return _monitor.get_usage_report()


def save_data() ->None:
    """Save usage data to files."""
    _monitor.save_data()


def generate_charts(output_dir: str=None) ->Dict[str, str]:
    """
    Generate charts visualizing deprecation usage.
    
    Args:
        output_dir: Directory to save charts
        
    Returns:
        Dict mapping chart names to file paths
    """
    return _monitor.generate_charts(output_dir)


def analyze_imports() ->Dict[str, List[Tuple[str, int, str]]]:
    """
    Analyze import statements for deprecated modules in the codebase.
    
    Returns:
        Dict mapping module names to lists of (file_path, line_number, line_content)
    """
    return _monitor.analyze_imports()


def get_migration_status() ->Dict[str, Any]:
    """
    Get the current migration status.
    
    Returns:
        Dict with migration status information
    """
    return _monitor.get_migration_status()
