"""
Client Code Migrator

This script helps migrate client code to use the standardized API endpoints.
It scans Python files for API calls to legacy endpoints and suggests replacements.
"""
import os
import re
import json
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@dataclass
class EndpointMapping:
    """Mapping between legacy and standardized endpoints"""
    legacy_path: str
    standardized_path: str
    methods: List[str]
    client_method: Optional[str] = None


@dataclass
class CodeLocation:
    """Location of code in a file"""
    file_path: str
    line_number: int
    line: str
    matched_text: str


@dataclass
class MigrationSuggestion:
    """Suggestion for migrating code"""
    location: CodeLocation
    legacy_endpoint: str
    standardized_endpoint: str
    suggested_code: str


@dataclass
class MigrationReport:
    """Report of migration suggestions"""
    suggestions: List[MigrationSuggestion] = field(default_factory=list)

    def add_suggestion(self, suggestion: MigrationSuggestion) ->None:
        """Add a suggestion to the report"""
        self.suggestions.append(suggestion)

    def to_dict(self) ->Dict:
        """Convert the report to a dictionary"""
        return {'summary': {'total_suggestions': len(self.suggestions),
            'files_affected': len(set(s.location.file_path for s in self.
            suggestions))}, 'suggestions': [{'file_path': suggestion.
            location.file_path, 'line_number': suggestion.location.
            line_number, 'line': suggestion.location.line, 'matched_text':
            suggestion.location.matched_text, 'legacy_endpoint': suggestion
            .legacy_endpoint, 'standardized_endpoint': suggestion.
            standardized_endpoint, 'suggested_code': suggestion.
            suggested_code} for suggestion in self.suggestions]}

    def to_json(self, indent: int=2) ->str:
        """Convert the report to JSON"""
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_file(self, file_path: str, indent: int=2) ->None:
        """Save the report to a file"""
        with open(file_path, 'w') as f:
            f.write(self.to_json(indent=indent))

    def print_summary(self) ->None:
        """Print a summary of the report"""
        print(f'Total suggestions: {len(self.suggestions)}')
        print(
            f'Files affected: {len(set(s.location.file_path for s in self.suggestions))}'
            )
        if self.suggestions:
            print('\nSample suggestions:')
            for suggestion in self.suggestions[:5]:
                print(f'\nFile: {suggestion.location.file_path}')
                print(
                    f'Line {suggestion.location.line_number}: {suggestion.location.line.strip()}'
                    )
                print(f'Legacy endpoint: {suggestion.legacy_endpoint}')
                print(
                    f'Standardized endpoint: {suggestion.standardized_endpoint}'
                    )
                print(f'Suggested code: {suggestion.suggested_code}')


def get_endpoint_mappings() ->List[EndpointMapping]:
    """
    Get mappings between legacy and standardized endpoints.

    Returns:
        List of EndpointMapping
    """
    return [EndpointMapping(legacy_path=
        '/api/v1/adaptive-layer/parameters/generate', standardized_path=
        '/api/v1/analysis/adaptations/parameters/generate', methods=['POST'
        ], client_method='generate_adaptive_parameters'), EndpointMapping(
        legacy_path='/api/v1/adaptive-layer/parameters/adjust',
        standardized_path='/api/v1/analysis/adaptations/parameters/adjust',
        methods=['POST'], client_method='adjust_parameters'),
        EndpointMapping(legacy_path=
        '/api/v1/adaptive-layer/strategy/update', standardized_path=
        '/api/v1/analysis/adaptations/strategy/update', methods=['POST'],
        client_method='update_strategy_parameters'), EndpointMapping(
        legacy_path='/api/v1/adaptive-layer/strategy/recommendations',
        standardized_path=
        '/api/v1/analysis/adaptations/strategy/recommendations', methods=[
        'POST'], client_method='generate_strategy_recommendations'),
        EndpointMapping(legacy_path=
        '/api/v1/adaptive-layer/strategy/effectiveness-trend',
        standardized_path=
        '/api/v1/analysis/adaptations/strategy/effectiveness-trend',
        methods=['POST'], client_method=
        'analyze_strategy_effectiveness_trend'), EndpointMapping(
        legacy_path='/api/v1/adaptive-layer/feedback/outcomes',
        standardized_path='/api/v1/analysis/adaptations/feedback/outcomes',
        methods=['POST'], client_method='record_strategy_outcome'),
        EndpointMapping(legacy_path=
        '/api/v1/adaptive-layer/adaptations/history', standardized_path=
        '/api/v1/analysis/adaptations/adaptations/history', methods=['GET'],
        client_method='get_adaptation_history'), EndpointMapping(
        legacy_path='/api/v1/adaptive-layer/parameters/history',
        standardized_path='/api/v1/analysis/adaptations/parameters/history',
        methods=['GET'], client_method='get_parameter_history'),
        EndpointMapping(legacy_path=
        '/api/v1/adaptive-layer/feedback/insights', standardized_path=
        '/api/v1/analysis/adaptations/feedback/insights', methods=['GET'],
        client_method='get_adaptation_insights'), EndpointMapping(
        legacy_path='/api/v1/adaptive-layer/feedback/performance',
        standardized_path=
        '/api/v1/analysis/adaptations/feedback/performance', methods=['GET'
        ], client_method='get_performance_by_regime'), EndpointMapping(
        legacy_path='/health', standardized_path=
        '/api/v1/analysis/health-checks', methods=['GET']), EndpointMapping
        (legacy_path='/health/live', standardized_path=
        '/api/v1/analysis/health-checks/liveness', methods=['GET']),
        EndpointMapping(legacy_path='/health/ready', standardized_path=
        '/api/v1/analysis/health-checks/readiness', methods=['GET'])]


@with_exception_handling
def scan_file_for_endpoints(file_path: str, mappings: List[EndpointMapping]
    ) ->List[MigrationSuggestion]:
    """
    Scan a file for API calls to legacy endpoints.

    Args:
        file_path: Path to the file
        mappings: List of endpoint mappings

    Returns:
        List of MigrationSuggestion
    """
    suggestions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except Exception as e:
            print(f'Skipping file {file_path}: {str(e)}')
            return suggestions
    except Exception as e:
        print(f'Skipping file {file_path}: {str(e)}')
        return suggestions
    for i, line in enumerate(lines):
        for mapping in mappings:
            if mapping.legacy_path in line:
                location = CodeLocation(file_path=file_path, line_number=i +
                    1, line=line, matched_text=mapping.legacy_path)
                suggested_code = line.replace(mapping.legacy_path, mapping.
                    standardized_path)
                if mapping.client_method:
                    if re.search(
                        '(requests|aiohttp|httpx|urllib|fetch|axios)\\.(get|post|put|delete)'
                        , line, re.IGNORECASE):
                        client_suggestion = f"""# Consider using the standardized client instead:
# from analysis_engine.clients.standardized import get_client_factory
# client = get_client_factory().get_adaptive_layer_client()
# result = await client.{mapping.client_method}(...)"""
                        suggested_code += f'\n{client_suggestion}'
                suggestion = MigrationSuggestion(location=location,
                    legacy_endpoint=mapping.legacy_path,
                    standardized_endpoint=mapping.standardized_path,
                    suggested_code=suggested_code)
                suggestions.append(suggestion)
    return suggestions


def scan_directory(directory: str, mappings: List[EndpointMapping],
    extensions: List[str]=['.py', '.js', '.ts']) ->List[MigrationSuggestion]:
    """
    Scan a directory for API calls to legacy endpoints.

    Args:
        directory: Directory to scan
        mappings: List of endpoint mappings
        extensions: File extensions to scan

    Returns:
        List of MigrationSuggestion
    """
    suggestions = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                file_suggestions = scan_file_for_endpoints(file_path, mappings)
                suggestions.extend(file_suggestions)
    return suggestions


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description=
        'Migrate client code to use standardized API endpoints')
    parser.add_argument('directory', help='Directory to scan', nargs='?',
        default='D:/MD/forex_trading_platform/analysis-engine-service')
    parser.add_argument('--output', '-o', help='Output file path', default=
        'client_migration_report.json')
    parser.add_argument('--extensions', '-e', help=
        'File extensions to scan (comma-separated)', default='.py,.js,.ts')
    args = parser.parse_args()
    extensions = args.extensions.split(',')
    mappings = get_endpoint_mappings()
    suggestions = scan_directory(args.directory, mappings, extensions)
    report = MigrationReport()
    for suggestion in suggestions:
        report.add_suggestion(suggestion)
    report.print_summary()
    report.save_to_file(args.output)
    print(f'Report saved to {args.output}')


if __name__ == '__main__':
    main()
