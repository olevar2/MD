"""
Rename files module.

This module provides functionality for...
"""

import os
import re
import shutil
import subprocess

def kebab_to_snake(name):
    """Convert kebab-case to snake_case."""
    return name.replace('-', '_')

def pascal_to_snake(name):
    """Convert PascalCase to snake_case."""
    # Add underscore before uppercase letters and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()

def camel_to_kebab(name):
    """Convert camelCase to kebab-case."""
    # Add hyphen before uppercase letters and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1)
    return s2.lower()

def rename_directory(old_path, new_path):
    """Rename a directory and update references in the codebase."""
    print(f"Renaming directory: {old_path} -> {new_path}")

    # Check if the new path already exists
    if os.path.exists(new_path):
        print(f"  Warning: {new_path} already exists, skipping")
        return False

    # Rename the directory
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.move(old_path, new_path)

    # Update references in the codebase
    old_name = os.path.basename(old_path)
    new_name = os.path.basename(new_path)

    # Use git grep to find references to the old name
    try:
        result = subprocess.run(
            ['git', 'grep', '-l', old_name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            files_to_update = result.stdout.strip().split('\n')

            for file_path in files_to_update:
                if not file_path or not os.path.exists(file_path):
                    continue

                # Read the file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Replace references to the old name
                new_content = content.replace(old_name, new_name)

                # Write the updated content
                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"  Updated references in {file_path}")

    except subprocess.CalledProcessError:
        print("  Warning: Failed to update references")

    return True

def rename_file(old_path, new_path):
    """Rename a file and update references in the codebase."""
    print(f"Renaming file: {old_path} -> {new_path}")

    # Check if the new path already exists
    if os.path.exists(new_path):
        print(f"  Warning: {new_path} already exists, skipping")
        return False

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Rename the file
    shutil.move(old_path, new_path)

    # Update references in the codebase
    old_name = os.path.basename(old_path)
    new_name = os.path.basename(new_path)

    # Use git grep to find references to the old name
    try:
        result = subprocess.run(
            ['git', 'grep', '-l', old_name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            files_to_update = result.stdout.strip().split('\n')

            for file_path in files_to_update:
                if not file_path or not os.path.exists(file_path):
                    continue

                # Read the file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Replace references to the old name
                new_content = content.replace(old_name, new_name)

                # Write the updated content
                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"  Updated references in {file_path}")

    except subprocess.CalledProcessError:
        print("  Warning: Failed to update references")

    return True

def rename_directories():
    """Rename directories to follow naming conventions."""
    # Directories to rename (old_path -> new_path)
    directories_to_rename = [
        # Top-level service directories should use kebab-case
        # Module directories should use snake_case
        ('api-gateway', 'api-gateway'),  # Already kebab-case, no change needed
        ('common-js-lib', 'common-js-lib'),  # Already kebab-case, no change needed
        ('common-lib', 'common-lib'),  # Already kebab-case, no change needed
        ('core-foundations', 'core-foundations'),  # Already kebab-case, no change needed
        ('reference-servers', 'reference-servers'),  # Already kebab-case, no change needed
        ('strategy-execution-engine', 'strategy-execution-engine'),  # Already kebab-case, no change needed

        # Module directories with kebab-case should be converted to snake_case
        ('ui-service/src/components/asset-detail', 'ui-service/src/components/asset_detail'),
        ('ui-service/src/components/feedback-loop', 'ui-service/src/components/feedback_loop'),
        ('ui-service/src/components/ml-workbench', 'ui-service/src/components/ml_workbench'),
        ('ui-service/src/components/ui-library', 'ui-service/src/components/ui_library'),
        ('ui-service/src/pages/chat-demo', 'ui-service/src/pages/chat_demo'),
    ]

    for old_path, new_path in directories_to_rename:
        if old_path != new_path:
            rename_directory(old_path, new_path)

def rename_files():
    """Rename files to follow naming conventions."""
    # Python files with PascalCase should be converted to snake_case
    python_files_to_rename = [
        ('analysis-engine-service/analysis_engine/interfaces/IAdvancedIndicator.py',
         'analysis-engine-service/analysis_engine/interfaces/i_advanced_indicator.py'),
        ('analysis-engine-service/analysis_engine/interfaces/IPatternRecognizer.py',
         'analysis-engine-service/analysis_engine/interfaces/i_pattern_recognizer.py'),
        ('chat_interface_template/ChatBackendService.py',
         'chat_interface_template/chat_backend_service.py'),
        ('feature-store-service/feature_store_service/interfaces/IIndicator.py',
         'feature-store-service/feature_store_service/interfaces/i_indicator.py'),
    ]

    for old_path, new_path in python_files_to_rename:
        if os.path.exists(old_path):
            rename_file(old_path, new_path)
        else:
            print(f"Warning: {old_path} does not exist, skipping")

    # JavaScript/TypeScript files with camelCase or PascalCase should be converted to kebab-case
    js_files_to_rename = [
        # Analysis Engine Service UI files
        ('analysis-engine-service/ui/api/apiClient.js',
         'analysis-engine-service/ui/api/api-client.js'),
        ('analysis-engine-service/ui/components/AnalysisDashboard.js',
         'analysis-engine-service/ui/components/analysis-dashboard.js'),
        ('analysis-engine-service/ui/components/ConfluenceDetectionWidget.js',
         'analysis-engine-service/ui/components/confluence-detection-widget.js'),
        ('analysis-engine-service/ui/components/DivergenceAnalysisWidget.js',
         'analysis-engine-service/ui/components/divergence-analysis-widget.js'),

        # Common JS Lib files
        ('common-js-lib/apiClient.js', 'common-js-lib/api-client.js'),
        ('common-js-lib/errorHandler.js', 'common-js-lib/error-handler.js'),
        ('common-js-lib/templates/ServiceClientTemplate.ts',
         'common-js-lib/templates/service-client-template.ts'),

        # UI Service files (just a subset for now)
        ('ui-service/src/api/apiClient.ts', 'ui-service/src/api/api-client.ts'),
        ('ui-service/src/components/ABTestMonitor.tsx', 'ui-service/src/components/ab-test-monitor.tsx'),
        ('ui-service/src/components/AssetDetailView.jsx', 'ui-service/src/components/asset-detail-view.jsx'),
        ('ui-service/src/components/CausalDashboard.tsx', 'ui-service/src/components/causal-dashboard.tsx'),
        ('ui-service/src/components/FeedbackDashboard.tsx', 'ui-service/src/components/feedback-dashboard.tsx'),
        ('ui-service/src/components/ModelExplainabilityVisualization.tsx',
         'ui-service/src/components/model-explainability-visualization.tsx'),
        ('ui-service/src/components/MultiAssetPortfolioDashboard.jsx',
         'ui-service/src/components/multi-asset-portfolio-dashboard.jsx'),
        ('ui-service/src/components/NetworkGraph.tsx', 'ui-service/src/components/network-graph.tsx'),
        ('ui-service/src/components/ParameterTuningInterface.tsx',
         'ui-service/src/components/parameter-tuning-interface.tsx'),
        ('ui-service/src/components/PortfolioBreakdown.tsx', 'ui-service/src/components/portfolio-breakdown.tsx'),
        ('ui-service/src/components/PositionMonitoringDashboard.jsx',
         'ui-service/src/components/position-monitoring-dashboard.jsx'),
        ('ui-service/src/components/RegimeAwareDashboard.tsx', 'ui-service/src/components/regime-aware-dashboard.tsx'),
        ('ui-service/src/components/RLEnvironmentConfig.tsx', 'ui-service/src/components/rl-environment-config.tsx'),
        ('ui-service/src/components/RLTrainingDashboard.tsx', 'ui-service/src/components/rl-training-dashboard.tsx'),
        ('ui-service/src/components/SignalVisualization.jsx', 'ui-service/src/components/signal-visualization.jsx'),
        ('ui-service/src/components/SignalVisualizer.tsx', 'ui-service/src/components/signal-visualizer.tsx'),
    ]

    for old_path, new_path in js_files_to_rename:
        if os.path.exists(old_path):
            rename_file(old_path, new_path)
        else:
            print(f"Warning: {old_path} does not exist, skipping")

if __name__ == "__main__":
    # Rename directories first
    rename_directories()

    # Then rename files
    rename_files()
