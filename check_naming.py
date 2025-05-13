"""
Check naming module.

This module provides functionality for...
"""

import os
import re

def check_naming(root_dir):
    """
    Check naming.
    
    Args:
        root_dir: Description of root_dir
    
    """

    results = {
        'non_compliant_dirs': [],
        'non_compliant_files': []
    }

    # Directories to skip
    skip_dirs = ['.git', '.venv', 'node_modules', '__pycache__']

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip directories
        should_skip = False
        for skip_dir in skip_dirs:
            if skip_dir in dirpath:
                should_skip = True
                break

        if should_skip:
            continue

        # Check directory names
        for dirname in dirnames[:]:  # Create a copy to modify during iteration
            # Skip directories
            if dirname in skip_dirs:
                dirnames.remove(dirname)
                continue

            # Service directories should use kebab-case
            if dirname.endswith('-service'):
                if not re.match(r'^[a-z0-9-]+$', dirname):
                    results['non_compliant_dirs'].append(os.path.join(dirpath, dirname))
            # Module directories should use snake_case
            elif '-' in dirname:
                results['non_compliant_dirs'].append(os.path.join(dirpath, dirname))

        # Check file names
        for filename in filenames:
            # Python files should use snake_case
            if filename.endswith('.py'):
                if not re.match(r'^[a-z0-9_]+\.py$', filename):
                    results['non_compliant_files'].append(os.path.join(dirpath, filename))
            # JavaScript/TypeScript files should use kebab-case
            elif filename.endswith(('.js', '.ts', '.jsx', '.tsx')):
                if not re.match(r'^[a-z0-9-]+\.[jt]sx?$', filename):
                    results['non_compliant_files'].append(os.path.join(dirpath, filename))

    return results

if __name__ == "__main__":
    results = check_naming('.')

    print('Non-compliant directories (should use snake_case for module dirs, kebab-case for service dirs):')
    for d in results['non_compliant_dirs']:
        print(f'- {d}')
    print(f'Total: {len(results["non_compliant_dirs"])}')

    print('\nNon-compliant files (Python files should use snake_case, JS/TS files should use kebab-case):')
    for f in results['non_compliant_files']:
        print(f'- {f}')
    print(f'Total: {len(results["non_compliant_files"])}')
