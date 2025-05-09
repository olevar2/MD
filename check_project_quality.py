import os
import re
import collections

def check_duplicate_files_by_name():
    """Find files with the same name in different directories."""
    print("Checking for files with the same name in different directories...")
    
    # Dictionary to store filenames and their paths
    filename_paths = collections.defaultdict(list)
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.venv', '.vscode'}
    
    # Walk through directory
    for root, dirs, files in os.walk('.'):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            filename_paths[filename].append(filepath)
    
    # Find duplicates
    duplicates = {filename: paths for filename, paths in filename_paths.items() if len(paths) > 1}
    
    if duplicates:
        print(f"\nFound {len(duplicates)} files with the same name in different directories:")
        for filename, paths in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
            if len(paths) > 2:  # Only show files with more than 2 occurrences
                print(f"\n{filename} (found in {len(paths)} locations):")
                for path in paths[:5]:  # Show only first 5 locations
                    print(f"  - {path}")
                if len(paths) > 5:
                    print(f"  - ... and {len(paths) - 5} more locations")
    else:
        print("No files with the same name in different directories found.")

def check_empty_directories():
    """Find empty directories."""
    print("\nChecking for empty directories...")
    
    empty_dirs = []
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.venv', '.vscode'}
    
    for root, dirs, files in os.walk('.', topdown=False):  # Bottom-up traversal
        # Skip ignored directories
        if any(ignore_dir in root for ignore_dir in ignore_dirs):
            continue
            
        # Check if directory is empty (no files and no subdirectories)
        if not files and not dirs:
            empty_dirs.append(root)
    
    if empty_dirs:
        print(f"\nFound {len(empty_dirs)} empty directories:")
        for dirpath in empty_dirs[:20]:  # Show only first 20
            print(f"  - {dirpath}")
        if len(empty_dirs) > 20:
            print(f"  - ... and {len(empty_dirs) - 20} more")
    else:
        print("No empty directories found.")

def check_potential_duplicate_modules():
    """Check for potential duplicate modules."""
    print("\nChecking for potential duplicate modules...")
    
    # Dictionary to store module names and their paths
    module_paths = collections.defaultdict(list)
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.venv', '.vscode'}
    
    # Walk through directory
    for root, dirs, files in os.walk('.'):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            if filename.endswith('.py'):
                # Extract module name (without extension)
                module_name = os.path.splitext(filename)[0]
                
                # Skip common module names like __init__, test_, etc.
                if module_name in ['__init__', '__main__'] or module_name.startswith('test_'):
                    continue
                
                filepath = os.path.join(root, filename)
                module_paths[module_name].append(filepath)
    
    # Find duplicates
    duplicates = {module: paths for module, paths in module_paths.items() if len(paths) > 1}
    
    if duplicates:
        print(f"\nFound {len(duplicates)} potential duplicate modules:")
        for module, paths in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
            print(f"\n{module}.py (found in {len(paths)} locations):")
            for path in paths:
                print(f"  - {path}")
    else:
        print("No potential duplicate modules found.")

def check_large_files():
    """Find unusually large files."""
    print("\nChecking for unusually large files...")
    
    large_files = []
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.venv', '.vscode'}
    
    # Extensions to check
    check_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yml', '.yaml', '.md', '.html', '.css', '.scss']
    
    # Walk through directory
    for root, dirs, files in os.walk('.'):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            # Only check files with specified extensions
            _, ext = os.path.splitext(filename)
            if ext.lower() not in check_extensions:
                continue
                
            filepath = os.path.join(root, filename)
            try:
                size = os.path.getsize(filepath)
                if size > 100 * 1024:  # Larger than 100KB
                    large_files.append((filepath, size))
            except Exception as e:
                print(f"Error checking file size for {filepath}: {e}")
    
    if large_files:
        print(f"\nFound {len(large_files)} unusually large files:")
        for filepath, size in sorted(large_files, key=lambda x: x[1], reverse=True)[:20]:
            print(f"  - {filepath}: {size / 1024:.2f} KB")
    else:
        print("No unusually large files found.")

def check_dependency_issues():
    """Check for dependency issues in package.json and requirements.txt files."""
    print("\nChecking for dependency management issues...")
    
    # Check Python requirements
    req_files = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file in ['requirements.txt', 'pyproject.toml', 'setup.py']:
                req_files.append(os.path.join(root, file))
    
    if len(req_files) > 1:
        print(f"\nFound {len(req_files)} Python dependency files:")
        for file in req_files:
            print(f"  - {file}")
        print("  Consider consolidating requirements or using a monorepo approach.")
    else:
        print("No issues found with Python dependency management.")
    
    # Check JavaScript dependencies
    package_jsons = []
    for root, _, files in os.walk('.'):
        if 'package.json' in files:
            package_jsons.append(os.path.join(root, 'package.json'))
    
    if len(package_jsons) > 1:
        print(f"\nFound {len(package_jsons)} package.json files:")
        for file in package_jsons:
            print(f"  - {file}")
        print("  Consider using a monorepo approach or consolidating dependencies.")
    else:
        print("No issues found with JavaScript dependency management.")

if __name__ == "__main__":
    print("Running project quality checks...\n")
    
    check_duplicate_files_by_name()
    check_empty_directories()
    check_potential_duplicate_modules()
    check_large_files()
    check_dependency_issues()
    
    print("\nProject quality check completed.")
