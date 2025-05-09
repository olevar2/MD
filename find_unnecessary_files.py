import os
import re

def find_unnecessary_files(directory):
    """Find potentially unnecessary files in a directory tree."""
    print(f"Scanning directory: {directory}")
    
    # Patterns for potentially unnecessary files
    unnecessary_patterns = [
        r'.*\.log$',                  # Log files
        r'.*\.tmp$',                  # Temporary files
        r'.*\.bak$',                  # Backup files
        r'.*\.swp$',                  # Vim swap files
        r'.*~$',                      # Backup files with tilde
        r'.*\.DS_Store$',             # macOS system files
        r'Thumbs\.db$',               # Windows thumbnail cache
        r'.*\.pyc$',                  # Python compiled files
        r'.*\.pyo$',                  # Python optimized files
        r'.*\.pyd$',                  # Python dynamic modules
        r'.*\.o$',                    # Object files
        r'.*\.so$',                   # Shared libraries
        r'.*\.dll$',                  # Windows DLLs
        r'.*\.exe$',                  # Executables
        r'.*\.class$',                # Java compiled files
        r'.*\.cache$',                # Cache files
        r'.*\.orig$',                 # Original files from merge conflicts
        r'.*\.rej$',                  # Rejected patches
    ]
    
    # Compile patterns
    patterns = [re.compile(pattern) for pattern in unnecessary_patterns]
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.vscode'}
    
    # List to store unnecessary files
    unnecessary_files = []
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Check if file matches any pattern
            if any(pattern.match(filename) for pattern in patterns):
                unnecessary_files.append(filepath)
    
    return unnecessary_files

def check_empty_directories(directory):
    """Find empty directories."""
    empty_dirs = []
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.vscode'}
    
    for root, dirs, files in os.walk(directory, topdown=False):  # Bottom-up traversal
        # Skip ignored directories
        if any(ignore_dir in root for ignore_dir in ignore_dirs):
            continue
            
        # Check if directory is empty (no files and no subdirectories)
        if not files and not dirs:
            empty_dirs.append(root)
    
    return empty_dirs

def check_duplicate_tasks(directory):
    """Look for potential duplicate task files."""
    task_files = []
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.vscode'}
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            if 'task' in filename.lower() or 'test' in filename.lower():
                filepath = os.path.join(root, filename)
                task_files.append(filepath)
    
    # Group by filename
    filename_groups = {}
    for filepath in task_files:
        filename = os.path.basename(filepath)
        if filename not in filename_groups:
            filename_groups[filename] = []
        filename_groups[filename].append(filepath)
    
    # Find duplicates
    potential_duplicates = {filename: paths for filename, paths in filename_groups.items() if len(paths) > 1}
    
    return potential_duplicates

if __name__ == "__main__":
    directory = "."  # Current directory
    
    # Find unnecessary files
    print("Checking for unnecessary files...")
    unnecessary_files = find_unnecessary_files(directory)
    
    if unnecessary_files:
        print(f"\nFound {len(unnecessary_files)} potentially unnecessary files:")
        for filepath in unnecessary_files:
            print(f"  - {filepath}")
    else:
        print("No unnecessary files found.")
    
    # Find empty directories
    print("\nChecking for empty directories...")
    empty_dirs = check_empty_directories(directory)
    
    if empty_dirs:
        print(f"\nFound {len(empty_dirs)} empty directories:")
        for dirpath in empty_dirs:
            print(f"  - {dirpath}")
    else:
        print("No empty directories found.")
    
    # Find potential duplicate tasks
    print("\nChecking for potential duplicate task files...")
    potential_duplicates = check_duplicate_tasks(directory)
    
    if potential_duplicates:
        print(f"\nFound {len(potential_duplicates)} potential duplicate task files:")
        for filename, paths in potential_duplicates.items():
            print(f"\n  {filename}:")
            for path in paths:
                print(f"    - {path}")
    else:
        print("No potential duplicate task files found.")
