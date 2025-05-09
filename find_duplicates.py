import os
import hashlib
from collections import defaultdict

def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)  # Read in 64kb chunks
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def find_duplicates(directory):
    """Find duplicate files in a directory tree."""
    print(f"Scanning directory: {directory}")
    
    # Dictionary to store file hashes and their paths
    hashes = defaultdict(list)
    
    # Extensions to ignore
    ignore_extensions = {'.pyc', '.pyo', '.pyd', '.git', '.DS_Store', '.idea', '__pycache__'}
    
    # Directories to ignore
    ignore_dirs = {'node_modules', '__pycache__', '.git', '.idea', 'venv', 'env', '.vscode'}
    
    # Count files processed
    file_count = 0
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in files:
            # Skip files with ignored extensions
            if any(filename.endswith(ext) for ext in ignore_extensions):
                continue
                
            filepath = os.path.join(root, filename)
            try:
                file_hash = get_file_hash(filepath)
                hashes[file_hash].append(filepath)
                file_count += 1
                if file_count % 100 == 0:
                    print(f"Processed {file_count} files...")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    print(f"Total files processed: {file_count}")
    
    # Find duplicates
    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    
    return duplicates

def print_duplicates(duplicates):
    """Print duplicate files in a readable format."""
    if not duplicates:
        print("No duplicate files found.")
        return
        
    print(f"Found {len(duplicates)} sets of duplicate files:")
    
    total_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
    print(f"Total duplicate files: {total_duplicates}")
    
    for i, (file_hash, paths) in enumerate(duplicates.items(), 1):
        print(f"\nDuplicate set #{i} (Hash: {file_hash}):")
        for path in paths:
            print(f"  - {path}")

if __name__ == "__main__":
    directory = "."  # Current directory
    duplicates = find_duplicates(directory)
    print_duplicates(duplicates)
