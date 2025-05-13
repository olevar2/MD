"""
Script to identify duplicate implementations across services.

This script scans the codebase for duplicate implementations of components
that could be consolidated into common-lib.
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

# Service directories to scan
SERVICE_DIRS = [
    "analysis-engine-service",
    "data-pipeline-service",
    "feature-store-service",
    "ml-integration-service",
    "ml-workbench-service",
    "monitoring-alerting-service",
    "portfolio-management-service",
    "strategy-execution-engine",
    "trading-gateway-service",
    "ui-service",
    "common-lib"
]

# Directories to skip
SKIP_DIRS = ['.git', '.venv', 'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist']

# Component patterns to look for
COMPONENT_PATTERNS = {
    "memory_optimized_dataframe": [
        r"class\s+MemoryOptimizedDataFrame",
        r"memory-efficient\s+wrapper\s+for\s+pandas\s+DataFrame"
    ],
    "exception_handling_bridge": [
        r"def\s+with_exception_handling",
        r"def\s+async_with_exception_handling",
        r"Decorator\s+to\s+add\s+standardized\s+exception\s+handling"
    ],
    "performance_monitoring": [
        r"class\s+\w+Monitoring",
        r"performance\s+monitoring",
        r"track_operation"
    ],
    "parallel_processor": [
        r"class\s+ParallelProcessor",
        r"class\s+OptimizedParallelProcessor",
        r"parallel\s+processing"
    ],
    "cache_manager": [
        r"class\s+CacheManager",
        r"class\s+AdaptiveCacheManager",
        r"cache\s+management"
    ]
}


class DuplicateImplementationFinder:
    """
    Class to find duplicate implementations across services.
    """
    
    def __init__(self, root_dir: str = '.'):
        """
        Initialize the finder.
        
        Args:
            root_dir: Root directory to scan
        """
        self.root_dir = root_dir
        self.results = {}
        self.component_files = defaultdict(list)
        self.file_hashes = {}
        self.similar_files = defaultdict(list)
    
    def scan_codebase(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan the codebase for duplicate implementations.
        
        Returns:
            Dictionary mapping component names to dictionaries with implementation details
        """
        # First pass: Find files that match component patterns
        for service_dir in SERVICE_DIRS:
            service_path = os.path.join(self.root_dir, service_dir)
            if not os.path.exists(service_path):
                continue
            
            for dirpath, dirnames, filenames in os.walk(service_path):
                # Skip directories
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                
                for filename in filenames:
                    if not filename.endswith('.py'):
                        continue
                    
                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, self.root_dir)
                    
                    try:
                        self._analyze_file(file_path, rel_path, service_dir)
                    except Exception as e:
                        print(f"Error analyzing {rel_path}: {str(e)}")
        
        # Second pass: Compute file hashes and find similar files
        self._compute_file_hashes()
        self._find_similar_files()
        
        # Third pass: Analyze results
        self._analyze_results()
        
        return self.results
    
    def _analyze_file(self, file_path: str, rel_path: str, service_dir: str) -> None:
        """
        Analyze a file for component patterns.
        
        Args:
            file_path: Path to the file
            rel_path: Relative path to the file
            service_dir: Service directory
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for component patterns
        for component_name, patterns in COMPONENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    self.component_files[component_name].append({
                        "file_path": rel_path,
                        "service": service_dir,
                        "content": content
                    })
                    break
    
    def _compute_file_hashes(self) -> None:
        """
        Compute hashes for all component files.
        """
        for component_name, files in self.component_files.items():
            for file_info in files:
                # Compute hash of file content
                content = file_info["content"]
                # Normalize whitespace and remove comments
                normalized_content = self._normalize_content(content)
                file_hash = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
                self.file_hashes[file_info["file_path"]] = file_hash
    
    def _normalize_content(self, content: str) -> str:
        """
        Normalize file content for comparison.
        
        Args:
            content: File content
            
        Returns:
            Normalized content
        """
        # Remove comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove import statements
        content = re.sub(r'import\s+.*', '', content)
        content = re.sub(r'from\s+.*\s+import\s+.*', '', content)
        
        return content.strip()
    
    def _find_similar_files(self) -> None:
        """
        Find similar files based on content hashes.
        """
        # Group files by hash
        hash_to_files = defaultdict(list)
        for file_path, file_hash in self.file_hashes.items():
            hash_to_files[file_hash].append(file_path)
        
        # Find duplicate hashes
        for file_hash, file_paths in hash_to_files.items():
            if len(file_paths) > 1:
                for file_path in file_paths:
                    self.similar_files[file_hash].append(file_path)
    
    def _analyze_results(self) -> None:
        """
        Analyze results and prepare final output.
        """
        for component_name, files in self.component_files.items():
            # Group files by service
            service_to_files = defaultdict(list)
            for file_info in files:
                service_to_files[file_info["service"]].append(file_info["file_path"])
            
            # Check if component is implemented in multiple services
            if len(service_to_files) > 1:
                self.results[component_name] = {
                    "services": list(service_to_files.keys()),
                    "implementations": [
                        {
                            "service": service,
                            "files": files
                        }
                        for service, files in service_to_files.items()
                    ],
                    "similar_files": []
                }
                
                # Add similar files
                for file_hash, file_paths in self.similar_files.items():
                    component_file_paths = [file_info["file_path"] for file_info in files]
                    if any(file_path in component_file_paths for file_path in file_paths):
                        self.results[component_name]["similar_files"].append({
                            "hash": file_hash,
                            "files": file_paths
                        })


def main():
    """Main function."""
    finder = DuplicateImplementationFinder()
    results = finder.scan_codebase()
    
    # Print results
    print("Duplicate Implementation Analysis")
    print("================================")
    
    for component_name, component_info in results.items():
        print(f"\n{component_name}:")
        print(f"  Implemented in {len(component_info['services'])} services: {', '.join(component_info['services'])}")
        
        for implementation in component_info["implementations"]:
            print(f"  {implementation['service']}:")
            for file_path in implementation["files"]:
                print(f"    {file_path}")
        
        if component_info["similar_files"]:
            print("  Similar files:")
            for similar_file_group in component_info["similar_files"]:
                print(f"    Hash: {similar_file_group['hash']}")
                for file_path in similar_file_group["files"]:
                    print(f"      {file_path}")
    
    # Save results to file
    with open('duplicate_implementation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to duplicate_implementation_analysis.json")


if __name__ == "__main__":
    main()
