#!/usr/bin/env python3
"""
Script to compile Protocol Buffer files for the Forex Trading Platform.
This script generates Python code from .proto files.
"""

import os
import subprocess
import sys
from pathlib import Path

# Root directory of the proto files
PROTO_ROOT = Path(__file__).parent.absolute()

# Output directory for generated Python code
OUTPUT_DIR = PROTO_ROOT.parent / "common_lib" / "grpc"

# Services to compile
SERVICES = [
    "common",
    "causal_analysis",
    "backtesting",
    "market_analysis",
    "analysis_coordinator"
]

def ensure_directory(directory):
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)

def compile_proto(proto_file, output_dir, proto_path):
    """Compile a single .proto file."""
    cmd = [
        "python", "-m", "grpc_tools.protoc",
        f"--proto_path={proto_path}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        proto_file
    ]
    
    print(f"Compiling {proto_file}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error compiling {proto_file}:")
        print(result.stderr)
        return False
    
    return True

def create_init_files(directory):
    """Create __init__.py files in all subdirectories."""
    for root, dirs, files in os.walk(directory):
        init_file = os.path.join(root, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass

def main():
    """Main function to compile all proto files."""
    # Ensure output directory exists
    ensure_directory(OUTPUT_DIR)
    
    # Create __init__.py in the output directory
    with open(os.path.join(OUTPUT_DIR, "__init__.py"), "w") as f:
        pass
    
    # Compile proto files for each service
    success = True
    for service in SERVICES:
        service_dir = PROTO_ROOT / service
        service_output_dir = OUTPUT_DIR / service
        
        ensure_directory(service_output_dir)
        
        # Create __init__.py in the service output directory
        with open(os.path.join(service_output_dir, "__init__.py"), "w") as f:
            pass
        
        # Find all .proto files in the service directory
        proto_files = list(service_dir.glob("*.proto"))
        
        if not proto_files:
            print(f"No .proto files found in {service_dir}")
            continue
        
        # Compile each .proto file
        for proto_file in proto_files:
            if not compile_proto(str(proto_file), str(OUTPUT_DIR), str(PROTO_ROOT)):
                success = False
    
    # Create __init__.py files in all subdirectories
    create_init_files(OUTPUT_DIR)
    
    if success:
        print("All proto files compiled successfully!")
    else:
        print("Some proto files failed to compile.")
        sys.exit(1)

if __name__ == "__main__":
    main()