"""
Wrapper script for running tests with the proper Python path.
"""
import os
import sys
import subprocess

# Add the project root directory to Python's path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Run pytest with the specified arguments
if __name__ == "__main__":
    test_path = "testing/test_indicator_integration.py"
    print(f"Running tests with project root in PYTHONPATH: {project_root}")
    subprocess.run([sys.executable, "-m", "pytest", test_path, "-v"])
