"""
Test runner for reliability components.
"""
import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_reliability_tests():
    """Run all reliability-related tests"""
    test_paths = [
        "tests/reliability/test_reliability_manager.py",
        "tests/verification/test_multi_level_verifier.py",
        "tests/verification/test_signal_filter.py",
        "tests/recovery/test_integrated_recovery.py"
    ]
    
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
        "-m", "not integration",  # Skip other integration tests
        *test_paths
    ]
    
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(run_reliability_tests())
