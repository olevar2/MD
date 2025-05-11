"""
Simple test script for the BaseIndicator class.
"""

import sys
import os
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the BaseIndicator class
from common_lib.indicators.base_indicator import BaseIndicator

# Create a simple subclass for testing
class SimpleMovingAverage(BaseIndicator):
    """Simple Moving Average indicator for testing."""
    
    category = "trend"
    name = "SimpleMovingAverage"
    default_params = {"window": 10}
    required_params = {"window": int}
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA for the 'close' column."""
        self.validate_input(data, ["close"])
        result = data.copy()
        window = self.params["window"]
        result[f"SMA_{window}"] = result["close"].rolling(window=window).mean()
        return result

def test_base_indicator():
    """Test the BaseIndicator class."""
    print("Testing BaseIndicator class...")
    
    # Create test data
    data = pd.DataFrame({
        "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    })
    
    # Create indicator instance
    sma = SimpleMovingAverage({"window": 3})
    print(f"Created indicator: {sma}")
    
    # Calculate indicator values
    result = sma.calculate(data)
    print(f"Calculated SMA values: {result['SMA_3'].tolist()}")
    
    # Test metadata
    metadata = SimpleMovingAverage.get_metadata()
    print(f"Metadata: {metadata}")
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    test_base_indicator()