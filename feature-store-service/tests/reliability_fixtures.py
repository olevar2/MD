"""
Test fixtures for reliability components.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for configuration files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(temp_config_dir):
    """Create a test configuration"""
    config = {
        "verification": {
            "input_validation": {
                "required_columns": {
                    "price_data": ["timestamp", "close", "volume"]
                }
            },
            "risk_limits": {
                "max_position_size": 1000,
                "max_leverage": 5
            }
        },
        "signal_filtering": {
            "price": {
                "outlier_std_threshold": 2.0,
                "window_size": 3
            }
        },
        "recovery": {
            "storage": {
                "state_dir": str(Path(temp_config_dir) / "states")
            }
        }
    }
    
    config_path = Path(temp_config_dir) / 'test_reliability.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    return str(config_path)

@pytest.fixture
def test_price_data():
    """Create test price data"""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-01', periods=5),
        'close': [100.0, 101.0, 102.0, 101.5, 102.5],
        'volume': [1000, 1100, 1200, 1150, 1250]
    })

@pytest.fixture
def test_historical_decisions():
    """Create test historical decisions"""
    return [
        {
            'decision': {'action': 'buy', 'confidence': 0.8},
            'context': {
                'market_condition': 'bullish',
                'volatility': 0.14,
                'trend': 0.75
            },
            'timestamp': datetime.utcnow()
        }
    ]
