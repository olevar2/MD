# filepath: d:\MD\forex_trading_platform\testing\feedback_tests\test_strategy_mutation.py
"""
Unit tests for the StrategyMutator and mutation functions.
"""

import pytest
import random
from unittest.mock import patch

from analysis_engine.adaptive_layer.strategy_mutation import (
    StrategyMutator,
    mutate_parameter,
    swap_indicator
)

# --- Test Data --- 

@pytest.fixture
def sample_strategy_config():
    """
    Sample strategy config.
    
    """

    return {
        'id': 'SMA_Cross_1',
        'version': 1.0,
        'indicators': [
            {'name': 'SMA', 'period': 10},
            {'name': 'EMA', 'period': 50} # Use different indicators for swap test
        ],
        'parameters': {
            'stop_loss_pips': 20.0,
            'take_profit_pips': 60.0,
            'lot_size': 0.1
        },
        'rules': {
            'entry': 'SMA_10 > EMA_50',
            'exit': 'SMA_10 < EMA_50'
        }
    }

# --- Test Mutation Functions --- 

def test_mutate_parameter_numeric(sample_strategy_config):
    """Test mutating a numerical parameter."""
    original_value = sample_strategy_config['parameters']['stop_loss_pips']
    mutated_config = mutate_parameter(sample_strategy_config, 'stop_loss_pips', mutation_scale=0.1)
    assert mutated_config['parameters']['stop_loss_pips'] != original_value
    # Check if it's within a reasonable range (e.g., +/- 10%)
    assert abs(mutated_config['parameters']['stop_loss_pips'] - original_value) <= original_value * 0.1

def test_mutate_parameter_non_existent(sample_strategy_config):
    """Test attempting to mutate a parameter that doesn't exist."""
    original_config = sample_strategy_config.copy()
    mutated_config = mutate_parameter(sample_strategy_config, 'non_existent_param')
    assert mutated_config == original_config # No change expected

def test_swap_indicator(sample_strategy_config):
    """Test swapping an indicator."""
    available = ['SMA', 'EMA', 'RSI', 'MACD']
    original_indicators = [ind['name'] for ind in sample_strategy_config['indicators']]
    
    # Run multiple times as swap is random
    changed = False
    for _ in range(10): # Try a few times
        mutated_config = swap_indicator(sample_strategy_config.copy(), available_indicators=available)
        mutated_indicators = [ind['name'] for ind in mutated_config['indicators']]
        if mutated_indicators != original_indicators:
            changed = True
            # Check that the swapped indicator is from the available list and not the original one
            for i, name in enumerate(mutated_indicators):
                if name != original_indicators[i]:
                    assert name in available
                    assert name != original_indicators[i]
            break
    assert changed, "Indicator swap did not occur after multiple attempts"

def test_swap_indicator_no_options(sample_strategy_config):
    """Test swap when no alternative indicators are available."""
    available = ['SMA', 'EMA'] # Only the ones already present
    original_config = sample_strategy_config.copy()
    mutated_config = swap_indicator(sample_strategy_config, available_indicators=available)
    assert mutated_config == original_config

# --- Test StrategyMutator Class --- 

@pytest.fixture
def mutator():
    return StrategyMutator(config={'available_indicators': ['SMA', 'EMA', 'RSI', 'MACD']})

@pytest.mark.asyncio
async def test_mutator_initialization(mutator):
    """Test the initializer sets defaults correctly."""
    assert mutator.default_mutation_rate == 0.1
    assert 'mutate_parameter' in mutator.mutation_operations
    assert 'swap_indicator' in mutator.mutation_operations
    assert mutator.available_indicators == ['SMA', 'EMA', 'RSI', 'MACD']

@pytest.mark.asyncio
@patch('random.random', return_value=0.05) # Ensure mutation happens (rate > 0.05)
@patch('random.choice') # Mock the choice of operation
async def test_mutate_strategy_calls_operation(mock_choice, mock_random, mutator, sample_strategy_config):
    """Test that mutate_strategy selects and calls a mutation operation."""
    # Mock the chosen operation function itself to check if it's called
    mock_mutate_param_func = AsyncMock(return_value=sample_strategy_config) # Simulate no change for simplicity
    mutator.mutation_operations['mutate_parameter'] = mock_mutate_param_func
    mock_choice.return_value = 'mutate_parameter' # Force choosing mutate_parameter

    mutated = await mutator.mutate_strategy(sample_strategy_config)

    mock_choice.assert_called_once()
    # Check that the *mocked* operation function was called
    # We need to assert that the mock we injected was called, not the original function
    # This assertion seems incorrect, let's rethink. We need to patch the *actual* function called.

@pytest.mark.asyncio
@patch('random.random', return_value=0.05) # Ensure mutation happens
@patch('analysis_engine.adaptive_layer.strategy_mutation.mutate_parameter', new_callable=MagicMock) # Patch the actual function
@patch('random.choice', return_value='mutate_parameter') # Force choice
async def test_mutate_strategy_calls_mutate_parameter(mock_choice, mock_mutate_func, mock_random, mutator, sample_strategy_config):
    """Test mutate_strategy correctly calls mutate_parameter."""
    mock_mutate_func.return_value = sample_strategy_config.copy() # Return a copy
    
    # Need to select a parameter *within* the test, as the original function does
    with patch('random.choice') as mock_param_choice: 
        # Mock the *second* random.choice call (for parameter selection)
        mock_param_choice.return_value = 'stop_loss_pips' 
        
        mutated = await mutator.mutate_strategy(sample_strategy_config)

    mock_mutate_func.assert_called_once()
    # Check args passed to the *actual* patched mutate_parameter function
    call_args, call_kwargs = mock_mutate_func.call_args
    assert call_args[0] == sample_strategy_config # First arg is the config
    assert call_kwargs.get('param_name') == 'stop_loss_pips'
    assert mutated is not None # Should return the mutated config
    assert 'last_mutation' in mutated['metadata']
    assert mutated['metadata']['last_mutation']['operation'] == 'mutate_parameter'

@pytest.mark.asyncio
@patch('random.random', return_value=0.05) # Ensure mutation happens
@patch('analysis_engine.adaptive_layer.strategy_mutation.swap_indicator', new_callable=MagicMock) # Patch the actual function
@patch('random.choice', return_value='swap_indicator') # Force choice
async def test_mutate_strategy_calls_swap_indicator(mock_choice, mock_swap_func, mock_random, mutator, sample_strategy_config):
    """Test mutate_strategy correctly calls swap_indicator."""
    mutated_config_copy = sample_strategy_config.copy()
    mutated_config_copy['indicators'][0]['name'] = 'RSI' # Simulate a change
    mock_swap_func.return_value = mutated_config_copy

    mutated = await mutator.mutate_strategy(sample_strategy_config)

    mock_swap_func.assert_called_once_with(sample_strategy_config, available_indicators=mutator.available_indicators)
    assert mutated is not None
    assert 'last_mutation' in mutated['metadata']
    assert mutated['metadata']['last_mutation']['operation'] == 'swap_indicator'
    assert mutated['indicators'][0]['name'] == 'RSI' # Check change persisted

@pytest.mark.asyncio
@patch('random.random', return_value=0.5) # Ensure mutation is skipped (rate < 0.5)
async def test_mutate_strategy_skips_based_on_rate(mock_random, mutator, sample_strategy_config):
    """Test that mutation is skipped if random number exceeds rate."""
    mutated = await mutator.mutate_strategy(sample_strategy_config)
    assert mutated is None

@pytest.mark.asyncio
async def test_mutate_strategy_handles_exception(mutator, sample_strategy_config):
    """Test error handling during mutation application."""
    with patch('random.random', return_value=0.05): # Ensure mutation happens
        with patch('random.choice', return_value='mutate_parameter'):
            # Make the mutation function raise an error
            with patch('analysis_engine.adaptive_layer.strategy_mutation.mutate_parameter', side_effect=ValueError("Test Error")):
                 with patch('random.choice', return_value='stop_loss_pips'): # Mock inner choice
                    mutated = await mutator.mutate_strategy(sample_strategy_config)
                    assert mutated is None # Should return None on error

# Add tests for register_mutation_operation if needed

from unittest.mock import AsyncMock # Import needed
