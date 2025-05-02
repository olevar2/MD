"""
Tests for Advanced Moving Average Indicators.
"""
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal
import ta # Import the reference library

# Import indicators to test
from feature_store_service.indicators.advanced_moving_averages import (
    TripleExponentialMovingAverage as TEMAIndicator, # Renamed to avoid clash
    DoubleExponentialMovingAverage as DEMAIndicator, # Renamed
    HullMovingAverage as HullMAIndicator,           # Renamed
    KaufmanAdaptiveMovingAverage as KAMAIndicator, # Renamed
    ZeroLagExponentialMovingAverage as ZLEMAIndicator, # Renamed
    ArnaudLegouxMovingAverage as ALMAIndicator,     # Renamed
    JurikMovingAverage as JMAIndicator              # Renamed
)

# Sample data for testing
@pytest.fixture
def sample_data() -> pd.DataFrame:
    # Simple sine wave data for predictable MA behavior
    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    close = 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, periods)) # Increased frequency
    high = close + 1
    low = close - 1
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = np.random.randint(100, 1000, size=periods).astype(float)
    data = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=index)
    # Add some NaNs
    data.loc[data.index[5:10], 'close'] = np.nan
    return data

@pytest.fixture
def short_data(sample_data) -> pd.DataFrame:
    return sample_data.head(15) # Shorter than many windows

@pytest.fixture
def constant_data() -> pd.DataFrame:
    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    data = pd.DataFrame({
        'open': 100.0,
        'high': 100.5,
        'low': 99.5,
        'close': 100.0,
        'volume': 100.0
    }, index=index)
    return data


# --- Test TEMA ---
def test_tema_calculation(sample_data):
    window = 10
    indicator = TEMAIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data.copy()) # Use copy to avoid modifying fixture
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Reference calculation using ta library
    # Note: Our implementation uses fillna=True by default in the base EMA
    expected_series = ta.trend.tema(sample_data['close'], window=window, fillna=True)
    expected_series.name = col_name

    # Compare, allowing for small float differences and ignoring NaNs at start if fillna differs
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-6)

def test_tema_edge_cases(short_data, constant_data):
    window = 10
    indicator = TEMAIndicator(window=window, column='close')

    # Short data
    result_short = indicator.calculate(short_data.copy())
    assert f'tema_{window}' in result_short.columns
    # Expect mostly NaNs due to short length and initial NaNs in data
    assert result_short[f'tema_{window}'].count() < 5

    # Constant data
    result_const = indicator.calculate(constant_data.copy())
    assert f'tema_{window}' in result_const.columns
    # TEMA of constant should be constant after initial period
    assert_series_equal(result_const[f'tema_{window}'].iloc[window*3:], pd.Series(100.0, index=result_const.index[window*3:], name=f'tema_{window}'), check_dtype=False, atol=1e-6)


# --- Test DEMA ---
def test_dema_calculation(sample_data):
    window = 10
    indicator = DEMAIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    expected_series = ta.trend.dema(sample_data['close'], window=window, fillna=True)
    expected_series.name = col_name

    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-6)

def test_dema_edge_cases(short_data, constant_data):
    window = 10
    indicator = DEMAIndicator(window=window, column='close')

    # Short data
    result_short = indicator.calculate(short_data.copy())
    assert f'dema_{window}' in result_short.columns
    assert result_short[f'dema_{window}'].count() < 5

    # Constant data
    result_const = indicator.calculate(constant_data.copy())
    assert f'dema_{window}' in result_const.columns
    assert_series_equal(result_const[f'dema_{window}'].iloc[window*2:], pd.Series(100.0, index=result_const.index[window*2:], name=f'dema_{window}'), check_dtype=False, atol=1e-6)


# --- Test HullMA ---
# Note: ta library doesn't have HMA directly. HMA = WMA(2*WMA(close, n/2) - WMA(close, n), sqrt(n))
def test_hma_calculation(sample_data):
    window = 14
    indicator = HullMAIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Manual reference calculation (based on definition)
    close = sample_data['close']
    wma_half_period = ta.trend.wma(close, window=window // 2, fillna=True)
    wma_full_period = ta.trend.wma(close, window=window, fillna=True)
    diff_wma = 2 * wma_half_period - wma_full_period
    sqrt_window = int(np.sqrt(window))
    expected_series = ta.trend.wma(diff_wma, window=sqrt_window, fillna=True)
    expected_series.name = col_name

    # HMA can have differences due to integer rounding of periods/sqrt
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-5)

def test_hma_edge_cases(short_data, constant_data):
    window = 14
    indicator = HullMAIndicator(window=window, column='close')

    # Short data
    result_short = indicator.calculate(short_data.copy())
    assert f'hma_{window}' in result_short.columns
    # HMA needs more data points
    assert result_short[f'hma_{window}'].count() == 0 # Likely all NaN

    # Constant data
    result_const = indicator.calculate(constant_data.copy())
    assert f'hma_{window}' in result_const.columns
    # HMA of constant should be constant after initial period
    # Need enough data points: window + int(sqrt(window)) - 1
    min_periods = window + int(np.sqrt(window)) - 1
    if len(result_const) > min_periods:
         assert_series_equal(result_const[f'hma_{window}'].iloc[min_periods:], pd.Series(100.0, index=result_const.index[min_periods:], name=f'hma_{window}'), check_dtype=False, atol=1e-6)


# --- Test KAMA ---
def test_kama_calculation(sample_data):
    window = 10
    pow1 = 2
    pow2 = 30
    indicator = KAMAIndicator(window=window, pow1=pow1, pow2=pow2, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Reference calculation
    expected_series = ta.trend.kama(sample_data['close'], window=window, pow1=pow1, pow2=pow2, fillna=True)
    expected_series.name = col_name

    # KAMA can have slight differences in initial values / NaN handling
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-5)

def test_kama_edge_cases(short_data, constant_data):
    window = 10
    pow1 = 2
    pow2 = 30
    indicator = KAMAIndicator(window=window, pow1=pow1, pow2=pow2, column='close')

    # Short data
    result_short = indicator.calculate(short_data.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() < 5 # Needs window+1

    # Constant data
    result_const = indicator.calculate(constant_data.copy())
    assert indicator.name in result_const.columns
    # KAMA of constant should be constant (ER=0, SC=fastest)
    # The 'ta' implementation might differ slightly here if ER calculation leads to NaN/0 issues
    # Check after window period
    assert_series_equal(result_const[indicator.name].iloc[window+1:], pd.Series(100.0, index=result_const.index[window+1:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test ZLEMA ---
# Note: ta library doesn't have ZLEMA directly. ZLEMA = EMA(close + (close - close.shift(lag)))
def test_zlema_calculation(sample_data):
    window = 14
    indicator = ZLEMAIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Manual reference calculation
    close = sample_data['close']
    lag = (window - 1) // 2
    momentum = close.diff(lag) # close - close.shift(lag)
    adjusted_close = close + momentum.fillna(0) # Add momentum, fill initial NaN
    expected_series = ta.trend.ema_indicator(adjusted_close, window=window, fillna=True)
    expected_series.name = col_name

    # Compare, note potential differences in handling initial momentum NaNs
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-5)

def test_zlema_edge_cases(short_data, constant_data):
    window = 14
    indicator = ZLEMAIndicator(window=window, column='close')

    # Short data
    result_short = indicator.calculate(short_data.copy())
    assert f'zlema_{window}' in result_short.columns
    assert result_short[f'zlema_{window}'].count() < 5 # Needs window + lag

    # Constant data
    result_const = indicator.calculate(constant_data.copy())
    assert f'zlema_{window}' in result_const.columns
    # ZLEMA of constant: momentum is 0, so it's just EMA(constant) = constant
    assert_series_equal(result_const[f'zlema_{window}'].iloc[window:], pd.Series(100.0, index=result_const.index[window:], name=f'zlema_{window}'), check_dtype=False, atol=1e-6)


# --- Test ALMA ---
# Note: ALMA is custom, no direct 'ta' equivalent. Test based on logic.
def test_alma_calculation(sample_data):
    window = 9
    sigma = 6.0
    offset = 0.85
    indicator = ALMAIndicator(window=window, sigma=sigma, offset=offset, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    # Check first non-NaN index is reasonable (window - 1)
    assert result_df[col_name].first_valid_index() == sample_data.index[window - 1]

    # Basic check: Should generally follow the close price trend
    assert result_df[col_name].iloc[-1] == pytest.approx(sample_data['close'].dropna().iloc[-1], abs=5)

    # TODO: Add comparison with known values if possible (e.g., from another library or manual calc)

def test_alma_edge_cases(short_data, constant_data):
    window = 9
    sigma = 6.0
    offset = 0.85
    indicator = ALMAIndicator(window=window, sigma=sigma, offset=offset, column='close')

    # Short data
    result_short = indicator.calculate(short_data.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() < (len(short_data) - window + 1) # Check count relative to possible values

    # Constant data
    result_const = indicator.calculate(constant_data.copy())
    assert indicator.name in result_const.columns
    # ALMA of constant should be constant
    assert_series_equal(result_const[indicator.name].iloc[window-1:], pd.Series(100.0, index=result_const.index[window-1:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test JMA ---
# Note: JMA is custom, no direct 'ta' equivalent. Test based on logic.
def test_jma_calculation(sample_data):
    window = 7
    phase = 0
    power = 2
    indicator = JMAIndicator(window=window, phase=phase, power=power) # Requires high, low, close
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    # Implementation starts calculation from index 1, check first value isn't NaN
    assert not pd.isna(result_df[col_name].iloc[1])

    # Basic check: Should generally follow the close price trend
    assert result_df[col_name].iloc[-1] == pytest.approx(sample_data['close'].dropna().iloc[-1], abs=5)

    # TODO: Add comparison with known values if possible

def test_jma_edge_cases(short_data, constant_data):
    window = 7
    phase = 0
    power = 2
    indicator = JMAIndicator(window=window, phase=phase, power=power)

    # Short data (needs high/low/close)
    short_data_jma = short_data[['high', 'low', 'close']].copy()
    result_short = indicator.calculate(short_data_jma)
    assert indicator.name in result_short.columns
    # JMA calculation is complex, check if *any* non-NaN values are produced
    assert result_short[indicator.name].count() > 0

    # Constant data
    constant_data_jma = constant_data[['high', 'low', 'close']].copy()
    result_const = indicator.calculate(constant_data_jma)
    assert indicator.name in result_const.columns
    # JMA of constant should be constant (volatility=0)
    # Check after a few periods as internal state stabilizes
    assert_series_equal(result_const[indicator.name].iloc[5:], pd.Series(100.0, index=result_const.index[5:], name=indicator.name), check_dtype=False, atol=1e-6)

# TODO: Add tests specifically for NaN handling within the series (already added some NaNs to sample_data)
