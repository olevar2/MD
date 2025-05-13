"""
Tests for Advanced Moving Average Indicators.
"""
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal
import ta
from core.advanced_moving_averages import TripleExponentialMovingAverage as TEMAIndicator, DoubleExponentialMovingAverage as DEMAIndicator, HullMovingAverage as HullMAIndicator, KaufmanAdaptiveMovingAverage as KAMAIndicator, ZeroLagExponentialMovingAverage as ZLEMAIndicator, ArnaudLegouxMovingAverage as ALMAIndicator, JurikMovingAverage as JMAIndicator

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Sample data.
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    close = 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, periods))
    high = close + 1
    low = close - 1
    open = close.shift(1).fillna(close.iloc[0])
    volume = np.random.randint(100, 1000, size=periods).astype(float)
    data = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=index)
    data.loc[data.index[5:10], 'close'] = np.nan
    return data

@pytest.fixture
def short_data(sample_data) -> pd.DataFrame:
    """
    Short data.
    
    Args:
        sample_data: Description of sample_data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    return sample_data.head(15)

@pytest.fixture
def constant_data() -> pd.DataFrame:
    """
    Constant data.
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.0, 'volume': 100.0}, index=index)
    return data

def test_tema_calculation(sample_data):
    window = 10
    indicator = TEMAIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.trend.tema(sample_data['close'], window=window, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_tema_edge_cases(short_data, constant_data):
    window = 10
    indicator = TEMAIndicator(window=window, column='close')
    result_short = indicator.calculate(short_data.copy())
    assert f'tema_{window}' in result_short.columns
    assert result_short[f'tema_{window}'].count() < 5
    result_const = indicator.calculate(constant_data.copy())
    assert f'tema_{window}' in result_const.columns
    assert_series_equal(result_const[f'tema_{window}'].iloc[window * 3:], pd.Series(100.0, index=result_const.index[window * 3:], name=f'tema_{window}'), check_dtype=False, atol=1e-06)

def test_dema_calculation(sample_data):
    window = 10
    indicator = DEMAIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.trend.dema(sample_data['close'], window=window, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_dema_edge_cases(short_data, constant_data):
    window = 10
    indicator = DEMAIndicator(window=window, column='close')
    result_short = indicator.calculate(short_data.copy())
    assert f'dema_{window}' in result_short.columns
    assert result_short[f'dema_{window}'].count() < 5
    result_const = indicator.calculate(constant_data.copy())
    assert f'dema_{window}' in result_const.columns
    assert_series_equal(result_const[f'dema_{window}'].iloc[window * 2:], pd.Series(100.0, index=result_const.index[window * 2:], name=f'dema_{window}'), check_dtype=False, atol=1e-06)

def test_hma_calculation(sample_data):
    window = 14
    indicator = HullMAIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    close = sample_data['close']
    wma_half_period = ta.trend.wma(close, window=window // 2, fillna=True)
    wma_full_period = ta.trend.wma(close, window=window, fillna=True)
    diff_wma = 2 * wma_half_period - wma_full_period
    sqrt_window = int(np.sqrt(window))
    expected_series = ta.trend.wma(diff_wma, window=sqrt_window, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-05)

def test_hma_edge_cases(short_data, constant_data):
    window = 14
    indicator = HullMAIndicator(window=window, column='close')
    result_short = indicator.calculate(short_data.copy())
    assert f'hma_{window}' in result_short.columns
    assert result_short[f'hma_{window}'].count() == 0
    result_const = indicator.calculate(constant_data.copy())
    assert f'hma_{window}' in result_const.columns
    min_periods = window + int(np.sqrt(window)) - 1
    if len(result_const) > min_periods:
        assert_series_equal(result_const[f'hma_{window}'].iloc[min_periods:], pd.Series(100.0, index=result_const.index[min_periods:], name=f'hma_{window}'), check_dtype=False, atol=1e-06)

def test_kama_calculation(sample_data):
    window = 10
    pow1 = 2
    pow2 = 30
    indicator = KAMAIndicator(window=window, pow1=pow1, pow2=pow2, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.trend.kama(sample_data['close'], window=window, pow1=pow1, pow2=pow2, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-05)

def test_kama_edge_cases(short_data, constant_data):
    window = 10
    pow1 = 2
    pow2 = 30
    indicator = KAMAIndicator(window=window, pow1=pow1, pow2=pow2, column='close')
    result_short = indicator.calculate(short_data.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() < 5
    result_const = indicator.calculate(constant_data.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window + 1:], pd.Series(100.0, index=result_const.index[window + 1:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_zlema_calculation(sample_data):
    window = 14
    indicator = ZLEMAIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    close = sample_data['close']
    lag = (window - 1) // 2
    momentum = close.diff(lag)
    adjusted_close = close + momentum.fillna(0)
    expected_series = ta.trend.ema_indicator(adjusted_close, window=window, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-05)

def test_zlema_edge_cases(short_data, constant_data):
    window = 14
    indicator = ZLEMAIndicator(window=window, column='close')
    result_short = indicator.calculate(short_data.copy())
    assert f'zlema_{window}' in result_short.columns
    assert result_short[f'zlema_{window}'].count() < 5
    result_const = indicator.calculate(constant_data.copy())
    assert f'zlema_{window}' in result_const.columns
    assert_series_equal(result_const[f'zlema_{window}'].iloc[window:], pd.Series(100.0, index=result_const.index[window:], name=f'zlema_{window}'), check_dtype=False, atol=1e-06)

def test_alma_calculation(sample_data):
    window = 9
    sigma = 6.0
    offset = 0.85
    indicator = ALMAIndicator(window=window, sigma=sigma, offset=offset, column='close')
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert result_df[col_name].first_valid_index() == sample_data.index[window - 1]
    assert result_df[col_name].iloc[-1] == pytest.approx(sample_data['close'].dropna().iloc[-1], abs=5)

def test_alma_edge_cases(short_data, constant_data):
    window = 9
    sigma = 6.0
    offset = 0.85
    indicator = ALMAIndicator(window=window, sigma=sigma, offset=offset, column='close')
    result_short = indicator.calculate(short_data.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() < len(short_data) - window + 1
    result_const = indicator.calculate(constant_data.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_jma_calculation(sample_data):
    window = 7
    phase = 0
    power = 2
    indicator = JMAIndicator(window=window, phase=phase, power=power)
    result_df = indicator.calculate(sample_data.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not pd.isna(result_df[col_name].iloc[1])
    assert result_df[col_name].iloc[-1] == pytest.approx(sample_data['close'].dropna().iloc[-1], abs=5)

def test_jma_edge_cases(short_data, constant_data):
    window = 7
    phase = 0
    power = 2
    indicator = JMAIndicator(window=window, phase=phase, power=power)
    short_data_jma = short_data[['high', 'low', 'close']].copy()
    result_short = indicator.calculate(short_data_jma)
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    constant_data_jma = constant_data[['high', 'low', 'close']].copy()
    result_const = indicator.calculate(constant_data_jma)
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[5:], pd.Series(100.0, index=result_const.index[5:], name=indicator.name), check_dtype=False, atol=1e-06)