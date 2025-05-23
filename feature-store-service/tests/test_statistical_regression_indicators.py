"""
Tests for Statistical and Regression Indicators.
"""
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal
from scipy import stats
from core.statistical_regression_indicators import StandardDeviationIndicator as StdDevIndicator, LinearRegressionIndicator as LinRegIndicator, LinearRegressionChannel as LinRegChannelIndicator, RSquaredIndicator as RSquaredIndicator

@pytest.fixture
def sample_data_stat() -> pd.DataFrame:
    """
    Sample data stat.
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    np.random.seed(44)
    trend = np.linspace(0, 20, periods)
    noise = np.random.randn(periods) * 2
    close = 100 + trend + noise
    high = close + np.abs(np.random.randn(periods))
    low = close - np.abs(np.random.randn(periods))
    open = close.shift(1).fillna(close.iloc[0])
    volume = np.random.randint(100, 1000, size=periods).astype(float)
    data = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=index)
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    data.loc[data.index[15:20], 'close'] = np.nan
    return data

@pytest.fixture
def short_data_stat(sample_data_stat) -> pd.DataFrame:
    """
    Short data stat.
    
    Args:
        sample_data_stat: Description of sample_data_stat
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    return sample_data_stat.head(15)

@pytest.fixture
def constant_data_stat() -> pd.DataFrame:
    """
    Constant data stat.
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.0, 'volume': 100.0}, index=index)
    return data

def test_stddev_indicator_calculation(sample_data_stat):
    window = 20
    bands = [1.0, 2.0]
    ma_type = 'sma'
    indicator = StdDevIndicator(window=window, bands=bands, moving_average_type=ma_type, column='close')
    result_df = indicator.calculate(sample_data_stat.copy())
    std_dev_name = indicator.std_dev_name
    ma_name = indicator.ma_name
    upper_1_name = f'{indicator.name_base}_upper_1_0'
    lower_1_name = f'{indicator.name_base}_lower_1_0'
    upper_2_name = f'{indicator.name_base}_upper_2_0'
    lower_2_name = f'{indicator.name_base}_lower_2_0'
    assert std_dev_name in result_df.columns
    assert ma_name in result_df.columns
    assert upper_1_name in result_df.columns
    assert lower_1_name in result_df.columns
    assert upper_2_name in result_df.columns
    assert lower_2_name in result_df.columns
    assert not result_df[std_dev_name].isnull().all()
    assert not result_df[ma_name].isnull().all()
    assert not pd.isna(result_df[std_dev_name].iloc[0])
    assert not pd.isna(result_df[ma_name].iloc[0])
    expected_ma = sample_data_stat['close'].rolling(window=window).mean()
    expected_std = sample_data_stat['close'].rolling(window=window).std()
    expected_upper1 = expected_ma + bands[0] * expected_std
    expected_lower1 = expected_ma - bands[0] * expected_std
    expected_upper2 = expected_ma + bands[1] * expected_std
    expected_lower2 = expected_ma - bands[1] * expected_std
    common_index = result_df.index[window - 1:]
    assert_series_equal(result_df.loc[common_index, ma_name], expected_ma.loc[common_index], check_dtype=False, atol=1e-06)
    assert_series_equal(result_df.loc[common_index, std_dev_name], expected_std.loc[common_index], check_dtype=False, atol=1e-06)
    assert_series_equal(result_df.loc[common_index, upper_1_name], expected_upper1.loc[common_index], check_dtype=False, atol=1e-06)
    assert_series_equal(result_df.loc[common_index, lower_1_name], expected_lower1.loc[common_index], check_dtype=False, atol=1e-06)
    assert_series_equal(result_df.loc[common_index, upper_2_name], expected_upper2.loc[common_index], check_dtype=False, atol=1e-06)
    assert_series_equal(result_df.loc[common_index, lower_2_name], expected_lower2.loc[common_index], check_dtype=False, atol=1e-06)

def test_stddev_indicator_edge_cases(short_data_stat, constant_data_stat):
    window = 20
    bands = [1.0, 2.0]
    ma_type = 'sma'
    indicator = StdDevIndicator(window=window, bands=bands, moving_average_type=ma_type, column='close')
    result_short = indicator.calculate(short_data_stat.copy())
    assert indicator.std_dev_name in result_short.columns
    assert result_short[indicator.std_dev_name].count() == 0
    assert result_short[indicator.ma_name].count() == 0
    result_const = indicator.calculate(constant_data_stat.copy())
    assert indicator.std_dev_name in result_const.columns
    assert_series_equal(result_const[indicator.ma_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.ma_name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.std_dev_name].iloc[window - 1:], pd.Series(0.0, index=result_const.index[window - 1:], name=indicator.std_dev_name), check_dtype=False, atol=1e-06)
    upper_1_name = f'{indicator.name_base}_upper_1_0'
    lower_1_name = f'{indicator.name_base}_lower_1_0'
    assert_series_equal(result_const[upper_1_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=upper_1_name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[lower_1_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=lower_1_name), check_dtype=False, atol=1e-06)

def rolling_linregress(y, window):
    """
    Rolling linregress.
    
    Args:
        y: Description of y
        window: Description of window
    
    """

    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    x = np.arange(len(y))
    slopes = np.full(len(y), np.nan)
    intercepts = np.full(len(y), np.nan)
    r_values = np.full(len(y), np.nan)
    predicted = np.full(len(y), np.nan)
    for i in range(window - 1, len(y)):
        y_slice = y.iloc[i - window + 1:i + 1]
        x_slice = x[i - window + 1:i + 1]
        mask = ~np.isnan(y_slice)
        if mask.sum() < 2:
            continue
        y_valid = y_slice[mask]
        x_valid = x_slice[mask]
        slope, intercept, r_value, _, _ = stats.linregress(x_valid, y_valid)
        slopes[i] = slope
        intercepts[i] = intercept
        r_values[i] = r_value
        predicted[i] = slope * x[i] + intercept
    return (pd.Series(slopes, index=y.index), pd.Series(intercepts, index=y.index), pd.Series(r_values, index=y.index), pd.Series(predicted, index=y.index))

def test_linreg_indicator_calculation(sample_data_stat):
    window = 14
    indicator = LinRegIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data_stat.copy())
    slope_name = indicator.slope_name
    intercept_name = indicator.intercept_name
    predicted_name = indicator.predicted_name
    assert slope_name in result_df.columns
    assert intercept_name in result_df.columns
    assert predicted_name in result_df.columns
    assert not result_df[slope_name].isnull().all()
    assert not result_df[intercept_name].isnull().all()
    assert not result_df[predicted_name].isnull().all()
    expected_slope, expected_intercept, _, expected_predicted = rolling_linregress(sample_data_stat['close'], window)
    common_index = result_df.index[window - 1:]
    assert_series_equal(result_df.loc[common_index, slope_name], expected_slope.loc[common_index], check_dtype=False, atol=1e-06)
    assert_series_equal(result_df.loc[common_index, intercept_name], expected_intercept.loc[common_index], check_dtype=False, atol=1e-06)
    assert_series_equal(result_df.loc[common_index, predicted_name], expected_predicted.loc[common_index], check_dtype=False, atol=1e-06)

def test_linreg_indicator_edge_cases(short_data_stat, constant_data_stat):
    window = 14
    indicator = LinRegIndicator(window=window, column='close')
    result_short = indicator.calculate(short_data_stat.copy())
    assert indicator.slope_name in result_short.columns
    assert result_short[indicator.slope_name].count() == 0
    assert result_short[indicator.intercept_name].count() == 0
    assert result_short[indicator.predicted_name].count() == 0
    result_const = indicator.calculate(constant_data_stat.copy())
    assert indicator.slope_name in result_const.columns
    assert_series_equal(result_const[indicator.slope_name].iloc[window - 1:], pd.Series(0.0, index=result_const.index[window - 1:], name=indicator.slope_name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.intercept_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.intercept_name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.predicted_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.predicted_name), check_dtype=False, atol=1e-06)

def test_linreg_channel_calculation(sample_data_stat):
    window = 14
    indicator = LinRegChannelIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data_stat.copy())
    upper_name = indicator.upper_name
    lower_name = indicator.lower_name
    center_name = indicator.center_name
    assert upper_name in result_df.columns
    assert lower_name in result_df.columns
    assert center_name in result_df.columns
    assert not result_df[upper_name].isnull().all()
    assert not result_df[lower_name].isnull().all()
    assert not result_df[center_name].isnull().all()
    _, _, _, predicted = rolling_linregress(sample_data_stat['close'], window)
    residuals = sample_data_stat['close'] - predicted
    max_residual = residuals.rolling(window=window).max()
    min_residual = residuals.rolling(window=window).min()
    expected_upper = predicted + max_residual
    expected_lower = predicted + min_residual
    common_index = result_df.index[window - 1:]
    assert_series_equal(result_df.loc[common_index, center_name], predicted.loc[common_index], check_dtype=False, atol=1e-06)
    assert_series_equal(result_df.loc[common_index, upper_name], expected_upper.loc[common_index], check_dtype=False, atol=1e-05)
    assert_series_equal(result_df.loc[common_index, lower_name], expected_lower.loc[common_index], check_dtype=False, atol=1e-05)

def test_linreg_channel_edge_cases(short_data_stat, constant_data_stat):
    window = 14
    indicator = LinRegChannelIndicator(window=window, column='close')
    result_short = indicator.calculate(short_data_stat.copy())
    assert indicator.upper_name in result_short.columns
    assert result_short[indicator.upper_name].count() == 0
    assert result_short[indicator.lower_name].count() == 0
    assert result_short[indicator.center_name].count() == 0
    result_const = indicator.calculate(constant_data_stat.copy())
    assert indicator.upper_name in result_const.columns
    assert_series_equal(result_const[indicator.center_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.center_name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.upper_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.upper_name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.lower_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.lower_name), check_dtype=False, atol=1e-06)

def test_rsquared_indicator_calculation(sample_data_stat):
    window = 14
    indicator = RSquaredIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data_stat.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    _, _, expected_r_value, _ = rolling_linregress(sample_data_stat['close'], window)
    expected_r_squared = expected_r_value ** 2
    common_index = result_df.index[window - 1:]
    assert_series_equal(result_df.loc[common_index, col_name], expected_r_squared.loc[common_index], check_dtype=False, atol=1e-06)
    assert (result_df[col_name].dropna() >= 0).all()
    assert (result_df[col_name].dropna() <= 1).all()

def test_rsquared_indicator_edge_cases(short_data_stat, constant_data_stat):
    window = 14
    indicator = RSquaredIndicator(window=window, column='close')
    result_short = indicator.calculate(short_data_stat.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() == 0
    result_const = indicator.calculate(constant_data_stat.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window - 1:], pd.Series(0.0, index=result_const.index[window - 1:], name=indicator.name), check_dtype=False, atol=1e-06)