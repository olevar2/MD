"""
Tests for Volume Analysis Indicators.
"""
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal
import ta
from feature_store_service.indicators.volume_analysis import MoneyFlowIndex as MoneyFlowIndexIndicator, EaseOfMovement as EaseOfMovementIndicator, VolumeWeightedAveragePrice as VWAPIndicator, VWAPBands as VWAPBandsIndicator, MarketFacilitationIndex as MarketFacilitationIndexIndicator, VolumeZoneOscillator as VZOIndicator, NVIAndPVI as NVIAndPVIIndicator, DemandIndex as DemandIndexIndicator, RelativeVolume as RelativeVolumeIndicator

@pytest.fixture
def sample_data_vol() -> pd.DataFrame:
    """
    Sample data vol.
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    np.random.seed(43)
    close = 100 + np.cumsum(np.random.randn(periods))
    high = close + np.abs(np.random.randn(periods)) * 2
    low = close - np.abs(np.random.randn(periods)) * 2
    open = close.shift(1).fillna(close.iloc[0])
    volume = np.random.randint(1000, 5000, size=periods).astype(float)
    volume[periods // 2] = 0
    volume[periods // 4] = np.nan
    data = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=index)
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    data.loc[data.index[10:15], 'close'] = np.nan
    return data

@pytest.fixture
def short_data_vol(sample_data_vol) -> pd.DataFrame:
    """
    Short data vol.
    
    Args:
        sample_data_vol: Description of sample_data_vol
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    return sample_data_vol.head(20)

@pytest.fixture
def constant_data_vol() -> pd.DataFrame:
    """
    Constant data vol.
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.0, 'volume': 1000.0}, index=index)
    return data

@pytest.fixture
def zero_volume_data(constant_data_vol) -> pd.DataFrame:
    """
    Zero volume data.
    
    Args:
        constant_data_vol: Description of constant_data_vol
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

    data = constant_data_vol.copy()
    data['volume'] = 0.0
    return data

def test_mfi_calculation(sample_data_vol):
    window = 14
    indicator = MoneyFlowIndexIndicator(window=window)
    result_df = indicator.calculate(sample_data_vol.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.volume.money_flow_index(sample_data_vol['high'], sample_data_vol['low'], sample_data_vol['close'], sample_data_vol['volume'], window=window, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_mfi_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    window = 14
    indicator = MoneyFlowIndexIndicator(window=window)
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    result_const = indicator.calculate(constant_data_vol.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(50.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert indicator.name in result_zero_vol.columns
    assert_series_equal(result_zero_vol[indicator.name].iloc[window:], pd.Series(50.0, index=result_zero_vol.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_eom_calculation(sample_data_vol):
    window = 14
    indicator = EaseOfMovementIndicator(window=window)
    result_df = indicator.calculate(sample_data_vol.copy())
    col_name = indicator.name
    signal_name = indicator.signal_name
    assert col_name in result_df.columns
    assert signal_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not result_df[signal_name].isnull().all()
    expected_eom = ta.volume.ease_of_movement(sample_data_vol['high'], sample_data_vol['low'], sample_data_vol['volume'], window=window, fillna=True)
    expected_sig = ta.volume.sma_ease_of_movement(sample_data_vol['high'], sample_data_vol['low'], sample_data_vol['volume'], window=window, fillna=True)
    expected_eom.name = col_name
    expected_sig.name = signal_name
    assert_series_equal(result_df[signal_name].dropna(), expected_sig.dropna(), check_dtype=False, atol=1e-06)

def test_eom_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    window = 14
    indicator = EaseOfMovementIndicator(window=window)
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.name in result_short.columns
    assert indicator.signal_name in result_short.columns
    assert result_short[indicator.name].count() > 0
    assert result_short[indicator.signal_name].count() > 0
    result_const = indicator.calculate(constant_data_vol.copy())
    assert indicator.name in result_const.columns
    assert indicator.signal_name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.signal_name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.signal_name), check_dtype=False, atol=1e-06)
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert indicator.name in result_zero_vol.columns
    assert indicator.signal_name in result_zero_vol.columns
    assert result_zero_vol[indicator.name].iloc[window:].isnull().all() or (result_zero_vol[indicator.name].iloc[window:] == 0).all()
    assert result_zero_vol[indicator.signal_name].iloc[window:].isnull().all() or (result_zero_vol[indicator.signal_name].iloc[window:] == 0).all()

def test_vwap_calculation(sample_data_vol):
    window = 14
    indicator = VWAPIndicator(window=window)
    result_df = indicator.calculate(sample_data_vol.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.volume.volume_weighted_average_price(sample_data_vol['high'], sample_data_vol['low'], sample_data_vol['close'], sample_data_vol['volume'], window=window, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_vwap_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    window = 14
    indicator = VWAPIndicator(window=window)
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    result_const = indicator.calculate(constant_data_vol.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.name), check_dtype=False, atol=1e-06)
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert indicator.name in result_zero_vol.columns
    assert result_zero_vol[indicator.name].iloc[window - 1:].isnull().all()

def test_vwap_bands_calculation(sample_data_vol):
    window = 14
    num_std = 2.0
    indicator = VWAPBandsIndicator(window=window, num_std_dev=num_std)
    result_df = indicator.calculate(sample_data_vol.copy())
    upper_name = indicator.upper_band_name
    lower_name = indicator.lower_band_name
    vwap_name = indicator.vwap_name
    assert vwap_name in result_df.columns
    assert upper_name in result_df.columns
    assert lower_name in result_df.columns
    assert not result_df[upper_name].isnull().all()
    assert not result_df[lower_name].isnull().all()
    assert not pd.isna(result_df[upper_name].iloc[0])
    assert not pd.isna(result_df[lower_name].iloc[0])
    assert (result_df[upper_name].dropna() >= result_df[vwap_name].dropna() - 1e-09).all()
    assert (result_df[lower_name].dropna() <= result_df[vwap_name].dropna() + 1e-09).all()
    vwap_series = result_df[vwap_name]
    typical_price = (sample_data_vol['high'] + sample_data_vol['low'] + sample_data_vol['close']) / 3
    sq_diff = ((typical_price - vwap_series) ** 2 * sample_data_vol['volume']).rolling(window=window).sum()
    vol_sum = sample_data_vol['volume'].rolling(window=window).sum()
    variance = sq_diff / vol_sum
    std_dev = np.sqrt(variance)
    expected_upper = vwap_series + num_std * std_dev
    expected_lower = vwap_series - num_std * std_dev
    assert_series_equal(result_df[upper_name].dropna(), expected_upper.dropna(), check_dtype=False, atol=1e-05)
    assert_series_equal(result_df[lower_name].dropna(), expected_lower.dropna(), check_dtype=False, atol=1e-05)

def test_vwap_bands_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    window = 14
    num_std = 2.0
    indicator = VWAPBandsIndicator(window=window, num_std_dev=num_std)
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.upper_band_name in result_short.columns
    assert indicator.lower_band_name in result_short.columns
    assert result_short[indicator.upper_band_name].count() > 0
    assert result_short[indicator.lower_band_name].count() > 0
    result_const = indicator.calculate(constant_data_vol.copy())
    assert indicator.upper_band_name in result_const.columns
    assert indicator.lower_band_name in result_const.columns
    assert_series_equal(result_const[indicator.upper_band_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.upper_band_name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.lower_band_name].iloc[window - 1:], pd.Series(100.0, index=result_const.index[window - 1:], name=indicator.lower_band_name), check_dtype=False, atol=1e-06)
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert indicator.upper_band_name in result_zero_vol.columns
    assert indicator.lower_band_name in result_zero_vol.columns
    assert result_zero_vol[indicator.upper_band_name].iloc[window - 1:].isnull().all()
    assert result_zero_vol[indicator.lower_band_name].iloc[window - 1:].isnull().all()

def test_bw_mfi_calculation(sample_data_vol):
    indicator = MarketFacilitationIndexIndicator()
    result_df = indicator.calculate(sample_data_vol.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.volume.market_facilitation_index(sample_data_vol['high'], sample_data_vol['low'], sample_data_vol['volume'], fillna=True)
    expected_series.name = col_name
    expected_series = expected_series.replace([np.inf, -np.inf], 0)
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_bw_mfi_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    indicator = MarketFacilitationIndexIndicator()
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    result_const = indicator.calculate(constant_data_vol.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[1:], pd.Series(0.001, index=result_const.index[1:], name=indicator.name), check_dtype=False, atol=1e-06)
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert indicator.name in result_zero_vol.columns
    assert_series_equal(result_zero_vol[indicator.name].iloc[1:], pd.Series(0.0, index=result_zero_vol.index[1:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_vzo_calculation(sample_data_vol):
    window = 14
    ema_window = 60
    indicator = VZOIndicator(window=window, ema_window=ema_window)
    result_df = indicator.calculate(sample_data_vol.copy())
    col_name = indicator.name
    ema_name = indicator.ema_name
    assert col_name in result_df.columns
    assert ema_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not result_df[ema_name].isnull().all()
    assert not pd.isna(result_df[col_name].iloc[0])
    assert not pd.isna(result_df[ema_name].iloc[0])
    assert result_df[col_name].iloc[window:].min() > -150
    assert result_df[col_name].iloc[window:].max() < 150

def test_vzo_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    window = 14
    ema_window = 60
    indicator = VZOIndicator(window=window, ema_window=ema_window)
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.name in result_short.columns
    assert indicator.ema_name in result_short.columns
    assert result_short[indicator.name].count() > 0
    assert result_short[indicator.ema_name].count() > 0
    result_const = indicator.calculate(constant_data_vol.copy())
    assert indicator.name in result_const.columns
    assert indicator.ema_name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.ema_name].iloc[ema_window:], pd.Series(0.0, index=result_const.index[ema_window:], name=indicator.ema_name), check_dtype=False, atol=1e-06)
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert indicator.name in result_zero_vol.columns
    assert indicator.ema_name in result_zero_vol.columns
    assert_series_equal(result_zero_vol[indicator.name].iloc[window:], pd.Series(0.0, index=result_zero_vol.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_zero_vol[indicator.ema_name].iloc[ema_window:], pd.Series(0.0, index=result_zero_vol.index[ema_window:], name=indicator.ema_name), check_dtype=False, atol=1e-06)

def test_nvi_pvi_calculation(sample_data_vol):
    nvi_ema = 255
    pvi_ema = 255
    indicator = NVIAndPVIIndicator(nvi_ema_window=nvi_ema, pvi_ema_window=pvi_ema)
    result_df = indicator.calculate(sample_data_vol.copy())
    nvi_name = indicator.nvi_name
    pvi_name = indicator.pvi_name
    nvi_signal = indicator.nvi_signal_name
    pvi_signal = indicator.pvi_signal_name
    assert nvi_name in result_df.columns
    assert pvi_name in result_df.columns
    assert nvi_signal in result_df.columns
    assert pvi_signal in result_df.columns
    assert result_df[nvi_name].iloc[0] == 1000.0
    assert result_df[pvi_name].iloc[0] == 1000.0
    assert not result_df[nvi_name].isnull().any()
    assert not result_df[pvi_name].isnull().any()
    assert not result_df[nvi_signal].isnull().any()
    assert not result_df[pvi_signal].isnull().any()
    assert not (result_df[nvi_name] == 1000.0).all()
    assert not (result_df[pvi_name] == 1000.0).all()

def test_nvi_pvi_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    nvi_ema = 255
    pvi_ema = 255
    indicator = NVIAndPVIIndicator(nvi_ema_window=nvi_ema, pvi_ema_window=pvi_ema)
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.nvi_name in result_short.columns
    assert indicator.pvi_name in result_short.columns
    assert indicator.nvi_signal_name in result_short.columns
    assert indicator.pvi_signal_name in result_short.columns
    assert result_short[indicator.nvi_name].count() == len(short_data_vol)
    result_const = indicator.calculate(constant_data_vol.copy())
    assert (result_const[indicator.nvi_name] == 1000.0).all()
    assert (result_const[indicator.pvi_name] == 1000.0).all()
    assert (result_const[indicator.nvi_signal_name] == 1000.0).all()
    assert (result_const[indicator.pvi_signal_name] == 1000.0).all()
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert (result_zero_vol[indicator.nvi_name] == 1000.0).all()
    assert (result_zero_vol[indicator.pvi_name] == 1000.0).all()
    assert (result_zero_vol[indicator.nvi_signal_name] == 1000.0).all()
    assert (result_zero_vol[indicator.pvi_signal_name] == 1000.0).all()

def test_demand_index_calculation(sample_data_vol):
    window = 10
    indicator = DemandIndexIndicator(window=window)
    result_df = indicator.calculate(sample_data_vol.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not pd.isna(result_df[col_name].iloc[0])
    assert result_df[col_name].iloc[window:].mean() == pytest.approx(1.0, abs=1.0)

def test_demand_index_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    window = 10
    indicator = DemandIndexIndicator(window=window)
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    result_const = indicator.calculate(constant_data_vol.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(1.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert indicator.name in result_zero_vol.columns
    assert_series_equal(result_zero_vol[indicator.name].iloc[window:], pd.Series(1.0, index=result_zero_vol.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_rvol_calculation(sample_data_vol):
    window = 50
    indicator = RelativeVolumeIndicator(window=window)
    result_df = indicator.calculate(sample_data_vol.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert result_df[col_name].iloc[0] == 1.0
    assert (result_df[col_name].dropna() > 0).all()
    avg_vol = sample_data_vol['volume'].rolling(window=window).mean().shift(1)
    expected_rvol = sample_data_vol['volume'] / avg_vol
    assert_series_equal(result_df[col_name].iloc[window:].dropna(), expected_rvol.iloc[window:].dropna(), check_dtype=False, atol=1e-06)

def test_rvol_edge_cases(short_data_vol, constant_data_vol, zero_volume_data):
    window = 50
    indicator = RelativeVolumeIndicator(window=window)
    result_short = indicator.calculate(short_data_vol.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    assert result_short[indicator.name].iloc[0] == 1.0
    result_const = indicator.calculate(constant_data_vol.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(1.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)
    result_zero_vol = indicator.calculate(zero_volume_data.copy())
    assert indicator.name in result_zero_vol.columns
    assert_series_equal(result_zero_vol[indicator.name].iloc[window:], pd.Series(1.0, index=result_zero_vol.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)