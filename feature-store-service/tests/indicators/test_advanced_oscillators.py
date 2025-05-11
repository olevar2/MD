"""
Tests for Advanced Oscillator Indicators.
"""
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal
import ta
from feature_store_service.indicators.advanced_oscillators import AwesomeOscillator as AwesomeOscillatorIndicator, AcceleratorOscillator as AcceleratorOscillatorIndicator, UltimateOscillatorIndicator as UltimateOscillator, DeMarker as DeMarkerIndicator, TRIXIndicatorImpl as TRIX, KSTIndicatorImpl as KST, ElderForceIndex as ElderForceIndexIndicator, RelativeVigorIndex as RVIIndicator, FisherTransform as FisherTransformIndicator, CoppockCurveIndicatorImpl as CoppockCurve, ChandeMomentumOscillator as CMOIndicator

@pytest.fixture
def sample_data_osc() -> pd.DataFrame:
    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    np.random.seed(42)
    base = 100 + np.cumsum(np.random.randn(periods) * 1.5)
    close = base + np.random.randn(periods) * 0.5
    high = close + np.abs(np.random.randn(periods)) * 2
    low = close - np.abs(np.random.randn(periods)) * 2
    open = close.shift(1).fillna(close.iloc[0])
    volume = np.random.randint(100, 1000, size=periods).astype(float)
    data = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=index)
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    data.loc[data.index[10:15], 'close'] = np.nan
    data.loc[data.index[20:25], 'high'] = np.nan
    data.loc[data.index[30:35], 'low'] = np.nan
    data.loc[data.index[40:45], 'volume'] = np.nan
    return data

@pytest.fixture
def short_data_osc(sample_data_osc) -> pd.DataFrame:
    return sample_data_osc.head(20)

@pytest.fixture
def constant_data_osc() -> pd.DataFrame:
    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.0, 'volume': 100.0}, index=index)
    return data

def test_awesome_oscillator_calculation(sample_data_osc):
    fast = 5
    slow = 34
    indicator = AwesomeOscillatorIndicator(fast_window=fast, slow_window=slow)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.momentum.awesome_oscillator(sample_data_osc['high'], sample_data_osc['low'], window1=fast, window2=slow, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_awesome_oscillator_edge_cases(short_data_osc, constant_data_osc):
    fast = 5
    slow = 34
    indicator = AwesomeOscillatorIndicator(fast_window=fast, slow_window=slow)
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() == 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[slow - 1:], pd.Series(0.0, index=result_const.index[slow - 1:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_accelerator_oscillator_calculation(sample_data_osc):
    fast_ao = 5
    slow_ao = 34
    sma_win = 5
    indicator = AcceleratorOscillatorIndicator(fast_ao_window=fast_ao, slow_ao_window=slow_ao, sma_window=sma_win)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    ao = ta.momentum.awesome_oscillator(sample_data_osc['high'], sample_data_osc['low'], window1=fast_ao, window2=slow_ao, fillna=True)
    sma_ao = ta.trend.sma_indicator(ao, window=sma_win, fillna=True)
    expected_series = ao - sma_ao
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_accelerator_oscillator_edge_cases(short_data_osc, constant_data_osc):
    fast_ao = 5
    slow_ao = 34
    sma_win = 5
    indicator = AcceleratorOscillatorIndicator(fast_ao_window=fast_ao, slow_ao_window=slow_ao, sma_window=sma_win)
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() == 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    min_periods = slow_ao + sma_win - 2
    assert_series_equal(result_const[indicator.name].iloc[min_periods:], pd.Series(0.0, index=result_const.index[min_periods:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_ultimate_oscillator_calculation(sample_data_osc):
    s, m, l = (7, 14, 28)
    ws, wm, wl = (4.0, 2.0, 1.0)
    indicator = UltimateOscillator(short_window=s, medium_window=m, long_window=l, short_weight=ws, medium_weight=wm, long_weight=wl)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.momentum.ultimate_oscillator(sample_data_osc['high'], sample_data_osc['low'], sample_data_osc['close'], window1=s, window2=m, window3=l, weight1=ws, weight2=wm, weight3=wl, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_ultimate_oscillator_edge_cases(short_data_osc, constant_data_osc):
    s, m, l = (7, 14, 28)
    ws, wm, wl = (4.0, 2.0, 1.0)
    indicator = UltimateOscillator(short_window=s, medium_window=m, long_window=l, short_weight=ws, medium_weight=wm, long_weight=wl)
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() == 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[l:], pd.Series(50.0, index=result_const.index[l:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_demarker_calculation(sample_data_osc):
    window = 14
    indicator = DeMarkerIndicator(window=window)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert result_df[col_name].iloc[0] == 50
    assert result_df[col_name].dropna().min() > -10
    assert result_df[col_name].dropna().max() < 110

def test_demarker_edge_cases(short_data_osc, constant_data_osc):
    window = 14
    indicator = DeMarkerIndicator(window=window)
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    assert result_short[indicator.name].iloc[0] == 50
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(50.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_trix_calculation(sample_data_osc):
    window = 14
    indicator = TRIX(window=window, column='close')
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.trend.trix(sample_data_osc['close'], window=window, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_trix_edge_cases(short_data_osc, constant_data_osc):
    window = 14
    indicator = TRIX(window=window, column='close')
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() == 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    min_periods = 3 * (window - 1) + 1
    assert_series_equal(result_const[indicator.name].iloc[min_periods:], pd.Series(0.0, index=result_const.index[min_periods:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_kst_calculation(sample_data_osc):
    indicator = KST()
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    signal_name = indicator.signal_name
    assert col_name in result_df.columns
    assert signal_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not result_df[signal_name].isnull().all()
    expected_kst = ta.trend.kst(sample_data_osc['close'], fillna=True)
    expected_sig = ta.trend.kst_sig(sample_data_osc['close'], fillna=True)
    expected_kst.name = col_name
    expected_sig.name = signal_name
    assert_series_equal(result_df[col_name].dropna(), expected_kst.dropna(), check_dtype=False, atol=1e-06)
    assert_series_equal(result_df[signal_name].dropna(), expected_sig.dropna(), check_dtype=False, atol=1e-06)

def test_kst_edge_cases(short_data_osc, constant_data_osc):
    indicator = KST()
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert indicator.signal_name in result_short.columns
    assert result_short[indicator.name].count() == 0
    assert result_short[indicator.signal_name].count() == 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert indicator.signal_name in result_const.columns
    roc4_n = 30
    sma4 = 15
    sign_n = 9
    min_periods_kst = roc4_n + sma4 - 1
    min_periods_sig = min_periods_kst + sign_n - 1
    assert_series_equal(result_const[indicator.name].iloc[min_periods_kst:], pd.Series(0.0, index=result_const.index[min_periods_kst:], name=indicator.name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.signal_name].iloc[min_periods_sig:], pd.Series(0.0, index=result_const.index[min_periods_sig:], name=indicator.signal_name), check_dtype=False, atol=1e-06)

def test_efi_calculation(sample_data_osc):
    window = 13
    indicator = ElderForceIndexIndicator(window=window)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.volume.force_index(sample_data_osc['close'], sample_data_osc['volume'], window=window, fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_efi_edge_cases(short_data_osc, constant_data_osc):
    window = 13
    indicator = ElderForceIndexIndicator(window=window)
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_rvi_calculation(sample_data_osc):
    window = 10
    signal = 4
    indicator = RVIIndicator(window=window, signal_window=signal)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    signal_name = indicator.signal_name
    assert col_name in result_df.columns
    assert signal_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not result_df[signal_name].isnull().all()
    assert not pd.isna(result_df[col_name].iloc[0])
    assert not pd.isna(result_df[signal_name].iloc[0])

def test_rvi_edge_cases(short_data_osc, constant_data_osc):
    window = 10
    signal = 4
    indicator = RVIIndicator(window=window, signal_window=signal)
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert indicator.signal_name in result_short.columns
    assert result_short[indicator.name].count() > 0
    assert result_short[indicator.signal_name].count() > 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert indicator.signal_name in result_const.columns
    min_periods_rvi = window + 3
    min_periods_sig = min_periods_rvi + 3
    assert_series_equal(result_const[indicator.name].iloc[min_periods_rvi:], pd.Series(0.0, index=result_const.index[min_periods_rvi:], name=indicator.name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.signal_name].iloc[min_periods_sig:], pd.Series(0.0, index=result_const.index[min_periods_sig:], name=indicator.signal_name), check_dtype=False, atol=1e-06)

def test_fisher_transform_calculation(sample_data_osc):
    window = 9
    indicator = FisherTransformIndicator(window=window)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    signal_name = indicator.signal_name
    assert col_name in result_df.columns
    assert signal_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not result_df[signal_name].isnull().all()
    assert not pd.isna(result_df[col_name].iloc[0])
    assert not pd.isna(result_df[signal_name].iloc[0])
    assert (result_df[signal_name].dropna() - result_df[col_name].shift(1).dropna()).abs().mean() < 1.0

def test_fisher_transform_edge_cases(short_data_osc, constant_data_osc):
    window = 9
    indicator = FisherTransformIndicator(window=window)
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert indicator.signal_name in result_short.columns
    assert result_short[indicator.name].count() > 0
    assert result_short[indicator.signal_name].count() > 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert indicator.signal_name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)
    assert_series_equal(result_const[indicator.signal_name].iloc[window + 1:], pd.Series(0.0, index=result_const.index[window + 1:], name=indicator.signal_name), check_dtype=False, atol=1e-06)

def test_coppock_curve_calculation(sample_data_osc):
    indicator = CoppockCurve()
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    expected_series = ta.trend.coppock_curve(sample_data_osc['close'], fillna=True)
    expected_series.name = col_name
    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-06)

def test_coppock_curve_edge_cases(short_data_osc, constant_data_osc):
    indicator = CoppockCurve()
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() == 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    w = 10
    s = 11
    l = 14
    min_periods = l + w - 1
    assert_series_equal(result_const[indicator.name].iloc[min_periods:], pd.Series(0.0, index=result_const.index[min_periods:], name=indicator.name), check_dtype=False, atol=1e-06)

def test_cmo_calculation(sample_data_osc):
    window = 9
    indicator = CMOIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not pd.isna(result_df[col_name].iloc[0])
    assert result_df[col_name].dropna().min() >= -100
    assert result_df[col_name].dropna().max() <= 100

def test_cmo_edge_cases(short_data_osc, constant_data_osc):
    window = 9
    indicator = CMOIndicator(window=window, column='close')
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() > 0
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-06)