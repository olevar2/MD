"""
Tests for Advanced Oscillator Indicators.
"""
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal
import ta # Import the reference library

# Import indicators to test
from feature_store_service.indicators.advanced_oscillators import (
    AwesomeOscillator as AwesomeOscillatorIndicator, # Renamed
    AcceleratorOscillator as AcceleratorOscillatorIndicator, # Renamed
    UltimateOscillatorIndicator as UltimateOscillator, # Already distinct
    DeMarker as DeMarkerIndicator, # Renamed
    TRIXIndicatorImpl as TRIX, # Already distinct
    KSTIndicatorImpl as KST, # Already distinct
    ElderForceIndex as ElderForceIndexIndicator, # Renamed
    RelativeVigorIndex as RVIIndicator, # Renamed
    FisherTransform as FisherTransformIndicator, # Renamed
    CoppockCurveIndicatorImpl as CoppockCurve, # Already distinct
    ChandeMomentumOscillator as CMOIndicator # Renamed
)

# Sample data for testing (more volatile for oscillators)
@pytest.fixture
def sample_data_osc() -> pd.DataFrame:
    periods = 100
    index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    # More volatile data
    np.random.seed(42) # for reproducibility
    base = 100 + np.cumsum(np.random.randn(periods) * 1.5)
    close = base + np.random.randn(periods) * 0.5
    high = close + np.abs(np.random.randn(periods)) * 2
    low = close - np.abs(np.random.randn(periods)) * 2
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = np.random.randint(100, 1000, size=periods).astype(float)
    data = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=index)
    # Ensure high >= close, low <= close etc.
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    # Add some NaNs
    data.loc[data.index[10:15], 'close'] = np.nan
    data.loc[data.index[20:25], 'high'] = np.nan
    data.loc[data.index[30:35], 'low'] = np.nan
    data.loc[data.index[40:45], 'volume'] = np.nan
    return data

@pytest.fixture
def short_data_osc(sample_data_osc) -> pd.DataFrame:
    return sample_data_osc.head(20) # Shorter than many windows

@pytest.fixture
def constant_data_osc() -> pd.DataFrame:
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

# --- Test AwesomeOscillator ---
def test_awesome_oscillator_calculation(sample_data_osc):
    fast = 5
    slow = 34
    indicator = AwesomeOscillatorIndicator(fast_window=fast, slow_window=slow)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Reference calculation
    expected_series = ta.momentum.awesome_oscillator(sample_data_osc['high'], sample_data_osc['low'], window1=fast, window2=slow, fillna=True)
    expected_series.name = col_name

    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-6)

def test_awesome_oscillator_edge_cases(short_data_osc, constant_data_osc):
    fast = 5
    slow = 34
    indicator = AwesomeOscillatorIndicator(fast_window=fast, slow_window=slow)

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    # AO needs slow window periods
    assert result_short[indicator.name].count() == 0 # Needs 34 periods, has only 20

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    # AO of constant data (midpoint = 100) should be 0 after initial period
    assert_series_equal(result_const[indicator.name].iloc[slow-1:], pd.Series(0.0, index=result_const.index[slow-1:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test AcceleratorOscillator ---
# Note: ta library doesn't have AC directly. AC = AO - SMA(AO, 5)
def test_accelerator_oscillator_calculation(sample_data_osc):
    fast_ao = 5
    slow_ao = 34
    sma_win = 5
    indicator = AcceleratorOscillatorIndicator(fast_ao_window=fast_ao, slow_ao_window=slow_ao, sma_window=sma_win)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Manual reference calculation
    ao = ta.momentum.awesome_oscillator(sample_data_osc['high'], sample_data_osc['low'], window1=fast_ao, window2=slow_ao, fillna=True)
    sma_ao = ta.trend.sma_indicator(ao, window=sma_win, fillna=True)
    expected_series = ao - sma_ao
    expected_series.name = col_name

    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-6)

def test_accelerator_oscillator_edge_cases(short_data_osc, constant_data_osc):
    fast_ao = 5
    slow_ao = 34
    sma_win = 5
    indicator = AcceleratorOscillatorIndicator(fast_ao_window=fast_ao, slow_ao_window=slow_ao, sma_window=sma_win)

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() == 0 # Needs slow_ao + sma_win - 1

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    # AO is 0, SMA(0) is 0, so AC should be 0
    min_periods = slow_ao + sma_win - 2 # Indexing starts from 0
    assert_series_equal(result_const[indicator.name].iloc[min_periods:], pd.Series(0.0, index=result_const.index[min_periods:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test UltimateOscillator ---
def test_ultimate_oscillator_calculation(sample_data_osc):
    s, m, l = 7, 14, 28
    ws, wm, wl = 4.0, 2.0, 1.0
    indicator = UltimateOscillator(
        short_window=s, medium_window=m, long_window=l,
        short_weight=ws, medium_weight=wm, long_weight=wl
    )
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Reference calculation
    expected_series = ta.momentum.ultimate_oscillator(
        sample_data_osc['high'], sample_data_osc['low'], sample_data_osc['close'],
        window1=s, window2=m, window3=l, weight1=ws, weight2=wm, weight3=wl, fillna=True
    )
    expected_series.name = col_name

    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-6)

def test_ultimate_oscillator_edge_cases(short_data_osc, constant_data_osc):
    s, m, l = 7, 14, 28
    ws, wm, wl = 4.0, 2.0, 1.0
    indicator = UltimateOscillator(
        short_window=s, medium_window=m, long_window=l,
        short_weight=ws, medium_weight=wm, long_weight=wl
    )

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert result_short[indicator.name].count() == 0 # Needs long window (28)

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    # BP = close - min(low, prev_close) = 100 - min(99.5, 100) = 0.5
    # TR = max(high, prev_close) - min(low, prev_close) = max(100.5, 100) - min(99.5, 100) = 100.5 - 99.5 = 1.0
    # AvgBP / AvgTR = (0.5 * n) / (1.0 * n) = 0.5
    # UO = 100 * [(4*0.5) + (2*0.5) + (1*0.5)] / (4+2+1) = 100 * [3.5] / 7 = 50.0
    assert_series_equal(result_const[indicator.name].iloc[l:], pd.Series(50.0, index=result_const.index[l:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test DeMarker ---
# Note: ta library doesn't have DeMarker. Test based on logic.
def test_demarker_calculation(sample_data_osc):
    window = 14
    indicator = DeMarkerIndicator(window=window)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    # Implementation fills initial NaNs with 50
    assert result_df[col_name].iloc[0] == 50

    # Basic check: Should be mostly between 0 and 100
    # Can slightly exceed due to SMA smoothing at start/end
    assert result_df[col_name].dropna().min() > -10
    assert result_df[col_name].dropna().max() < 110

    # TODO: Add comparison with known values if possible

def test_demarker_edge_cases(short_data_osc, constant_data_osc):
    window = 14
    indicator = DeMarkerIndicator(window=window)

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    # Needs window+1 periods for SMA
    assert result_short[indicator.name].count() > 0 # Should produce some values
    assert result_short[indicator.name].iloc[0] == 50 # Check fillna

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    # DeMax = max(0, high - prev_high) = max(0, 100.5 - 100.5) = 0
    # DeMin = max(0, prev_low - low) = max(0, 99.5 - 99.5) = 0
    # SMA(DeMax) = 0, SMA(DeMin) = 0
    # DeM = SMA(DeMax) / (SMA(DeMax) + SMA(DeMin)) = 0 / (0 + 0) -> NaN, but implementation handles division by zero
    # Expect 50 due to division by zero handling or initial fillna
    # Check after window period
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(50.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test TRIX ---
def test_trix_calculation(sample_data_osc):
    window = 14
    indicator = TRIX(window=window, column='close')
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Reference calculation
    expected_series = ta.trend.trix(sample_data_osc['close'], window=window, fillna=True)
    expected_series.name = col_name

    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-6)

def test_trix_edge_cases(short_data_osc, constant_data_osc):
    window = 14
    indicator = TRIX(window=window, column='close')

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    # TRIX needs 3*window periods effectively
    assert result_short[indicator.name].count() == 0

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    # EMA(const) = const. ROC(const) = 0.
    min_periods = 3 * (window - 1) + 1 # For triple EMA
    assert_series_equal(result_const[indicator.name].iloc[min_periods:], pd.Series(0.0, index=result_const.index[min_periods:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test KST ---
def test_kst_calculation(sample_data_osc):
    indicator = KST() # Use defaults
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name
    signal_name = indicator.signal_name

    assert col_name in result_df.columns
    assert signal_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    assert not result_df[signal_name].isnull().all()

    # Reference calculation
    expected_kst = ta.trend.kst(sample_data_osc['close'], fillna=True)
    expected_sig = ta.trend.kst_sig(sample_data_osc['close'], fillna=True)
    expected_kst.name = col_name
    expected_sig.name = signal_name

    assert_series_equal(result_df[col_name].dropna(), expected_kst.dropna(), check_dtype=False, atol=1e-6)
    assert_series_equal(result_df[signal_name].dropna(), expected_sig.dropna(), check_dtype=False, atol=1e-6)

def test_kst_edge_cases(short_data_osc, constant_data_osc):
    indicator = KST() # Use defaults

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert indicator.signal_name in result_short.columns
    # KST needs roc4_n + sma4 - 1 = 30 + 15 - 1 = 44 periods
    assert result_short[indicator.name].count() == 0
    assert result_short[indicator.signal_name].count() == 0

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert indicator.signal_name in result_const.columns
    # ROC(const) = 0. SMA(0) = 0. KST = 0. Signal = 0.
    roc4_n = 30; sma4 = 15; sign_n = 9
    min_periods_kst = roc4_n + sma4 - 1
    min_periods_sig = min_periods_kst + sign_n - 1
    assert_series_equal(result_const[indicator.name].iloc[min_periods_kst:], pd.Series(0.0, index=result_const.index[min_periods_kst:], name=indicator.name), check_dtype=False, atol=1e-6)
    assert_series_equal(result_const[indicator.signal_name].iloc[min_periods_sig:], pd.Series(0.0, index=result_const.index[min_periods_sig:], name=indicator.signal_name), check_dtype=False, atol=1e-6)


# --- Test ElderForceIndex ---
def test_efi_calculation(sample_data_osc):
    window = 13
    indicator = ElderForceIndexIndicator(window=window)
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Reference calculation
    expected_series = ta.volume.force_index(sample_data_osc['close'], sample_data_osc['volume'], window=window, fillna=True)
    expected_series.name = col_name

    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-6)

def test_efi_edge_cases(short_data_osc, constant_data_osc):
    window = 13
    indicator = ElderForceIndexIndicator(window=window)

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    # Needs window periods for EMA
    assert result_short[indicator.name].count() > 0 # Should produce some values

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    # FI = (close - prev_close) * volume = (100 - 100) * 100 = 0
    # EMA(0) = 0
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test RVI ---
# Note: ta library doesn't have RVI. Test based on logic.
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
    # Implementation fills NaNs
    assert not pd.isna(result_df[col_name].iloc[0])
    assert not pd.isna(result_df[signal_name].iloc[0])

    # Basic check: Should oscillate, often between 0 and 100 but not strictly bounded

    # TODO: Add comparison with known values if possible

def test_rvi_edge_cases(short_data_osc, constant_data_osc):
    window = 10
    signal = 4
    indicator = RVIIndicator(window=window, signal_window=signal)

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert indicator.signal_name in result_short.columns
    # Needs window + 3 periods for SWMA
    assert result_short[indicator.name].count() > 0
    assert result_short[indicator.signal_name].count() > 0

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert indicator.signal_name in result_const.columns
    # Numerator = close - open = 100 - 100 = 0
    # Denominator = high - low = 100.5 - 99.5 = 1
    # SWMA(0) = 0, SWMA(1) = 1. RVI = 0 / 1 = 0. Signal = 0.
    min_periods_rvi = window + 3
    min_periods_sig = min_periods_rvi + 3
    assert_series_equal(result_const[indicator.name].iloc[min_periods_rvi:], pd.Series(0.0, index=result_const.index[min_periods_rvi:], name=indicator.name), check_dtype=False, atol=1e-6)
    assert_series_equal(result_const[indicator.signal_name].iloc[min_periods_sig:], pd.Series(0.0, index=result_const.index[min_periods_sig:], name=indicator.signal_name), check_dtype=False, atol=1e-6)


# --- Test FisherTransform ---
# Note: ta library doesn't have Fisher. Test based on logic.
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
    # Implementation fills NaNs
    assert not pd.isna(result_df[col_name].iloc[0])
    assert not pd.isna(result_df[signal_name].iloc[0])

    # Basic check: Fisher values can be large, signal follows Fisher
    assert (result_df[signal_name].dropna() - result_df[col_name].shift(1).dropna()).abs().mean() < 1.0 # Signal should lag Fisher

    # TODO: Add comparison with known values if possible

def test_fisher_transform_edge_cases(short_data_osc, constant_data_osc):
    window = 9
    indicator = FisherTransformIndicator(window=window)

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    assert indicator.signal_name in result_short.columns
    # Needs window periods
    assert result_short[indicator.name].count() > 0
    assert result_short[indicator.signal_name].count() > 0

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    assert indicator.signal_name in result_const.columns
    # Median price = (100.5 + 99.5) / 2 = 100
    # MaxH = 100, MinL = 100
    # Value1 = (100 - 100) / (100 - 100) -> NaN handled as 0 in implementation?
    # Fish = 0.5 * log((1+Value1)/(1-Value1)) -> 0.5 * log(1/1) = 0
    # Check after window period
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-6)
    assert_series_equal(result_const[indicator.signal_name].iloc[window+1:], pd.Series(0.0, index=result_const.index[window+1:], name=indicator.signal_name), check_dtype=False, atol=1e-6)


# --- Test CoppockCurve ---
def test_coppock_curve_calculation(sample_data_osc):
    indicator = CoppockCurve() # Use defaults
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()

    # Reference calculation
    expected_series = ta.trend.coppock_curve(sample_data_osc['close'], fillna=True)
    expected_series.name = col_name

    assert_series_equal(result_df[col_name].dropna(), expected_series.dropna(), check_dtype=False, atol=1e-6)

def test_coppock_curve_edge_cases(short_data_osc, constant_data_osc):
    indicator = CoppockCurve() # Use defaults: w=10, s=11, l=14

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    # Needs l + w - 1 = 14 + 10 - 1 = 23 periods
    assert result_short[indicator.name].count() == 0

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    # ROC(const) = 0. WMA(0) = 0. Coppock = 0.
    w=10; s=11; l=14
    min_periods = l + w - 1
    assert_series_equal(result_const[indicator.name].iloc[min_periods:], pd.Series(0.0, index=result_const.index[min_periods:], name=indicator.name), check_dtype=False, atol=1e-6)


# --- Test CMO ---
# Note: ta library doesn't have CMO. Test based on logic.
def test_cmo_calculation(sample_data_osc):
    window = 9
    indicator = CMOIndicator(window=window, column='close')
    result_df = indicator.calculate(sample_data_osc.copy())
    col_name = indicator.name

    assert col_name in result_df.columns
    assert not result_df[col_name].isnull().all()
    # Implementation fills NaNs
    assert not pd.isna(result_df[col_name].iloc[0])

    # Basic check: Should be between -100 and 100
    assert result_df[col_name].dropna().min() >= -100
    assert result_df[col_name].dropna().max() <= 100

    # TODO: Add comparison with known values if possible

def test_cmo_edge_cases(short_data_osc, constant_data_osc):
    window = 9
    indicator = CMOIndicator(window=window, column='close')

    # Short data
    result_short = indicator.calculate(short_data_osc.copy())
    assert indicator.name in result_short.columns
    # Needs window periods
    assert result_short[indicator.name].count() > 0

    # Constant data
    result_const = indicator.calculate(constant_data_osc.copy())
    assert indicator.name in result_const.columns
    # diff = 0. sum_pos = 0, sum_neg = 0.
    # CMO = 100 * (0 - 0) / (0 + 0) -> NaN, handled as 0 in implementation
    assert_series_equal(result_const[indicator.name].iloc[window:], pd.Series(0.0, index=result_const.index[window:], name=indicator.name), check_dtype=False, atol=1e-6)
