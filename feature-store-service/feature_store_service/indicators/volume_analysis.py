"""
Volume Analysis Indicators Module.

This module contains implementations for advanced volume-based technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from feature_store_service.indicators.base_indicator import BaseIndicator

class VWAPBands(BaseIndicator):
    """
    Volume Weighted Average Price (VWAP) Bands Indicator.

    Calculates the VWAP and standard deviation bands around it.

    Parameters:
    -----------
    period : int, optional
        The lookback period for VWAP calculation (default: 14).
    std_dev_multiplier : float, optional
        The multiplier for the standard deviation bands (default: 1.0).
    source_col : str, optional
        The source column to use for typical price calculation if 'high', 'low', 'close' aren't all present.
        Not typically used if OHLCV data is standard. (default: 'close')

    Attributes:
    -----------
    period : int
        The lookback period.
    std_dev_multiplier : float
        The standard deviation multiplier.
    source_col : str
        Source column name.

    Methods:
    --------
    calculate(data: pd.DataFrame) -> pd.DataFrame:
        Calculate the VWAP Bands values.
    _validate_params():
        Validate the indicator parameters.
    """
    category = "volume"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 14},
        "std_dev_multiplier": {"type": "float", "min": 0.1, "max": 5.0, "default": 1.0},
        "source_col": {"type": "str", "options": ["open", "high", "low", "close"], "default": "close"} # Less common, but for flexibility
    }

    def __init__(self, period: int = 14, std_dev_multiplier: float = 1.0, source_col: str = 'close', **kwargs):
        """
        Initialize the VWAPBands indicator.

        Args:
            period (int, optional): Lookback period. Defaults to 14.
            std_dev_multiplier (float, optional): Standard deviation multiplier. Defaults to 1.0.
            source_col (str, optional): Source column if typical price cannot be calculated. Defaults to 'close'.
        """
        # Although BaseIndicator has no __init__, we store params for clarity and potential future use.
        # The name isn't strictly needed by BaseIndicator but good practice.
        self.name = f"VWAPBands_{period}_{std_dev_multiplier}"
        self.period = period
        self.std_dev_multiplier = std_dev_multiplier
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
        """Validate the parameters."""
        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")
        if not isinstance(self.std_dev_multiplier, (float, int)) or self.std_dev_multiplier <= 0:
            raise ValueError(f"Standard deviation multiplier must be positive, got {self.std_dev_multiplier}")
        # source_col validation might depend on expected data columns

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the VWAP Bands.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'high', 'low', 'close', 'volume' columns.

        Returns:
            pd.DataFrame: DataFrame with 'VWAP_{period}', 'VWAP_UpperBand_{period}',
                          and 'VWAP_LowerBand_{period}' columns added.
        """
        required_cols = ['high', 'low', 'close', 'volume']
        self.validate_input(data, required_cols)

        typical_price = (data['high'] + data['low'] + data['close']) / 3
        tpv = typical_price * data['volume']

        # Calculate cumulative sums using rolling window
        cumulative_tpv = tpv.rolling(window=self.period, min_periods=self.period).sum()
        cumulative_volume = data['volume'].rolling(window=self.period, min_periods=self.period).sum()

        # Calculate VWAP, handle potential division by zero
        vwap = cumulative_tpv / cumulative_volume
        vwap.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero if volume is zero

        # Calculate standard deviation of typical price over the period
        typical_price_std_dev = typical_price.rolling(window=self.period, min_periods=self.period).std()

        # Calculate Bands
        upper_band = vwap + (self.std_dev_multiplier * typical_price_std_dev)
        lower_band = vwap - (self.std_dev_multiplier * typical_price_std_dev)

        # Prepare output DataFrame
        output = pd.DataFrame(index=data.index)
        output[f'VWAP_{self.period}'] = vwap
        output[f'VWAP_UpperBand_{self.period}'] = upper_band
        output[f'VWAP_LowerBand_{self.period}'] = lower_band

        return output

class MarketFacilitationIndex(BaseIndicator):
    \"\"\"
    Market Facilitation Index (BW MFI) Indicator.

    Measures the efficiency of price movement by relating price change to volume.
    BW MFI = (High - Low) / Volume

    Parameters:
    -----------
    None

    Attributes:
    -----------
    None

    Methods:
    --------
    calculate(data: pd.DataFrame) -> pd.DataFrame:
        Calculate the Market Facilitation Index values.
    _validate_params():
        Validate the indicator parameters (none needed).
    \"\"\"
    category = \"volume\"
    default_params = {} # No parameters

    def __init__(self, **kwargs):
        \"\"\"Initialize the MarketFacilitationIndex indicator.\"\"\"
        self.name = "MarketFacilitationIndex"
        # No parameters to validate or store
        self._validate_params()

    def _validate_params(self):
        \"\"\"Validate the parameters (none needed).\"\"\"
        pass # No parameters to validate

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\""
        Calculate the Market Facilitation Index.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'high', 'low', 'volume' columns.

        Returns:
            pd.DataFrame: DataFrame with 'MFI_BW' column added.
        \"\""
        required_cols = ['high', 'low', 'volume']
        self.validate_input(data, required_cols)

        # Calculate MFI
        mfi = (data['high'] - data['low']) / data['volume']

        # Handle potential division by zero or zero range
        mfi.replace([np.inf, -np.inf], 0, inplace=True) # If volume is 0, MFI is arguably 0
        mfi.fillna(0, inplace=True) # If high == low, MFI is 0

        # Prepare output DataFrame
        output = pd.DataFrame(index=data.index)
        output['MFI_BW'] = mfi

        return output

class VolumeZoneOscillator(BaseIndicator):
    \"\"\"
    Volume Zone Oscillator (VZO) Indicator.

    Measures buying and selling pressure relative to price zones.

    Parameters:
    -----------\n    period : int, optional
        The lookback period for the EMA of the relationship between closing price and volume (default: 14).
    ema_period : int, optional
        The lookback period for the EMA of the VZO percentage (default: 60). Used for smoothing the final VZO.

    Attributes:
    -----------\n    period : int
        The primary lookback period.
    ema_period : int
        The smoothing EMA period.

    Methods:
    --------\n    calculate(data: pd.DataFrame) -> pd.DataFrame:
        Calculate the Volume Zone Oscillator values.
    _validate_params():
        Validate the indicator parameters.
    \"\"\"
    category = "volume"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 14},
        "ema_period": {"type": "int", "min": 2, "max": 200, "default": 60}
    }

    def __init__(self, period: int = 14, ema_period: int = 60, **kwargs):
        \"\""
        Initialize the VolumeZoneOscillator indicator.

        Args:\n            period (int, optional): Lookback period for price/volume relationship. Defaults to 14.
            ema_period (int, optional): Lookback period for VZO smoothing. Defaults to 60.
        \"\""
        self.name = f"VZO_{period}_{ema_period}"
        self.period = period
        self.ema_period = ema_period
        self._validate_params()

    def _validate_params(self):
        \"\"\"Validate the parameters.\"\"\"
        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")
        if not isinstance(self.ema_period, int) or self.ema_period <= 1:
            raise ValueError(f"EMA period must be an integer greater than 1, got {self.ema_period}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\""
        Calculate the Volume Zone Oscillator.

        Args:\n            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'close', 'volume' columns.

        Returns:\n            pd.DataFrame: DataFrame with 'VZO_{period}_{ema_period}' column added.
        \"\""
        required_cols = ['close', 'volume']
        self.validate_input(data, required_cols)

        # Calculate R = sign(Close - Close.shift(1)) * Volume
        r = np.sign(data['close'].diff(1)) * data['volume']
        r.iloc[0] = 0 # First value is NaN, set to 0

        # Calculate EMA of R over 'period'
        ema_r = r.ewm(span=self.period, adjust=False).mean()

        # Calculate EMA of Volume over 'period'
        ema_volume = data['volume'].ewm(span=self.period, adjust=False).mean()

        # Calculate VZO percentage = 100 * (EMA of R / EMA of Volume)
        # Handle potential division by zero
        vzo_percent = 100 * (ema_r / ema_volume.replace(0, np.nan)) # Replace 0 volume with NaN to avoid division error, then fill
        vzo_percent.fillna(method='ffill', inplace=True) # Forward fill NaNs resulting from zero volume
        vzo_percent.fillna(0, inplace=True) # Fill any remaining NaNs at the beginning

        # Calculate EMA of VZO percentage over 'ema_period'
        vzo_smoothed = vzo_percent.ewm(span=self.ema_period, adjust=False).mean()

        # Prepare output DataFrame
        output = pd.DataFrame(index=data.index)
        output[f'VZO_{self.period}_{self.ema_period}'] = vzo_smoothed

        return output

class EaseOfMovement(BaseIndicator):
    \"\""
    Ease of Movement (EOM) Indicator.

    Relates price change to volume, highlighting periods where prices move easily on low volume.

    Parameters:
    -----------\n    period : int, optional
        The lookback period for the moving average of the EOM value (default: 14).

    Attributes:
    -----------\n    period : int
        The lookback period for the moving average.

    Methods:
    --------\n    calculate(data: pd.DataFrame) -> pd.DataFrame:
        Calculate the Ease of Movement values.
    _validate_params():
        Validate the indicator parameters.
    \"\"\"
    category = "volume"
    default_params = {
        "period": {"type": "int", "min": 1, "max": 200, "default": 14}
    }

    def __init__(self, period: int = 14, **kwargs):
        \"\""
        Initialize the EaseOfMovement indicator.

        Args:\n            period (int, optional): Lookback period for the moving average. Defaults to 14.
        \"\""
        self.name = f"EOM_{period}"
        self.period = period
        self._validate_params()

    def _validate_params(self):
        \"\"\"Validate the parameters.\"\"\"
        if not isinstance(self.period, int) or self.period < 1:
            raise ValueError(f"Period must be a positive integer, got {self.period}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\""
        Calculate the Ease of Movement.

        Args:\n            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'high', 'low', 'volume' columns.

        Returns:\n            pd.DataFrame: DataFrame with 'EOM_{period}' column added.
        \"\""
        required_cols = ['high', 'low', 'volume']
        self.validate_input(data, required_cols)

        # Calculate Midpoint Move
        midpoint = (data['high'] + data['low']) / 2
        midpoint_move = midpoint.diff(1)
        midpoint_move.iloc[0] = 0 # First value is NaN

        # Calculate Box Ratio = (Volume / Scale Factor) / (High - Low)
        # Scale factor to make EOM values reasonable, often 100,000,000 or similar
        # We need to handle High == Low and Volume == 0 cases.
        scale_factor = 100_000_000 # Common scale factor
        high_low_range = data['high'] - data['low']
        # Avoid division by zero if high == low or volume == 0
        box_ratio_denominator = high_low_range.replace(0, np.nan) # Replace 0 range with NaN
        box_ratio = (data['volume'] / scale_factor) / box_ratio_denominator
        box_ratio.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero volume
        box_ratio.fillna(method='ffill', inplace=True) # Fill NaNs, maybe from 0 range or 0 volume
        box_ratio.fillna(1, inplace=True) # If still NaN (e.g., start), use 1 to avoid issues? Or 0? Let's use 1.

        # Calculate 1-Period EOM = Midpoint Move / Box Ratio
        eom_1_period = midpoint_move / box_ratio
        eom_1_period.fillna(0, inplace=True) # Fill NaNs at the beginning

        # Calculate EOM = Moving Average of 1-Period EOM
        eom = eom_1_period.rolling(window=self.period, min_periods=self.period).mean()

        # Prepare output DataFrame
        output = pd.DataFrame(index=data.index)
        output[f'EOM_{self.period}'] = eom

        return output

class NVIAndPVI(BaseIndicator):
    \"\""
    Negative Volume Index (NVI) and Positive Volume Index (PVI) Indicator.

    NVI assumes smart money is active on low volume days.
    PVI assumes crowd behavior influences high volume days.

    Parameters:
    -----------\n    None

    Attributes:
    -----------\n    None

    Methods:
    --------\n    calculate(data: pd.DataFrame) -> pd.DataFrame:
        Calculate the NVI and PVI values.
    _validate_params():
        Validate the indicator parameters (none needed).
    \"\""
    category = "volume"
    default_params = {} # No parameters

    def __init__(self, **kwargs):
        \"\"\"Initialize the NVIAndPVI indicator.\"\"\"
        self.name = "NVI_PVI"
        self._validate_params()

    def _validate_params(self):
        \"\"\"Validate the parameters (none needed).\"\"\"
        pass # No parameters

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\""
        Calculate the Negative Volume Index (NVI) and Positive Volume Index (PVI).

        Args:\n            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'close', 'volume' columns.

        Returns:\n            pd.DataFrame: DataFrame with 'NVI' and 'PVI' columns added.
        \"\""
        required_cols = ['close', 'volume']
        self.validate_input(data, required_cols)

        close_change_pct = data['close'].pct_change()
        volume_change = data['volume'].diff()

        nvi = pd.Series(index=data.index, dtype=float)
        pvi = pd.Series(index=data.index, dtype=float)

        # Initialize NVI and PVI with a starting value (e.g., 1000)
        nvi.iloc[0] = 1000.0
        pvi.iloc[0] = 1000.0

        for i in range(1, len(data)):
            if volume_change.iloc[i] < 0: # Volume decreased
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + close_change_pct.iloc[i])
                pvi.iloc[i] = pvi.iloc[i-1] # PVI unchanged
            elif volume_change.iloc[i] > 0: # Volume increased
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + close_change_pct.iloc[i])
                nvi.iloc[i] = nvi.iloc[i-1] # NVI unchanged
            else: # Volume unchanged
                nvi.iloc[i] = nvi.iloc[i-1]
                pvi.iloc[i] = pvi.iloc[i-1]

        # Handle potential NaNs in percentage change at the start
        nvi.fillna(method='ffill', inplace=True)
        pvi.fillna(method='ffill', inplace=True)

        # Prepare output DataFrame
        output = pd.DataFrame(index=data.index)
        output['NVI'] = nvi
        output['PVI'] = pvi

        return output

class DemandIndex(BaseIndicator):
    \"\""
    Demand Index Indicator.

    Combines price and volume to measure buying and selling pressure.

    Parameters:
    -----------\n    di_period : int, optional
        The lookback period for calculating Buying Power (BP) and Selling Power (SP) moving averages (default: 10).
    ma_period : int, optional
        The lookback period for the moving average of the Demand Index ratio (default: 3).

    Attributes:
    -----------\n    di_period : int
        The primary lookback period.
    ma_period : int
        The moving average period for the final index.

    Methods:
    --------\n    calculate(data: pd.DataFrame) -> pd.DataFrame:
        Calculate the Demand Index values.
    _validate_params():
        Validate the indicator parameters.
    \"\""
    category = "volume"
    default_params = {
        "di_period": {"type": "int", "min": 2, "max": 200, "default": 10},
        "ma_period": {"type": "int", "min": 1, "max": 50, "default": 3}
    }

    def __init__(self, di_period: int = 10, ma_period: int = 3, **kwargs):
        \"\""
        Initialize the DemandIndex indicator.

        Args:\n            di_period (int, optional): Lookback period for BP/SP. Defaults to 10.
            ma_period (int, optional): Lookback period for DI ratio MA. Defaults to 3.
        \"\""
        self.name = f"DemandIndex_{di_period}_{ma_period}"
        self.di_period = di_period
        self.ma_period = ma_period
        self._validate_params()

    def _validate_params(self):
        \"\"\"Validate the parameters.\"\"\"
        if not isinstance(self.di_period, int) or self.di_period <= 1:
            raise ValueError(f"DI period must be an integer greater than 1, got {self.di_period}")
        if not isinstance(self.ma_period, int) or self.ma_period < 1:
            raise ValueError(f"MA period must be a positive integer, got {self.ma_period}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\""
        Calculate the Demand Index.

        Note: The exact formula for Demand Index can vary slightly between sources,
              especially regarding the price ratio calculation and scaling.
              This implementation follows a common interpretation.

        Args:\n            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'open', 'high', 'low', 'close', 'volume' columns.

        Returns:\n            pd.DataFrame: DataFrame with 'DemandIndex_{di_period}_{ma_period}' column added.
        \"\""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        self.validate_input(data, required_cols)

        # Price Move Ratio (simplified version)
        # Avoid division by zero if open == close
        price_ratio = (data['high'] + data['low'] + 2 * data['close']) / (data['open'] * 4).replace(0, np.nan)
        price_ratio.fillna(1, inplace=True) # If open is 0, assume ratio is 1

        # Calculate Buying Power (BP) and Selling Power (SP)
        bp = pd.Series(index=data.index, dtype=float)
        sp = pd.Series(index=data.index, dtype=float)

        if data['close'].iloc[0] >= data['open'].iloc[0]:
            bp.iloc[0] = data['volume'].iloc[0] / price_ratio.iloc[0]
            sp.iloc[0] = data['volume'].iloc[0] * price_ratio.iloc[0]
        else:
            bp.iloc[0] = data['volume'].iloc[0] * price_ratio.iloc[0]
            sp.iloc[0] = data['volume'].iloc[0] / price_ratio.iloc[0]

        for i in range(1, len(data)):
            if data['close'].iloc[i] >= data['open'].iloc[i]:
                bp.iloc[i] = data['volume'].iloc[i] / price_ratio.iloc[i]
                sp.iloc[i] = data['volume'].iloc[i] * price_ratio.iloc[i]
            else:
                bp.iloc[i] = data['volume'].iloc[i] * price_ratio.iloc[i]
                sp.iloc[i] = data['volume'].iloc[i] / price_ratio.iloc[i]

        # Smooth BP and SP
        bp_ma = bp.rolling(window=self.di_period, min_periods=self.di_period).mean()
        sp_ma = sp.rolling(window=self.di_period, min_periods=self.di_period).mean()

        # Calculate Demand Index (DI) Ratio
        # Avoid division by zero if bp_ma + sp_ma is zero
        di_ratio_denominator = (bp_ma + sp_ma).replace(0, np.nan)
        di_ratio = (bp_ma - sp_ma) / di_ratio_denominator
        di_ratio.fillna(0, inplace=True) # If sum is 0, ratio is 0

        # Smooth DI Ratio
        demand_index = di_ratio.rolling(window=self.ma_period, min_periods=self.ma_period).mean()

        # Prepare output DataFrame
        output = pd.DataFrame(index=data.index)
        output[f'DemandIndex_{self.di_period}_{self.ma_period}'] = demand_index

        return output

class RelativeVolume(BaseIndicator):
    \"\""
    Relative Volume (RVOL) Indicator.

    Compares the current volume to the average volume over a specified period.

    Parameters:
    -----------\n    period : int, optional
        The lookback period for calculating the average volume (default: 50).

    Attributes:
    -----------\n    period : int
        The lookback period for the average volume.

    Methods:
    --------\n    calculate(data: pd.DataFrame) -> pd.DataFrame:
        Calculate the Relative Volume values.
    _validate_params():
        Validate the indicator parameters.
    \"\""
    category = "volume"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 50}
    }

    def __init__(self, period: int = 50, **kwargs):
        \"\""
        Initialize the RelativeVolume indicator.

        Args:\n            period (int, optional): Lookback period for average volume. Defaults to 50.
        \"\""
        self.name = f"RVOL_{period}"
        self.period = period
        self._validate_params()

    def _validate_params(self):
        \"\"\"Validate the parameters.\"\"\"
        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\""
        Calculate the Relative Volume.

        Args:\n            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'volume' column.

        Returns:\n            pd.DataFrame: DataFrame with 'RVOL_{period}' column added.
        \"\""
        required_cols = ['volume']
        self.validate_input(data, required_cols)

        # Calculate average volume over the period
        average_volume = data['volume'].rolling(window=self.period, min_periods=self.period).mean()

        # Calculate Relative Volume = Current Volume / Average Volume
        # Avoid division by zero if average_volume is zero
        relative_volume = data['volume'] / average_volume.replace(0, np.nan)
        relative_volume.fillna(1, inplace=True) # If avg vol is 0, RVOL is arguably 1 (or undefined, 1 seems reasonable)

        # Prepare output DataFrame
        output = pd.DataFrame(index=data.index)
        output[f'RVOL_{self.period}'] = relative_volume

        return output

class MoneyFlowIndex(BaseIndicator):
    \"\""
    Money Flow Index (MFI) Indicator.

    A volume-weighted RSI that measures buying and selling pressure.

    Parameters:
    -----------\n    period : int, optional
        The lookback period for MFI calculation (default: 14).

    Attributes:
    -----------\n    period : int
        The lookback period.

    Methods:
    --------\n    calculate(data: pd.DataFrame) -> pd.DataFrame:
        Calculate the Money Flow Index values.
    _validate_params():
        Validate the indicator parameters.
    \"\""
    category = "volume" # Also considered an oscillator
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 14}
    }

    def __init__(self, period: int = 14, **kwargs):
        \"\""
        Initialize the MoneyFlowIndex indicator.

        Args:\n            period (int, optional): Lookback period. Defaults to 14.
        \"\""
        self.name = f"MFI_{period}"
        self.period = period
        self._validate_params()

    def _validate_params(self):
        \"\"\"Validate the parameters.\"\"\"
        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\""
        Calculate the Money Flow Index.

        Args:\n            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'high', 'low', 'close', 'volume' columns.

        Returns:\n            pd.DataFrame: DataFrame with 'MFI_{period}' column added.
        \"\""
        required_cols = ['high', 'low', 'close', 'volume']
        self.validate_input(data, required_cols)

        # Calculate Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        # Calculate Raw Money Flow = Typical Price * Volume
        raw_money_flow = typical_price * data['volume']

        # Calculate Positive and Negative Money Flow
        typical_price_change = typical_price.diff(1)
        positive_money_flow = raw_money_flow.where(typical_price_change > 0, 0)
        negative_money_flow = raw_money_flow.where(typical_price_change < 0, 0)

        # Calculate Money Flow Ratio = Sum of Positive MF / Sum of Negative MF over the period
        positive_mf_sum = positive_money_flow.rolling(window=self.period, min_periods=self.period).sum()
        negative_mf_sum = negative_money_flow.rolling(window=self.period, min_periods=self.period).sum()

        # Avoid division by zero if negative_mf_sum is zero
        money_flow_ratio = positive_mf_sum / negative_mf_sum.replace(0, np.nan)
        # If negative_mf_sum is 0, MFI should be 100. Handle this case.
        money_flow_ratio.fillna(np.inf, inplace=True) # Temporarily mark division by zero as infinity

        # Calculate Money Flow Index = 100 - (100 / (1 + Money Flow Ratio))
        mfi = 100 - (100 / (1 + money_flow_ratio))
        mfi[money_flow_ratio == np.inf] = 100 # Where negative_mf_sum was 0, MFI is 100
        mfi.fillna(50, inplace=True) # Fill initial NaNs with 50 (neutral)

        # Prepare output DataFrame
        output = pd.DataFrame(index=data.index)
        output[f'MFI_{self.period}'] = mfi

        return output

# --- End of volume analysis indicators ---

