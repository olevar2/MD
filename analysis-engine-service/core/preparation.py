"""
Preparation module.

This module provides functionality for...
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import List, Optional, Dict, Any
from analysis_engine.analysis.indicators import IndicatorClient


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CausalDataPreparationError(Exception):
    """Custom exception for errors during causal data preparation."""
    pass


class CausalDataPreparation:
    """
    Handles data preparation steps specifically tailored for causal inference analysis.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the CausalDataPreparation class.

        Args:
            data (pd.DataFrame): The input time-series data, expected to have a DatetimeIndex.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise CausalDataPreparationError(
                'Input data must have a DatetimeIndex.')
        self.data = data.copy()
        self._validate_input_data()

    def _validate_input_data(self):
        """Basic validation of the input DataFrame."""
        if self.data.empty:
            raise CausalDataPreparationError('Input data cannot be empty.')

    @with_resilience('check_stationarity')
    def check_stationarity(self, column: str, significance_level: float=0.05
        ) ->Dict[str, Any]:
        """
        Checks the stationarity of a time series using the Augmented Dickey-Fuller test.

        Args:
            column (str): The name of the column (time series) to check.
            significance_level (float): The significance level for the test.

        Returns:
            Dict[str, Any]: A dictionary containing the ADF statistic, p-value, and whether the series is stationary.

        Raises:
            CausalDataPreparationError: If the column does not exist or contains NaNs/Infs.
        """
        if column not in self.data.columns:
            raise CausalDataPreparationError(
                f"Column '{column}' not found in data.")
        series = self.data[column].dropna()
        if np.isinf(series).any():
            raise CausalDataPreparationError(
                f"Column '{column}' contains infinite values.")
        if len(series) < 10:
            raise CausalDataPreparationError(
                f"Column '{column}' has insufficient data points ({len(series)}) for ADF test."
                )
        result = adfuller(series)
        p_value = result[1]
        is_stationary = p_value < significance_level
        return {'adf_statistic': result[0], 'p_value': p_value,
            'is_stationary': is_stationary, 'critical_values': result[4]}

    def difference_series(self, column: str, order: int=1, inplace: bool=False
        ) ->Optional[pd.DataFrame]:
        """
        Applies differencing to a time series to make it stationary.

        Args:
            column (str): The name of the column to difference.
            order (int): The order of differencing.
            inplace (bool): If True, modifies the internal DataFrame directly. Otherwise, returns a new DataFrame.

        Returns:
            Optional[pd.DataFrame]: The DataFrame with the differenced series if inplace is False, else None.

        Raises:
            CausalDataPreparationError: If the column does not exist.
        """
        if column not in self.data.columns:
            raise CausalDataPreparationError(
                f"Column '{column}' not found for differencing.")
        diff_series = self.data[column].diff(periods=order).dropna()
        new_col_name = f'{column}_diff_{order}'
        if inplace:
            self.data[new_col_name] = diff_series
            self.data.dropna(subset=[new_col_name], inplace=True)
            return None
        else:
            new_data = self.data.copy()
            new_data[new_col_name] = diff_series
            new_data.dropna(subset=[new_col_name], inplace=True)
            return new_data

    def handle_missing_data(self, method: str='forward_fill', columns:
        Optional[List[str]]=None, inplace: bool=True, **kwargs) ->Optional[pd
        .DataFrame]:
        """
        Handles missing data using various strategies suitable for time-series.

        Args:
            method (str): The imputation method ('forward_fill', 'backward_fill', 'interpolation', 'mean', 'median').
            columns (Optional[List[str]]): Specific columns to apply imputation. If None, applies to all columns.
            inplace (bool): If True, modifies the internal DataFrame directly. Otherwise, returns a new DataFrame.
            **kwargs: Additional arguments for specific methods (e.g., 'limit' for ffill/bfill, 'method' for interpolate).

        Returns:
            Optional[pd.DataFrame]: The DataFrame with missing values handled if inplace is False, else None.

        Raises:
            CausalDataPreparationError: If an invalid method is specified.
        """
        target_data = self.data if inplace else self.data.copy()
        cols_to_process = columns if columns else target_data.columns
        for col in cols_to_process:
            if col not in target_data.columns:
                print(
                    f"Warning: Column '{col}' specified for missing data handling not found."
                    )
                continue
            if target_data[col].isnull().any():
                if method == 'forward_fill':
                    target_data[col].ffill(inplace=True, limit=kwargs.get(
                        'limit'))
                elif method == 'backward_fill':
                    target_data[col].bfill(inplace=True, limit=kwargs.get(
                        'limit'))
                elif method == 'interpolation':
                    interp_method = kwargs.get('interpolation_method', 'linear'
                        )
                    target_data[col].interpolate(method=interp_method,
                        inplace=True, limit_direction='both', limit_area=None)
                elif method == 'mean':
                    mean_val = target_data[col].mean()
                    target_data[col].fillna(mean_val, inplace=True)
                elif method == 'median':
                    median_val = target_data[col].median()
                    target_data[col].fillna(median_val, inplace=True)
                else:
                    raise CausalDataPreparationError(
                        f'Invalid missing data handling method: {method}')
                if target_data[col].isnull().any():
                    target_data[col].bfill(inplace=True)
                    target_data.dropna(subset=[col], inplace=True)
        if not inplace:
            return target_data
        return None

    def identify_potential_confounders(self, treatment_col: str,
        outcome_col: str, potential_confounder_cols: List[str], threshold:
        float=0.7) ->List[str]:
        """
        Identifies potential confounding variables based on correlation with treatment and outcome.
        Note: This is a simplistic approach. Proper confounder identification requires domain knowledge.

        Args:
            treatment_col (str): The name of the treatment variable column.
            outcome_col (str): The name of the outcome variable column.
            potential_confounder_cols (List[str]): List of columns to check as potential confounders.
            threshold (float): The absolute correlation threshold to consider a variable a potential confounder.

        Returns:
            List[str]: A list of column names identified as potential confounders based on the criteria.

        Raises:
            CausalDataPreparationError: If specified columns do not exist.
        """
        required_cols = [treatment_col, outcome_col
            ] + potential_confounder_cols
        for col in required_cols:
            if col not in self.data.columns:
                raise CausalDataPreparationError(
                    f"Column '{col}' not found for confounder identification.")
        confounders = []
        correlations = self.data[required_cols].corr()
        for col in potential_confounder_cols:
            corr_with_treatment = abs(correlations.loc[col, treatment_col])
            corr_with_outcome = abs(correlations.loc[col, outcome_col])
            if (corr_with_treatment > threshold and corr_with_outcome >
                threshold):
                confounders.append(col)
        return confounders

    def assess_data_quality_for_causal(self, treatment_col: str,
        required_cols: List[str]) ->Dict[str, Any]:
        """
        Performs basic data quality checks relevant to causal inference assumptions.

        Args:
            treatment_col (str): The name of the treatment variable column.
            required_cols (List[str]): List of essential columns (treatment, outcome, confounders) that must be present.

        Returns:
            Dict[str, Any]: A dictionary containing results of the quality checks (e.g., missing values, positivity check).

        Raises:
            CausalDataPreparationError: If essential columns are missing after preparation steps.
        """
        results = {}
        missing_essentials = [col for col in required_cols if col not in
            self.data.columns]
        if missing_essentials:
            raise CausalDataPreparationError(
                f'Essential columns missing: {missing_essentials}')
        nans_in_essentials = self.data[required_cols].isnull().sum()
        results['nans_in_essential_cols'] = nans_in_essentials[
            nans_in_essentials > 0].to_dict()
        if nans_in_essentials.sum() > 0:
            print(
                f"Warning: NaNs detected in essential columns after preparation: {results['nans_in_essential_cols']}"
                )
        if self.data[treatment_col].nunique() < 10:
            treatment_counts = self.data[treatment_col].value_counts(normalize
                =True)
            results['treatment_distribution'] = treatment_counts.to_dict()
            if (treatment_counts < 0.01).any():
                print(
                    f"Warning: Potential positivity violation. Treatment distribution: {results['treatment_distribution']}"
                    )
        else:
            results['treatment_distribution'] = (
                'Continuous or high cardinality treatment - positivity check skipped.'
                )
        return results

    @with_resilience('get_prepared_data')
    def get_prepared_data(self) ->pd.DataFrame:
        """
        Returns the prepared DataFrame.
        """
        return self.data.copy()


if __name__ == '__main__':
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({'price': np.random.randn(100).cumsum() + 50,
        'volume': np.random.randint(100, 1000, size=100), 'sentiment': np.
        random.rand(100) * 2 - 1, 'intervention': np.random.choice([0, 1],
        size=100, p=[0.8, 0.2]), 'confounder1': np.random.randn(100) * 5,
        'confounder2': np.random.randn(100) * 10}, index=dates)
    data.loc[data.sample(frac=0.05).index, 'price'] = np.nan
    data.loc[data.sample(frac=0.03).index, 'sentiment'] = np.nan
    data.loc['2023-01-10':'2023-01-15', 'confounder1'] = np.nan
    print('Original Data Head:\n', data.head())
    print('\nOriginal Data Info:\n', data.info())
    try:
        preparer = CausalDataPreparation(data)
        preparer.handle_missing_data(method='interpolation', columns=[
            'price', 'sentiment'], inplace=True)
        preparer.handle_missing_data(method='forward_fill', columns=[
            'confounder1'], inplace=True, limit=3)
        preparer.handle_missing_data(method='backward_fill', inplace=True)
        print('\nData after handling missing values:\n', preparer.
            get_prepared_data().info())
        stationarity_check = preparer.check_stationarity('price')
        print(f"\nStationarity check for 'price': {stationarity_check}")
        if not stationarity_check['is_stationary']:
            print("Differencing 'price' series...")
            preparer.difference_series('price', order=1, inplace=True)
            print('Data info after differencing:\n', preparer.
                get_prepared_data().info())
            stationarity_check_diff = preparer.check_stationarity(
                'price_diff_1')
            print(
                f"Stationarity check for 'price_diff_1': {stationarity_check_diff}"
                )
        potential_confounders = ['volume', 'sentiment', 'confounder1',
            'confounder2']
        outcome = ('price_diff_1' if 'price_diff_1' in preparer.
            get_prepared_data().columns else 'price')
        identified_confounders = preparer.identify_potential_confounders(
            treatment_col='intervention', outcome_col=outcome,
            potential_confounder_cols=potential_confounders, threshold=0.1)
        print(
            f'\nPotential confounders identified (correlation > 0.1): {identified_confounders}'
            )
        essential_cols = [outcome, 'intervention'] + identified_confounders
        quality_assessment = preparer.assess_data_quality_for_causal(
            treatment_col='intervention', required_cols=essential_cols)
        print(f'\nData quality assessment results: {quality_assessment}')
        final_data = preparer.get_prepared_data()
        print('\nFinal Prepared Data Head:\n', final_data.head())
        print('\nFinal Prepared Data Info:\n', final_data.info())
    except CausalDataPreparationError as e:
        print(f'\nError during data preparation: {e}')
    except Exception as e:
        print(f'\nAn unexpected error occurred: {e}')
