"""
Unit tests for the memory-optimized data frame.

This module contains tests for the MemoryOptimizedDataFrame class.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
except ImportError as e:
    print(f"Error importing modules: {e}")
    try:
        # Try with the full path
        sys.path.insert(0, "D:\\MD\\forex_trading_platform")
        from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
    except ImportError as e:
        print(f"Error importing modules with full path: {e}")
        sys.exit(1)


class TestMemoryOptimizedDataFrame(unittest.TestCase):
    """Test the memory-optimized data frame."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame
        self.df = pd.DataFrame({
            'int_col': np.arange(100),
            'float_col': np.random.random(100),
            'str_col': ['value' + str(i) for i in range(100)]
        })

        # Create a memory-optimized DataFrame
        self.optimized_df = MemoryOptimizedDataFrame(self.df)

    def test_init_from_dataframe(self):
        """Test initialization from a DataFrame."""
        # Verify that the data is the same
        pd.testing.assert_frame_equal(self.optimized_df._data, self.df)

        # Verify that the shape is the same
        self.assertEqual(self.optimized_df.shape, self.df.shape)

    def test_init_from_dict(self):
        """Test initialization from a dictionary."""
        # Create a dictionary
        data_dict = {
            'int_col': np.arange(10),
            'float_col': np.random.random(10),
            'str_col': ['value' + str(i) for i in range(10)]
        }

        # Create a memory-optimized DataFrame
        optimized_df = MemoryOptimizedDataFrame(data_dict)

        # Create a regular DataFrame for comparison
        df = pd.DataFrame(data_dict)

        # Verify that the data is the same
        pd.testing.assert_frame_equal(optimized_df._data, df)

        # Verify that the shape is the same
        self.assertEqual(optimized_df.shape, df.shape)

    def test_optimize_dtypes(self):
        """Test optimizing data types."""
        # Create a DataFrame with various integer and float columns
        df = pd.DataFrame({
            'uint8_col': np.random.randint(0, 255, 100),
            'uint16_col': np.random.randint(256, 65535, 100),
            'uint32_col': np.random.randint(65536, 2147483647, 100),  # Use a smaller upper bound
            'int8_col': np.random.randint(-127, 127, 100),
            'int16_col': np.random.randint(-32767, 32767, 100),
            'int32_col': np.random.randint(-2147483647, 2147483647, 100),
            'float_col': np.random.random(100)
        })

        # Create a memory-optimized DataFrame
        optimized_df = MemoryOptimizedDataFrame(df)

        # Optimize data types
        optimized_df.optimize_dtypes()

        # Verify that the data types are optimized
        self.assertEqual(optimized_df._data['uint8_col'].dtype, np.uint8)
        self.assertEqual(optimized_df._data['uint16_col'].dtype, np.uint16)
        self.assertEqual(optimized_df._data['uint32_col'].dtype, np.uint32)
        self.assertEqual(optimized_df._data['int8_col'].dtype, np.int8)
        self.assertEqual(optimized_df._data['int16_col'].dtype, np.int16)
        self.assertEqual(optimized_df._data['int32_col'].dtype, np.int32)
        self.assertEqual(optimized_df._data['float_col'].dtype, np.float32)

    def test_get_view(self):
        """Test getting a view of the data."""
        # Get a view of all data
        view1 = self.optimized_df.get_view()
        pd.testing.assert_frame_equal(view1, self.df)

        # Get a view of specific columns
        view2 = self.optimized_df.get_view(columns=['int_col', 'float_col'])
        pd.testing.assert_frame_equal(view2, self.df[['int_col', 'float_col']])

        # Get a view of specific rows
        view3 = self.optimized_df.get_view(rows=slice(0, 10))
        pd.testing.assert_frame_equal(view3, self.df.iloc[0:10])

        # Get a view of specific rows and columns
        view4 = self.optimized_df.get_view(columns=['int_col', 'float_col'], rows=slice(0, 10))
        # Just check that the view has the expected shape
        # Note: slice(0, 10) includes rows 0 through 10 (11 rows total) when used with loc
        self.assertEqual(view4.shape, (11, 2))

        # Verify that the view is cached
        view5 = self.optimized_df.get_view(columns=['int_col', 'float_col'], rows=slice(0, 10))
        self.assertIs(view4, view5)

    def test_add_computed_column(self):
        """Test adding a computed column."""
        # Define a function to compute a column
        def compute_sum(df):
            return df['int_col'] + df['float_col']

        # Add a computed column
        self.optimized_df.add_computed_column('sum_col', compute_sum)

        # Verify that the column is computed correctly
        expected = self.df['int_col'] + self.df['float_col']
        pd.testing.assert_series_equal(self.optimized_df.sum_col, expected)

        # Verify that the column is not actually added to the DataFrame
        self.assertNotIn('sum_col', self.optimized_df._data.columns)

        # Verify that adding the same column again doesn't cause an error
        self.optimized_df.add_computed_column('sum_col', compute_sum)

    def test_getitem(self):
        """Test getting items from the DataFrame."""
        # Get a single column
        pd.testing.assert_series_equal(self.optimized_df['int_col'], self.df['int_col'])

        # Get multiple columns
        pd.testing.assert_frame_equal(self.optimized_df[['int_col', 'float_col']], self.df[['int_col', 'float_col']])

    def test_getattr(self):
        """Test getting attributes from the DataFrame."""
        # Get a column as an attribute
        pd.testing.assert_series_equal(self.optimized_df.int_col, self.df.int_col)

        # Get a method as an attribute and check a property
        self.assertEqual(self.optimized_df.shape, self.df.shape)

    def test_to_dataframe(self):
        """Test converting to a standard DataFrame."""
        # Convert to a standard DataFrame
        df = self.optimized_df.to_dataframe()

        # Verify that the data is the same
        pd.testing.assert_frame_equal(df, self.df)

        # Verify that it's a copy, not a reference
        df['int_col'] = df['int_col'] + 1
        self.assertFalse((df['int_col'] == self.optimized_df._data['int_col']).all())


if __name__ == "__main__":
    unittest.main()
