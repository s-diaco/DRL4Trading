"""
How to add a custom column:
Define a class that implements custom_col_base.CustomColumn.
'self.data_table' is a pandas DataFrame containing full data. You can use it
to calculate your own column (see bellow examples).

Notes:
- Return value of the function should be a tuple containing:
    Added column name (string),
    Added column data (pd.Series).
- For every column to be used in the training or predicting process you have to
add class name and column name to "USER_DEFINED_FEATURES" and "DATA_COLUMNS"
section of the settings file (config/settings.py).
- List of columns is shown in log before trainig or predicting start.
"""

import pandas as pd

from preprocess_data import custom_col_base
from preprocess_data.custom_columns_helper import divide_array as div_arr
from preprocess_data.custom_columns_helper import sma_ratio
from ta.momentum import RSIIndicator


class ChangeCol(custom_col_base.CustomColumn):
    """'change' column"""

    def add_column(self) -> pd.Series:
        """add calculated column: 'change'"""
        new_column_name = 'change'
        tic_day_ch = self.data_table.close-self.data_table.open
        index_day_ch = self.data_table.index_close-self.data_table.index_open
        day_tic_c = self.data_table.close
        day_index_c = self.data_table.index_close
        new_column = div_arr(
            tic_day_ch/day_tic_c,
            index_day_ch/day_index_c,
            day_index_c,
            False
        )
        return new_column_name, new_column


class DailyVarianceCol(custom_col_base.CustomColumn):
    """'daily_variance' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'daily_variance'"""
        new_column_name = 'daily_variance'
        new_column = div_arr(
            self.data_table.high-self.data_table.low,
            self.data_table.close,
            self.data_table.close,
            True
        )
        return new_column_name, new_column


class SMARatioIndicator(custom_col_base.CustomColumn):
    """'Simple Moving Average Ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'sma_ratio'"""
        new_column_name = 'sma_ratio'
        new_column = sma_ratio(
            col=self.data_table.close,
            short_ma=5,
            long_ma=30
        )
        return new_column_name, new_column


class VolumeSMARatioCol(custom_col_base.CustomColumn):
    """'volume_sma_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'volume_sma_ratio'"""
        new_column_name = 'volume_sma_ratio'
        new_column = sma_ratio(
            col=self.data_table.volume,
            short_ma=5,
            long_ma=30
        )
        return new_column_name, new_column


class CountSMARatioCol(custom_col_base.CustomColumn):
    """'count_sma_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'count_sma_ratio'"""
        new_column_name = 'count_sma_ratio'
        new_column = sma_ratio(
            col=self.data_table.count,
            short_ma=5,
            long_ma=30
        )
        return new_column_name, new_column


class IndvBSRatioCol(custom_col_base.CustomColumn):
    """'indv_b_s_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'indv_b_s_ratio'"""
        new_column_name = 'indv_buy_sell_ratio'
        new_column = div_arr(
            self.data_table.individual_buy_count,
            self.data_table.individual_sell_count,
            self.data_table.close,
            True
        )
        return new_column_name, new_column


class CorpBSRatioCol(custom_col_base.CustomColumn):
    """'corp_buy_sell_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'corp_buy_sell_ratio'"""
        new_column_name = 'corp_b_s_ratio'
        new_column = div_arr(
            self.data_table.corporate_buy_count,
            self.data_table.corporate_sell_count,
            self.data_table.close,
            True
        )
        return new_column_name, new_column


class IndCorpBRatioCol(custom_col_base.CustomColumn):
    """'ind_corp_buy_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'ind_corp_b_ratio'"""
        new_column_name = 'ind_corp_buy_ratio'
        new_column = div_arr(
            self.data_table.individual_buy_vol,
            self.data_table.corporate_buy_vol,
            self.data_table.close,
            True
        )
        return new_column_name, new_column

class RSIIndicatorCol(custom_col_base.CustomColumn):
    """'RSI' Technical indicator data feature"""

    def add_column(self) -> pd.Series:
        """Add calculated column 'rsi'"""
        new_column_name = 'rsi'
        new_column = RSIIndicator(
            close=self.data_table.close,
            fillna=True).rsi()
        return new_column_name, new_column
