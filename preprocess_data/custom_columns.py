"""
How to add a custom column:
Define a class that implements custom_col_base.CustomColumn.
'self.data_table' is a pandas DataFrame containing full data. You can use it
to calculate your own column (see bellow examples).
Notes:
- Return value of the function should be a tuple containing:
    Added column name (string),
    Added column data (pd.Series).
- List of available columns are shown in log before adding custom columns.
"""

import pandas as pd

from preprocess_data import custom_col_base
from preprocess_data.custom_columns_helper import divide_array as div_arr, sma_ratio


class ChangeCol(custom_col_base.CustomColumn):
    """'change' column"""

    def add_column(self) -> pd.Series:
        """add calculated column: 'change'"""
        new_column_name = 'change'
        new_column = div_arr(
            (self.data_table.close-self.data_table.open)/self.data_table.close,
            (self.data_table.index_close-self.data_table.index_open)/self.data_table.index_close,
            self.data_table.index_close,
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
            col = self.data_table.close,
            short_ma = 5,
            long_ma = 30
        )
        return new_column_name, new_column

class VolumeMARatioCol(custom_col_base.CustomColumn):
    """'volume_ma_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'volume_ma_ratio'"""
        new_column_name = 'volume_ma_ratio'
        new_column = div_arr(
            self.data_table.volume_5_sma,
            self.data_table.volume_30_sma,
            self.data_table.close,
            False
        )
        return new_column_name, new_column


class CountMARatioCol(custom_col_base.CustomColumn):
    """'count_ma_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'count_ma_ratio'"""
        new_column_name = 'count_ma_ratio'
        new_column = div_arr(
            self.data_table.count_5_sma,
            self.data_table.count_30_sma,
            self.data_table.close,
            False
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
