"""
How to add a custom column:
define a class that implements custom_column.CustomColumn.
'self.data' is a 'pandas.DataFrame' object containing the full database and you 
can use it to calculate your own column (see bellow implementions ).
Return value of the function should be your new column and type of the return 
value should be 'pd.Series'.
You will get an 'unsuccessful' log message if you use data from a column that 
isn't in the 'data'.
Available columns:
    1- Every column in csv files
    2- Columns defined in config file as 'TECHNICAL_INDICATORS_LIST'
A complete list of columns are shown in log before adding costom columns.
The new column's name will be same as the defined class name.
"""

import pandas as pd

from preprocess_data import custom_column
from preprocess_data.custom_columns_helper import divide_array as div_arr


class change(custom_column.CustomColumn):
    """'Change' column"""

    def add_column(self) -> pd.Series:
        """add calculated column: 'change'"""
        new_column = div_arr(
            (self.data.close-self.data.open)/self.data.close,
            (self.data.index_close-self.data.index_open)/self.data.index_close,
            self.data.index_close,
            False
        )
        return new_column


class daily_variance(custom_column.CustomColumn):
    """'daily_variance' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'daily_variance'"""
        new_column = div_arr(
            self.data.high-self.data.low,
            self.data.close,
            self.data.close,
            True
        )
        return new_column


class volume_ma_ratio(custom_column.CustomColumn):
    """'volume_ma_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'volume_ma_ratio'"""
        new_column = div_arr(
            self.data.volume_5_sma,
            self.data.volume_30_sma,
            self.data.close,
            False
        )
        return new_column


class count_ma_ratio(custom_column.CustomColumn):
    """'count_ma_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'count_ma_ratio'"""
        new_column = div_arr(
            self.data.count_5_sma,
            self.data.count_30_sma,
            self.data.close,
            False
        )
        return new_column


class ma_ratio(custom_column.CustomColumn):
    """'ma_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'ma_ratio'"""
        new_column = div_arr(
            self.data.close_5_sma,
            self.data.close_30_sma,
            self.data.close,
            False
        )
        return new_column


class indv_buy_sell_ratio(custom_column.CustomColumn):
    """'indv_buy_sell_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'indv_buy_sell_ratio'"""
        new_column = div_arr(
            self.data.individual_buy_count,
            self.data.individual_sell_count,
            self.data.close,
            True
        )
        return new_column


class corp_buy_sell_ratio(custom_column.CustomColumn):
    """'corp_buy_sell_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'corp_buy_sell_ratio'"""
        new_column = div_arr(
            self.data.corporate_buy_count,
            self.data.corporate_sell_count,
            self.data.close,
            True
        )
        return new_column


class ind_corp_buy_ratio(custom_column.CustomColumn):
    """'ind_corp_buy_ratio' column"""

    def add_column(self) -> pd.Series:
        """add calculated column 'ind_corp_buy_ratio'"""
        new_column = div_arr(
            self.data.individual_buy_vol,
            self.data.corporate_buy_vol,
            self.data.close,
            True
        )
        return new_column
