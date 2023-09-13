"""
# TODO: Change this file to use the new user features.
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
from ta.momentum import RSIIndicator
from tensortrade.feed.core.base import IterableStream

from preprocess_data.custom_columns_helper import divide_array as div_arr
from preprocess_data.custom_columns_helper import sma_ratio


class ChangeStream(IterableStream):
    """stream of 'change' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        price_diff = source.close - source.open
        index_diff = source.index_close - source.index_open
        change_source = div_arr(
            price_diff/source.close,
            index_diff/source.index_close,
            source.index_close,
            False
        )
        super().__init__(source=change_source, dtype="float")


class DailyVarianceStream(IterableStream):
    """stream of 'daily_variance' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        high_minus_low = source.high - source.low
        new_source = div_arr(
            high_minus_low,
            source.close,
            source.close,
            True
        )
        super().__init__(source=new_source, dtype="float")


class SMARatioIndStream(IterableStream):
    """stream of 'sma_ratio' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        new_source = sma_ratio(
            col=source.close,
            short_ma=5,
            long_ma=30
        )
        super().__init__(source=new_source, dtype="float")


class VolumeSMARatioStream(IterableStream):
    """stream of 'volume_sma_ratio' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        new_source = sma_ratio(
            col=source.volume,
            short_ma=5,
            long_ma=30
        )
        super().__init__(source=new_source, dtype="float")


class CountSMARatioStream(IterableStream):
    """stream of 'count_sma_ratio' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        new_source = sma_ratio(
            col=source.count,
            short_ma=5,
            long_ma=30
        )
        super().__init__(source=new_source, dtype="float")


class IndvBSRatioStream(IterableStream):
    """stream of 'indv_b_s_ratio' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        new_source = div_arr(
            source.individual_buy_count,
            source.individual_sell_count,
            source.close,
            True
        )
        super().__init__(source=new_source, dtype="float")


class CorpBSRatioStream(IterableStream):
    """stream of 'corp_b_s_ratio' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        new_source = div_arr(
            source.corporate_buy_count,
            source.corporate_sell_count,
            source.close,
            True
        )
        super().__init__(source=new_source, dtype="float")


class IndCorpBRatioStream(IterableStream):
    """stream of 'ind_corp_b_ratio' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        new_source = div_arr(
            self.data_table.individual_buy_vol,
            self.data_table.corporate_buy_vol,
            self.data_table.close,
            True
        )
        super().__init__(source=new_source, dtype="float")


class RSIIndicatorStream(IterableStream):
    """stream of 'rsi' values"""

    def __init__(self, source: IterableStream):
        source = pd.DataFrame(source.iterable)
        new_source = RSIIndicator(
            close=source.close,
            fillna=True).rsi()
        super().__init__(source=new_source, dtype="float")
