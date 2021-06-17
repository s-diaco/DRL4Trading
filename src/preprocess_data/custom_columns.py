"""
How to add a custom column:
define a function with one parameter named 'data'.
'data' is a 'pandas.DataFrame' object containing the full database and you 
can use it to calculate your own column (see bellow functions ).
Return value of the function should be your new column and type of the return 
value should be 'pd.Series'.
You will get an 'unsuccessful' log message if you use data from a column that 
isn't in the 'data'.
Available columns:
    1- Every columan in csv files
    2- Columns defined in config file as 'TECHNICAL_INDICATORS_LIST'
A complete list of columns are shown in log before adding costom columns.
The new column's name will be same as the defined function name.
"""

import numpy as np
from numpy.core.multiarray import where
from numpy.core.numeric import zeros_like

def change(data):
    """add calculated column: 'change'"""
    new_column = np.divide(
        ((data.close-data.open)/data.close),
        ((data.index_close-data.index_open)/data.index_close),
        out=np.ones_like(data.index_close),
        where=(data.index_close-data.index_open) != 0)
    return new_column

def daily_variance(data):
    """add calculated column 'daily_variance'"""
    new_column = np.divide(
        data.high-data.low,
        data.close,
        out=zeros_like(data.close),
        where=data.close != 0)
    return new_column

def volume_ma_ratio(data):
    """add calculated column 'volume_ma_ratio'"""
    new_column = np.divide(
        data.volume_5_sma,
        data.volume_30_sma,
        out=np.ones_like(data.close),
        where=data.volume_30_sma != 0)
    return new_column

    processed['count_ma_ratio'] = np.divide(
        processed.count_5_sma,
        processed.count_30_sma,
        out=np.ones_like(processed.close),
        where=processed.count_30_sma != 0)
    processed['ma_ratio'] = np.divide(
        processed.close_5_sma,
        processed.close_30_sma,
        out=np.ones_like(processed.close),
        where=processed.close_30_sma != 0)
    processed['indv_buy_sell_ratio'] = np.divide(
        processed.individual_buy_count,
        processed.individual_sell_count,
        out=np.zeros_like(processed.close),
        where=processed.individual_sell_count != 0)
    processed['corp_buy_sell_ratio'] = np.divide(
        processed.corporate_buy_count,
        processed.corporate_sell_count,
        out=np.zeros_like(processed.close),
        where=processed.corporate_sell_count != 0)
    processed['ind_corp_buy_ratio'] = np.divide(
        processed.individual_buy_vol,
        processed.corporate_buy_vol,
        out=np.zeros_like(processed.close),
        where=processed.corporate_buy_vol != 0)

    return processed
