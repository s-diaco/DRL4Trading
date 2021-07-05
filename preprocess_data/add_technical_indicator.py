"""
calcualte technical indicators
"""
import logging
import pandas as pd
from stockstats import StockDataFrame as stock_df


def add_technical_indicator(
    data: pd.DataFrame,
    tech_indicator_list: list
) -> pd.DataFrame:
    """
    calcualte technical indicators
    add technical inidactors using "stockstats" package

    Parameters:
            data (df): pandas dataframe
            tech_indicator_list (list): list of stockstats indicators

    Returns: 
            df (pd.DataFrame): data with indicators
    """
    df = data.copy()
    stock = stock_df.retype(df)

    for indicator in tech_indicator_list:
        if not stock[indicator].empty:
            df[indicator] = stock[indicator]
    return df
