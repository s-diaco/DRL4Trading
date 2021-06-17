import logging
from typing import Tuple

import numpy as np
import pandas as pd
from config import config
from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from numpy.core.multiarray import where
from numpy.core.numeric import zeros_like
from stockstats import StockDataFrame as Sdf

from . import csv_data, user_calculated_columns


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
    stock = Sdf.retype(df.copy())
    unique_ticker = stock.tic.unique()

    for indicator in tech_indicator_list:
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            try:
                temp_indicator = stock[stock.tic ==
                                       unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                indicator_df = indicator_df.append(
                    temp_indicator, ignore_index=True
                )
            except Exception as e:
                logging.error(e)
        df[indicator] = indicator_df
    return df


def new_column_from_client_func(client_func, data):
    '''
    Create "series" from a given function and dataframe

            Parameters:
                    client_func (callable): function used to create the column
                    data (pd.DataFrame): data used to calculate new columns

            Returns:
                    column (pd.Series): calculated column

            Raises:
                    TypeError: if the column type is not pd.Series
    '''
    # TODO check if there are any Nan or inf values in new column
    column = client_func(data)
    if isinstance(column, pd.Series):
        return column
    else:
        raise TypeError(f'Type of return value for "{str(client_func)}" \
            func should be "pd.Series"')


def add_user_defined_features(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Add data from functions in 'user_calculated_columns.py'.

            Parameters:
                    data (pd.DataFrame): data used to calculate new columns

            Returns:
                    data (pd.DataFrame): the updated dataframe
    '''
    logging.info(f'Adding custom columns')
    for i in dir(user_calculated_columns):
        item = getattr(user_calculated_columns, i)
        if callable(item):
            # add new column to dataframe
            try:
                data[str(item)] = new_column_from_client_func(item, data)
            except:
                logging.info(f'Add column "{str(item)}": unsuccessful!')
    return data


def preprocess_data(tic_list, start_date, end_date) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # if not os.path.exists("./" + config.DATA_SAVE_DIR):
    #    os.makedirs("./" + config.DATA_SAVE_DIR)
    # if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    #    os.makedirs("./" + config.TRAINED_MODEL_DIR)
    # if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    #    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    # if not os.path.exists("./" + config.RESULTS_DIR):
    #    os.makedirs("./" + config.RESULTS_DIR)

    # from config.py start_date is a string
    logging.info(f'Train start date: {start_date}')
    # from config.py end_date is a string
    logging.info(f'Train end date: {end_date}')
    logging.info(f'Tickers: {tic_list}')
    data_loader = csv_data.CSVData(
        start_date=start_date,
        end_date=end_date,
        baseline_file_name=config.BASELINE_FILE_NAME,
        baseline_dir=config.BASELINE_DIR,
        ticker_list=tic_list,
        csv_dirs=config.TICKER_CSV_DIR_LIST,
        has_daily_trading_limit=config.HAS_DAILY_TRADING_LIMIT,
        use_baseline_data=config.USE_BASELINE_DATA)
    raw_df = data_loader.fetch_data()

    # Preprocess Data
    processed_data = add_technical_indicator(
        data=raw_df,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST
    )

    processed_data = add_user_defined_features(
        processed_data
    )

    logging.info(f'Preprocessed data (tail): \n {processed_data.tail()}')
    logging.info(f'Sample size: {len(processed_data)}')
    logging.info(f'Training column names: {processed_data.columns}')

    return processed_data
