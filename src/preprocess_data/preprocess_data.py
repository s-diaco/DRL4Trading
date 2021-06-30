"""preprocess data before using it"""
import inspect
import logging
import types

import pandas as pd

from preprocess_data import csv_data, custom_columns


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
    logging.info('Adding custom columns')
    for i in dir(custom_columns):
        item = getattr(custom_columns, i)
        if callable(item):
            # add new column to dataframe
            try:
                func_params=inspect.signature(item)
                first_param = next(iter(func_params.parameters))
                if first_param == 'data':
                    data[i] = new_column_from_client_func(item, data)
            except AttributeError:
                logging.info(f'Add column "{i}": unsuccessful!')
            except ValueError:
                pass
    return data


def preprocess_data(tic_list, start_date, end_date,
                    field_mappings, baseline_filed_mappings,
                    csv_file_info) -> pd.DataFrame:
    """preprocess data before using"""
    logging.info(f'Train start date: {start_date}')
    logging.info(f'Train end date: {end_date}')
    logging.info(f'Tickers: {tic_list}')
    logging.info(f'Fetching data from csv files')
    data_loader = csv_data.CSVData(
        start_date=start_date,
        end_date=end_date,
        ticker_list=tic_list,
        csv_dirs=csv_file_info["dir_list"],
        baseline_file_name=csv_file_info["baseline_file_name"],
        has_daily_trading_limit=csv_file_info["has_daily_trading_limit"],
        use_baseline_data=csv_file_info["use_baseline_data"],
        baseline_filed_mappings=baseline_filed_mappings,
        baseline_date_column_name=csv_file_info["baseline_date_column_name"]
        )
    processed_data = data_loader.fetch_data(
        field_mappings = field_mappings,
        date_column=csv_file_info["date_column_name"])

    # Preprocess Data
    processed_data = add_user_defined_features(
        processed_data
    )

    logging.info(f'Preprocessed data (tail): \n{processed_data.tail()}')
    logging.info(f'Sample size: {len(processed_data)}')
    logging.info(f'Columns after preprocess: {processed_data.columns}')
    return processed_data

def get_baseline_df(
        start_date,
        end_date,
        baseline_filed_mappings,
        csv_file_info
):
    """
    return a dataframe for baseline data
    """
    data_loader = csv_data.CSVData(
        start_date=start_date,
        end_date=end_date,
        baseline_file_name=csv_file_info["baseline_file_name"],
        use_baseline_data=csv_file_info["use_baseline_data"],
        baseline_filed_mappings=baseline_filed_mappings,
        baseline_date_column_name=csv_file_info["baseline_date_column_name"]
    )
    baseline_df = data_loader.baseline_df
    return baseline_df

