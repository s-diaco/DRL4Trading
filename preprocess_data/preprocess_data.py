"""preprocess data before using it"""
import logging

import pandas as pd

from preprocess_data import csv_data, custom_col_base, custom_columns


def col_from_cls(client_class, data):
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
    column_cls = client_class(data)
    col_name, col_data = column_cls.add_column()
    if isinstance(col_data, pd.Series):
        return col_name, col_data
    else:
        raise TypeError('Method "add_column()" has to return "pd.Series"')


def add_user_defined_features(data: pd.DataFrame, user_cols) -> pd.DataFrame:
    '''
    Add data from functions in 'user_calculated_columns.py'.

            Parameters:
                    data (pd.DataFrame): data used to calculate new columns
                    user_cols (list): user class names to use

            Returns:
                    data (pd.DataFrame): the updated dataframe
    '''
    logging.info('Adding custom columns')
    for col_cls in custom_col_base.CustomColumn.__subclasses__():
        cls_name = col_cls.__name__
        if(col_cls.__module__ == custom_columns.__name__):
            if cls_name in user_cols:
                try:
                    # add new column to dataframe
                    new_col_name, new_col = col_from_cls(col_cls, data)
                    data[new_col_name] = new_col
                    logging.info(f'Add column "{new_col_name}" ✅')
                except Exception as e:
                    logging.info(f'Add column from user class "{cls_name}" '
                    f'❌: {str(e)}')

    return data


def preprocess_data(tic_list, start_date, end_date,
                    field_mappings, baseline_filed_mappings,
                    csv_file_info, user_columns) -> pd.DataFrame:
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
        processed_data, user_columns
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

