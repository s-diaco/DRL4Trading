"""
Get data from [multiple] dirs
"""
import pandas as pd
import logging
import pathlib


class ExternalData:
    """
    tools to get data related to a specific ticker
    from third party sources

    Args:
        ticker: str
            The name of ticker
        csv_dirs: Path
            Directories to save data file

    Returns:
        A pandas dataframe
    """

    def __init__(
        self,
        first_date: str,
        last_date: str,
    ):
        self.first_day = first_date
        self.last_day = last_date

    def fetch_from_csv(
        self, csv_dirs: list, ticker: str, date_column: str = "date"
        ) -> pd.DataFrame:
        """Fetch data from csv files"""
        ret_val = pd.DataFrame()
        for csv_dir in csv_dirs:
            logging.info(f"fetching data for {ticker}")
            price_file_name = f'{ticker}.csv'
            if (csv_dir/pathlib.Path(price_file_name)).is_file():
                # price file exist. open and read it
                price_df = pd.read_csv(
                    csv_dir/price_file_name,
                    index_col=date_column,
                    parse_dates=[date_column],
                    header=0,
                    date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
                )
                price_df = price_df.loc[self.first_day:self.last_day]
                if ret_val.empty:
                    ret_val = price_df
                else:
                    ret_val = ret_val.join(price_df)
            else:
                # ticker file doesn't exist
                raise ValueError(
                    f'There is no file for {ticker} in dir "{csv_dir}"'
                )
        return ret_val

    def fetch_baseline_from_csv(
        self, dir: str, file_name: str, date_column: str = "date"
        ) -> pd.DataFrame:
        """Fetch baseline data from csv file"""
        bl_file_name = f'{file_name}.csv'
        logging.info(f"fetching baseline {bl_file_name}.")
        baseline_full_path = dir/pathlib.Path(bl_file_name)
        if baseline_full_path.is_file():
            # baseline file exist. open and read it
            bl_df = pd.read_csv(
                baseline_full_path,
                index_col=date_column,
                parse_dates=[date_column],
                header=0,
                date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
            )
            bl_df = bl_df.loc[self.first_day:self.last_day]
        else:
            # ticker file doesn't exist
            raise ValueError(
                f'There is no "{bl_file_name} in dir "{dir}"'
            )
        return bl_df
