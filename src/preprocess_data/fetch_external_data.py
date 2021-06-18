"""
Get csv data from [multiple] dirs
"""
import logging
import pathlib

import pandas as pd


class ExternalData:
    """
    Tools to get data from third party sources
    """

    def __init__(
        self,
        first_date: str,
        last_date: str,
    ):
        """
        Initialize class variables.

        Parameters:
                first_date (str): Strat of data (%Y-%m-%d)
                last_date (str): End of the data (%Y-%m-%d)
        """
        self.first_day = first_date
        self.last_day = last_date

    def fetch_from_csv(
        self, csv_dirs: list,
        ticker: str,
        field_mappins: list = None,
        date_column: str = "date"
    ) -> pd.DataFrame:
        """Fetch data from csv files"""
        ret_val = pd.DataFrame()
        for csv_dir in csv_dirs:
            logging.info(f"fetching data for {ticker}")
            price_file_name = f'{ticker}.csv'
            csv_df = self._get_single_csv(
                file_name=csv_dir/pathlib.Path(price_file_name),
                date_column=date_column,
                field_mappins=field_mappins
            )
            if ret_val.empty:
                ret_val = csv_df
            else:
                ret_val = ret_val.join(csv_df)
        return ret_val

    def _get_single_csv(
        self,
        file_name: pathlib.Path,
        date_column: str,
        field_mappins: list = None
    ) -> pd.DataFrame:
        """Fetch data from a csv file"""
        csv_df = pd.read_csv(
            file_name,
            index_col=date_column,
            parse_dates=[date_column],
            header=0,
            date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"),
        )
        csv_df = csv_df.loc[self.first_day:self.last_day]
        if field_mappins:
            csv_df = csv_df.rename(columns=field_mappins)
        return csv_df

    def fetch_baseline_from_csv(
        self, file_name: str,
        date_column: str = "date",
        field_mappins: list = None
    ) -> pd.DataFrame:
        """Fetch baseline data from csv file"""
        logging.info(f"fetching baseline {file_name}.")
        baseline_full_path = pathlib.Path(file_name)
        bl_df = self._get_single_csv(
            baseline_full_path,
            date_column,
            field_mappins
        )
        return bl_df
